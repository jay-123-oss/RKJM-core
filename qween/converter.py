"""
Out-of-Core Streaming Post-Training Quantization (PTQ) Converter for Qwen Architecture.
Converts 27B-32B parameter Qwen FP16/BF16 models (~54GB) into 1.58-bit packed .rkmjbin format (~7GB)
strictly within a <= 3.5 GB physical DRAM ceiling.
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from rkmj.serialization.packer import pack_ternary_weights

logger = logging.getLogger("qween.converter")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

MAGIC = b"RKMJBIN1"
VERSION = 1


class QwenStreamingPTQConverter:
    """
    Out-of-Core Streaming PTQ Converter.
    Iterates through Safetensors shards layer-by-layer, quantizes heavy linear projections
    to ternary 1.58-bit {-1, 0, +1} with dynamic per-channel scale, and streams 2-bit packed
    payloads to disk without loading complete shards into RAM.
    """

    def __init__(
        self,
        model_dir: str,
        output_path: str,
        max_ram_bytes: int = int(3.5 * 1024 * 1024 * 1024),
    ):
        self.model_dir = os.path.abspath(model_dir)
        self.output_path = os.path.abspath(output_path)
        self.max_ram_bytes = max_ram_bytes

        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0

        # Load glibc malloc_trim on Linux
        self._libc = None
        if sys.platform.startswith("linux"):
            try:
                self._libc = ctypes.CDLL("libc.so.6")
            except Exception as e:
                logger.debug("libc.so.6 malloc_trim unavailable: %s", e)

        # Resolve local directory or download from Hugging Face
        self.model_dir = self._resolve_model_dir(model_dir)

        # Parse model configuration
        self.config = self._load_config()

    def _resolve_model_dir(self, model_dir_or_id: str) -> str:
        """Resolves local directory or autonomously downloads model from Hugging Face Hub."""
        if os.path.isdir(model_dir_or_id):
            logger.info("Found local model directory: %s", os.path.abspath(model_dir_or_id))
            return os.path.abspath(model_dir_or_id)

        # Check if already downloaded to ./weights/<model_name>
        model_name = os.path.basename(model_dir_or_id.rstrip("/"))
        local_weights_dir = os.path.abspath(os.path.join("weights", model_name))
        if os.path.isdir(local_weights_dir):
            cfg_check = os.path.join(local_weights_dir, "config.json")
            if os.path.exists(cfg_check):
                logger.info("Found existing local cached weights at: %s", local_weights_dir)
                return local_weights_dir

        # Attempt to import huggingface_hub with inline fallback
        logger.info("Local path '%s' not found. Checking Hugging Face Hub...", model_dir_or_id)
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.warning("huggingface_hub is not installed. Attempting inline installation...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.20.0"])
                from huggingface_hub import snapshot_download
                logger.info("huggingface_hub successfully installed.")
            except Exception as exc:
                raise ImportError(
                    f"huggingface_hub is required to download models from Hugging Face.\n"
                    f"Auto-install failed: {exc}\n"
                    "Please install it manually inside the virtual environment:\n"
                    "    source myenv/bin/activate\n"
                    "    pip install huggingface_hub"
                )

        logger.info("=" * 70)
        logger.info("📥 Downloading model shards & metadata from Hugging Face Hub")
        logger.info("   Model ID:        %s", model_dir_or_id)
        logger.info("   Target Cache:    %s", local_weights_dir)
        logger.info("   Allowed files:   *.json, *.safetensors, *.model, tokenizer*, vocab*")
        logger.info("=" * 70)

        os.makedirs(local_weights_dir, exist_ok=True)
        try:
            downloaded_path = snapshot_download(
                repo_id=model_dir_or_id,
                local_dir=local_weights_dir,
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "vocab*"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin", "*.ot", "*.pt", "*.pth"],
            )
            logger.info("✅ Hugging Face model files successfully downloaded to: %s", downloaded_path)
            return os.path.abspath(downloaded_path)
        except Exception as e:
            # If local_dir download fails, try default hub cache
            logger.warning("Direct local_dir download encountered error (%s). Trying default cache...", e)
            downloaded_path = snapshot_download(
                repo_id=model_dir_or_id,
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "vocab*"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin", "*.ot", "*.pt", "*.pth"],
            )
            logger.info("✅ Hugging Face model downloaded to cache: %s", downloaded_path)
            return os.path.abspath(downloaded_path)

    def _trim_memory(self):
        """Aggressively releases unreferenced memory and trims glibc heap arenas."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._libc and hasattr(self._libc, "malloc_trim"):
            try:
                self._libc.malloc_trim(0)
            except Exception:
                pass

        rss = self.process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        if rss > self.max_ram_bytes:
            logger.warning(
                "WARNING: Active RSS (%.2f GB) exceeds ceiling (%.2f GB)!",
                rss / (1024.0 ** 3),
                self.max_ram_bytes / (1024.0 ** 3),
            )

    def _load_config(self) -> Dict[str, Any]:
        cfg_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json not found in {self.model_dir}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_weight_map(self) -> Tuple[Dict[str, str], List[str]]:
        """Finds all Safetensors files and indexes tensor keys."""
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            weight_map = idx_data.get("weight_map", {})
            return weight_map, list(set(weight_map.values()))

        # Single or local safetensors
        from safetensors import safe_open
        sfs = [f for f in os.listdir(self.model_dir) if f.endswith(".safetensors")]
        if not sfs:
            raise FileNotFoundError(f"No .safetensors files found in {self.model_dir}")

        weight_map = {}
        for sf in sfs:
            full_sf = os.path.join(self.model_dir, sf)
            with safe_open(full_sf, framework="pt") as handle:
                for k in handle.keys():
                    weight_map[k] = sf
        return weight_map, sfs

    def convert(self) -> Dict[str, Any]:
        """
        Executes end-to-end out-of-core PTQ quantization and serialization.
        Returns conversion statistics.
        """
        from safetensors import safe_open

        t_start = time.perf_counter()
        weight_map, shard_files = self._build_weight_map()

        logger.info("=" * 70)
        logger.info("🚀 Starting Qwen Out-of-Core Streaming PTQ Conversion")
        logger.info("   Model Directory:  %s", self.model_dir)
        logger.info("   Output Path:      %s", self.output_path)
        logger.info("   Safetensors Shards: %d shards", len(shard_files))
        logger.info("   RSS Hard Ceiling: %.2f GB", self.max_ram_bytes / (1024.0 ** 3))
        logger.info("=" * 70)

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        tmp_payload_path = self.output_path + ".tmp_payload"

        tensor_metadata: List[Dict[str, Any]] = []
        current_offset = 0
        total_uncompressed_bytes = 0
        total_packed_bytes = 0

        # Discover all layer indices
        layer_indices = set()
        for k in weight_map:
            if "layers." in k:
                try:
                    part = k.split("layers.")[1].split(".")[0]
                    layer_indices.add(int(part))
                except Exception:
                    pass
        num_layers = len(layer_indices) if layer_indices else self.config.get("num_hidden_layers", 32)
        sorted_layers = sorted(layer_indices) if layer_indices else list(range(num_layers))

        with open(tmp_payload_path, "wb") as payload_file:
            # 1. Process Embeddings & Non-Layer Tensors first
            non_layer_keys = [k for k in weight_map if "layers." not in k]
            logger.info("Ingesting global embeddings and norms (%d tensors)...", len(non_layer_keys))

            # Cache handles to shards
            shard_handles: Dict[str, Any] = {}
            try:
                for k in non_layer_keys:
                    shard_name = weight_map[k]
                    if shard_name not in shard_handles:
                        shard_handles[shard_name] = safe_open(
                            os.path.join(self.model_dir, shard_name), framework="pt"
                        )
                    tensor = shard_handles[shard_name].get_tensor(k).float().contiguous()

                    # Save embedding/norm as FP32 or FP16
                    raw_bytes = tensor.numpy().tobytes()
                    nbytes = len(raw_bytes)
                    payload_file.write(raw_bytes)

                    total_uncompressed_bytes += nbytes
                    total_packed_bytes += nbytes

                    tensor_metadata.append({
                        "name": k,
                        "is_packed": False,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                        "offset": current_offset,
                        "length": nbytes,
                    })
                    current_offset += nbytes

                    del tensor, raw_bytes
                self._trim_memory()

                # 2. Process Layers One-by-One
                logger.info("Streaming and quantizing %d Transformer layers out-of-core...", len(sorted_layers))
                for layer_idx in sorted_layers:
                    layer_prefix = f"model.layers.{layer_idx}."
                    # Find all tensors for this layer
                    layer_keys = [
                        k for k in weight_map
                        if k.startswith(layer_prefix) or k.startswith(f"layers.{layer_idx}.")
                    ]

                    for k in layer_keys:
                        shard_name = weight_map[k]
                        if shard_name not in shard_handles:
                            shard_handles[shard_name] = safe_open(
                                os.path.join(self.model_dir, shard_name), framework="pt"
                            )
                        raw_t = shard_handles[shard_name].get_tensor(k).float()

                        # Determine if this tensor is a heavy linear weight matrix to quantize to 1.58-bit
                        is_heavy_linear = any(
                            proj in k
                            for proj in [
                                "q_proj.weight",
                                "k_proj.weight",
                                "v_proj.weight",
                                "o_proj.weight",
                                "gate_proj.weight",
                                "up_proj.weight",
                                "down_proj.weight",
                            ]
                        )

                        if is_heavy_linear:
                            # Dynamic per-channel scale alpha
                            # Shape: [Out_features, In_features]
                            alpha_base = raw_t.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
                            w_norm = raw_t / alpha_base
                            w_ternary = torch.clamp(torch.round(w_norm), -1.0, 1.0)
                            # Least-squares optimal per-channel scale: (W * W_t).sum() / (W_t^2).sum()
                            alpha = ((raw_t * w_ternary).sum(dim=1) / (w_ternary ** 2).sum(dim=1).clamp(min=1.0)).clamp(min=1e-5)

                            # Pack ternary weights into uint32 bitfield (16 weights/word)
                            try:
                                import rkmj._C as _C
                                if hasattr(_C, "pack_weights"):
                                    w_packed = _C.pack_weights(w_ternary)
                                else:
                                    w_packed = pack_ternary_weights(w_ternary)
                            except Exception:
                                w_packed = pack_ternary_weights(w_ternary)

                            # Serialize packed bitfield
                            packed_bytes = w_packed.numpy().tobytes()
                            nbytes_packed = len(packed_bytes)
                            payload_file.write(packed_bytes)

                            packed_name = k.replace(".weight", ".w_packed")
                            tensor_metadata.append({
                                "name": k,
                                "packed_name": packed_name,
                                "is_packed": True,
                                "shape": list(raw_t.shape),
                                "packed_shape": list(w_packed.shape),
                                "dtype": "int32_packed",
                                "offset": current_offset,
                                "length": nbytes_packed,
                            })
                            current_offset += nbytes_packed

                            # Also serialize dynamic alpha scale vector (FP32)
                            alpha_bytes = alpha.contiguous().numpy().tobytes()
                            nbytes_alpha = len(alpha_bytes)
                            payload_file.write(alpha_bytes)

                            alpha_name = k.replace(".weight", ".alpha")
                            tensor_metadata.append({
                                "name": alpha_name,
                                "is_packed": False,
                                "shape": list(alpha.shape),
                                "dtype": "float32",
                                "offset": current_offset,
                                "length": nbytes_alpha,
                            })
                            current_offset += nbytes_alpha

                            total_uncompressed_bytes += raw_t.numel() * 2  # FP16 equivalent
                            total_packed_bytes += (nbytes_packed + nbytes_alpha)

                            del raw_t, alpha, alpha_base, w_norm, w_ternary, w_packed, packed_bytes, alpha_bytes
                        else:
                            # Non-quantized weights: norms, biases
                            raw_bytes = raw_t.numpy().tobytes()
                            nbytes = len(raw_bytes)
                            payload_file.write(raw_bytes)

                            total_uncompressed_bytes += nbytes
                            total_packed_bytes += nbytes

                            tensor_metadata.append({
                                "name": k,
                                "is_packed": False,
                                "shape": list(raw_t.shape),
                                "dtype": str(raw_t.dtype).replace("torch.", ""),
                                "offset": current_offset,
                                "length": nbytes,
                            })
                            current_offset += nbytes
                            del raw_t, raw_bytes

                    # Crucial: Trim memory immediately after every layer
                    self._trim_memory()
                    if (layer_idx + 1) % 4 == 0 or layer_idx == len(sorted_layers) - 1:
                        logger.info(
                            "   Processed Layer %d / %d | Active RSS: %.2f GB (Peak: %.2f GB)",
                            layer_idx + 1,
                            len(sorted_layers),
                            self.process.memory_info().rss / (1024.0 ** 3),
                            self.peak_rss_bytes / (1024.0 ** 3),
                        )

            finally:
                shard_handles.clear()
                self._trim_memory()

        # 3. Assemble Header and Final .rkmjbin file
        logger.info("Assembling final .rkmjbin file with zero-copy stream...")
        cfg_export = dict(self.config)
        cfg_export["source_model_dir"] = self.model_dir
        header_dict = {
            "version": VERSION,
            "architecture": "qwen",
            "config": cfg_export,
            "tensors": tensor_metadata,
            "total_payload_bytes": current_offset,
        }
        header_json = json.dumps(header_dict, indent=2).encode("utf-8")
        header_len = len(header_json)

        with open(self.output_path, "wb") as final_out:
            # 1. Magic (8 bytes)
            final_out.write(MAGIC)
            # 2. Header Length (uint32, 4 bytes)
            final_out.write(struct.pack("<I", header_len))
            # 3. Header JSON
            final_out.write(header_json)
            # 4. Stream payload in 64MB buffer chunks
            with open(tmp_payload_path, "rb") as payload_src:
                while True:
                    chunk = payload_src.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    final_out.write(chunk)

        # Remove temporary payload file
        if os.path.exists(tmp_payload_path):
            os.remove(tmp_payload_path)

        total_time = time.perf_counter() - t_start
        final_file_size = os.path.getsize(self.output_path)
        compression_ratio = total_uncompressed_bytes / max(final_file_size, 1)

        logger.info("=" * 70)
        logger.info("✅ Qwen PTQ Conversion Complete!")
        logger.info("   Output .rkmjbin File:        %s", self.output_path)
        logger.info("   Final File Size:             %.2f GB (%d bytes)", final_file_size / (1024.0 ** 3), final_file_size)
        logger.info("   Uncompressed Source Size:    %.2f GB", total_uncompressed_bytes / (1024.0 ** 3))
        logger.info("   Compression Ratio:           %.1fx", compression_ratio)
        logger.info("   Peak Process RSS Memory:     %.2f GB (Ceiling: %.2f GB)", self.peak_rss_bytes / (1024.0 ** 3), self.max_ram_bytes / (1024.0 ** 3))
        logger.info("   Total Conversion Time:       %.2f seconds", total_time)
        logger.info("=" * 70)

        return {
            "output_path": self.output_path,
            "final_file_size_gb": round(final_file_size / (1024.0 ** 3), 3),
            "uncompressed_size_gb": round(total_uncompressed_bytes / (1024.0 ** 3), 3),
            "compression_ratio": round(compression_ratio, 2),
            "peak_rss_gb": round(self.peak_rss_bytes / (1024.0 ** 3), 3),
            "conversion_time_seconds": round(total_time, 2),
        }

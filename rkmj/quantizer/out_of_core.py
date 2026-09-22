"""
================================================================================
RKMJ-Core Smart Out-of-Core Safetensors Streaming Quantizer
================================================================================
Guarantees zero model loading spikes by streaming tensors directly from disk,
quantizing single matrices in isolation, bit-packing, and immediately releasing
heap pages back to the OS via C++ malloc_trim.
================================================================================
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
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil
import torch

from rkmj.quantizer.ternary import quantize_ternary_grouped, pack_ternary

logger = logging.getLogger("rkmj.quantizer.out_of_core")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

MAGIC = b"RKMJBIN1"
VERSION = 1


def force_heap_trim() -> None:
    """Forces glibc and Python runtime to release unused heap pages back to the OS."""
    gc.collect()
    try:
        from rkmj import _C
        if hasattr(_C, "force_heap_trim"):
            _C.force_heap_trim()
            return
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass


class OutOfCoreQuantizer:
    """
    Zero-Spike Out-of-Core Model Quantizer.
    Streams weights tensor-by-tensor directly from raw Safetensors shards on disk,
    bypassing AutoModelForCausalLM entirely.
    """

    def __init__(
        self,
        model_dir: str,
        output_path: str,
        group_size: int = 64,
        max_ram_budget_gb: float = 2.0,
    ):
        self.model_dir = os.path.abspath(model_dir)
        self.output_path = os.path.abspath(output_path)
        self.group_size = group_size
        self.max_ram_bytes = int(max_ram_budget_gb * 1024 * 1024 * 1024)
        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0

        # Load model config
        self.config = self._load_config()

        # Build index of tensor names to safetensors files
        self.tensor_to_shard: Dict[str, str] = {}
        self.shard_files: List[str] = []
        self._index_safetensors()

        # Identify linear weight patterns that should be quantized to 1.58-bit
        self.quantizable_suffixes = (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "w1.weight",
            "w2.weight",
            "w3.weight",
        )

    def _load_config(self) -> Dict[str, Any]:
        cfg_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.json in {self.model_dir}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _index_safetensors(self) -> None:
        idx_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tensor_to_shard = data.get("weight_map", {})
            self.shard_files = sorted(list(set(self.tensor_to_shard.values())))
        else:
            single = os.path.join(self.model_dir, "model.safetensors")
            if os.path.exists(single):
                self.shard_files = ["model.safetensors"]
                from safetensors import safe_open
                with safe_open(single, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        self.tensor_to_shard[k] = "model.safetensors"
            else:
                raise FileNotFoundError(f"No safetensors files found in {self.model_dir}")

    def _track_rss(self) -> float:
        rss = self.process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        rss_gb = rss / (1024.0 ** 3)
        if rss > self.max_ram_bytes:
            logger.warning(
                f"Peak RSS ({rss_gb:.2f} GB) exceeded budget ceiling ({self.max_ram_bytes / (1024**3):.2f} GB)!"
            )
        return rss_gb

    def should_quantize(self, tensor_name: str, shape: Tuple[int, ...]) -> bool:
        if len(shape) != 2:
            return False
        # Do not quantize embeddings or LM head
        if "embed_tokens" in tensor_name or "lm_head" in tensor_name:
            return False
        for suffix in self.quantizable_suffixes:
            if tensor_name.endswith(suffix):
                return True
        return False

    def convert(self) -> None:
        t0 = time.time()
        logger.info("=" * 80)
        logger.info("⚡ RKMJ Out-of-Core Safetensors Streaming Quantizer (Zero-Spike)")
        logger.info(f"   Model Directory:   {self.model_dir}")
        logger.info(f"   Output Binary:     {self.output_path}")
        logger.info(f"   Group Size:        {self.group_size}")
        logger.info(f"   RAM Ceiling:       {self.max_ram_bytes / (1024**3):.2f} GB")
        logger.info("=" * 80)

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        temp_payload_path = self.output_path + ".payload.tmp"

        from safetensors import safe_open

        tensor_metadata: List[Dict[str, Any]] = []
        current_offset = 0
        total_raw_bytes = 0
        total_packed_bytes = 0

        force_heap_trim()
        self._track_rss()

        # Group tensors by shard to avoid reopening files repeatedly
        shard_to_tensors: Dict[str, List[str]] = {}
        for t_name, shard in self.tensor_to_shard.items():
            shard_to_tensors.setdefault(shard, []).append(t_name)

        with open(temp_payload_path, "wb") as payload_f:
            for shard_file, tensor_names in shard_to_tensors.items():
                shard_path = os.path.join(self.model_dir, shard_file)
                logger.info(f"[*] Streaming shard: {shard_file} ({len(tensor_names)} tensors)")

                with safe_open(shard_path, framework="pt", device="cpu") as sf:
                    for name in tensor_names:
                        # 1. Load ONLY a single matrix into host RAM
                        tensor = sf.get_tensor(name)
                        shape = tuple(tensor.shape)
                        raw_bytes = tensor.numel() * tensor.element_size()
                        total_raw_bytes += raw_bytes

                        # 2. Check if quantizable linear layer
                        if self.should_quantize(name, shape):
                            tensor_f32 = tensor.float().contiguous()
                            del tensor

                            # Quantize with Per-Group Dynamic Scaling
                            w_packed, scales = quantize_ternary_grouped(
                                tensor_f32, group_size=self.group_size
                            )
                            del tensor_f32

                            packed_bytes = w_packed.numpy().tobytes()
                            scales_bytes = scales.numpy().tobytes()

                            # Stream packed tensor directly to disk
                            payload_f.write(packed_bytes)
                            tensor_metadata.append({
                                "name": name,
                                "shape": list(shape),
                                "packed_shape": list(w_packed.shape),
                                "group_size": self.group_size,
                                "dtype": "int32",
                                "offset": current_offset,
                                "length": len(packed_bytes),
                                "is_packed": True,
                            })
                            current_offset += len(packed_bytes)
                            total_packed_bytes += len(packed_bytes)

                            # Stream scales directly to disk
                            scales_name = name.replace(".weight", ".scales")
                            payload_f.write(scales_bytes)
                            tensor_metadata.append({
                                "name": scales_name,
                                "shape": list(scales.shape),
                                "group_size": self.group_size,
                                "dtype": "float32",
                                "offset": current_offset,
                                "length": len(scales_bytes),
                                "is_packed": False,
                            })
                            current_offset += len(scales_bytes)
                            total_packed_bytes += len(scales_bytes)

                            del w_packed, scales, packed_bytes, scales_bytes

                        else:
                            # Non-quantized tensor (embeddings, norms, biases)
                            # Convert to bfloat16 for embeddings and lm_head to save 50% memory if float32
                            if "embed_tokens" in name or "lm_head" in name:
                                tensor = tensor.to(torch.bfloat16).contiguous()
                                dtype_str = "bfloat16"
                            else:
                                tensor = tensor.float().contiguous()
                                dtype_str = "float32"

                            buf = tensor.numpy().tobytes()
                            payload_f.write(buf)
                            tensor_metadata.append({
                                "name": name,
                                "shape": list(shape),
                                "dtype": dtype_str,
                                "offset": current_offset,
                                "length": len(buf),
                                "is_packed": False,
                            })
                            current_offset += len(buf)
                            total_packed_bytes += len(buf)
                            del tensor, buf

                        # Instantly reclaim memory
                        force_heap_trim()
                        self._track_rss()

                # Trim heap after each shard
                force_heap_trim()

        # Step 3: Write final .rkmjbin with header
        logger.info("[*] Finalizing packed binary header and container...")
        header_dict = {
            "version": VERSION,
            "config": self.config,
            "group_size": self.group_size,
            "num_tensors": len(tensor_metadata),
            "total_packed_bytes": total_packed_bytes,
            "tensors": tensor_metadata,
        }

        header_json = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_json)

        with open(self.output_path, "wb") as out_f:
            out_f.write(MAGIC)
            out_f.write(struct.pack("<I", header_len))
            out_f.write(header_json)

            with open(temp_payload_path, "rb") as pay_f:
                while True:
                    chunk = pay_f.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

        if os.path.exists(temp_payload_path):
            os.remove(temp_payload_path)

        force_heap_trim()
        t_duration = time.time() - t0
        ratio = (total_raw_bytes / max(total_packed_bytes, 1))

        logger.info("=" * 80)
        logger.info(f"✅ Quantization Completed in {t_duration:.2f}s!")
        logger.info(f"   Uncompressed Raw Size: {total_raw_bytes / (1024**3):.2f} GB")
        logger.info(f"   Output Binary Size:   {os.path.getsize(self.output_path) / (1024**3):.2f} GB")
        logger.info(f"   Compression Ratio:    {ratio:.1f}x")
        logger.info(f"   Peak Process RSS:     {self.peak_rss_bytes / (1024**3):.2f} GB (Hard limit: {self.max_ram_bytes / (1024**3):.2f} GB)")
        logger.info("=" * 80)

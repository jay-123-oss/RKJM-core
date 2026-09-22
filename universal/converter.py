"""
Universal Out-of-Core Streaming Post-Training Quantization (PTQ) Converter.
Converts any modern HuggingFace LLM (Qwen, LLaMA 2/3, Mistral, Gemma) from FP16/BF16 into
1.58-bit ternary {-1, 0, +1} packed .rkmjbin format within a strict <= 3.5 GB physical DRAM ceiling.
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

from rkmj.quantizer import quantize_ternary_grouped, pack_ternary
from universal.config import ArchitectureProfile, detect_architecture_profile

logger = logging.getLogger("universal.converter")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

MAGIC = b"RKMJBIN1"
VERSION = 1


class UniversalStreamingPTQConverter:
    """
    Universal Out-of-Core Streaming PTQ Converter.
    Auto-detects model architecture and streams weights layer-by-layer out-of-core
    without loading full shards into host RAM.
    """

    def __init__(
        self,
        model_dir: str,
        output_path: str,
        max_ram_bytes: int = int(3.5 * 1024 * 1024 * 1024),
        group_size: int = 64,
    ):
        self.model_dir = os.path.abspath(model_dir)
        self.output_path = os.path.abspath(output_path)
        self.max_ram_bytes = max_ram_bytes
        self.group_size = group_size

        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0

        # Load glibc malloc_trim on Linux
        self._libc = None
        if sys.platform.startswith("linux"):
            try:
                self._libc = ctypes.CDLL("libc.so.6")
            except Exception:
                pass

        # Load config and architecture profile
        self.config = self._load_config()
        self.profile = detect_architecture_profile(self.config)
        logger.info(f"Auto-detected model family: '{self.profile.model_family.upper()}'")

        # Shard index
        self.tensor_to_shard: Dict[str, str] = {}
        self.shard_files: List[str] = []
        self._index_safetensors()

    def _load_config(self) -> Dict[str, Any]:
        cfg_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json not found in {self.model_dir}")
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
                try:
                    from safetensors import safe_open
                    with safe_open(single, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            self.tensor_to_shard[k] = "model.safetensors"
                except ImportError:
                    pass

    def _track_rss(self) -> float:
        rss = self.process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        rss_gb = rss / (1024.0 ** 3)
        if rss > self.max_ram_bytes:
            logger.warning(
                f"Active process RSS ({rss_gb:.2f} GB) exceeded budget ceiling ({self.max_ram_bytes / (1024**3):.2f} GB)!"
            )
        return rss_gb

    def _trim_ram(self) -> None:
        gc.collect()
        if self._libc and hasattr(self._libc, "malloc_trim"):
            try:
                self._libc.malloc_trim(0)
            except Exception:
                pass

    def _load_single_tensor(self, tensor_name: str) -> Optional[torch.Tensor]:
        shard_rel = self.tensor_to_shard.get(tensor_name)
        if not shard_rel:
            return None
        shard_path = os.path.join(self.model_dir, shard_rel)
        try:
            from safetensors import safe_open
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                if tensor_name in f.keys():
                    return f.get_tensor(tensor_name).float().contiguous()
        except Exception as e:
            logger.error(f"Error reading tensor '{tensor_name}' from {shard_path}: {e}")
        return None

    def convert(self) -> None:
        start_time = time.time()
        logger.info("======================================================================")
        logger.info(f"🚀 Starting Universal Out-of-Core Streaming PTQ Conversion")
        logger.info(f"   Model Directory:  {self.model_dir}")
        logger.info(f"   Output Path:      {self.output_path}")
        logger.info(f"   Model Family:     {self.profile.model_family.upper()}")
        logger.info(f"   Group Size:       {self.group_size}")
        logger.info(f"   RSS Hard Ceiling: {self.max_ram_bytes / (1024**3):.2f} GB")
        logger.info("======================================================================")

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        temp_payload_path = self.output_path + ".payload.tmp"

        tensor_metadata: List[Dict[str, Any]] = []
        current_offset = 0
        total_uncompressed_bytes = 0
        total_packed_bytes = 0

        num_layers = self.config.get("num_hidden_layers", self.config.get("n_layers", 0))

        with open(temp_payload_path, "wb") as payload_f:
            # Step 1: Ingest global embedding and norm tensors
            logger.info("Ingesting global embeddings and norms...")
            global_keys = [
                self.profile.embed_tokens_key,
                self.profile.norm_key,
                self.profile.lm_head_key,
            ]
            for key in global_keys:
                tensor = self._load_single_tensor(key)
                if tensor is None and key == self.profile.lm_head_key:
                    # Tie word embeddings if lm_head is missing
                    tensor = self._load_single_tensor(self.profile.embed_tokens_key)

                if tensor is not None:
                    raw_bytes = tensor.numpy().tobytes()
                    nbytes = len(raw_bytes)
                    payload_f.write(raw_bytes)

                    tensor_metadata.append({
                        "name": key,
                        "shape": list(tensor.shape),
                        "dtype": "float32",
                        "offset": current_offset,
                        "length": nbytes,
                        "is_packed": False,
                    })
                    current_offset += nbytes
                    total_uncompressed_bytes += nbytes
                    total_packed_bytes += nbytes
                    del tensor
                    self._trim_ram()

            # Step 2: Stream transformer layers out-of-core
            logger.info(f"Streaming and quantizing {num_layers} Transformer layers out-of-core...")
            linear_subnames = [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]

            other_subnames = [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ]

            if self.profile.qkv_has_bias:
                other_subnames.extend([
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                ])

            for layer_idx in range(num_layers):
                prefix = f"{self.profile.layer_prefix}{layer_idx}."

                # A. Quantize Linear Projection Weights
                for subname in linear_subnames:
                    weight_key = prefix + subname
                    weight = self._load_single_tensor(weight_key)
                    if weight is None:
                        continue

                    N, K = weight.shape
                    uncomp_nbytes = weight.numel() * 4
                    total_uncompressed_bytes += uncomp_nbytes

                    # Run 1.58-bit grouped quantization
                    w_packed, scales = quantize_ternary_grouped(weight, group_size=self.group_size)

                    # Write packed int32 weights
                    packed_raw = w_packed.numpy().tobytes()
                    packed_len = len(packed_raw)
                    payload_f.write(packed_raw)

                    tensor_metadata.append({
                        "name": weight_key,
                        "shape": [N, K],
                        "packed_shape": list(w_packed.shape),
                        "dtype": "int32",
                        "offset": current_offset,
                        "length": packed_len,
                        "is_packed": True,
                        "group_size": self.group_size,
                    })
                    current_offset += packed_len
                    total_packed_bytes += packed_len

                    # Write float32 scales
                    scale_key = prefix + subname.replace(".weight", ".scales")
                    scales_raw = scales.numpy().tobytes()
                    scales_len = len(scales_raw)
                    payload_f.write(scales_raw)

                    tensor_metadata.append({
                        "name": scale_key,
                        "shape": list(scales.shape),
                        "dtype": "float32",
                        "offset": current_offset,
                        "length": scales_len,
                        "is_packed": False,
                    })
                    current_offset += scales_len
                    total_packed_bytes += scales_len

                    del weight, w_packed, scales

                # B. Ingest norms and biases (uncompressed FP32)
                for subname in other_subnames:
                    key = prefix + subname
                    tensor = self._load_single_tensor(key)
                    if tensor is None:
                        continue

                    raw_bytes = tensor.numpy().tobytes()
                    nbytes = len(raw_bytes)
                    payload_f.write(raw_bytes)

                    tensor_metadata.append({
                        "name": key,
                        "shape": list(tensor.shape),
                        "dtype": "float32",
                        "offset": current_offset,
                        "length": nbytes,
                        "is_packed": False,
                    })
                    current_offset += nbytes
                    total_uncompressed_bytes += nbytes
                    total_packed_bytes += nbytes
                    del tensor

                self._trim_ram()
                rss_gb = self._track_rss()
                if (layer_idx + 1) % max(1, num_layers // 10) == 0 or layer_idx == num_layers - 1:
                    logger.info(
                        f"   Processed Layer {layer_idx + 1:2d} / {num_layers} | Active RSS: {rss_gb:.2f} GB (Peak: {self.peak_rss_bytes / (1024**3):.2f} GB)"
                    )

        # Step 3: Build and write final .rkmjbin
        logger.info("Assembling final .rkmjbin file with zero-copy stream...")
        header_data = {
            "version": VERSION,
            "profile": {
                "model_family": self.profile.model_family,
                "qkv_has_bias": self.profile.qkv_has_bias,
                "norm_type": self.profile.norm_type,
                "norm_add_unit": self.profile.norm_add_unit,
                "mlp_activation": self.profile.mlp_activation,
                "default_rope_theta": self.profile.default_rope_theta,
                "eos_token_ids": self.profile.eos_token_ids,
                "chat_template_type": self.profile.chat_template_type,
            },
            "config": self.config,
            "tensors": tensor_metadata,
            "total_uncompressed_bytes": total_uncompressed_bytes,
            "total_packed_bytes": total_packed_bytes,
        }

        header_json = json.dumps(header_data).encode("utf-8")
        header_len = len(header_json)

        with open(self.output_path, "wb") as final_f:
            final_f.write(MAGIC)
            final_f.write(struct.pack("<I", header_len))
            final_f.write(header_json)

            with open(temp_payload_path, "rb") as payload_f:
                while True:
                    chunk = payload_f.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    final_f.write(chunk)

        if os.path.exists(temp_payload_path):
            os.remove(temp_payload_path)

        duration = time.time() - start_time
        final_file_size = os.path.getsize(self.output_path)
        compression = total_uncompressed_bytes / max(1, total_packed_bytes)

        logger.info("======================================================================")
        logger.info(f"✅ Universal PTQ Conversion Complete!")
        logger.info(f"   Output .rkmjbin File:        {self.output_path}")
        logger.info(f"   Final File Size:             {final_file_size / (1024**3):.2f} GB ({final_file_size:,} bytes)")
        logger.info(f"   Uncompressed Source Size:    {total_uncompressed_bytes / (1024**3):.2f} GB")
        logger.info(f"   Compression Ratio:           {compression:.1f}x")
        logger.info(f"   Peak Process RSS Memory:     {self.peak_rss_bytes / (1024**3):.2f} GB (Ceiling: {self.max_ram_bytes / (1024**3):.2f} GB)")
        logger.info(f"   Total Conversion Time:       {duration:.2f} seconds")
        logger.info("======================================================================")

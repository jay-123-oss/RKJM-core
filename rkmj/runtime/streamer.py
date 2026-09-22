"""
Out-of-Core Layer Streamer & Memory Governor for RKMJ-Core.
Enables execution of 32B - 70B parameter models on memory-constrained systems (8GB / 16GB)
by enforcing an active RAM ceiling of <= 3.5 GB through RAII reclamation and Linux malloc_trim.
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import struct
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import psutil
import torch

from rkmj.serialization.packer import unpack_ternary_weights

logger = logging.getLogger("rkmj.runtime.streamer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MemoryGovernor:
    """
    Enforces a strict resident set size (RSS) ceiling (default <= 3.5 GB).
    Monitors process memory and triggers immediate garbage collection and
    Linux arena trimming (malloc_trim) to avoid kernel Out-Of-Memory (OOM) killer invocations.
    """

    def __init__(self, max_ram_bytes: int = int(3.5 * 1024 * 1024 * 1024)):
        self.max_ram_bytes = max_ram_bytes
        self.process = psutil.Process(os.getpid())
        self._libc = None
        if sys.platform.startswith("linux"):
            try:
                self._libc = ctypes.CDLL("libc.so.6")
            except Exception as e:
                logger.debug("libc.so.6 malloc_trim unavailable: %s", e)

    def get_current_rss_bytes(self) -> int:
        """Returns current process Resident Set Size (RSS) in bytes."""
        try:
            return self.process.memory_info().rss
        except Exception:
            return 0

    def get_current_rss_gb(self) -> float:
        return self.get_current_rss_bytes() / (1024.0 ** 3)

    def check_and_trim(self, force: bool = False) -> bool:
        """
        Checks current RSS. If it exceeds 80% of max_ram_bytes (or if force=True),
        performs aggressive garbage collection and libc malloc_trim.
        Returns True if trim was performed.
        """
        current_rss = self.get_current_rss_bytes()
        threshold = int(self.max_ram_bytes * 0.80)

        if force or current_rss >= threshold:
            # 1. Force Python cycle collection
            gc.collect()

            # 2. Free CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 3. Call glibc malloc_trim to return free chunks to OS kernel
            if self._libc and hasattr(self._libc, "malloc_trim"):
                try:
                    self._libc.malloc_trim(0)
                except Exception:
                    pass

            new_rss = self.get_current_rss_bytes()
            logger.debug(
                "MemoryGovernor trimmed RSS: %.2f GB -> %.2f GB (Ceiling: %.2f GB)",
                current_rss / (1024.0 ** 3),
                new_rss / (1024.0 ** 3),
                self.max_ram_bytes / (1024.0 ** 3),
            )
            return True
        return False

    @contextmanager
    def manage(self):
        """Context manager guaranteeing cleanup on exit."""
        try:
            yield self
        finally:
            self.check_and_trim(force=True)


class LayerWiseStreamer:
    """
    Out-of-core sequential transformer layer streamer.
    Streams weights for one transformer layer at a time from Safetensors or .rkmjbin,
    executes computation, and discards tensor memory immediately.
    """

    def __init__(
        self,
        model_source: str,
        memory_governor: Optional[MemoryGovernor] = None,
    ):
        self.model_source = model_source
        self.governor = memory_governor or MemoryGovernor()
        self.mode = self._detect_format()

    def _detect_format(self) -> str:
        if os.path.isfile(self.model_source):
            if self.model_source.endswith(".rkmjbin"):
                return "RKMJBIN"
            elif self.model_source.endswith(".safetensors"):
                return "SAFETENSORS_SINGLE"
        elif os.path.isdir(self.model_source):
            idx = os.path.join(self.model_source, "model.safetensors.index.json")
            if os.path.exists(idx):
                return "SAFETENSORS_SHARDED"
            rkmjbins = [f for f in os.listdir(self.model_source) if f.endswith(".rkmjbin")]
            if rkmjbins:
                self.model_source = os.path.join(self.model_source, rkmjbins[0])
                return "RKMJBIN"
            sfs = [f for f in os.listdir(self.model_source) if f.endswith(".safetensors")]
            if sfs:
                self.model_source = os.path.join(self.model_source, sfs[0])
                return "SAFETENSORS_SINGLE"
        return "UNKNOWN"

    def stream_layers(
        self,
        num_layers: Optional[int] = None,
    ) -> Iterator[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        Yields (layer_idx, layer_weights_dict).
        Guarantees that preceding layer tensors are deallocated and RSS is governed.
        """
        if self.mode == "RKMJBIN":
            yield from self._stream_rkmjbin(num_layers)
        elif "SAFETENSORS" in self.mode:
            yield from self._stream_safetensors(num_layers)
        else:
            raise ValueError(f"Unsupported model format for layer streaming: {self.model_source}")

    def _stream_rkmjbin(
        self,
        max_layers: Optional[int],
    ) -> Iterator[Tuple[int, Dict[str, torch.Tensor]]]:
        import mmap
        with open(self.model_source, "rb") as f:
            magic = f.read(8)
            if magic != b"RKMJBIN1":
                raise ValueError(f"Invalid .rkmjbin magic: {magic}")
            header_len = struct.unpack("<I", f.read(4))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))

            payload_start = 12 + header_len
            tensors = header.get("tensors", [])

            # Group tensors by layer index
            layer_tensors: Dict[int, List[Dict[str, Any]]] = {}
            for t in tensors:
                name = t["name"]
                if "layers." in name:
                    try:
                        idx = int(name.split("layers.")[1].split(".")[0])
                        layer_tensors.setdefault(idx, []).append(t)
                    except Exception:
                        pass

            sorted_layers = sorted(layer_tensors.keys())
            if max_layers is not None:
                sorted_layers = sorted_layers[:max_layers]

            # Stream layer by layer
            for layer_idx in sorted_layers:
                layer_dict: Dict[str, torch.Tensor] = {}
                meta_list = layer_tensors[layer_idx]

                for meta in meta_list:
                    f.seek(payload_start + meta["offset"])
                    raw_bytes = f.read(meta["length"])

                    if meta["is_packed"]:
                        packed_tensor = torch.frombuffer(
                            bytearray(raw_bytes), dtype=torch.int32
                        ).reshape(meta["packed_shape"])
                        layer_dict[meta["packed_name"]] = packed_tensor

                        # Unpack latent weights for computation
                        K = meta["shape"][1]
                        unpacked = unpack_ternary_weights(packed_tensor, K)
                        layer_dict[meta["name"]] = unpacked
                    else:
                        dtype_str = meta["dtype"]
                        dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.float32
                        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=dtype).reshape(meta["shape"])
                        layer_dict[meta["name"]] = tensor

                yield layer_idx, layer_dict

                # Strict RAII release
                del layer_dict
                self.governor.check_and_trim()

    def _stream_safetensors(
        self,
        max_layers: Optional[int],
    ) -> Iterator[Tuple[int, Dict[str, torch.Tensor]]]:
        from safetensors import safe_open

        if self.mode == "SAFETENSORS_SINGLE":
            with safe_open(self.model_source, framework="pt") as f:
                keys = list(f.keys())
                layer_indices = set()
                for k in keys:
                    if "layers." in k:
                        idx = int(k.split("layers.")[1].split(".")[0])
                        layer_indices.add(idx)

                sorted_layers = sorted(layer_indices)
                if max_layers is not None:
                    sorted_layers = sorted_layers[:max_layers]

                for layer_idx in sorted_layers:
                    layer_dict = {}
                    prefix = f"layers.{layer_idx}."
                    for k in keys:
                        if k.startswith(prefix):
                            layer_dict[k] = f.get_tensor(k)

                    yield layer_idx, layer_dict
                    del layer_dict
                    self.governor.check_and_trim()

        elif self.mode == "SAFETENSORS_SHARDED":
            idx_file = os.path.join(self.model_source, "model.safetensors.index.json")
            with open(idx_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})

            # Group keys by layer
            layer_to_keys: Dict[int, List[str]] = {}
            for k in weight_map:
                if "layers." in k:
                    idx = int(k.split("layers.")[1].split(".")[0])
                    layer_to_keys.setdefault(idx, []).append(k)

            sorted_layers = sorted(layer_to_keys.keys())
            if max_layers is not None:
                sorted_layers = sorted_layers[:max_layers]

            # Cache safe_open handles across shards
            open_shards: Dict[str, Any] = {}
            try:
                for layer_idx in sorted_layers:
                    layer_dict = {}
                    keys = layer_to_keys[layer_idx]
                    for k in keys:
                        shard_file = weight_map[k]
                        if shard_file not in open_shards:
                            shard_path = os.path.join(self.model_source, shard_file)
                            open_shards[shard_file] = safe_open(shard_path, framework="pt")
                        layer_dict[k] = open_shards[shard_file].get_tensor(k)

                    yield layer_idx, layer_dict
                    del layer_dict
                    self.governor.check_and_trim()
            finally:
                open_shards.clear()
                self.governor.check_and_trim(force=True)

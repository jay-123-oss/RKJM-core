"""
Hardware Profiler & Execution Tier Decision Engine for RKMJ-Core.
Autonomously inspects host hardware resources (RAM, SIMD, Storage I/O) against
arbitrary model scale (1B to 256B+) to dynamically route execution tiers without OOMs.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import platform
import subprocess
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import psutil

logger = logging.getLogger("rkmj.runtime.profiler")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ExecutionTier(str, Enum):
    """Execution tier selected based on model scale and host memory headroom."""
    IN_RAM = "IN_RAM"                         # Fits in < 40% available RAM. Pinned contiguous DRAM.
    BALANCED_MMAP = "BALANCED_MMAP"           # Fits in 40% - 90% available RAM. Zero-copy mmap with page cache.
    LAYER_STREAM = "LAYER_STREAM"             # Exceeds 90% RAM. Out-of-core layer streaming capped at <= 3.5 GB.
    DOUBLE_BUFFERED_RING = "DOUBLE_BUFFERED_RING"  # Extreme scales (100B+) / NVMe: Overlapped compute & I/O.


class SystemHardwareProfiler:
    """Profiles host memory, CPU instruction set extensions, and storage media throughput."""

    @staticmethod
    def get_total_ram_gb() -> float:
        """Returns physical RAM in gigabytes."""
        return psutil.virtual_memory().total / (1024.0 ** 3)

    @staticmethod
    def get_available_ram_gb(safety_factor: float = 0.75) -> float:
        """
        Returns safe available RAM in gigabytes after applying safety headroom.
        Default safety factor is 0.75 (reserving 25% for OS and active buffers).
        """
        avail = psutil.virtual_memory().available / (1024.0 ** 3)
        return max(0.25, avail * safety_factor)

    @staticmethod
    def detect_cpu_features() -> Dict[str, bool]:
        """Detects SIMD vector extensions on x86_64 or ARM64."""
        flags = {
            "avx512": False,
            "avx512_vpopcntdq": False,
            "avx2": False,
            "bmi2": False,
            "neon": False,
        }

        arch = platform.machine().lower()
        if arch in ("x86_64", "amd64"):
            if os.path.exists("/proc/cpuinfo"):
                try:
                    with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                        cpuinfo = f.read().lower()
                        flags["avx512"] = "avx512f" in cpuinfo
                        flags["avx512_vpopcntdq"] = "avx512_vpopcntdq" in cpuinfo
                        flags["avx2"] = "avx2" in cpuinfo
                        flags["bmi2"] = "bmi2" in cpuinfo
                except Exception:
                    pass
        elif arch in ("aarch64", "arm64"):
            flags["neon"] = True

        # Check compiled engine backend if available
        try:
            import rkmj._C as _C
            if hasattr(_C, "get_simd_backend_name"):
                backend = _C.get_simd_backend_name().lower()
                if "avx-512" in backend:
                    flags["avx512"] = True
                if "avx2" in backend:
                    flags["avx2"] = True
                if "neon" in backend:
                    flags["neon"] = True
        except ImportError:
            pass

        return flags

    @staticmethod
    def detect_storage_profile(target_path: str = ".") -> Dict[str, Any]:
        """
        Classifies storage media throughput (NVME, SATA_SSD, or HDD) for the target path.
        """
        abs_path = os.path.abspath(target_path)
        profile = {
            "path": abs_path,
            "type": "SATA_SSD",
            "is_fast_storage": True,
            "read_speed_mb_s": 500.0,
        }

        # 1. Inspect Linux sysfs rotational flag if available
        try:
            stat_res = os.stat(abs_path)
            major = os.major(stat_res.st_dev)
            minor = os.minor(stat_res.st_dev)
            rot_path = f"/sys/dev/block/{major}:{minor}/queue/rotational"
            if os.path.exists(rot_path):
                with open(rot_path, "r") as f:
                    is_rotational = (f.read().strip() == "1")
                    if is_rotational:
                        profile["type"] = "HDD"
                        profile["is_fast_storage"] = False
                        profile["read_speed_mb_s"] = 120.0
                        return profile
        except Exception:
            pass

        # 2. Heuristic based on device name or mount point
        if "nvme" in abs_path.lower():
            profile["type"] = "NVME"
            profile["is_fast_storage"] = True
            profile["read_speed_mb_s"] = 2500.0
            return profile

        # 3. Quick non-destructive micro-benchmark (read small existing file if available)
        try:
            check_file = abs_path if os.path.isfile(abs_path) else None
            if check_file is None:
                for root, _, files in os.walk(abs_path):
                    for file in files:
                        p = os.path.join(root, file)
                        if os.path.getsize(p) >= 1024 * 1024:
                            check_file = p
                            break
                    if check_file:
                        break

            if check_file and os.path.getsize(check_file) >= 1024 * 1024:
                file_size = min(os.path.getsize(check_file), 8 * 1024 * 1024)
                t0 = time.perf_counter()
                with open(check_file, "rb") as f:
                    _ = f.read(file_size)
                elapsed = max(time.perf_counter() - t0, 1e-6)
                mb_s = (file_size / (1024.0 * 1024.0)) / elapsed
                profile["read_speed_mb_s"] = mb_s
                if mb_s > 1000.0:
                    profile["type"] = "NVME"
                elif mb_s > 250.0:
                    profile["type"] = "SATA_SSD"
                else:
                    profile["type"] = "HDD"
                    profile["is_fast_storage"] = False
        except Exception:
            pass

        return profile


class ModelFootprintEstimator:
    """
    Analyzes model configurations, Safetensors indices, and .rkmjbin headers
    to determine parameter counts, layer counts, and 1.58-bit footprint metrics.
    """

    @classmethod
    def estimate(cls, model_source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns model scale metadata:
          - total_params: int
          - num_layers: int
          - hidden_size: int
          - intermediate_size: int
          - total_bytes_1_58bit: int (bytes needed in 2-bit packed representation)
          - single_layer_bytes: int (bytes needed for 1 transformer layer)
        """
        if isinstance(model_source, dict):
            return cls._estimate_from_config_dict(model_source)

        if not os.path.exists(model_source):
            # Check if it's a model ID or string representation
            raise FileNotFoundError(f"Model source path does not exist: {model_source}")

        if os.path.isfile(model_source):
            if model_source.endswith(".rkmjbin"):
                return cls._estimate_from_rkmjbin(model_source)
            elif model_source.endswith(".safetensors"):
                return cls._estimate_from_safetensors_file(model_source)
            elif model_source.endswith(".json"):
                with open(model_source, "r", encoding="utf-8") as f:
                    return cls._estimate_from_config_dict(json.load(f))

        # Model directory
        rkmjbin_files = glob.glob(os.path.join(model_source, "*.rkmjbin"))
        if rkmjbin_files:
            return cls._estimate_from_rkmjbin(rkmjbin_files[0])

        index_file = os.path.join(model_source, "model.safetensors.index.json")
        if os.path.exists(index_file):
            return cls._estimate_from_safetensors_index(index_file, model_source)

        config_file = os.path.join(model_source, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                return cls._estimate_from_config_dict(cfg_data)

        # Fallback: sum all file sizes in directory
        total_file_bytes = sum(
            os.path.getsize(os.path.join(root, file))
            for root, _, files in os.walk(model_source)
            for file in files
        )
        # Approximate 2 bytes per param if uncompressed fp16
        approx_params = total_file_bytes // 2
        return cls._compute_footprint_metrics(
            total_params=approx_params,
            num_layers=32,
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
        )

    @classmethod
    def _estimate_from_rkmjbin(cls, filepath: str) -> Dict[str, Any]:
        """Parse header of .rkmjbin file to extract metadata."""
        import struct
        with open(filepath, "rb") as f:
            magic = f.read(8)
            if magic != b"RKMJBIN1":
                raise ValueError(f"Invalid .rkmjbin magic: {magic}")
            header_len = struct.unpack("<I", f.read(4))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))

        config = header.get("config", {})
        tensors = header.get("tensors", [])

        total_packed_bytes = sum(t["length"] for t in tensors)
        num_layers = config.get("n_layers", 16)
        hidden_size = config.get("dim", 2048)
        intermediate_size = config.get("intermediate_dim", hidden_size * 4)
        vocab_size = config.get("vocab_size", 32000)

        # Estimate parameters from tensor shapes
        total_params = sum(
            int(abs(math_prod(t["shape"])))
            for t in tensors
            if "shape" in t
        )
        if total_params == 0:
            total_params = total_packed_bytes * 4

        single_layer_bytes = total_packed_bytes // max(num_layers, 1)
        return {
            "total_params": total_params,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "total_bytes_1_58bit": total_packed_bytes,
            "single_layer_bytes": single_layer_bytes,
        }

    @classmethod
    def _estimate_from_safetensors_index(cls, index_path: str, model_dir: str) -> Dict[str, Any]:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        weight_map = data.get("weight_map", {})
        metadata = data.get("metadata", {})
        total_params = metadata.get("total_size")

        config_path = os.path.join(model_dir, "config.json")
        cfg_data = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)

        if total_params is None:
            # Infer from layer count in weight_map
            layer_indices = set()
            for key in weight_map:
                if "layers." in key:
                    try:
                        part = key.split("layers.")[1].split(".")[0]
                        layer_indices.add(int(part))
                    except Exception:
                        pass
            num_layers = len(layer_indices) if layer_indices else cfg_data.get("num_hidden_layers", 32)
        else:
            num_layers = cfg_data.get("num_hidden_layers", 32)

        hidden_size = cfg_data.get("hidden_size", 4096)
        intermediate_size = cfg_data.get("intermediate_size", int(hidden_size * 8 / 3))
        vocab_size = cfg_data.get("vocab_size", 32000)

        return cls._compute_footprint_metrics(
            total_params=total_params or (num_layers * hidden_size * hidden_size * 12),
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
        )

    @classmethod
    def _estimate_from_safetensors_file(cls, filepath: str) -> Dict[str, Any]:
        import struct
        with open(filepath, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))

        total_params = 0
        layer_indices = set()
        for k, v in header.items():
            if k == "__metadata__":
                continue
            shape = v.get("shape", [])
            total_params += math_prod(shape)
            if "layers." in k:
                try:
                    part = k.split("layers.")[1].split(".")[0]
                    layer_indices.add(int(part))
                except Exception:
                    pass

        num_layers = len(layer_indices) if layer_indices else 32
        return cls._compute_footprint_metrics(
            total_params=total_params,
            num_layers=num_layers,
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
        )

    @classmethod
    def _estimate_from_config_dict(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layers") or 32
        hidden_size = cfg.get("hidden_size") or cfg.get("dim") or 4096
        intermediate_size = cfg.get("intermediate_size") or cfg.get("intermediate_dim") or int(hidden_size * 8 / 3)
        vocab_size = cfg.get("vocab_size") or 32000

        # Calculate exact transformer parameters:
        # Per layer:
        # - Q, K, V, O projections: 4 * (hidden_size * hidden_size)
        # - MLP gate, up, down projections: 3 * (hidden_size * intermediate_size)
        # - Norms: 2 * hidden_size
        per_layer_params = 4 * (hidden_size * hidden_size) + 3 * (hidden_size * intermediate_size) + 2 * hidden_size
        embed_params = vocab_size * hidden_size
        lm_head_params = vocab_size * hidden_size
        total_params = embed_params + (num_layers * per_layer_params) + lm_head_params

        return cls._compute_footprint_metrics(
            total_params=total_params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
        )

    @classmethod
    def _compute_footprint_metrics(
        cls,
        total_params: int,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        vocab_size: int,
    ) -> Dict[str, Any]:
        # 1.58-bit packed weights: 2 bits per weight = 0.25 bytes
        # Latent/Alpha scale factors + norms + embeddings in FP32 / FP16 = ~0.05 bytes/param
        total_bytes_1_58bit = int(total_params * 0.28)
        single_layer_params = max(1, total_params // max(num_layers, 1))
        single_layer_bytes = int(single_layer_params * 0.28)

        return {
            "total_params": total_params,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "total_bytes_1_58bit": total_bytes_1_58bit,
            "single_layer_bytes": single_layer_bytes,
        }


def math_prod(iterable) -> int:
    """Product of elements in iterable."""
    res = 1
    for x in iterable:
        res *= int(x)
    return res


class DynamicExecutionRouter:
    """
    Compares model memory requirements against host hardware headroom.
    Automatically resolves optimal runtime tier and outputs diagnostic metrics.
    """

    @classmethod
    def resolve(
        cls,
        model_source: Union[str, Dict[str, Any]],
        override_tier: Optional[str] = None,
        safety_ram_gb: Optional[float] = None,
    ) -> Tuple[ExecutionTier, Dict[str, Any]]:
        """
        Dynamically select the execution tier.
        Returns: (ExecutionTier, diagnostic_info_dict)
        """
        profiler = SystemHardwareProfiler()
        total_ram_gb = profiler.get_total_ram_gb()
        safe_ram_gb = safety_ram_gb if safety_ram_gb is not None else profiler.get_available_ram_gb()
        simd_features = profiler.detect_cpu_features()

        target_dir = model_source if isinstance(model_source, str) else "."
        storage_profile = profiler.detect_storage_profile(target_dir)

        footprint = ModelFootprintEstimator.estimate(model_source)
        model_bytes = footprint["total_bytes_1_58bit"]
        model_gb = model_bytes / (1024.0 ** 3)
        single_layer_mb = footprint["single_layer_bytes"] / (1024.0 * 1024.0)
        total_params_b = footprint["total_params"] / 1e9

        # Selection algorithm
        tier: ExecutionTier
        rationale: str

        if override_tier:
            try:
                tier = ExecutionTier(override_tier.upper())
                rationale = f"User explicit override to {tier.value}."
            except ValueError:
                logger.warning("Invalid override_tier '%s', resolving automatically.", override_tier)
                tier = None

        if not override_tier or tier is None:
            # 1. Tier 1 (IN_RAM): Small Models (< 40% of available safe RAM)
            if model_gb <= (safe_ram_gb * 0.40):
                tier = ExecutionTier.IN_RAM
                rationale = (
                    f"Model footprint ({model_gb:.2f} GB) fits comfortably in available RAM "
                    f"({model_gb:.2f} GB <= 40% of safe RAM {safe_ram_gb:.2f} GB). Pinned DRAM selected."
                )
            # 2. Tier 2 (BALANCED_MMAP): Medium Models (40% - 90% of available safe RAM)
            elif model_gb <= (safe_ram_gb * 0.90):
                tier = ExecutionTier.BALANCED_MMAP
                rationale = (
                    f"Model footprint ({model_gb:.2f} GB) fits in RAM working set "
                    f"(40%-90% of safe RAM {safe_ram_gb:.2f} GB). Zero-copy MMAP selected."
                )
            # 3. Tier 4 (DOUBLE_BUFFERED_RING): Extreme Models (>= 100B params)
            elif footprint["total_params"] >= 100_000_000_000:
                tier = ExecutionTier.DOUBLE_BUFFERED_RING
                rationale = (
                    f"Extreme model scale ({total_params_b:.1f}B params) detected. "
                    f"C++ double-buffered prefetch ring selected to overlap I/O with compute."
                )
            # 4. Tier 3 (LAYER_STREAM): Large Models (32B - 70B on constrained systems)
            else:
                tier = ExecutionTier.LAYER_STREAM
                rationale = (
                    f"Model footprint ({model_gb:.2f} GB) exceeds 90% of available RAM ({safe_ram_gb:.2f} GB). "
                    f"Out-of-core layer-by-layer streaming with strict <= 3.5 GB allocation ceiling selected."
                )

        diag = {
            "tier": tier.value,
            "rationale": rationale,
            "model_scale": {
                "total_params_billion": round(total_params_b, 2),
                "num_layers": footprint["num_layers"],
                "hidden_size": footprint["hidden_size"],
                "model_footprint_gb": round(model_gb, 3),
                "single_layer_mb": round(single_layer_mb, 2),
            },
            "host_hardware": {
                "total_physical_ram_gb": round(total_ram_gb, 2),
                "available_safe_ram_gb": round(safe_ram_gb, 2),
                "storage_type": storage_profile["type"],
                "storage_speed_mb_s": round(storage_profile["read_speed_mb_s"], 1),
                "simd_features": simd_features,
            },
        }

        logger.info("=" * 65)
        logger.info("🧠 RKMJ Dynamic Execution Tier Resolved: %s", tier.value)
        logger.info("   Model Scale:       %.2fB parameters (%d layers, %.2f GB packed)", total_params_b, footprint["num_layers"], model_gb)
        logger.info("   Available RAM:     %.2f GB / %.2f GB total", safe_ram_gb, total_ram_gb)
        logger.info("   Storage Profile:   %s (~%.0f MB/s)", storage_profile["type"], storage_profile["read_speed_mb_s"])
        logger.info("   Rationale:         %s", rationale)
        logger.info("=" * 65)

        return tier, diag

"""
Native Binary Serialization Format (.rkmjbin) for 1.58-bit RKMJ Models.

Features:
  - Stores 1.58-bit ternary weights as packed 2-bit uint32 arrays (15.8x compression ratio).
  - Header: Magic header b"RKMJBIN1", metadata JSON (tensor shapes, dtypes, offsets, config).
  - Body: Sequential binary payload of packed buffers and FP32 scale factors.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from rkmj.nn.linear import CSALinear

MAGIC = b"RKMJBIN1"
VERSION = 1


def save_rkmjbin(
    model: nn.Module,
    filepath: str,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> int:
    """
    Export an RKMJ model to the native .rkmjbin format.
    Automatically packs all CSALinear layers into 2-bit uint32 arrays.
    """
    model.eval()

    # Pre-pack all CSALinear layers
    for m in model.modules():
        if isinstance(m, CSALinear):
            m.pack_weights_for_inference()

    state_dict = model.state_dict()
    tensor_metadata = []
    payload_parts = []
    current_offset = 0

    total_uncompressed_bytes = 0
    total_packed_bytes = 0

    for name, tensor in state_dict.items():
        is_csa_weight = name.endswith("latent_weight")
        packed_name = name.replace("latent_weight", "w_packed") if is_csa_weight else None

        # If it's a latent weight and we have packed weights, serialize w_packed instead
        if is_csa_weight and packed_name in state_dict:
            packed_tensor = state_dict[packed_name].cpu().contiguous()
            raw_data = packed_tensor.numpy().tobytes()
            nbytes = len(raw_data)

            uncompressed_bytes = tensor.numel() * 4  # FP32 bytes
            total_uncompressed_bytes += uncompressed_bytes
            total_packed_bytes += nbytes

            meta = {
                "name": name,
                "packed_name": packed_name,
                "is_packed": True,
                "shape": list(tensor.shape),
                "packed_shape": list(packed_tensor.shape),
                "dtype": "int32_packed",
                "offset": current_offset,
                "length": nbytes,
            }
        elif name.endswith("w_packed"):
            # Already handled with latent_weight
            continue
        else:
            t = tensor.cpu().contiguous()
            raw_data = t.numpy().tobytes()
            nbytes = len(raw_data)
            total_uncompressed_bytes += nbytes
            total_packed_bytes += nbytes

            meta = {
                "name": name,
                "is_packed": False,
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "offset": current_offset,
                "length": nbytes,
            }

        tensor_metadata.append(meta)
        payload_parts.append(raw_data)
        current_offset += nbytes

    header_dict = {
        "version": VERSION,
        "config": config or {},
        "tensors": tensor_metadata,
        "total_payload_bytes": current_offset,
    }
    header_json = json.dumps(header_dict, indent=2).encode("utf-8")
    header_len = len(header_json)

    with open(filepath, "wb") as f:
        # 1. Magic (8 bytes)
        f.write(MAGIC)
        # 2. Header Length (uint32, 4 bytes)
        f.write(struct.pack("<I", header_len))
        # 3. Header JSON
        f.write(header_json)
        # 4. Binary Payload
        for chunk in payload_parts:
            f.write(chunk)

    file_size = os.path.getsize(filepath)
    if verbose:
        compression = total_uncompressed_bytes / max(total_packed_bytes, 1)
        print(f"Exported .rkmjbin model to: {filepath}")
        print(f"  - Total File Size:               {file_size:,} bytes")
        print(f"  - Uncompressed Weight Equivalent:{total_uncompressed_bytes:,} bytes")
        print(f"  - Weight Compression Factor:     {compression:.1f}x")

    return file_size


def load_rkmjbin(
    filepath: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load tensors and configuration from an .rkmjbin binary file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Invalid .rkmjbin file magic: {magic}")

        header_len = struct.unpack("<I", f.read(4))[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)

        payload_start = 8 + 4 + header_len
        state_dict = {}

        for meta in header["tensors"]:
            f.seek(payload_start + meta["offset"])
            raw = f.read(meta["length"])

            if meta["is_packed"]:
                # Reconstruct int32 packed tensor
                arr = torch.frombuffer(bytearray(raw), dtype=torch.int32)
                packed_tensor = arr.reshape(meta["packed_shape"]).to(device)
                state_dict[meta["packed_name"]] = packed_tensor

                # Unpack into latent_weight representation for initial model state
                K = meta["shape"][1]
                from rkmj.serialization.packer import unpack_ternary_weights
                unpacked = unpack_ternary_weights(packed_tensor, K)
                state_dict[meta["name"]] = unpacked
            else:
                dtype_name = meta["dtype"]
                dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else torch.float32
                arr = torch.frombuffer(bytearray(raw), dtype=dtype)
                state_dict[meta["name"]] = arr.reshape(meta["shape"]).to(device)

    return state_dict, header.get("config", {})

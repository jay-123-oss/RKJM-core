"""
Numerically Stable 1.58-Bit Quantization Pipeline for RKMJ-Core.
Implements Per-Group Dynamic Scaling (gamma_g = mean(|W_g|)) and dynamic zero-threshold (eps = 0.5 * gamma_g)
with strict 2-bit aligned SIMD packing into uint32 integers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch

try:
    import rkmj._C as _C
    HAS_CPP_PACK = hasattr(_C, "quantize_grouped")
except ImportError:
    _C = None
    HAS_CPP_PACK = False


def quantize_ternary_grouped(
    weight: torch.Tensor,
    group_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a 2D weight matrix [out_features, in_features] into 1.58-bit ternary {-1, 0, +1}
    using Per-Group Dynamic Scaling (gamma_g) and dynamic zero thresholding.

    Args:
        weight: Float32 / BFloat16 / Float16 2D weight tensor [N, K].
        group_size: Quantization group size along input dimension (default: 64).

    Returns:
        w_packed: Packed int32 bitfield [N, (K + 15) // 16].
        scales: Per-group float32 scaling factors [N, (K + group_size - 1) // group_size].
    """
    if weight.dim() != 2:
        raise ValueError(f"quantize_ternary_grouped expects 2D tensor, got shape {list(weight.shape)}")

    # Fast path via C++ multi-threaded kernel
    if HAS_CPP_PACK and _C is not None:
        try:
            return _C.quantize_grouped(weight, group_size)
        except Exception:
            pass

    # High-precision vectorized PyTorch fallback
    N, K = weight.shape
    w = weight.float().contiguous()
    num_groups = (K + group_size - 1) // group_size
    pad_len = num_groups * group_size - K

    if pad_len > 0:
        w_padded = torch.nn.functional.pad(w, (0, pad_len))
    else:
        w_padded = w

    # Reshape into groups: [N, num_groups, group_size]
    w_grouped = w_padded.view(N, num_groups, group_size)

    # 1. Per-group dynamic scale gamma_g = mean(|W_g|)
    gamma_g = w_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    threshold = 0.5 * gamma_g

    # 2. Dynamic ternary assignment with zero threshold
    w_ternary = torch.zeros_like(w_grouped)
    w_ternary = torch.where(w_grouped >= threshold, 1.0, w_ternary)
    w_ternary = torch.where(w_grouped <= -threshold, -1.0, w_ternary)

    # 3. Frobenius least-squares scale refinement: (W * W_t).sum() / (W_t^2).sum()
    dot = (w_grouped * w_ternary).sum(dim=-1)
    norm_sq = (w_ternary ** 2).sum(dim=-1).clamp(min=1e-6)
    scales = (dot / norm_sq).clamp(min=1e-5)

    # Slice back to original K dimension
    w_t_flat = w_ternary.view(N, -1)[:, :K].contiguous()

    # 4. Pack into uint32 bitfield
    w_packed = pack_ternary(w_t_flat)
    return w_packed, scales


def dequantize_ternary_grouped(
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    group_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantizes per-group packed weights into full precision continuous tensor:
    W_dequant = W_ternary * scale_g.
    """
    if HAS_CPP_PACK and _C is not None:
        try:
            return _C.dequantize_grouped(w_packed, scales, in_features, group_size).to(dtype)
        except Exception:
            pass

    w_t = unpack_ternary(w_packed, in_features)
    N, K = w_t.shape
    num_groups = scales.shape[1]

    # Expand per-group scales along K
    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :K]
    return (w_t * scales_expanded).to(dtype)


def pack_ternary(w_ternary: torch.Tensor) -> torch.Tensor:
    """
    Packs a 2D ternary matrix [-1, 0, +1] into 2-bit aligned uint32 words (16 weights/word).
    Bit codes: 00 = 0, 01 = +1, 10 = -1.
    """
    if _C is not None and hasattr(_C, "pack_weights_2bit"):
        try:
            return _C.pack_weights_2bit(w_ternary)
        except Exception:
            pass

    w = w_ternary.contiguous().float()
    N, K = w.shape
    num_words = (K + 15) // 16
    pad_k = num_words * 16 - K

    if pad_k > 0:
        w = torch.nn.functional.pad(w, (0, pad_k))

    # Map to 2-bit codes: 0 -> 0, +1 -> 1, -1 -> 2
    codes = torch.zeros_like(w, dtype=torch.int32)
    codes[w > 0.5] = 1
    codes[w < -0.5] = 2

    codes_reshaped = codes.view(N, num_words, 16)
    shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=w.device).view(1, 1, 16)
    packed = (codes_reshaped << shifts).sum(dim=-1, dtype=torch.int32)
    return packed


def unpack_ternary(
    w_packed: torch.Tensor,
    in_features: Optional[int] = None,
    K: Optional[int] = None,
) -> torch.Tensor:
    """
    Unpacks a 2D packed int32 bitfield back into a float32 ternary tensor [-1, 0, +1].
    """
    feat = in_features if in_features is not None else K
    if feat is None:
        raise ValueError("unpack_ternary requires in_features or K to be specified")

    if _C is not None and hasattr(_C, "unpack_weights_2bit"):
        try:
            return _C.unpack_weights_2bit(w_packed, feat)
        except Exception:
            pass

    N, num_words = w_packed.shape
    device = w_packed.device
    shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=device).view(1, 1, 16)

    # Broadcast unpack
    words = w_packed.unsqueeze(-1)  # [N, num_words, 1]
    codes = (words >> shifts) & 0x03

    unpacked = torch.zeros((N, num_words, 16), dtype=torch.float32, device=device)
    unpacked[codes == 1] = 1.0
    unpacked[codes == 2] = -1.0

    return unpacked.view(N, num_words * 16)[:, :in_features].contiguous()

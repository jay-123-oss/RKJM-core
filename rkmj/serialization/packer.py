"""
Bit-packing utilities for 1.58-bit (Ternary) weights.
Translates between FP32 ternary values {-1.0, 0.0, +1.0} and 2-bit packed uint32 integers.
"""

from __future__ import annotations

import torch


def quantize_to_ternary(
    weight: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize an FP32 weight matrix into ternary values {-1, 0, +1}
    using dynamic per-channel scale alpha = mean(|W|).
    """
    if alpha is None:
        alpha = weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    else:
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)
        alpha = alpha.clamp(min=1e-8)

    w_normalized = weight / alpha
    w_ternary = torch.clamp(torch.round(w_normalized), -1.0, 1.0)
    return w_ternary, alpha.squeeze(1)


def pack_ternary_weights(w_ternary: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch fallback for packing a 2D [N, K] ternary matrix into 2-bit uint32 words.
    Each 32-bit word stores 16 weights (2 bits each: 00=0, 01=+1, 10=-1).
    Handles non-divisible channel dimensions without memory misalignment.
    """
    w_ternary = w_ternary.contiguous()
    N, K = w_ternary.shape
    num_words = (K + 15) // 16
    packed = torch.zeros((N, num_words), dtype=torch.int32, device=w_ternary.device)

    # Map -1.0 -> 2, 0.0 -> 0, +1.0 -> 1
    codes = torch.zeros_like(w_ternary, dtype=torch.int64)
    codes[w_ternary > 0.5] = 1
    codes[w_ternary < -0.5] = 2

    for i in range(16):
        if i >= K:
            break
        cols = torch.arange(i, K, 16, device=w_ternary.device)
        if len(cols) > 0:
            shift = 2 * i
            chunk = codes[:, cols].to(torch.int32) << shift
            packed[:, : len(cols)] |= chunk

    return packed.contiguous()


def unpack_ternary_weights(w_packed: torch.Tensor, K: int) -> torch.Tensor:
    """
    Pure PyTorch fallback for unpacking uint32 words back into a [N, K] FP32 ternary matrix.
    Handles non-divisible channel dimensions without memory misalignment.
    """
    w_packed = w_packed.contiguous()
    N = w_packed.shape[0]
    num_words = w_packed.shape[1]
    unpacked = torch.zeros((N, K), dtype=torch.float32, device=w_packed.device)

    for i in range(16):
        if i >= K:
            break
        cols = torch.arange(i, K, 16, device=w_packed.device)
        if len(cols) > 0:
            shift = 2 * i
            # Extract 2 bits
            code = (w_packed[:, : len(cols)] >> shift) & 0x3
            # Map 1 -> +1.0, 2 -> -1.0, 0 -> 0.0
            val = torch.zeros_like(code, dtype=torch.float32)
            val[code == 1] = 1.0
            val[code == 2] = -1.0
            unpacked[:, cols] = val

    return unpacked.contiguous()


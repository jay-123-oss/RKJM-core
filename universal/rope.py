"""
Universal Rotary Position Embedding (RoPE) for RKMJ-Core.
Supports arbitrary base theta (10,000 to 1,000,000+), head dimensions, and sequence offsets.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class UniversalRotaryEmbedding(nn.Module):
    """
    Computes and caches rotary position embeddings (cos, sin) for arbitrary base theta.
    Handles single-token autoregressive decoding and full prompt prefill sequences.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base_theta: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base_theta = float(base_theta)

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len, device=device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=device or self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cos, sin) slices of shape [1, seq_len, 1, head_dim] broadcastable to Q and K.
        """
        end_pos = start_pos + seq_len
        if end_pos > self.max_seq_len or self.cos_cached.device != x.device:
            new_len = max(end_pos, self.max_seq_len * 2)
            self._build_cache(new_len, device=x.device)

        cos = self.cos_cached[start_pos:end_pos].view(1, seq_len, 1, self.dim)
        sin = self.sin_cached[start_pos:end_pos].view(1, seq_len, 1, self.dim)
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embedding to query and key states.
    q, k shape: [B, num_heads, T, head_dim]
    cos, sin shape: [1, T, 1, head_dim] -> transposed to [1, 1, T, head_dim]
    """
    cos = cos.transpose(1, 2)
    sin = sin.transpose(1, 2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

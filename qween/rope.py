"""
Rotary Position Embedding (RoPE) for Qwen Architecture.
Supports high-throughput vector-compatible rotary transformation for queries and keys.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to query and key tensors.
    q: [B, num_heads, seq_len, head_dim]
    k: [B, num_kv_heads, seq_len, head_dim]
    cos: [1, 1, seq_len, head_dim] or broadcastable
    sin: [1, 1, seq_len, head_dim] or broadcastable
    """
    # Ensure cos and sin match q/k floating point precision and device
    cos = cos.to(q.device, dtype=q.dtype)
    sin = sin.to(q.device, dtype=q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QwenRotaryEmbedding(nn.Module):
    """
    Precomputed Rotary Position Embedding tables for Qwen models.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Inverse frequency calculation
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build initial cache
        self._build_cache(self.max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        # Concat to full head_dim
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cached (cos, sin) tensors sliced from start_pos to start_pos + seq_len.
        """
        total_len = start_pos + seq_len
        if total_len > self.cos_cached.shape[2]:
            self._build_cache(max(self.max_position_embeddings, total_len))
        return (
            self.cos_cached[:, :, start_pos:total_len, :],
            self.sin_cached[:, :, start_pos:total_len, :],
        )

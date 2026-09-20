"""
Full 1.58-bit Transformer Block combining Pre-RMSNorm, CSASelfAttention, and CSAMLP.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from rkmj.nn.norm import RMSNorm
from rkmj.nn.attention import CSASelfAttention
from rkmj.nn.mlp import CSAMLP


class CSATransformerBlock(nn.Module):
    """
    RKMJ 1.58-bit Transformer Block.

    Architecture (Pre-Norm with Residual Connections):
      1. h = x + Attention(RMSNorm(x))
      2. y = h + MLP(RMSNorm(h))

    All heavy projection weights are computed via our C++ CSA OpenMP engine.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        eps: float = 1e-6,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim

        # Pre-Norm normalization layers
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)

        # Core 1.58-bit CSA Blocks
        self.self_attn = CSASelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            attn_dropout=attn_dropout,
        )
        self.mlp = CSAMLP(
            dim=dim,
            intermediate_dim=intermediate_dim,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        layer_idx: int = 0,
        kv_cache: Optional[Any] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        # Sub-layer 1: Residual Self-Attention with Pre-RMSNorm
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_x,
            causal_mask=causal_mask,
            layer_idx=layer_idx,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        h = x + attn_out

        # Sub-layer 2: Residual SwiGLU MLP with Pre-RMSNorm
        norm_h = self.post_attention_layernorm(h)
        mlp_out = self.mlp(norm_h)
        out = h + mlp_out

        return out

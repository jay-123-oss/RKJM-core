"""
1.58-bit Multi-Head / Grouped-Query Self-Attention Module.
Powered by CSALinear projections for Q, K, V, and O matrices.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn.linear import CSALinear


class CSASelfAttention(nn.Module):
    """
    Multi-Head / Grouped-Query Self-Attention with 1.58-bit CSALinear Projections.

    Key Features:
      - Uses CSALinear for Q, K, V, and O heavy matrix projections.
      - Keeps scaled dot-product attention and softmax in FP32 for numerical stability.
      - Supports standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).
      - Built-in causal masking for autoregressive generative models.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        bias: bool = False,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        kv_dim = self.num_kv_heads * self.head_dim

        # 1.58-bit CSA Linear Projections
        self.q_proj = CSALinear(dim, dim, bias=bias)
        self.k_proj = CSALinear(dim, kv_dim, bias=bias)
        self.v_proj = CSALinear(dim, kv_dim, bias=bias)
        self.o_proj = CSALinear(dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        layer_idx: int = 0,
        kv_cache: Optional[Any] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections with 1.58-bit CSA Engine
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [Batch, NumHeads, SeqLen, HeadDim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # KV-Cache update & retrieval
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v, start_pos=start_pos)

        total_kv_len = k.size(2)

        # Handle Grouped-Query Attention (zero-allocation 5D strided expansion)
        if self.num_kv_groups > 1:
            # Reshape q to [Batch, NumKVHeads, NumKVGroups, SeqLen, HeadDim]
            q_gqa = q.view(batch_size, self.num_kv_heads, self.num_kv_groups, seq_len, self.head_dim)
            # Expand k and v without memory allocation (zero-copy strided views)
            k_gqa = k.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.num_kv_groups, total_kv_len, self.head_dim)
            v_gqa = v.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.num_kv_groups, total_kv_len, self.head_dim)

            scores = torch.matmul(q_gqa, k_gqa.transpose(-2, -1)) * self.scale

            if not (seq_len == 1 and kv_cache is not None):
                if causal_mask and seq_len > 1:
                    mask = torch.triu(
                        torch.full((seq_len, total_kv_len), float("-inf"), device=x.device),
                        diagonal=start_pos + 1,
                    )
                    scores = scores + mask

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            context = torch.matmul(attn_weights, v_gqa)
            context = context.view(batch_size, self.num_heads, seq_len, self.head_dim)
        else:
            # Scaled dot-product attention (MHA / MQA)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if not (seq_len == 1 and kv_cache is not None):
                if causal_mask and seq_len > 1:
                    mask = torch.triu(
                        torch.full((seq_len, total_kv_len), float("-inf"), device=x.device),
                        diagonal=start_pos + 1,
                    )
                    scores = scores + mask

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            context = torch.matmul(attn_weights, v)

        # Context aggregation
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # Final projection via CSA O_proj
        return self.o_proj(context)

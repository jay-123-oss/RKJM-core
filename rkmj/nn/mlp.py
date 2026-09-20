"""
1.58-bit SwiGLU FeedForward / MLP Module (LLaMA / Mistral style).
Powered by CSALinear projections for gate, up, and down projection matrices.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn.linear import CSALinear


class CSAMLP(nn.Module):
    """
    SwiGLU FeedForward Network powered by 1.58-bit CSALinear layers.

    Formula:
        output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        # Standard LLaMA heuristic: 8/3 * dim (rounded to multiple of 64 for cache-line alignment)
        if intermediate_dim is None:
            intermediate_dim = int(2 * 4 * dim / 3)
            intermediate_dim = ((intermediate_dim + 63) // 64) * 64

        self.dim = dim
        self.intermediate_dim = intermediate_dim

        # SwiGLU Projections powered by CSALinear
        self.gate_proj = CSALinear(dim, intermediate_dim, bias=bias)
        self.up_proj = CSALinear(dim, intermediate_dim, bias=bias)
        self.down_proj = CSALinear(intermediate_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: SiLU(gate) * up followed by down projection
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

"""
Root Mean Square Normalization (RMSNorm) Module.
Standard Pre-Norm implementation used in LLaMA / Mistral style architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Normalizes activations across the last feature dimension without centering around mean:
        y = (x / sqrt(mean(x^2) + eps)) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        norm_x = x_fp32 * torch.rsqrt(variance + self.eps)
        out = norm_x * self.weight.float()
        return out.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"

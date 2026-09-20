"""
Watch Framework Core Neural Network Modules (watch_nn.py).

Implements a LLaMA-style 1.58-bit Transformer Architecture powered by our
custom CPU Carry-Save Addition (CSA) / Popcount engine with Straight-Through
Estimator (STE) Autograd backward propagation.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure custom C++ extensions are accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import csa_autograd_cpu
    AUTOGRAD_AVAILABLE = True
except ImportError:
    AUTOGRAD_AVAILABLE = False

try:
    import csa_linear_cpu
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) as used in LLaMA / Mistral.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


class CSALinear(nn.Module):
    """
    1.58-bit Ternary Linear Layer with Carry-Save Addition (CSA) Engine.

    Features:
      - Training Mode: Full backward autograd via C++ OpenMP Straight-Through Estimator (STE).
      - Inference Mode: Packed 2-bit weight storage (16x RAM reduction) using OpenMP Popcount.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_words = (in_features + 15) // 16

        # High-precision latent weight for optimizer updates (STE)
        self.latent_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        # Per-channel dynamic scale factor alpha
        self.alpha = nn.Parameter(torch.ones(out_features, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # Packed buffer for frozen / inference execution
        self.register_buffer(
            "w_packed",
            torch.zeros((out_features, self.num_words), dtype=torch.int32),
            persistent=False,
        )
        self.is_packed = False

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with scaled Kaiming uniform."""
        nn.init.kaiming_uniform_(self.latent_weight, a=math.sqrt(5))
        with torch.no_grad():
            # Initial alpha = mean absolute weight value per row
            self.alpha.copy_(self.latent_weight.abs().mean(dim=1).clamp(min=1e-5))

    def pack_weights_for_inference(self):
        """Pack ternary weights into uint32 bitfields to optimize inference memory and speed."""
        if not INFERENCE_AVAILABLE:
            return
        with torch.no_grad():
            alpha_col = self.alpha.unsqueeze(1).clamp(min=1e-8)
            w_ternary = torch.clamp(torch.round(self.latent_weight / alpha_col), -1.0, 1.0)
            self.w_packed.copy_(csa_linear_cpu.pack_weights(w_ternary))
            self.is_packed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        In training (or when requiring grads), executes the C++ autograd kernel.
        When packed and evaluation mode, executes the ultra-fast packed kernel.
        """
        x = x.contiguous()
        if self.training or not self.is_packed or x.requires_grad:
            if AUTOGRAD_AVAILABLE:
                return csa_autograd_cpu.csa_linear(x, self.latent_weight, self.alpha, self.bias)
            else:
                # Pure PyTorch fallback if C++ extension not compiled
                alpha_col = self.alpha.unsqueeze(1).clamp(min=1e-8)
                w_q = torch.clamp(torch.round(self.latent_weight / alpha_col), -1.0, 1.0)
                x_sign = torch.where(x >= 0.0, 1.0, -1.0)
                y = self.alpha * F.linear(x_sign, w_q)
                if self.bias is not None:
                    y = y + self.bias
                return y
        else:
            # Frozen inference path
            return csa_linear_cpu.forward(x, self.w_packed, self.alpha, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, packed={self.is_packed}"
        )


class CSASelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with 1.58-bit CSALinear Q, K, V, O projections.
    Keeps scaled dot-product attention and softmax in FP32 for numerical stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = False,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1.58-bit CSA Linear Projections
        self.q_proj = CSALinear(dim, dim, bias=bias)
        self.k_proj = CSALinear(dim, dim, bias=bias)
        self.v_proj = CSALinear(dim, dim, bias=bias)
        self.o_proj = CSALinear(dim, dim, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections with 1.58-bit CSA Engine
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [Batch, NumHeads, SeqLen, HeadDim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal_mask and seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Context aggregation
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # Final projection via CSA O_proj
        return self.o_proj(context)


class CSAMLP(nn.Module):
    """
    SwiGLU FeedForward Network (LLaMA / Mistral style) with 1.58-bit CSALinear layers.
    Formula: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, dim: int, intermediate_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        # Standard LLaMA heuristic: 8/3 * dim (rounded to multiple of 64/128)
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
        # SwiGLU: SiLU(gate) * up followed by down projection
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class CSATransformerBlock(nn.Module):
    """
    Full 1.58-bit Transformer Block.

    Architecture (Pre-Norm with Residual Connections):
      1. h = x + Attention(RMSNorm(x))
      2. y = h + MLP(RMSNorm(h))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        intermediate_dim: Optional[int] = None,
        eps: float = 1e-6,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim

        # Pre-Norm layers
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)

        # Core 1.58-bit CSA Blocks
        self.self_attn = CSASelfAttention(
            dim=dim,
            num_heads=num_heads,
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
    ) -> torch.Tensor:
        # Sub-layer 1: Residual Self-Attention with Pre-RMSNorm
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(norm_x, causal_mask=causal_mask)
        h = x + attn_out

        # Sub-layer 2: Residual SwiGLU MLP with Pre-RMSNorm
        norm_h = self.post_attention_layernorm(h)
        mlp_out = self.mlp(norm_h)
        out = h + mlp_out

        return out


__all__ = [
    "RMSNorm",
    "CSALinear",
    "CSASelfAttention",
    "CSAMLP",
    "CSATransformerBlock",
]

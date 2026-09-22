"""
Neural Network Layers for RKMJ-Core 1.58-Bit Architecture.
Provides CSALinear, RMSNorm, and CSATransformerBlock drop-in replacements for PyTorch.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import rkmj._C as _C
except ImportError:
    _C = None

from rkmj.quantizer.ternary import quantize_ternary_grouped, pack_ternary, unpack_ternary


class CSALinearFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) PyTorch Autograd Function.
    - Forward pass: Discrete ternary evaluation W_quant in {-1, 0, +1}.
    - Backward pass: Gradient pass-through directly to latent shadow weights W_latent
      bounded by the indicator function I(|W_latent| <= 1.0).
    """

    @staticmethod
    def forward(ctx, x, latent_weight, alpha, bias=None):
        ctx.save_for_backward(x, latent_weight, alpha)
        ctx.has_bias = bias is not None

        if not x.is_cuda and _C is not None and hasattr(_C, "csa_ste_forward"):
            try:
                return _C.csa_ste_forward(x, latent_weight, alpha, bias)
            except Exception:
                pass

        # Pure PyTorch fallback (executed on CUDA and CPU)
        alpha_col = alpha.view(-1, 1).clamp(min=1e-5)
        w_norm = latent_weight / alpha_col
        w_quant = torch.clamp(torch.round(w_norm), -1.0, 1.0)
        effective_w = w_quant * alpha_col

        out = F.linear(x, effective_w, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, latent_weight, alpha = ctx.saved_tensors
        has_bias = ctx.has_bias

        if not grad_output.is_cuda and _C is not None and hasattr(_C, "csa_ste_backward"):
            try:
                gx, gw, ga, gb = _C.csa_ste_backward(grad_output, x, latent_weight, alpha, has_bias)
                return gx, gw, ga, gb if has_bias else None
            except Exception:
                pass

        # Pure PyTorch STE fallback
        alpha_col = alpha.view(-1, 1).clamp(min=1e-5)
        w_norm = latent_weight / alpha_col
        w_quant = torch.clamp(torch.round(w_norm), -1.0, 1.0)
        effective_w = w_quant * alpha_col

        grad_x = grad_output.matmul(effective_w)

        # Gradient to latent weight with clipping mask I(|W_latent| <= 1.0)
        grad_out_flat = grad_output.view(-1, grad_output.shape[-1])
        x_flat = x.view(-1, x.shape[-1])

        grad_w_unclipped = grad_out_flat.t().matmul(x_flat) * alpha_col
        ste_mask = (latent_weight.abs() <= 1.0).float()
        grad_latent = grad_w_unclipped * ste_mask

        # Gradient to scale factor alpha
        x_wq = x_flat.matmul(w_quant.t())
        grad_alpha = (grad_out_flat * x_wq).sum(dim=0)

        grad_bias = grad_out_flat.sum(dim=0) if has_bias else None
        return grad_x, grad_latent, grad_alpha, grad_bias


class CSALinear(nn.Module):
    """
    RKMJ 1.58-bit Carry-Save Addition (CSA) Linear Layer.
    - Training mode: Full-precision latent shadow weights with C++ STE autograd.
    - Inference mode: 2-bit aligned packed weights with vectorized tiled GEMV/GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 64,
        allocate_latent: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_words = (in_features + 15) // 16
        self.num_groups = (in_features + group_size - 1) // group_size

        if allocate_latent:
            self.latent_weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=torch.float32)
            )
            self.register_buffer(
                "w_packed",
                torch.zeros((out_features, self.num_words), dtype=torch.int32),
                persistent=True,
            )
        else:
            self.register_parameter("latent_weight", None)
            self.register_buffer("w_packed", None, persistent=False)

        self.alpha = nn.Parameter(torch.ones(out_features, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.is_packed = False
        self.reset_parameters()

    def reset_parameters(self):
        if self.latent_weight is not None:
            nn.init.kaiming_uniform_(self.latent_weight, a=math.sqrt(5))
            with torch.no_grad():
                self.alpha.copy_(self.latent_weight.abs().mean(dim=1).clamp(min=1e-5))

    def pack_weights_for_inference(self):
        """Quantizes latent weights and packs into 2-bit integer bitfields."""
        with torch.no_grad():
            if self.latent_weight is not None:
                w_packed, scales = quantize_ternary_grouped(self.latent_weight, self.group_size)
                self.register_buffer("w_packed", w_packed, persistent=True)
                self.register_buffer("scales", scales, persistent=True)
                self.is_packed = True
                # Release heavy latent parameter to free DRAM
                self.register_parameter("latent_weight", None)

    def set_weight(self, weight: torch.Tensor):
        """Sets pre-dequantized float32 / bfloat16 weights directly for inference."""
        self.register_buffer("dequantized_weight", weight.contiguous(), persistent=False)
        self.register_parameter("latent_weight", None)
        self.register_buffer("w_packed", None, persistent=False)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(weight.dtype)

    def set_dequantized_weights(
        self,
        w_ternary: torch.Tensor,
        alpha: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Pre-computes dequantized weights W_dequant = W_ternary * alpha for continuous inference."""
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)
        self.register_buffer("dequantized_weight", (w_ternary * alpha).to(dtype).contiguous(), persistent=False)
        self.register_parameter("latent_weight", None)
        self.register_buffer("w_packed", None, persistent=False)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()

        # 1. High-throughput continuous dequantized path
        if hasattr(self, "dequantized_weight") and self.dequantized_weight is not None:
            return F.linear(x, self.dequantized_weight, self.bias)

        # 2. Packed 2-bit tiled GEMV / GEMM path
        if self.is_packed and self.w_packed is not None and hasattr(self, "scales"):
            if _C is not None and hasattr(_C, "gemm_tiled_csa"):
                try:
                    return _C.gemm_tiled_csa(x, self.w_packed, self.scales, self.bias, self.group_size)
                except Exception:
                    pass

        # 3. Training STE autograd path
        if self.latent_weight is not None:
            return CSALinearFunction.apply(x, self.latent_weight, self.alpha, self.bias)

        raise RuntimeError("CSALinear: Neither latent_weight, dequantized_weight, nor w_packed is initialized.")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        norm_x = x_f * torch.rsqrt(variance + self.eps)
        return (self.weight * norm_x).to(orig_dtype)


class CSATransformerBlock(nn.Module):
    """Complete 1.58-Bit Transformer Block powered by CSALinear and RMSNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: Optional[int] = None,
        n_heads: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        group_size: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        heads = num_heads if num_heads is not None else n_heads
        if heads is None:
            heads = max(1, dim // 64)
        inter_dim = intermediate_dim if intermediate_dim is not None else int(dim * 8 // 3)

        self.dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads

        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.q_proj = CSALinear(dim, dim, bias=False, group_size=group_size)
        self.k_proj = CSALinear(dim, dim, bias=False, group_size=group_size)
        self.v_proj = CSALinear(dim, dim, bias=False, group_size=group_size)
        self.o_proj = CSALinear(dim, dim, bias=False, group_size=group_size)

        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.gate_proj = CSALinear(dim, inter_dim, bias=False, group_size=group_size)
        self.up_proj = CSALinear(dim, inter_dim, bias=False, group_size=group_size)
        self.down_proj = CSALinear(inter_dim, dim, bias=False, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm Self-Attention
        norm_x = self.input_layernorm(x)
        B, T, C = norm_x.shape

        q = self.q_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn_out)

        # Pre-norm SwiGLU FeedForward
        norm_x2 = self.post_attention_layernorm(x)
        mlp_out = self.down_proj(F.silu(self.gate_proj(norm_x2)) * self.up_proj(norm_x2))
        x = x + mlp_out
        return x

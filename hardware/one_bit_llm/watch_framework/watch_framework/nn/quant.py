"""Quantization primitives and custom Straight-Through Estimator (STE) autograd functions.

Implements:
1. dynamic_alpha: alpha = (1 / N) * sum(|W_i|)
2. ste_sign: 1-bit quantization to {-1, +1} with identity gradient within [-1, 1]
3. WatchGridAutogradFunction: autograd bridge with C++ OpenMP backend and pure-torch fallback.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from watch_framework import _C
    CPP_AVAILABLE = True
except ImportError:
    _C = None
    CPP_AVAILABLE = False


def dynamic_alpha(w: Tensor) -> Tensor:
    """Compute the dynamic scale factor alpha = (1 / N) * sum(|W_i|)."""
    return w.abs().mean()


class _STESignFunction(torch.autograd.Function):
    """Straight-Through Estimator sign quantization with [-1, 1] hard clipping."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        # Binarize to {-1.0, +1.0} where x >= 0 -> +1.0, x < 0 -> -1.0
        return torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        # STE indicator mask: Indicator(|x| <= 1.0)
        mask = (x.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask


def ste_sign(x: Tensor) -> Tensor:
    """Apply straight-through estimator sign function."""
    return _STESignFunction.apply(x)


class WatchGridAutogradFunction(torch.autograd.Function):
    """Autograd function bridging PyTorch to the C++ OpenMP CSA GEMM kernel with pure PyTorch fallback."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        force_fallback: bool = False,
    ) -> Tensor:
        if x.dim() < 2:
            raise ValueError(f"Input x must have at least 2 dimensions, got shape {x.shape}")
        if weight.dim() != 2:
            raise ValueError(f"Weight must be 2D [out_features, in_features], got shape {weight.shape}")
        if x.shape[-1] != weight.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: x has {x.shape[-1]}, weight has {weight.shape[1]}"
            )

        orig_shape = x.shape
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        x_2d = x.reshape(-1, in_features).contiguous()
        weight_contig = weight.contiguous()
        bias_contig = bias.contiguous() if bias is not None else None

        # Execute C++ accelerated backend if available
        if CPP_AVAILABLE and not force_fallback and x.is_cpu and x.dtype == torch.float32 and weight.dtype == torch.float32:
            try:
                # _C.forward returns: [y_2d, x_q, w_q, alpha, y_unscaled]
                res = _C.forward(x_2d, weight_contig, bias_contig)
                y_2d = res[0]
                x_q = res[1]
                w_q = res[2]
                alpha = res[3]
                ctx.save_for_backward(x_2d, weight_contig, x_q, w_q, alpha)
                ctx.has_bias = bias is not None
                ctx.orig_shape = orig_shape
                ctx.in_features = in_features
                ctx.out_features = out_features
                ctx.used_cpp = True
                out_shape = list(orig_shape[:-1]) + [out_features]
                return y_2d.reshape(out_shape)
            except Exception:
                # Fallback on any runtime error
                pass

        # Pure PyTorch fallback
        alpha = dynamic_alpha(weight_contig)
        x_q = torch.where(x_2d >= 0.0, torch.ones_like(x_2d), -torch.ones_like(x_2d))
        w_q = torch.where(weight_contig >= 0.0, torch.ones_like(weight_contig), -torch.ones_like(weight_contig))

        y_unscaled = F.linear(x_q, w_q)
        y_2d = alpha * y_unscaled
        if bias_contig is not None:
            y_2d = y_2d + bias_contig.unsqueeze(0)

        ctx.save_for_backward(x_2d, weight_contig, x_q, w_q, alpha)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.used_cpp = False

        out_shape = list(orig_shape[:-1]) + [out_features]
        return y_2d.reshape(out_shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        x_2d, weight, x_q, w_q, alpha = ctx.saved_tensors
        grad_out_contig = grad_output.contiguous()
        grad_out_2d = grad_out_contig.reshape(-1, ctx.out_features).to(torch.float32)

        if ctx.used_cpp and CPP_AVAILABLE:
            try:
                # _C.backward returns: [grad_x, grad_weight, grad_bias, grad_alpha]
                grads = _C.backward(
                    grad_out_2d,
                    x_2d,
                    weight,
                    x_q,
                    w_q,
                    alpha,
                    ctx.has_bias,
                )
                grad_x = grads[0].reshape(ctx.orig_shape).to(grad_output.dtype)
                grad_weight = grads[1].to(weight.dtype)
                grad_bias = grads[2].to(weight.dtype) if ctx.has_bias else None
                return grad_x, grad_weight, grad_bias, None
            except Exception:
                pass

        # Pure PyTorch STE backward implementation
        # 1. dL/dX = alpha * (dL/dY @ W_q)
        grad_x_2d = alpha * torch.matmul(grad_out_2d, w_q)
        grad_x = grad_x_2d.reshape(ctx.orig_shape).to(grad_output.dtype)

        # 2. dL/dW = alpha * (dL/dY^T @ X_q) * Indicator(|W| <= 1.0)
        ste_mask = (weight.abs() <= 1.0).to(grad_out_2d.dtype)
        grad_w_unclipped = alpha * torch.matmul(grad_out_2d.t(), x_q)
        grad_weight = (grad_w_unclipped * ste_mask).to(weight.dtype)

        # 3. dL/dbias
        grad_bias = grad_out_2d.sum(dim=0).to(weight.dtype) if ctx.has_bias else None

        return grad_x, grad_weight, grad_bias, None


def watch_grid_matmul(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    force_fallback: bool = False,
) -> Tensor:
    """Compute the 1-bit Watch Grid forward pass with dynamic alpha scaling and STE backward."""
    return WatchGridAutogradFunction.apply(x, weight, bias, force_fallback)

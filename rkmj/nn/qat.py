"""
Quantization-Aware Training (QAT) Module for RKMJ-Core.
Features:
  - TernaryQuantizeSTE: Straight-Through Estimator with dynamic alpha scaling and hard-tanh gradient clipping.
  - ActivationQuantizeSTE: 1-bit sign quantization with gradient clipping.
  - QATCSALinear: Drop-in replacement for nn.Linear / CSALinear maintaining FP32/BF16 master weights.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn.linear import CSALinear, CPP_ENGINE_AVAILABLE

try:
    import rkmj._C as _C
except ImportError:
    _C = None


class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for 1.58-bit Ternary Weight Quantization.

    Forward:
      alpha_i = (1 / d_in) * sum_{j=1}^{d_in} |W_{ij}| + eps
      W_scaled = W / alpha
      W_quant = clamp(round(W_scaled), -1, 1)
      Returns (W_quant, alpha)

    Backward (Hard-Tanh Gradient Window):
      dL / dW = (dL / dW_quant) * I(|W_scaled| <= 1.0)
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_in = weight.size(-1)
        # Compute dynamic per-channel scale alpha: shape [d_out, 1]
        alpha = weight.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
        w_scaled = weight / alpha

        # Quantize to ternary {-1, 0, +1}
        w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0)

        # Save for backward pass
        ctx.save_for_backward(w_scaled)
        ctx.eps = eps

        return w_quant, alpha.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_w_quant: torch.Tensor, grad_alpha: Optional[torch.Tensor] = None):
        (w_scaled,) = ctx.saved_tensors

        # Hard-tanh gradient clipping: zero out gradients where |W_scaled| > 1.0
        # Prevents master weights from exploding or drifting into saturation
        grad_mask = (w_scaled.abs() <= 1.0).to(grad_w_quant.dtype)
        grad_weight = grad_w_quant * grad_mask

        return grad_weight, None


class ActivationQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for 1-bit Activation Sign Binarization.

    Forward:
      x_sign = where(x >= 0, +1.0, -1.0)

    Backward:
      dL / dx = (dL / dx_sign) * I(|x| <= 1.0)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        x_sign = torch.where(x >= 0.0, 1.0, -1.0)
        ctx.save_for_backward(x)
        return x_sign

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        # Gradient clipping window [-1.0, 1.0]
        grad_mask = (x.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * grad_mask


class QATCSALinear(CSALinear):
    """
    Quantization-Aware Training (QAT) 1.58-bit Linear Layer.

    Maintains full-precision master weights for optimizer updates while simulating
    1.58-bit ternary quantization during the forward pass with STE gradient flow.

    Modes:
      - Training (self.training=True):
          Runs simulated ternary GEMM via TernaryQuantizeSTE and ActivationQuantizeSTE.
      - Evaluation (self.training=False):
          Packs weights into 2-bit bitfields and invokes the ultra-fast C++ OpenMP CSA engine.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        # Full-precision master weight parameter for gradient descent
        self.master_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        self.reset_master_parameters()

        # Keep latent_weight synchronized to master_weight
        with torch.no_grad():
            self.latent_weight.data.copy_(self.master_weight.data)
            self.alpha.data.copy_(self.master_weight.abs().mean(dim=1).clamp(min=1e-5))

    def reset_master_parameters(self):
        """Initialize master weight with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.master_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.master_weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def synchronize_latent(self):
        """Copy master weights to latent weights for inference."""
        with torch.no_grad():
            self.latent_weight.data.copy_(self.master_weight.data)
            self.alpha.data.copy_(self.master_weight.abs().mean(dim=1).clamp(min=1e-5))

    def pack_weights_for_inference(self):
        """Pack master weights into uint32 2-bit bitfields."""
        with torch.no_grad():
            self.synchronize_latent()
            alpha_col = self.alpha.unsqueeze(1).clamp(min=1e-8)
            w_ternary = torch.clamp(torch.round(self.master_weight / alpha_col), -1.0, 1.0)
            if CPP_ENGINE_AVAILABLE and hasattr(_C, "pack_weights"):
                self.w_packed.copy_(_C.pack_weights(w_ternary.float()))
            else:
                from rkmj.serialization.packer import pack_ternary_weights
                self.w_packed.copy_(pack_ternary_weights(w_ternary.float()))
            self.is_packed = True

    def export_packed_weights(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return (w_packed, alpha, bias) tuple for serialization."""
        if not self.is_packed:
            self.pack_weights_for_inference()
        return self.w_packed, self.alpha, self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic mode switching:
          - Training: simulated 1.58-bit ternary forward pass with STE autograd.
          - Evaluation: fast multi-ISA C++ CSA popcount engine.
        """
        if self.training or x.requires_grad:
            # Simulated 1.58-bit forward pass with STE
            w_quant, alpha = TernaryQuantizeSTE.apply(self.master_weight)
            x_sign = ActivationQuantizeSTE.apply(x)

            # Effective forward calculation: y = alpha * (x_sign @ w_quant^T) + bias
            y = alpha * F.linear(x_sign, w_quant)
            if self.bias is not None:
                y = y + self.bias
            return y
        else:
            # Inference execution: ensure weights are packed and call C++ engine
            if not self.is_packed:
                self.pack_weights_for_inference()

            if CPP_ENGINE_AVAILABLE and hasattr(_C, "csa_forward"):
                return _C.csa_forward(x, self.w_packed, self.alpha, self.bias)
            else:
                return super().forward(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={'TRAIN' if self.training else 'EVAL'}"
        )

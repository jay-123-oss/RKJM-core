"""
1.58-bit (Ternary) Linear Layer powered by Carry-Save Addition (CSA) Bitwise Popcount Engine.
Drop-in replacement for torch.nn.Linear.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt import of compiled C++ engine
try:
    import rkmj._C as _C
    CPP_ENGINE_AVAILABLE = True
except ImportError:
    try:
        # Fallback to local build if running from source tree
        import csa_autograd_cpu as _C_autograd
        import csa_linear_cpu as _C_linear
        CPP_ENGINE_AVAILABLE = True
    except ImportError:
        _C = None
        CPP_ENGINE_AVAILABLE = False


class CSALinear(nn.Module):
    """
    RKMJ 1.58-bit Carry-Save Addition (CSA) Linear Layer.

    Key Features:
      - Training Mode: Full backward autograd via C++ OpenMP Straight-Through Estimator (STE).
      - Inference Mode: Packed 2-bit weight storage (15.8x RAM reduction) using OpenMP Popcount.
      - Dynamic Alpha Scaling: Per-channel scale alpha = mean(|W|) to preserve representation power.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, allocate_latent: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_words = (in_features + 15) // 16

        # High-precision latent weight for optimizer updates (STE)
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

        # Dynamic per-channel scale factor alpha
        self.alpha = nn.Parameter(torch.ones(out_features, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.is_packed = False

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize latent weights using Kaiming uniform."""
        if self.latent_weight is not None:
            nn.init.kaiming_uniform_(self.latent_weight, a=math.sqrt(5))
            with torch.no_grad():
                self.alpha.copy_(self.latent_weight.abs().mean(dim=1).clamp(min=1e-5))

    def update_alpha(self):
        """Synchronize dynamic scale factor alpha to current latent weight magnitude."""
        with torch.no_grad():
            self.alpha.copy_(self.latent_weight.abs().mean(dim=1).clamp(min=1e-5))

    def pack_weights_for_inference(self):
        """Pack ternary weights into uint32 bitfields to optimize inference memory and speed."""
        with torch.no_grad():
            alpha_col = self.alpha.unsqueeze(1).clamp(min=1e-8)
            w_ternary = torch.clamp(torch.round(self.latent_weight / alpha_col), -1.0, 1.0)
            if CPP_ENGINE_AVAILABLE and hasattr(_C, "pack_weights"):
                self.w_packed.copy_(_C.pack_weights(w_ternary))
            elif CPP_ENGINE_AVAILABLE and "_C_linear" in globals() and _C_linear is not None:
                self.w_packed.copy_(_C_linear.pack_weights(w_ternary))
            else:
                # Python packing fallback
                from rkmj.serialization.packer import pack_ternary_weights
                self.w_packed.copy_(pack_ternary_weights(w_ternary))
            self.is_packed = True

    def unpack_weights(self) -> torch.Tensor:
        """Unpack 2-bit integer bitfields back into FP32 ternary matrix [-1, 0, +1]."""
        if CPP_ENGINE_AVAILABLE and hasattr(_C, "unpack_weights"):
            return _C.unpack_weights(self.w_packed, self.in_features)
        elif CPP_ENGINE_AVAILABLE and "_C_linear" in globals() and _C_linear is not None:
            return _C_linear.unpack_weights(self.w_packed, self.in_features)
        else:
            from rkmj.serialization.packer import unpack_ternary_weights
            return unpack_ternary_weights(self.w_packed, self.in_features)

    def set_dequantized_weights(self, w_ternary: torch.Tensor, alpha: torch.Tensor, dtype: torch.dtype = torch.bfloat16):
        """Pre-computes dequantized weights W_dequant = W_ternary * alpha for continuous activation inference."""
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)
        self.register_buffer("dequantized_weight", (w_ternary * alpha).to(dtype).contiguous(), persistent=False)
        self.register_parameter("latent_weight", None)
        self.register_buffer("w_packed", None, persistent=False)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        - If dequantized_weight buffer is set: executes high-throughput vectorized linear forward.
        - In training (or when requiring grads): executes C++ autograd kernel.
        - When packed and in eval mode: executes bitwise popcount kernel.
        """
        x = x.contiguous()

        if hasattr(self, "dequantized_weight") and self.dequantized_weight is not None:
            return F.linear(x, self.dequantized_weight, self.bias)

        if self.training or not self.is_packed or x.requires_grad:
            if CPP_ENGINE_AVAILABLE and hasattr(_C, "csa_linear"):
                return _C.csa_linear(x, self.latent_weight, self.alpha, self.bias)
            elif CPP_ENGINE_AVAILABLE and "_C_autograd" in globals() and _C_autograd is not None:
                return _C_autograd.csa_linear(x, self.latent_weight, self.alpha, self.bias)
            else:
                # Pure PyTorch fallback if C++ engine is not compiled
                alpha_col = self.alpha.unsqueeze(1).clamp(min=1e-8)
                w_norm = self.latent_weight / alpha_col
                w_q_val = torch.clamp(torch.round(w_norm), -1.0, 1.0)
                # Straight-Through Estimator (STE) gradient bypass
                w_q = w_norm + (w_q_val - w_norm).detach()
                x_sign_val = torch.where(x >= 0.0, 1.0, -1.0)
                x_sign = x + (x_sign_val - x).detach()
                y = self.alpha * F.linear(x_sign, w_q)
                if self.bias is not None:
                    y = y + self.bias
                return y
        else:
            # Ultra-fast frozen inference path (Popcount CSA Engine)
            if CPP_ENGINE_AVAILABLE and hasattr(_C, "csa_forward"):
                return _C.csa_forward(x, self.w_packed, self.alpha, self.bias)
            elif CPP_ENGINE_AVAILABLE and "_C_linear" in globals() and _C_linear is not None:
                return _C_linear.forward(x, self.w_packed, self.alpha, self.bias)
            else:
                w_ternary = self.unpack_weights()
                x_sign = torch.where(x >= 0.0, 1.0, -1.0)
                y = self.alpha * F.linear(x_sign, w_ternary)
                if self.bias is not None:
                    y = y + self.bias
                return y

    def forward_fused_rmsnorm(
        self,
        x: torch.Tensor,
        rmsnorm_weight: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Fused Operator Forward:
        RMSNorm -> in-L1 activation sign quantization -> 1.58-bit CSA GEMM.
        Avoids writing intermediate unquantized activations back to DRAM.
        """
        if (
            not self.training
            and self.is_packed
            and not x.requires_grad
            and CPP_ENGINE_AVAILABLE
            and hasattr(_C, "fused_rmsnorm_csa_forward")
        ):
            return _C.fused_rmsnorm_csa_forward(
                x.contiguous(),
                rmsnorm_weight.contiguous(),
                float(eps),
                self.w_packed,
                self.alpha,
                self.bias
            )
        # Fallback: compute standard RMSNorm then regular forward
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rmsnorm_weight
        return self.forward(norm_x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, packed={self.is_packed}"
        )

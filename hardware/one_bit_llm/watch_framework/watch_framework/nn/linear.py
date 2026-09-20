"""WatchLinear: 1-bit bitwise Carry-Save Addition (CSA) drop-in replacement for nn.Linear."""

from __future__ import annotations

import math
from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .quant import watch_grid_matmul, dynamic_alpha, ste_sign


class WatchLinear(nn.Module):
    """PyTorch-native Linear layer powered by 1-bit CSA GEMM and dynamic alpha scaling.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``True``, the layer learns an additive bias. Default: ``False``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"Features must be positive, got in_features={in_features}, out_features={out_features}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Latent FP32 weight parameter optimized by STE
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # Flag for FP32 warmup phase vs 1-bit QAT
        self._qat_enabled: bool = True

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_qat_mode(self, enabled: bool = True) -> None:
        """Toggle between 1-bit QAT execution and FP32 warmup execution."""
        self._qat_enabled = enabled

    @property
    def dynamic_alpha(self) -> Tensor:
        """Calculate and return the current dynamic scale factor alpha."""
        return dynamic_alpha(self.weight)

    @property
    def quantized_weight(self) -> Tensor:
        """Return the quantized ternary/binary {-1, +1} weight tensor."""
        return ste_sign(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. In QAT mode, invokes the 1-bit CSA GEMM kernel."""
        if not self._qat_enabled:
            # FP32 warmup path
            return F.linear(x, self.weight, self.bias)

        return watch_grid_matmul(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, qat={self._qat_enabled}"

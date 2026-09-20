"""WatchConv2d: 1-bit bitwise 2D Convolution via im2col (unfold) and WatchLinear GEMM."""

from __future__ import annotations

import math
from typing import Tuple, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .quant import watch_grid_matmul, dynamic_alpha, ste_sign


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


class WatchConv2d(nn.Module):
    """PyTorch-native 2D Convolution layer powered by im2col and 1-bit CSA GEMM.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        bias: If True, adds a learnable bias to the output. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        self.k_features = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        # Weight representation: [out_channels, in_channels, K_h, K_w]
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self._qat_enabled: bool = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform for Conv2d."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in > 0:
                bound = 1.0 / math.sqrt(fan_in)
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
        """Forward pass using im2col unfold followed by bitwise GEMM."""
        if not self._qat_enabled:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
            )

        batch_size, _, h_in, w_in = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        h_out = (h_in + 2 * ph - kh) // sh + 1
        w_out = (w_in + 2 * pw - kw) // sw + 1

        # 1. im2col unfold: [B, C_in * kh * kw, L] where L = h_out * w_out
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # Permute to [B * L, C_in * kh * kw]
        L = x_unfolded.shape[-1]
        cols = x_unfolded.transpose(1, 2).reshape(batch_size * L, self.k_features)

        # 2. Reshape weight to 2D matrix: [out_channels, C_in * kh * kw]
        weight_2d = self.weight.reshape(self.out_channels, self.k_features)

        # 3. Bitwise 1-bit CSA GEMM
        out_2d = watch_grid_matmul(cols, weight_2d, self.bias)

        # 4. Reshape back to [B, out_channels, h_out, w_out]
        out_reshaped = out_2d.reshape(batch_size, L, self.out_channels).transpose(1, 2)
        out = out_reshaped.reshape(batch_size, self.out_channels, h_out, w_out)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None}, qat={self._qat_enabled}"
        )

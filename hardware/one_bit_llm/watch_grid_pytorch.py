"""Production PyTorch integration for the Watch Grid ternary linear layer.

The module exposes an ``nn.Linear``-compatible interface while retaining a
latent FP32 parameter for optimization.  During forward execution the
parameter is quantized to {-1, 0, +1}; the packed representation is sent to
the Triton backend when CUDA is available and to the exact PyTorch fallback on
CPU.  The custom backward implements the masked STE specified by the design.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from hybrid_bitwise_matmul import hybrid_ternary_matmul, pack_ternary_torch


PathLike = Union[str, os.PathLike[str]]


def quantize_ternary(weight_fp32: Tensor) -> Tensor:
    """Apply the architecture's nearest-integer ternary quantizer."""

    return torch.clamp(torch.round(weight_fp32), -1.0, 1.0)


def quantize_scaled_ternary(weight_fp32: Tensor) -> Tuple[Tensor, Tensor]:
    """Return normalized ternary weights and one scale per output row."""

    alpha = weight_fp32.abs().mean(dim=1, keepdim=True).clamp_min(1.0e-8)
    normalized = weight_fp32 / alpha
    return quantize_ternary(normalized), alpha.squeeze(1)


class WatchGridAutogradFunction(torch.autograd.Function):
    """Autograd bridge for packed Watch Grid execution.

    Inputs are ``X``, latent ``W_fp32``, optional bias, and an optional packed
    cache.  The fourth input is deliberately non-differentiable hardware
    state; returning ``None`` for it keeps optimizer state attached only to
    the latent parameter.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        weight_fp32: Tensor,
        bias: Optional[Tensor],
        packed_weight: Optional[Tensor],
    ) -> Tensor:
        if x.ndim != 2:
            raise ValueError("WatchGridAutogradFunction expects a rank-2 activation matrix")
        if weight_fp32.ndim != 2 or weight_fp32.dtype not in (torch.float32, torch.float64):
            raise TypeError("weight_fp32 must be a rank-2 FP32/FP64 tensor")
        if x.shape[1] != weight_fp32.shape[1]:
            raise ValueError("activation and weight feature dimensions do not match")

        w_q, alpha = quantize_scaled_ternary(weight_fp32)
        w_q = w_q.to(torch.float32)
        if packed_weight is None or packed_weight.numel() == 0:
            packed_weight = pack_ternary_torch(w_q.to(torch.int32))
        output = hybrid_ternary_matmul(x, packed_weight) * alpha.to(torch.float32)
        if bias is not None:
            output = output + bias.to(output.dtype)

        ctx.save_for_backward(x, w_q, weight_fp32, alpha)
        ctx.has_bias = bias is not None
        ctx.input_shape = tuple(x.shape)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        x, w_q, weight_fp32, alpha = ctx.saved_tensors
        grad_matrix = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float32)
        x_matrix = x.reshape(-1, x.shape[-1]).to(torch.float32)

        # dL/dW = (dL/dY)^T @ X * 1_{|W_fp| <= 1.5}.
        ste_mask = weight_fp32.abs().le(1.5)
        grad_weight = grad_matrix.transpose(0, 1).matmul(x_matrix)
        grad_weight = grad_weight * ste_mask.to(grad_weight.dtype)

        # The hardware forward consumes bipolar activation bits, so its input
        # gradient is the corresponding straight-through matrix product.
        grad_input = (grad_matrix * alpha.to(torch.float32)).matmul(w_q)
        grad_input = grad_input.reshape(ctx.input_shape).to(dtype=x.dtype)
        grad_bias = grad_matrix.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None


class WatchGridLinear(nn.Module):
    """Drop-in linear layer backed by a ternary Watch Grid execution path.

    ``packing_factor`` is the register width in bits.  With two bits per
    ternary value, the default 32-bit register stores sixteen weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        packing_factor: int = 32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if packing_factor != 32:
            raise ValueError("the current backend supports 32-bit packed registers only")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.packing_factor = int(packing_factor)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        words = (in_features + 15) // 16
        self.register_buffer("packed_weight", torch.zeros(out_features, words, dtype=torch.int32))
        self.register_buffer("alpha", torch.ones(out_features, dtype=torch.float32))
        self._packed_weight_version = -1
        self._inference_only = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1.0 / self.in_features**0.5
            nn.init.uniform_(self.bias, -bound, bound)
        self._packed_weight_version = -1

    @classmethod
    def from_linear(cls, layer: nn.Linear, packing_factor: int = 32) -> "WatchGridLinear":
        """Create a Watch Grid layer with a standard linear layer's weights."""

        result = cls(layer.in_features, layer.out_features, layer.bias is not None, packing_factor)
        with torch.no_grad():
            result.weight.copy_(layer.weight.float())
            if layer.bias is not None and result.bias is not None:
                result.bias.copy_(layer.bias.float())
        result._packed_weight_version = -1
        return result

    def _refresh_packed_weight(self) -> None:
        if self.weight is None:
            raise RuntimeError("packed-only layer has no latent weights to repack")
        if self._packed_weight_version == self.weight._version:
            return
        with torch.no_grad():
            quantized, alpha = quantize_scaled_ternary(self.weight)
            packed = pack_ternary_torch(quantized.to(torch.int32))
            self.packed_weight.resize_(packed.shape)
            self.packed_weight.copy_(packed)
            self.alpha.copy_(alpha)
        self._packed_weight_version = self.weight._version

    def training_mode(self) -> "WatchGridLinear":
        """Enable latent-weight training and dynamic packed-cache refresh."""

        if self.weight is None:
            raise RuntimeError("a packed-only layer cannot return to training without latent weights")
        self.train(True)
        self._inference_only = False
        self.weight.requires_grad_(True)
        return self

    def inference_mode(self) -> "WatchGridLinear":
        """Freeze FP32 state and use the packed register cache for inference."""

        self.eval()
        self._refresh_packed_weight()
        self._inference_only = True
        self.weight.requires_grad_(False)
        return self

    def clamp_latent_weights(self, minimum: float = -1.2, maximum: float = 1.2) -> "WatchGridLinear":
        """Keep trainable latent weights inside the STE support envelope."""

        if self.weight is None:
            raise RuntimeError("packed-only layer has no latent weights to clamp")
        if minimum >= maximum:
            raise ValueError("minimum must be smaller than maximum")
        with torch.no_grad():
            self.weight.clamp_(minimum, maximum)
        return self

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape[:-1]
        if x.shape[-1] != self.in_features:
            raise ValueError("the final input dimension must equal in_features")
        x_matrix = x.reshape(-1, self.in_features)

        if self._inference_only:
            output = hybrid_ternary_matmul(x_matrix, self.packed_weight) * self.alpha
            if self.bias is not None:
                output = output + self.bias.to(output.dtype)
        else:
            self._refresh_packed_weight()
            output = WatchGridAutogradFunction.apply(
                x_matrix, self.weight, self.bias, self.packed_weight
            )
        return output.reshape(*original_shape, self.out_features)

    def export_packed_checkpoint(self, file_path: PathLike) -> Dict[str, Any]:
        """Save packed weights and metadata without serializing latent weights."""

        self._refresh_packed_weight()
        payload: Dict[str, Any] = {
            "format": "watch-grid-packed-v2",
            "shape": (self.out_features, self.in_features),
            "packing_factor": self.packing_factor,
            "ternary_encoding": {0: 0, 1: 1, -1: 3},
            "packed_weight": self.packed_weight.detach().cpu(),
            "bias": None if self.bias is None else self.bias.detach().cpu(),
            "scale": self.alpha.detach().cpu(),
        }
        torch.save(payload, file_path)
        return {
            "path": str(file_path),
            "fp32_bytes": self.weight.numel() * 4,
            "packed_bytes": self.packed_weight.numel() * 4,
            "compression_ratio": (self.weight.numel() * 4)
            / max(1, self.packed_weight.numel() * 4),
        }

    def load_packed_checkpoint(self, file_path: PathLike) -> "WatchGridLinear":
        """Load packed deployment state without reconstructing FP32 weights."""

        payload = torch.load(file_path, map_location="cpu", weights_only=True)
        if payload.get("format") != "watch-grid-packed-v2":
            raise ValueError("unsupported Watch Grid checkpoint format")
        if tuple(payload["shape"]) != (self.out_features, self.in_features):
            raise ValueError("checkpoint shape does not match this layer")
        if int(payload["packing_factor"]) != self.packing_factor:
            raise ValueError("checkpoint packing factor does not match this layer")
        packed = payload["packed_weight"]
        if packed.dtype != torch.int32:
            raise TypeError("packed checkpoint must contain int32 registers")
        scale = payload["scale"]
        if not isinstance(scale, Tensor) or tuple(scale.shape) != (self.out_features,):
            raise ValueError("checkpoint scale must contain one value per output row")
        self.packed_weight.resize_(packed.shape)
        self.packed_weight.copy_(packed)
        self.alpha.copy_(scale)
        if self.bias is not None and payload["bias"] is not None:
            self.bias.data.copy_(payload["bias"])
        self.register_parameter("weight", None)
        self._packed_weight_version = -1
        self._inference_only = True
        self.eval()
        return self


__all__ = [
    "WatchGridAutogradFunction",
    "WatchGridLinear",
    "quantize_scaled_ternary",
    "quantize_ternary",
]
"""
Ternary LoRA (Low-Rank Adaptation) Module for RKMJ-Core.
Features:
  - TernaryLoRALinear: 1.58-bit frozen base weights executed via C++ CSA popcount engine,
    combined with trainable FP16/BF16/FP32 low-rank adapter matrices (A and B).
  - apply_ternary_lora: Injects Ternary LoRA adapters into any transformer model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn.linear import CSALinear, CPP_ENGINE_AVAILABLE

try:
    import rkmj._C as _C
except ImportError:
    _C = None


class TernaryLoRALinear(nn.Module):
    """
    Ternary Low-Rank Adaptation (LoRA) Linear Layer.

    Forward:
      y = CSA_Forward(x, W_0, alpha) + (gamma / r) * (x @ A^T @ B^T) + bias

    Properties:
      - Base weight W_0 is frozen in 1.58-bit ternary precision.
      - Adapter matrices:
          A in R^{r x d_in}  initialized via Kaiming uniform / Gaussian
          B in R^{d_out x r} initialized to 0 (so Delta W = 0 at step 0)
      - Extremely parameter-efficient: only (d_in + d_out) * r parameters trained.
    """

    def __init__(
        self,
        base_linear: nn.Linear | CSALinear,
        rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.merged = False

        # 1. Base 1.58-bit Linear Layer (Frozen)
        if isinstance(base_linear, CSALinear):
            self.base = base_linear
        else:
            self.base = CSALinear(
                self.in_features,
                self.out_features,
                bias=base_linear.bias is not None
            )
            with torch.no_grad():
                self.base.latent_weight.data.copy_(base_linear.weight.data)
                if base_linear.bias is not None:
                    self.base.bias.data.copy_(base_linear.bias.data)
                self.base.update_alpha()

        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()
        self.base.pack_weights_for_inference()

        # 2. Trainable Low-Rank Adapters (A and B)
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, dtype=dtype)
        )

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize adapter A with Kaiming uniform and B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge_lora(self):
        """
        Merge trainable LoRA update Delta W = (gamma / r) * (B @ A) back into
        base latent weights, re-quantize to 1.58-bit ternary, and repack.
        """
        if self.merged:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base.latent_weight.data.add_(delta_w.to(self.base.latent_weight.dtype))
            self.base.update_alpha()
            self.base.pack_weights_for_inference()

        self.merged = True

    def unmerge_lora(self):
        """Undo LoRA merge and restore base weight."""
        if not self.merged:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base.latent_weight.data.sub_(delta_w.to(self.base.latent_weight.dtype))
            self.base.update_alpha()
            self.base.pack_weights_for_inference()

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute frozen CSA forward + low-rank adapter contribution.
        """
        # Base ternary forward pass (C++ CSA Popcount Engine)
        base_out = self.base(x)

        if self.merged:
            return base_out

        # Adapter path: (x @ A^T) @ B^T * scaling
        x_dropped = self.lora_dropout(x.to(self.lora_A.dtype))
        lora_act = F.linear(x_dropped, self.lora_A)
        lora_out = F.linear(lora_act, self.lora_B) * self.scaling

        return base_out + lora_out.to(base_out.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, lora_alpha={self.lora_alpha}, merged={self.merged}"
        )


def apply_ternary_lora(
    model: nn.Module,
    rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    target_layers: Optional[List[str]] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> nn.Module:
    """
    Inject TernaryLoRALinear into all linear layers matching target_layers.
    Freezes all non-LoRA parameters across the model.

    Args:
        model: Hugging Face or PyTorch model.
        rank: Rank r of the LoRA projection matrices.
        lora_alpha: LoRA scaling factor gamma.
        lora_dropout: Dropout probability on input to LoRA adapter.
        target_layers: Names of linear projections to adapt (defaults to attention & MLP projections).
        dtype: Data type for adapter parameters.
        verbose: Print injection statistics.

    Returns:
        Adapted model with only LoRA adapter parameters requiring gradients.
    """
    targets = set(target_layers or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # First freeze all existing parameters
    for param in model.parameters():
        param.requires_grad = False

    injected_count = 0

    def _replace_with_lora(module: nn.Module, parent_name: str = ""):
        nonlocal injected_count

        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(child, (nn.Linear, CSALinear)):
                if name in targets or any(t in name for t in targets):
                    lora_layer = TernaryLoRALinear(
                        base_linear=child,
                        rank=rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        dtype=dtype,
                    )
                    setattr(module, name, lora_layer)
                    injected_count += 1
                    continue

            _replace_with_lora(child, full_name)

    _replace_with_lora(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"[RKMJ LoRA] Successfully injected Ternary LoRA:")
        print(f"  • Adapted {injected_count} layers with rank={rank}, alpha={lora_alpha}.")
        print(f"  • Trainable Parameters: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / max(1, total_params):.3f}%)")

    return model

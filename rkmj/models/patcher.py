"""
Model Patcher & Rewriter Module for RKMJ-Core.
Automatically injects Quantization-Aware Training (QATCSALinear) layers
into Hugging Face and PyTorch transformer models while preserving sensitive boundary layers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
import torch
import torch.nn as nn

from rkmj.nn.qat import QATCSALinear
from rkmj.nn.linear import CSALinear
from rkmj.serialization.rkmjbin import save_rkmjbin

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PRESERVED_LAYER_PATTERNS = [
    "embed_tokens",
    "wte",
    "lm_head",
    "norm",
    "layernorm",
    "input_layernorm",
    "post_attention_layernorm",
]


def prepare_model_for_qat(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    preserved_layers: Optional[List[str]] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> nn.Module:
    """
    Recursively traverse model and replace standard nn.Linear layers matching target_layers
    with QATCSALinear layers, initializing master weights from the pretrained model.

    Preserves embeddings (embed_tokens), output classification heads (lm_head),
    and layer normalization layers (RMSNorm, LayerNorm) in high precision.

    Args:
        model: Hugging Face or PyTorch causal language model.
        target_layers: Names of linear projections to quantize (e.g. q_proj, gate_proj).
        preserved_layers: Names/substrings of layers that must NEVER be quantized.
        dtype: Data type for master weights (torch.float32 or torch.bfloat16).
        verbose: Print layer swap summary.

    Returns:
        The patched model ready for QAT fine-tuning.
    """
    targets = set(target_layers or DEFAULT_TARGET_LAYERS)
    preserved = set(preserved_layers or PRESERVED_LAYER_PATTERNS)

    swapped_count = 0
    skipped_count = 0

    def _should_preserve(full_name: str) -> bool:
        lower_name = full_name.lower()
        for p in preserved:
            if p.lower() in lower_name:
                return True
        return False

    def _replace_layers(module: nn.Module, parent_name: str = ""):
        nonlocal swapped_count, skipped_count

        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if _should_preserve(full_name):
                skipped_count += 1
                continue

            if isinstance(child, nn.Linear):
                # Check if leaf module name matches target layers
                if name in targets or any(t in name for t in targets):
                    has_bias = child.bias is not None
                    qat_linear = QATCSALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=has_bias,
                        dtype=dtype,
                        device=child.weight.device,
                    )

                    # Copy pretrained weight and bias
                    with torch.no_grad():
                        qat_linear.master_weight.data.copy_(child.weight.data.to(dtype))
                        if has_bias:
                            qat_linear.bias.data.copy_(child.bias.data.to(dtype))
                        qat_linear.synchronize_latent()

                    setattr(module, name, qat_linear)
                    swapped_count += 1
                    continue

            # Recurse down
            _replace_layers(child, full_name)

    _replace_layers(model)

    if verbose:
        print(f"[RKMJ Patcher] Successfully patched model for QAT:")
        print(f"  • Injected {swapped_count} QATCSALinear modules.")
        print(f"  • Preserved {skipped_count} sensitive high-precision layers.")

    return model


def export_model_to_rkmjbin(
    model: nn.Module,
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> int:
    """
    Pack all QATCSALinear and CSALinear layers into native 2-bit uint32 format
    and export the model to an optimized .rkmjbin file.
    """
    model.eval()

    # Pack all linear layers
    packed_count = 0
    for m in model.modules():
        if isinstance(m, (QATCSALinear, CSALinear)):
            m.pack_weights_for_inference()
            packed_count += 1

    if verbose:
        print(f"[RKMJ Patcher] Pre-packed {packed_count} linear layers for inference.")

    return save_rkmjbin(model, save_path, config=config, verbose=verbose)

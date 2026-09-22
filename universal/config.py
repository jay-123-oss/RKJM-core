"""
Universal Model Architecture Configuration and Auto-Detection.
Detects architecture types across Qwen, LLaMA, Mistral, Gemma, and Phi model families.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("universal.config")


@dataclass
class ArchitectureProfile:
    """Metadata profile defining architecture traits and tensor key patterns."""
    model_family: str  # "qwen", "llama", "mistral", "gemma", "phi", etc.
    qkv_has_bias: bool = False
    norm_type: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    norm_add_unit: bool = False  # Gemma uses RMSNorm(x * (1 + w))
    mlp_activation: str = "silu"  # "silu", "gelu", etc.
    default_rope_theta: float = 10000.0
    embedding_weight_scale: Optional[float] = None  # Gemma scales embeddings by sqrt(dim)
    eos_token_ids: List[int] = field(default_factory=lambda: [2])
    chat_template_type: str = "chatml"  # "chatml", "llama3", "mistral", "gemma"

    # Tensor name mapping prefixes
    layer_prefix: str = "model.layers."
    embed_tokens_key: str = "model.embed_tokens.weight"
    norm_key: str = "model.norm.weight"
    lm_head_key: str = "lm_head.weight"


def detect_architecture_profile(config_dict: Dict[str, Any]) -> ArchitectureProfile:
    """
    Inspects HuggingFace config.json and returns the appropriate ArchitectureProfile.
    """
    model_type = config_dict.get("model_type", "").lower()
    architectures = [a.lower() for a in config_dict.get("architectures", [])]
    arch_str = " ".join(architectures)

    # 1. Qwen Family
    if "qwen2" in model_type or "qwen" in model_type or "qwen" in arch_str:
        return ArchitectureProfile(
            model_family="qwen",
            qkv_has_bias=True,
            norm_type="rmsnorm",
            norm_add_unit=False,
            mlp_activation="silu",
            default_rope_theta=1000000.0,
            eos_token_ids=[151645, 151643],
            chat_template_type="chatml",
            layer_prefix="model.layers.",
            embed_tokens_key="model.embed_tokens.weight",
            norm_key="model.norm.weight",
            lm_head_key="lm_head.weight",
        )

    # 2. LLaMA Family (LLaMA 2, LLaMA 3, 3.1, 3.2, Vicuna, Alpaca)
    if "llama" in model_type or "llama" in arch_str:
        # Check if LLaMA 3
        is_llama3 = (
            config_dict.get("vocab_size", 0) >= 128000
            or config_dict.get("rope_theta", 10000.0) >= 500000.0
        )
        return ArchitectureProfile(
            model_family="llama",
            qkv_has_bias=False,
            norm_type="rmsnorm",
            norm_add_unit=False,
            mlp_activation="silu",
            default_rope_theta=500000.0 if is_llama3 else 10000.0,
            eos_token_ids=[128009, 128001] if is_llama3 else [2],
            chat_template_type="llama3" if is_llama3 else "llama2",
            layer_prefix="model.layers.",
            embed_tokens_key="model.embed_tokens.weight",
            norm_key="model.norm.weight",
            lm_head_key="lm_head.weight",
        )

    # 3. Mistral / Mixtral Family
    if "mistral" in model_type or "mistral" in arch_str:
        return ArchitectureProfile(
            model_family="mistral",
            qkv_has_bias=False,
            norm_type="rmsnorm",
            norm_add_unit=False,
            mlp_activation="silu",
            default_rope_theta=config_dict.get("rope_theta", 1000000.0),
            eos_token_ids=[2],
            chat_template_type="mistral",
            layer_prefix="model.layers.",
            embed_tokens_key="model.embed_tokens.weight",
            norm_key="model.norm.weight",
            lm_head_key="lm_head.weight",
        )

    # 4. Gemma / Gemma 2 Family
    if "gemma" in model_type or "gemma" in arch_str:
        dim = config_dict.get("hidden_size", 2048)
        return ArchitectureProfile(
            model_family="gemma",
            qkv_has_bias=False,
            norm_type="rmsnorm",
            norm_add_unit=True,
            mlp_activation="gelu",
            default_rope_theta=10000.0,
            embedding_weight_scale=float(dim ** 0.5),
            eos_token_ids=[1],
            chat_template_type="gemma",
            layer_prefix="model.layers.",
            embed_tokens_key="model.embed_tokens.weight",
            norm_key="model.norm.weight",
            lm_head_key="lm_head.weight",
        )

    # Generic Auto-Fallback (Default Transformer)
    logger.warning(
        f"Unknown model_type '{model_type}'. Falling back to standard LLaMA-compatible Transformer profile."
    )
    return ArchitectureProfile(
        model_family="generic",
        qkv_has_bias=False,
        norm_type="rmsnorm",
        norm_add_unit=False,
        mlp_activation="silu",
        default_rope_theta=config_dict.get("rope_theta", 10000.0),
        eos_token_ids=[2],
        chat_template_type="chatml",
        layer_prefix="model.layers.",
        embed_tokens_key="model.embed_tokens.weight",
        norm_key="model.norm.weight",
        lm_head_key="lm_head.weight",
    )

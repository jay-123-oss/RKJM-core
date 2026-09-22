"""
Universal Multi-Model Adapter for RKMJ-Core.
Supports Qwen, LLaMA, Mistral, and Gemma models out-of-core.
"""

from .config import ArchitectureProfile, detect_architecture_profile
from .converter import UniversalStreamingPTQConverter
from .runner import (
    UniversalLocalRunner,
    UniversalAttention,
    UniversalMLP,
    UniversalTransformerLayer,
)
from .rope import UniversalRotaryEmbedding, apply_rotary_pos_emb

__all__ = [
    "ArchitectureProfile",
    "detect_architecture_profile",
    "UniversalStreamingPTQConverter",
    "UniversalLocalRunner",
    "UniversalAttention",
    "UniversalMLP",
    "UniversalTransformerLayer",
    "UniversalRotaryEmbedding",
    "apply_rotary_pos_emb",
]

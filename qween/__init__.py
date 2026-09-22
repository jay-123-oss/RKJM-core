"""
Backward-compatibility shim for Qwen Engine (now powered by Universal Multi-Model Adapter).
"""

from universal import (
    UniversalStreamingPTQConverter as QwenStreamingPTQConverter,
    UniversalLocalRunner as QwenLocalRunner,
    UniversalRotaryEmbedding as QwenRotaryEmbedding,
    apply_rotary_pos_emb,
)
from qween.bootstrap import check_dependencies, ensure_dependencies

__all__ = [
    "QwenRotaryEmbedding",
    "apply_rotary_pos_emb",
    "QwenStreamingPTQConverter",
    "QwenLocalRunner",
    "check_dependencies",
    "ensure_dependencies",
]

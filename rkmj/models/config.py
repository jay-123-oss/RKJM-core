"""
Configuration dataclasses for RKMJ-Core Models.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class RKMJConfig:
    """
    Hyperparameter configuration for RKMJ 1.58-bit Transformer Models.
    """
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    intermediate_dim: Optional[int] = None
    max_seq_len: int = 1024
    norm_eps: float = 1e-6
    attn_dropout: float = 0.0
    bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.intermediate_dim is None:
            # LLaMA heuristic: 8/3 * dim padded to multiples of 64
            dim_calc = int(2 * 4 * self.dim / 3)
            self.intermediate_dim = ((dim_calc + 63) // 64) * 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RKMJConfig:
        return cls(**data)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> RKMJConfig:
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

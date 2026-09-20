"""
RKMJ-Core: Revolutionary 1.58-bit (Ternary) LLM Framework for CPU.
Replaces FP32 GEMM with Carry-Save Addition (CSA) and bitwise popcount.
"""

from rkmj import nn
from rkmj import models
from rkmj import serialization
from rkmj import engine

__version__ = "0.1.0"

__all__ = [
    "nn",
    "models",
    "serialization",
    "engine",
    "__version__",
]

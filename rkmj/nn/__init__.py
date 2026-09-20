"""
RKMJ-Core Neural Network Layers (rkmj.nn).
Drop-in replacement layers powered by 1.58-bit Carry-Save Addition (CSA) engine.
"""

from rkmj.nn.norm import RMSNorm
from rkmj.nn.linear import CSALinear
from rkmj.nn.attention import CSASelfAttention
from rkmj.nn.mlp import CSAMLP
from rkmj.nn.block import CSATransformerBlock
from rkmj.nn.audio import AudioPatchEmbed, RKMJAudioTransformer

__all__ = [
    "RMSNorm",
    "CSALinear",
    "CSASelfAttention",
    "CSAMLP",
    "CSATransformerBlock",
    "AudioPatchEmbed",
    "RKMJAudioTransformer",
]


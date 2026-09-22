"""
RKMJ-Core Neural Network Layers (rkmj.nn).
Drop-in replacement layers powered by 1.58-bit Carry-Save Addition (CSA) engine.
"""

from rkmj.nn.layers import RMSNorm, CSALinear, CSATransformerBlock, CSALinearFunction
from rkmj.nn.audio import AudioPatchEmbed, RKMJAudioTransformer
from rkmj.nn.qat import QATCSALinear, TernaryQuantizeSTE, ActivationQuantizeSTE
from rkmj.nn.lora import TernaryLoRALinear, apply_ternary_lora

__all__ = [
    "RMSNorm",
    "CSALinear",
    "CSASelfAttention",
    "CSAMLP",
    "CSATransformerBlock",
    "AudioPatchEmbed",
    "RKMJAudioTransformer",
    "QATCSALinear",
    "TernaryQuantizeSTE",
    "ActivationQuantizeSTE",
    "TernaryLoRALinear",
    "apply_ternary_lora",
]

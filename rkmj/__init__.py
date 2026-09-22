"""
RKMJ-Core: High-Performance 1.58-Bit (Ternary {-1, 0, +1}) AI Systems Framework.
Enterprise-grade CPU Execution, STE Autograd, Dynamic Memory Arena, and Grouped Quantizer.
"""

import os
from typing import Optional, Union
import torch

# Submodules
from rkmj import core
from rkmj import quantizer
from rkmj import nn
from rkmj import optim
from rkmj import trainer
from rkmj import tokenizer
from rkmj import models
from rkmj import serialization

# Key classes and functions
from rkmj.core import MemorySupervisor, get_supervisor, load, force_heap_trim
from rkmj.quantizer import (
    quantize_ternary_grouped,
    dequantize_ternary_grouped,
    pack_ternary,
    unpack_ternary,
)
from rkmj.nn import CSALinear, RMSNorm, CSATransformerBlock
from rkmj.optim import AdamSTE
from rkmj.trainer import Trainer
from rkmj.tokenizer import TokenizerPreserver
from rkmj.auto import AutoModel

__version__ = "0.2.0"

def init(num_threads: Optional[int] = None) -> None:
    """
    Initialize RKMJ-Core hardware scheduler, OpenMP thread affinity, and memory supervisor.
    """
    try:
        from rkmj import _C
        if hasattr(_C, "set_thread_affinity"):
            threads = num_threads if num_threads is not None else -1
            _C.set_thread_affinity(threads)
    except ImportError:
        pass

    supervisor = get_supervisor()
    supervisor.start_background_watchdog(interval_sec=2.0)

def quantize(
    tensor_or_model: Union[torch.Tensor, torch.nn.Module],
    group_size: int = 64,
) -> Union[tuple, torch.nn.Module]:
    """
    Convenience API for 1.58-bit grouped ternary quantization.
    If a tensor is passed, returns (w_packed, scales).
    If a PyTorch module is passed, returns a quantized CSA module.
    """
    if isinstance(tensor_or_model, torch.Tensor):
        return quantize_ternary_grouped(tensor_or_model, group_size=group_size)
    elif isinstance(tensor_or_model, torch.nn.Module):
        # Convert Linear layers to CSALinear
        for name, module in tensor_or_model.named_children():
            if isinstance(module, torch.nn.Linear):
                csa_layer = CSALinear.from_linear(module, group_size=group_size)
                setattr(tensor_or_model, name, csa_layer)
            else:
                quantize(module, group_size=group_size)
        return tensor_or_model
    else:
        raise TypeError(f"Expected torch.Tensor or torch.nn.Module, got {type(tensor_or_model)}")

__all__ = [
    "core",
    "quantizer",
    "nn",
    "optim",
    "trainer",
    "tokenizer",
    "models",
    "serialization",
    "AutoModel",
    "MemorySupervisor",
    "get_supervisor",
    "quantize_ternary_grouped",
    "dequantize_ternary_grouped",
    "pack_ternary",
    "unpack_ternary",
    "CSALinear",
    "RMSNorm",
    "CSATransformerBlock",
    "AdamSTE",
    "Trainer",
    "TokenizerPreserver",
    "init",
    "quantize",
    "load",
    "force_heap_trim",
    "__version__",
]

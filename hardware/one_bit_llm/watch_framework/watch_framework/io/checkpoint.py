"""1-Bit Bit-Packed Serialization Engine (.wfbin).

Implements:
1. wf.save_checkpoint(model, filepath): Converts {-1, +1} weights to binary bitmasks,
   packs them into uint8 bytes via numpy.packbits (32x compression), and exports .wfbin.
2. wf.load_checkpoint(model, filepath): Reads .wfbin, unpacks bytes back to {-1, +1},
   and updates model weights in-place with 100% numerical parity.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict
import numpy as np
import torch
from torch import nn

from ..nn.linear import WatchLinear
from ..nn.conv import WatchConv2d
from ..nn.quant import dynamic_alpha

MAGIC_HEADER = b"WFBIN_V1\x00"


def save_checkpoint(model: nn.Module, filepath: str) -> None:
    """Save a model with 1-bit quantized layers packed into uint8 bytes (32x weight compression).

    Args:
        model: PyTorch model containing WatchLinear or WatchConv2d layers.
        filepath: Output path ending in .wfbin.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    watch_layers: Dict[str, Any] = {}
    extra_state: Dict[str, Any] = {}

    # Iterate named modules to extract WatchGrid components
    for name, module in model.named_modules():
        if isinstance(module, (WatchLinear, WatchConv2d)):
            w = module.weight.detach().cpu()
            alpha_val = float(dynamic_alpha(w).item())

            # Convert {-1, +1} to binary mask {0, 1} where w >= 0 -> 1, w < 0 -> 0
            bin_mask = (w >= 0.0).numpy().astype(np.uint8)
            numel = int(w.numel())
            packed_bytes = np.packbits(bin_mask)

            bias_array = (
                module.bias.detach().cpu().numpy()
                if module.bias is not None
                else None
            )

            watch_layers[name] = {
                "layer_type": "WatchLinear" if isinstance(module, WatchLinear) else "WatchConv2d",
                "shape": tuple(w.shape),
                "numel": numel,
                "alpha": alpha_val,
                "packed_weights": packed_bytes,
                "bias": bias_array,
            }

    # Save any non-watch parameters or buffers
    state_dict = model.state_dict()
    for param_name, tensor in state_dict.items():
        # Check if this parameter belongs to an already-packed watch layer's weight
        matched = False
        for watch_name in watch_layers:
            if param_name == f"{watch_name}.weight" or (watch_name and param_name.startswith(f"{watch_name}.")):
                matched = True
                break
        if not matched:
            extra_state[param_name] = tensor.cpu()

    payload = {
        "header": MAGIC_HEADER,
        "watch_layers": watch_layers,
        "extra_state": extra_state,
    }

    with open(filepath, "wb") as f:
        f.write(MAGIC_HEADER)
        pickle.dump(payload, f, protocol=5)


def load_checkpoint(model: nn.Module, filepath: str) -> None:
    """Load a compressed .wfbin checkpoint, unpack weights, and restore model parameters in-place.

    Args:
        model: PyTorch model matching the checkpoint architecture.
        filepath: Path to the .wfbin binary checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    with open(filepath, "rb") as f:
        header = f.read(len(MAGIC_HEADER))
        if header != MAGIC_HEADER:
            raise ValueError(f"Invalid .wfbin header in {filepath}")
        payload = pickle.load(f)

    watch_layers = payload.get("watch_layers", {})
    extra_state = payload.get("extra_state", {})

    named_modules = dict(model.named_modules())

    for name, layer_info in watch_layers.items():
        if name not in named_modules:
            continue
        module = named_modules[name]
        shape = layer_info["shape"]
        numel = layer_info["numel"]
        alpha_val = layer_info["alpha"]
        packed_bytes = layer_info["packed_weights"]
        bias_array = layer_info["bias"]

        # Unpack uint8 byte array back to {0, 1} binary bits and truncate padding
        unpacked_bits = np.unpackbits(packed_bytes)[:numel]

        # Map {0, 1} back to {-1.0, +1.0}
        w_sign = np.where(unpacked_bits > 0, 1.0, -1.0).astype(np.float32)

        # Reconstruct latent weight scaled by alpha: W = alpha * W_q
        # Note: dynamic_alpha(alpha * W_q) == alpha and sign(alpha * W_q) == W_q
        w_restored = (w_sign * alpha_val).reshape(shape)

        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(w_restored).to(module.weight.device))
            if bias_array is not None and module.bias is not None:
                module.bias.copy_(torch.from_numpy(bias_array).to(module.bias.device))

    # Restore extra state
    if extra_state:
        filtered_state = {k: v.to(model.state_dict()[k].device) for k, v in extra_state.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state, strict=False)

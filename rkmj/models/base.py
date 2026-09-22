"""
Base Generative Model Class for RKMJ Architecture.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from rkmj.models.config import RKMJConfig
from rkmj.nn.linear import CSALinear
from rkmj.serialization.rkmjbin import load_rkmjbin, save_rkmjbin


class RKMJBaseModel(nn.Module):
    """
    Abstract Base Class for RKMJ Generative Models.
    Provides parameter counting, memory analysis, 2-bit packing, and serialization methods.
    """

    def __init__(self, config: RKMJConfig):
        super().__init__()
        self.config = config

    def count_parameters(self) -> Dict[str, int]:
        """Count total, latent CSA, and standard FP32 parameters."""
        total = sum(p.numel() for p in self.parameters())
        csa = sum(
            p.numel()
            for m in self.modules()
            if isinstance(m, CSALinear)
            for p in m.parameters()
        )
        standard = total - csa
        return {
            "total": total,
            "csa_1_58_bit": csa,
            "standard_fp32": standard,
            "csa_ratio_pct": (100.0 * csa / max(total, 1)),
        }

    def pack_for_inference(self):
        """Pack all CSALinear layers into 2-bit uint32 arrays for high-speed inference."""
        self.eval()
        for m in self.modules():
            if isinstance(m, CSALinear):
                m.pack_weights_for_inference()

    def pack_weights_for_inference(self):
        """Alias for pack_for_inference."""
        self.pack_for_inference()

    def unpack_weights(self):
        """Unpack all 2-bit weights back into FP32 ternary representation."""
        for m in self.modules():
            if isinstance(m, CSALinear):
                m.unpack_weights()
                m.is_packed = False

    def save_pretrained(self, save_directory: str, filename: str = "model.rkmjbin") -> str:
        """Save the model and configuration in the native .rkmjbin format."""
        os.makedirs(save_directory, exist_ok=True)
        file_path = os.path.join(save_directory, filename)
        config_path = os.path.join(save_directory, "config.json")

        self.config.save(config_path)
        save_rkmjbin(self, file_path, config=self.config.to_dict())
        return file_path

    @classmethod
    def from_pretrained(cls, save_directory: str, filename: str = "model.rkmjbin", device: str = "cpu"):
        """Load an RKMJ model from a directory containing .rkmjbin and config.json."""
        config_path = os.path.join(save_directory, "config.json")
        file_path = os.path.join(save_directory, filename)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {save_directory}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {filename} in {save_directory}")

        config = RKMJConfig.load(config_path)
        model = cls(config).to(device)

        state_dict, _ = load_rkmjbin(file_path, device=torch.device(device))
        model.load_state_dict(state_dict, strict=False)
        return model

"""
Multimodal Audio Processing Layers for RKMJ-Core 1.58-bit Engine.

Provides:
  - AudioPatchEmbed: Projects continuous 2D Mel-spectrograms into discrete sequence tokens.
  - RKMJAudioTransformer: Bidirectional 1.58-bit transformer backbone with RMSNorm + CSALinear head.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn.block import CSATransformerBlock
from rkmj.nn.linear import CSALinear
from rkmj.nn.norm import RMSNorm


class AudioPatchEmbed(nn.Module):
    """
    Audio Patch Embedding Layer for Continuous Temporal/Spectral Signals.

    Ingests 2D Mel-spectrogram inputs of shape (Batch, 1, n_mels, time_steps)
    and uses a standard precision (FP32) Conv2d boundary patch projection
    to transform continuous spectrogram patches into sequence tokens of shape
    (Batch, n_patches, dim).
    """

    def __init__(
        self,
        n_mels: int = 128,
        time_steps: int = 1024,
        patch_size: Union[int, Tuple[int, int]] = (16, 16),
        in_channels: int = 1,
        dim: int = 512,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.n_mels = n_mels
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim

        self.grid_size = (n_mels // patch_size[0], time_steps // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Boundary Conv2d projection maintained in standard precision (FP32)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel-spectrogram tensor of shape (Batch, 1, n_mels, time_steps)
               or (Batch, n_mels, time_steps).

        Returns:
            Projected tokens of shape (Batch, n_patches, dim).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Boundary Conv2D: [B, in_channels, n_mels, time_steps] -> [B, dim, grid_h, grid_w]
        x = self.proj(x)

        # Flatten spatial grid into sequence: [B, dim, grid_h, grid_w] -> [B, n_patches, dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class RKMJAudioTransformer(nn.Module):
    """
    RKMJ 1.58-bit Audio Transformer for Keyword Spotting & Audio Classification.

    Architecture:
      1. AudioPatchEmbed: 2D Mel-spectrogram -> Sequence Tokens [B, N, D]
      2. Learnable [CLS] classification token prepended -> [B, N + 1, D]
      3. Learnable temporal/spectral positional embeddings added
      4. Bidirectional Backbone: Stack of CSATransformerBlocks with causal_mask=False
      5. Classification Head: RMSNorm + 1.58-bit CSALinear classifier
    """

    def __init__(
        self,
        n_mels: int = 128,
        time_steps: int = 1024,
        patch_size: Union[int, Tuple[int, int]] = (16, 16),
        in_channels: int = 1,
        num_classes: int = 10,
        dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        eps: float = 1e-6,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        # 1. Boundary Spectrogram Patch Projection
        self.patch_embed = AudioPatchEmbed(
            n_mels=n_mels,
            time_steps=time_steps,
            patch_size=patch_size,
            in_channels=in_channels,
            dim=dim,
        )
        num_patches = self.patch_embed.num_patches

        # 2. Learnable CLS Token & Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # 3. 1.58-bit CSA Transformer Backbone (Bidirectional Attention)
        self.blocks = nn.ModuleList([
            CSATransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_dim=intermediate_dim,
                eps=eps,
                attn_dropout=attn_dropout,
                bias=bias,
            )
            for _ in range(depth)
        ])

        # 4. Classification Head: RMSNorm + 1.58-bit CSALinear Head
        self.norm = RMSNorm(dim, eps=eps)
        self.head = CSALinear(dim, num_classes, bias=bias)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize positional embeddings and CLS token."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def pack_weights_for_inference(self):
        """Pack all constituent CSALinear projection layers for frozen 2-bit inference."""
        for m in self.modules():
            if isinstance(m, CSALinear):
                m.pack_weights_for_inference()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel-spectrogram tensor of shape (Batch, 1, n_mels, time_steps)
               or (Batch, n_mels, time_steps).

        Returns:
            Logits tensor of shape (Batch, num_classes).
        """
        # [B, N, D]
        tokens = self.patch_embed(x)
        batch_size = tokens.shape[0]

        # Prepend [CLS] token: [B, 1 + N, D]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, tokens), dim=1)

        # Add positional embedding
        seq_len = x_seq.size(1)
        if seq_len == self.pos_embed.size(1):
            x_seq = x_seq + self.pos_embed
        elif seq_len < self.pos_embed.size(1):
            x_seq = x_seq + self.pos_embed[:, :seq_len, :]
        else:
            # Interpolate if spectrogram length exceeds predefined time steps
            pos_tokens = self.pos_embed[:, 1:, :].transpose(1, 2)
            pos_interp = F.interpolate(
                pos_tokens, size=seq_len - 1, mode="linear", align_corners=False
            ).transpose(1, 2)
            full_pos = torch.cat((self.pos_embed[:, :1, :], pos_interp), dim=1)
            x_seq = x_seq + full_pos

        # Bidirectional Transformer Backbone (causal_mask=False)
        for block in self.blocks:
            x_seq = block(x_seq, causal_mask=False)

        # Extract [CLS] token representation
        cls_rep = x_seq[:, 0]
        norm_cls = self.norm(cls_rep)
        logits = self.head(norm_cls)
        return logits

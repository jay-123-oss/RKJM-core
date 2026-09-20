"""
Unit tests for RKMJ-Core Multimodal Audio Module (AudioPatchEmbed and RKMJAudioTransformer).
Verifies:
  - Tensor shapes through AudioPatchEmbed for 2D Mel-spectrogram inputs.
  - Bidirectional attention and classification head in RKMJAudioTransformer.
  - Backward gradient propagation through C++ STE kernel.
  - 2-bit weight packing and frozen inference execution.
"""

from __future__ import annotations

import os
import sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.nn.audio import AudioPatchEmbed, RKMJAudioTransformer
from rkmj.nn.linear import CSALinear


class TestAudioModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_audio_patch_embed_shape_4d(self):
        """Test AudioPatchEmbed with 4D Mel-spectrogram (B, 1, n_mels, time_steps)."""
        batch_size = 2
        n_mels = 128
        time_steps = 1024
        dim = 256
        patch_size = (16, 16)

        patch_embed = AudioPatchEmbed(
            n_mels=n_mels,
            time_steps=time_steps,
            patch_size=patch_size,
            in_channels=1,
            dim=dim,
        )

        expected_patches = (n_mels // 16) * (time_steps // 16)
        self.assertEqual(patch_embed.num_patches, expected_patches)

        x = torch.randn(batch_size, 1, n_mels, time_steps)
        tokens = patch_embed(x)

        self.assertEqual(tokens.shape, (batch_size, expected_patches, dim))
        self.assertFalse(torch.isnan(tokens).any())

    def test_audio_patch_embed_shape_3d(self):
        """Test AudioPatchEmbed with 3D input (B, n_mels, time_steps)."""
        batch_size = 3
        n_mels = 64
        time_steps = 256
        dim = 128
        patch_size = (16, 16)

        patch_embed = AudioPatchEmbed(
            n_mels=n_mels,
            time_steps=time_steps,
            patch_size=patch_size,
            dim=dim,
        )

        expected_patches = (n_mels // 16) * (time_steps // 16)
        x = torch.randn(batch_size, n_mels, time_steps)
        tokens = patch_embed(x)

        self.assertEqual(tokens.shape, (batch_size, expected_patches, dim))
        self.assertFalse(torch.isnan(tokens).any())

    def test_audio_transformer_forward_shape(self):
        """Test forward pass of RKMJAudioTransformer for audio classification."""
        batch_size = 2
        n_mels = 64
        time_steps = 128
        num_classes = 10
        dim = 64

        model = RKMJAudioTransformer(
            n_mels=n_mels,
            time_steps=time_steps,
            patch_size=(16, 16),
            num_classes=num_classes,
            dim=dim,
            depth=2,
            num_heads=4,
        )
        model.eval()

        x = torch.randn(batch_size, 1, n_mels, time_steps)
        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertFalse(torch.isnan(logits).any())

    def test_audio_transformer_backward_gradients_ste(self):
        """
        Verify backward gradient propagation from audio classification loss through
        RMSNorm, CSATransformerBlocks, and boundary AudioPatchEmbed via C++ STE kernel.
        """
        batch_size = 2
        n_mels = 64
        time_steps = 128
        num_classes = 5
        dim = 64

        model = RKMJAudioTransformer(
            n_mels=n_mels,
            time_steps=time_steps,
            patch_size=(16, 16),
            num_classes=num_classes,
            dim=dim,
            depth=2,
            num_heads=4,
            bias=True,
        )
        model.train()

        x = torch.randn(batch_size, 1, n_mels, time_steps, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # 1. Input Spectrogram Gradients
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertFalse(torch.isnan(x.grad).any())

        # 2. Boundary Conv2d Patch Projection Gradients
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)
        self.assertEqual(model.patch_embed.proj.weight.grad.shape, model.patch_embed.proj.weight.shape)

        # 3. Learnable Token & Position Embedding Gradients
        self.assertIsNotNone(model.cls_token.grad)
        self.assertIsNotNone(model.pos_embed.grad)

        # 4. Backbone CSATransformerBlock STE Gradients (C++ STE Kernel)
        first_block = model.blocks[0]
        self.assertIsNotNone(first_block.self_attn.q_proj.latent_weight.grad)
        self.assertIsNotNone(first_block.self_attn.k_proj.latent_weight.grad)
        self.assertIsNotNone(first_block.self_attn.v_proj.latent_weight.grad)
        self.assertIsNotNone(first_block.self_attn.o_proj.latent_weight.grad)
        self.assertIsNotNone(first_block.mlp.gate_proj.latent_weight.grad)
        self.assertIsNotNone(first_block.mlp.up_proj.latent_weight.grad)
        self.assertIsNotNone(first_block.mlp.down_proj.latent_weight.grad)

        # 5. Classification Head STE Gradients
        self.assertIsNotNone(model.head.latent_weight.grad)
        self.assertIsNotNone(model.head.alpha.grad)
        self.assertIsNotNone(model.head.bias.grad)

    def test_audio_transformer_inference_packing(self):
        """Test inference execution after packing weights into native 2-bit format."""
        batch_size = 2
        n_mels = 64
        time_steps = 128
        num_classes = 8
        dim = 64

        model = RKMJAudioTransformer(
            n_mels=n_mels,
            time_steps=time_steps,
            patch_size=(16, 16),
            num_classes=num_classes,
            dim=dim,
            depth=2,
            num_heads=4,
        )

        # Pack weights for inference
        model.pack_weights_for_inference()
        model.eval()

        # Check all CSALinear layers are packed
        csa_count = 0
        for m in model.modules():
            if isinstance(m, CSALinear):
                self.assertTrue(m.is_packed)
                csa_count += 1
        self.assertGreater(csa_count, 0)

        # Execute forward pass with packed CSA Popcount engine
        x = torch.randn(batch_size, 1, n_mels, time_steps)
        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertFalse(torch.isnan(logits).any())


if __name__ == "__main__":
    unittest.main()

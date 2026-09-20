"""
Unit tests for Straight-Through Estimator (STE) autograd gradient propagation.
"""

from __future__ import annotations

import os
import sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.nn.linear import CSALinear
from rkmj.nn.block import CSATransformerBlock


class TestBackwardSTE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_csa_linear_gradients(self):
        layer = CSALinear(64, 32, bias=True)
        x = torch.randn(4, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertIsNotNone(layer.latent_weight.grad)
        self.assertIsNotNone(layer.alpha.grad)
        self.assertIsNotNone(layer.bias.grad)

    def test_transformer_block_gradients(self):
        dim = 128
        block = CSATransformerBlock(dim=dim, num_heads=4)
        x = torch.randn(2, 16, dim, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # Verify Attention Projections
        self.assertIsNotNone(block.self_attn.q_proj.latent_weight.grad)
        self.assertIsNotNone(block.self_attn.k_proj.latent_weight.grad)
        self.assertIsNotNone(block.self_attn.v_proj.latent_weight.grad)
        self.assertIsNotNone(block.self_attn.o_proj.latent_weight.grad)

        # Verify MLP Projections
        self.assertIsNotNone(block.mlp.gate_proj.latent_weight.grad)
        self.assertIsNotNone(block.mlp.up_proj.latent_weight.grad)
        self.assertIsNotNone(block.mlp.down_proj.latent_weight.grad)

        # Verify RMSNorm layers
        self.assertIsNotNone(block.input_layernorm.weight.grad)
        self.assertIsNotNone(block.post_attention_layernorm.weight.grad)

    def test_optimizer_convergence(self):
        layer = CSALinear(32, 16)
        layer.alpha.requires_grad = False
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
        x = torch.randn(8, 32)
        target = torch.randn(8, 16)
        criterion = nn.MSELoss()

        init_loss = criterion(layer(x), target).item()
        for _ in range(5):
            opt.zero_grad()
            l = criterion(layer(x), target)
            l.backward()
            opt.step()
            layer.update_alpha()
        final_loss = criterion(layer(x), target).item()

        self.assertLess(final_loss, init_loss)


if __name__ == "__main__":
    unittest.main()

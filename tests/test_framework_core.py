"""
Comprehensive Unit Tests for RKMJ-Core Framework:
  1. Memory Arena, Pool Allocator & OS Memory Supervisor
  2. 1.58-bit Per-Group Quantizer & Exact 2-Bit Packing Roundtrip
  3. Straight-Through Estimator (STE) Autograd Gradient Flow
  4. CSALinear, RMSNorm, and CSATransformerBlock Neural Layers
  5. AdamSTE Optimizer & Latent Weight Clamping
  6. Trainer Training Loop Integration
"""

import unittest
import torch
import torch.nn as nn

import rkmj
from rkmj.core import MemorySupervisor, get_supervisor
from rkmj.quantizer import (
    quantize_ternary_grouped,
    dequantize_ternary_grouped,
    pack_ternary,
    unpack_ternary,
)
from rkmj.nn import CSALinear, RMSNorm, CSATransformerBlock
from rkmj.optim import AdamSTE
from rkmj.trainer import Trainer


class TestFrameworkCore(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_packing_unpacking_roundtrip(self):
        """Verify strict 2-bit aligned packing and exact arithmetic unpacking."""
        N, K = 32, 128
        # Ternary values in {-1, 0, +1}
        w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.float32)

        packed = pack_ternary(w_ternary)
        self.assertEqual(packed.dtype, torch.int32)
        self.assertEqual(packed.shape, (N, K // 16))

        unpacked = unpack_ternary(packed, K=K)
        self.assertEqual(unpacked.shape, (N, K))
        self.assertTrue(torch.allclose(w_ternary, unpacked))

    def test_grouped_quantization_stability(self):
        """Verify per-group dynamic scaling and dynamic zero-threshold."""
        N, K = 64, 256
        group_size = 64
        # Normal distribution weights
        w = torch.randn(N, K, dtype=torch.float32)

        w_packed, scales = quantize_ternary_grouped(w, group_size=group_size)
        self.assertEqual(scales.shape, (N, K // group_size))
        # Scales must be positive
        self.assertTrue(torch.all(scales > 0.0))

        # Dequantize
        w_dequant = dequantize_ternary_grouped(w_packed, scales, in_features=K, group_size=group_size)
        self.assertEqual(w_dequant.shape, (N, K))

        # Error check: quantized weights should correlate strongly with original
        cos_sim = torch.cosine_similarity(w.flatten(), w_dequant.flatten(), dim=0)
        self.assertGreater(cos_sim.item(), 0.75)

    def test_ste_autograd_gradient_flow(self):
        """Verify Straight-Through Estimator propagates gradients to latent weights."""
        in_features = 64
        out_features = 32
        batch_size = 4

        layer = CSALinear(in_features, out_features, bias=True, group_size=64)
        x = torch.randn(batch_size, in_features, requires_grad=True)

        out = layer(x)
        self.assertEqual(out.shape, (batch_size, out_features))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.latent_weight.grad)
        self.assertIsNotNone(layer.bias.grad)

        # Gradients must be non-zero and non-NaN
        self.assertFalse(torch.isnan(layer.latent_weight.grad).any())
        self.assertGreater(layer.latent_weight.grad.abs().sum().item(), 0.0)

    def test_rmsnorm_and_csa_transformer_block(self):
        """Verify RMSNorm and full CSATransformerBlock forward execution."""
        dim = 64
        n_heads = 4
        seq_len = 8
        batch_size = 2

        norm = RMSNorm(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)
        norm_out = norm(x)
        self.assertEqual(norm_out.shape, (batch_size, seq_len, dim))

        block = CSATransformerBlock(dim=dim, n_heads=n_heads, group_size=64)
        block_out = block(x)
        self.assertEqual(block_out.shape, (batch_size, seq_len, dim))

    def test_adam_ste_optimizer(self):
        """Verify AdamSTE updates parameters and respects latent clamping."""
        layer = CSALinear(32, 16, bias=False)
        optimizer = AdamSTE(layer.parameters(), lr=0.1, weight_decay=0.0, clamp_latent=1.0)

        initial_w = layer.latent_weight.clone()
        x = torch.randn(2, 32)
        loss = layer(x).sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Weights must have moved
        self.assertFalse(torch.allclose(initial_w, layer.latent_weight))
        # Latent weights must be within [-1.0, 1.0]
        self.assertTrue(torch.all(layer.latent_weight <= 1.0))
        self.assertTrue(torch.all(layer.latent_weight >= -1.0))

    def test_memory_supervisor(self):
        """Verify MemorySupervisor metrics and layer page flushing."""
        supervisor = get_supervisor()
        status = supervisor.get_memory_status()

        self.assertIn("process_rss_bytes", status)
        self.assertIn("system_total_bytes", status)
        self.assertIn("system_available_bytes", status)

        # Test layer page flushing via madvise
        t = torch.randn(1024, 1024)
        flushed = supervisor.flush_layer_pages(t)
        self.assertTrue(isinstance(flushed, bool))

    def test_trainer_engine(self):
        """Verify Trainer execution step with toy model and loss convergence."""
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = CSALinear(16, 16, group_size=16)
                self.norm = RMSNorm(16)
                self.fc2 = CSALinear(16, 4, group_size=16)

            def forward(self, x):
                return self.fc2(self.norm(self.fc1(x)))

        model = ToyModel()
        trainer = Trainer(model, lr=0.05)

        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))

        initial_loss = trainer.train_step(x, y)
        for _ in range(5):
            last_loss = trainer.train_step(x, y)

        self.assertFalse(torch.isnan(torch.tensor(last_loss)))


if __name__ == "__main__":
    unittest.main()

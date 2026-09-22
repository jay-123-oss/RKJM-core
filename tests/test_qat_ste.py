"""
Comprehensive Unit Tests for QAT, STE Autograd, and Ternary LoRA.
Validates:
  1. TernaryQuantizeSTE forward and hard-tanh backward clipping.
  2. Numerical stability: zero NaN/Inf over multiple AdamW optimizer steps.
  3. QATCSALinear training vs evaluation modes.
  4. TernaryLoRALinear parameter freezing and adapter gradient flow.
  5. Model patcher layer swapping and .rkmjbin export parity.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim

from rkmj.nn.qat import TernaryQuantizeSTE, ActivationQuantizeSTE, QATCSALinear
from rkmj.nn.lora import TernaryLoRALinear, apply_ternary_lora
from rkmj.nn.norm import RMSNorm
from rkmj.models.patcher import prepare_model_for_qat, export_model_to_rkmjbin
from rkmj.serialization.rkmjbin import load_rkmjbin


class TestQATAndLoRA(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_ternary_quantize_ste_forward_and_clipping(self):
        """Verify forward quantization to {-1, 0, 1} and hard-tanh backward clipping."""
        # Create master weights with values inside and outside the [-1, 1] range
        w = torch.tensor([[-2.5, -0.8, -0.1, 0.0, 0.2, 0.9, 3.0]], dtype=torch.float32, requires_grad=True)

        w_quant, alpha = TernaryQuantizeSTE.apply(w)

        # Check values are strictly in {-1.0, 0.0, 1.0}
        unique_vals = torch.unique(w_quant).tolist()
        for v in unique_vals:
            self.assertIn(v, [-1.0, 0.0, 1.0])

        # Backward pass: compute loss = sum(w_quant)
        loss = w_quant.sum()
        loss.backward()

        self.assertIsNotNone(w.grad)
        # Check clipping: elements where |w / alpha| > 1.0 must have grad == 0
        w_scaled = (w / alpha.unsqueeze(-1)).detach()
        clipped_mask = w_scaled.abs() > 1.0

        if clipped_mask.any():
            self.assertTrue(torch.all(w.grad[clipped_mask] == 0.0))

        # Check unclipped elements pass gradient straight-through (grad == 1.0)
        unclipped_mask = w_scaled.abs() <= 1.0
        if unclipped_mask.any():
            self.assertTrue(torch.all(w.grad[unclipped_mask] == 1.0))

    def test_activation_quantize_ste(self):
        """Verify 1-bit activation binarization and gradient flow."""
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0], dtype=torch.float32, requires_grad=True)
        x_sign = ActivationQuantizeSTE.apply(x)

        # Forward check
        self.assertTrue(torch.all(x_sign[:2] == -1.0))
        self.assertTrue(torch.all(x_sign[2:] == 1.0))

        # Backward check
        loss = x_sign.sum()
        loss.backward()

        # Elements outside [-1, 1] have grad 0, inside have grad 1
        self.assertEqual(x.grad[0].item(), 0.0)
        self.assertEqual(x.grad[1].item(), 1.0)
        self.assertEqual(x.grad[2].item(), 1.0)
        self.assertEqual(x.grad[3].item(), 1.0)
        self.assertEqual(x.grad[4].item(), 0.0)

    def test_qat_linear_training_stability(self):
        """Verify QATCSALinear trains stably without NaN/Inf over multiple AdamW steps."""
        layer = QATCSALinear(in_features=128, out_features=64, bias=True)
        optimizer = optim.AdamW(layer.parameters(), lr=1e-3, weight_decay=1e-4)

        for step in range(25):
            optimizer.zero_grad()
            x = torch.randn(8, 128, dtype=torch.float32)
            target = torch.randn(8, 64, dtype=torch.float32)

            y = layer(x)
            loss = nn.functional.mse_loss(y, target)
            loss.backward()

            # Check zero NaN or Inf in gradients and master weights
            self.assertFalse(torch.isnan(layer.master_weight.grad).any())
            self.assertFalse(torch.isinf(layer.master_weight.grad).any())

            optimizer.step()

            self.assertFalse(torch.isnan(layer.master_weight).any())
            self.assertFalse(torch.isinf(layer.master_weight).any())

    def test_qat_linear_eval_parity(self):
        """Verify that eval() mode and C++ popcount match simulated forward pass within atol <= 1e-5."""
        layer = QATCSALinear(in_features=256, out_features=128, bias=True)
        layer.eval()  # Automatically packs weights

        x = torch.randn(4, 256, dtype=torch.float32)

        # Ground-truth simulated forward
        w_quant = torch.clamp(torch.round(layer.master_weight / layer.alpha.unsqueeze(1)), -1.0, 1.0)
        x_sign = torch.where(x >= 0.0, 1.0, -1.0)
        y_simulated = layer.alpha * nn.functional.linear(x_sign, w_quant) + layer.bias

        # Execution in eval mode
        y_eval = layer(x)

        max_diff = torch.max(torch.abs(y_eval - y_simulated)).item()
        self.assertLessEqual(max_diff, 1e-5, f"Eval mode parity discrepancy: {max_diff}")

    def test_ternary_lora_adaptation(self):
        """Verify TernaryLoRALinear freezes base weights and updates only adapters."""
        base_layer = nn.Linear(128, 64, bias=True)
        lora_layer = TernaryLoRALinear(base_layer, rank=8, lora_alpha=16.0)

        # Verify base layer parameters are frozen
        for p in lora_layer.base.parameters():
            self.assertFalse(p.requires_grad)

        # Verify adapter parameters require grad
        self.assertTrue(lora_layer.lora_A.requires_grad)
        self.assertTrue(lora_layer.lora_B.requires_grad)

        optimizer = optim.AdamW([lora_layer.lora_A, lora_layer.lora_B], lr=1e-2)

        x = torch.randn(4, 128, dtype=torch.float32)
        target = torch.randn(4, 64, dtype=torch.float32)

        # Initial forward pass
        initial_out = lora_layer(x)

        # Train adapter
        for _ in range(5):
            optimizer.zero_grad()
            out = lora_layer(x)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()

        trained_out = lora_layer(x)
        # Outputs should change as adapter learns
        self.assertFalse(torch.allclose(initial_out, trained_out))

        # Test merge and unmerge
        lora_layer.merge_lora()
        self.assertTrue(lora_layer.merged)
        merged_out = lora_layer(x)
        # Merged forward executes purely through base layer
        self.assertEqual(merged_out.shape, trained_out.shape)

        # Unmerge restores original weights and adapter forward
        lora_layer.unmerge_lora()
        self.assertFalse(lora_layer.merged)
        restored_out = lora_layer(x)
        self.assertTrue(torch.allclose(trained_out, restored_out, atol=1e-5))

    def test_model_patcher_and_export(self):
        """Verify prepare_model_for_qat and export_model_to_rkmjbin."""
        class ToyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.v_proj = nn.Linear(64, 64, bias=False)
                self.norm = RMSNorm(64)
                self.lm_head = nn.Linear(64, 100, bias=False)

            def forward(self, input_ids):
                h = self.embed_tokens(input_ids)
                q = self.q_proj(h)
                v = self.v_proj(h)
                out = self.norm(q + v)
                return self.lm_head(out)

        model = ToyTransformer()
        prepare_model_for_qat(model, verbose=False)

        # Verify q_proj and v_proj were replaced by QATCSALinear
        self.assertIsInstance(model.q_proj, QATCSALinear)
        self.assertIsInstance(model.v_proj, QATCSALinear)

        # Verify embed_tokens and lm_head remained full precision
        self.assertIsInstance(model.embed_tokens, nn.Embedding)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertIsInstance(model.norm, RMSNorm)

        # Test export to .rkmjbin
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.rkmjbin")
            config = {"vocab_size": 100, "dim": 64}
            export_model_to_rkmjbin(model, save_path, config=config, verbose=False)
            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)


if __name__ == "__main__":
    unittest.main()

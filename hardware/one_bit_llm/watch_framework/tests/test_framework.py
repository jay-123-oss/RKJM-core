"""Official Verification Suite for watch_framework (Phase 3).

Tests:
1. Mathematical parity of dynamic alpha, sign quantization, and STE backward clipping.
2. Bitwise CSA GEMM OpenMP kernel vs reference ternary matrix operations.
3. WatchLinear and WatchConv2d execution and autograd gradient flow.
4. End-to-end 3-layer WatchMLP training with wf.Trainer asserting strict loss convergence.
5. Bit-packed .wfbin serialization verifying >20x file size reduction over .pt checkpoints.
6. Checkpoint reload in-place updating and 100% numerical parity assertion.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure watch_framework package directory is discoverable
here = os.path.abspath(os.path.dirname(__file__))
pkg_root = os.path.abspath(os.path.join(here, ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

import watch_framework as wf
from watch_framework import _C


class WatchMLP(nn.Module):
    """3-layer multi-layer perceptron built entirely with 1-bit WatchLinear layers."""

    def __init__(self, in_features: int = 64, hidden_features: int = 128, num_classes: int = 4):
        super().__init__()
        self.fc1 = wf.nn.WatchLinear(in_features, hidden_features, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = wf.nn.WatchLinear(hidden_features, hidden_features, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = wf.nn.WatchLinear(hidden_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class TestWatchFramework(unittest.TestCase):
    """Comprehensive test suite verifying Phase 3 specifications."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_01_dynamic_alpha_and_ste_sign(self):
        """Verify dynamic alpha calculation: alpha = (1 / N) * sum(|W_i|)."""
        w = torch.tensor([[-2.0, 1.0], [0.5, -1.5]], dtype=torch.float32)
        expected_alpha = (2.0 + 1.0 + 0.5 + 1.5) / 4.0
        computed_alpha = wf.nn.dynamic_alpha(w)
        self.assertAlmostEqual(computed_alpha.item(), expected_alpha, places=6)

        # Test STE sign function
        x = torch.tensor([-2.5, -0.1, 0.0, 0.5, 3.0], dtype=torch.float32, requires_grad=True)
        x_q = wf.nn.ste_sign(x)
        expected_signs = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(x_q, expected_signs))

        # Backward test: grad clipped by Indicator(|x| <= 1.0)
        loss = x_q.sum()
        loss.backward()
        # |x|: [2.5, 0.1, 0.0, 0.5, 3.0] -> indicator: [0, 1, 1, 1, 0]
        expected_grad = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        self.assertTrue(torch.equal(x.grad, expected_grad))

    def test_02_csa_gemm_kernel_correctness(self):
        """Verify C++ OpenMP CSA GEMM kernel produces exact dot products."""
        B, K, N = 8, 128, 16
        x_q = torch.where(torch.randn(B, K) >= 0, 1.0, -1.0)
        w_q = torch.where(torch.randn(N, K) >= 0, 1.0, -1.0)

        # Reference matrix multiplication
        y_ref = torch.matmul(x_q, w_q.t())

        # C++ OpenMP CSA bitwise kernel
        y_csa = _C.csa_gemm(x_q, w_q)

        self.assertTrue(torch.allclose(y_ref, y_csa, atol=1e-5))

    def test_03_watch_linear_forward_backward(self):
        """Verify WatchLinear layer execution and gradient flow."""
        layer = wf.nn.WatchLinear(in_features=32, out_features=16, bias=True)
        x = torch.randn(4, 32, requires_grad=True)

        y = layer(x)
        self.assertEqual(y.shape, (4, 16))

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        self.assertEqual(layer.weight.grad.shape, (16, 32))
        self.assertEqual(layer.bias.grad.shape, (16,))

    def test_04_watch_conv2d_im2col(self):
        """Verify WatchConv2d execution via unfold and 1-bit GEMM."""
        conv = wf.nn.WatchConv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True)
        x = torch.randn(2, 3, 16, 16, requires_grad=True)

        out = conv(x)
        self.assertEqual(out.shape, (2, 8, 16, 16))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNotNone(conv.bias.grad)
        self.assertEqual(conv.weight.grad.shape, (8, 3, 3, 3))

    def test_05_trainer_convergence_and_decreasing_loss(self):
        """Train WatchMLP on synthetic multi-class data and assert strictly decreasing loss across epochs."""
        num_samples = 100
        in_features = 32
        hidden_features = 128
        num_classes = 4

        # Synthetic multi-class data with controlled seed
        torch.manual_seed(1)
        X = torch.randn(num_samples, in_features)
        W_teacher = torch.randn(in_features, num_classes)
        Y = (X @ W_teacher).argmax(dim=-1)

        dataset = TensorDataset(X, Y)
        train_loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        model = WatchMLP(in_features=in_features, hidden_features=hidden_features, num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()

        trainer = wf.Trainer(model, optimizer, criterion)
        history = trainer.fit(train_loader, epochs=10, warmup_epochs=0, verbose=False)

        losses = history["train_loss"]
        self.assertEqual(len(losses), 10)

        # Assert loss strictly decreases over the 10 epochs
        is_strictly_decreasing = all(losses[i] < losses[i - 1] for i in range(1, len(losses)))
        self.assertTrue(
            is_strictly_decreasing,
            f"Loss did not strictly decrease across all epochs: {losses}",
        )
        self.assertLess(losses[-1], losses[0])

    def test_06_checkpoint_compression_and_parity(self):
        """Export model to .wfbin, verify >20x file size reduction, and assert 100% output parity."""
        in_features, hidden_features, num_classes = 128, 256, 8
        model = WatchMLP(in_features=in_features, hidden_features=hidden_features, num_classes=num_classes)

        with tempfile.TemporaryDirectory() as tmpdir:
            wfbin_path = os.path.join(tmpdir, "model_compressed.wfbin")
            pt_path = os.path.join(tmpdir, "model_fp32.pt")

            # 1. Save compressed .wfbin and standard PyTorch .pt
            wf.save_checkpoint(model, wfbin_path)
            torch.save(model.state_dict(), pt_path)

            wfbin_size = os.path.getsize(wfbin_path)
            pt_size = os.path.getsize(pt_path)
            compression_ratio = pt_size / wfbin_size

            print(
                f"\n[Compression Test] PT Size: {pt_size:,} bytes | "
                f"WFBIN Size: {wfbin_size:,} bytes | "
                f"Ratio: {compression_ratio:.2f}x"
            )

            # Assert file size reduction is >20x
            self.assertGreater(
                compression_ratio,
                20.0,
                f"Compression ratio {compression_ratio:.2f}x is not > 20x",
            )

            # 2. Test output before saving
            test_x = torch.randn(8, in_features)
            model.eval()
            with torch.no_grad():
                out_before = model(test_x)

            # 3. Reload checkpoint into a fresh model instance
            reloaded_model = WatchMLP(in_features=in_features, hidden_features=hidden_features, num_classes=num_classes)
            wf.load_checkpoint(reloaded_model, wfbin_path)
            reloaded_model.eval()

            with torch.no_grad():
                out_after = reloaded_model(test_x)

            # 4. Assert 100% output parity
            max_diff = (out_before - out_after).abs().max().item()
            print(f"[Parity Test] Max difference after reload: {max_diff:.2e}")
            self.assertTrue(
                torch.allclose(out_before, out_after, atol=1e-5),
                f"Reloaded model outputs diverged: max diff = {max_diff}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

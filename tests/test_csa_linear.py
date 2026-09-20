"""
Unit tests for rkmj.nn.CSALinear forward pass and inference packing.
"""

from __future__ import annotations

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.nn.linear import CSALinear


class TestCSALinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape_2d(self):
        batch, in_f, out_f = 4, 128, 64
        layer = CSALinear(in_f, out_f)
        x = torch.randn(batch, in_f)
        out = layer(x)
        self.assertEqual(out.shape, (batch, out_f))
        self.assertFalse(torch.isnan(out).any())

    def test_forward_shape_3d(self):
        batch, seq, in_f, out_f = 2, 32, 256, 128
        layer = CSALinear(in_f, out_f)
        x = torch.randn(batch, seq, in_f)
        out = layer(x)
        self.assertEqual(out.shape, (batch, seq, out_f))
        self.assertFalse(torch.isnan(out).any())

    def test_packed_inference_mode(self):
        in_f, out_f = 128, 64
        layer = CSALinear(in_f, out_f)
        x = torch.randn(2, in_f)

        # Dynamic pass
        out_dynamic = layer(x)

        # Pack weights
        layer.pack_weights_for_inference()
        layer.eval()
        self.assertTrue(layer.is_packed)

        # Packed popcount pass
        out_packed = layer(x)
        self.assertEqual(out_packed.shape, (2, out_f))
        # High numerical consistency
        diff = (out_dynamic - out_packed).abs().max().item()
        self.assertLess(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()

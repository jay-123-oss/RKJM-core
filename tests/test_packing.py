"""
Unit tests for 1.58-bit ternary bit-packing and unpacking fidelity.
"""

from __future__ import annotations

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.serialization.packer import (
    quantize_to_ternary,
    pack_ternary_weights,
    unpack_ternary_weights,
)


class TestBitPacking(unittest.TestCase):
    def test_roundtrip_multiples_of_16(self):
        N, K = 32, 64
        # Generate random ternary matrix in {-1.0, 0.0, 1.0}
        w_ternary = torch.randint(-1, 2, (N, K)).float()

        packed = pack_ternary_weights(w_ternary)
        self.assertEqual(packed.shape, (N, K // 16))

        unpacked = unpack_ternary_weights(packed, K)
        self.assertEqual(unpacked.shape, (N, K))
        self.assertTrue(torch.equal(w_ternary, unpacked))

    def test_roundtrip_non_multiple_of_16(self):
        N, K = 16, 53  # Non-multiple of 16
        w_ternary = torch.randint(-1, 2, (N, K)).float()

        packed = pack_ternary_weights(w_ternary)
        unpacked = unpack_ternary_weights(packed, K)
        self.assertTrue(torch.equal(w_ternary, unpacked))

    def test_quantization(self):
        w_fp32 = torch.randn(16, 32)
        w_ternary, alpha = quantize_to_ternary(w_fp32)

        unique_vals = set(w_ternary.unique().tolist())
        self.assertTrue(unique_vals.issubset({-1.0, 0.0, 1.0}))
        self.assertEqual(alpha.shape, (16,))


if __name__ == "__main__":
    unittest.main()

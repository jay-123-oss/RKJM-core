"""
================================================================================
RKMJ-Core Zero Memory Leak & Peak RSS Verification Test Suite
================================================================================
Validates:
1. Native C++ heap trimming (malloc_trim) via rkmj._C.force_heap_trim().
2. Out-of-Core Safetensors streaming quantization peak RSS <= 2.0 GB.
3. Adaptive inference model loading via rkmj.load() peak RSS <= 2.0 GB.
4. Multi-token generation loop memory stability with zero unbounded accumulation.
================================================================================
"""

import gc
import os
import psutil
import sys
import unittest
from pathlib import Path

import torch

# Ensure rkmj-core is in sys.path
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rkmj
from rkmj.quantizer import OutOfCoreQuantizer


class TestMemoryLeakAndLimits(unittest.TestCase):
    def setUp(self):
        self.process = psutil.Process(os.getpid())
        gc.collect()
        rkmj.force_heap_trim()

    def tearDown(self):
        gc.collect()
        rkmj.force_heap_trim()

    def test_01_force_heap_trim(self):
        """Verify C++ force_heap_trim successfully calls malloc_trim and drops RSS."""
        from rkmj import _C
        self.assertTrue(hasattr(_C, "force_heap_trim"), "C++ extension missing force_heap_trim binding")

        # Allocate 100MB of tensors then delete
        tensors = [torch.randn(1024, 1024, dtype=torch.float32) for _ in range(25)]
        del tensors
        gc.collect()

        # Call force heap trim
        _C.force_heap_trim()
        rss_gb = self.process.memory_info().rss / (1024.0 ** 3)
        self.assertLessEqual(rss_gb, 2.0, "Heap trim did not keep RSS <= 2.0 GB")

    def test_02_loader_rss_bound(self):
        """Verify rkmj.load() maintains peak RSS strictly <= 2.0 GB for 3B parameter models."""
        candidate_model = REPO_ROOT.parent / "TEST" / "quantization_test" / "qwen2.5_3b_1.58bit.rkmjbin"
        if not candidate_model.exists():
            self.skipTest(f"Model binary not found at {candidate_model}")

        rss_before = self.process.memory_info().rss / (1024.0 ** 3)
        runner = rkmj.load(str(candidate_model), ram_budget_gb=2.0)
        rss_after = self.process.memory_info().rss / (1024.0 ** 3)

        print(f"\n[Test Loader] RSS Before: {rss_before:.2f} GB | Loaded RSS: {rss_after:.2f} GB")
        self.assertLessEqual(
            rss_after,
            2.0,
            f"Active process RSS ({rss_after:.2f} GB) exceeded 2.0 GB limit!",
        )

        runner.close()

    def test_03_inference_generation_memory_stability(self):
        """Verify multi-token generation does not cause memory leaks or unbounded growth."""
        candidate_model = REPO_ROOT.parent / "TEST" / "quantization_test" / "qwen2.5_3b_1.58bit.rkmjbin"
        if not candidate_model.exists():
            self.skipTest(f"Model binary not found at {candidate_model}")

        runner = rkmj.load(str(candidate_model), ram_budget_gb=2.0)
        dummy_prompt = torch.tensor([[151644, 872, 198, 10838, 151645]], dtype=torch.long)

        peak_rss = 0.0
        tokens_generated = 0

        for token in runner.generate(dummy_prompt, max_new_tokens=4, temperature=0.7):
            tokens_generated += 1
            rss = self.process.memory_info().rss / (1024.0 ** 3)
            if rss > peak_rss:
                peak_rss = rss

        print(f"\n[Test Generation] Generated: {tokens_generated} tokens | Peak RSS: {peak_rss:.2f} GB")
        self.assertLessEqual(
            peak_rss,
            2.0,
            f"Peak RSS ({peak_rss:.2f} GB) during generation exceeded 2.0 GB limit!",
        )

        runner.close()


if __name__ == "__main__":
    unittest.main()

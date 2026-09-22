"""
Unit tests for Hardware & Model-Scale Adaptive Dynamic Architecture in RKMJ-Core.
Tests:
  - SystemHardwareProfiler & SIMD / Storage detection
  - ModelFootprintEstimator across 1B, 8B, 70B, 256B scales
  - DynamicExecutionRouter 4-tier decision engine
  - MemoryGovernor and LayerWiseStreamer
  - C++ AsyncRingStreamer double-buffered prefetcher
  - Unified AutoModel interface across execution tiers
"""

import os
import tempfile
import unittest
import torch

from rkmj.models.config import RKMJConfig
from rkmj.models.llama import RKMJLlamaForCausalLM
from rkmj.runtime.profiler import (
    ExecutionTier,
    SystemHardwareProfiler,
    ModelFootprintEstimator,
    DynamicExecutionRouter,
)
from rkmj.runtime.streamer import MemoryGovernor, LayerWiseStreamer
from rkmj.serialization.rkmjbin import save_rkmjbin
from rkmj.auto import AutoModel, StreamedTransformerModel

try:
    from rkmj._C import AsyncRingStreamer
except ImportError:
    AsyncRingStreamer = None


class TestRuntimeAdaptiveSuite(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config = RKMJConfig(
            vocab_size=500,
            dim=64,
            n_layers=3,
            n_heads=2,
            max_seq_len=256,
        )
        self.model = RKMJLlamaForCausalLM(self.config)
        self.model.pack_weights_for_inference()
        self.model.eval()

        self.bin_path = os.path.join(self.tmpdir.name, "model.rkmjbin")
        save_rkmjbin(self.model, self.bin_path, config=self.config.to_dict(), verbose=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_hardware_profiler(self):
        """Test host RAM, SIMD, and storage media profiling."""
        profiler = SystemHardwareProfiler()
        total_ram = profiler.get_total_ram_gb()
        safe_ram = profiler.get_available_ram_gb()

        self.assertGreater(total_ram, 0.5)
        self.assertGreater(safe_ram, 0.2)
        self.assertLessEqual(safe_ram, total_ram)

        cpu_flags = profiler.detect_cpu_features()
        self.assertIsInstance(cpu_flags, dict)
        self.assertIn("avx2", cpu_flags)
        self.assertIn("neon", cpu_flags)

        storage = profiler.detect_storage_profile(self.tmpdir.name)
        self.assertIn(storage["type"], ["NVME", "SATA_SSD", "HDD"])
        self.assertGreater(storage["read_speed_mb_s"], 0)

    def test_model_footprint_estimator(self):
        """Estimate model footprints across 1B, 8B, 70B, and 256B scales."""
        # 1B scale config
        cfg_1b = {"num_hidden_layers": 24, "hidden_size": 2048, "intermediate_size": 5504, "vocab_size": 32000}
        fp_1b = ModelFootprintEstimator.estimate(cfg_1b)
        self.assertGreater(fp_1b["total_params"], 0.8e9)
        self.assertLess(fp_1b["total_params"], 1.5e9)
        # 1B in 1.58-bit packed is ~300 MB
        self.assertLess(fp_1b["total_bytes_1_58bit"], 500 * 1024 * 1024)

        # 70B scale config
        cfg_70b = {"num_hidden_layers": 80, "hidden_size": 8192, "intermediate_size": 28672, "vocab_size": 128000}
        fp_70b = ModelFootprintEstimator.estimate(cfg_70b)
        self.assertGreater(fp_70b["total_params"], 60e9)
        # 70B in 1.58-bit packed is ~20 GB
        self.assertLess(fp_70b["total_bytes_1_58bit"], 25 * 1024 * 1024 * 1024)

        # Real .rkmjbin file footprint
        fp_bin = ModelFootprintEstimator.estimate(self.bin_path)
        self.assertEqual(fp_bin["num_layers"], 3)
        self.assertGreater(fp_bin["total_bytes_1_58bit"], 0)

    def test_dynamic_execution_router(self):
        """Validate dynamic execution tier decisions under simulated RAM environments."""
        # 1. Small model (< 40% RAM) -> IN_RAM
        cfg_small = {"num_hidden_layers": 12, "hidden_size": 1024, "intermediate_size": 2816, "vocab_size": 32000}
        tier, diag = DynamicExecutionRouter.resolve(cfg_small, safety_ram_gb=16.0)
        self.assertEqual(tier, ExecutionTier.IN_RAM)

        # 2. Medium model (40% - 90% RAM) -> BALANCED_MMAP
        cfg_med = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 11008, "vocab_size": 32000}
        # 8B in 1.58-bit is ~2.3 GB. On 4.0 GB safe RAM, 2.3 GB is > 40% and <= 90%
        tier, diag = DynamicExecutionRouter.resolve(cfg_med, safety_ram_gb=4.0)
        self.assertEqual(tier, ExecutionTier.BALANCED_MMAP)

        # 3. Large 70B model with 4GB RAM -> LAYER_STREAM
        cfg_70b = {"num_hidden_layers": 80, "hidden_size": 8192, "intermediate_size": 28672, "vocab_size": 32000}
        tier, diag = DynamicExecutionRouter.resolve(cfg_70b, safety_ram_gb=4.0)
        self.assertEqual(tier, ExecutionTier.LAYER_STREAM)

        # 4. Extreme 120B+ model -> DOUBLE_BUFFERED_RING
        cfg_120b = {"num_hidden_layers": 120, "hidden_size": 10240, "intermediate_size": 32768, "vocab_size": 64000}
        tier, diag = DynamicExecutionRouter.resolve(cfg_120b, safety_ram_gb=32.0)
        self.assertEqual(tier, ExecutionTier.DOUBLE_BUFFERED_RING)

        # 5. User override
        tier, _ = DynamicExecutionRouter.resolve(cfg_small, override_tier="LAYER_STREAM")
        self.assertEqual(tier, ExecutionTier.LAYER_STREAM)

    def test_memory_governor_and_streamer(self):
        """Test MemoryGovernor and layer-by-layer sequential streaming from .rkmjbin."""
        governor = MemoryGovernor(max_ram_bytes=int(3.5 * 1024 * 1024 * 1024))
        rss_bytes = governor.get_current_rss_bytes()
        self.assertGreater(rss_bytes, 0)

        # Test trimming
        trimmed = governor.check_and_trim(force=True)
        self.assertTrue(trimmed)

        # Stream layers
        streamer = LayerWiseStreamer(self.bin_path, memory_governor=governor)
        layers_streamed = 0
        for layer_idx, weights in streamer.stream_layers():
            self.assertEqual(layer_idx, layers_streamed)
            self.assertIsInstance(weights, dict)
            self.assertTrue(len(weights) > 0)
            layers_streamed += 1

        self.assertEqual(layers_streamed, 3)

    def test_cpp_async_ring_streamer(self):
        """Test C++ AsyncRingStreamer prefetching and buffer swapping."""
        if AsyncRingStreamer is None:
            self.skipTest("C++ AsyncRingStreamer extension not compiled")

        # Create dummy file with 3 frames
        frame_size = 1024
        test_file = os.path.join(self.tmpdir.name, "stream_test.bin")
        with open(test_file, "wb") as f:
            for i in range(3):
                f.write(bytes([i + 1] * frame_size))

        offsets = [0, frame_size, frame_size * 2]
        sizes = [frame_size, frame_size, frame_size]

        streamer = AsyncRingStreamer(test_file, offsets, sizes)
        self.assertEqual(streamer.get_num_layers(), 3)
        self.assertGreaterEqual(streamer.get_buffer_size_bytes(), frame_size)

        streamer.start(0)

        # Acquire Layer 0
        buf0 = streamer.acquire_compute_layer(0)
        self.assertEqual(buf0.numel(), frame_size)
        self.assertEqual(buf0[0].item(), 1)

        # Signal compute done, prefetch next
        streamer.release_and_prefetch_next()

        # Acquire Layer 1
        buf1 = streamer.acquire_compute_layer(1)
        self.assertEqual(buf1.numel(), frame_size)
        self.assertEqual(buf1[0].item(), 2)

        streamer.stop()

    def test_automodel_from_pretrained(self):
        """Test AutoModel.from_pretrained loading across different execution tiers."""
        # 1. Auto mode (for small test model, resolves to IN_RAM)
        model_auto = AutoModel.from_pretrained(self.bin_path, mode="auto")
        self.assertIsNotNone(model_auto)
        self.assertEqual(model_auto.execution_tier, ExecutionTier.IN_RAM)

        # Forward pass
        input_ids = torch.randint(0, 500, (1, 8))
        logits, _ = model_auto(input_ids)
        self.assertEqual(logits.shape, (1, 8, 500))

        # 2. Explicit Tier 3: LAYER_STREAM
        model_stream = AutoModel.from_pretrained(self.bin_path, mode="layer_stream")
        self.assertIsInstance(model_stream, StreamedTransformerModel)
        self.assertEqual(model_stream.execution_tier, ExecutionTier.LAYER_STREAM)

        logits_stream, _ = model_stream(input_ids)
        self.assertEqual(logits_stream.shape, (1, 8, 500))

        # Generation with Streamed model
        out_ids = model_stream.generate(input_ids, max_new_tokens=3, temperature=0.0)
        self.assertEqual(out_ids.shape, (1, 11))

        # 3. Explicit Tier 2: BALANCED_MMAP
        model_mmap = AutoModel.from_pretrained(self.bin_path, mode="balanced_mmap")
        self.assertEqual(model_mmap.execution_tier, ExecutionTier.BALANCED_MMAP)
        logits_mmap, _ = model_mmap(input_ids)
        self.assertEqual(logits_mmap.shape, (1, 8, 500))


if __name__ == "__main__":
    unittest.main()

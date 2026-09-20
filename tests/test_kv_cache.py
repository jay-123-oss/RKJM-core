"""
Unit tests for RKMJ contiguous KVCache engine, attention integration, and two-phase generation.
"""

from __future__ import annotations

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.engine.cache import KVCache
from rkmj.engine.generator import RKMJGenerator, GenerationMetrics
from rkmj.nn.attention import CSASelfAttention
from rkmj.models.config import RKMJConfig
from rkmj.models.llama import RKMJLlamaForCausalLM


class TestKVCache(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_cache_allocation_and_shape(self):
        n_layers = 4
        max_batch = 2
        n_kv_heads = 4
        max_seq = 128
        head_dim = 32

        cache = KVCache(
            n_layers=n_layers,
            max_batch_size=max_batch,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq,
            head_dim=head_dim,
        )

        expected_shape = (n_layers, max_batch, n_kv_heads, max_seq, head_dim)
        self.assertEqual(cache.k.shape, expected_shape)
        self.assertEqual(cache.v.shape, expected_shape)
        self.assertEqual(cache.seen_tokens, 0)
        self.assertEqual(cache.current_pos, 0)

    def test_inplace_update_and_zero_realloc(self):
        cache = KVCache(n_layers=2, max_batch_size=1, n_kv_heads=2, max_seq_len=64, head_dim=16)

        # Record original underlying data pointers to verify zero re-allocation
        orig_k_ptr = cache.k.data_ptr()
        orig_v_ptr = cache.v.data_ptr()

        # Phase 1: Prefill 4 tokens
        k_prefill = torch.randn(1, 2, 4, 16)
        v_prefill = torch.randn(1, 2, 4, 16)

        k_view, v_view = cache.update(layer_idx=0, k_state=k_prefill, v_state=v_prefill)
        self.assertEqual(k_view.shape, (1, 2, 4, 16))
        self.assertEqual(v_view.shape, (1, 2, 4, 16))
        self.assertTrue(torch.allclose(k_view, k_prefill))
        self.assertTrue(torch.allclose(v_view, v_prefill))
        self.assertEqual(cache.k.data_ptr(), orig_k_ptr)
        self.assertEqual(cache.v.data_ptr(), orig_v_ptr)

        cache.increment_seen(4)
        self.assertEqual(cache.seen_tokens, 4)

        # Phase 2: Decode 1 token
        k_dec = torch.randn(1, 2, 1, 16)
        v_dec = torch.randn(1, 2, 1, 16)

        k_view, v_view = cache.update(layer_idx=0, k_state=k_dec, v_state=v_dec)
        self.assertEqual(k_view.shape, (1, 2, 5, 16))
        self.assertEqual(v_view.shape, (1, 2, 5, 16))
        self.assertTrue(torch.allclose(k_view[:, :, -1:, :], k_dec))
        self.assertTrue(torch.allclose(v_view[:, :, -1:, :], v_dec))
        # Ensure memory address remains identical (strict zero-allocation)
        self.assertEqual(cache.k.data_ptr(), orig_k_ptr)
        self.assertEqual(cache.v.data_ptr(), orig_v_ptr)

        # Reset
        cache.reset()
        self.assertEqual(cache.seen_tokens, 0)
        self.assertTrue(torch.all(cache.k == 0))
        self.assertTrue(torch.all(cache.v == 0))

    def test_gqa_attention_with_cache(self):
        # 8 query heads, 2 KV heads -> num_kv_groups = 4
        dim = 128
        num_heads = 8
        num_kv_heads = 2
        head_dim = dim // num_heads  # 16
        attn = CSASelfAttention(dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads)

        cache = KVCache(n_layers=1, max_batch_size=1, n_kv_heads=num_kv_heads, max_seq_len=32, head_dim=head_dim)

        # Step 1: Prefill 3 tokens
        x_prompt = torch.randn(1, 3, dim)
        out_prefill = attn(x_prompt, layer_idx=0, kv_cache=cache, start_pos=0)
        self.assertEqual(out_prefill.shape, (1, 3, dim))
        cache.increment_seen(3)

        # Step 2: Decode 1 token
        x_token = torch.randn(1, 1, dim)
        out_dec = attn(x_token, layer_idx=0, kv_cache=cache, start_pos=cache.seen_tokens)
        self.assertEqual(out_dec.shape, (1, 1, dim))
        cache.increment_seen(1)
        self.assertEqual(cache.seen_tokens, 4)

    def test_logits_equivalence_cached_vs_noncached(self):
        """Verify that cached autoregressive step produces identical logits to full recomputation."""
        config = RKMJConfig(
            vocab_size=100,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_dim=128,
            max_seq_len=64,
        )
        model = RKMJLlamaForCausalLM(config)
        model.eval()

        prompt_ids = torch.tensor([[1, 5, 12, 34, 78]])
        T = prompt_ids.shape[1]

        # 1. Non-cached full forward
        with torch.no_grad():
            logits_full, _ = model(prompt_ids)

        # 2. Cached prefill forward
        cache = KVCache(
            n_layers=config.n_layers,
            max_batch_size=1,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=64,
            head_dim=config.dim // config.n_heads,
        )
        with torch.no_grad():
            logits_cached_prefill, _ = model(prompt_ids, kv_cache=cache, start_pos=0)

        # Logits at prefill should be numerically identical
        diff_prefill = (logits_full - logits_cached_prefill).abs().max().item()
        self.assertLess(diff_prefill, 1e-4)
        cache.increment_seen(T)

        # 3. Next token decode step
        next_token = torch.tensor([[42]])
        seq_concat = torch.cat([prompt_ids, next_token], dim=1)

        # Full recomputation logits
        with torch.no_grad():
            logits_full_next, _ = model(seq_concat)
            full_next_step_logits = logits_full_next[:, -1, :]

        # Cached single-token step
        with torch.no_grad():
            logits_cached_next, _ = model(next_token, kv_cache=cache, start_pos=cache.seen_tokens)
            cached_next_step_logits = logits_cached_next[:, -1, :]

        diff_decode = (full_next_step_logits - cached_next_step_logits).abs().max().item()
        self.assertLess(diff_decode, 1e-4)

    def test_rkmj_generator_two_phase_and_metrics(self):
        config = RKMJConfig(
            vocab_size=50,
            dim=32,
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            intermediate_dim=64,
            max_seq_len=64,
        )
        model = RKMJLlamaForCausalLM(config)

        def encode(text: str):
            return [ord(c) % 50 for c in text]

        def decode(ids):
            return "".join([chr((i % 26) + 65) for i in ids])

        generator = RKMJGenerator(model, encode, decode)
        prompt = "Hello"
        out_text, metrics = generator.generate(
            prompt,
            max_new_tokens=5,
            temperature=0.0,
            return_metrics=True,
        )

        self.assertIsInstance(out_text, str)
        self.assertTrue(out_text.startswith(prompt))
        self.assertIsInstance(metrics, GenerationMetrics)
        self.assertEqual(metrics.total_tokens, 5)
        self.assertEqual(metrics.decode_tokens, 4)
        self.assertGreater(metrics.decode_tokens_per_sec, 0.0)


if __name__ == "__main__":
    unittest.main()

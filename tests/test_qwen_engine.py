"""
Unit tests for Qwen Out-of-Core Streaming PTQ Converter & Local 4GB Runner (in qween/).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Bootstrap sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
_rkmj_core = Path(__file__).resolve().parent.parent
for p in [str(_repo_root), str(_rkmj_core)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from safetensors.torch import save_file

from qween.rope import QwenRotaryEmbedding, apply_rotary_pos_emb
from qween.converter import QwenStreamingPTQConverter
from qween.runner import QwenLocalRunner


class TestQwenEngineSuite(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = os.path.join(self.tmpdir.name, "synthetic_qwen")
        os.makedirs(self.model_dir, exist_ok=True)

        self.hidden_size = 128
        self.num_heads = 4
        self.num_kv_heads = 2
        self.intermediate_size = 256
        self.vocab_size = 1000
        self.num_layers = 3
        self.head_dim = self.hidden_size // self.num_heads

        # 1. Write config.json
        self.config = {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_layers,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "rope_theta": 1000000.0,
        }
        with open(os.path.join(self.model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        # 2. Create synthetic Safetensors shards
        shard1_dict = {
            "model.embed_tokens.weight": torch.randn(self.vocab_size, self.hidden_size),
        }
        # Layer 0 & Layer 1 in Shard 1
        for l in range(2):
            p = f"model.layers.{l}."
            shard1_dict[p + "input_layernorm.weight"] = torch.ones(self.hidden_size)
            shard1_dict[p + "post_attention_layernorm.weight"] = torch.ones(self.hidden_size)
            shard1_dict[p + "self_attn.q_proj.weight"] = torch.randn(self.hidden_size, self.hidden_size)
            shard1_dict[p + "self_attn.k_proj.weight"] = torch.randn(self.num_kv_heads * self.head_dim, self.hidden_size)
            shard1_dict[p + "self_attn.v_proj.weight"] = torch.randn(self.num_kv_heads * self.head_dim, self.hidden_size)
            shard1_dict[p + "self_attn.o_proj.weight"] = torch.randn(self.hidden_size, self.hidden_size)
            shard1_dict[p + "mlp.gate_proj.weight"] = torch.randn(self.intermediate_size, self.hidden_size)
            shard1_dict[p + "mlp.up_proj.weight"] = torch.randn(self.intermediate_size, self.hidden_size)
            shard1_dict[p + "mlp.down_proj.weight"] = torch.randn(self.hidden_size, self.intermediate_size)

        shard1_path = os.path.join(self.model_dir, "model-00001-of-00002.safetensors")
        save_file(shard1_dict, shard1_path)

        # Layer 2, norm, lm_head in Shard 2
        p2 = "model.layers.2."
        shard2_dict = {
            p2 + "input_layernorm.weight": torch.ones(self.hidden_size),
            p2 + "post_attention_layernorm.weight": torch.ones(self.hidden_size),
            p2 + "self_attn.q_proj.weight": torch.randn(self.hidden_size, self.hidden_size),
            p2 + "self_attn.k_proj.weight": torch.randn(self.num_kv_heads * self.head_dim, self.hidden_size),
            p2 + "self_attn.v_proj.weight": torch.randn(self.num_kv_heads * self.head_dim, self.hidden_size),
            p2 + "self_attn.o_proj.weight": torch.randn(self.hidden_size, self.hidden_size),
            p2 + "mlp.gate_proj.weight": torch.randn(self.intermediate_size, self.hidden_size),
            p2 + "mlp.up_proj.weight": torch.randn(self.intermediate_size, self.hidden_size),
            p2 + "mlp.down_proj.weight": torch.randn(self.hidden_size, self.intermediate_size),
            "model.norm.weight": torch.ones(self.hidden_size),
            "lm_head.weight": torch.randn(self.vocab_size, self.hidden_size),
        }
        shard2_path = os.path.join(self.model_dir, "model-00002-of-00002.safetensors")
        save_file(shard2_dict, shard2_path)

        # 3. Create model.safetensors.index.json
        weight_map = {}
        for k in shard1_dict:
            weight_map[k] = "model-00001-of-00002.safetensors"
        for k in shard2_dict:
            weight_map[k] = "model-00002-of-00002.safetensors"

        index_data = {
            "metadata": {"total_size": 1000000},
            "weight_map": weight_map,
        }
        with open(os.path.join(self.model_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_qwen_rope(self):
        """Test Rotary Position Embedding forward pass and rotation."""
        rotary = QwenRotaryEmbedding(dim=32, max_position_embeddings=512)
        seq_len = 16
        cos, sin = rotary(torch.zeros(1, 16), seq_len)
        self.assertEqual(cos.shape, (1, 1, 16, 32))
        self.assertEqual(sin.shape, (1, 1, 16, 32))

        # Test rotation application
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 2, 16, 32)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertFalse(torch.isnan(q_rot).any())
        self.assertFalse(torch.isnan(k_rot).any())

    def test_qwen_streaming_ptq_conversion(self):
        """Test layer-by-layer out-of-core conversion from Safetensors to .rkmjbin."""
        out_bin = os.path.join(self.tmpdir.name, "qwen_ternary.rkmjbin")
        converter = QwenStreamingPTQConverter(
            model_dir=self.model_dir,
            output_path=out_bin,
            max_ram_bytes=int(3.5 * 1024 * 1024 * 1024),
        )

        stats = converter.convert()
        self.assertTrue(os.path.exists(out_bin))
        self.assertGreater(stats["final_file_size_gb"], 0)
        self.assertGreaterEqual(stats["compression_ratio"], 1.0)
        self.assertLessEqual(stats["peak_rss_gb"], 3.5)

        # Inspect generated binary file
        with open(out_bin, "rb") as f:
            magic = f.read(8)
            self.assertEqual(magic, b"RKMJBIN1")

    def test_qwen_local_runner_forward_and_generation(self):
        """Test local inference and autoregressive generation with QwenLocalRunner."""
        out_bin = os.path.join(self.tmpdir.name, "qwen_ternary.rkmjbin")
        converter = QwenStreamingPTQConverter(
            model_dir=self.model_dir,
            output_path=out_bin,
        )
        converter.convert()

        runner = QwenLocalRunner(out_bin)
        self.assertEqual(runner.num_layers, 3)
        self.assertEqual(runner.dim, self.hidden_size)
        self.assertEqual(runner.vocab_size, self.vocab_size)

        # Test forward pass
        input_ids = torch.tensor([[12, 45, 67, 89]])
        logits = runner.forward(input_ids)
        self.assertEqual(logits.shape, (1, 4, self.vocab_size))
        self.assertFalse(torch.isnan(logits).any())

        # Test autoregressive generation
        generated = list(runner.generate(input_ids, max_new_tokens=4, temperature=0.0))
        self.assertEqual(len(generated), 4)
        for tok in generated:
            self.assertIsInstance(tok, int)
            self.assertTrue(0 <= tok < self.vocab_size)

        runner.close()


if __name__ == "__main__":
    unittest.main()

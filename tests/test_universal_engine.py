"""
Unit tests for Universal Multi-Model Engine:
  1. Architecture Auto-Detection (Qwen, LLaMA, Mistral, Gemma)
  2. Out-of-Core Streaming PTQ Conversion
  3. UniversalLocalRunner forward evaluation & autoregressive token generation
  4. Backward-compatibility shim verification for qween imports
"""

import json
import os
import shutil
import tempfile
import unittest
import torch
from safetensors.torch import save_file

from universal.config import detect_architecture_profile
from universal.converter import UniversalStreamingPTQConverter
from universal.runner import UniversalLocalRunner


class TestUniversalEngineSuite(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_size = 500
        self.hidden_size = 64
        self.num_heads = 4
        self.num_kv_heads = 2
        self.num_layers = 2
        self.intermediate_size = 128

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_architecture_profile_detection(self):
        """Verify automatic profile detection across different model families."""
        # Qwen
        qwen_cfg = {"model_type": "qwen2", "vocab_size": 151936}
        prof = detect_architecture_profile(qwen_cfg)
        self.assertEqual(prof.model_family, "qwen")
        self.assertTrue(prof.qkv_has_bias)
        self.assertEqual(prof.default_rope_theta, 1000000.0)

        # LLaMA 3
        llama3_cfg = {"model_type": "llama", "vocab_size": 128256, "rope_theta": 500000.0}
        prof = detect_architecture_profile(llama3_cfg)
        self.assertEqual(prof.model_family, "llama")
        self.assertFalse(prof.qkv_has_bias)
        self.assertEqual(prof.default_rope_theta, 500000.0)

        # Mistral
        mistral_cfg = {"model_type": "mistral", "vocab_size": 32000}
        prof = detect_architecture_profile(mistral_cfg)
        self.assertEqual(prof.model_family, "mistral")

        # Gemma
        gemma_cfg = {"model_type": "gemma", "hidden_size": 2048}
        prof = detect_architecture_profile(gemma_cfg)
        self.assertEqual(prof.model_family, "gemma")
        self.assertTrue(prof.norm_add_unit)

    def _create_synthetic_model(self, model_type: str = "llama") -> str:
        model_dir = os.path.join(self.temp_dir, f"synthetic_{model_type}")
        os.makedirs(model_dir, exist_ok=True)

        config = {
            "model_type": model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0 if model_type != "qwen2" else 1000000.0,
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create synthetic weights
        tensors = {
            "model.embed_tokens.weight": torch.randn(self.vocab_size, self.hidden_size),
            "model.norm.weight": torch.ones(self.hidden_size),
            "lm_head.weight": torch.randn(self.vocab_size, self.hidden_size),
        }

        has_bias = (model_type == "qwen2")
        head_dim = self.hidden_size // self.num_heads

        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            tensors[f"{p}input_layernorm.weight"] = torch.ones(self.hidden_size)
            tensors[f"{p}post_attention_layernorm.weight"] = torch.ones(self.hidden_size)

            tensors[f"{p}self_attn.q_proj.weight"] = torch.randn(self.hidden_size, self.hidden_size)
            tensors[f"{p}self_attn.k_proj.weight"] = torch.randn(self.num_kv_heads * head_dim, self.hidden_size)
            tensors[f"{p}self_attn.v_proj.weight"] = torch.randn(self.num_kv_heads * head_dim, self.hidden_size)
            tensors[f"{p}self_attn.o_proj.weight"] = torch.randn(self.hidden_size, self.hidden_size)

            if has_bias:
                tensors[f"{p}self_attn.q_proj.bias"] = torch.zeros(self.hidden_size)
                tensors[f"{p}self_attn.k_proj.bias"] = torch.zeros(self.num_kv_heads * head_dim)
                tensors[f"{p}self_attn.v_proj.bias"] = torch.zeros(self.num_kv_heads * head_dim)

            tensors[f"{p}mlp.gate_proj.weight"] = torch.randn(self.intermediate_size, self.hidden_size)
            tensors[f"{p}mlp.up_proj.weight"] = torch.randn(self.intermediate_size, self.hidden_size)
            tensors[f"{p}mlp.down_proj.weight"] = torch.randn(self.hidden_size, self.intermediate_size)

        save_file(tensors, os.path.join(model_dir, "model.safetensors"))
        return model_dir

    def test_universal_conversion_and_inference_llama(self):
        """Verify conversion and local execution for LLaMA-style models."""
        model_dir = self._create_synthetic_model("llama")
        out_bin = os.path.join(self.temp_dir, "llama_ternary.rkmjbin")

        converter = UniversalStreamingPTQConverter(model_dir, out_bin)
        converter.convert()
        self.assertTrue(os.path.exists(out_bin))
        self.assertGreater(os.path.getsize(out_bin), 0)

        runner = UniversalLocalRunner(out_bin)
        self.assertEqual(runner.profile.model_family, "llama")
        self.assertEqual(runner.num_layers, self.num_layers)

        # Forward pass
        input_ids = torch.tensor([[10, 20, 30]])
        logits = runner.forward(input_ids)
        self.assertEqual(logits.shape, (1, 3, self.vocab_size))

        # Generation
        tokens = list(runner.generate(input_ids, max_new_tokens=3, temperature=0.0))
        self.assertEqual(len(tokens), 3)
        runner.close()

    def test_backward_compatibility_shim(self):
        """Verify that importing from qween still works transparently."""
        from qween import QwenStreamingPTQConverter, QwenLocalRunner
        model_dir = self._create_synthetic_model("qwen2")
        out_bin = os.path.join(self.temp_dir, "qwen_shim.rkmjbin")

        converter = QwenStreamingPTQConverter(model_dir, out_bin)
        converter.convert()
        self.assertTrue(os.path.exists(out_bin))

        runner = QwenLocalRunner(out_bin)
        self.assertEqual(runner.profile.model_family, "qwen")
        input_ids = torch.tensor([[5, 15]])
        logits = runner.forward(input_ids)
        self.assertEqual(logits.shape, (1, 2, self.vocab_size))
        runner.close()


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for RKMJ Multimodal Suite (Ternary Whisper and Ternary LLaVA).
"""

import unittest
import torch

from rkmj.models.config import RKMJConfig
from rkmj.multimodal.whisper import (
    LogMelSpectrogram,
    TernaryWhisper,
    TernaryWhisperEncoder,
    TernaryWhisperDecoder,
)
from rkmj.multimodal.llava import VisionProjector, TernaryLLaVA


class TestMultimodalSuite(unittest.TestCase):

    def test_log_mel_spectrogram_shape(self):
        """Test audio waveform conversion into Log-Mel spectrogram."""
        extractor = LogMelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
        # 1 second of audio at 16kHz
        audio = torch.randn(2, 16000)
        mel = extractor(audio)
        # 16000 / 160 = 100 frames (+1 boundary)
        self.assertEqual(mel.shape[0], 2)
        self.assertEqual(mel.shape[1], 80)
        self.assertTrue(mel.shape[2] >= 100)
        self.assertFalse(torch.isnan(mel).any())

    def test_ternary_whisper_forward_and_packing(self):
        """Test TernaryWhisper forward pass with packed weights and generation."""
        model = TernaryWhisper(
            n_mels=80,
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            d_ff=128,
            encoder_layers=2,
            decoder_layers=2,
        )
        audio = torch.randn(2, 16000)
        decoder_ids = torch.randint(0, 1000, (2, 8))

        # Forward pass in training/STE mode
        logits = model(audio, decoder_ids)
        self.assertEqual(logits.shape, (2, 8, 1000))
        self.assertFalse(torch.isnan(logits).any())

        # Test weight packing
        model.pack_weights_for_inference()
        model.eval()

        # Forward pass in packed inference mode
        logits_packed = model(audio, decoder_ids)
        self.assertEqual(logits_packed.shape, (2, 8, 1000))

        # Test autoregressive generation
        out_ids = model.generate(audio, max_new_tokens=4, start_token_id=1, eos_token_id=999)
        self.assertEqual(out_ids.shape[0], 2)
        self.assertEqual(out_ids.shape[1], 5)

    def test_vision_projector_and_llava(self):
        """Test Ternary LLaVA projector and multimodal sequence forward pass."""
        config = RKMJConfig(
            vocab_size=1000,
            dim=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=256,
        )
        model = TernaryLLaVA(config=config, vision_dim=128, image_token_id=-200)

        # 4 visual tokens per sequence, batch size 2
        vision_embeds = torch.randn(2, 4, 128)
        # Text prompt of 6 tokens with image placeholder at pos 1: [10, -200, 20, 30, 40, 50]
        input_ids = torch.tensor([
            [10, -200, 20, 30, 40, 50],
            [10, -200, 25, 35, 45, 55],
        ])

        # Test embedding merge
        merged = model.merge_input_embeddings(input_ids, vision_embeds)
        # 5 text tokens + 4 vision tokens = 9 total tokens
        self.assertEqual(merged.shape, (2, 9, 64))

        # Forward pass
        logits, _ = model(input_ids=input_ids, vision_embeds=vision_embeds)
        self.assertEqual(logits.shape, (2, 9, 1000))

        # Pack weights
        model.pack_weights_for_inference()
        model.eval()

        # Generation
        out_tokens = model.generate(
            input_ids=input_ids,
            vision_embeds=vision_embeds,
            max_new_tokens=3,
            temperature=0.0,
        )
        # Initial 6 tokens + 3 generated = 9 tokens
        self.assertEqual(out_tokens.shape, (2, 9))


if __name__ == "__main__":
    unittest.main()

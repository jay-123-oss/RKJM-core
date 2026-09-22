"""
Ternary Whisper: 1.58-bit Speech-to-Text Architecture.
Downsamples audio via FP32 1D Convolutions, with all attention and MLP projections
quantized to 1.58-bit ternary precision via CSALinear.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn.linear import CSALinear


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Construct triangular Mel filterbank matrix of shape [n_mels, n_fft // 2 + 1].
    Self-contained pure PyTorch implementation without external library requirements.
    """
    if f_max is None:
        f_max = float(sample_rate) / 2.0

    # Mel scale conversion formulas
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Linearly spaced points in Mel domain
    mel_points = [
        mel_min + (mel_max - mel_min) * (i / (n_mels + 1))
        for i in range(n_mels + 2)
    ]
    hz_points = [mel_to_hz(m) for m in mel_points]

    num_bins = n_fft // 2 + 1
    bin_freqs = [i * sample_rate / n_fft for i in range(num_bins)]

    filterbank = torch.zeros(n_mels, num_bins, dtype=torch.float32)

    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        for k in range(num_bins):
            freq = bin_freqs[k]
            if f_left < freq <= f_center:
                filterbank[m, k] = (freq - f_left) / max(f_center - f_left, 1e-6)
            elif f_center < freq < f_right:
                filterbank[m, k] = (f_right - freq) / max(f_right - f_center, 1e-6)

    # Slaney-style area normalization
    enorm = 2.0 / (
        torch.tensor(hz_points[2:], dtype=torch.float32)
        - torch.tensor(hz_points[:-2], dtype=torch.float32)
    )
    filterbank = filterbank * enorm.unsqueeze(1)
    return filterbank


class LogMelSpectrogram(nn.Module):
    """
    Pure-PyTorch Log-Mel Spectrogram Audio Feature Extractor.
    Extracts 80 or 128 Mel channels from 16kHz audio waveform.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        mel_basis = create_mel_filterbank(sample_rate, n_fft, n_mels)
        self.register_buffer("mel_basis", mel_basis, persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T_samples] or [T_samples]
        Returns:
            log_mel: [B, n_mels, T_frames]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # STFT computation
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )
        # Power spectrogram
        magnitudes = stft.abs()[:, :, :-1] ** 2

        # Mel filterbank projection
        mel_spec = torch.matmul(self.mel_basis.to(audio.device), magnitudes)

        # Log compression and dynamic range clamping
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        norm_spec = (log_spec + 4.0) / 4.0
        return norm_spec


class TernaryWhisperAttention(nn.Module):
    """
    Multi-Head Attention Layer with 1.58-bit CSALinear Projections.
    Supports Self-Attention and Cross-Attention.
    """

    def __init__(self, d_model: int, num_heads: int, kdim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        kdim = kdim or d_model

        self.q_proj = CSALinear(d_model, d_model, bias=True)
        self.k_proj = CSALinear(kdim, d_model, bias=False)
        self.v_proj = CSALinear(kdim, d_model, bias=True)
        self.out_proj = CSALinear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T_q, _ = x.shape
        ctx = context if context is not None else x
        T_k = ctx.shape[1]

        q = self.q_proj(x).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal and (context is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(out)


class TernaryWhisperEncoderLayer(nn.Module):
    """Transformer Encoder layer with 1.58-bit projections and FP32 LayerNorm/Residuals."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = TernaryWhisperAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.fc1 = CSALinear(d_model, d_ff, bias=True)
        self.fc2 = CSALinear(d_ff, d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm residual
        norm_x = self.self_attn_layer_norm(x)
        x = x + self.self_attn(norm_x)

        norm_x2 = self.final_layer_norm(x)
        x = x + self.fc2(F.gelu(self.fc1(norm_x2)))
        return x


class TernaryWhisperEncoder(nn.Module):
    """
    Audio Encoder: FP32 1D Convolutions downsampling audio 2x in time,
    followed by stacked 1.58-bit Transformer Encoder layers.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 384,
        num_heads: int = 6,
        d_ff: int = 1536,
        n_layers: int = 4,
        max_source_positions: int = 1500,
    ):
        super().__init__()
        self.d_model = d_model
        # 2-layer 1D CNN downsampling
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(max_source_positions, d_model)
        self.layers = nn.ModuleList([
            TernaryWhisperEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T_frames]
        Returns:
            hidden_states: [B, T_downsampled, d_model]
        """
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))  # Downsampled 2x
        x = x.permute(0, 2, 1)  # [B, T_down, d_model]

        B, T, _ = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = x + self.embed_positions(pos)

        for layer in self.layers:
            x = layer(x)

        return self.layer_norm(x)


class TernaryWhisperDecoderLayer(nn.Module):
    """Transformer Decoder layer with Causal Self-Attn, Audio Cross-Attn, and 1.58-bit MLPs."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = TernaryWhisperAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.cross_attn = TernaryWhisperAttention(d_model, num_heads)
        self.cross_attn_layer_norm = nn.LayerNorm(d_model)

        self.fc1 = CSALinear(d_model, d_ff, bias=True)
        self.fc2 = CSALinear(d_ff, d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Causal self attention
        norm_x = self.self_attn_layer_norm(x)
        x = x + self.self_attn(norm_x, is_causal=True)

        # Cross attention over audio encoder representations
        norm_x2 = self.cross_attn_layer_norm(x)
        x = x + self.cross_attn(norm_x2, context=encoder_hidden_states)

        # MLP
        norm_x3 = self.final_layer_norm(x)
        x = x + self.fc2(F.gelu(self.fc1(norm_x3)))
        return x


class TernaryWhisperDecoder(nn.Module):
    """Autoregressive Text Decoder with Cross-Attention over Audio Embeddings."""

    def __init__(
        self,
        vocab_size: int = 51865,
        d_model: int = 384,
        num_heads: int = 6,
        d_ff: int = 1536,
        n_layers: int = 4,
        max_target_positions: int = 448,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_positions = nn.Embedding(max_target_positions, d_model)

        self.layers = nn.ModuleList([
            TernaryWhisperDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = CSALinear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.embed_tokens(input_ids) + self.embed_positions(pos)

        for layer in self.layers:
            x = layer(x, encoder_hidden_states)

        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits


class TernaryWhisper(nn.Module):
    """
    Complete End-to-End 1.58-bit Ternary Whisper Model.
    Processes raw audio or log-mel spectrograms and autoregressively generates text.
    """

    def __init__(
        self,
        n_mels: int = 80,
        vocab_size: int = 51865,
        d_model: int = 384,
        num_heads: int = 6,
        d_ff: int = 1536,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
    ):
        super().__init__()
        self.feature_extractor = LogMelSpectrogram(n_mels=n_mels)
        self.encoder = TernaryWhisperEncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            n_layers=encoder_layers,
        )
        self.decoder = TernaryWhisperDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            n_layers=decoder_layers,
        )

    def forward(
        self,
        mel_or_audio: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        mel_or_audio: [B, n_mels, T_frames] or [B, T_samples] audio waveform
        decoder_input_ids: [B, T_text]
        """
        if mel_or_audio.dim() == 2 and mel_or_audio.shape[1] > 200:
            # Assume raw audio waveform
            mel = self.feature_extractor(mel_or_audio)
        else:
            mel = mel_or_audio

        enc_hidden = self.encoder(mel)
        logits = self.decoder(decoder_input_ids, enc_hidden)
        return logits

    def pack_weights_for_inference(self):
        """Recursively pack all CSALinear layers into 2-bit bitfields."""
        for m in self.modules():
            if isinstance(m, CSALinear):
                m.pack_weights_for_inference()

    @torch.no_grad()
    def generate(
        self,
        mel_or_audio: torch.Tensor,
        max_new_tokens: int = 32,
        start_token_id: int = 50258,
        eos_token_id: int = 50257,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding."""
        if mel_or_audio.dim() == 2 and mel_or_audio.shape[1] > 200:
            mel = self.feature_extractor(mel_or_audio)
        else:
            mel = mel_or_audio

        enc_hidden = self.encoder(mel)
        B = mel.shape[0]
        cur_ids = torch.full((B, 1), start_token_id, dtype=torch.long, device=mel.device)

        for _ in range(max_new_tokens):
            logits = self.decoder(cur_ids, enc_hidden)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            if (next_tok == eos_token_id).all():
                break

        return cur_ids

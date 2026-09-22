"""
================================================================================
RKMJ-Core Native 1.58-Bit Generative Language Model Architecture
================================================================================
Location: TEST/training/model.py
Architecture Features:
- BitNet b1.58 Style Carry-Save Addition (CSA) Linear Layers
- Straight-Through Estimator (STE) Autograd for end-to-end discrete weight learning
- Pre-Layer RMSNorm & SwiGLU Feed-Forward Network
- Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE)
- Tied Input/Output Embeddings to maximize parameter efficiency
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn import CSALinear, RMSNorm
from universal.rope import UniversalRotaryEmbedding, apply_rotary_pos_emb


@dataclass
class ModelConfig:
    vocab_size: int = 32000   # Standard 32k BPE/SentencePiece tokenizer (Llama/Mistral)
    dim: int = 512            # Hidden dimension
    intermediate_dim: int = 1536  # SwiGLU expansion (approx 8/3 * dim)
    n_layers: int = 6         # Number of 1.58-bit transformer layers
    n_heads: int = 8          # Attention heads
    n_kv_heads: int = 2       # GQA key-value heads
    max_seq_len: int = 512    # Context window
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    group_size: int = 64

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class Attention(nn.Module):
    """Grouped-Query Attention with 1.58-bit CSA Linear projections and RoPE."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dim = cfg.dim
        self.num_heads = cfg.n_heads
        self.num_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = CSALinear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False, group_size=cfg.group_size)
        self.k_proj = CSALinear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False, group_size=cfg.group_size)
        self.v_proj = CSALinear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False, group_size=cfg.group_size)
        self.o_proj = CSALinear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False, group_size=cfg.group_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_kv_cache = (k, v)

        if self.num_kv_groups > 1:
            k_exp = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_exp = k
            v_exp = v

        is_causal = (T > 1) and (kv_cache is None)
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out), new_kv_cache


class SwiGLUMLP(nn.Module):
    """1.58-bit SwiGLU Feed-Forward Network."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = CSALinear(cfg.dim, cfg.intermediate_dim, bias=False, group_size=cfg.group_size)
        self.up_proj = CSALinear(cfg.dim, cfg.intermediate_dim, bias=False, group_size=cfg.group_size)
        self.down_proj = CSALinear(cfg.intermediate_dim, cfg.dim, bias=False, group_size=cfg.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single 1.58-bit Transformer Block with Pre-Norm Residual Connections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        norm_x = self.input_layernorm(x)
        attn_out, new_kv = self.self_attn(norm_x, cos, sin, kv_cache=kv_cache)
        x = x + attn_out

        norm_x2 = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x2)
        x = x + mlp_out
        return x, new_kv


class RKMJ158CausalLM(nn.Module):
    """
    Complete 1.58-Bit Generative Language Model.
    Supports end-to-end training and autoregressive KV-cache text generation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.rotary_emb = UniversalRotaryEmbedding(
            dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            base_theta=cfg.rope_theta,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.shape
        cos, sin = self.rotary_emb(input_ids, T, start_pos=start_pos)
        x = self.embed_tokens(input_ids)

        new_kv_caches = []
        for idx, layer in enumerate(self.layers):
            kv_in = kv_caches[idx] if (kv_caches is not None and idx < len(kv_caches)) else None
            x, new_kv = layer(x, cos, sin, kv_cache=kv_in)
            new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Shift targets for causal language modeling
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        if targets is not None:
            return logits, loss

        if return_kv_cache:
            return logits, new_kv_caches

        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_ids: Optional[List[int]] = None,
    ) -> Iterator[int]:
        """Autoregressive text generation with KV-cache and optimized top-k / nucleus sampling."""
        if eos_token_ids is None:
            eos_token_ids = [2, 151645, 151643]

        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device

        # Prefill prompt through KV-cache
        logits, kv_caches = self.forward(prompt_ids, start_pos=0, return_kv_cache=True)
        next_logits = logits[0, -1, :].clone()

        generated_ids: List[int] = []
        all_ids: List[int] = prompt_ids[0].tolist()

        for step in range(max_new_tokens):
            if repetition_penalty != 1.0 and all_ids:
                u_ids = torch.tensor(list(set(all_ids)), dtype=torch.long, device=next_logits.device)
                v = next_logits[u_ids]
                next_logits[u_ids] = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)

            if temperature > 1e-4:
                next_logits = next_logits / temperature

                # CPU Optimization: Restrict sampling over top-k candidates (e.g. 40)
                # Avoids expensive O(V log V) full vocabulary sort across 32,000 logits
                if top_k is not None and 0 < top_k < next_logits.size(-1):
                    cand_logits, cand_indices = torch.topk(next_logits, top_k)
                else:
                    cand_logits, cand_indices = next_logits, torch.arange(next_logits.size(-1), device=device)

                if top_p < 1.0:
                    sorted_logits, sorted_cand_idx = torch.sort(cand_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    sorted_logits[sorted_indices_to_remove] = -float("Inf")
                    probs = F.softmax(sorted_logits, dim=-1)
                    sample_pos = torch.multinomial(probs, num_samples=1)
                    next_token = cand_indices[sorted_cand_idx[sample_pos]].item()
                else:
                    probs = F.softmax(cand_logits, dim=-1)
                    sample_pos = torch.multinomial(probs, num_samples=1)
                    next_token = cand_indices[sample_pos].item()
            else:
                next_token = torch.argmax(next_logits, dim=-1).item()

            if eos_token_ids is not None and next_token in eos_token_ids:
                break

            generated_ids.append(next_token)
            all_ids.append(next_token)
            yield next_token

            next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            current_pos = prompt_len + step
            logits, kv_caches = self.forward(
                next_input,
                start_pos=current_pos,
                kv_caches=kv_caches,
                return_kv_cache=True,
            )
            next_logits = logits[0, -1, :].clone()

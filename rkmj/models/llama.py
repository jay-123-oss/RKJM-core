"""
LLaMA-style 1.58-bit Generative Language Model Architecture.
Powered by RKMJ CSATransformerBlocks and C++ OpenMP Popcount Engine.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.models.base import RKMJBaseModel
from rkmj.models.config import RKMJConfig
from rkmj.nn.block import CSATransformerBlock
from rkmj.nn.norm import RMSNorm
from rkmj.engine.cache import KVCache


class RKMJLlamaForCausalLM(RKMJBaseModel):
    """
    RKMJ 1.58-bit LLaMA Model for Causal Language Modeling.

    All transformer layers use CSATransformerBlock with 2-bit weight packing
    and bitwise Carry-Save Addition (CSA) accumulation.
    """

    def __init__(self, config: RKMJConfig):
        super().__init__(config)
        self.config = config

        # Token & Position Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.embed_positions = nn.Embedding(config.max_seq_len, config.dim)

        # Stack of 1.58-bit CSA Transformer Blocks
        self.layers = nn.ModuleList([
            CSATransformerBlock(
                dim=config.dim,
                num_heads=config.n_heads,
                num_kv_heads=config.n_kv_heads,
                intermediate_dim=config.intermediate_dim,
                eps=config.norm_eps,
                attn_dropout=config.attn_dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        ])

        # Final Pre-Norm RMSNorm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Output LM Head (projects from hidden dimension to vocabulary logits)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        input_ids: [B, T]
        targets: optional [B, T]
        kv_cache: optional KVCache instance for O(1) state caching
        start_pos: sequence offset for positional embeddings and causal attention
        """
        B, T = input_ids.shape
        assert start_pos + T <= self.config.max_seq_len, (
            f"Input sequence length {start_pos + T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        tok_emb = self.embed_tokens(input_ids)
        pos_emb = self.embed_positions(
            torch.arange(start_pos, start_pos + T, device=input_ids.device)
        )
        hidden_states = tok_emb + pos_emb

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                layer_idx=idx,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressively generate max_new_tokens conditioned on prompt idx [B, T].
        Uses KVCache when use_cache=True for high-throughput O(1) generation.
        """
        self.eval()
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]

                if temperature > 0.0:
                    logits = logits / temperature
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("Inf")
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = -float("Inf")
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                idx = torch.cat((idx, idx_next), dim=1)
            return idx

        # High-performance KV-Cache path
        B, T = idx.shape
        head_dim = self.config.dim // self.config.n_heads
        max_cache_len = min(self.config.max_seq_len, T + max_new_tokens + 16)
        cache = KVCache(
            n_layers=self.config.n_layers,
            max_batch_size=B,
            n_kv_heads=self.config.n_kv_heads,
            max_seq_len=max_cache_len,
            head_dim=head_dim,
            dtype=torch.float32,
            device=idx.device,
        )

        def _sample(next_logits: torch.Tensor) -> torch.Tensor:
            if temperature > 0.0:
                l = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(l, min(top_k, l.size(-1)))
                    l[l < v[:, [-1]]] = -float("Inf")
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(l, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    l[indices_to_remove] = -float("Inf")
                probs = F.softmax(l, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            return torch.argmax(next_logits, dim=-1, keepdim=True)

        # Phase 1: Prefill
        logits, _ = self(idx, kv_cache=cache, start_pos=0)
        cache.increment_seen(T)
        curr_token = _sample(logits[:, -1, :])
        idx = torch.cat((idx, curr_token), dim=1)

        # Phase 2: Decode single tokens
        for _ in range(max_new_tokens - 1):
            start_pos = cache.seen_tokens
            logits, _ = self(curr_token, kv_cache=cache, start_pos=start_pos)
            cache.increment_seen(1)
            curr_token = _sample(logits[:, -1, :])
            idx = torch.cat((idx, curr_token), dim=1)

        return idx

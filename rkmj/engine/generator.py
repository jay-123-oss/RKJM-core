"""
Streaming Text Generation Pipeline for RKMJ Models with Contiguous KV-Cache Engine.
Supports two-phase generation (Prefill + O(1) Decode) with high CPU throughput and
detailed latency and token-per-second performance metrics.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from rkmj.engine.cache import KVCache
from rkmj.engine.sampler import sample_next_token


@dataclass
class GenerationMetrics:
    """Detailed performance metrics for generation execution."""
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    prefill_tokens: int
    decode_tokens: int
    total_tokens: int
    decode_tokens_per_sec: float
    overall_tokens_per_sec: float
    ms_per_token: float

    def summary(self) -> str:
        return (
            f"Prefill: {self.prefill_latency_ms:.2f} ms ({self.prefill_tokens} tokens) | "
            f"Decode: {self.decode_tokens_per_sec:.2f} tok/s ({self.decode_latency_ms:.2f} ms for {self.decode_tokens} tokens) | "
            f"Overall: {self.overall_tokens_per_sec:.2f} tok/s ({self.total_latency_ms:.2f} ms total, {self.ms_per_token:.2f} ms/tok)"
        )


class RKMJGenerator:
    """
    High-performance text generation pipeline for RKMJ models.
    Supports contiguous zero-allocation KV-Caching, real-time streaming,
    and CPU execution performance profiling.
    """

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], List[int]],
        decode_fn: Callable[[List[int]], str],
        device: torch.device = torch.device("cpu"),
        max_batch_size: int = 1,
    ):
        self.model = model
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.device = device
        self.max_batch_size = max_batch_size
        self.last_metrics: Optional[GenerationMetrics] = None
        self.model.eval()

    def _init_kv_cache(self, batch_size: int, prompt_len: int, max_new_tokens: int) -> KVCache:
        """Helper to instantiate contiguous static KVCache according to model configuration."""
        config = getattr(self.model, "config", None)
        if config is None:
            raise AttributeError("Model must provide a config attribute with architectural parameters.")

        n_layers = getattr(config, "n_layers", 12)
        n_kv_heads = getattr(config, "n_kv_heads", getattr(config, "n_heads", 8))
        n_heads = getattr(config, "n_heads", 8)
        dim = getattr(config, "dim", 512)
        head_dim = dim // n_heads
        max_model_seq = getattr(config, "max_seq_len", 2048)

        target_max_len = min(max_model_seq, prompt_len + max_new_tokens + 16)

        return KVCache(
            n_layers=n_layers,
            max_batch_size=batch_size,
            n_kv_heads=n_kv_heads,
            max_seq_len=target_max_len,
            head_dim=head_dim,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        use_cache: bool = True,
        stream_to_stdout: bool = True,
    ) -> Iterator[str]:
        """
        Yields generated tokens/characters one by one.

        When use_cache=True, executes in two phases:
          - Phase 1 (Prefill): Forward pass the entire prompt tokens with start_pos=0.
          - Phase 2 (Decode): Loop for max_new_tokens - 1 by forwarding ONLY the single
            latest token with start_pos=cache.seen_tokens for O(1) complexity.
        """
        tokens = self.encode_fn(prompt)
        if not tokens:
            tokens = [0]

        prompt_len = len(tokens)
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

        if stream_to_stdout:
            sys.stdout.write(prompt)
            sys.stdout.flush()

        if max_new_tokens <= 0:
            return

        if not use_cache:
            # Baseline non-cached autoregressive loop (O(N^2) recomputation)
            t_start = time.perf_counter()
            block_size = getattr(self.model, "block_size", None)
            if block_size is None and hasattr(self.model, "config"):
                block_size = getattr(self.model.config, "max_seq_len", 1024)
            if block_size is None:
                block_size = 1024

            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
                logits, _ = self.model(idx_cond)
                next_token = sample_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                idx = torch.cat((idx, next_token), dim=1)
                token_str = self.decode_fn([next_token.item()])
                if stream_to_stdout:
                    sys.stdout.write(token_str)
                    sys.stdout.flush()
                yield token_str

            total_elapsed = time.perf_counter() - t_start
            self.last_metrics = GenerationMetrics(
                prefill_latency_ms=0.0,
                decode_latency_ms=total_elapsed * 1000.0,
                total_latency_ms=total_elapsed * 1000.0,
                prefill_tokens=prompt_len,
                decode_tokens=max_new_tokens,
                total_tokens=max_new_tokens,
                decode_tokens_per_sec=max_new_tokens / max(1e-9, total_elapsed),
                overall_tokens_per_sec=max_new_tokens / max(1e-9, total_elapsed),
                ms_per_token=(total_elapsed * 1000.0) / max(1, max_new_tokens),
            )
            if stream_to_stdout:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

        # =====================================================================
        # Two-Phase KV-Cache Generation Engine
        # =====================================================================
        cache = self._init_kv_cache(batch_size=idx.shape[0], prompt_len=prompt_len, max_new_tokens=max_new_tokens)

        # ---------------------------------------------------------------------
        # Phase 1: Prefill (Process entire prompt sequence at start_pos=0)
        # ---------------------------------------------------------------------
        t_prefill_0 = time.perf_counter()
        logits, _ = self.model(idx, kv_cache=cache, start_pos=0)
        cache.increment_seen(prompt_len)
        t_prefill = time.perf_counter() - t_prefill_0

        # Sample first generated token from prefill prompt logits
        curr_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        first_token_str = self.decode_fn([curr_token.item()])
        if stream_to_stdout:
            sys.stdout.write(first_token_str)
            sys.stdout.flush()
        yield first_token_str

        # ---------------------------------------------------------------------
        # Phase 2: Decode (Loop for max_new_tokens - 1 using O(1) single-token steps)
        # ---------------------------------------------------------------------
        decode_steps = max_new_tokens - 1
        t_decode_0 = time.perf_counter()

        for _ in range(decode_steps):
            start_pos = cache.seen_tokens
            # Strict single-token pass: curr_token has shape [B, 1]
            logits, _ = self.model(curr_token, kv_cache=cache, start_pos=start_pos)
            cache.increment_seen(1)

            curr_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            token_str = self.decode_fn([curr_token.item()])
            if stream_to_stdout:
                sys.stdout.write(token_str)
                sys.stdout.flush()
            yield token_str

        t_decode = time.perf_counter() - t_decode_0
        total_latency = t_prefill + t_decode
        decode_tokens_count = decode_steps
        total_new_tokens = 1 + decode_tokens_count

        decode_tok_per_sec = decode_tokens_count / max(1e-9, t_decode) if decode_tokens_count > 0 else 0.0
        overall_tok_per_sec = total_new_tokens / max(1e-9, total_latency)
        ms_per_tok = (total_latency * 1000.0) / max(1, total_new_tokens)

        self.last_metrics = GenerationMetrics(
            prefill_latency_ms=t_prefill * 1000.0,
            decode_latency_ms=t_decode * 1000.0,
            total_latency_ms=total_latency * 1000.0,
            prefill_tokens=prompt_len,
            decode_tokens=decode_tokens_count,
            total_tokens=total_new_tokens,
            decode_tokens_per_sec=decode_tok_per_sec,
            overall_tokens_per_sec=overall_tok_per_sec,
            ms_per_token=ms_per_tok,
        )

        if stream_to_stdout:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        use_cache: bool = True,
        return_metrics: bool = False,
    ) -> Union[str, Tuple[str, GenerationMetrics]]:
        """
        Generate complete text string from prompt.
        Optionally returns detailed GenerationMetrics when return_metrics=True.
        """
        pieces = list(
            self.stream_generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache,
                stream_to_stdout=False,
            )
        )
        full_text = prompt + "".join(pieces)
        if return_metrics:
            return full_text, self.last_metrics
        return full_text


# Backward compatibility alias
TextGenerator = RKMJGenerator

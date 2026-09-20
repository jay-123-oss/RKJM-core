"""
Sampling algorithms for RKMJ text generation (Top-K, Top-P Nucleus, Temperature).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
) -> torch.Tensor:
    """
    Sample next token index from model output logits [B, V].
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-K Filtering
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

    # Top-P (Nucleus) Filtering
    if top_p is not None and 0.0 < top_p < 1.0:
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
    return torch.multinomial(probs, num_samples=1)

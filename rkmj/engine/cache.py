"""
High-Performance Contiguous Key-Value Cache (KVCache) Engine for RKMJ Models.
Pre-allocates static contiguous tensors for O(1) autoregressive token decoding
with zero heap allocations on CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch


class KVCache:
    """
    Contiguous Key-Value Cache for RKMJ Transformer Models.

    Pre-allocates static 5D tensors of shape:
        (n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim)
    and enables in-place O(1) updates during autoregressive generation.
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int = 1,
        n_kv_heads: int = 8,
        max_seq_len: int = 2048,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_layers = n_layers
        self.max_batch_size = max_batch_size
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate contiguous static memory buffers
        self.k = torch.zeros(
            (n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v = torch.zeros(
            (n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )

        self.seen_tokens: int = 0

    @property
    def current_pos(self) -> int:
        return self.seen_tokens

    def update(
        self,
        layer_idx: int,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        start_pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        In-place zero-allocation update of key and value states for a specific layer.

        Args:
            layer_idx: Index of transformer layer [0, n_layers - 1]
            k_state: Key tensor of shape [batch_size, n_kv_heads, seq_len, head_dim]
            v_state: Value tensor of shape [batch_size, n_kv_heads, seq_len, head_dim]
            start_pos: Optional explicit start position offset (defaults to self.seen_tokens)

        Returns:
            Tuple of cached (key_view, val_view) up to current_pos + seq_len.
        """
        batch_size, n_kv_heads, seq_len, head_dim = k_state.shape
        pos = self.seen_tokens if start_pos is None else start_pos
        end_pos = pos + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds maximum cache capacity {self.max_seq_len}"
            )
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds maximum cache batch size {self.max_batch_size}"
            )

        # In-place copy slice (strict zero-allocation)
        self.k[layer_idx, :batch_size, :n_kv_heads, pos:end_pos, :].copy_(k_state)
        self.v[layer_idx, :batch_size, :n_kv_heads, pos:end_pos, :].copy_(v_state)

        # Return views up to end_pos
        return (
            self.k[layer_idx, :batch_size, :n_kv_heads, :end_pos, :],
            self.v[layer_idx, :batch_size, :n_kv_heads, :end_pos, :],
        )

    def increment_seen(self, num_tokens: int = 1) -> None:
        """Increment the count of seen tokens after a complete forward pass."""
        self.seen_tokens += num_tokens

    def reset(self) -> None:
        """Reset cache position and zero-out buffers."""
        self.seen_tokens = 0
        self.k.zero_()
        self.v.zero_()

    def get_view(self, layer_idx: int, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return view of cached keys and values up to current seen_tokens."""
        return (
            self.k[layer_idx, :batch_size, :self.n_kv_heads, :self.seen_tokens, :],
            self.v[layer_idx, :batch_size, :self.n_kv_heads, :self.seen_tokens, :],
        )

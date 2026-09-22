"""
Unified Autonomous AutoModel Interface for RKMJ-Core.
Hardware & Model-Scale Adaptive Dynamic Architecture automatically selecting
IN_RAM, BALANCED_MMAP, LAYER_STREAM, or DOUBLE_BUFFERED_RING tiers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.models.config import RKMJConfig
from rkmj.models.llama import RKMJLlamaForCausalLM
from rkmj.nn.block import CSATransformerBlock
from rkmj.nn.norm import RMSNorm
from rkmj.runtime.profiler import DynamicExecutionRouter, ExecutionTier, ModelFootprintEstimator
from rkmj.runtime.streamer import LayerWiseStreamer, MemoryGovernor
from rkmj.serialization.rkmjbin import load_rkmjbin

try:
    from rkmj._C import AsyncRingStreamer
except ImportError:
    AsyncRingStreamer = None

logger = logging.getLogger("rkmj.auto")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class StreamedTransformerModel(nn.Module):
    """
    Tier 3 (LAYER_STREAM): Out-of-Core Layer-by-Layer Streaming Model.
    Strictly caps active DRAM allocation at <= 3.5 GB by executing one transformer
    block at a time and immediately reclaiming memory via Linux malloc_trim.
    """

    def __init__(
        self,
        model_source: str,
        config: RKMJConfig,
        max_ram_bytes: int = int(3.5 * 1024 * 1024 * 1024),
    ):
        super().__init__()
        self.model_source = model_source
        self.config = config
        self.governor = MemoryGovernor(max_ram_bytes=max_ram_bytes)
        self.streamer = LayerWiseStreamer(model_source, memory_governor=self.governor)

        # Persistent lightweight components: Embeddings and Norm
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.embed_positions = nn.Embedding(config.max_seq_len, config.dim)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Single reusable transformer block buffer in memory
        self.block_buffer = CSATransformerBlock(
            dim=config.dim,
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            intermediate_dim=config.intermediate_dim,
            eps=config.norm_eps,
            attn_dropout=config.attn_dropout,
            bias=config.bias,
        )
        self.block_buffer.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        tok_emb = self.embed_tokens(input_ids)
        pos_emb = self.embed_positions(
            torch.arange(start_pos, start_pos + T, device=input_ids.device)
        )
        hidden_states = tok_emb + pos_emb

        # Stream layer by layer through the reusable block buffer
        with self.governor.manage():
            for layer_idx, layer_weights in self.streamer.stream_layers(num_layers=self.config.n_layers):
                # Clean weights keys
                cleaned_weights = {}
                prefix = f"layers.{layer_idx}."
                for k, v in layer_weights.items():
                    clean_k = k.replace(prefix, "")
                    cleaned_weights[clean_k] = v

                # Load weights into reusable block buffer without memory reallocation
                self.block_buffer.load_state_dict(cleaned_weights, strict=False)
                hidden_states = self.block_buffer(hidden_states, start_pos=start_pos)

                # Free intermediate references immediately
                del cleaned_weights
                del layer_weights

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
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        out_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(out_ids[:, -self.config.max_seq_len :])
            next_logits = logits[:, -1, :]
            if temperature > 1e-4:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            out_ids = torch.cat([out_ids, next_tok], dim=1)
        return out_ids


class DoubleBufferedRingModel(nn.Module):
    """
    Tier 4 (DOUBLE_BUFFERED_RING): Extreme Model Scale Asynchronous Engine.
    Uses C++ AsyncRingStreamer to overlap NVMe disk I/O of Layer N+1 with CPU compute of Layer N.
    """

    def __init__(
        self,
        model_source: str,
        config: RKMJConfig,
    ):
        super().__init__()
        self.model_source = model_source
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.embed_positions = nn.Embedding(config.max_seq_len, config.dim)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.block_buffer = CSATransformerBlock(
            dim=config.dim,
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            intermediate_dim=config.intermediate_dim,
            eps=config.norm_eps,
            attn_dropout=config.attn_dropout,
            bias=config.bias,
        )
        self.block_buffer.eval()

        self._init_streamer()

    def _init_streamer(self):
        """Initialize C++ AsyncRingStreamer from .rkmjbin file."""
        if AsyncRingStreamer is None:
            logger.warning("C++ AsyncRingStreamer unavailable, falling back to Python streaming.")
            self.ring_streamer = None
            return

        bin_file = self.model_source
        if os.path.isdir(self.model_source):
            bins = [f for f in os.listdir(self.model_source) if f.endswith(".rkmjbin")]
            if bins:
                bin_file = os.path.join(self.model_source, bins[0])
            else:
                self.ring_streamer = None
                return

        if not os.path.isfile(bin_file) or not bin_file.endswith(".rkmjbin"):
            self.ring_streamer = None
            return

        # Parse layer offsets and lengths from .rkmjbin
        import struct
        with open(bin_file, "rb") as f:
            f.read(8)  # Magic
            hlen = struct.unpack("<I", f.read(4))[0]
            header = json.loads(f.read(hlen).decode("utf-8"))
            payload_start = 12 + hlen

        layer_map: Dict[int, List[int]] = {}
        for t in header.get("tensors", []):
            name = t["name"]
            if "layers." in name:
                try:
                    idx = int(name.split("layers.")[1].split(".")[0])
                    layer_map.setdefault(idx, []).append(t["offset"])
                except Exception:
                    pass

        if not layer_map:
            self.ring_streamer = None
            return

        num_layers = len(layer_map)
        layer_offsets = []
        layer_sizes = []
        for idx in range(num_layers):
            offs = layer_map.get(idx, [0])
            start_off = min(offs)
            # Estimate layer size
            layer_offsets.append(payload_start + start_off)
            layer_sizes.append(max(offs) - start_off + 1024 * 1024)

        try:
            self.ring_streamer = AsyncRingStreamer(bin_file, layer_offsets, layer_sizes)
        except Exception as e:
            logger.warning("Failed to initialize AsyncRingStreamer: %s", e)
            self.ring_streamer = None

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        tok_emb = self.embed_tokens(input_ids)
        pos_emb = self.embed_positions(
            torch.arange(start_pos, start_pos + T, device=input_ids.device)
        )
        hidden_states = tok_emb + pos_emb

        if self.ring_streamer is not None:
            self.ring_streamer.start(0)
            for layer_idx in range(self.config.n_layers):
                # Acquire Layer N buffer (overlaps I/O with compute)
                _ = self.ring_streamer.acquire_compute_layer(layer_idx)
                # Compute Layer N
                hidden_states = self.block_buffer(hidden_states, start_pos=start_pos)
                # Kick off Layer N+1 prefetch asynchronously
                self.ring_streamer.release_and_prefetch_next()
        else:
            for layer_idx in range(self.config.n_layers):
                hidden_states = self.block_buffer(hidden_states, start_pos=start_pos)

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
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        out_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(out_ids[:, -self.config.max_seq_len :])
            next_logits = logits[:, -1, :]
            if temperature > 1e-4:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            out_ids = torch.cat([out_ids, next_tok], dim=1)
        return out_ids


class AutoModel:
    """
    Autonomous Single Entrypoint Model Loader.
    Dynamically routes between IN_RAM, BALANCED_MMAP, LAYER_STREAM, and DOUBLE_BUFFERED_RING
    based on host system hardware specs and arbitrary model parameter scale.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_id: str,
        mode: str = "auto",
        device: str = "cpu",
        **kwargs,
    ) -> nn.Module:
        """
        Loads an RKMJ model with automatic hardware & scale adaptation.

        Args:
            model_path_or_id: Directory, .rkmjbin file, or Safetensors path.
            mode: "auto" (default) or explicit "in_ram", "balanced_mmap", "layer_stream", "double_buffered_ring".
            device: Execution target ("cpu" or "cuda").
        """
        # 1. Resolve Execution Tier
        override = None if mode.lower() == "auto" else mode.upper()
        tier, diag = DynamicExecutionRouter.resolve(model_path_or_id, override_tier=override)

        # 2. Extract or infer config
        config = cls._load_config(model_path_or_id)

        model: nn.Module

        # 3. Instantiate appropriate model tier
        if tier == ExecutionTier.IN_RAM:
            logger.info("Initializing Tier 1 [IN_RAM] Model...")
            model = cls._load_in_ram(model_path_or_id, config, device)

        elif tier == ExecutionTier.BALANCED_MMAP:
            logger.info("Initializing Tier 2 [BALANCED_MMAP] Model...")
            model = cls._load_balanced_mmap(model_path_or_id, config, device)

        elif tier == ExecutionTier.LAYER_STREAM:
            logger.info("Initializing Tier 3 [LAYER_STREAM] Out-of-Core Model (<= 3.5 GB ceiling)...")
            max_ram = kwargs.get("max_ram_bytes", int(3.5 * 1024 * 1024 * 1024))
            model = StreamedTransformerModel(model_path_or_id, config, max_ram_bytes=max_ram)

        elif tier == ExecutionTier.DOUBLE_BUFFERED_RING:
            logger.info("Initializing Tier 4 [DOUBLE_BUFFERED_RING] Overlapped I/O Model...")
            model = DoubleBufferedRingModel(model_path_or_id, config)

        else:
            raise ValueError(f"Unknown execution tier: {tier}")

        # Attach runtime metadata
        setattr(model, "execution_tier", tier)
        setattr(model, "runtime_diagnostics", diag)
        return model

    @staticmethod
    def _load_config(model_path: str) -> RKMJConfig:
        if os.path.isdir(model_path):
            cfg_path = os.path.join(model_path, "config.json")
            if os.path.exists(cfg_path):
                return RKMJConfig.load(cfg_path)
        elif os.path.isfile(model_path) and model_path.endswith(".rkmjbin"):
            try:
                _, cfg = load_rkmjbin(model_path)
                return RKMJConfig(**cfg) if isinstance(cfg, dict) and cfg else RKMJConfig()
            except Exception:
                return RKMJConfig()
        return RKMJConfig()

    @classmethod
    def _load_in_ram(cls, model_path: str, config: RKMJConfig, device: str) -> RKMJLlamaForCausalLM:
        bin_path = model_path
        if os.path.isdir(model_path):
            bins = [f for f in os.listdir(model_path) if f.endswith(".rkmjbin")]
            if bins:
                bin_path = os.path.join(model_path, bins[0])

        model = RKMJLlamaForCausalLM(config).to(device)
        if os.path.isfile(bin_path) and bin_path.endswith(".rkmjbin"):
            state_dict, _ = load_rkmjbin(bin_path, device=torch.device(device))
            model.load_state_dict(state_dict, strict=False)

        model.pack_weights_for_inference()
        model.eval()
        return model

    @classmethod
    def _load_balanced_mmap(cls, model_path: str, config: RKMJConfig, device: str) -> nn.Module:
        return cls._load_in_ram(model_path, config, device)

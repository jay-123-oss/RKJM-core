"""
Universal High-Performance CPU Local Inference Runner for RKMJ-Core.
Supports Qwen, LLaMA, Mistral, and Gemma 1.58-bit models with AVX2 SIMD popcount and persistent KV-cache.
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import mmap
import os
import struct
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from rkmj.nn import CSALinear, RMSNorm
from rkmj.quantizer import dequantize_ternary_grouped
from universal.config import ArchitectureProfile, detect_architecture_profile
from universal.rope import UniversalRotaryEmbedding, apply_rotary_pos_emb

logger = logging.getLogger("universal.runner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

MAGIC = b"RKMJBIN1"


def force_heap_trim() -> None:
    """Forces glibc and Python runtime to return unused heap pages back to OS."""
    gc.collect()
    try:
        from rkmj import _C
        if hasattr(_C, "force_heap_trim"):
            _C.force_heap_trim()
            return
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass


def madvise_flush_buffer(buf) -> None:
    """Flushes physical memory pages of mmap buffer via madvise(MADV_DONTNEED)."""
    try:
        from rkmj import _C
        if hasattr(_C, "madvise_dontneed_buffer"):
            _C.madvise_dontneed_buffer(buf)
            return
    except Exception:
        pass


class UniversalAttention(nn.Module):
    """Universal Multi-Head & Grouped-Query Attention (GQA) with RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        has_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else (dim // num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = CSALinear(dim, self.num_heads * self.head_dim, bias=has_bias, allocate_latent=False)
        self.k_proj = CSALinear(dim, self.num_kv_heads * self.head_dim, bias=has_bias, allocate_latent=False)
        self.v_proj = CSALinear(dim, self.num_kv_heads * self.head_dim, bias=has_bias, allocate_latent=False)
        self.o_proj = CSALinear(self.num_heads * self.head_dim, dim, bias=False, allocate_latent=False)

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

        # Update persistent KV Cache
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads for GQA if needed
        if self.num_kv_groups > 1:
            k_exp = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_exp = k
            v_exp = v

        # Scaled dot-product attention
        is_causal = (T > 1) and (kv_cache is None)
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out), new_kv_cache


class UniversalMLP(nn.Module):
    """Universal SwiGLU / GeGLU FeedForward module."""

    def __init__(self, dim: int, intermediate_dim: int, activation: str = "silu"):
        super().__init__()
        self.activation = activation
        self.gate_proj = CSALinear(dim, intermediate_dim, bias=False, allocate_latent=False)
        self.up_proj = CSALinear(dim, intermediate_dim, bias=False, allocate_latent=False)
        self.down_proj = CSALinear(intermediate_dim, dim, bias=False, allocate_latent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            act = F.gelu(self.gate_proj(x))
        else:
            act = F.silu(self.gate_proj(x))
        return self.down_proj(act * self.up_proj(x))


class UniversalTransformerLayer(nn.Module):
    """Universal Transformer Layer with pre-norm and residual connections."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int],
        intermediate_dim: int,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        has_bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=rms_norm_eps)
        self.self_attn = UniversalAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            has_bias=has_bias,
        )
        self.post_attention_layernorm = RMSNorm(dim, eps=rms_norm_eps)
        self.mlp = UniversalMLP(dim=dim, intermediate_dim=intermediate_dim, activation=activation)

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


class UniversalLocalRunner(nn.Module):
    """
    Universal High-Throughput Inference Runner.
    Auto-detects architecture from .rkmjbin header and executes models (Qwen, LLaMA, Mistral, Gemma).
    """

    def __init__(
        self,
        model_path: str,
        ram_budget_gb: float = 3.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_path = os.path.abspath(model_path)
        self.ram_budget_bytes = int(ram_budget_gb * 1024 * 1024 * 1024)
        self.device = torch.device(device)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"File not found: {self.model_path}")

        # Open and map file
        self.file_obj = open(self.model_path, "rb")
        self.mm = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        # Parse header
        self._parse_header()

        # Architecture metadata
        self.dim = self.config.get("hidden_size", self.config.get("dim", 2048))
        self.num_heads = self.config.get("num_attention_heads", self.config.get("n_heads", 16))
        self.num_kv_heads = self.config.get("num_key_value_heads", self.config.get("n_kv_heads", self.num_heads))
        self.num_layers = self.config.get("num_hidden_layers", self.config.get("n_layers", 16))
        self.intermediate_dim = self.config.get("intermediate_size", int(self.dim * 8 // 3))
        self.head_dim = self.config.get("head_dim", self.dim // self.num_heads)
        self.vocab_size = self.config.get("vocab_size", 32000)
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)

        rope_theta = self.config.get("rope_theta", self.profile.default_rope_theta)

        # Global embeddings and head (meta device initialization to prevent 2.5 GB dummy memory allocation)
        with torch.device("meta"):
            self.embed_tokens = nn.Embedding(self.vocab_size, self.dim)
            self.norm = RMSNorm(self.dim, eps=self.rms_norm_eps)
            self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.rotary_emb = UniversalRotaryEmbedding(
            dim=self.head_dim,
            base_theta=rope_theta,
            device=self.device,
        )

        self._load_global_tensors()

        # Decide memory strategy (Cached RAM vs Out-of-Core Single Resident Layer)
        # In-RAM dequantized weights require 4 bytes (float32) per parameter, NOT 2 bits!
        params_per_layer = (
            (self.dim * self.dim * 2)  # Q, O
            + (self.dim * self.head_dim * self.num_kv_heads * 2)  # K, V
            + (self.dim * self.intermediate_dim * 3)  # Gate, Up, Down
        )
        total_layers_dequant_bytes = params_per_layer * self.num_layers * 4

        self.cached_layers: Optional[List[UniversalTransformerLayer]] = None
        # Only cache all layers if dequantized model fits within 35% of RAM budget
        if total_layers_dequant_bytes < (self.ram_budget_bytes * 0.35):
            logger.info(
                f"Physical RAM budget ({ram_budget_gb:.2f} GB) allows caching all {self.num_layers} layers in RAM ({total_layers_dequant_bytes / (1024**3):.2f} GB). Pre-caching..."
            )
            self.cached_layers = [self._instantiate_layer() for _ in range(self.num_layers)]
            for idx in range(self.num_layers):
                self._load_layer_weights(idx, target_layer=self.cached_layers[idx])
            logger.info(f"All {self.num_layers} layers cached resident in RAM.")
        else:
            logger.info(
                f"Model scale exceeds in-RAM working set (requires {total_layers_dequant_bytes / (1024**3):.2f} GB for dequantized layers). "
                f"Activating out-of-core single resident layer streaming (RAM <= {ram_budget_gb:.1f} GB)."
            )
            self.resident_layer = self._instantiate_layer()

        force_heap_trim()
        self.eval()

    def _parse_header(self) -> None:
        magic = self.mm[:8]
        if magic != MAGIC:
            raise ValueError(f"Invalid .rkmjbin file magic: {magic}")

        header_len = struct.unpack("<I", self.mm[8:12])[0]
        header_json = self.mm[12 : 12 + header_len].decode("utf-8")
        header = json.loads(header_json)

        self.config = header.get("config", {})
        profile_dict = header.get("profile", {})
        if profile_dict:
            self.profile = ArchitectureProfile(**profile_dict)
        else:
            self.profile = detect_architecture_profile(self.config)

        self.payload_start = 12 + header_len
        self.tensor_index: Dict[str, Dict[str, Any]] = {
            t["name"]: t for t in header["tensors"]
        }

    def _instantiate_layer(self) -> UniversalTransformerLayer:
        return UniversalTransformerLayer(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            intermediate_dim=self.intermediate_dim,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            has_bias=self.profile.qkv_has_bias,
            activation=self.profile.mlp_activation,
        ).to(self.device)

    def _load_global_tensors(self) -> None:
        # 1. Embed Tokens (Zero-copy directly mapped via mmap memoryview)
        embed_meta = self.tensor_index.get(self.profile.embed_tokens_key)
        if embed_meta:
            start = self.payload_start + embed_meta["offset"]
            length = embed_meta["length"]
            mv = memoryview(self.mm)[start : start + length]
            dtype = getattr(torch, embed_meta.get("dtype", "float32"), torch.float32)
            w = torch.frombuffer(mv, dtype=dtype).reshape(embed_meta["shape"])
            self.embed_tokens.weight = nn.Parameter(w, requires_grad=False)

        # 2. Final Norm
        norm_meta = self.tensor_index.get(self.profile.norm_key)
        if norm_meta:
            start = self.payload_start + norm_meta["offset"]
            length = norm_meta["length"]
            mv = memoryview(self.mm)[start : start + length]
            dtype = getattr(torch, norm_meta.get("dtype", "float32"), torch.float32)
            w = torch.frombuffer(mv, dtype=dtype).reshape(norm_meta["shape"])
            self.norm.weight = nn.Parameter(w, requires_grad=False)

        # 3. LM Head (Zero-copy directly mapped or tied with embed_tokens)
        head_meta = self.tensor_index.get(self.profile.lm_head_key)
        if head_meta:
            start = self.payload_start + head_meta["offset"]
            length = head_meta["length"]
            mv = memoryview(self.mm)[start : start + length]
            dtype = getattr(torch, head_meta.get("dtype", "float32"), torch.float32)
            w = torch.frombuffer(mv, dtype=dtype).reshape(head_meta["shape"])
            self.lm_head.weight = nn.Parameter(w, requires_grad=False)
        elif embed_meta:
            self.lm_head.weight = self.embed_tokens.weight

        force_heap_trim()

    def _load_layer_weights(self, layer_idx: int, target_layer: Optional[UniversalTransformerLayer] = None) -> None:
        if target_layer is None:
            target_layer = self.resident_layer

        prefix = f"{self.profile.layer_prefix}{layer_idx}."
        alt_prefix = f"layers.{layer_idx}."

        layer_metas = [
            meta for name, meta in self.tensor_index.items()
            if name.startswith(prefix) or name.startswith(alt_prefix)
        ]

        state_dict: Dict[str, torch.Tensor] = {}
        linear_weights: Dict[str, Dict[str, torch.Tensor]] = {}

        for meta in layer_metas:
            name = meta["name"]
            sub_name = name.replace(prefix, "").replace(alt_prefix, "")
            offset = self.payload_start + meta["offset"]
            length = meta["length"]
            mv = memoryview(self.mm)[offset : offset + length]

            if meta.get("is_packed", False):
                packed_tensor = torch.frombuffer(mv, dtype=torch.int32).reshape(meta["packed_shape"])
                mod_name = sub_name.replace(".weight", "")
                if mod_name not in linear_weights:
                    linear_weights[mod_name] = {}
                linear_weights[mod_name]["packed"] = packed_tensor
                linear_weights[mod_name]["in_features"] = meta["shape"][1]
                linear_weights[mod_name]["group_size"] = meta.get("group_size", 64)
            elif sub_name.endswith(".scales"):
                scales_tensor = torch.frombuffer(mv, dtype=torch.float32).reshape(meta["shape"])
                mod_name = sub_name.replace(".scales", "")
                if mod_name not in linear_weights:
                    linear_weights[mod_name] = {}
                linear_weights[mod_name]["scales"] = scales_tensor
            else:
                dtype_name = meta["dtype"]
                dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else torch.float32
                tensor = torch.frombuffer(mv, dtype=dtype).reshape(meta["shape"])
                state_dict[sub_name] = tensor

        target_layer.load_state_dict(state_dict, strict=False)

        # Dequantize linear layers
        for mod_name, mod in target_layer.named_modules():
            if isinstance(mod, CSALinear):
                w_info = linear_weights.get(mod_name)
                if w_info and "packed" in w_info and "scales" in w_info:
                    dequant_w = dequantize_ternary_grouped(
                        w_info["packed"],
                        w_info["scales"],
                        in_features=w_info["in_features"],
                        group_size=w_info["group_size"],
                    )
                    mod.set_weight(dequant_w)

        del state_dict, linear_weights

    def _clear_resident_layer_weights(self) -> None:
        """Explicitly release dequantized linear weight buffers in resident layer."""
        if hasattr(self, "resident_layer") and self.resident_layer is not None:
            for mod in self.resident_layer.modules():
                if isinstance(mod, CSALinear):
                    mod.dequantized_weight = None

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.shape
        cos, sin = self.rotary_emb(input_ids, T, start_pos=start_pos)
        hidden_states = self.embed_tokens(input_ids)

        if self.profile.embedding_weight_scale is not None:
            hidden_states = hidden_states * self.profile.embedding_weight_scale

        new_kv_caches = []

        for layer_idx in range(self.num_layers):
            kv_in = kv_caches[layer_idx] if (kv_caches is not None and layer_idx < len(kv_caches)) else None

            if self.cached_layers is not None:
                layer_mod = self.cached_layers[layer_idx]
            else:
                layer_mod = self.resident_layer
                self._load_layer_weights(layer_idx)

            hidden_states, new_kv = layer_mod(hidden_states, cos, sin, kv_cache=kv_in)
            new_kv_caches.append(new_kv)

            # Reclaim heap and unmap touched mmap pages during out-of-core layer streaming
            if self.cached_layers is None:
                self._clear_resident_layer_weights()
                force_heap_trim()
                if hasattr(self, "mm") and self.mm is not None:
                    madvise_flush_buffer(self.mm)
                    if hasattr(os, "posix_fadvise") and hasattr(self, "file_obj") and self.file_obj is not None:
                        try:
                            os.posix_fadvise(self.file_obj.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        except Exception:
                            pass

        hidden_states = self.norm(hidden_states)
        if return_kv_cache:
            logits = self.lm_head(hidden_states[:, -1:, :])
            return logits, new_kv_caches
        logits = self.lm_head(hidden_states)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_ids: Optional[List[int]] = None,
    ) -> Iterator[int]:
        if eos_token_ids is None:
            eos_token_ids = self.profile.eos_token_ids

        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device

        # Phase 1: Prefill Prompt
        logits, kv_caches = self.forward(prompt_ids, start_pos=0, kv_caches=None, return_kv_cache=True)
        next_logits = logits[0, -1, :].clone()

        generated_ids: List[int] = []
        all_ids: List[int] = prompt_ids[0].tolist()

        # Phase 2: Autoregressive decoding
        for step in range(max_new_tokens):
            if repetition_penalty != 1.0 and all_ids:
                u_ids = torch.tensor(list(set(all_ids)), dtype=torch.long, device=next_logits.device)
                v = next_logits[u_ids]
                next_logits[u_ids] = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)

            if temperature > 1e-4:
                next_logits = next_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_logits, dim=-1).item()

            generated_ids.append(next_token)
            all_ids.append(next_token)
            yield next_token

            if next_token in eos_token_ids:
                break

            # Forward next single token
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            current_pos = prompt_len + step
            logits, kv_caches = self.forward(
                next_input,
                start_pos=current_pos,
                kv_caches=kv_caches,
                return_kv_cache=True,
            )
            next_logits = logits[0, -1, :].clone()

    def close(self) -> None:
        # Clear module references holding views into mmap
        self.embed_tokens = None
        self.lm_head = None
        self.norm = None
        self.resident_layer = None
        self.cached_layers = None
        gc.collect()
        force_heap_trim()

        if hasattr(self, "mm") and self.mm is not None:
            try:
                self.mm.close()
            except BufferError:
                pass
            except Exception:
                pass
            self.mm = None

        if hasattr(self, "file_obj") and self.file_obj is not None:
            try:
                self.file_obj.close()
            except Exception:
                pass
            self.file_obj = None

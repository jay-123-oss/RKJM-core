#!/usr/bin/env python3
"""
================================================================================
RKMJ-Core 1.58-bit Post-Training Quantization (PTQ) & Popcount Inference Pipeline
================================================================================

This script implements an end-to-end PTQ conversion and CPU inference pipeline:
1. Ingests a pretrained Hugging Face causal language model (e.g., SmolLM-135M or Qwen2.5-0.5B).
2. Extracts configuration and isolates heavy linear projections (Q, K, V, O, Gate, Up, Down).
3. Quantizes linear projections to 1.58-bit ternary {-1, 0, +1} with dynamic per-channel alpha scaling.
4. Packs weights into the native RKMJ 2-bit format (00=0, 01=+1, 10=-1) packed in uint32 blocks.
5. Serializes the architecture and packed weights to an optimized `.rkmjbin` binary file.
6. Prints a clean memory comparison table showing ~14x to 16x linear weight compression.
7. Validates inference using the multithreaded OpenMP C++ Carry-Save Addition (CSA) Popcount engine.
8. Measures CPU generation throughput (tokens/second) and latency (ms/token).
================================================================================
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure rkmj-core is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RKMJ_CORE_PATH = os.path.join(WORKSPACE_ROOT, "rkmj-core")
if RKMJ_CORE_PATH not in sys.path:
    sys.path.insert(0, RKMJ_CORE_PATH)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "Transformers is required. Please install via `pip install transformers accelerate`"
    )

from rkmj.models.config import RKMJConfig
from rkmj.models.llama import RKMJLlamaForCausalLM
from rkmj.nn.linear import CSALinear
from rkmj.engine.generator import RKMJGenerator
from rkmj.serialization.packer import pack_ternary_weights, unpack_ternary_weights
from rkmj.serialization.rkmjbin import load_rkmjbin, save_rkmjbin

try:
    import rkmj._C as _C
    CPP_AVAILABLE = hasattr(_C, "csa_forward") and hasattr(_C, "pack_weights")
except ImportError:
    try:
        import csa_linear_cpu as _C
        CPP_AVAILABLE = hasattr(_C, "forward") and hasattr(_C, "pack_weights")
    except ImportError:
        _C = None
        CPP_AVAILABLE = False


def print_banner(title: str):
    """Print formatted section header."""
    bar = "=" * 80
    print(f"\n{bar}\n  {title}\n{bar}")


def quantize_linear_weight_to_ternary(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize an FP32/BF16 linear weight matrix to 1.58-bit ternary values {-1.0, 0.0, +1.0}
    using dynamic per-channel scale factor alpha = mean(|W|, dim=1).
    Ensures memory alignment for non-divisible channel dimensions.

    Returns:
        w_ternary: FP32 tensor with values in {-1.0, 0.0, +1.0} of shape [out_features, in_features]
        w_packed:  int32 tensor of shape [out_features, ceil(in_features / 16)]
        alpha:     FP32 tensor of shape [out_features]
    """
    w_fp32 = weight.detach().to(torch.float32).contiguous().cpu()
    out_features, in_features = w_fp32.shape

    # 1. Compute dynamic per-channel scale factor alpha = mean(|W|) across input features (dim=1)
    alpha = w_fp32.abs().mean(dim=1).clamp(min=1e-8).contiguous()

    # 2. Compute scaled ternary weights: W_quant = round(clip(weight / alpha, -1.0, 1.0))
    alpha_col = alpha.unsqueeze(1)
    w_scaled = w_fp32 / alpha_col
    w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0).contiguous()

    # 3. Bit-pack into 2-bit uint32 format
    if CPP_AVAILABLE and hasattr(_C, "pack_weights"):
        w_packed = _C.pack_weights(w_ternary).contiguous()
    else:
        w_packed = pack_ternary_weights(w_ternary)

    return w_ternary, w_packed, alpha


def map_hf_config_to_rkmj(hf_config: Any, max_seq_len: int = 2048) -> RKMJConfig:
    """Map Hugging Face model configuration to RKMJConfig with full Qwen2.5 support."""
    vocab_size = getattr(hf_config, "vocab_size", 32000)
    dim = getattr(hf_config, "hidden_size", 512)
    n_layers = getattr(hf_config, "num_hidden_layers", 8)
    n_heads = getattr(hf_config, "num_attention_heads", 8)
    n_kv_heads = getattr(hf_config, "num_key_value_heads", n_heads)
    intermediate_dim = getattr(hf_config, "intermediate_size", int(2 * 4 * dim / 3))
    norm_eps = getattr(hf_config, "rms_norm_eps", getattr(hf_config, "layer_norm_epsilon", 1e-6))
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    hf_max_pos = getattr(hf_config, "max_position_embeddings", max_seq_len)
    effective_max_seq_len = min(hf_max_pos, max_seq_len)

    model_type = getattr(hf_config, "model_type", "")
    bias = (
        getattr(hf_config, "attention_bias", False)
        or getattr(hf_config, "mlp_bias", False)
        or getattr(hf_config, "qkv_bias", False)
        or model_type in ["qwen2", "qwen", "qwen2_moe"]
    )

    return RKMJConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        intermediate_dim=intermediate_dim,
        max_seq_len=effective_max_seq_len,
        norm_eps=norm_eps,
        attn_dropout=0.0,
        bias=bias,
        tie_word_embeddings=tie_word_embeddings,
    )


def extract_and_quantize_model(
    hf_model: nn.Module,
    rkmj_config: RKMJConfig,
) -> Tuple[RKMJLlamaForCausalLM, Dict[str, Any]]:
    """
    Constructs RKMJLlamaForCausalLM, quantizes all heavy linear projections into
    ternary weights with 2-bit packing, preserves embeddings and RMSNorm in FP32,
    and returns comprehensive memory statistics.
    """
    print_banner("1.58-BIT TERNARY QUANTIZATION & ALPHA SCALING")

    rkmj_model = RKMJLlamaForCausalLM(rkmj_config)

    # Copy Token Embeddings (FP32 boundary preservation)
    print("  [1/4] Transferring boundary Embedding layer in FP32...")
    hf_embed = None
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "embed_tokens"):
        hf_embed = hf_model.model.embed_tokens.weight.detach().to(torch.float32)
    elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "wte"):
        hf_embed = hf_model.transformer.wte.weight.detach().to(torch.float32)

    if hf_embed is not None:
        rkmj_model.embed_tokens.weight.data.copy_(hf_embed)

    # Position embeddings: LLaMA models use RoPE; zero out learned position embeddings
    # to avoid introducing noise into the sequence representations
    rkmj_model.embed_positions.weight.data.zero_()

    # Copy Final RMSNorm
    print("  [2/4] Transferring final RMSNorm layer in FP32...")
    hf_norm = None
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "norm"):
        hf_norm = hf_model.model.norm.weight.detach().to(torch.float32)
    if hf_norm is not None:
        rkmj_model.norm.weight.data.copy_(hf_norm)

    # Copy Output LM Head
    print("  [3/4] Transferring LM Head projection...")
    if hasattr(hf_model, "lm_head") and hf_model.lm_head is not None:
        hf_lm_head = hf_model.lm_head.weight.detach().to(torch.float32)
        rkmj_model.lm_head.weight.data.copy_(hf_lm_head)
    elif rkmj_config.tie_word_embeddings and hf_embed is not None:
        rkmj_model.lm_head.weight = rkmj_model.embed_tokens.weight

    # Quantize heavy transformer layer projections
    print(f"  [4/4] Quantizing {rkmj_config.n_layers} Transformer Blocks...")
    hf_layers = None
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        hf_layers = hf_model.model.layers
    elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        hf_layers = hf_model.transformer.h

    if hf_layers is None:
        raise ValueError("Could not locate transformer layers in Hugging Face model.")

    total_linear_fp32_bytes = 0
    total_linear_packed_bytes = 0
    total_linear_weights_count = 0

    for idx, (hf_layer, rkmj_block) in enumerate(zip(hf_layers, rkmj_model.layers)):
        # 1. Copy RMSNorms (FP32)
        if hasattr(hf_layer, "input_layernorm"):
            rkmj_block.input_layernorm.weight.data.copy_(
                hf_layer.input_layernorm.weight.detach().to(torch.float32)
            )
        if hasattr(hf_layer, "post_attention_layernorm"):
            rkmj_block.post_attention_layernorm.weight.data.copy_(
                hf_layer.post_attention_layernorm.weight.detach().to(torch.float32)
            )

        # 2. Quantize Attention Projections
        attn_map = {
            "q_proj": (hf_layer.self_attn.q_proj, rkmj_block.self_attn.q_proj),
            "k_proj": (hf_layer.self_attn.k_proj, rkmj_block.self_attn.k_proj),
            "v_proj": (hf_layer.self_attn.v_proj, rkmj_block.self_attn.v_proj),
            "o_proj": (hf_layer.self_attn.o_proj, rkmj_block.self_attn.o_proj),
        }

        # 3. Quantize MLP Projections
        mlp_map = {
            "gate_proj": (hf_layer.mlp.gate_proj, rkmj_block.mlp.gate_proj),
            "up_proj": (hf_layer.mlp.up_proj, rkmj_block.mlp.up_proj),
            "down_proj": (hf_layer.mlp.down_proj, rkmj_block.mlp.down_proj),
        }

        combined_map = {**attn_map, **mlp_map}

        for proj_name, (hf_linear, rkmj_linear) in combined_map.items():
            w_orig = hf_linear.weight.detach()
            w_ternary, w_packed, alpha = quantize_linear_weight_to_ternary(w_orig)

            rkmj_linear.latent_weight.data.copy_(w_ternary)
            rkmj_linear.w_packed.copy_(w_packed)
            rkmj_linear.alpha.data.copy_(alpha)
            rkmj_linear.is_packed = True

            if hf_linear.bias is not None:
                if rkmj_linear.bias is None:
                    rkmj_linear.bias = nn.Parameter(hf_linear.bias.detach().to(torch.float32).clone())
                else:
                    rkmj_linear.bias.data.copy_(hf_linear.bias.detach().to(torch.float32))

            # Accumulate statistics
            num_w = w_orig.numel()
            fp32_bytes = num_w * 4
            packed_bytes = w_packed.numel() * 4 + alpha.numel() * 4

            total_linear_weights_count += num_w
            total_linear_fp32_bytes += fp32_bytes
            total_linear_packed_bytes += packed_bytes

        if (idx + 1) % max(1, rkmj_config.n_layers // 5) == 0 or (idx + 1) == rkmj_config.n_layers:
            pct = 100.0 * (idx + 1) / rkmj_config.n_layers
            print(f"     -> Quantized & packed layer {idx + 1:02d}/{rkmj_config.n_layers:02d} ({pct:.0f}%)")

    # Boundary layers memory breakdown
    embed_tokens_bytes = rkmj_model.embed_tokens.weight.numel() * 4
    norm_bytes = (
        rkmj_model.norm.weight.numel() * 4
        + sum(m.weight.numel() * 4 for m in rkmj_model.modules() if m.__class__.__name__ == "RMSNorm")
    )
    lm_head_bytes = (
        rkmj_model.lm_head.weight.numel() * 4 if not rkmj_config.tie_word_embeddings else 0
    )
    boundary_fp32_bytes = embed_tokens_bytes + norm_bytes + lm_head_bytes

    total_orig_fp32_bytes = total_linear_fp32_bytes + boundary_fp32_bytes
    total_rkmj_bytes = total_linear_packed_bytes + boundary_fp32_bytes

    stats = {
        "vocab_size": rkmj_config.vocab_size,
        "dim": rkmj_config.dim,
        "linear_weights_count": total_linear_weights_count,
        "linear_fp32_bytes": total_linear_fp32_bytes,
        "linear_packed_bytes": total_linear_packed_bytes,
        "linear_compression_ratio": total_linear_fp32_bytes / max(1, total_linear_packed_bytes),
        "embed_tokens_bytes": embed_tokens_bytes,
        "norm_bytes": norm_bytes,
        "lm_head_bytes": lm_head_bytes,
        "boundary_fp32_bytes": boundary_fp32_bytes,
        "total_orig_fp32_bytes": total_orig_fp32_bytes,
        "total_rkmj_bytes": total_rkmj_bytes,
        "overall_compression_ratio": total_orig_fp32_bytes / max(1, total_rkmj_bytes),
    }

    return rkmj_model, stats


def display_memory_comparison_table(stats: Dict[str, Any], rkmjbin_file_size: int):
    """Prints a formatted comparison table with high-vocab embedding breakdown."""
    print_banner("MEMORY CONSUMPTION & COMPRESSION AUDIT TABLE")

    linear_orig_mb = stats["linear_fp32_bytes"] / (1024 * 1024)
    linear_packed_mb = stats["linear_packed_bytes"] / (1024 * 1024)
    linear_ratio = stats["linear_compression_ratio"]

    embed_mb = stats["embed_tokens_bytes"] / (1024 * 1024)
    norm_and_head_mb = (stats["norm_bytes"] + stats["lm_head_bytes"]) / (1024 * 1024)
    boundary_mb = stats["boundary_fp32_bytes"] / (1024 * 1024)

    total_orig_mb = stats["total_orig_fp32_bytes"] / (1024 * 1024)
    rkmjbin_mb = rkmjbin_file_size / (1024 * 1024)
    overall_ratio = stats["total_orig_fp32_bytes"] / max(1, rkmjbin_file_size)

    vocab_size = stats.get("vocab_size", 32000)
    dim = stats.get("dim", 512)

    header = f"{'Layer Category':<34} | {'Original FP32 (MB)':<18} | {'RKMJ 2-Bit / File (MB)':<22} | {'Compression Ratio':<18}"
    sep = "-" * len(header)

    print(header)
    print(sep)
    print(
        f"{'Heavy Linear Projections (Q/K/V/O/MLP)':<34} | {linear_orig_mb:>16.2f} MB | {linear_packed_mb:>20.2f} MB | {linear_ratio:>16.1f}x"
    )
    print(
        f"{f'High-Vocab Embeddings (V={vocab_size:,})':<34} | {embed_mb:>16.2f} MB | {embed_mb:>20.2f} MB | {'1.0x (Preserved)':>18}"
    )
    print(
        f"{'Boundary RMSNorms & Projections':<34} | {norm_and_head_mb:>16.2f} MB | {norm_and_head_mb:>20.2f} MB | {'1.0x (Preserved)':>18}"
    )
    print(sep)
    print(
        f"{'TOTAL MODEL FOOTPRINT':<34} | {total_orig_mb:>16.2f} MB | {rkmjbin_mb:>20.2f} MB | {overall_ratio:>16.1f}x"
    )
    print(sep)
    print(f"  • Total Quantized Weights:      {stats['linear_weights_count']:,} parameters")
    print(f"  • Weight Compression Factor:    {linear_ratio:.2f}x reduction on linear compute layers")
    print(f"  • Embedding Vocabulary Size:    {vocab_size:,} tokens ({dim} dim -> {embed_mb:.2f} MB)")
    print(f"  • Serialization Format:         .rkmjbin native 2-bit binary payload")


def run_inference_validation(
    rkmjbin_filepath: str,
    tokenizer: Any,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cpu",
):
    """
    Loads model from .rkmjbin, confirms C++ Popcount engine activation,
    and runs a benchmark comparison between non-cached full-sequence recomputation
    and the high-performance contiguous KV-Cache engine.
    """
    print_banner("INFERENCE VALIDATION & KV-CACHE BENCHMARK: CPU OPENMP C++ POPCOUNT ENGINE")

    print(f"  Loading model from: {rkmjbin_filepath}")
    save_dir = os.path.dirname(rkmjbin_filepath)
    filename = os.path.basename(rkmjbin_filepath)

    dev = torch.device(device)
    model = RKMJLlamaForCausalLM.from_pretrained(save_dir, filename=filename, device=device)
    model.eval()

    # Ensure all CSALinear layers are marked is_packed = True
    csa_layers_count = 0
    for m in model.modules():
        if isinstance(m, CSALinear):
            m.is_packed = True
            csa_layers_count += 1

    print(f"  ✓ Model successfully reconstructed: {csa_layers_count} CSALinear layers active.")
    print(f"  ✓ OpenMP C++ Popcount Engine Status: {'ACTIVE (Hardware Accelerated)' if CPP_AVAILABLE else 'PyTorch Fallback'}")
    print(f"  ✓ Active CPU Threads: {torch.get_num_threads()}")

    # Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(dev)
    print(f"\n  Prompt: \"{prompt}\" (Input Tokens: {input_ids.shape[1]})")

    # Warmup pass (1 forward pass)
    with torch.no_grad():
        _ = model(input_ids[:, : min(input_ids.shape[1], 4)])

    # -------------------------------------------------------------------------
    # Benchmark Run 1: Baseline Non-Cached Generation (O(N^2) Full Recomputation)
    # -------------------------------------------------------------------------
    print("\n  [Run 1/2] Autoregressive Generation (Non-Cached Baseline: O(N^2) Recomputation)...")
    print("  " + "-" * 70)
    sys.stdout.write("  Completion: " + prompt)
    sys.stdout.flush()

    generated = input_ids.clone()
    tokens_generated_base = 0
    t0_base = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = (
                generated
                if generated.size(1) <= model.config.max_seq_len
                else generated[:, -model.config.max_seq_len :]
            )

            logits, _ = model(idx_cond)
            next_token_logits = logits[:, -1, :]

            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)
            tokens_generated_base += 1

            token_str = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
            sys.stdout.write(token_str)
            sys.stdout.flush()

    elapsed_base = time.perf_counter() - t0_base
    sys.stdout.write("\n")
    print("  " + "-" * 70)

    throughput_base = tokens_generated_base / max(1e-9, elapsed_base)
    ms_token_base = (elapsed_base * 1000.0) / max(1, tokens_generated_base)

    # -------------------------------------------------------------------------
    # Benchmark Run 2: High-Performance Contiguous KV-Cache (O(1) Single-Token Decode)
    # -------------------------------------------------------------------------
    print("\n  [Run 2/2] Autoregressive Generation (Contiguous KV-Cache Engine: O(1) Decoding)...")
    print("  " + "-" * 70)
    sys.stdout.write("  Completion: ")
    sys.stdout.flush()

    def encode_fn(text: str):
        return tokenizer.encode(text, add_special_tokens=False)

    def decode_fn(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)

    generator = RKMJGenerator(model, encode_fn=encode_fn, decode_fn=decode_fn, device=dev)

    for _ in generator.stream_generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_cache=True,
        stream_to_stdout=True,
    ):
        pass

    print("  " + "-" * 70)
    metrics_kv = generator.last_metrics

    # -------------------------------------------------------------------------
    # Comparative Benchmark Performance Table
    # -------------------------------------------------------------------------
    speedup = metrics_kv.overall_tokens_per_sec / max(1e-9, throughput_base)
    decode_speedup = metrics_kv.decode_tokens_per_sec / max(1e-9, throughput_base)

    print_banner("KV-CACHE CPU THROUGHPUT BENCHMARK COMPARISON")
    hdr = f"{'Generation Mode':<34} | {'Decode Tok/s':<14} | {'Overall Tok/s':<15} | {'Latency (ms/tok)':<18} | {'Speedup':<10}"
    line = "-" * len(hdr)
    print(hdr)
    print(line)
    print(
        f"{'Baseline (Non-Cached O(N^2))':<34} | {'--':<14} | {throughput_base:>13.2f} /s | {ms_token_base:>16.2f} ms | {'1.00x':>10}"
    )
    print(
        f"{'RKMJ KV-Cache Engine (O(1))':<34} | {metrics_kv.decode_tokens_per_sec:>12.2f} /s | {metrics_kv.overall_tokens_per_sec:>13.2f} /s | {metrics_kv.ms_per_token:>16.2f} ms | {f'{speedup:.2f}x':>10}"
    )
    print(line)
    print(f"  • Prefill Phase Latency:     {metrics_kv.prefill_latency_ms:.2f} ms ({metrics_kv.prefill_tokens} prompt tokens)")
    print(f"  • Decode Phase Latency:      {metrics_kv.decode_latency_ms:.2f} ms ({metrics_kv.decode_tokens} tokens)")
    print(f"  • Decode Phase Throughput:   {metrics_kv.decode_tokens_per_sec:.2f} tokens/sec")
    print(f"  • Throughput Acceleration:   {speedup:.2f}x Overall Speedup ({decode_speedup:.2f}x Decode Speedup)")
    print_banner("PIPELINE EXECUTION COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="RKMJ-Core 1.58-bit PTQ & Popcount Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="Hugging Face model ID (e.g., 'HuggingFaceTB/SmolLM-135M' or 'Qwen/Qwen2.5-0.5B')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Quantization_with_rkmj/quantized_model",
        help="Directory to save the serialized .rkmjbin file and config",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="model.rkmjbin",
        help="Target binary filename",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Sample prompt to evaluate generation latency and output",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=25,
        help="Number of new tokens to generate during inference validation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-K sampling cutoff",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length limit for RKMJ model",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of CPU OpenMP threads (0 = auto / PyTorch default)",
    )
    parser.add_argument(
        "--from_config",
        action="store_true",
        help="Instantiate model from config (synthetic weights) for rapid offline pipeline validation",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Only run download, quantization, and export; skip inference validation",
    )

    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    print_banner(f"RKMJ 1.58-BIT PTQ PIPELINE: {args.model_id}")
    print(f"Target Architecture: Hugging Face -> RKMJ LLaMA Causal LM")
    print(f"Precision Target:    1.58-bit Ternary {{-1, 0, +1}} (2-bit Bit-Packed)")
    print(f"Engine Backend:      Carry-Save Addition (CSA) OpenMP Popcount")

    # Step 1: Ingestion
    print_banner(f"STEP 1: INGESTING MODEL '{args.model_id}'")
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    hf_config = AutoConfig.from_pretrained(args.model_id)
    if args.from_config:
        print("  [Rapid Mode] Instantiating model architecture from config with synthetic weights...")
        hf_model = AutoModelForCausalLM.from_config(hf_config)
    else:
        print("  Loading pretrained weights in FP32...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    hf_model.eval()

    # Translate config
    rkmj_config = map_hf_config_to_rkmj(hf_config, max_seq_len=args.max_seq_len)
    print("  ✓ Configuration successfully extracted:")
    print(f"    - Hidden Dimension:    {rkmj_config.dim}")
    print(f"    - Intermediate Dim:    {rkmj_config.intermediate_dim}")
    print(f"    - Transformer Layers:  {rkmj_config.n_layers}")
    print(f"    - Attention Heads:     {rkmj_config.n_heads} (KV Heads: {rkmj_config.n_kv_heads})")
    print(f"    - Vocabulary Size:     {rkmj_config.vocab_size}")
    print(f"    - Max Sequence Length: {rkmj_config.max_seq_len}")

    # Step 2: Quantization & 2-Bit Packing
    rkmj_model, stats = extract_and_quantize_model(hf_model, rkmj_config)

    # Step 3: Native Serialization
    print_banner("STEP 3: NATIVE SERIALIZATION (.RKMJBIN)")
    os.makedirs(args.output_dir, exist_ok=True)
    out_bin_path = os.path.join(args.output_dir, args.output_filename)

    saved_path = rkmj_model.save_pretrained(args.output_dir, filename=args.output_filename)
    file_size = os.path.getsize(saved_path)

    # Print Comparison Table
    display_memory_comparison_table(stats, file_size)

    # Step 4: Inference Validation
    if not args.skip_inference:
        run_inference_validation(
            rkmjbin_filepath=out_bin_path,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device="cpu",
        )
    else:
        print("\n[INFO] Inference validation skipped as requested (--skip_inference).")


if __name__ == "__main__":
    main()

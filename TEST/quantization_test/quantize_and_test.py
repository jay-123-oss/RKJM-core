"""
================================================================================
RKMJ-Core External Quantization & Inference Test Script
================================================================================
Location: /home/jaydeep/Documents/rkjm-core/TEST/quantization_test/quantize_and_test.py
NOTE: This script is maintained OUTSIDE the core framework directory (`rkmj-core/`).
It imports the framework externally via sys.path.
================================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------
# 1. External Framework Import (Strictly outside rkmj-core)
# ----------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
FRAMEWORK_DIR = REPO_ROOT / "rkmj-core"

if not FRAMEWORK_DIR.exists():
    raise FileNotFoundError(f"Framework directory not found at: {FRAMEWORK_DIR}")

# Insert framework at top of sys.path
sys.path.insert(0, str(FRAMEWORK_DIR))

try:
    import rkmj
    from universal import UniversalStreamingPTQConverter, UniversalLocalRunner
    from rkmj.core import get_supervisor
    print("=" * 80)
    print(f"✅ Successfully imported RKMJ-Core Framework externally from:")
    print(f"   {FRAMEWORK_DIR}")
    print(f"   Framework Version: {rkmj.__version__}")
    print("=" * 80)
except ImportError as e:
    print(f"❌ Failed to import RKMJ framework: {e}")
    sys.exit(1)

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def run_test(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    local_dir: Optional[str] = None,
    output_bin_name: str = "qwen2.5_3b_1.58bit.rkmjbin",
    group_size: int = 64,
    ram_budget_gb: float = 3.5,
    prompt: str = "Explain the advantages of 1.58-bit ternary neural networks in 3 bullet points.",
    max_new_tokens: int = 64,
):
    print("\n" + "=" * 80)
    print(f"🎯 STEP 1: ACQUIRING 3-4B PARAMETER MODEL")
    print("=" * 80)

    output_bin_path = CURRENT_DIR / output_bin_name
    manual_dir = CURRENT_DIR / "manual_model"

    # 1. Check if user passed explicit --local-dir
    if local_dir and os.path.exists(local_dir):
        model_dir = os.path.abspath(local_dir)
        print(f"✅ Using local model directory provided via --local-dir:")
        print(f"   {model_dir}")
    # 2. Check if user manually pasted model into TEST/quantization_test/manual_model/
    elif manual_dir.exists() and (manual_dir / "config.json").exists():
        model_dir = str(manual_dir.resolve())
        print(f"✅ Found manually placed model in default directory:")
        print(f"   {model_dir}")
    # 3. Otherwise download via HuggingFace
    else:
        models_cache_dir = CURRENT_DIR / "source_weights"
        models_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Local manual folder not found. Downloading via HuggingFace: {model_id}")
        print(f"[*] Local cache directory: {models_cache_dir}")

        try:
            model_dir = snapshot_download(
                repo_id=model_id,
                local_dir=str(models_cache_dir / model_id.replace("/", "--")),
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "tokenizer*",
                ],
                ignore_patterns=[
                    "*.bin",
                    "*.pt",
                    "*.pth",
                ],
            )
            print(f"✅ Model files downloaded and verified at: {model_dir}")
        except Exception as e:
            print(f"❌ Failed to download model weights: {e}")
            return

    # ------------------------------------------------------------------
    # 2. 1.58-bit Out-of-Core Streaming Quantization
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"⚡ STEP 2: OUT-OF-CORE 1.58-BIT STREAMING QUANTIZATION")
    print(f"   Max RAM Ceiling: {ram_budget_gb:.2f} GB")
    print(f"   Group Size:      {group_size}")
    print(f"   Output Binary:   {output_bin_path}")
    print("=" * 80)

    max_ram_bytes = int(ram_budget_gb * 1024 * 1024 * 1024)

    converter = UniversalStreamingPTQConverter(
        model_dir=model_dir,
        output_path=str(output_bin_path),
        max_ram_bytes=max_ram_bytes,
        group_size=group_size,
    )

    t_quant_start = time.time()
    converter.convert()
    t_quant_duration = time.time() - t_quant_start

    bin_size_bytes = os.path.getsize(output_bin_path)
    bin_size_gb = bin_size_bytes / (1024.0 ** 3)
    bin_size_mb = bin_size_bytes / (1024.0 ** 2)

    print("\n" + "-" * 80)
    print(f"✅ Quantization Completed in {t_quant_duration:.2f} seconds!")
    print(f"   Packed .rkmjbin File Size: {bin_size_mb:.2f} MB ({bin_size_gb:.3f} GB)")
    print(f"   Peak Process RSS:         {converter.peak_rss_bytes / (1024**3):.2f} GB")
    print("-" * 80)

    # ------------------------------------------------------------------
    # 3. Model Loading & Inference Test
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"🚀 STEP 3: HIGH-SPEED CPU LOCAL INFERENCE TEST")
    print("=" * 80)

    print(f"[*] Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"[*] Initializing UniversalLocalRunner for: {output_bin_path}")
    runner = UniversalLocalRunner(
        model_path=str(output_bin_path),
        ram_budget_gb=ram_budget_gb,
    )

    supervisor = get_supervisor()
    mem_status = supervisor.get_memory_status()
    print(f"[*] Active Process Physical RSS: {mem_status['process_rss_gb']:.2f} GB")
    print(f"[*] Host Available RAM:          {mem_status['system_available_gb']:.2f} GB")

    # Format ChatML prompt if tokenizer supports chat templates
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt

    input_ids = torch.tensor([tokenizer.encode(formatted_prompt, add_special_tokens=True)])

    print(f"\nPrompt: '{prompt}'")
    print("-" * 80)
    print("Model Output:\n")

    tokens_generated = 0
    t_gen_start = time.time()

    for token_id in runner.generate(
        prompt_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    ):
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        sys.stdout.write(token_str)
        sys.stdout.flush()
        tokens_generated += 1

    t_gen_duration = time.time() - t_gen_start
    tok_per_sec = tokens_generated / max(t_gen_duration, 1e-4)

    print("\n" + "-" * 80)
    print(f"\n📊 FINAL TELEMETRY & PERFORMANCE METRICS:")
    print(f"   • Model Architecture:   {runner.profile.model_family.upper()} ({runner.num_layers} layers)")
    print(f"   • Quantized File Size:  {bin_size_mb:.2f} MB ({bin_size_gb:.3f} GB)")
    print(f"   • Generated Tokens:     {tokens_generated} tokens")
    print(f"   • Total Time:           {t_gen_duration:.2f}s")
    print(f"   • Throughput:           {tok_per_sec:.2f} tok/s")
    print(f"   • Process RAM RSS:      {supervisor.get_process_rss_gb():.2f} GB (Strictly <= {ram_budget_gb:.1f} GB)")
    print("=" * 80)

    runner.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RKMJ External 3-4B Model Quantization & Inference Test")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--local-dir", "-d", type=str, default=None, help="Path to local folder containing manually downloaded model files")
    parser.add_argument("--output-bin", type=str, default="qwen2.5_3b_1.58bit.rkmjbin", help="Output .rkmjbin filename")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--max-ram-gb", type=float, default=3.5, help="Physical RAM budget limit (default: 3.5 GB)")
    parser.add_argument("--prompt", type=str, default="Explain the advantages of 1.58-bit ternary neural networks in 3 bullet points.", help="Test prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to generate")

    args = parser.parse_args()

    run_test(
        model_id=args.model_id,
        local_dir=args.local_dir,
        output_bin_name=args.output_bin,
        group_size=args.group_size,
        ram_budget_gb=args.max_ram_gb,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

"""
Universal High-Performance CPU Local Generation CLI.
"""

import argparse
import sys
import time
import torch
from transformers import AutoTokenizer
from universal.runner import UniversalLocalRunner


def main():
    parser = argparse.ArgumentParser(description="Universal 1.58-bit Local Inference Runner")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to .rkmjbin model file")
    parser.add_argument("--tokenizer", "-t", type=str, required=True, help="Hugging Face model ID or path for tokenizer")
    parser.add_argument("--prompt", "-p", type=str, default="Explain what a 1.58-bit model is.", help="User prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--ram-budget-gb", type=float, default=3.5, help="Physical RAM budget ceiling")

    args = parser.parse_args()

    print(f"[INFO] Loading tokenizer from: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"[INFO] Initializing UniversalLocalRunner for: {args.model}")
    runner = UniversalLocalRunner(args.model, ram_budget_gb=args.ram_budget_gb)

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_special_tokens=True)])
    print(f"\n--- Generation (Prompt: '{args.prompt}') ---")

    tokens_generated = 0
    t0 = time.time()
    for token_id in runner.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        sys.stdout.write(token_str)
        sys.stdout.flush()
        tokens_generated += 1

    dt = time.time() - t0
    tok_s = tokens_generated / max(dt, 1e-4)
    print(f"\n\n--- Stats: {tokens_generated} tokens in {dt:.2f}s ({tok_s:.2f} tok/s) ---")
    runner.close()


if __name__ == "__main__":
    main()

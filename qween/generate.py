"""
CLI Tool: Local Autoregressive Generation for Qwen 1.58-bit Models within < 3.5 GB RAM.
Usage:
    python qween/generate.py --model models/qwen-0.5b-ternary.rkmjbin --prompt "Artificial intelligence is" --max-tokens 32
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 1. Auto-detect project virtualenv if global python without torch is invoked
try:
    import torch
except ImportError:
    for candidate in [
        Path(__file__).resolve().parent.parent / "myenv" / "bin" / "python",
        Path(__file__).resolve().parent.parent.parent / "myenv" / "bin" / "python",
    ]:
        if candidate.exists() and sys.executable != str(candidate):
            os.execv(str(candidate), [str(candidate)] + sys.argv)

# Bootstrap sys.path to allow running from both repo root and rkmj-core
_current_dir = Path(__file__).resolve().parent
_rkmj_core_dir = _current_dir.parent if (_current_dir.parent / "rkmj").exists() else _current_dir.parent / "rkmj-core"

for p in [str(_current_dir.parent), str(_rkmj_core_dir), str(_current_dir)]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch
from qween.bootstrap import check_dependencies, ensure_dependencies
from qween.runner import QwenLocalRunner


def main():
    # Bootstrap check
    _missing, _ = check_dependencies()
    if _missing:
        ensure_dependencies(auto_install=True)

    parser = argparse.ArgumentParser(
        description="Local High-Throughput Inference Runner for Qwen Architecture under <= 3.5 GB RAM"
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to .rkmjbin quantized Qwen model",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="Artificial intelligence is",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty factor (default: 1.15)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not apply chat template for instruct models (use raw prompt text)",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default=None,
        help="Path to tokenizer directory or Hugging Face repo ID",
    )

    args = parser.parse_args()

    print(f"Loading Qwen model from: {args.model}")
    runner = QwenLocalRunner(args.model)
    print(f"Model loaded. Config: {runner.num_layers} layers, dim={runner.dim}, vocab={runner.vocab_size}")

    # Resolve tokenizer location
    tokenizer_path = args.tokenizer
    if not tokenizer_path:
        # Priority 1: Check standard local instruct weights
        preferred_local = "/home/jaydeep/Documents/rkjm-core/weights/Qwen2.5-0.5B-Instruct"
        if os.path.isdir(preferred_local) and os.path.exists(os.path.join(preferred_local, "tokenizer.json")):
            tokenizer_path = preferred_local
        else:
            src_dir = runner.config.get("source_model_dir")
            if src_dir and os.path.isdir(src_dir) and os.path.exists(os.path.join(src_dir, "tokenizer.json")):
                tokenizer_path = src_dir
            elif os.path.isdir("weights"):
                for sub in os.listdir("weights"):
                    candidate = os.path.join("weights", sub)
                    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "tokenizer.json")):
                        tokenizer_path = candidate
                        break

    if not tokenizer_path:
        tokenizer_path = "Qwen/Qwen2.5-0.5B-Instruct"

    # Tokenizer loading
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        prompt_str = args.prompt
        # Format with chat template for instruct models if not raw and no special tags
        if not args.raw and hasattr(tokenizer, "chat_template") and tokenizer.chat_template and "<|im_start|>" not in prompt_str:
            messages = [{"role": "user", "content": prompt_str}]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False)
        else:
            input_ids = tokenizer.encode(prompt_str, return_tensors="pt")

        decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [151645, 151643]
        if hasattr(tokenizer, "get_vocab") and "<|im_end|>" in tokenizer.get_vocab():
            eos_ids.append(tokenizer.get_vocab()["<|im_end|>"])
    except Exception as exc:
        print(f"[WARN] Failed to load tokenizer from '{tokenizer_path}': {exc}. Using byte encoder fallback.")
        raw_bytes = args.prompt.encode("utf-8")
        input_ids = torch.tensor([[b for b in raw_bytes][:128]], dtype=torch.long)
        decode_fn = lambda ids: bytes([t % 256 for t in ids]).decode("utf-8", errors="replace")
        eos_ids = [0]

    print(f"\nPrompt: {args.prompt}")
    print("Generating: ", end="", flush=True)

    t0 = time.perf_counter()
    tokens_generated = 0
    token_list = []

    try:
        for token_id in runner.generate(
            prompt_ids=input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_ids=eos_ids,
        ):
            token_list.append(token_id)
            tokens_generated += 1
            chunk_str = decode_fn([token_id])
            print(chunk_str, end="", flush=True)

        elapsed = time.perf_counter() - t0
        tps = tokens_generated / max(elapsed, 1e-5)
        print(f"\n\nDone! Generated {tokens_generated} tokens in {elapsed:.2f}s ({tps:.2f} tokens/s).")
    finally:
        runner.close()


if __name__ == "__main__":
    main()

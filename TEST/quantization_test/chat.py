"""
================================================================================
RKMJ-Core External Interactive Chat Script
================================================================================
Location: TEST/quantization_test/chat.py
Loads quantized .rkmjbin binary and starts an interactive multi-turn chat session.
================================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# Bootstrap environment paths
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
FRAMEWORK_DIR = REPO_ROOT / "rkmj-core"

if not FRAMEWORK_DIR.exists():
    raise FileNotFoundError(f"Framework directory not found at: {FRAMEWORK_DIR}")

sys.path.insert(0, str(FRAMEWORK_DIR))

try:
    import rkmj
    from universal import UniversalLocalRunner
    from rkmj.core import get_supervisor
except ImportError as e:
    print(f"❌ Failed to import RKMJ framework: {e}")
    sys.exit(1)

import torch
from transformers import AutoTokenizer


def interactive_chat(
    model_path: str,
    tokenizer_path: str,
    ram_budget_gb: float = 3.5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    max_new_tokens: int = 256,
    system_prompt: str = "You are a helpful, concise, and intelligent AI assistant.",
):
    print("=" * 80)
    print("🤖 RKMJ-Core 1.58-Bit Interactive Chat")
    print(f"   Framework Version: {rkmj.__version__}")
    print(f"   Model Path:        {model_path}")
    print(f"   Tokenizer Source:  {tokenizer_path}")
    print(f"   RAM Budget Limit:  {ram_budget_gb:.1f} GB")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    # 1. Load Tokenizer
    print("\n[*] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    # 2. Load Model Runner
    print("[*] Loading model into RKMJ adaptive loader...")
    t0 = time.time()
    try:
        runner = rkmj.load(
            model_path=model_path,
            ram_budget_gb=ram_budget_gb,
        )
        print(f"✅ Model loaded in {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    supervisor = get_supervisor()
    mem_status = supervisor.get_memory_status()
    print(f"[*] Memory RSS: {mem_status['process_rss_gb']:.2f} GB / Available Host: {mem_status['system_available_gb']:.2f} GB")

    # Determine EOS tokens
    eos_ids = [151645, 151643]  # <|im_end|>, <|endoftext|>
    if tokenizer.eos_token_id and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)

    print("\n" + "=" * 80)
    print("💬 Chat session started! (Commands: 'exit' or 'quit' to stop, 'clear' to reset context)")
    print("=" * 80 + "\n")

    history: List[Dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    try:
        while True:
            try:
                user_input = input("\n👤 You > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Exiting chat...")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Bye!")
                break

            if user_input.lower() in ("clear", "reset"):
                history = []
                if system_prompt:
                    history.append({"role": "system", "content": system_prompt})
                print("🧹 Conversation context cleared.")
                continue

            # Append user message
            history.append({"role": "user", "content": user_input})

            # Format prompt using chat template
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    history,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = f"System: {system_prompt}\n" if system_prompt else ""
                for msg in history:
                    formatted_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
                formatted_prompt += "Assistant: "

            input_ids = torch.tensor([tokenizer.encode(formatted_prompt, add_special_tokens=True)])

            print("\n🤖 Assistant > ", end="", flush=True)

            tokens_generated = 0
            response_tokens: List[int] = []
            t_start = time.time()

            for token_id in runner.generate(
                prompt_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_ids=eos_ids,
            ):
                if token_id in eos_ids:
                    break
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                sys.stdout.write(token_str)
                sys.stdout.flush()
                response_tokens.append(token_id)
                tokens_generated += 1

            t_elapsed = time.time() - t_start
            speed = tokens_generated / max(t_elapsed, 1e-4)

            # Store assistant response in history
            full_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            history.append({"role": "assistant", "content": full_response})

            rss = supervisor.get_process_rss_gb()
            print(f"\n\n\033[90m[{tokens_generated} tokens | {speed:.2f} tok/s | RAM: {rss:.2f} GB]\033[0m")

    finally:
        runner.close()


if __name__ == "__main__":
    import argparse

    default_model = str(CURRENT_DIR / "qwen2.5_3b_1.58bit.rkmjbin")
    default_tokenizer = "/home/jaydeep/Downloads/Qwen2.5-3B"
    if not os.path.exists(default_tokenizer):
        default_tokenizer = "Qwen/Qwen2.5-3B-Instruct"

    parser = argparse.ArgumentParser(description="RKMJ-Core 1.58-Bit Interactive Chat")
    parser.add_argument("--model", "-m", type=str, default=default_model, help="Path to .rkmjbin model")
    parser.add_argument("--tokenizer", "-t", type=str, default=default_tokenizer, help="Path to tokenizer directory or HF repo")
    parser.add_argument("--ram-budget", type=float, default=2.0, help="Physical RAM budget limit (GB, default: 2.0)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum new tokens per turn")
    parser.add_argument("--system", type=str, default="You are a helpful, concise, and intelligent AI assistant.", help="System prompt")

    args = parser.parse_args()

    interactive_chat(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        ram_budget_gb=args.ram_budget,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_tokens,
        system_prompt=args.system,
    )

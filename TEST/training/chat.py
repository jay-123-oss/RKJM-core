"""
================================================================================
RKMJ-Core Trained 1.58-Bit Interactive Text-to-Text Chat
================================================================================
Location: TEST/training/chat.py
Loads the native 1.58-bit model trained with C++ STE autograd and runs
interactive conversational generation with streaming output.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import torch
from transformers import AutoTokenizer

# Bootstrap paths
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
FRAMEWORK_DIR = REPO_ROOT / "rkmj-core"

if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import rkmj
from model import ModelConfig, RKMJ158CausalLM


def chat(
    checkpoint_path: str = "TEST/training/rkmj_158_trained.pt",
    tokenizer_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    max_new_tokens: int = 128,
    device: str = "auto",
):
    # ==============================================================================
    # 🚀 Hardware Device Auto-Detection (CUDA GPU vs Multi-threaded CPU)
    # ==============================================================================
    if device in ("auto", None):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    if target_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        device_desc = f"NVIDIA CUDA ({torch.cuda.get_device_name(target_device)})"
    else:
        CPU_CORES = os.cpu_count() or 4
        torch.set_num_threads(CPU_CORES)
        torch.set_num_interop_threads(2)
        device_desc = f"CPU ({CPU_CORES} Compute Threads | 2 Interop Threads)"

    print("=" * 80)
    print("🤖 RKMJ-Core 1.58-Bit Trained Model Interactive Chat")
    print(f"   Framework Version: {rkmj.__version__}")
    print(f"   Compute Device:    {device_desc}")
    print(f"   Checkpoint:        {checkpoint_path}")
    print(f"   Tokenizer:         {tokenizer_path}")
    print(f"   Sampling:          top-k={top_k}, top-p={top_p}, temp={temperature}")
    print("=" * 80)

    full_ckpt_path = REPO_ROOT / checkpoint_path
    if not full_ckpt_path.exists():
        print(f"❌ Checkpoint file not found: {full_ckpt_path}")
        print("💡 Please train the model first by running:")
        print("   ./myenv/bin/python TEST/training/train.py")
        return

    # 1. Load Tokenizer
    print("\n[*] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Could not load {tokenizer_path}: {e}. Falling back to TinyLlama 32k...")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"✅ Tokenizer loaded successfully (Vocab: {len(tokenizer):,}).")

    # 2. Checkpoint Loading (PyTorch 2.6+ safe unpickling compatibility)
    print(f"\n[*] Loading model weights into RKMJ158CausalLM on {target_device}...")
    t0 = time.time()
    try:
        torch.serialization.add_safe_globals([ModelConfig])
    except Exception:
        pass

    checkpoint = torch.load(str(full_ckpt_path), map_location=target_device, weights_only=False)
    cfg: ModelConfig = checkpoint["config"]
    model = RKMJ158CausalLM(cfg).to(target_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    unique_params = sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())
    print(f"✅ Model loaded in {time.time() - t0:.2f}s (Architecture: {cfg.n_layers}L | {cfg.dim}D | {cfg.n_heads}H | {unique_params:,} Params).")

    print("\n" + "=" * 80)
    print("💬 Interactive Text-to-Text session started! Type your prompt or query.")
    print("   (Commands: 'exit' or 'quit' to stop, 'clear' to reset)")
    print("=" * 80 + "\n")

    # 3. Comprehensive Stop / Termination Tokens
    stop_token_ids: Set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)

    # Convert special delimiters to IDs if available in tokenizer vocabulary
    for tag in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"]:
        tid = tokenizer.convert_tokens_to_ids(tag)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_token_ids.add(tid)

    # Standard fallback EOS IDs across LLaMA / Qwen families
    stop_token_ids.update([2, 151645, 151643])

    while True:
        try:
            user_input = input("\n👤 You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 Bye!")
            break

        # Format with standard chat delimiters
        if "<|im_start|>" not in user_input:
            formatted_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = user_input

        input_ids = torch.tensor([tokenizer.encode(formatted_prompt, add_special_tokens=False)], device=target_device)

        print("\n🤖 1.58-Bit AI > ", end="", flush=True)

        tokens_generated = 0
        t_start = time.time()
        generated_token_ids: List[int] = []
        printed_len = 0
        full_decoded_text = ""

        for token_id in model.generate(
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_ids=list(stop_token_ids),
        ):
            # Immediate termination on stop token
            if token_id in stop_token_ids:
                break

            generated_token_ids.append(token_id)
            tokens_generated += 1

            # Incremental space-preserving decoding for BPE / SentencePiece
            current_text = tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            # Prevent role-switch hallucinations
            if "<|im_end|>" in current_text or "<|im_start|>" in current_text:
                # Stop immediately before printing the delimiter
                break

            if len(current_text) > printed_len:
                diff = current_text[printed_len:]
                sys.stdout.write(diff)
                sys.stdout.flush()
                printed_len = len(current_text)
                full_decoded_text = current_text

        # Strip any trailing special tags or stray whitespace
        cleaned_tail = re.sub(r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|</s>)", "", full_decoded_text[printed_len:]).rstrip()
        if cleaned_tail:
            sys.stdout.write(cleaned_tail)
            sys.stdout.flush()

        t_elapsed = time.time() - t_start
        speed = tokens_generated / max(t_elapsed, 1e-4)
        print(f"\n\n\033[90m[{tokens_generated} tokens generated in {t_elapsed:.2f}s | {speed:.2f} tok/s]\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKMJ-Core 1.58-Bit Interactive Chat")
    parser.add_argument("--checkpoint", type=str, default="TEST/training/rkmj_158_trained.pt")
    parser.add_argument("--tokenizer", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=128)

    args = parser.parse_args()

    chat(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        device=args.device,
    )

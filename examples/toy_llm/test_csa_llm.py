"""
Interactive Testing and Prompt Evaluation for 1.58-bit CSALanguageModel.

Features:
1. Loads trained weights from csa_toy_llm.pt.
2. Supports interactive terminal prompt loop (type your prompt and watch it complete!).
3. Demonstrates real-time token streaming.
4. Allows toggling 2-bit packed weight inference mode (csa_linear_cpu popcount engine).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from train_toy_csa_llm import CSALanguageModel, CharTokenizer
from watch_nn import CSALinear


def load_model(checkpoint_path: str = "csa_toy_llm.pt", device: torch.device = torch.device("cpu")):
    """Load trained CSALanguageModel and Tokenizer from disk."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found! "
            f"Please run 'myenv/bin/python train_toy_csa_llm.py' first to train the model."
        )

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Recreate tokenizer from saved vocabulary
    vocab_text = "".join(ckpt["vocab"])
    tokenizer = CharTokenizer(vocab_text)

    cfg = ckpt["config"]
    model = CSALanguageModel(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded successfully! ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model, tokenizer, cfg


@torch.no_grad()
def stream_generate(
    model: CSALanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    device: torch.device = torch.device("cpu"),
):
    """
    Autoregressively generates text character-by-character with live streaming output.
    """
    model.eval()

    # Encode prompt (fallback to newline if empty)
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [tokenizer.stoi.get("\n", 0)]

    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    # Print prompt without newline
    sys.stdout.write(prompt)
    sys.stdout.flush()

    t0 = time.perf_counter()
    tokens_generated = 0

    for _ in range(max_new_tokens):
        # Crop context to model block size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]

        # Forward pass
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Last token logits

        if temperature > 0.0:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
        next_char = tokenizer.decode([idx_next.item()])
        sys.stdout.write(next_char)
        sys.stdout.flush()
        tokens_generated += 1

    elapsed = time.perf_counter() - t0
    sys.stdout.write("\n")
    speed = tokens_generated / elapsed if elapsed > 0 else 0
    print(f"\n[Generated {tokens_generated} tokens in {elapsed:.2f}s (~{speed:.1f} tokens/sec)]")


def main():
    parser = argparse.ArgumentParser(description="Test and prompt the 1.58-bit CSA Toy LLM")
    parser.add_argument("--checkpoint", type=str, default="csa_toy_llm.pt", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt string")
    parser.add_argument("--tokens", type=int, default=250, help="Number of characters to generate")
    parser.add_argument("--temp", type=float, default=0.75, help="Sampling temperature (0.0 to 1.5)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling cutoff")
    parser.add_argument("--pack", action="store_true", help="Pack weights into 2-bit bitfields for inference")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tokenizer, cfg = load_model(args.checkpoint, device)

    if args.pack:
        print("Packing CSA ternary weights into 2-bit uint32 words for inference...")
        for m in model.modules():
            if isinstance(m, CSALinear):
                m.pack_weights_for_inference()
        print("  ✓ 2-bit weight packing active!")

    # If a prompt was provided via CLI, run it once
    if args.prompt is not None:
        print(f"\nGenerating from prompt: {repr(args.prompt)}")
        print("-" * 60)
        stream_generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        print("-" * 60)
        return

    # Preset demonstration prompts
    demo_prompts = [
        "ROMEO: ",
        "KING: ",
        "To be or not to be",
        "JULIET: ",
    ]

    print("\n" + "=" * 70)
    print("DEMO PROMPT TESTING (1.58-bit CSA Transformer Block)")
    print("=" * 70)

    for p in demo_prompts:
        print(f"\n>>> Prompt: {repr(p)}")
        print("-" * 60)
        stream_generate(
            model,
            tokenizer,
            prompt=p,
            max_new_tokens=150,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        print("-" * 60)

    # Interactive Loop
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE: Type any prompt (or 'exit' / 'quit' to stop):")
    print("=" * 70)

    try:
        while True:
            user_input = input("\nEnter Prompt > ")
            if not user_input or user_input.lower().strip() in ("exit", "quit", "q"):
                print("Exiting interactive test.")
                break
            print("-" * 60)
            stream_generate(
                model,
                tokenizer,
                prompt=user_input,
                max_new_tokens=args.tokens,
                temperature=args.temp,
                top_k=args.top_k,
                device=device,
            )
            print("-" * 60)
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")


if __name__ == "__main__":
    main()

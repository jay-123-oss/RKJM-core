"""
Interactive CLI Chat and Streaming Text Generation with RKMJ-Core.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.engine.generator import TextGenerator
from rkmj.models.llama import RKMJLlamaForCausalLM
from rkmj.models.config import RKMJConfig


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with RKMJ-Core 1.58-bit LLM")
    parser.add_argument("--model_dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--model_file", type=str, default="shakespeare_model.rkmjbin")
    parser.add_argument("--pack", action="store_true", help="Use 2-bit packed weights for fast CPU inference")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_file)
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        print("Please train a model first: python examples/train_shakespeare.py")
        sys.exit(1)

    print("Loading RKMJ 1.58-bit model...")
    model = RKMJLlamaForCausalLM.from_pretrained(args.model_dir, args.model_file)

    if args.pack:
        print("Packing weights into 2-bit uint32 bitfields (15.8x RAM reduction)...")
        model.pack_for_inference()
        print("  ✓ Hardware Popcount CSA engine active!")

    # Simple character vocabulary fallback
    vocab_path = os.path.join(args.model_dir, "input.txt")
    if os.path.exists(vocab_path):
        chars = sorted(list(set(open(vocab_path, "r", encoding="utf-8").read())))
    else:
        chars = [chr(i) for i in range(128)]

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode_fn = lambda s: [stoi[c] for c in s if c in stoi]
    decode_fn = lambda ids: "".join([itos[i] for i in ids if i in itos])

    generator = TextGenerator(model, encode_fn, decode_fn)

    print("\n" + "=" * 70)
    print("RKMJ-CORE INTERACTIVE CHAT (Type 'quit' or 'exit' to stop)")
    print("=" * 70)

    while True:
        try:
            prompt = input("\nYou > ")
            if not prompt or prompt.lower().strip() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            print("\nRKMJ > ", end="")
            list(generator.stream_generate(prompt, max_new_tokens=args.tokens, temperature=args.temp))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

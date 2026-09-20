"""
End-to-End 1.58-bit LLM Training on Tiny Shakespeare using RKMJ-Core.
"""

from __future__ import annotations

import math
import os
import sys
import time
import urllib.request
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure rkmj is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.models.config import RKMJConfig
from rkmj.models.llama import RKMJLlamaForCausalLM
from rkmj.nn.linear import CSALinear

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = os.path.join(os.path.dirname(__file__), "input.txt")


def get_dataset() -> str:
    if not os.path.exists(DATA_FILE):
        # Also check workspace root
        root_input = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../input.txt"))
        if os.path.exists(root_input):
            with open(root_input, "r", encoding="utf-8") as f:
                return f.read()
        print(f"Downloading Tiny Shakespeare to {DATA_FILE}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, indices: list[int]) -> str:
        return "".join([self.itos[i] for i in indices if i in self.itos])


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, iters=20):
    model.eval()
    out = {}
    for split, d in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(d, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print("=" * 75)
    print("RKMJ-CORE: TRAINING 1.58-BIT LLAMA MODEL ON TINY SHAKESPEARE")
    print("=" * 75)

    device = torch.device("cpu")
    torch.manual_seed(42)

    raw_text = get_dataset()
    tokenizer = CharTokenizer(raw_text)
    data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    batch_size = 16
    block_size = 64
    max_iters = 500
    learning_rate = 1e-3

    config = RKMJConfig(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        intermediate_dim=384,
        max_seq_len=block_size,
    )

    model = RKMJLlamaForCausalLM(config).to(device)
    params = model.count_parameters()
    print(f"Model Summary:")
    print(f"  - Total Parameters:              {params['total']:,}")
    print(f"  - 1.58-bit CSA Parameters:       {params['csa_1_58_bit']:,} ({params['csa_ratio_pct']:.1f}%)")

    # Set alpha.requires_grad = False to allow dynamic weight magnitude tracking
    for m in model.modules():
        if isinstance(m, CSALinear):
            m.alpha.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

    # Initial Untrained Generation
    print("\n" + "-" * 75)
    print("SAMPLE TEXT GENERATION (ITERATION 0 - UNTRAINED):")
    print("-" * 75)
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    untrained_out = model.generate(prompt, max_new_tokens=150, temperature=0.8)
    print(tokenizer.decode(untrained_out[0].tolist()))
    print("-" * 75)

    # Training Loop
    start_time = time.perf_counter()
    print("\nTraining in progress...")

    for iter_num in range(1, max_iters + 1):
        lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * iter_num / max_iters))
        lr = max(lr, 1e-4)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        # Update dynamic alpha
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, CSALinear):
                    m.update_alpha()

        if iter_num % 100 == 0 or iter_num == max_iters:
            elapsed = time.perf_counter() - start_time
            ms_per_step = (elapsed / iter_num) * 1000.0
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, device)
            print(
                f"[Step {iter_num:3d} / {max_iters}] "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"Speed: {ms_per_step:.1f} ms/step"
            )

    print(f"\nTraining completed in {time.perf_counter() - start_time:.2f}s.")

    # Export to .rkmjbin
    bin_path = os.path.join(os.path.dirname(__file__), "shakespeare_model.rkmjbin")
    model.save_pretrained(os.path.dirname(bin_path), "shakespeare_model.rkmjbin")

    # Sample Trained Generation
    print("\n" + "=" * 75)
    print("SAMPLE TEXT GENERATION (ITERATION 500 - TRAINED 1.58-BIT MODEL):")
    print("=" * 75)
    prompt_text = "ROMEO:\n"
    prompt_toks = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
    trained_out = model.generate(prompt_toks, max_new_tokens=200, temperature=0.7)
    print(tokenizer.decode(trained_out[0].tolist()))
    print("=" * 75)


if __name__ == "__main__":
    main()

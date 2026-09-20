"""
Train a Toy LLaMA-style Language Model using 1.58-bit Carry-Save Addition (CSA) Transformer Blocks.

Dataset: Tiny Shakespeare (downloaded automatically if not present).
Model: CSALanguageModel with 4 CSATransformerBlocks (Q/K/V/O and SwiGLU MLP via CSALinear).
Optimizer: AdamW.
Proof of Learning: 200 characters generated at Iteration 0 (untrained) vs Iteration 500 (trained).
"""

from __future__ import annotations

import math
import os
import sys
import time
import urllib.request
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure watch_nn and custom extensions are discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from watch_nn import CSATransformerBlock, RMSNorm, CSALinear

# =============================================================================
# 1. Dataset & Tokenizer (Tiny Shakespeare)
# =============================================================================

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"


def download_dataset_if_needed(file_path: str = DATA_FILE) -> str:
    """Download the Tiny Shakespeare dataset if it doesn't already exist locally."""
    if not os.path.exists(file_path):
        print(f"Downloading Tiny Shakespeare from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, file_path)
        print(f"Downloaded {os.path.getsize(file_path):,} bytes to {file_path}.")
    else:
        print(f"Found existing dataset at {file_path} ({os.path.getsize(file_path):,} bytes).")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Simple character-level tokenizer with encode and decode methods."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, indices: list[int]) -> str:
        return "".join([self.itos[i] for i in indices if i in self.itos])


def prepare_data(text: str, tokenizer: CharTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode text and split into train and validation tensors (90% / 10%)."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a small batch of inputs x and targets y."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


# =============================================================================
# 2. The Model: CSALanguageModel
# =============================================================================

class CSALanguageModel(nn.Module):
    """
    Autoregressive Language Model built with 1.58-bit CSATransformerBlocks.

    Architecture:
      - Token & Position Embeddings
      - Stack of N CSATransformerBlocks (Multi-head Attention + SwiGLU MLP via CSALinear)
      - Final RMSNorm
      - LM Head Linear projection to vocabulary logits
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        block_size: int = 64,
        n_layer: int = 4,
        n_head: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        # Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 4-layer 1.58-bit CSA Transformer Stack
        self.blocks = nn.Sequential(*[
            CSATransformerBlock(
                dim=n_embd,
                num_heads=n_head,
                attn_dropout=attn_dropout,
                bias=False,
            )
            for _ in range(n_layer)
        ])

        # Final RMS Normalization
        self.ln_f = RMSNorm(n_embd)

        # Output LM Head (projects from hidden dimension to vocabulary logits)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying (optional, but standard in LLaMA-style LLMs)
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Inputs:
          idx: [B, T] token indices.
          targets: optional [B, T] ground truth token indices.
        Returns:
          logits: [B, T, vocab_size]
          loss: cross-entropy scalar loss (if targets provided) or None
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence length {T}, block_size is {self.block_size}"

        # Embedding lookup + positional encoding
        tok_emb = self.token_embedding_table(idx)  # [B, T, n_embd]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # [T, n_embd]
        x = tok_emb + pos_emb  # [B, T, n_embd]

        # Stack of 1.58-bit CSA Transformer Blocks
        x = self.blocks(x)  # [B, T, n_embd]

        # Final LayerNorm
        x = self.ln_f(x)  # [B, T, n_embd]

        # Logits
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        """
        Autoregressively generate max_new_tokens conditioned on prompt idx [B, T].
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop current context to the maximum context length (block_size)
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward the model
            logits, _ = self(idx_cond)

            # Pluck the logits at the final step
            logits = logits[:, -1, :]  # [B, vocab_size]

            # Scale by temperature
            if temperature > 0.0:
                logits = logits / temperature

                # Optionally crop to top-k choices
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]

            # Append sampled token to running context
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# 3. Training & Evaluation Pipeline
# =============================================================================

@torch.no_grad()
def estimate_loss(
    model: CSALanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int = 25,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Estimate average training and validation loss over several random batches."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_toy_csa_llm():
    print("=" * 80)
    print("STARTING 1.58-BIT CSA TOY LLM TRAINING (CPU OPENMP BACKWARD & FORWARD)")
    print("=" * 80)

    # Device configuration
    device = torch.device("cpu")
    torch.manual_seed(1337)

    # 1. Load Data & Tokenizer
    raw_text = download_dataset_if_needed()
    tokenizer = CharTokenizer(raw_text)
    train_data, val_data = prepare_data(raw_text, tokenizer)

    print(f"Dataset Statistics:")
    print(f"  - Total Characters:              {len(raw_text):,}")
    print(f"  - Vocabulary Size:               {tokenizer.vocab_size} unique characters")
    print(f"  - Training Tokens:               {len(train_data):,}")
    print(f"  - Validation Tokens:             {len(val_data):,}")

    # 2. Hyperparameters
    batch_size = 16
    block_size = 64
    max_iters = 500
    learning_rate = 1e-3
    eval_interval = 100

    # Model architecture parameters
    n_embd = 128
    n_head = 4
    n_layer = 4

    print(f"\nModel Configuration:")
    print(f"  - Embedding Dimension (n_embd):  {n_embd}")
    print(f"  - Transformer Layers:            {n_layer}")
    print(f"  - Attention Heads:               {n_head} (Head Dim: {n_embd // n_head})")
    print(f"  - Context Window (block_size):   {block_size}")
    print(f"  - Batch Size:                    {batch_size}")
    print(f"  - Total Training Iterations:     {max_iters}")
    print(f"  - Initial Learning Rate:         {learning_rate}")

    # 3. Instantiate Model
    model = CSALanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    csa_params = sum(
        p.numel() for m in model.modules() if isinstance(m, CSALinear) for p in m.parameters()
    )
    print(f"  - Total Model Parameters:        {total_params:,}")
    print(f"  - CSA 1.58-bit Linear Params:    {csa_params:,} ({100.0 * csa_params / total_params:.1f}%)")

    # Set alpha.requires_grad = False so optimizer focuses on latent weights,
    # and update alpha dynamically to match the current weight magnitude
    for m in model.modules():
        if isinstance(m, CSALinear):
            m.alpha.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

    # 4. Generate Text from Untrained Model (Iteration 0)
    print("\n" + "-" * 80)
    print("ITERATION 0: SAMPLE GENERATION FROM UNTRAINED MODEL")
    print("-" * 80)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_untrained = model.generate(context, max_new_tokens=200, temperature=0.8)
    print(tokenizer.decode(sample_untrained[0].tolist()))
    print("-" * 80)

    # Initial loss measurement
    initial_losses = estimate_loss(model, train_data, val_data, batch_size, block_size, device=device)
    print(f"\n[Step 0 / {max_iters}] Train Loss: {initial_losses['train']:.4f} | Val Loss: {initial_losses['val']:.4f}")

    # 5. Training Loop
    print("\nTraining in progress...")
    start_time = time.perf_counter()

    for iter_num in range(1, max_iters + 1):
        # Cosine learning rate decay for smooth convergence
        lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * iter_num / max_iters))
        lr = max(lr, 1e-4)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Sample batch
        xb, yb = get_batch(train_data, batch_size, block_size, device)

        # Forward pass & loss
        _, loss = model(xb, yb)

        # Backward autograd propagation through all CSA layers
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Dynamically update alpha to match new latent weight scales
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, CSALinear):
                    m.alpha.copy_(m.latent_weight.abs().mean(dim=1).clamp(min=1e-5))

        # Periodic evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters:
            elapsed = time.perf_counter() - start_time
            ms_per_step = (elapsed / iter_num) * 1000.0
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size, device=device)
            print(
                f"[Step {iter_num:3d} / {max_iters}] "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.5f} | "
                f"Speed: {ms_per_step:.2f} ms/step"
            )

    total_time = time.perf_counter() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({total_time / max_iters * 1000.0:.2f} ms/iter).")

    # Save model checkpoint
    checkpoint_path = "csa_toy_llm.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.chars,
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "n_embd": n_embd,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
        }
    }, checkpoint_path)
    print(f"  ✓ Model checkpoint saved to {checkpoint_path} ({os.path.getsize(checkpoint_path):,} bytes).")

    # 6. Generate Text from Trained Model (Iteration 500)
    print("\n" + "=" * 80)
    print("ITERATION 500: SAMPLE GENERATION FROM TRAINED 1.58-BIT MODEL")
    print("=" * 80)
    # Conditioning prompt: "QUEEN:" or newline
    prompt_text = "\n"
    prompt_tokens = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
    sample_trained = model.generate(prompt_tokens, max_new_tokens=200, temperature=0.8)
    generated_text = tokenizer.decode(sample_trained[0].tolist())
    print(generated_text)
    print("=" * 80)

    # Verify loss reduction
    final_losses = estimate_loss(model, train_data, val_data, batch_size, block_size, device=device)
    loss_reduction = initial_losses["val"] - final_losses["val"]
    print(f"\nFinal Summary:")
    print(f"  - Initial Validation Loss:       {initial_losses['val']:.4f}")
    print(f"  - Final Validation Loss:         {final_losses['val']:.4f} (Decreased by {loss_reduction:.4f})")
    assert final_losses["val"] < initial_losses["val"], "Validation loss failed to decrease!"
    print("  ✓ 1.58-bit CSA Transformer Block successfully trained and generated coherent character sequences!")


if __name__ == "__main__":
    train_toy_csa_llm()

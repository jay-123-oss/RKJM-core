"""
End-to-End Multimodal Vision Transformer (ViT) Training on CIFAR-10 using RKMJ-Core.
Validates the 1.58-bit ternary CSALinear engine on continuous 2D spatial gradients
with bidirectional self-attention (causal_mask=False) and custom C++ OpenMP STE autograd.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Ensure rkmj-core is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
for candidate in [
    os.path.join(current_dir, ".."),
    os.path.join(current_dir, "..", "rkmj-core"),
    os.path.join(current_dir, "rkmj-core"),
    os.path.abspath(os.path.join(current_dir, "../..", "rkmj-core")),
    "/home/jaydeep/Documents/cuda2d/rkmj-core",
]:
    cand = os.path.abspath(candidate)
    if os.path.exists(os.path.join(cand, "rkmj")) and cand not in sys.path:
        sys.path.insert(0, cand)

from rkmj.nn.block import CSATransformerBlock
from rkmj.nn.linear import CSALinear
from rkmj.nn.norm import RMSNorm


class PatchEmbedding(nn.Module):
    """
    Boundary Conv2d patch embedding layer mapping 32x32 RGB images
    into 64 spatial patch embeddings of dimension embed_dim.
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 4, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class RKMJVisionTransformer(nn.Module):
    """
    RKMJ 1.58-bit Vision Transformer (RKMJ-ViT) for CIFAR-10 Classification.

    Architecture:
      1. Boundary Conv2d Patch Embedding (32x32 -> 64 patches of dim 128).
      2. Learnable [CLS] token prepended to sequence (seq_len = 65).
      3. Learnable 1D Position Embeddings.
      4. depth=4 layers of CSATransformerBlock with bidirectional attention (causal_mask=False).
      5. Classification Head: RMSNorm -> CSALinear(128, 10).
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        intermediate_dim: int = 384,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64 patches
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # 2. [CLS] Token & Position Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # 3. 1.58-bit CSA Transformer Backbone
        self.blocks = nn.ModuleList([
            CSATransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                intermediate_dim=intermediate_dim,
                attn_dropout=attn_dropout,
                bias=False,
            )
            for _ in range(depth)
        ])

        # 4. Classification Head
        self.norm = RMSNorm(embed_dim)
        self.head = CSALinear(embed_dim, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Initialize Conv2d patch embedding
        nn.init.kaiming_normal_(self.patch_embed.proj.weight, mode="fan_out", nonlinearity="relu")
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding: [B, 64, embed_dim]
        x = self.patch_embed(x)

        # Prepend [CLS] token: [B, 65, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encodings
        x = x + self.pos_embed

        # 1.58-bit Transformer backbone with bidirectional attention (causal_mask=False)
        for block in self.blocks:
            x = block(x, causal_mask=False)

        # Extract [CLS] token representation: [B, embed_dim]
        cls_out = x[:, 0]

        # Classification Head
        cls_out = self.norm(cls_out)
        logits = self.head(cls_out)

        return logits


def get_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares standard CIFAR-10 train and validation DataLoaders."""
    # CIFAR-10 standard normalization statistics
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates validation loss and Top-1 accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / max(1, total)
    top1_acc = 100.0 * correct / max(1, total)
    return avg_loss, top1_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
):
    """Executes the complete training loop with step logging and validation."""
    print("=" * 80)
    print("  RKMJ-CORE 1.58-BIT VISION TRANSFORMER (ViT) CIFAR-10 TRAINING")
    print("=" * 80)
    print(f"Device:                {device}")
    print(f"CPU Threads:           {torch.get_num_threads()}")
    print(f"Epochs:                {epochs}")
    print(f"Training Batches:      {len(train_loader)} (Batch Size: {train_loader.batch_size})")
    print(f"Validation Batches:    {len(val_loader)}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        t_epoch_start = time.perf_counter()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, (images, targets) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)

            # Backpropagation through C++ STE OpenMP engine
            loss.backward()

            # Gradient clipping to ensure numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Synchronize dynamic alpha scaling for all CSALinear layers
            for m in model.modules():
                if isinstance(m, CSALinear):
                    m.update_alpha()

            running_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            running_total += targets.size(0)

            if step % 100 == 0 or step == len(train_loader):
                batch_loss = running_loss / running_total
                batch_acc = 100.0 * running_correct / running_total
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | Step [{step:03d}/{len(train_loader):03d}] "
                    f"| Loss: {batch_loss:.4f} | Batch Acc: {batch_acc:.2f}%"
                )

        epoch_time = time.perf_counter() - t_epoch_start

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print("-" * 80)
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Complete in {epoch_time:.2f}s "
            f"| Train Loss: {running_loss / running_total:.4f} | Train Acc: {100.0 * running_correct / running_total:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Top-1 Acc: {val_acc:.2f}%"
        )
        print("-" * 80)

    print("=" * 80)
    print("  TRAINING & VALIDATION PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="RKMJ-Core 1.58-bit Vision Transformer (ViT) on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and eval")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for CIFAR-10 data")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = auto)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on ('cpu' or 'cuda')")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick 1-epoch dry-run test")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)

    # Initialize model
    model = RKMJVisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        intermediate_dim=384,
    ).to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    csa_layers = sum(1 for m in model.modules() if isinstance(m, CSALinear))

    print(f"RKMJ-ViT Architecture Initialized:")
    print(f"  • Total Parameters:     {total_params:,}")
    print(f"  • Trainable Parameters: {trainable_params:,}")
    print(f"  • CSALinear Layers:     {csa_layers}")

    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    epochs = 1 if args.dry_run else args.epochs
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
    )


if __name__ == "__main__":
    main()

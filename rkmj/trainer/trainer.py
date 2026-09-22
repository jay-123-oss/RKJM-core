"""
High-Level Training & Fine-Tuning Engine for RKMJ-Core Models.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rkmj.optim.optim import AdamSTE
from rkmj.nn.layers import CSALinear


class Trainer:
    """
    High-level Trainer for native 1.58-bit model training and parameter-efficient fine-tuning on CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or AdamSTE(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Executes a single forward and backward optimization step using C++ STE."""
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(x)

        if logits.dim() == 3:
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            loss = self.criterion(logits, y)

        loss.backward()
        self.optimizer.step()

        # Synchronize dynamic scale factors alpha post-step
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, CSALinear) and m.latent_weight is not None:
                    m.alpha.copy_(m.latent_weight.abs().mean(dim=1).clamp(min=1e-5))

        return float(loss.item())

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, Any]:
        """Runs full training loop across epochs."""
        history = []
        t0 = time.time()

        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                elif isinstance(batch, dict):
                    x, y = batch["input_ids"], batch["labels"]
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                loss = self.train_step(x, y)
                total_loss += loss
                steps += 1

            avg_loss = total_loss / max(1, steps)
            history.append(avg_loss)
            if callback:
                callback(epoch, avg_loss)

        total_time = time.time() - t0
        return {
            "epochs": epochs,
            "final_loss": history[-1] if history else 0.0,
            "loss_history": history,
            "elapsed_seconds": round(total_time, 2),
        }

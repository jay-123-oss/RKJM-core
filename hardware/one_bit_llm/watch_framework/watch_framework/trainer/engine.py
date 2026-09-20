"""Trainer API for 1-bit QAT and FP32 warm-up training."""

from __future__ import annotations

import time
from typing import Dict, List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..nn.linear import WatchLinear
from ..nn.conv import WatchConv2d


class Trainer:
    """High-level training and evaluation engine for 1-bit Watch Grid models.

    Args:
        model: PyTorch module to train.
        optimizer: PyTorch optimizer (e.g. AdamW or SGD).
        criterion: Loss function (e.g. CrossEntropyLoss).
        device: Target execution device ('cpu' or 'cuda').
        grad_clip: Maximum gradient norm for clipping. Default: 1.0.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_clip = float(grad_clip)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def _set_model_qat_mode(self, enabled: bool) -> None:
        """Enable or disable QAT mode on all Watch layers."""
        for module in self.model.modules():
            if isinstance(module, (WatchLinear, WatchConv2d)):
                module.set_qat_mode(enabled)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        warmup_epochs: int = 1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model over specified epochs with an FP32 warm-up phase.

        Args:
            train_loader: DataLoader providing training batches (inputs, targets).
            val_loader: Optional DataLoader providing validation batches.
            epochs: Total number of epochs to train.
            warmup_epochs: Number of initial epochs in FP32 before activating 1-bit QAT.
            verbose: If True, prints per-epoch metrics.

        Returns:
            Dictionary containing training history metrics.
        """
        for epoch in range(epochs):
            t0 = time.time()
            is_warmup = epoch < warmup_epochs

            # Configure QAT mode based on warmup
            self._set_model_qat_mode(not is_warmup)
            mode_str = "FP32 Warmup" if is_warmup else "1-Bit QAT (STE)"

            self.model.train()
            total_train_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs)

                loss.backward()

                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_train_loss / max(num_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            # Validation evaluation
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

            elapsed = time.time() - t0
            if verbose:
                val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}" if val_loader else ""
                print(
                    f"Epoch [{epoch+1}/{epochs}] ({mode_str}) - Train Loss: {avg_train_loss:.4f}"
                    f"{val_info} - {elapsed:.2f}s"
                )

        return self.history

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on a validation dataset.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        num_batches = 0

        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs, targets = batch.to(self.device), None

            outputs = self.model(inputs)
            if targets is not None:
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Calculate classification accuracy if output is logits
                if outputs.dim() >= 2 and targets.dim() == 1:
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == targets).sum().item()
                    total_samples += targets.size(0)
            else:
                loss = self.criterion(outputs)
                total_loss += loss.item()

            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        acc = (correct / total_samples) if total_samples > 0 else 0.0
        return avg_loss, acc

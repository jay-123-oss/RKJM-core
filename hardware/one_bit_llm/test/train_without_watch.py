"""Train a Baseline FP32 Regression Model using Standard PyTorch."""

import os
import sys
import time
import torch
import torch.nn as nn

here = os.path.abspath(os.path.dirname(__file__))
from dataset import get_dataloaders


class BaselineRegressionNet(nn.Module):
    """Deep Regression Network using standard FP32 nn.Linear layers with LayerNorm and residual connections."""

    def __init__(self, in_features: int = 32, hidden_features: int = 128):
        super().__init__()
        self.proj_in = nn.Linear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.dense1 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.dense2 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.head = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        h = h + self.dense1(self.norm1(h))
        h = h + self.dense2(self.norm2(h))
        return self.head(h)


def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    return 1.0 - (ss_res / max(ss_tot, 1e-8))


def train_baseline(epochs: int = 30, batch_size: int = 64, lr: float = 0.002):
    print("=" * 70)
    print("STARTING STANDARD PYTORCH (FP32 BASELINE) REGRESSION TRAINING")
    print("=" * 70)

    train_loader, val_loader, data = get_dataloaders(batch_size=batch_size)
    in_features = data["in_features"]

    model = BaselineRegressionNet(in_features=in_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    start_time = time.time()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        t0_epoch = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        history["val_loss"].append(avg_val_loss)

        epoch_time = time.time() - t0_epoch
        print(
            f"Epoch [{epoch+1}/{epochs}] (FP32 Dense) - Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} - {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time

    # Save standard PyTorch checkpoint
    save_path_pt = os.path.join(here, "baseline_model.pt")
    torch.save(model.state_dict(), save_path_pt)
    pt_size = os.path.getsize(save_path_pt)

    # Evaluation
    model.eval()
    with torch.no_grad():
        x_val, y_val = data["x_val"], data["y_val"]
        t0_inf = time.time()
        val_preds = model(x_val)
        inf_time_ms = (time.time() - t0_inf) * 1000.0

        val_mse = criterion(val_preds, y_val).item()
        val_rmse = val_mse ** 0.5
        val_r2 = compute_r2_score(y_val, val_preds)

    print("\n[Baseline PyTorch Training Complete]")
    print(f"Training Time        : {total_time:.2f} seconds")
    print(f"Final Train MSE Loss : {history['train_loss'][-1]:.4f}")
    print(f"Final Val MSE Loss   : {val_mse:.4f} (RMSE: {val_rmse:.4f})")
    print(f"Validation R2 Score  : {val_r2:.4f}")
    print(f"Inference Latency    : {inf_time_ms:.2f} ms for 2,000 samples ({x_val.shape[0] / (inf_time_ms / 1000):,.0f} samples/sec)")
    print(f"Saved Checkpoint .pt : {pt_size:,} bytes")

    return {
        "model_name": "Standard PyTorch (FP32 Dense)",
        "train_time": total_time,
        "train_loss": history["train_loss"][-1],
        "val_loss": val_mse,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "inf_time_ms": inf_time_ms,
        "throughput": x_val.shape[0] / (inf_time_ms / 1000),
        "file_size_bytes": pt_size,
    }


if __name__ == "__main__":
    train_baseline()

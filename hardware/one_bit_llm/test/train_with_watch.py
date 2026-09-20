"""Train a 1-Bit Regression Model using the Watch Framework."""

import os
import sys
import time
import torch
import torch.nn as nn

# Ensure watch_framework is discoverable
here = os.path.abspath(os.path.dirname(__file__))
pkg_root = os.path.abspath(os.path.join(here, "..", "watch_framework"))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

import watch_framework as wf
from dataset import get_dataloaders


class WatchRegressionNet(nn.Module):
    """Deep Regression Network utilizing 1-Bit WatchLinear CSA layers with LayerNorm and residual connections."""

    def __init__(self, in_features: int = 32, hidden_features: int = 128):
        super().__init__()
        self.proj_in = nn.Linear(in_features, hidden_features)
        # 1-Bit Watch Grid Core Blocks
        self.norm1 = nn.LayerNorm(hidden_features)
        self.watch1 = wf.nn.WatchLinear(hidden_features, hidden_features, bias=True)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.watch2 = wf.nn.WatchLinear(hidden_features, hidden_features, bias=True)
        # Regression Head
        self.head = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        h = h + self.watch1(self.norm1(h))
        h = h + self.watch2(self.norm2(h))
        return self.head(h)


def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    return 1.0 - (ss_res / max(ss_tot, 1e-8))


def train_watch(epochs: int = 30, batch_size: int = 64, lr: float = 0.002):
    print("=" * 70)
    print("STARTING WATCH FRAMEWORK (1-BIT QAT) REGRESSION TRAINING")
    print("=" * 70)

    train_loader, val_loader, data = get_dataloaders(batch_size=batch_size)
    in_features = data["in_features"]

    model = WatchRegressionNet(in_features=in_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    trainer = wf.Trainer(model, optimizer, criterion, device="cpu")

    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=epochs, warmup_epochs=1, verbose=True)
    total_time = time.time() - start_time

    # Save 1-bit bit-packed checkpoint
    save_path_wfbin = os.path.join(here, "watch_model.wfbin")
    wf.save_checkpoint(model, save_path_wfbin)
    wfbin_size = os.path.getsize(save_path_wfbin)

    # Save equivalent standard .pt for size comparison
    save_path_pt = os.path.join(here, "watch_model_uncompressed.pt")
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

    print("\n[Watch Framework Training Complete]")
    print(f"Training Time        : {total_time:.2f} seconds")
    print(f"Final Train MSE Loss : {history['train_loss'][-1]:.4f}")
    print(f"Final Val MSE Loss   : {val_mse:.4f} (RMSE: {val_rmse:.4f})")
    print(f"Validation R2 Score  : {val_r2:.4f}")
    print(f"Inference Latency    : {inf_time_ms:.2f} ms for 2,000 samples ({x_val.shape[0] / (inf_time_ms / 1000):,.0f} samples/sec)")
    print(f"Compressed .wfbin    : {wfbin_size:,} bytes")
    print(f"Uncompressed .pt     : {pt_size:,} bytes (Compression: {pt_size / wfbin_size:.2f}x)")

    return {
        "model_name": "Watch Framework (1-Bit QAT)",
        "train_time": total_time,
        "train_loss": history["train_loss"][-1],
        "val_loss": val_mse,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "inf_time_ms": inf_time_ms,
        "throughput": x_val.shape[0] / (inf_time_ms / 1000),
        "file_size_bytes": wfbin_size,
        "raw_pt_size_bytes": pt_size,
    }


if __name__ == "__main__":
    train_watch()

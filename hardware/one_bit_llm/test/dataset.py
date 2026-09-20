"""10,000-Row Regression Dataset Generator and DataLoaders."""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_FILE = os.path.join(os.path.dirname(__file__), "data_regression_10k.pt")


def generate_regression_data(
    num_samples: int = 1000000,
    in_features: int = 32,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Generate 10,000 samples for a non-linear regression task and save deterministically."""
    torch.manual_seed(seed)

    # 1. Inputs: 32 continuous features
    X = torch.randn(num_samples, in_features)

    # 2. Linear ground-truth weights
    w_true = torch.randn(in_features, 1) * 0.8

    # 3. Non-linear components (interactions & trigonometric activations)
    nonlinear = (
        0.5 * torch.sin(X[:, 0:1] * X[:, 1:2])
        + 0.4 * torch.cos(X[:, 2:3] * 1.5)
        + 0.3 * torch.tanh(X[:, 3:4] + X[:, 4:5])
    )

    # 4. Target with Gaussian noise
    noise = 0.1 * torch.randn(num_samples, 1)
    y_raw = torch.matmul(X, w_true) + nonlinear + noise

    # Standardize target for numerical stability
    y = (y_raw - y_raw.mean()) / y_raw.std()

    # Split train/val
    split_idx = int(num_samples * train_ratio)
    x_train, y_train = X[:split_idx], y[:split_idx]
    x_val, y_val = X[split_idx:], y[split_idx:]

    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "in_features": in_features,
        "num_samples": num_samples,
    }

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    torch.save(data, DATA_FILE)
    print(f"Generated {num_samples} samples saved to: {DATA_FILE}")
    print(f"Train samples: {x_train.shape[0]} | Validation samples: {x_val.shape[0]}")
    return data


def get_dataloaders(batch_size: int = 64):
    """Load the pre-generated dataset into PyTorch DataLoaders."""
    if not os.path.exists(DATA_FILE):
        generate_regression_data()

    data = torch.load(DATA_FILE, weights_only=False)
    train_dataset = TensorDataset(data["x_train"], data["y_train"])
    val_dataset = TensorDataset(data["x_val"], data["y_val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, data


if __name__ == "__main__":
    generate_regression_data()

"""End-to-end QAT, serialization, timing, and memory validation.

Run on any machine with the workspace environment::

    .venv/bin/python test_end2end_training.py

The script intentionally uses a small synthetic classification problem so it
is useful on CPU-only development machines.  On CUDA it additionally reports
peak allocated VRAM for the dense and Watch Grid models.
"""

from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn

from watch_grid_pytorch import WatchGridLinear


class StandardMLP(nn.Module):
    """Three-layer dense baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int, classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class WatchGridMLP(nn.Module):
    """The same topology with Watch Grid linear layers."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int, classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            WatchGridLinear(input_dim, hidden_dim),
            nn.ReLU(),
            WatchGridLinear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            WatchGridLinear(bottleneck_dim, classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)

    @classmethod
    def from_standard(cls, model: StandardMLP) -> "WatchGridMLP":
        result = cls(1, 1, 1, 1)
        dense_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        result.layers = nn.Sequential(
            WatchGridLinear.from_linear(dense_layers[0]),
            nn.ReLU(),
            WatchGridLinear.from_linear(dense_layers[1]),
            nn.ReLU(),
            WatchGridLinear.from_linear(dense_layers[2]),
        )
        return result


def make_dataset(seed: int, samples: int, input_dim: int, classes: int) -> Tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, input_dim, generator=generator)
    teacher = torch.randn(input_dim, classes, generator=generator)
    labels = (inputs @ teacher).argmax(dim=1)
    return inputs, labels


def train_for_50_steps(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    name: str,
    learning_rate: float,
) -> Tuple[list[float], list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    epoch_times: list[float] = []
    print(f"\n{name} training log")
    for epoch in range(5):
        epoch_start = time.perf_counter()
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            for module in model.modules():
                if isinstance(module, WatchGridLinear):
                    module.clamp_latent_weights()
            loss_value = float(loss.detach())
            losses.append(loss_value)
            print(f"  step {epoch * 10 + step + 1:02d}/50 loss={loss_value:.6f}")
        epoch_times.append(time.perf_counter() - epoch_start)
    return losses, epoch_times


def peak_cuda_bytes(model: nn.Module, inputs: Tensor, labels: Tensor) -> int:
    """Measure peak allocated bytes for one representative training step."""

    if not torch.cuda.is_available():
        return 0
    model.cuda()
    inputs = inputs.cuda()
    labels = labels.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    torch.cuda.reset_peak_memory_stats()
    loss = nn.CrossEntropyLoss()(model(inputs), labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def export_watch_model(model: WatchGridMLP, directory: Path) -> int:
    total = 0
    for index, layer in enumerate(model.layers):
        if isinstance(layer, WatchGridLinear):
            info = layer.export_packed_checkpoint(directory / f"watch_layer_{index}.pt")
            total += Path(info["path"]).stat().st_size
    return total


def main() -> None:
    torch.manual_seed(21)
    input_dim, hidden_dim, bottleneck_dim, classes = 32, 64, 32, 4
    inputs, labels = make_dataset(22, 256, input_dim, classes)

    standard = StandardMLP(input_dim, hidden_dim, bottleneck_dim, classes)
    watch = WatchGridMLP.from_standard(standard)
    standard_losses, standard_epoch_times = train_for_50_steps(
        standard, inputs, labels, "Standard nn.Linear", learning_rate=2.0e-3
    )
    watch_losses, watch_epoch_times = train_for_50_steps(
        watch, inputs, labels, "WatchGridLinear", learning_rate=5.0e-3
    )

    assert watch_losses[-1] < watch_losses[0], "WatchGridLinear loss did not decrease"
    assert standard_losses[-1] < standard_losses[0], "baseline loss did not decrease"

    standard_parameter_bytes = parameter_bytes(standard)
    watch_parameter_bytes = parameter_bytes(watch)
    with tempfile.TemporaryDirectory(prefix="watch-grid-qat-") as directory_name:
        directory = Path(directory_name)
        standard_path = directory / "standard_state.pt"
        torch.save(standard.state_dict(), standard_path)
        packed_bytes = export_watch_model(watch, directory)
        standard_disk_bytes = standard_path.stat().st_size

    print("\nValidation summary")
    print(f"  standard loss: {standard_losses[0]:.6f} -> {standard_losses[-1]:.6f}")
    print(f"  watch loss:    {watch_losses[0]:.6f} -> {watch_losses[-1]:.6f}")
    print("\nPer-epoch timing")
    for index, (dense_time, watch_time) in enumerate(zip(standard_epoch_times, watch_epoch_times), 1):
        print(f"  epoch {index}: standard={dense_time:.4f}s watch={watch_time:.4f}s speedup={dense_time / watch_time:.2f}x")
    print("\nMemory and serialized footprint")
    print(f"  standard parameter memory (FP32): {standard_parameter_bytes / 1024:.1f} KiB")
    print(f"  Watch latent parameter memory (FP32): {watch_parameter_bytes / 1024:.1f} KiB")
    print(f"  standard checkpoint: {standard_disk_bytes / 1024:.1f} KiB")
    print(f"  packed Watch checkpoints: {packed_bytes / 1024:.1f} KiB")
    print(f"  disk reduction: {standard_disk_bytes / max(1, packed_bytes):.2f}x")
    if torch.cuda.is_available():
        print(f"  standard peak allocated VRAM: {peak_cuda_bytes(standard, inputs, labels) / 2**20:.2f} MiB")
        print(f"  Watch peak allocated VRAM: {peak_cuda_bytes(watch, inputs, labels) / 2**20:.2f} MiB")
    else:
        print("  CUDA unavailable: peak allocated VRAM = 0 MiB for both runs")
    print("  status: PASS")


if __name__ == "__main__":
    main()
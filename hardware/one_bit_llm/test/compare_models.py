"""Comparison Runner: Watch Framework (1-Bit QAT) vs Standard PyTorch (FP32).

Executes both models on the identical 10,000-row regression dataset,
evaluates side-by-side metrics, and prints a comparative analysis.
"""

import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

from dataset import generate_regression_data
from train_without_watch import train_baseline
from train_with_watch import train_watch


def print_comparison_table(baseline: dict, watch: dict):
    ratio = baseline["file_size_bytes"] / watch["file_size_bytes"]

    print("\n" + "=" * 80)
    print("                10,000-ROW REGRESSION BENCHMARK COMPARISON               ")
    print("=" * 80)
    header = f"{'Metric / Feature':<32} | {'Standard PyTorch (FP32)':<22} | {'Watch Framework (1-Bit)':<22}"
    print(header)
    print("-" * 80)

    rows = [
        ("Weight Precision", "32-bit Float (FP32)", "1-bit Sign {-1, +1} + Alpha"),
        ("Arithmetic Core", "Dense FP32 BLAS", "Bitwise Carry-Save Addition (CSA)"),
        ("Final Train MSE Loss", f"{baseline['train_loss']:.4f}", f"{watch['train_loss']:.4f}"),
        ("Final Validation MSE Loss", f"{baseline['val_loss']:.4f}", f"{watch['val_loss']:.4f}"),
        ("Validation RMSE", f"{baseline['val_rmse']:.4f}", f"{watch['val_rmse']:.4f}"),
        ("Validation R2 Score", f"{baseline['val_r2']:.4f}", f"{watch['val_r2']:.4f}"),
        ("Total Training Time", f"{baseline['train_time']:.2f} s", f"{watch['train_time']:.2f} s"),
        ("Inference Latency (2k)", f"{baseline['inf_time_ms']:.2f} ms", f"{watch['inf_time_ms']:.2f} ms"),
        ("Inference Throughput", f"{baseline['throughput']:,.0f} samples/s", f"{watch['throughput']:,.0f} samples/s"),
        ("Checkpoint File Size", f"{baseline['file_size_bytes']:,} bytes (.pt)", f"{watch['file_size_bytes']:,} bytes (.wfbin)"),
        ("Model Compression Ratio", "1.00x (Baseline)", f"{ratio:.2f}x Smaller!"),
    ]

    for label, val_base, val_watch in rows:
        print(f"{label:<32} | {val_base:<22} | {val_watch:<22}")

    print("=" * 80)
    print("\n[KEY ARCHITECTURAL INSIGHTS]")
    print(f"1. Memory Compression: Watch Framework achieved a {ratio:.2f}x reduction in checkpoint size.")
    print(f"2. Regression Parity : 1-bit weights with dynamic alpha captured the non-linear dataset (R2 = {watch['val_r2']:.4f}).")
    print("3. Hardware Alignment: Bit-packed serialization (.wfbin) matches 2D Watch Grid CSA register layout.")
    print("=" * 80)


def main():
    # 1. Ensure dataset exists
    generate_regression_data(num_samples=100000, seed=42)

    # 2. Run baseline training
    res_baseline = train_baseline(epochs=30, batch_size=64, lr=0.002)

    # 3. Run watch framework training
    res_watch = train_watch(epochs=30, batch_size=64, lr=0.002)

    # 4. Compare and display results
    print_comparison_table(res_baseline, res_watch)


if __name__ == "__main__":
    main()

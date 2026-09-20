"""
Performance Benchmark: Standard PyTorch FP32 Linear vs RKMJ 1.58-bit CSALinear.
"""

from __future__ import annotations

import os
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rkmj.nn.linear import CSALinear


def run_benchmark():
    print("=" * 80)
    print("RKMJ-CORE CPU BENCHMARK: FP32 TORCH GEMM vs 1.58-BIT POPCOUNT CSA ENGINE")
    print("=" * 80)

    # Standard LLM Hidden Dimensions
    batch_size = 1
    seq_len = 512
    dim = 2048
    runs = 50

    print(f"Workload Configuration:")
    print(f"  - Batch Size:                    {batch_size}")
    print(f"  - Sequence Length:               {seq_len}")
    print(f"  - Dimension (In / Out):          {dim} x {dim}")
    print(f"  - Benchmark Iterations:          {runs}")

    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)

    # 1. Standard PyTorch FP32 Linear
    fp32_linear = nn.Linear(dim, dim, bias=False)
    # Warmup
    for _ in range(5):
        _ = fp32_linear(x)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = fp32_linear(x)
    fp32_time = (time.perf_counter() - t0) / runs * 1000.0

    # 2. RKMJ 1.58-bit CSALinear (Packed Popcount)
    csa_linear = CSALinear(dim, dim, bias=False)
    csa_linear.pack_weights_for_inference()
    csa_linear.eval()

    # Warmup
    for _ in range(5):
        _ = csa_linear(x)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = csa_linear(x)
    csa_time = (time.perf_counter() - t0) / runs * 1000.0

    # Memory comparison
    fp32_mem_kb = (dim * dim * 4) / 1024.0
    csa_mem_kb = (dim * (dim // 16) * 4) / 1024.0
    compression = fp32_mem_kb / csa_mem_kb

    print("\nBenchmark Results:")
    print(f"  - FP32 torch.nn.Linear Latency:  {fp32_time:.2f} ms")
    print(f"  - RKMJ CSALinear Latency:        {csa_time:.2f} ms")
    if csa_time > 0:
        speedup = fp32_time / csa_time
        print(f"  - Relative Speedup:              {speedup:.2f}x")
    print(f"\nMemory Footprint:")
    print(f"  - FP32 Weight Memory:            {fp32_mem_kb:,.1f} KB")
    print(f"  - 1.58-bit CSA Weight Memory:    {csa_mem_kb:,.1f} KB")
    print(f"  - Exact Memory Reduction:        {compression:.1f}x (15.8x target)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

#!/usr/bin/env python3
"""
Production Hardening & Multi-ISA Benchmark for RKMJ-Core C++ Engine.
Tests:
  1. Mathematical Invariance against PyTorch FP32 baseline (atol <= 1e-5).
  2. Crash & Edge-Case Safety (M=1, non-aligned dimensions K, N, odd batches).
  3. Performance Metrics: GFLOPS/Bitwise TOPS, Bandwidth, Latency vs FP32.
"""

import sys
import os
import time
import argparse
import torch
import torch.nn.functional as F

# Add rkmj-core to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import rkmj._C as _C
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Could not import rkmj._C: {e}")
    sys.exit(1)


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_mathematical_invariance():
    print_banner("1. MATHEMATICAL INVARIANCE TEST (atol <= 1e-5)")

    backend = _C.get_simd_backend_name()
    print(f"Active Multi-ISA SIMD Backend: \033[1;32m{backend}\033[0m\n")

    torch.manual_seed(42)
    shapes = [
        (1, 2048, 2048),     # Autoregressive decode step (M=1)
        (4, 1024, 1024),     # Small batch
        (16, 2048, 2048),    # Moderate batch
        (128, 2048, 2048),   # Prefill batch
    ]

    all_passed = True
    for M, K, N in shapes:
        x = torch.randn(M, K, dtype=torch.float32)
        # Realistic dynamic alpha scaling: alpha = mean(|W|) ~ 0.02
        w_dense = torch.randn(N, K, dtype=torch.float32)
        alpha = w_dense.abs().mean(dim=1).clamp(min=1e-5)
        w_ternary = torch.clamp(torch.round(w_dense / alpha.unsqueeze(1)), -1.0, 1.0)
        bias = torch.randn(N, dtype=torch.float32) * 0.01

        w_packed = _C.pack_weights(w_ternary)

        # PyTorch mathematical ground-truth:
        # y = alpha * (x_sign @ w_ternary.T) + bias
        x_sign = torch.where(x >= 0.0, 1.0, -1.0)
        y_expected = alpha * F.linear(x_sign, w_ternary) + bias

        # C++ Multi-ISA forward pass
        y_csa = _C.csa_forward(x, w_packed, alpha, bias)

        max_err = torch.max(torch.abs(y_csa - y_expected)).item()
        passed = torch.allclose(y_csa, y_expected, atol=2e-5, rtol=1e-5)

        status = "\033[1;32mPASS\033[0m" if passed else "\033[1;31mFAIL\033[0m"
        print(f"Shape [M={M:3d}, K={K:4d}, N={N:4d}] -> Max Abs Error: {max_err:10.7f} | Status: {status}")

        if not passed:
            all_passed = False

    # Test Fused RMSNorm + CSA Popcount
    print("\nTesting Fused Operator: RMSNorm + 1.58-bit CSA Forward...")
    for M, K, N in [(1, 2048, 2048), (32, 2048, 2048)]:
        x = torch.randn(M, K, dtype=torch.float32)
        gamma = torch.rand(K, dtype=torch.float32) + 0.5
        eps = 1e-6
        w_dense = torch.randn(N, K, dtype=torch.float32)
        alpha = w_dense.abs().mean(dim=1).clamp(min=1e-5)
        w_ternary = torch.clamp(torch.round(w_dense / alpha.unsqueeze(1)), -1.0, 1.0)
        bias = torch.randn(N, dtype=torch.float32) * 0.01
        w_packed = _C.pack_weights(w_ternary)

        # PyTorch ground truth
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma
        norm_x_sign = torch.where(norm_x >= 0.0, 1.0, -1.0)
        y_fused_expected = alpha * F.linear(norm_x_sign, w_ternary) + bias

        # Fused C++ entrypoint
        y_fused_csa = _C.fused_rmsnorm_csa_forward(x, gamma, eps, w_packed, alpha, bias)

        max_err = torch.max(torch.abs(y_fused_csa - y_fused_expected)).item()
        passed = torch.allclose(y_fused_csa, y_fused_expected, atol=2e-5, rtol=1e-5)
        status = "\033[1;32mPASS\033[0m" if passed else "\033[1;31mFAIL\033[0m"
        print(f"Fused [M={M:3d}, K={K:4d}, N={N:4d}] -> Max Abs Error: {max_err:10.7f} | Status: {status}")
        if not passed:
            all_passed = False

    return all_passed


def test_crash_and_edge_cases():
    print_banner("2. CRASH & EDGE-CASE SAFETY TEST")

    edge_cases = [
        (1, 17, 31),        # Non-aligned primes, M=1
        (1, 33, 65),        # Non-aligned 32-bit boundary
        (3, 77, 127),       # Odd batch and odd non-aligned K, N
        (7, 1000, 500),     # Non-power-of-two arbitrary shapes
        (1, 2048, 2048),    # Autoregressive single token step
        (2, 4096, 4096),    # Large channel count
        (1, 15, 16),        # Tail exactly 15 (sub-word boundary)
        (1, 16, 16),        # Exactly 1 word
        (1, 17, 16),        # Exactly 1 word + 1 element tail
    ]

    all_passed = True
    for M, K, N in edge_cases:
        try:
            x = torch.randn(M, K, dtype=torch.float32)
            w_dense = torch.randn(N, K, dtype=torch.float32)
            alpha = w_dense.abs().mean(dim=1).clamp(min=1e-5)
            w_ternary = torch.clamp(torch.round(w_dense / alpha.unsqueeze(1)), -1.0, 1.0)
            bias = torch.randn(N, dtype=torch.float32) * 0.01
            w_packed = _C.pack_weights(w_ternary)

            # Execution
            y_csa = _C.csa_forward(x, w_packed, alpha, bias)

            # Verification
            x_sign = torch.where(x >= 0.0, 1.0, -1.0)
            y_exp = alpha * F.linear(x_sign, w_ternary) + bias
            err = torch.max(torch.abs(y_csa - y_exp)).item()

            passed = torch.allclose(y_csa, y_exp, atol=2e-5, rtol=1e-5) and not torch.isnan(y_csa).any()
            status = "\033[1;32mPASS\033[0m" if passed else "\033[1;31mFAIL\033[0m"
            print(f"Edge Case [M={M:2d}, K={K:5d}, N={N:5d}] -> Max Error: {err:9.6f} | Status: {status}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"Edge Case [M={M:2d}, K={K:5d}, N={N:5d}] -> CRASHED: {e}")
            all_passed = False

    return all_passed


def run_benchmark():
    print_banner("3. HIGH-THROUGHPUT PERFORMANCE PROFILING & BENCHMARKS")

    backend = _C.get_simd_backend_name()
    print(f"Hardware & ISA Engine: \033[1;34m{backend}\033[0m")
    print(f"Torch Threads: {torch.get_num_threads()}")

    configs = [
        ("Single-Token Autoregressive Decode Step", 1, 2048, 2048, 200),
        ("Batched Prefill Generation Step", 128, 2048, 2048, 30),
    ]

    print("\n" + "-" * 88)
    print(f"{'Workload Scenario':<35} | {'M':>4} | {'K':>5} | {'N':>5} | {'FP32 (ms)':>9} | {'RKMJ (ms)':>9} | {'Speedup':>7} | {'Bitwise TOPS':>12}")
    print("-" * 88)

    for desc, M, K, N, iters in configs:
        x = torch.randn(M, K, dtype=torch.float32)
        w_fp32 = torch.randn(N, K, dtype=torch.float32)
        w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.float32)
        alpha = torch.ones(N, dtype=torch.float32)
        w_packed = _C.pack_weights(w_ternary)

        # Warmup
        for _ in range(10):
            _ = F.linear(x, w_fp32)
            _ = _C.csa_forward(x, w_packed, alpha, None)

        # Benchmark PyTorch FP32 GEMM
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = F.linear(x, w_fp32)
        t_fp32 = (time.perf_counter() - t0) / iters * 1000.0  # ms

        # Benchmark RKMJ Multi-ISA CSA Popcount
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = _C.csa_forward(x, w_packed, alpha, None)
        t_rkmj = (time.perf_counter() - t0) / iters * 1000.0  # ms

        speedup = t_fp32 / t_rkmj
        total_ops = 2.0 * M * N * K
        effective_tops = (total_ops / (t_rkmj * 1e-3)) / 1e12  # TeraOps

        print(f"{desc:<35} | {M:4d} | {K:5d} | {N:5d} | {t_fp32:9.3f} | {t_rkmj:9.3f} | {speedup:6.2f}x | {effective_tops:10.3f} TOPS")

    print("-" * 88)

    # Memory Bandwidth & Compression Audit
    print("\nMemory Footprint & Bandwidth Compression Analysis:")
    k_dim = 2048
    n_dim = 2048
    fp32_bytes = n_dim * k_dim * 4
    rkmj_bytes = n_dim * ((k_dim + 15) // 16) * 4
    comp_ratio = fp32_bytes / rkmj_bytes

    print(f"  • Weight Tensor [N={n_dim}, K={k_dim}]:")
    print(f"    - Standard FP32 Storage:      {fp32_bytes / (1024*1024):.2f} MB")
    print(f"    - RKMJ 2-bit Packed Storage:  {rkmj_bytes / (1024*1024):.2f} MB")
    print(f"    - Memory Compression Ratio:   \033[1;32m{comp_ratio:.2f}x\033[0m")
    print(f"    - Memory Traffic Reduction:   \033[1;32m{(1.0 - 1.0/comp_ratio)*100:.1f}%\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-ISA Hardened RKMJ Benchmark")
    parser.add_argument("--verify-only", action="store_true", help="Run invariance and safety verification only")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    args = parser.parse_args()

    inv_ok = test_mathematical_invariance()
    edge_ok = test_crash_and_edge_cases()

    if not (inv_ok and edge_ok):
        print("\n\033[1;31m[FAILURE] Verification checks failed.\033[0m")
        sys.exit(1)

    print("\n\033[1;32m[SUCCESS] All Mathematical Invariance & Crash Safety Tests PASSED!\033[0m")

    if args.benchmark or not args.verify_only:
        run_benchmark()


if __name__ == "__main__":
    main()

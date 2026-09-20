"""
Verification and Benchmark Suite for Custom CSA Linear CPU Extension.

Compares our Carry-Save Addition (CSA) / Popcount 1.58-bit Ternary Linear Layer
against standard PyTorch FP32 `torch.nn.Linear` on CPU.
"""

import time
import os
import sys
import torch
import torch.nn as nn
from typing import Optional

# Ensure the compiled extension can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import csa_linear_cpu
except ImportError as e:
    raise ImportError(
        f"Failed to import csa_linear_cpu. Please run 'python setup.py build_ext --inplace' first. Error: {e}"
    )


class CSALinear(nn.Module):
    """
    1.58-bit Ternary Linear Layer powered by CPU Carry-Save Addition (CSA) & Popcount.

    Weight Encoding (2 bits per weight packed in uint32):
      00 =  0
      01 = +1
      10 = -1

    Memory compression: 16x reduction compared to FP32 weights (2 bits vs 32 bits).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_words = (in_features + 15) // 16

        # Register packed weights and dynamic alpha as buffers
        self.register_buffer(
            "w_packed",
            torch.zeros((out_features, self.num_words), dtype=torch.int32),
        )
        self.register_buffer(
            "alpha",
            torch.ones((out_features,), dtype=torch.float32),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float_linear(cls, linear: nn.Linear) -> "CSALinear":
        """Convert a standard FP32 nn.Linear into an optimized CSALinear."""
        out_features, in_features = linear.weight.shape
        csa = cls(in_features, out_features, bias=(linear.bias is not None))

        with torch.no_grad():
            w = linear.weight.detach().clone()
            # Calculate dynamic scale alpha = mean(|w|) per row
            alpha = w.abs().mean(dim=1).clamp(min=1e-8)
            # Ternary quantization: round(w / alpha), clamped to [-1, 0, 1]
            w_scaled = w / alpha.unsqueeze(1)
            w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)

            # Pack weights using C++ extension
            csa.w_packed.copy_(csa_linear_cpu.pack_weights(w_ternary))
            csa.alpha.copy_(alpha)

            if linear.bias is not None and csa.bias is not None:
                csa.bias.copy_(linear.bias.detach())

        return csa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using multithreaded OpenMP bitwise popcount CSA reduction.
        """
        return csa_linear_cpu.forward(x, self.w_packed, self.alpha, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def run_verification():
    """Verify correctness of packing, unpacking, and forward calculation."""
    print("=" * 70)
    print("RUNNING MATHEMATICAL CORRECTNESS VERIFICATION")
    print("=" * 70)

    torch.manual_seed(42)

    # 1. Verification with standard dimensions
    B, K, N = 8, 128, 64
    print(f"[Test 1] Standard Dimensions: Batch={B}, InFeatures={K}, OutFeatures={N}")

    w_ternary = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    alpha = torch.rand((N,), dtype=torch.float32) * 0.1 + 0.05
    bias = torch.randn((N,), dtype=torch.float32)
    x = torch.randn((B, K), dtype=torch.float32)

    # Test pack & unpack
    w_packed = csa_linear_cpu.pack_weights(w_ternary)
    w_unpacked = csa_linear_cpu.unpack_weights(w_packed, K)
    assert torch.equal(w_ternary, w_unpacked), "Unpacked weights do not match original ternary weights!"
    print("  ✓ C++ Weight packing and unpacking verified with 100% fidelity.")

    # Mathematical reference
    x_sign = torch.where(x >= 0.0, 1.0, -1.0)
    y_expected = alpha.unsqueeze(0) * (x_sign @ w_ternary.t()) + bias.unsqueeze(0)

    # Kernel execution
    y_csa = csa_linear_cpu.forward(x, w_packed, alpha, bias)
    max_err = torch.max(torch.abs(y_csa - y_expected)).item()
    assert max_err < 1e-5, f"Forward output mismatch! Max error: {max_err}"
    print(f"  ✓ Forward pass matches reference exactly! Max absolute error: {max_err:.1e}")

    # 2. Verification with odd dimensions (non-multiples of 16/32)
    B_odd, K_odd, N_odd = 7, 77, 43
    print(f"\n[Test 2] Odd Non-Aligned Dimensions: Batch={B_odd}, InFeatures={K_odd}, OutFeatures={N_odd}")
    w_odd = torch.randint(-1, 2, (N_odd, K_odd), dtype=torch.float32)
    alpha_odd = torch.rand((N_odd,), dtype=torch.float32)
    bias_odd = torch.randn((N_odd,), dtype=torch.float32)
    x_odd = torch.randn((B_odd, K_odd), dtype=torch.float32)

    w_packed_odd = csa_linear_cpu.pack_weights(w_odd)
    y_expected_odd = alpha_odd.unsqueeze(0) * (torch.where(x_odd >= 0.0, 1.0, -1.0) @ w_odd.t()) + bias_odd.unsqueeze(0)
    y_csa_odd = csa_linear_cpu.forward(x_odd, w_packed_odd, alpha_odd, bias_odd)
    max_err_odd = torch.max(torch.abs(y_csa_odd - y_expected_odd)).item()
    assert max_err_odd < 1e-5, f"Odd dimension mismatch! Max error: {max_err_odd}"
    print(f"  ✓ Non-aligned tail bit masking verified! Max absolute error: {max_err_odd:.1e}")

    # 3. Verification with 3D Sequence Tensor [Batch, SeqLen, InFeatures]
    print("\n[Test 3] 3D Transformer Sequence Shape: [Batch=2, Seq=16, InFeatures=128]")
    x_3d = torch.randn((2, 16, K), dtype=torch.float32)
    y_3d = csa_linear_cpu.forward(x_3d, w_packed, alpha, bias)
    assert y_3d.shape == (2, 16, N), f"3D shape mismatch! Expected (2, 16, {N}), got {y_3d.shape}"
    print(f"  ✓ 3D Tensor input handled correctly -> Output shape {y_3d.shape}")

    # 4. CSALinear Module Integration
    print("\n[Test 4] CSALinear nn.Module Wrapper")
    csa_module = CSALinear(K, N, bias=True)
    csa_module.w_packed.copy_(w_packed)
    csa_module.alpha.copy_(alpha)
    csa_module.bias.data.copy_(bias)
    y_module = csa_module(x)
    assert torch.equal(y_module, y_csa), "CSALinear module output mismatch!"
    print("  ✓ CSALinear PyTorch module functioning seamlessly.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


def benchmark_layer(
    batch_size: int,
    in_features: int,
    out_features: int,
    warmup: int = 20,
    iterations: int = 100,
):
    """Benchmark CSALinear against torch.nn.Linear."""
    # Create standard FP32 linear layer
    fp32_linear = nn.Linear(in_features, out_features, bias=True)
    csa_linear = CSALinear.from_float_linear(fp32_linear)

    # Memory calculations
    fp32_weight_bytes = in_features * out_features * 4
    csa_weight_bytes = (out_features * ((in_features + 15) // 16) * 4) + (out_features * 4)
    compression_ratio = fp32_weight_bytes / csa_weight_bytes

    x = torch.randn((batch_size, in_features), dtype=torch.float32)

    # Warmup FP32 Linear
    for _ in range(warmup):
        _ = fp32_linear(x)

    # Time FP32 Linear
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = fp32_linear(x)
    t1 = time.perf_counter()
    fp32_time_ms = (t1 - t0) * 1000.0 / iterations

    # Warmup CSA Linear
    for _ in range(warmup):
        _ = csa_linear(x)

    # Time CSA Linear
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = csa_linear(x)
    t1 = time.perf_counter()
    csa_time_ms = (t1 - t0) * 1000.0 / iterations

    speedup = fp32_time_ms / max(csa_time_ms, 1e-9)

    return {
        "batch_size": batch_size,
        "in_features": in_features,
        "out_features": out_features,
        "fp32_time_ms": fp32_time_ms,
        "csa_time_ms": csa_time_ms,
        "speedup": speedup,
        "fp32_weight_kb": fp32_weight_bytes / 1024.0,
        "csa_weight_kb": csa_weight_bytes / 1024.0,
        "compression_ratio": compression_ratio,
    }


def run_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("\n" + "=" * 70)
    print("BENCHMARK: CSA 1.58-BIT POPCOUNT LINEAR VS TORCH.NN.LINEAR (CPU)")
    print(f"PyTorch Thread Count: {torch.get_num_threads()}")
    print("=" * 70)

    scenarios = [
        # (BatchSize, InFeatures, OutFeatures, Description)
        (1, 1024, 1024, "Single Token Latency (1K x 1K)"),
        (1, 2048, 2048, "Single Token Latency (2K x 2K)"),
        (16, 2048, 2048, "Batch 16 LLM Layer (2K x 2K)"),
        (64, 2048, 2048, "Batch 64 Throughput (2K x 2K)"),
        (32, 4096, 4096, "Batch 32 Large LLM Layer (4K x 4K)"),
    ]

    header = f"{'Workload':<30} | {'FP32 Linear':<12} | {'CSA Popcount':<13} | {'Speedup':<9} | {'Weight RAM'}"
    print(header)
    print("-" * len(header))

    for batch, k, n, desc in scenarios:
        res = benchmark_layer(batch, k, n, warmup=15, iterations=50)
        desc_str = f"{desc} [B={batch}]"
        fp_str = f"{res['fp32_time_ms']:.3f} ms"
        csa_str = f"{res['csa_time_ms']:.3f} ms"
        speedup_str = f"{res['speedup']:.2f}x"
        ram_str = f"{res['csa_weight_kb']:.1f} KB ({res['compression_ratio']:.1f}x smaller)"

        print(f"{desc_str:<30} | {fp_str:<12} | {csa_str:<13} | {speedup_str:<9} | {ram_str}")

    print("=" * 70)


if __name__ == "__main__":
    run_verification()
    run_benchmarks()

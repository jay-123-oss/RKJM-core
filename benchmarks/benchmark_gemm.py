import time
import torch
from torch.utils.cpp_extension import load

# 1. Compile C++ Extension
gemm_cpp = load(
    name="gemm_cpp",
    sources=["gemm_kernel.cpp"],
    extra_cflags=["-O3", "-mavx2", "-mfma", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False
)

# Matrix dimensions (N x N)
N = 1024
A = torch.randn(N, N, dtype=torch.float32)
B = torch.randn(N, N, dtype=torch.float32)
C_custom = torch.zeros(N, N, dtype=torch.float32)

# Total Floating Point Operations for GEMM = 2 * M * N * K
total_flops = 2.0 * (N ** 3)

# Warmup Runs
for _ in range(3):
    gemm_cpp.custom_gemm(A, B, C_custom)
    C_native = torch.matmul(A, B)

# Benchmark Custom OpenMP + AVX2 Kernel
iterations = 10
start = time.perf_counter()
for _ in range(iterations):
    gemm_cpp.custom_gemm(A, B, C_custom)
end = time.perf_counter()

custom_time = (end - start) / iterations
custom_gflops = (total_flops / custom_time) / 1e9

# Benchmark PyTorch Native (MKL / OneDNN)
start = time.perf_counter()
for _ in range(iterations):
    C_native = torch.matmul(A, B)
end = time.perf_counter()

native_time = (end - start) / iterations
native_gflops = (total_flops / native_time) / 1e9

# Validation Check
is_correct = torch.allclose(C_custom, C_native, atol=1e-3)

print("=" * 50)
print(f"Matrix Size: {N}x{N}")
print(f"Correctness Check: {'PASS' if is_correct else 'FAIL'}")
print("=" * 50)
print(f"Custom Kernel (AVX2+OMP): {custom_time*1000:.2f} ms | {custom_gflops:.2f} GFLOPS")
print(f"PyTorch Native (MKL/oneDNN): {native_time*1000:.2f} ms | {native_gflops:.2f} GFLOPS")
print("=" * 50)
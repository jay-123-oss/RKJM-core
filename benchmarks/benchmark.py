import time
import torch
from torch.utils.cpp_extension import load

print("Compiling C++ Extension with OpenMP & AVX2...")
custom_ext = load(
    name="custom_extension",
    sources=["kernel.cpp"],
    extra_cflags=["-O3", "-fopenmp", "-mavx2", "-mfma"],
    extra_ldflags=["-fopenmp"],
    verbose=False
)
print("Compilation Successful!\n")

N = 10_000_000  # 10 Million elements
device = torch.device("cpu")

x = torch.randn(N, dtype=torch.float32, device=device)
out_cpp = torch.empty(N, dtype=torch.float32, device=device)

# Warmup run
custom_ext.custom_kernel_omp(x, out_cpp)

# Benchmark execution
iterations = 10
start = time.perf_counter()
for _ in range(iterations):
    custom_ext.custom_kernel_omp(x, out_cpp)
end = time.perf_counter()

avg_latency_ms = ((end - start) / iterations) * 1000

# Verification against standard PyTorch
out_pytorch = x * 2.0 + 5.0
is_correct = torch.allclose(out_cpp, out_pytorch, atol=1e-5)

print("=" * 50)
print("--- BENCHMARK & VALIDATION RESULTS ---")
print("=" * 50)
print(f"Execution Device   : {device.type.upper()}")
print(f"Elements Processed : {N:,}")
print(f"Correctness Check  : {'PASS' if is_correct else 'FAIL'}")
print(f"Average Latency    : {avg_latency_ms:.4f} ms")
print("=" * 50)
"""Benchmark the Watch Grid Triton model against dense PyTorch matmul.

Examples
--------
CUDA benchmark using the requested size sweep::

    .venv/bin/python benchmark_watch_grid.py --sizes 1024 2048 4096 8192

Print planning numbers without allocating a GPU::

    .venv/bin/python benchmark_watch_grid.py --mock

The mock table is illustrative, not a measurement.  Real results depend on
GPU architecture, clocks, compiler version, and the selected tile sizes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import torch

try:
    import triton

    from hybrid_bitwise_matmul import (
        TRITON_AVAILABLE,
        hybrid_bitwise_matmul,
        hybrid_ste_backward,
        pack_ternary_torch,
    )
except ImportError:
    triton = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False
    from hybrid_bitwise_matmul import hybrid_bitwise_matmul, hybrid_ste_backward, pack_ternary_torch


@dataclass
class Measurement:
    size: int
    path: str
    forward_ms: float
    backward_ms: float
    forward_tops: float
    backward_tops: float
    forward_gbs: float
    backward_gbs: float


def _time_cuda(function: Callable[[], torch.Tensor], warmup: int, repetitions: int) -> float:
    """Use Triton's calibrated CUDA timer, falling back to CUDA events."""

    if TRITON_AVAILABLE and triton is not None:
        return float(triton.testing.do_bench(function, warmup=warmup, rep=repetitions))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repetitions)


def _tops(size: int, milliseconds: float) -> float:
    return (2.0 * size**3) / (milliseconds * 1.0e9)


def _gbs(byte_count: int, milliseconds: float) -> float:
    return byte_count / (milliseconds * 1.0e6)


def benchmark_size(size: int, warmup: int = 25, repetitions: int = 100) -> List[Measurement]:
    """Benchmark one square M=N=K problem in FP16 and FP32 reference modes."""

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for real benchmarks; use --mock on CPU hosts")
    device = torch.device("cuda")
    torch.manual_seed(11)
    x_fp16 = torch.randn((size, size), device=device, dtype=torch.float16)
    x_fp32 = x_fp16.float()
    weights_q = torch.randint(-1, 2, (size, size), device=device, dtype=torch.int32)
    packed = pack_ternary_torch(weights_q)
    dense_fp16 = weights_q.to(torch.float16)
    dense_fp32 = weights_q.to(torch.float32)
    grad_y = torch.randn((size, size), device=device, dtype=torch.float32)
    latent = torch.randn((size, size), device=device, dtype=torch.float32)

    cases = [
        ("hybrid-bitwise", x_fp16, lambda: hybrid_bitwise_matmul(x_fp16, packed)),
        ("torch-fp16", x_fp16, lambda: torch.matmul(x_fp16, dense_fp16.t())),
        ("torch-fp32", x_fp32, lambda: torch.matmul(x_fp32, dense_fp32.t())),
    ]
    results: List[Measurement] = []
    for name, activation, forward in cases:
        forward_ms = _time_cuda(forward, warmup, repetitions)
        if name == "hybrid-bitwise":
            backward = lambda: hybrid_ste_backward(grad_y, activation.float(), latent)
            forward_bytes = size * size * activation.element_size() + packed.numel() * 4 + size * size * 4
            backward_bytes = size * size * (4 + 4 + 4 + 4)
        else:
            backward = lambda: torch.matmul(grad_y.t(), activation.float())
            forward_bytes = size * size * activation.element_size() * 2 + size * size * 4
            backward_bytes = size * size * (4 + activation.element_size() + 4)
        backward_ms = _time_cuda(backward, warmup, repetitions)
        results.append(
            Measurement(
                size=size,
                path=name,
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                forward_tops=_tops(size, forward_ms),
                backward_tops=_tops(size, backward_ms),
                forward_gbs=_gbs(forward_bytes, forward_ms),
                backward_gbs=_gbs(backward_bytes, backward_ms),
            )
        )
    return results


def print_results(results: Sequence[Measurement]) -> None:
    print("Measured CUDA results")
    print("size  path             fwd_ms  bwd_ms  fwd_TOPS  bwd_TOPS  fwd_GB/s  bwd_GB/s")
    for result in results:
        print(
            f"{result.size:4d}  {result.path:16s} "
            f"{result.forward_ms:7.3f} {result.backward_ms:7.3f} "
            f"{result.forward_tops:8.2f} {result.backward_tops:8.2f} "
            f"{result.forward_gbs:9.1f} {result.backward_gbs:9.1f}"
        )


def print_mock_table() -> None:
    """Print a planning table; values are intentionally labeled simulated."""

    print("\nSimulated planning results (not measured on this host)")
    print("size   hybrid_ms  fp16_ms  fp32_ms  hybrid_TOPS  fp16_TFLOPS  rationale")
    rows = [
        (1024, 0.18, 0.10, 0.19, 11.9, 21.5, "packing reduces weight traffic"),
        (2048, 0.62, 0.74, 1.35, 27.7, 23.2, "bitwise path scales with compressed weights"),
        (4096, 2.45, 5.72, 10.6, 56.2, 24.0, "memory pressure dominates dense paths"),
        (8192, 9.80, 44.8, 83.0, 112.4, 24.5, "packed reads avoid most weight bandwidth"),
    ]
    for size, hybrid, fp16, fp32, tops, tflops, rationale in rows:
        print(f"{size:4d}   {hybrid:9.2f} {fp16:8.2f} {fp32:8.2f} {tops:11.1f} {tflops:12.1f}  {rationale}")
    print("Bandwidth figures must be collected from the measured table; the mock table only illustrates scaling.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--mock", action="store_true", help="print the simulated planning table only")
    args = parser.parse_args()

    if args.mock:
        print_mock_table()
        return
    if not torch.cuda.is_available():
        print("No CUDA device detected; real Triton timing was not run.")
        print_mock_table()
        return

    measurements: List[Measurement] = []
    for size in args.sizes:
        print(f"Benchmarking M=N=K={size} ...")
        measurements.extend(benchmark_size(size, args.warmup, args.repetitions))
    print_results(measurements)
    print_mock_table()


if __name__ == "__main__":
    main()
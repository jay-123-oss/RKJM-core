"""
Production Benchmark & Load Testing Script for RKMJ-Core Inference Server.
Measures Time-to-First-Token (TTFT), Inter-Token Latency (ITL), and Requests-Per-Second (RPS).
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any

import torch

from rkmj.models.config import RKMJConfig
from rkmj.models.llama import RKMJLlamaForCausalLM
from rkmj.server.engine import AsyncInferenceEngine


async def benchmark_client(
    engine: AsyncInferenceEngine,
    request_id: int,
    prompt_len: int = 16,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    prompt = [10 + (i % 100) for i in range(prompt_len)]
    t_start = time.perf_counter()
    first_token_time = None
    token_times: List[float] = []

    last_time = t_start
    token_count = 0

    async for token_id, is_finished, _ in engine.generate_stream(
        prompt_token_ids=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
    ):
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = (now - t_start) * 1000.0  # ms
        else:
            token_times.append((now - last_time) * 1000.0)  # ms
        last_time = now
        token_count += 1
        if is_finished:
            break

    total_time = (time.perf_counter() - t_start) * 1000.0  # ms
    avg_itl = statistics.mean(token_times) if token_times else 0.0

    return {
        "request_id": request_id,
        "token_count": token_count,
        "ttft_ms": first_token_time or total_time,
        "avg_itl_ms": avg_itl,
        "total_time_ms": total_time,
    }


async def run_benchmark(
    concurrency_levels: List[int] = [1, 4, 8],
    tokens_per_req: int = 24,
):
    print("=" * 70)
    print("🚀 RKMJ-Core 1.58-bit Inference Server Load Benchmark")
    print("=" * 70)

    # Initialize small model for fast CPU benchmarking
    config = RKMJConfig(vocab_size=1000, dim=128, n_layers=4, n_heads=4, max_seq_len=512)
    model = RKMJLlamaForCausalLM(config)
    model.pack_weights_for_inference()
    model.eval()

    engine = AsyncInferenceEngine(model, max_batch_size=16)
    engine.start()

    # Warmup
    print("Warming up inference engine...")
    await benchmark_client(engine, request_id=0, max_new_tokens=4)
    print("Warmup complete.\n")

    results = []

    for concurrency in concurrency_levels:
        num_requests = concurrency * 2
        print(f"Testing Concurrency = {concurrency} ({num_requests} total requests)...")

        t_bench_start = time.perf_counter()
        tasks = [
            benchmark_client(engine, request_id=i, max_new_tokens=tokens_per_req)
            for i in range(num_requests)
        ]
        metrics = await asyncio.gather(*tasks)
        bench_duration = time.perf_counter() - t_bench_start

        ttfts = [m["ttft_ms"] for m in metrics]
        itls = [m["avg_itl_ms"] for m in metrics if m["avg_itl_ms"] > 0]
        total_tokens = sum(m["token_count"] for m in metrics)

        p50_ttft = statistics.median(ttfts)
        p95_ttft = sorted(ttfts)[int(0.95 * len(ttfts))]
        p50_itl = statistics.median(itls) if itls else 0.0
        rps = num_requests / bench_duration
        tps = total_tokens / bench_duration

        results.append({
            "concurrency": concurrency,
            "rps": rps,
            "tps": tps,
            "p50_ttft_ms": p50_ttft,
            "p95_ttft_ms": p95_ttft,
            "p50_itl_ms": p50_itl,
        })

    await engine.stop()

    print("\n" + "=" * 70)
    print(f"{'Concurrency':<12} | {'RPS':<8} | {'TPS':<8} | {'TTFT p50 (ms)':<14} | {'ITL p50 (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['concurrency']:<12} | {r['rps']:<8.2f} | {r['tps']:<8.1f} | "
            f"{r['p50_ttft_ms']:<14.2f} | {r['p50_itl_ms']:<12.2f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

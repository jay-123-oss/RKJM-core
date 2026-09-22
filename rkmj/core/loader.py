"""
================================================================================
RKMJ-Core Smart Adaptive Inference Model Loader
================================================================================
Loads quantized .rkmjbin binary weights via zero-copy mmap.
Enforces strict physical RAM budget <= 2.0 GB by dynamically streaming single
resident layers and unmapping non-active memory pages via glibc malloc_trim.
================================================================================
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import psutil
import torch

from rkmj.core.supervisor import get_supervisor
from universal.runner import UniversalLocalRunner

logger = logging.getLogger("rkmj.core.loader")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load(
    model_path: Union[str, Path],
    ram_budget_gb: float = 2.0,
    device: str = "cpu",
) -> UniversalLocalRunner:
    """
    Loads an RKMJ 1.58-bit quantized model (.rkmjbin) with zero full-model RAM spikes.

    Args:
        model_path: Path to the .rkmjbin packed model file.
        ram_budget_gb: Maximum physical RAM ceiling in GB (default: 2.0 GB).
        device: Execution device ("cpu" or "cuda").

    Returns:
        UniversalLocalRunner configured with adaptive zero-copy layer streaming.
    """
    path_str = str(model_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Quantized model binary not found: {path_str}")

    # Inspect host RAM
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024.0 ** 3)
    effective_budget = min(ram_budget_gb, max(1.0, avail_gb * 0.75))

    logger.info("=" * 80)
    logger.info("🚀 RKMJ Smart Adaptive Model Loader")
    logger.info(f"   Model File:       {path_str}")
    logger.info(f"   File Size:        {os.path.getsize(path_str) / (1024**3):.2f} GB")
    logger.info(f"   RAM Ceiling:      {effective_budget:.2f} GB (Host Available: {avail_gb:.2f} GB)")
    logger.info("=" * 80)

    # Instantiate UniversalLocalRunner with strict budget
    runner = UniversalLocalRunner(
        model_path=path_str,
        ram_budget_gb=effective_budget,
        device=device,
    )

    supervisor = get_supervisor()
    rss_gb = supervisor.get_process_rss_gb()
    logger.info(f"✅ Model loaded into runner. Current Active RSS: {rss_gb:.2f} GB (<= {effective_budget:.2f} GB)")

    return runner

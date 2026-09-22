"""
CLI Tool: Convert Hugging Face Safetensors Qwen Models to 1.58-bit .rkmjbin.
Usage:
    python qween/convert.py --input /path/to/Qwen2.5-32B --output /path/to/qwen-32b-ternary.rkmjbin
"""

import argparse
import os
import sys
from pathlib import Path

# 1. Auto-detect project virtualenv if global python without torch is invoked
try:
    import torch
except ImportError:
    for candidate in [
        Path(__file__).resolve().parent.parent / "myenv" / "bin" / "python",
        Path(__file__).resolve().parent.parent.parent / "myenv" / "bin" / "python",
    ]:
        if candidate.exists() and sys.executable != str(candidate):
            os.execv(str(candidate), [str(candidate)] + sys.argv)

# Bootstrap sys.path to allow running from both repo root and rkmj-core
_current_dir = Path(__file__).resolve().parent
_rkmj_core_dir = _current_dir.parent if (_current_dir.parent / "rkmj").exists() else _current_dir.parent / "rkmj-core"
_parent_dir = _current_dir.parent

for p in [str(_current_dir.parent), str(_rkmj_core_dir), str(_current_dir)]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from qween.bootstrap import check_dependencies, ensure_dependencies
from qween.converter import QwenStreamingPTQConverter


def main():
    # Bootstrap check
    _missing, _ = check_dependencies()
    if _missing:
        ensure_dependencies(auto_install=True)

    parser = argparse.ArgumentParser(
        description="Out-of-Core Streaming PTQ Converter for Qwen Architecture"
    )
    parser.add_argument(
        "--input",
        "-i",
        "--model-dir",
        dest="input",
        required=True,
        help="Path to local directory OR Hugging Face repo ID (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Target output path for the packed .rkmjbin binary file",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=3.5,
        help="Strict physical DRAM budget ceiling in GB (default: 3.5)",
    )

    args = parser.parse_args()

    # Ensure parent output directory exists
    out_parent = os.path.dirname(os.path.abspath(args.output))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    converter = QwenStreamingPTQConverter(
        model_dir=args.input,
        output_path=args.output,
        max_ram_bytes=int(args.max_ram_gb * 1024 * 1024 * 1024),
    )

    try:
        stats = converter.convert()
        print("\nConversion succeeded:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nError during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

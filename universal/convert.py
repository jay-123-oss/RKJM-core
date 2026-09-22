"""
Universal Streaming PTQ Conversion CLI.
Converts any Hugging Face LLM (Qwen, LLaMA, Mistral, Gemma) into 1.58-bit .rkmjbin.
"""

import argparse
import sys
from universal.converter import UniversalStreamingPTQConverter


def main():
    parser = argparse.ArgumentParser(description="Universal 1.58-bit Streaming PTQ Converter")
    parser.add_argument("--model-dir", "-m", type=str, required=True, help="Path to Hugging Face model directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path for output .rkmjbin file")
    parser.add_argument("--max-ram-gb", type=float, default=3.5, help="Physical RAM ceiling in GB (default: 3.5)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")

    args = parser.parse_args()
    max_ram_bytes = int(args.max_ram_gb * 1024 * 1024 * 1024)

    converter = UniversalStreamingPTQConverter(
        model_dir=args.model_dir,
        output_path=args.output,
        max_ram_bytes=max_ram_bytes,
        group_size=args.group_size,
    )
    converter.convert()


if __name__ == "__main__":
    main()

import os
import sys
import platform
import shutil
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Detect CUDA availability
has_cuda = (torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1") and shutil.which("nvcc") is not None

arch = platform.machine().lower()

cxx_args = [
    "-std=c++20",
    "-O3",
    "-fopenmp",
    "-fPIC",
    "-Wall",
]

if arch in ("x86_64", "amd64"):
    cxx_args.extend([
        "-mavx2",
        "-mfma",
        "-mbmi2",
    ])
elif arch in ("aarch64", "arm64"):
    cxx_args.extend([
        "-march=armv8-a+simd",
    ])

extra_link_args = [
    "-fopenmp",
]

sources = [
    "csrc/bindings.cpp",
    "csrc/allocator.cpp",
    "csrc/pack.cpp",
    "csrc/kernels_avx2.cpp",
    "csrc/scheduler.cpp",
    "csrc/autograd_ste.cpp",
    "csrc/cpu/csa_linear_cpu.cpp",
    "csrc/cpu/csa_autograd_cpu.cpp",
    "csrc/cuda/bitwise_host.cpp",
    "csrc/io/mmap_streamer.cpp",
]

if has_cuda:
    print("[INFO] Building RKMJ-Core with CUDA hardware acceleration enabled.")
    cxx_args.append("-DWITH_CUDA")
    sources.append("csrc/cuda/bitwise_kernel.cu")
    ext_modules = [
        CUDAExtension(
            name="rkmj._C",
            sources=sources,
            include_dirs=[
                os.path.abspath("csrc/include"),
            ],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--use_fast_math",
                ],
            },
            extra_link_args=extra_link_args,
        )
    ]
else:
    print(f"[INFO] Building RKMJ-Core in CPU mode with multi-ISA vectorization on {arch}.")
    ext_modules = [
        CppExtension(
            name="rkmj._C",
            sources=sources,
            include_dirs=[
                os.path.abspath("csrc/include"),
            ],
            extra_compile_args=cxx_args,
            extra_link_args=extra_link_args,
        )
    ]

setup(
    name="rkmj",
    version="0.1.0",
    description="RKMJ-Core: 1.58-bit Carry-Save Addition (CSA) Transformer Framework (CPU & CUDA)",
    author="RKMJ AI Team",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
    ],
)

# RKMJ-Core

> **A Revolutionary 1.58-bit (Ternary) Large Language Model Framework Optimized for Ultra-Fast CPU Execution.**

RKMJ-Core replaces heavy FP32 matrix multiplications (GEMM) with **Carry-Save Addition (CSA)** and bitwise **popcount** hardware instructions, achieving:
- **15.8x Memory Compression Ratio** compared to standard FP32 weights.
- **Ultra-Fast CPU Inference** powered by AVX2, BMI2 (`_pext_u64`), and OpenMP multi-threading.
- **End-to-End PyTorch Drop-in API** (`rkmj.nn.CSALinear`, `rkmj.nn.CSATransformerBlock`).
- **Full Training Capability** using custom C++ Straight-Through Estimator (STE) autograd.

---

## 📁 Repository Structure

```
rkmj-core/
├── csrc/                             # Native C++ Hardware Kernels (OpenMP)
│   ├── include/
│   │   ├── csa_common.h              # Packing & popcount utilities
│   │   └── csa_autograd.h            # STE gradient calculation headers
│   ├── cpu/
│   │   ├── csa_linear_cpu.cpp        # Optimized OpenMP forward pass
│   │   └── csa_autograd_cpu.cpp      # Multithreaded STE backward pass
│   └── bindings.cpp                  # PyTorch PyBind11 bindings
│
├── rkmj/                             # Main Framework Python Package
│   ├── __init__.py                   # Core exports: from rkmj import nn, models
│   │
│   ├── nn/                           # Drop-in Replacement Neural Network Layers
│   │   ├── __init__.py
│   │   ├── linear.py                 # rkmj.nn.CSALinear (calls C++ engine)
│   │   ├── attention.py              # 1.58-bit Multi-Head / Grouped Query Attention
│   │   ├── mlp.py                    # 1.58-bit SwiGLU FeedForward module
│   │   ├── norm.py                   # RMSNorm implementation
│   │   └── block.py                  # rkmj.nn.CSATransformerBlock
│   │
│   ├── models/                       # Full Model Architectures
│   │   ├── __init__.py
│   │   ├── base.py                   # Base generative model class
│   │   ├── llama.py                  # Llama-style 1.58-bit Generative LLM
│   │   └── config.py                 # RKMJConfig (dataclass for parameters)
│   │
│   ├── serialization/                # Bit-Packing & Model Export
│   │   ├── __init__.py
│   │   ├── packer.py                 # FP32 <-> 2-bit ternary pack/unpack routines
│   │   └── rkmjbin.py                # Native .rkmjbin binary format reader/writer
│   │
│   └── engine/                       # Generation & Samplers
│       ├── __init__.py
│       ├── sampler.py                # Top-K, Top-P, Temperature sampling
│       └── generator.py              # Text generation stream / pipeline
│
├── rust_core/                        # wave_brain_core Integration
│   ├── Cargo.toml                    # Rust build config
│   └── src/lib.rs                    # Unsupervised context learning & symbolic loops
│
├── examples/                         # End-to-end Runnable Demos
│   ├── train_shakespeare.py          # 1.58-bit Toy LLM training script
│   ├── interactive_chat.py           # Interactive CLI chat loop
│   └── benchmark_cpu.py              # FP32 vs RKMJ Popcount benchmark
│
├── tests/                            # Unit & Mathematical Tests
│   ├── test_csa_linear.py            # Forward correctness tests
│   ├── test_backward_ste.py          # Autograd gradient flow tests
│   └── test_packing.py               # Bit-packing fidelity checks
│
├── .gitignore
├── pyproject.toml
├── setup.py                          # Compiles C++ extension as `rkmj._C`
└── README.md                         # Framework documentation & benchmarks
```

---

## ⚡ Quick Start

### 1. Build and Install C++ Extension
```bash
cd rkmj-core
pip install -e .
```

### 2. Train a 1.58-bit LLaMA Model on CPU
```bash
python examples/train_shakespeare.py
```

### 3. Interactive CLI Generation
```bash
python examples/interactive_chat.py --pack
```

### 4. Run CPU Benchmarks
```bash
python examples/benchmark_cpu.py
```

### 5. Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 🔬 Core Math: 1.58-bit Popcount Engine

Weights are quantized to $\{-1, 0, +1\}$ and packed into 32-bit integers using 2-bit encoding:
- `00` = 0
- `01` = +1
- `10` = -1

Activations are binarized by sign into bitmasks. The matrix multiplication is calculated via:
$$\text{Output} = \alpha \cdot \Big(\text{popcount}(\text{pos\_hits}) - \text{popcount}(\text{neg\_hits})\Big) + \text{bias}$$

Hardware acceleration uses Intel/AMD **BMI2 `_pext`** instructions and OpenMP multi-threading.

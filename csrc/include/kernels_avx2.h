#pragma once

#include <torch/extension.h>
#include <cstdint>

namespace rkmj {
namespace cpu {

// Tiled Cache-Aware GEMV on 2-bit Packed Weights
// Handles single-token autoregressive decoding (Batch size B = 1, sequence length M = 1).
torch::Tensor gemv_tiled_csa_cpu(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias,
    int64_t group_size = 64
);

// Tiled Cache-Aware GEMM on 2-bit Packed Weights
// Handles sequence prefill (Sequence length M > 1).
torch::Tensor gemm_tiled_csa_cpu(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias,
    int64_t group_size = 64
);

} // namespace cpu
} // namespace rkmj

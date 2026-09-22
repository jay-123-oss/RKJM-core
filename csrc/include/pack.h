#pragma once

#include <cstdint>
#include <vector>
#include <torch/extension.h>

namespace rkmj {
namespace quant {

// Packs 2D ternary tensor [-1, 0, +1] into 2-bit aligned uint32 bitfield
// 16 weights per uint32 word.
// Bit encoding: 00 = 0, 01 = +1, 10 = -1
torch::Tensor pack_ternary_2bit_cpu(torch::Tensor w_ternary);

// Unpacks 2D packed uint32 bitfield into ternary tensor [-1, 0, +1]
// Uses arithmetic right-shifts and bitwise masking to guarantee sign accuracy.
torch::Tensor unpack_ternary_2bit_cpu(torch::Tensor w_packed, int64_t K);

// Per-Group Quantization:
// Computes gamma_g = mean(|W_g|), applies threshold eps = 0.5 * gamma_g,
// outputs packed bitfield and per-group scales alpha_g.
std::pair<torch::Tensor, torch::Tensor> quantize_grouped_cpu(
    torch::Tensor weight,
    int64_t group_size = 64
);

// Dequantizes per-group packed weights into FP32 / BF16
torch::Tensor dequantize_grouped_cpu(
    torch::Tensor w_packed,
    torch::Tensor scales,
    int64_t in_features,
    int64_t group_size = 64
);

} // namespace quant
} // namespace rkmj

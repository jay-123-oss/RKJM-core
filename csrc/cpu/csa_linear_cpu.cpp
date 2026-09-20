#include "../include/csa_common.h"
#include <torch/extension.h>
#include <omp.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace rkmj {
namespace csa {

// -----------------------------------------------------------------------------
// csa_linear_forward: Pure Bitwise Carry-Save Addition Forward Pass
//
// Math:
//   y = alpha * (popcount(pos_hits) - popcount(neg_hits)) + bias
//
// No FP32 matrix multiplication (GEMM) is performed.
// OpenMP parallelized across batches and output features.
// -----------------------------------------------------------------------------

torch::Tensor csa_linear_forward(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor alpha,
    c10::optional<torch::Tensor> bias = c10::nullopt
) {
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(w_packed.is_cpu(), "w_packed must be a CPU tensor");
    TORCH_CHECK(alpha.is_cpu(), "alpha must be a CPU tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w_packed.is_contiguous(), "w_packed must be contiguous");
    TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w_packed.scalar_type() == torch::kInt32, "w_packed must be int32 (uint32_t bitfields)");
    TORCH_CHECK(alpha.scalar_type() == torch::kFloat32, "alpha must be float32");

    const bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        TORCH_CHECK(bias.value().is_cpu(), "bias must be a CPU tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().scalar_type() == torch::kFloat32, "bias must be float32");
    }

    const auto orig_sizes = x.sizes();
    const int64_t K = x.size(-1);
    const int64_t B = x.numel() / K;

    TORCH_CHECK(w_packed.dim() == 2, "w_packed must be 2D [N, num_words]");
    const int64_t N = w_packed.size(0);
    const int64_t num_words = w_packed.size(1);
    const int64_t expected_words = (K + 15) / 16;
    TORCH_CHECK(num_words == expected_words, "w_packed columns mismatch feature dimension K");

    const bool alpha_is_per_channel = (alpha.numel() == N);
    TORCH_CHECK(alpha.numel() == 1 || alpha_is_per_channel, "alpha must have 1 or N elements");

    if (has_bias) {
        TORCH_CHECK(bias.value().numel() == N, "bias must have N elements");
    }

    // Allocate output tensor
    std::vector<int64_t> out_sizes(orig_sizes.begin(), orig_sizes.end() - 1);
    out_sizes.push_back(N);
    auto y = torch::empty(out_sizes, x.options());

    // -------------------------------------------------------------------------
    // Step 1: Pack Activation Signs (x >= 0 ? 1 : 0) into 16-bit bitfields
    // -------------------------------------------------------------------------
    std::vector<uint16_t> x_packed(B * num_words);
    const float* x_ptr = x.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        pack_activation_signs_row(x_ptr + b * K, &x_packed[b * num_words], K);
    }

    // -------------------------------------------------------------------------
    // Step 2: Bitwise CSA Popcount Accumulation with OpenMP
    // -------------------------------------------------------------------------
    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    const float* alpha_ptr = alpha.data_ptr<float>();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();

    const int64_t num_words_64 = num_words / 2;
    const bool has_tail = (K % 16 != 0);
    const uint32_t tail_mask = has_tail ? ((1U << (K % 16)) - 1U) : 0xFFFFU;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            const uint16_t* x_row = &x_packed[b * num_words];
            const uint32_t* w_row = &w_ptr[n * num_words];
            const float a = alpha_is_per_channel ? alpha_ptr[n] : alpha_ptr[0];
            const float b_val = has_bias ? bias_ptr[n] : 0.0f;

            int32_t total_diff = 0;

#if defined(__BMI2__)
            const uint64_t* w_row64 = reinterpret_cast<const uint64_t*>(w_row);
            const uint32_t* x_row32 = reinterpret_cast<const uint32_t*>(x_row);

            int64_t w64 = 0;
            // 4x unrolled 64-bit word accumulator (64 weights per loop iteration)
            for (; w64 + 3 < num_words_64; w64 += 4) {
                uint64_t w0 = w_row64[w64];
                uint64_t w1 = w_row64[w64 + 1];
                uint64_t w2 = w_row64[w64 + 2];
                uint64_t w3 = w_row64[w64 + 3];

                uint64_t x0 = x_row32[w64];
                uint64_t x1 = x_row32[w64 + 1];
                uint64_t x2 = x_row32[w64 + 2];
                uint64_t x3 = x_row32[w64 + 3];

                uint64_t p0 = extract_even_bits_u64(w0);
                uint64_t n0 = extract_odd_bits_u64(w0);
                uint64_t p1 = extract_even_bits_u64(w1);
                uint64_t n1 = extract_odd_bits_u64(w1);
                uint64_t p2 = extract_even_bits_u64(w2);
                uint64_t n2 = extract_odd_bits_u64(w2);
                uint64_t p3 = extract_even_bits_u64(w3);
                uint64_t n3 = extract_odd_bits_u64(w3);

                uint64_t pos0 = (p0 & x0) | (n0 & ~x0);
                uint64_t neg0 = (n0 & x0) | (p0 & ~x0);
                uint64_t pos1 = (p1 & x1) | (n1 & ~x1);
                uint64_t neg1 = (n1 & x1) | (p1 & ~x1);
                uint64_t pos2 = (p2 & x2) | (n2 & ~x2);
                uint64_t neg2 = (n2 & x2) | (p2 & ~x2);
                uint64_t pos3 = (p3 & x3) | (n3 & ~x3);
                uint64_t neg3 = (n3 & x3) | (p3 & ~x3);

                total_diff += (__builtin_popcountll(pos0) - __builtin_popcountll(neg0))
                            + (__builtin_popcountll(pos1) - __builtin_popcountll(neg1))
                            + (__builtin_popcountll(pos2) - __builtin_popcountll(neg2))
                            + (__builtin_popcountll(pos3) - __builtin_popcountll(neg3));
            }

            for (; w64 < num_words_64; ++w64) {
                uint64_t w_val = w_row64[w64];
                uint64_t x_val = x_row32[w64];
                uint64_t pos_mask = extract_even_bits_u64(w_val);
                uint64_t neg_mask = extract_odd_bits_u64(w_val);

                if (w64 == num_words_64 - 1 && (num_words % 2 == 0) && has_tail) {
                    uint64_t full_tail = 0xFFFFFFFFULL | (static_cast<uint64_t>(tail_mask) << 16);
                    pos_mask &= full_tail;
                    neg_mask &= full_tail;
                }

                uint64_t pos_hits = (pos_mask & x_val) | (neg_mask & ~x_val);
                uint64_t neg_hits = (neg_mask & x_val) | (pos_mask & ~x_val);
                total_diff += __builtin_popcountll(pos_hits) - __builtin_popcountll(neg_hits);
            }

            int64_t w = num_words_64 * 2;
#else
            int64_t w = 0;
#endif
            for (; w < num_words; ++w) {
                uint32_t word = w_row[w];
                uint32_t pos_mask = extract_even_bits_u32(word);
                uint32_t neg_mask = extract_odd_bits_u32(word);
                uint32_t x_mask = x_row[w];

                if (w == num_words - 1 && has_tail) {
                    pos_mask &= tail_mask;
                    neg_mask &= tail_mask;
                }

                uint32_t pos_hits = (pos_mask & x_mask) | (neg_mask & ~x_mask);
                uint32_t neg_hits = (neg_mask & x_mask) | (pos_mask & ~x_mask);
                total_diff += __builtin_popcount(pos_hits) - __builtin_popcount(neg_hits);
            }

            y_ptr[b * N + n] = a * static_cast<float>(total_diff) + b_val;
        }
    }

    return y;
}

// -----------------------------------------------------------------------------
// Packing & Unpacking Helper Functions for Python Binding
// -----------------------------------------------------------------------------

torch::Tensor pack_weights_cpu(torch::Tensor w_ternary) {
    TORCH_CHECK(w_ternary.dim() == 2, "w_ternary must be 2D [N, K]");
    TORCH_CHECK(w_ternary.is_contiguous(), "w_ternary must be contiguous");
    const int64_t N = w_ternary.size(0);
    const int64_t K = w_ternary.size(1);
    const int64_t num_words = (K + 15) / 16;

    auto packed = torch::empty({N, num_words}, torch::kInt32);
    const float* src = w_ternary.data_ptr<float>();
    uint32_t* dst = reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>());

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        pack_ternary_row(src + n * K, dst + n * num_words, K);
    }
    return packed;
}

torch::Tensor unpack_weights_cpu(torch::Tensor w_packed, int64_t K) {
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must be 2D [N, num_words]");
    TORCH_CHECK(w_packed.is_contiguous(), "w_packed must be contiguous");
    const int64_t N = w_packed.size(0);

    auto unpacked = torch::empty({N, K}, torch::kFloat32);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    float* dst = unpacked.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        unpack_ternary_row(src + n * w_packed.size(1), dst + n * K, K);
    }
    return unpacked;
}

} // namespace csa
} // namespace rkmj

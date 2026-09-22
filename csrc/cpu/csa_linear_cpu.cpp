#include "../include/csa_common.h"
#include "../include/simd_isa.hpp"
#include "../include/fast_popcount.hpp"
#include "../include/fused_ops.hpp"
#include "../include/kv_cache_ops.hpp"

#include <torch/extension.h>
#include <omp.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace rkmj {
namespace csa {

// -----------------------------------------------------------------------------
// Core Tiled CSA GEMM Kernel Implementation
// -----------------------------------------------------------------------------

static inline void run_csa_gemm_tiled(
    const uint16_t* __restrict__ x_packed_ptr,
    const uint32_t* __restrict__ w_ptr,
    const float* __restrict__ alpha_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y_ptr,
    int64_t M,
    int64_t N,
    int64_t num_words,
    bool has_tail,
    uint32_t tail_mask,
    bool alpha_is_per_channel,
    bool has_bias,
    simd::SimdBackend backend
) {
    constexpr int64_t TILE_M = 32;
    constexpr int64_t TILE_N = 64;

    // Adaptive Thread Pooling Threshold:
    // If problem size is small (e.g. single-token decode with small N), avoid OpenMP overhead
    const bool use_parallel = (M * N >= 512);

    if (!use_parallel) {
        // Single-threaded fast path
        for (int64_t m = 0; m < M; ++m) {
            const uint16_t* x_row = x_packed_ptr + m * num_words;
            float* y_row = y_ptr + m * N;

            for (int64_t n = 0; n < N; ++n) {
                const uint32_t* w_row = w_ptr + n * num_words;
                const float a = alpha_is_per_channel ? alpha_ptr[n] : alpha_ptr[0];
                const float b_val = has_bias ? bias_ptr[n] : 0.0f;

                int32_t diff = simd::popcount_diff_dispatch(
                    w_row, x_row, num_words, has_tail, tail_mask, backend
                );
                y_row[n] = a * static_cast<float>(diff) + b_val;
            }
        }
        return;
    }

    if (M == 1) {
        // Single-token autoregressive decode step: parallelize strictly over N
        const uint16_t* x_row = x_packed_ptr;
        float* y_row = y_ptr;

        #pragma omp parallel for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            const uint32_t* w_row = w_ptr + n * num_words;
            const float a = alpha_is_per_channel ? alpha_ptr[n] : alpha_ptr[0];
            const float b_val = has_bias ? bias_ptr[n] : 0.0f;

            int32_t diff = simd::popcount_diff_dispatch(
                w_row, x_row, num_words, has_tail, tail_mask, backend
            );
            y_row[n] = a * static_cast<float>(diff) + b_val;
        }
        return;
    }

    // 2D Cache-Blocked Tiled Loop (L1/L2 Cache Conscious)
    const int64_t num_m_tiles = (M + TILE_M - 1) / TILE_M;
    const int64_t num_n_tiles = (N + TILE_N - 1) / TILE_N;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t tm = 0; tm < num_m_tiles; ++tm) {
        for (int64_t tn = 0; tn < num_n_tiles; ++tn) {
            const int64_t m_start = tm * TILE_M;
            const int64_t m_end = std::min(m_start + TILE_M, M);
            const int64_t n_start = tn * TILE_N;
            const int64_t n_end = std::min(n_start + TILE_N, N);

            for (int64_t m = m_start; m < m_end; ++m) {
                const uint16_t* x_row = x_packed_ptr + m * num_words;
                float* y_row = y_ptr + m * N;

                for (int64_t n = n_start; n < n_end; ++n) {
                    const uint32_t* w_row = w_ptr + n * num_words;
                    const float a = alpha_is_per_channel ? alpha_ptr[n] : alpha_ptr[0];
                    const float b_val = has_bias ? bias_ptr[n] : 0.0f;

                    int32_t diff = simd::popcount_diff_dispatch(
                        w_row, x_row, num_words, has_tail, tail_mask, backend
                    );
                    y_row[n] = a * static_cast<float>(diff) + b_val;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// csa_linear_forward: Hardened Multi-ISA CSA Popcount Forward Pass
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
    const int64_t M = x.numel() / K;

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

    // 64-byte aligned temporary activation sign bitfield buffer
    simd::AlignedBuffer<uint16_t, 64> x_packed(M * num_words);
    const float* x_ptr = x.data_ptr<float>();

    #pragma omp parallel for schedule(static) if (M > 4)
    for (int64_t m = 0; m < M; ++m) {
        pack_activation_signs_row(x_ptr + m * K, x_packed.data() + m * num_words, K);
    }

    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    const float* alpha_ptr = alpha.data_ptr<float>();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();

    const bool has_tail = (K % 16 != 0);
    const uint32_t tail_mask = has_tail ? ((1U << (K % 16)) - 1U) : 0xFFFFU;
    const simd::SimdBackend backend = simd::get_simd_backend();

    run_csa_gemm_tiled(
        x_packed.data(), w_ptr, alpha_ptr, bias_ptr, y_ptr,
        M, N, num_words, has_tail, tail_mask, alpha_is_per_channel, has_bias, backend
    );

    return y;
}

// -----------------------------------------------------------------------------
// fused_rmsnorm_csa_forward: Fused RMSNorm + Activation Quantization + CSA GEMM
// -----------------------------------------------------------------------------

torch::Tensor fused_rmsnorm_csa_forward(
    torch::Tensor x,
    torch::Tensor rmsnorm_weight,
    double eps,
    torch::Tensor w_packed,
    torch::Tensor alpha,
    c10::optional<torch::Tensor> bias = c10::nullopt
) {
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(rmsnorm_weight.is_cpu(), "rmsnorm_weight must be a CPU tensor");
    TORCH_CHECK(w_packed.is_cpu(), "w_packed must be a CPU tensor");
    TORCH_CHECK(alpha.is_cpu(), "alpha must be a CPU tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(rmsnorm_weight.is_contiguous(), "rmsnorm_weight must be contiguous");
    TORCH_CHECK(w_packed.is_contiguous(), "w_packed must be contiguous");
    TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(rmsnorm_weight.scalar_type() == torch::kFloat32, "rmsnorm_weight must be float32");
    TORCH_CHECK(w_packed.scalar_type() == torch::kInt32, "w_packed must be int32");
    TORCH_CHECK(alpha.scalar_type() == torch::kFloat32, "alpha must be float32");

    const bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        TORCH_CHECK(bias.value().is_cpu() && bias.value().is_contiguous(), "bias must be contiguous CPU");
    }

    const auto orig_sizes = x.sizes();
    const int64_t K = x.size(-1);
    const int64_t M = x.numel() / K;

    TORCH_CHECK(rmsnorm_weight.numel() == K, "rmsnorm_weight must have K elements");
    const int64_t N = w_packed.size(0);
    const int64_t num_words = w_packed.size(1);
    const int64_t expected_words = (K + 15) / 16;
    TORCH_CHECK(num_words == expected_words, "w_packed columns mismatch feature dimension K");

    const bool alpha_is_per_channel = (alpha.numel() == N);
    TORCH_CHECK(alpha.numel() == 1 || alpha_is_per_channel, "alpha must have 1 or N elements");

    std::vector<int64_t> out_sizes(orig_sizes.begin(), orig_sizes.end() - 1);
    out_sizes.push_back(N);
    auto y = torch::empty(out_sizes, x.options());

    // 64-byte aligned buffer for packed activation signs
    simd::AlignedBuffer<uint16_t, 64> x_packed(M * num_words);
    const float* x_ptr = x.data_ptr<float>();
    const float* gamma_ptr = rmsnorm_weight.data_ptr<float>();

    // Fused RMSNorm & 1-bit sign packing in L1 cache
    #pragma omp parallel for schedule(static) if (M > 4)
    for (int64_t m = 0; m < M; ++m) {
        fused::fused_rmsnorm_pack_activation_row(
            x_ptr + m * K, gamma_ptr, static_cast<float>(eps),
            x_packed.data() + m * num_words, K
        );
    }

    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    const float* alpha_ptr = alpha.data_ptr<float>();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;
    float* y_ptr = y.data_ptr<float>();

    const bool has_tail = (K % 16 != 0);
    const uint32_t tail_mask = has_tail ? ((1U << (K % 16)) - 1U) : 0xFFFFU;
    const simd::SimdBackend backend = simd::get_simd_backend();

    run_csa_gemm_tiled(
        x_packed.data(), w_ptr, alpha_ptr, bias_ptr, y_ptr,
        M, N, num_words, has_tail, tail_mask, alpha_is_per_channel, has_bias, backend
    );

    return y;
}

// -----------------------------------------------------------------------------
// Packing & Unpacking Helpers
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

std::string get_simd_backend_name() {
    return simd::backend_to_string(simd::get_simd_backend());
}

} // namespace csa
} // namespace rkmj

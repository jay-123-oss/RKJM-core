#pragma once

#include "simd_isa.hpp"
#include <cstdint>
#include <cmath>
#include <immintrin.h>

namespace rkmj {
namespace fused {

// -----------------------------------------------------------------------------
// Vectorized RMSNorm + 1-bit Sign Bit-Packing into L1 Cache
//
// Math:
//   rsqrt_scale = 1.0 / sqrt(mean(x^2) + eps)
//   x'_k = x_k * rsqrt_scale * gamma_k
//   sign_bit_k = (x'_k >= 0.0f) ? 1 : 0
//
// Never writes the intermediate unquantized x'_k to main DRAM.
// Packs directly into 16-bit bitfields (uint16_t) in L1 cache.
// -----------------------------------------------------------------------------

inline void fused_rmsnorm_pack_activation_row(
    const float* __restrict__ x_row,
    const float* __restrict__ gamma,
    float eps,
    uint16_t* __restrict__ dst_packed,
    int64_t K
) {
    // Step 1: Compute sum of squares sum(x^2)
    float sum_sq = 0.0f;
    int64_t k = 0;

#if defined(__AVX2__)
    __m256 v_sum256 = _mm256_setzero_ps();
    for (; k + 7 < K; k += 8) {
        __m256 vx = _mm256_loadu_ps(x_row + k);
        v_sum256 = _mm256_fmadd_ps(vx, vx, v_sum256);
    }
    // Horizontal add 8 floats in v_sum256
    __m128 v_low = _mm256_castps256_ps128(v_sum256);
    __m128 v_high = _mm256_extractf128_ps(v_sum256, 1);
    __m128 v_sum128 = _mm_add_ps(v_low, v_high);
    v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);
    v_sum128 = _mm_hadd_ps(v_sum128, v_sum128);
    sum_sq = _mm_cvtss_f32(v_sum128);
#endif

    for (; k < K; ++k) {
        float val = x_row[k];
        sum_sq += val * val;
    }

    const float mean_sq = sum_sq / static_cast<float>(K);
    const float rsqrt_scale = 1.0f / std::sqrt(mean_sq + eps);

    // Step 2: Fused Normalization & 1-bit Sign Bit-Packing
    const int64_t num_words = (K + 15) / 16;
    for (int64_t w = 0; w < num_words; ++w) {
        uint16_t mask = 0U;
        const int64_t k_start = w * 16;
        const int64_t k_end = std::min(k_start + 16, K);

#if defined(__AVX2__)
        if (k_start + 16 <= K) {
            __m256 x0 = _mm256_loadu_ps(x_row + k_start);
            __m256 x1 = _mm256_loadu_ps(x_row + k_start + 8);
            __m256 g0 = _mm256_loadu_ps(gamma + k_start);
            __m256 g1 = _mm256_loadu_ps(gamma + k_start + 8);
            __m256 v_scale = _mm256_set1_ps(rsqrt_scale);

            // Normalized scaled activation: x * rsqrt_scale * gamma
            __m256 norm0 = _mm256_mul_ps(_mm256_mul_ps(x0, v_scale), g0);
            __m256 norm1 = _mm256_mul_ps(_mm256_mul_ps(x1, v_scale), g1);

            __m256 zero = _mm256_setzero_ps();
            __m256 cmp0 = _mm256_cmp_ps(norm0, zero, _CMP_GE_OQ);
            __m256 cmp1 = _mm256_cmp_ps(norm1, zero, _CMP_GE_OQ);

            int m0 = _mm256_movemask_ps(cmp0);
            int m1 = _mm256_movemask_ps(cmp1);
            mask = static_cast<uint16_t>((m1 << 8) | m0);
            dst_packed[w] = mask;
            continue;
        }
#endif

        for (int64_t idx = k_start; idx < k_end; ++idx) {
            float norm_val = x_row[idx] * rsqrt_scale * gamma[idx];
            if (norm_val >= 0.0f) {
                mask |= (1U << (idx - k_start));
            }
        }
        dst_packed[w] = mask;
    }
}

} // namespace fused
} // namespace rkmj

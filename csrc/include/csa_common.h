#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace rkmj {
namespace csa {

// -----------------------------------------------------------------------------
// Bitwise Bit-Extraction Intrinsics
//
// 1.58-bit (Ternary) 2-bit Encoding:
//   00 = 0
//   01 = +1
//   10 = -1
//   11 = reserved / unused
//
// Even bits (pos_mask) indicate +1 positions.
// Odd bits (neg_mask)  indicate -1 positions.
// BMI2 hardware instruction _pext gathers all even/odd bits in a single CPU cycle.
// -----------------------------------------------------------------------------

static inline uint32_t extract_even_bits_u32(uint32_t w) {
#if defined(__BMI2__)
    return _pext_u32(w, 0x55555555U);
#else
    uint32_t r = 0;
    for (int i = 0; i < 16; ++i) {
        r |= ((w >> (2 * i)) & 1U) << i;
    }
    return r;
#endif
}

static inline uint32_t extract_odd_bits_u32(uint32_t w) {
#if defined(__BMI2__)
    return _pext_u32(w, 0xAAAAAAAAU);
#else
    uint32_t r = 0;
    for (int i = 0; i < 16; ++i) {
        r |= ((w >> (2 * i + 1)) & 1U) << i;
    }
    return r;
#endif
}

#if defined(__BMI2__)
static inline uint64_t extract_even_bits_u64(uint64_t w) {
    return _pext_u64(w, 0x5555555555555555ULL);
}

static inline uint64_t extract_odd_bits_u64(uint64_t w) {
    return _pext_u64(w, 0xAAAAAAAAAAAAAAAAULL);
}
#endif

// -----------------------------------------------------------------------------
// Bit-Packing Utilities: FP32 / Int8 Ternary <-> 2-bit Packed Words
// 16 weights per uint32_t word (15.8x compression ratio vs FP32)
// -----------------------------------------------------------------------------

inline void pack_ternary_row(
    const float* src,
    uint32_t* dst_packed,
    int64_t K
) {
    const int64_t num_words = (K + 15) / 16;
    for (int64_t w = 0; w < num_words; ++w) {
        uint32_t word_val = 0U;
        const int64_t k_start = w * 16;
        const int64_t k_end = std::min(k_start + 16, K);

        for (int64_t k = k_start; k < k_end; ++k) {
            const float v = src[k];
            uint32_t code = 0U;
            if (v > 0.5f) {
                code = 1U;      // 01 = +1
            } else if (v < -0.5f) {
                code = 2U;      // 10 = -1
            }                   // 00 = 0
            word_val |= (code << (2 * (k - k_start)));
        }
        dst_packed[w] = word_val;
    }
}

inline void unpack_ternary_row(
    const uint32_t* src_packed,
    float* dst,
    int64_t K
) {
    const int64_t num_words = (K + 15) / 16;
    for (int64_t w = 0; w < num_words; ++w) {
        const uint32_t word_val = src_packed[w];
        const int64_t k_start = w * 16;
        const int64_t k_end = std::min(k_start + 16, K);

        for (int64_t k = k_start; k < k_end; ++k) {
            const uint32_t code = (word_val >> (2 * (k - k_start))) & 0x3U;
            if (code == 1U) {
                dst[k] = 1.0f;
            } else if (code == 2U) {
                dst[k] = -1.0f;
            } else {
                dst[k] = 0.0f;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Activation 1-bit Sign Binarization:
// 16 activations packed into uint16_t mask (1 bit per feature: 1 if >= 0, 0 if < 0)
// -----------------------------------------------------------------------------

inline void pack_activation_signs_row(
    const float* x_row,
    uint16_t* x_packed_row,
    int64_t K
) {
    const int64_t num_words = (K + 15) / 16;
    for (int64_t w = 0; w < num_words; ++w) {
        uint16_t mask = 0U;
        const int64_t k_start = w * 16;
        const int64_t k_end = std::min(k_start + 16, K);

#if defined(__AVX2__)
        if (k_start + 16 <= K) {
            __m256 v0 = _mm256_loadu_ps(x_row + k_start);
            __m256 v1 = _mm256_loadu_ps(x_row + k_start + 8);
            __m256 zero = _mm256_setzero_ps();
            __m256 cmp0 = _mm256_cmp_ps(v0, zero, _CMP_GE_OQ);
            __m256 cmp1 = _mm256_cmp_ps(v1, zero, _CMP_GE_OQ);
            int m0 = _mm256_movemask_ps(cmp0);
            int m1 = _mm256_movemask_ps(cmp1);
            mask = static_cast<uint16_t>((m1 << 8) | m0);
            x_packed_row[w] = mask;
            continue;
        }
#endif
        for (int64_t k = k_start; k < k_end; ++k) {
            if (x_row[k] >= 0.0f) {
                mask |= (1U << (k - k_start));
            }
        }
        x_packed_row[w] = mask;
    }
}

} // namespace csa
} // namespace rkmj

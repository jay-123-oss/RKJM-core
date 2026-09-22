#pragma once

#include "simd_isa.hpp"
#include <cstdint>
#include <immintrin.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace rkmj {
namespace simd {

// -----------------------------------------------------------------------------
// Bit Extraction Utilities
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
// 1. Scalar Popcount Difference Kernel
// -----------------------------------------------------------------------------

inline int32_t popcount_diff_scalar(
    const uint32_t* __restrict__ w_row,
    const uint16_t* __restrict__ x_row,
    int64_t num_words,
    bool has_tail,
    uint32_t tail_mask
) {
    int32_t total_diff = 0;
    for (int64_t w = 0; w < num_words; ++w) {
        uint32_t word = w_row[w];
        uint32_t pos_mask = extract_even_bits_u32(word);
        uint32_t neg_mask = extract_odd_bits_u32(word);
        uint32_t x_mask = static_cast<uint32_t>(x_row[w]);

        if (w == num_words - 1 && has_tail) {
            pos_mask &= tail_mask;
            neg_mask &= tail_mask;
        }

        uint32_t pos_hits = (pos_mask & x_mask) | (neg_mask & ~x_mask);
        uint32_t neg_hits = (neg_mask & x_mask) | (pos_mask & ~x_mask);

#if defined(__GNUC__) || defined(__clang__)
        total_diff += __builtin_popcount(pos_hits) - __builtin_popcount(neg_hits);
#else
        total_diff += static_cast<int32_t>(_mm_popcnt_u32(pos_hits)) - static_cast<int32_t>(_mm_popcnt_u32(neg_hits));
#endif
    }
    return total_diff;
}

// -----------------------------------------------------------------------------
// 2. AVX2 + BMI2 Accelerated Kernel (64-bit unrolled popcount + bit extraction)
// -----------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)
inline int32_t popcount_diff_avx2_bmi2(
    const uint32_t* __restrict__ w_row,
    const uint16_t* __restrict__ x_row,
    int64_t num_words,
    bool has_tail,
    uint32_t tail_mask
) {
    const int64_t num_words_64 = num_words / 2;
    int32_t total_diff = 0;

#if defined(__BMI2__)
    const uint64_t* w_row64 = reinterpret_cast<const uint64_t*>(w_row);
    const uint32_t* x_row32 = reinterpret_cast<const uint32_t*>(x_row);

    int64_t w64 = 0;
    // 4x unrolled 64-bit word accumulator (64 ternary weights per loop iteration)
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

        total_diff += (static_cast<int32_t>(__builtin_popcountll(pos0)) - static_cast<int32_t>(__builtin_popcountll(neg0)))
                    + (static_cast<int32_t>(__builtin_popcountll(pos1)) - static_cast<int32_t>(__builtin_popcountll(neg1)))
                    + (static_cast<int32_t>(__builtin_popcountll(pos2)) - static_cast<int32_t>(__builtin_popcountll(neg2)))
                    + (static_cast<int32_t>(__builtin_popcountll(pos3)) - static_cast<int32_t>(__builtin_popcountll(neg3)));
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
        total_diff += static_cast<int32_t>(__builtin_popcountll(pos_hits)) - static_cast<int32_t>(__builtin_popcountll(neg_hits));
    }

    int64_t w = num_words_64 * 2;
#else
    int64_t w = 0;
#endif

    for (; w < num_words; ++w) {
        uint32_t word = w_row[w];
        uint32_t pos_mask = extract_even_bits_u32(word);
        uint32_t neg_mask = extract_odd_bits_u32(word);
        uint32_t x_mask = static_cast<uint32_t>(x_row[w]);

        if (w == num_words - 1 && has_tail) {
            pos_mask &= tail_mask;
            neg_mask &= tail_mask;
        }

        uint32_t pos_hits = (pos_mask & x_mask) | (neg_mask & ~x_mask);
        uint32_t neg_hits = (neg_mask & x_mask) | (pos_mask & ~x_mask);
        total_diff += static_cast<int32_t>(__builtin_popcount(pos_hits)) - static_cast<int32_t>(__builtin_popcount(neg_hits));
    }

    return total_diff;
}
#endif

// -----------------------------------------------------------------------------
// 3. AVX-512 + VPOPCNTDQ Vectorized Kernel
// -----------------------------------------------------------------------------

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512bw,avx512vpopcntdq")))
inline int32_t popcount_diff_avx512(
    const uint32_t* __restrict__ w_row,
    const uint16_t* __restrict__ x_row,
    int64_t num_words,
    bool has_tail,
    uint32_t tail_mask
) {
    const int64_t num_words_64 = num_words / 2;
    int64_t w64 = 0;
    int32_t total_diff = 0;

#if defined(__BMI2__)
    // AVX-512 vector accumulator with 512-bit vector popcount
    // 8x 64-bit integers per 512-bit vector (256 weights per vector iteration)
    __m512i acc_diff = _mm512_setzero_si512();

    const uint64_t* w_row64 = reinterpret_cast<const uint64_t*>(w_row);
    const uint32_t* x_row32 = reinterpret_cast<const uint32_t*>(x_row);

    alignas(64) uint64_t pos_buf[8];
    alignas(64) uint64_t neg_buf[8];

    for (; w64 + 7 < num_words_64; w64 += 8) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint64_t w_val = w_row64[w64 + i];
            uint64_t x_val = x_row32[w64 + i];
            uint64_t p = extract_even_bits_u64(w_val);
            uint64_t n = extract_odd_bits_u64(w_val);
            pos_buf[i] = (p & x_val) | (n & ~x_val);
            neg_buf[i] = (n & x_val) | (p & ~x_val);
        }

        __m512i v_pos = _mm512_load_si512(reinterpret_cast<const __m512i*>(pos_buf));
        __m512i v_neg = _mm512_load_si512(reinterpret_cast<const __m512i*>(neg_buf));

        __m512i cnt_pos = _mm512_popcnt_epi64(v_pos);
        __m512i cnt_neg = _mm512_popcnt_epi64(v_neg);

        __m512i diff = _mm512_sub_epi64(cnt_pos, cnt_neg);
        acc_diff = _mm512_add_epi64(acc_diff, diff);
    }

    // Horizontal reduce 512-bit accumulator
    alignas(64) int64_t reduced[8];
    _mm512_store_si512(reinterpret_cast<__m512i*>(reduced), acc_diff);
    for (int i = 0; i < 8; ++i) {
        total_diff += static_cast<int32_t>(reduced[i]);
    }
#endif

    // Remaining 64-bit words via AVX2 / fast BMI2
    for (; w64 < num_words_64; ++w64) {
        uint64_t w_val = reinterpret_cast<const uint64_t*>(w_row)[w64];
        uint64_t x_val = reinterpret_cast<const uint32_t*>(x_row)[w64];
        uint64_t pos_mask = extract_even_bits_u64(w_val);
        uint64_t neg_mask = extract_odd_bits_u64(w_val);

        if (w64 == num_words_64 - 1 && (num_words % 2 == 0) && has_tail) {
            uint64_t full_tail = 0xFFFFFFFFULL | (static_cast<uint64_t>(tail_mask) << 16);
            pos_mask &= full_tail;
            neg_mask &= full_tail;
        }

        uint64_t pos_hits = (pos_mask & x_val) | (neg_mask & ~x_val);
        uint64_t neg_hits = (neg_mask & x_val) | (pos_mask & ~x_val);
        total_diff += static_cast<int32_t>(__builtin_popcountll(pos_hits)) - static_cast<int32_t>(__builtin_popcountll(neg_hits));
    }

    for (int64_t w = num_words_64 * 2; w < num_words; ++w) {
        uint32_t word = w_row[w];
        uint32_t pos_mask = extract_even_bits_u32(word);
        uint32_t neg_mask = extract_odd_bits_u32(word);
        uint32_t x_mask = static_cast<uint32_t>(x_row[w]);

        if (w == num_words - 1 && has_tail) {
            pos_mask &= tail_mask;
            neg_mask &= tail_mask;
        }

        uint32_t pos_hits = (pos_mask & x_mask) | (neg_mask & ~x_mask);
        uint32_t neg_hits = (neg_mask & x_mask) | (pos_mask & ~x_mask);
        total_diff += static_cast<int32_t>(__builtin_popcount(pos_hits)) - static_cast<int32_t>(__builtin_popcount(neg_hits));
    }

    return total_diff;
}
#endif

// -----------------------------------------------------------------------------
// 4. ARM64 NEON Vectorized Kernel
// -----------------------------------------------------------------------------

#if defined(__ARM_NEON)
inline int32_t popcount_diff_neon(
    const uint32_t* __restrict__ w_row,
    const uint16_t* __restrict__ x_row,
    int64_t num_words,
    bool has_tail,
    uint32_t tail_mask
) {
    int32_t total_diff = 0;
    int64_t w = 0;

    // Vectorized 128-bit NEON loop (16 bytes per iteration)
    uint32x4_t v_acc_pos = vdupq_n_u32(0);
    uint32x4_t v_acc_neg = vdupq_n_u32(0);

    for (; w + 3 < num_words; w += 4) {
        // Extract even and odd bits for 4 words
        uint32_t p0 = extract_even_bits_u32(w_row[w]);
        uint32_t n0 = extract_odd_bits_u32(w_row[w]);
        uint32_t p1 = extract_even_bits_u32(w_row[w + 1]);
        uint32_t n1 = extract_odd_bits_u32(w_row[w + 1]);
        uint32_t p2 = extract_even_bits_u32(w_row[w + 2]);
        uint32_t n2 = extract_odd_bits_u32(w_row[w + 2]);
        uint32_t p3 = extract_even_bits_u32(w_row[w + 3]);
        uint32_t n3 = extract_odd_bits_u32(w_row[w + 3]);

        uint32_t x0 = x_row[w];
        uint32_t x1 = x_row[w + 1];
        uint32_t x2 = x_row[w + 2];
        uint32_t x3 = x_row[w + 3];

        uint32_t pos0 = (p0 & x0) | (n0 & ~x0);
        uint32_t neg0 = (n0 & x0) | (p0 & ~x0);
        uint32_t pos1 = (p1 & x1) | (n1 & ~x1);
        uint32_t neg1 = (n1 & x1) | (p1 & ~x1);
        uint32_t pos2 = (p2 & x2) | (n2 & ~x2);
        uint32_t neg2 = (n2 & x2) | (p2 & ~x2);
        uint32_t pos3 = (p3 & x3) | (n3 & ~x3);
        uint32_t neg3 = (n3 & x3) | (p3 & ~x3);

        uint32x4_t v_pos = {pos0, pos1, pos2, pos3};
        uint32x4_t v_neg = {neg0, neg1, neg2, neg3};

        // vcntq_u8 byte-wise popcount
        uint8x16_t cnt_pos8 = vcntq_u8(vreinterpretq_u8_u32(v_pos));
        uint8x16_t cnt_neg8 = vcntq_u8(vreinterpretq_u8_u32(v_neg));

        // Pairwise add 8-bit -> 16-bit -> 32-bit
        uint16x8_t p16 = vpaddlq_u8(cnt_pos8);
        uint16x8_t n16 = vpaddlq_u8(cnt_neg8);

        v_acc_pos = vpadalq_u16(v_acc_pos, p16);
        v_acc_neg = vpadalq_u16(v_acc_neg, n16);
    }

    total_diff += vaddvq_u32(v_acc_pos) - vaddvq_u32(v_acc_neg);

    // Scalar tail
    for (; w < num_words; ++w) {
        uint32_t word = w_row[w];
        uint32_t pos_mask = extract_even_bits_u32(word);
        uint32_t neg_mask = extract_odd_bits_u32(word);
        uint32_t x_mask = static_cast<uint32_t>(x_row[w]);

        if (w == num_words - 1 && has_tail) {
            pos_mask &= tail_mask;
            neg_mask &= tail_mask;
        }

        uint32_t pos_hits = (pos_mask & x_mask) | (neg_mask & ~x_mask);
        uint32_t neg_hits = (neg_mask & x_mask) | (pos_mask & ~x_mask);
        total_diff += __builtin_popcount(pos_hits) - __builtin_popcount(neg_hits);
    }

    return total_diff;
}
#endif

// -----------------------------------------------------------------------------
// Universal Dispatcher Function
// -----------------------------------------------------------------------------

inline int32_t popcount_diff_dispatch(
    const uint32_t* __restrict__ w_row,
    const uint16_t* __restrict__ x_row,
    int64_t num_words,
    bool has_tail,
    uint32_t tail_mask,
    SimdBackend backend
) {
    switch (backend) {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        case SimdBackend::AVX512_VPOPCNT:
            return popcount_diff_avx512(w_row, x_row, num_words, has_tail, tail_mask);
#endif
#if defined(__x86_64__) || defined(_M_X64)
        case SimdBackend::AVX2_BMI2:
            return popcount_diff_avx2_bmi2(w_row, x_row, num_words, has_tail, tail_mask);
#endif
#if defined(__ARM_NEON)
        case SimdBackend::ARM_NEON:
            return popcount_diff_neon(w_row, x_row, num_words, has_tail, tail_mask);
#endif
        case SimdBackend::SCALAR:
        default:
            return popcount_diff_scalar(w_row, x_row, num_words, has_tail, tail_mask);
    }
}

} // namespace simd
} // namespace rkmj

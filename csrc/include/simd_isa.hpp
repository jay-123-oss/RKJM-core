#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <new>
#include <memory>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif
#endif

namespace rkmj {
namespace simd {

enum class SimdBackend : uint8_t {
    SCALAR = 0,
    ARM_NEON = 1,
    AVX2_BMI2 = 2,
    AVX512_VPOPCNT = 3
};

inline const char* backend_to_string(SimdBackend b) {
    switch (b) {
        case SimdBackend::AVX512_VPOPCNT:
            return "AVX-512 (VPOPCNTDQ/BW/F)";
        case SimdBackend::AVX2_BMI2:
            return "AVX2 + BMI2 (Fast Popcount)";
        case SimdBackend::ARM_NEON:
            return "ARM64 NEON (VCNT)";
        case SimdBackend::SCALAR:
        default:
            return "Scalar Fallback";
    }
}

// -----------------------------------------------------------------------------
// CPU Feature Detection (Runtime CPUID / Architecture Check)
// -----------------------------------------------------------------------------

inline SimdBackend detect_best_simd_backend() {
#if defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON)
        return SimdBackend::ARM_NEON;
    #else
        return SimdBackend::SCALAR;
    #endif
#elif defined(__x86_64__) || defined(_M_X64)
    // Check for AVX-512 F, BW, and VPOPCNTDQ
    bool has_avx512 = false;
    bool has_avx2_bmi2 = false;

    #if defined(__GNUC__) || defined(__clang__)
        // Use GCC / Clang built-in detection first if available
        #if defined(__builtin_cpu_supports)
            bool avx2 = __builtin_cpu_supports("avx2");
            bool bmi2 = __builtin_cpu_supports("bmi2");
            bool popcnt = __builtin_cpu_supports("popcnt");
            if (avx2 && bmi2 && popcnt) {
                has_avx2_bmi2 = true;
            }
            bool avx512f = __builtin_cpu_supports("avx512f");
            bool avx512bw = __builtin_cpu_supports("avx512bw");
            bool avx512vpopcntdq = __builtin_cpu_supports("avx512vpopcntdq");
            if (avx512f && avx512bw && avx512vpopcntdq) {
                has_avx512 = true;
            }
        #endif
    #endif

    // Direct CPUID verification fallback
    if (!has_avx512) {
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        // Leaf 7, Subleaf 0
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            bool avx2_flag = (ebx & (1U << 5)) != 0;
            bool bmi2_flag = (ebx & (1U << 8)) != 0;
            bool avx512f_flag = (ebx & (1U << 16)) != 0;
            bool avx512bw_flag = (ebx & (1U << 30)) != 0;
            bool avx512vpopcntdq_flag = (ecx & (1U << 14)) != 0;

            if (avx2_flag && bmi2_flag) {
                has_avx2_bmi2 = true;
            }
            if (avx512f_flag && avx512bw_flag && avx512vpopcntdq_flag) {
                has_avx512 = true;
            }
        }
    }

    if (has_avx512) {
        return SimdBackend::AVX512_VPOPCNT;
    }
    if (has_avx2_bmi2) {
        return SimdBackend::AVX2_BMI2;
    }
    return SimdBackend::SCALAR;
#else
    return SimdBackend::SCALAR;
#endif
}

inline SimdBackend get_simd_backend() {
    static const SimdBackend backend = detect_best_simd_backend();
    return backend;
}

// -----------------------------------------------------------------------------
// 64-Byte Cache-Line Aligned Memory Buffer (Zero-Copy & AVX-512 Ready)
// -----------------------------------------------------------------------------

template <typename T, size_t Alignment = 64>
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t count = 0)
        : data_(nullptr), count_(0), capacity_(0) {
        if (count > 0) {
            allocate(count);
        }
    }

    ~AlignedBuffer() {
        deallocate();
    }

    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // Move-constructible
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.count_ = 0;
        other.capacity_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            count_ = other.count_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.count_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        if (count <= capacity_ && data_ != nullptr) {
            count_ = count;
            return;
        }
        deallocate();
        count_ = count;
        capacity_ = count;
        size_t bytes = count * sizeof(T);
        // Round up to multiple of Alignment
        bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
        if (bytes == 0) {
            data_ = nullptr;
            return;
        }

#if defined(_MSC_VER)
        data_ = static_cast<T*>(_aligned_malloc(bytes, Alignment));
        if (!data_) throw std::bad_alloc();
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, bytes) != 0 || !ptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<T*>(ptr);
#endif
    }

    void zero() {
        if (data_ && count_ > 0) {
            std::memset(data_, 0, count_ * sizeof(T));
        }
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T& operator[](size_t idx) noexcept { return data_[idx]; }
    const T& operator[](size_t idx) const noexcept { return data_[idx]; }

    size_t size() const noexcept { return count_; }
    size_t capacity() const noexcept { return capacity_; }

private:
    void deallocate() {
        if (data_) {
#if defined(_MSC_VER)
            _aligned_free(data_);
#else
            std::free(data_);
#endif
            data_ = nullptr;
        }
        count_ = 0;
        capacity_ = 0;
    }

    T* data_;
    size_t count_;
    size_t capacity_;
};

} // namespace simd
} // namespace rkmj

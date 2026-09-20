#include <torch/extension.h>
#include <omp.h>
#include <immintrin.h>
#include <iostream>

void custom_kernel_omp(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous");

    float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    int64_t N = input.numel();

    // Verification: Runtime par kitne threads available hain print karein
    static bool printed = false;
    if (!printed) {
        std::cout << "[INFO] OpenMP Max Threads: " << omp_get_max_threads() << std::endl;
        printed = true;
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < N; i += 8) {
        if (i + 8 <= N) {
            __m256 va = _mm256_loadu_ps(&in_ptr[i]);
            __m256 v_mult = _mm256_set1_ps(2.0f);
            __m256 v_add = _mm256_set1_ps(5.0f);
            
            __m256 vres = _mm256_fmadd_ps(va, v_mult, v_add);
            _mm256_storeu_ps(&out_ptr[i], vres);
        } else {
            for (int64_t j = i; j < N; j++) {
                out_ptr[j] = in_ptr[j] * 2.0f + 5.0f;
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_kernel_omp", &custom_kernel_omp, "Fast OpenMP + AVX2 C++ Kernel");
}
#include <torch/extension.h>
#include <omp.h>
#include <immintrin.h>

void custom_gemm_omp(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "Tensors must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* a_ptr = A.data_ptr<float>();
    const float* b_ptr = B.data_ptr<float>();
    float* c_ptr = C.data_ptr<float>();

    // Zero-initialize C matrix
    std::fill(c_ptr, c_ptr + (M * N), 0.0f);

    int BLOCK_SIZE = 64; // Cache Tiling size

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                
                int i_max = std::min(i0 + BLOCK_SIZE, M);
                int j_max = std::min(j0 + BLOCK_SIZE, N);
                int k_max = std::min(k0 + BLOCK_SIZE, K);

                for (int i = i0; i < i_max; ++i) {
                    for (int k = k0; k < k_max; ++k) {
                        float a_val = a_ptr[i * K + k];
                        __m256 va = _mm256_set1_ps(a_val);

                        int j = j0;
                        // AVX2 SIMD Loop (8 floats per cycle)
                        for (; j + 7 < j_max; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&b_ptr[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&c_ptr[i * N + j]);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&c_ptr[i * N + j], vc);
                        }
                        // Scalar fallback for remaining columns
                        for (; j < j_max; ++j) {
                            c_ptr[i * N + j] += a_val * b_ptr[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_gemm", &custom_gemm_omp, "Parallel AVX2 Tiled GEMM Kernel");
}
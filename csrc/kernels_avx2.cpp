#include "include/kernels_avx2.h"
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace rkmj {
namespace cpu {

torch::Tensor gemv_tiled_csa_cpu(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias,
    int64_t group_size
) {
    x = x.contiguous().to(torch::kFloat32);
    w_packed = w_packed.contiguous();
    scales = scales.contiguous().to(torch::kFloat32);

    const int64_t K = x.numel();
    const int64_t N = w_packed.size(0);
    const int64_t num_words = w_packed.size(1);
    const int64_t num_groups = scales.size(1);

    auto out = torch::zeros({1, N}, torch::kFloat32);

    const float* x_ptr = x.data_ptr<float>();
    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    const float* scales_ptr = scales.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const float* bias_ptr = (bias.has_value() && bias.value().defined()) 
                            ? bias.value().contiguous().data_ptr<float>() 
                            : nullptr;

    // Cache-aware Tiled Loop: Tile across rows N (L2 tile size = 64 rows)
    constexpr int64_t TILE_N = 64;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t tn = 0; tn < N; tn += TILE_N) {
        const int64_t n_end = std::min(tn + TILE_N, N);

        for (int64_t row = tn; row < n_end; ++row) {
            const uint32_t* row_w = w_ptr + row * num_words;
            const float* row_scales = scales_ptr + row * num_groups;

            float row_acc = 0.0f;

            // Iterate by quantization groups (e.g. 64 elements = 4 words)
            const int64_t words_per_group = (group_size + 15) / 16;

            for (int64_t g = 0; g < num_groups; ++g) {
                const int64_t w_start = g * words_per_group;
                const int64_t w_end = std::min(w_start + words_per_group, num_words);
                const float scale_val = row_scales[g];

                float group_acc = 0.0f;

                for (int64_t w_idx = w_start; w_idx < w_end; ++w_idx) {
                    const uint32_t word = row_w[w_idx];
                    const int64_t col_start = w_idx * 16;
                    const int64_t col_end = std::min(col_start + 16, K);

                    // Unroll 16 weights
                    for (int64_t col = col_start; col < col_end; ++col) {
                        const uint32_t code = (word >> ((col - col_start) * 2)) & 0x03;
                        if (code == 1) {
                            group_acc += x_ptr[col];
                        } else if (code == 2) {
                            group_acc -= x_ptr[col];
                        }
                    }
                }

                row_acc += group_acc * scale_val;
            }

            if (bias_ptr) {
                row_acc += bias_ptr[row];
            }
            out_ptr[row] = row_acc;
        }
    }

    return out;
}

torch::Tensor gemm_tiled_csa_cpu(
    torch::Tensor x,
    torch::Tensor w_packed,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias,
    int64_t group_size
) {
    x = x.contiguous().to(torch::kFloat32);
    w_packed = w_packed.contiguous();
    scales = scales.contiguous().to(torch::kFloat32);

    const int64_t orig_dim0 = x.dim() > 2 ? x.size(0) : 1;
    const int64_t M = x.numel() / x.size(-1);
    const int64_t K = x.size(-1);
    const int64_t N = w_packed.size(0);

    // If batch size is 1, dispatch directly to optimized gemv
    if (M == 1) {
        auto res = gemv_tiled_csa_cpu(x, w_packed, scales, bias, group_size);
        if (x.dim() == 3) {
            return res.view({orig_dim0, 1, N});
        }
        return res;
    }

    auto x_2d = x.view({M, K});
    auto out_2d = torch::zeros({M, N}, torch::kFloat32);

    const float* x_ptr = x_2d.data_ptr<float>();
    const uint32_t* w_ptr = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    const float* scales_ptr = scales.data_ptr<float>();
    float* out_ptr = out_2d.data_ptr<float>();

    const float* bias_ptr = (bias.has_value() && bias.value().defined()) 
                            ? bias.value().contiguous().data_ptr<float>() 
                            : nullptr;

    const int64_t num_words = w_packed.size(1);
    const int64_t num_groups = scales.size(1);
    const int64_t words_per_group = (group_size + 15) / 16;

    // Parallelize across batch M and rows N
    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t row = 0; row < N; ++row) {
            const float* row_x = x_ptr + m * K;
            const uint32_t* row_w = w_ptr + row * num_words;
            const float* row_scales = scales_ptr + row * num_groups;

            float acc = 0.0f;

            for (int64_t g = 0; g < num_groups; ++g) {
                const int64_t w_start = g * words_per_group;
                const int64_t w_end = std::min(w_start + words_per_group, num_words);
                const float scale = row_scales[g];

                float g_acc = 0.0f;
                for (int64_t w_idx = w_start; w_idx < w_end; ++w_idx) {
                    const uint32_t word = row_w[w_idx];
                    const int64_t col_start = w_idx * 16;
                    const int64_t col_end = std::min(col_start + 16, K);

                    for (int64_t col = col_start; col < col_end; ++col) {
                        const uint32_t code = (word >> ((col - col_start) * 2)) & 0x03;
                        if (code == 1) {
                            g_acc += row_x[col];
                        } else if (code == 2) {
                            g_acc -= row_x[col];
                        }
                    }
                }
                acc += g_acc * scale;
            }

            if (bias_ptr) {
                acc += bias_ptr[row];
            }
            out_ptr[m * N + row] = acc;
        }
    }

    if (x.dim() == 3) {
        return out_2d.view({orig_dim0, x.size(1), N});
    }
    return out_2d;
}

} // namespace cpu
} // namespace rkmj

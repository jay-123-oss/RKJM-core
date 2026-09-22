#include "include/pack.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace rkmj {
namespace quant {

torch::Tensor pack_ternary_2bit_cpu(torch::Tensor w_ternary) {
    TORCH_CHECK(w_ternary.dim() == 2, "pack_ternary_2bit_cpu: Input must be 2D matrix [N, K].");
    w_ternary = w_ternary.contiguous().to(torch::kFloat32);

    const int64_t N = w_ternary.size(0);
    const int64_t K = w_ternary.size(1);
    const int64_t num_words = (K + 15) / 16;

    auto w_packed = torch::zeros({N, num_words}, torch::kInt32);

    const float* src_ptr = w_ternary.data_ptr<float>();
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(w_packed.data_ptr<int32_t>());

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < N; ++row) {
        const float* row_src = src_ptr + row * K;
        uint32_t* row_dst = dst_ptr + row * num_words;

        for (int64_t w_idx = 0; w_idx < num_words; ++w_idx) {
            uint32_t packed_val = 0;
            const int64_t col_start = w_idx * 16;
            const int64_t col_end = std::min(col_start + 16, K);

            for (int64_t col = col_start; col < col_end; ++col) {
                const float val = row_src[col];
                uint32_t code = 0; // 00 = 0
                if (val > 0.5f) {
                    code = 1;      // 01 = +1
                } else if (val < -0.5f) {
                    code = 2;      // 10 = -1
                }
                packed_val |= (code << ((col - col_start) * 2));
            }
            row_dst[w_idx] = packed_val;
        }
    }

    return w_packed;
}

torch::Tensor unpack_ternary_2bit_cpu(torch::Tensor w_packed, int64_t K) {
    TORCH_CHECK(w_packed.dim() == 2, "unpack_ternary_2bit_cpu: Input must be 2D [N, num_words].");
    w_packed = w_packed.contiguous();

    const int64_t N = w_packed.size(0);
    const int64_t num_words = w_packed.size(1);
    TORCH_CHECK(num_words == (K + 15) / 16, "unpack_ternary_2bit_cpu: Dimension mismatch with K.");

    auto w_unpacked = torch::zeros({N, K}, torch::kFloat32);

    const uint32_t* src_ptr = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    float* dst_ptr = w_unpacked.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < N; ++row) {
        const uint32_t* row_src = src_ptr + row * num_words;
        float* row_dst = dst_ptr + row * K;

        for (int64_t w_idx = 0; w_idx < num_words; ++w_idx) {
            const uint32_t word = row_src[w_idx];
            const int64_t col_start = w_idx * 16;
            const int64_t col_end = std::min(col_start + 16, K);

            for (int64_t col = col_start; col < col_end; ++col) {
                const uint32_t code = (word >> ((col - col_start) * 2)) & 0x03;
                float weight = 0.0f;
                if (code == 1) {
                    weight = 1.0f;
                } else if (code == 2) {
                    weight = -1.0f;
                }
                row_dst[col] = weight;
            }
        }
    }

    return w_unpacked;
}

std::pair<torch::Tensor, torch::Tensor> quantize_grouped_cpu(
    torch::Tensor weight,
    int64_t group_size
) {
    TORCH_CHECK(weight.dim() == 2, "quantize_grouped_cpu: Weight must be 2D [N, K].");
    TORCH_CHECK(group_size > 0, "quantize_grouped_cpu: group_size must be positive.");
    weight = weight.contiguous().to(torch::kFloat32);

    const int64_t N = weight.size(0);
    const int64_t K = weight.size(1);
    const int64_t num_groups = (K + group_size - 1) / group_size;
    const int64_t num_words = (K + 15) / 16;

    auto w_packed = torch::zeros({N, num_words}, torch::kInt32);
    auto scales = torch::zeros({N, num_groups}, torch::kFloat32);

    const float* src_ptr = weight.data_ptr<float>();
    uint32_t* dst_packed = reinterpret_cast<uint32_t*>(w_packed.data_ptr<int32_t>());
    float* dst_scales = scales.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t row = 0; row < N; ++row) {
        const float* row_src = src_ptr + row * K;
        uint32_t* row_packed = dst_packed + row * num_words;
        float* row_scales = dst_scales + row * num_groups;

        std::vector<float> row_ternary(K, 0.0f);

        // 1. Group-wise mean and thresholding
        for (int64_t g = 0; g < num_groups; ++g) {
            const int64_t g_start = g * group_size;
            const int64_t g_end = std::min(g_start + group_size, K);
            const int64_t g_len = g_end - g_start;

            float abs_sum = 0.0f;
            for (int64_t j = g_start; j < g_end; ++j) {
                abs_sum += std::abs(row_src[j]);
            }
            const float gamma_g = abs_sum / static_cast<float>(g_len);
            const float threshold = 0.5f * gamma_g;

            float dot_prod = 0.0f;
            float norm_sq = 0.0f;

            for (int64_t j = g_start; j < g_end; ++j) {
                const float w_val = row_src[j];
                float t_val = 0.0f;
                if (w_val >= threshold) {
                    t_val = 1.0f;
                } else if (w_val <= -threshold) {
                    t_val = -1.0f;
                }
                row_ternary[j] = t_val;
                dot_prod += w_val * t_val;
                norm_sq += t_val * t_val;
            }

            // Least-squares optimal per-group alpha: (W * W_t).sum() / (W_t^2).sum()
            float alpha = (norm_sq > 1e-6f) ? (dot_prod / norm_sq) : gamma_g;
            if (alpha < 1e-5f) alpha = 1e-5f;
            row_scales[g] = alpha;
        }

        // 2. Pack row into 2-bit aligned uint32 bitfield
        for (int64_t w_idx = 0; w_idx < num_words; ++w_idx) {
            uint32_t packed_val = 0;
            const int64_t col_start = w_idx * 16;
            const int64_t col_end = std::min(col_start + 16, K);

            for (int64_t col = col_start; col < col_end; ++col) {
                const float t = row_ternary[col];
                uint32_t code = 0;
                if (t > 0.5f) code = 1;
                else if (t < -0.5f) code = 2;
                packed_val |= (code << ((col - col_start) * 2));
            }
            row_packed[w_idx] = packed_val;
        }
    }

    return {w_packed, scales};
}

torch::Tensor dequantize_grouped_cpu(
    torch::Tensor w_packed,
    torch::Tensor scales,
    int64_t in_features,
    int64_t group_size
) {
    w_packed = w_packed.contiguous();
    scales = scales.contiguous().to(torch::kFloat32);

    const int64_t N = w_packed.size(0);
    const int64_t K = in_features;
    const int64_t num_words = w_packed.size(1);
    const int64_t num_groups = scales.size(1);

    auto out = torch::zeros({N, K}, torch::kFloat32);

    const uint32_t* src_packed = reinterpret_cast<const uint32_t*>(w_packed.data_ptr<int32_t>());
    const float* src_scales = scales.data_ptr<float>();
    float* dst_out = out.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < N; ++row) {
        const uint32_t* row_packed = src_packed + row * num_words;
        const float* row_scales = src_scales + row * num_groups;
        float* row_out = dst_out + row * K;

        for (int64_t col = 0; col < K; ++col) {
            const int64_t w_idx = col / 16;
            const int64_t bit_offset = (col % 16) * 2;
            const uint32_t code = (row_packed[w_idx] >> bit_offset) & 0x03;

            float t = 0.0f;
            if (code == 1) t = 1.0f;
            else if (code == 2) t = -1.0f;

            const int64_t g = col / group_size;
            const float alpha = (g < num_groups) ? row_scales[g] : 1.0f;

            row_out[col] = t * alpha;
        }
    }

    return out;
}

} // namespace quant
} // namespace rkmj

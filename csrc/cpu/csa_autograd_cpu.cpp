#include "../include/csa_common.h"
#include "../include/csa_autograd.h"
#include <torch/extension.h>
#include <torch/autograd.h>
#include <omp.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace rkmj {
namespace autograd {

// =============================================================================
// CSALinearFunction Forward Implementation
// =============================================================================

torch::Tensor CSALinearFunction::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor x,
    torch::Tensor latent_weight,
    torch::Tensor alpha,
    c10::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(latent_weight.is_cpu(), "latent_weight must be a CPU tensor");
    TORCH_CHECK(alpha.is_cpu(), "alpha must be a CPU tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(latent_weight.is_contiguous(), "latent_weight must be contiguous");
    TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(latent_weight.scalar_type() == torch::kFloat32, "latent_weight must be float32");
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

    TORCH_CHECK(latent_weight.dim() == 2, "latent_weight must be 2D [N, K]");
    const int64_t N = latent_weight.size(0);
    TORCH_CHECK(latent_weight.size(1) == K, "Feature dimension K mismatch");

    const bool alpha_is_per_channel = (alpha.numel() == N);
    TORCH_CHECK(alpha.numel() == 1 || alpha_is_per_channel, "alpha must have 1 or N elements");

    if (has_bias) {
        TORCH_CHECK(bias.value().numel() == N, "bias must have N elements");
    }

    const int64_t num_words = (K + 15) / 16;
    const int64_t num_words_64 = num_words / 2;
    const bool has_tail = (K % 16 != 0);
    const uint32_t tail_mask = has_tail ? ((1U << (K % 16)) - 1U) : 0xFFFFU;

    // -------------------------------------------------------------------------
    // 1. Quantize Latent Weight to Ternary {-1, 0, +1} & Pack into uint32
    // -------------------------------------------------------------------------
    auto w_q = torch::empty({N, K}, torch::kFloat32);
    std::vector<uint32_t> w_packed(N * num_words, 0U);

    const float* lw_ptr = latent_weight.data_ptr<float>();
    const float* alpha_ptr = alpha.data_ptr<float>();
    float* wq_ptr = w_q.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const float a = alpha_is_per_channel ? alpha_ptr[n] : alpha_ptr[0];
        const float scale = (std::abs(a) > 1e-8f) ? (1.0f / a) : 1.0f;
        const float* row_in = lw_ptr + n * K;
        float* row_out = wq_ptr + n * K;
        uint32_t* row_packed = &w_packed[n * num_words];

        for (int64_t w = 0; w < num_words; ++w) {
            uint32_t word_val = 0U;
            const int64_t k_start = w * 16;
            const int64_t k_end = std::min(k_start + 16, K);

            for (int64_t k = k_start; k < k_end; ++k) {
                const float scaled_w = row_in[k] * scale;
                const float q = std::clamp(std::round(scaled_w), -1.0f, 1.0f);
                row_out[k] = q;

                uint32_t code = 0U;
                if (q > 0.5f) {
                    code = 1U;      // 01 = +1
                } else if (q < -0.5f) {
                    code = 2U;      // 10 = -1
                }                   // 00 = 0
                word_val |= (code << (2 * (k - k_start)));
            }
            row_packed[w] = word_val;
        }
    }

    // -------------------------------------------------------------------------
    // 2. Pack Activation Signs (x >= 0 ? 1 : 0)
    // -------------------------------------------------------------------------
    std::vector<uint16_t> x_packed(B * num_words);
    const float* x_ptr = x.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        rkmj::csa::pack_activation_signs_row(x_ptr + b * K, &x_packed[b * num_words], K);
    }

    // -------------------------------------------------------------------------
    // 3. Bitwise Popcount Carry-Save Accumulation
    // -------------------------------------------------------------------------
    std::vector<int64_t> out_sizes(orig_sizes.begin(), orig_sizes.end() - 1);
    out_sizes.push_back(N);
    auto y = torch::empty(out_sizes, x.options());
    auto y_unscaled = torch::empty({B, N}, torch::kFloat32);

    float* y_ptr = y.data_ptr<float>();
    float* yu_ptr = y_unscaled.data_ptr<float>();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            const uint16_t* x_row = &x_packed[b * num_words];
            const uint32_t* w_row = &w_packed[n * num_words];
            const float a = alpha_is_per_channel ? alpha_ptr[n] : alpha_ptr[0];
            const float b_val = has_bias ? bias_ptr[n] : 0.0f;

            int32_t total_diff = 0;

#if defined(__BMI2__)
            const uint64_t* w_row64 = reinterpret_cast<const uint64_t*>(w_row);
            const uint32_t* x_row32 = reinterpret_cast<const uint32_t*>(x_row);

            int64_t w64 = 0;
            for (; w64 + 3 < num_words_64; w64 += 4) {
                uint64_t w0 = w_row64[w64];
                uint64_t w1 = w_row64[w64 + 1];
                uint64_t w2 = w_row64[w64 + 2];
                uint64_t w3 = w_row64[w64 + 3];

                uint64_t x0 = x_row32[w64];
                uint64_t x1 = x_row32[w64 + 1];
                uint64_t x2 = x_row32[w64 + 2];
                uint64_t x3 = x_row32[w64 + 3];

                uint64_t p0 = rkmj::csa::extract_even_bits_u64(w0);
                uint64_t n0 = rkmj::csa::extract_odd_bits_u64(w0);
                uint64_t p1 = rkmj::csa::extract_even_bits_u64(w1);
                uint64_t n1 = rkmj::csa::extract_odd_bits_u64(w1);
                uint64_t p2 = rkmj::csa::extract_even_bits_u64(w2);
                uint64_t n2 = rkmj::csa::extract_odd_bits_u64(w2);
                uint64_t p3 = rkmj::csa::extract_even_bits_u64(w3);
                uint64_t n3 = rkmj::csa::extract_odd_bits_u64(w3);

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
                uint64_t pos_mask = rkmj::csa::extract_even_bits_u64(w_val);
                uint64_t neg_mask = rkmj::csa::extract_odd_bits_u64(w_val);

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
                uint32_t pos_mask = rkmj::csa::extract_even_bits_u32(word);
                uint32_t neg_mask = rkmj::csa::extract_odd_bits_u32(word);
                uint32_t x_mask = x_row[w];

                if (w == num_words - 1 && has_tail) {
                    pos_mask &= tail_mask;
                    neg_mask &= tail_mask;
                }

                uint32_t pos_hits = (pos_mask & x_mask) | (neg_mask & ~x_mask);
                uint32_t neg_hits = (neg_mask & x_mask) | (pos_mask & ~x_mask);
                total_diff += __builtin_popcount(pos_hits) - __builtin_popcount(neg_hits);
            }

            yu_ptr[b * N + n] = static_cast<float>(total_diff);
            y_ptr[b * N + n] = a * static_cast<float>(total_diff) + b_val;
        }
    }

    // Save tensors and metadata for backward pass
    ctx->save_for_backward({x, latent_weight, alpha, y_unscaled, w_q});
    ctx->saved_data["has_bias"] = has_bias;
    ctx->saved_data["B"] = B;
    ctx->saved_data["K"] = K;
    ctx->saved_data["N"] = N;

    return y;
}

// =============================================================================
// CSALinearFunction Backward Implementation (STE)
// =============================================================================

torch::autograd::variable_list CSALinearFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs
) {
    torch::autograd::variable_list saved = ctx->get_saved_variables();
    torch::Tensor x = saved[0];
    torch::Tensor latent_weight = saved[1];
    torch::Tensor alpha = saved[2];
    torch::Tensor y_unscaled = saved[3];
    torch::Tensor w_q = saved[4];

    const bool has_bias = ctx->saved_data["has_bias"].toBool();
    const int64_t B = ctx->saved_data["B"].toInt();
    const int64_t K = ctx->saved_data["K"].toInt();
    const int64_t N = ctx->saved_data["N"].toInt();

    auto grad_out = grad_outputs[0].contiguous().reshape({B, N});
    const bool alpha_is_per_channel = (alpha.numel() == N);

    // -------------------------------------------------------------------------
    // Compute grad_scaled = grad_output * alpha [B, N]
    // -------------------------------------------------------------------------
    torch::Tensor grad_scaled = alpha_is_per_channel 
                                ? (grad_out * alpha.unsqueeze(0)) 
                                : (grad_out * alpha);

    // -------------------------------------------------------------------------
    // a. grad_input = grad_scaled @ quantized_weight [B, K]
    // -------------------------------------------------------------------------
    auto grad_x_2d = at::mm(grad_scaled, w_q);
    auto grad_x = grad_x_2d.reshape(x.sizes());

    // -------------------------------------------------------------------------
    // b. grad_latent_weight = grad_scaled^T @ x [N, K]
    // -------------------------------------------------------------------------
    auto grad_latent = at::mm(grad_scaled.t(), x.reshape({B, K}));

    // -------------------------------------------------------------------------
    // c. grad_alpha = sum_b (grad_output * y_unscaled)
    // -------------------------------------------------------------------------
    torch::Tensor grad_alpha;
    if (alpha_is_per_channel) {
        grad_alpha = (grad_out * y_unscaled).sum(0);
    } else {
        grad_alpha = (grad_out * y_unscaled).sum().reshape_as(alpha);
    }

    // Optional bias gradient: sum_b (grad_output)
    torch::Tensor grad_bias;
    if (has_bias) {
        grad_bias = grad_out.sum(0);
    }

    torch::autograd::variable_list result;
    result.push_back(grad_x);
    result.push_back(grad_latent);
    result.push_back(grad_alpha);
    if (has_bias) {
        result.push_back(grad_bias);
    } else {
        result.push_back(torch::Tensor());
    }
    return result;
}

torch::Tensor csa_linear_forward_autograd(
    torch::Tensor x,
    torch::Tensor latent_weight,
    torch::Tensor alpha,
    c10::optional<torch::Tensor> bias
) {
    if (bias.has_value() && bias.value().defined()) {
        return CSALinearFunction::apply(x, latent_weight, alpha, bias.value());
    }
    return CSALinearFunction::apply(x, latent_weight, alpha, c10::nullopt);
}

} // namespace autograd
} // namespace rkmj

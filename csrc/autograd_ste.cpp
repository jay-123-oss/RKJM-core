#include "include/autograd_ste.h"
#include <omp.h>
#include <cmath>
#include <vector>

namespace rkmj {
namespace autograd {

torch::Tensor csa_linear_ste_forward(
    torch::Tensor x,
    torch::Tensor latent_weight,
    torch::Tensor alpha,
    c10::optional<torch::Tensor> bias
) {
    x = x.contiguous().to(torch::kFloat32);
    latent_weight = latent_weight.contiguous().to(torch::kFloat32);
    alpha = alpha.contiguous().to(torch::kFloat32);

    const int64_t orig_dim0 = x.dim() > 2 ? x.size(0) : 1;
    const int64_t M = x.numel() / x.size(-1);
    const int64_t K = x.size(-1);
    const int64_t N = latent_weight.size(0);

    auto x_2d = x.view({M, K});
    auto w_quant = torch::zeros({N, K}, torch::kFloat32);
    auto out = torch::zeros({M, N}, torch::kFloat32);

    const float* w_ptr = latent_weight.data_ptr<float>();
    const float* a_ptr = alpha.data_ptr<float>();
    float* wq_ptr = w_quant.data_ptr<float>();

    // Quantize latent weights to {-1, 0, +1}
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < N; ++row) {
        const float a = std::max(a_ptr[row], 1e-5f);
        const float* src_row = w_ptr + row * K;
        float* dst_row = wq_ptr + row * K;

        for (int64_t col = 0; col < K; ++col) {
            const float val = src_row[col] / a;
            float q = std::round(val);
            if (q > 1.0f) q = 1.0f;
            else if (q < -1.0f) q = -1.0f;
            dst_row[col] = q;
        }
    }

    // Forward GEMM: Y = X * (W_quant * alpha)^T + bias
    auto effective_w = w_quant * alpha.view({N, 1});
    out = torch::matmul(x_2d, effective_w.t());

    if (bias.has_value() && bias.value().defined()) {
        out = out + bias.value();
    }

    if (x.dim() == 3) {
        return out.view({orig_dim0, x.size(1), N});
    }
    return out;
}

std::vector<torch::Tensor> csa_linear_ste_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor latent_weight,
    torch::Tensor alpha,
    bool has_bias
) {
    grad_output = grad_output.contiguous().to(torch::kFloat32);
    x = x.contiguous().to(torch::kFloat32);
    latent_weight = latent_weight.contiguous().to(torch::kFloat32);
    alpha = alpha.contiguous().to(torch::kFloat32);

    const int64_t M = x.numel() / x.size(-1);
    const int64_t K = x.size(-1);
    const int64_t N = latent_weight.size(0);

    auto grad_out_2d = grad_output.view({M, N});
    auto x_2d = x.view({M, K});

    // 1. Recompute W_quant
    auto w_quant = torch::zeros({N, K}, torch::kFloat32);
    const float* w_ptr = latent_weight.data_ptr<float>();
    const float* a_ptr = alpha.data_ptr<float>();
    float* wq_ptr = w_quant.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < N; ++row) {
        const float a = std::max(a_ptr[row], 1e-5f);
        const float* src_row = w_ptr + row * K;
        float* dst_row = wq_ptr + row * K;

        for (int64_t col = 0; col < K; ++col) {
            const float val = src_row[col] / a;
            float q = std::round(val);
            if (q > 1.0f) q = 1.0f;
            else if (q < -1.0f) q = -1.0f;
            dst_row[col] = q;
        }
    }

    // 2. grad_x = grad_output * (W_quant * alpha)
    auto effective_w = w_quant * alpha.view({N, 1});
    auto grad_x = torch::matmul(grad_out_2d, effective_w);

    // 3. grad_W = grad_output^T * x
    auto grad_w_unclipped = torch::matmul(grad_out_2d.t(), x_2d) * alpha.view({N, 1});

    // 4. Straight-Through Estimator Gradient Mask: I(|W_latent| <= 1.0)
    auto grad_latent = torch::zeros_like(latent_weight);
    const float* g_src = grad_w_unclipped.data_ptr<float>();
    float* g_dst = grad_latent.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < N; ++row) {
        const float* src_row = w_ptr + row * K;
        const float* grad_row = g_src + row * K;
        float* dst_row = g_dst + row * K;

        for (int64_t col = 0; col < K; ++col) {
            const float val = src_row[col];
            // STE pass-through only if |W_latent| <= 1.0
            if (std::abs(val) <= 1.0f) {
                dst_row[col] = grad_row[col];
            } else {
                dst_row[col] = 0.0f;
            }
        }
    }

    // 5. grad_alpha = sum(grad_output * (x * W_quant^T)) along batch
    auto x_wq = torch::matmul(x_2d, w_quant.t());
    auto grad_alpha = (grad_out_2d * x_wq).sum(0);

    // 6. grad_bias
    torch::Tensor grad_bias;
    if (has_bias) {
        grad_bias = grad_out_2d.sum(0);
    } else {
        grad_bias = torch::Tensor();
    }

    return {grad_x.view(x.sizes()), grad_latent, grad_alpha, grad_bias};
}

} // namespace autograd
} // namespace rkmj

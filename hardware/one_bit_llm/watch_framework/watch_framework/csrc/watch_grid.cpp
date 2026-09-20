#include "watch_grid.h"
#include <torch/extension.h>
#include <torch/autograd.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <cstdint>

// -----------------------------------------------------------------------------
// Bitwise CSA (Carry-Save Addition) / Popcount Fast 1-bit GEMM
// -----------------------------------------------------------------------------
// For 1-bit binary values X_q, W_q in {-1, +1}:
// Let B_x, B_w in {0, 1} where val > 0 -> 1, val < 0 -> 0.
// x_k * w_k = +1 if B_x == B_w, else -1
// x_k * w_k = 1 - 2 * (B_x ^ B_w)
// Sum_k (x_k * w_k) = K - 2 * popcount(B_x ^ B_w)
// -----------------------------------------------------------------------------

torch::Tensor watch_grid_csa_gemm(
    const torch::Tensor& x_q,
    const torch::Tensor& w_q
) {
    TORCH_CHECK(x_q.is_contiguous(), "x_q must be contiguous");
    TORCH_CHECK(w_q.is_contiguous(), "w_q must be contiguous");
    TORCH_CHECK(x_q.dim() == 2, "x_q must be a 2D tensor [B, K]");
    TORCH_CHECK(w_q.dim() == 2, "w_q must be a 2D tensor [N, K]");
    TORCH_CHECK(x_q.size(1) == w_q.size(1), "Inner dimension K must match");

    const int64_t B = x_q.size(0);
    const int64_t K = x_q.size(1);
    const int64_t N = w_q.size(0);

    auto out = torch::empty({B, N}, x_q.options().dtype(torch::kFloat32));

    const float* x_ptr = x_q.data_ptr<float>();
    const float* w_ptr = w_q.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const int64_t num_words = (K + 63) / 64;

    // Pack binary representation: 64 bits per uint64_t
    std::vector<uint64_t> x_bits(B * num_words, 0ULL);
    std::vector<uint64_t> w_bits(N * num_words, 0ULL);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t w = 0; w < num_words; ++w) {
            uint64_t word_val = 0ULL;
            const int64_t k_start = w * 64;
            const int64_t k_end = std::min(k_start + 64, K);
            for (int64_t k = k_start; k < k_end; ++k) {
                if (x_ptr[b * K + k] > 0.0f) {
                    word_val |= (1ULL << (k - k_start));
                }
            }
            x_bits[b * num_words + w] = word_val;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t w = 0; w < num_words; ++w) {
            uint64_t word_val = 0ULL;
            const int64_t k_start = w * 64;
            const int64_t k_end = std::min(k_start + 64, K);
            for (int64_t k = k_start; k < k_end; ++k) {
                if (w_ptr[n * K + k] > 0.0f) {
                    word_val |= (1ULL << (k - k_start));
                }
            }
            w_bits[n * num_words + w] = word_val;
        }
    }

    // Carry-Save Popcount Reduction Grid
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            const uint64_t* bx = &x_bits[b * num_words];
            const uint64_t* bw = &w_bits[n * num_words];
            int64_t total_diff = 0;

            #pragma omp simd reduction(+:total_diff)
            for (int64_t w = 0; w < num_words; ++w) {
                uint64_t xor_val = bx[w] ^ bw[w];
                // For the last word, mask out unused bits
                if (w == num_words - 1 && (K % 64 != 0)) {
                    uint64_t valid_mask = (1ULL << (K % 64)) - 1ULL;
                    xor_val &= valid_mask;
                }
                total_diff += __builtin_popcountll(xor_val);
            }

            int64_t dot = K - 2 * total_diff;
            out_ptr[b * N + n] = static_cast<float>(dot);
        }
    }

    return out;
}

// -----------------------------------------------------------------------------
// Forward Pass: Dynamic Alpha, Quantization, GEMM, and Bias
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> watch_grid_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.defined(), "Input x must be defined");
    TORCH_CHECK(weight.defined(), "Weight must be defined");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");

    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();

    const auto orig_shape = x.sizes();
    const int64_t in_features = weight.size(1);
    const int64_t out_features = weight.size(0);

    TORCH_CHECK(x_contig.size(-1) == in_features,
        "Input feature dimension (", x_contig.size(-1),
        ") must match weight in_features (", in_features, ")");

    // Flatten leading batch dimensions to 2D matrix [B, K]
    const int64_t B = x.numel() / in_features;
    auto x_2d = x_contig.reshape({B, in_features});

    // 1. Dynamic scale factor (Alpha): alpha = (1 / N) * sum(|W_i|)
    torch::Tensor alpha = weight_contig.abs().mean();

    // 2. Sign quantization: {-1, +1}
    torch::Tensor x_q = torch::where(x_2d >= 0.0f,
                                     torch::ones_like(x_2d),
                                     -torch::ones_like(x_2d));

    torch::Tensor w_q = torch::where(weight_contig >= 0.0f,
                                     torch::ones_like(weight_contig),
                                     -torch::ones_like(weight_contig));

    // 3. Bitwise CSA GEMM: Y_unscaled = X_q @ W_q^T
    torch::Tensor y_unscaled = watch_grid_csa_gemm(x_q, w_q);

    // 4. Output scaling: Y = alpha * Y_unscaled
    torch::Tensor y_2d = y_unscaled * alpha;

    // 5. Optional bias
    if (bias.has_value() && bias.value().defined()) {
        auto b = bias.value().contiguous();
        TORCH_CHECK(b.numel() == out_features, "Bias size must match out_features");
        y_2d = y_2d + b.view({1, out_features});
    }

    // Reshape output to match input batch dimensions + [out_features]
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
    out_shape.push_back(out_features);
    torch::Tensor y = y_2d.reshape(out_shape);

    // Returns: [y, x_q, w_q, alpha, y_unscaled]
    return {y, x_q, w_q, alpha, y_unscaled};
}

// -----------------------------------------------------------------------------
// Backward Pass: Straight-Through Estimator (STE)
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> watch_grid_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& alpha,
    bool has_bias
) {
    TORCH_CHECK(grad_output.defined(), "grad_output must be defined");
    auto grad_out_contig = grad_output.contiguous().to(torch::kFloat32);

    const int64_t in_features = weight.size(1);
    const int64_t out_features = weight.size(0);
    const int64_t B = grad_out_contig.numel() / out_features;

    auto grad_out_2d = grad_out_contig.reshape({B, out_features});

    // 1. Grad Input: dL/dX = alpha * (dL/dY @ W_q)
    torch::Tensor grad_x_2d = alpha * torch::matmul(grad_out_2d, w_q);
    torch::Tensor grad_x = grad_x_2d.reshape(x.sizes()).to(x.dtype());

    // 2. Grad Weight with STE indicator clip: dL/dW = alpha * (dL/dY^T @ X_q) * Indicator(|W| <= 1.0)
    torch::Tensor ste_mask = (weight.abs() <= 1.0f).to(weight.dtype());
    torch::Tensor grad_w_unclipped = alpha * torch::matmul(grad_out_2d.t(), x_q);
    torch::Tensor grad_weight = grad_w_unclipped * ste_mask;

    // 3. Grad Alpha: dL/dalpha = sum(dL/dY * (X_q @ W_q^T))
    torch::Tensor y_unscaled = torch::matmul(x_q, w_q.t());
    torch::Tensor grad_alpha = (grad_out_2d * y_unscaled).sum();

    // 4. Grad Bias (if present)
    torch::Tensor grad_bias;
    if (has_bias) {
        grad_bias = grad_out_2d.sum(0);
    }

    return {grad_x, grad_weight, grad_bias, grad_alpha};
}

// -----------------------------------------------------------------------------
// PyTorch Autograd Function Integration
// -----------------------------------------------------------------------------
class WatchGridFunction : public torch::autograd::Function<WatchGridFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor x,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias
    ) {
        auto res = watch_grid_forward(x, weight, bias);
        torch::Tensor y = res[0];
        torch::Tensor x_q = res[1];
        torch::Tensor w_q = res[2];
        torch::Tensor alpha = res[3];

        bool has_bias = bias.has_value() && bias.value().defined();
        ctx->save_for_backward({x, weight, x_q, w_q, alpha});
        ctx->saved_data["has_bias"] = has_bias;

        return {y};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor x = saved[0];
        torch::Tensor weight = saved[1];
        torch::Tensor x_q = saved[2];
        torch::Tensor w_q = saved[3];
        torch::Tensor alpha = saved[4];
        bool has_bias = ctx->saved_data["has_bias"].toBool();

        auto grads = watch_grid_backward(grad_outputs[0], x, weight, x_q, w_q, alpha, has_bias);

        torch::autograd::variable_list result(3);
        result[0] = grads[0]; // grad_x
        result[1] = grads[1]; // grad_weight
        if (has_bias) {
            result[2] = grads[2]; // grad_bias
        } else {
            result[2] = torch::Tensor();
        }
        return result;
    }
};

torch::Tensor watch_grid_autograd_apply(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias
) {
    return WatchGridFunction::apply(x, weight, bias)[0];
}

// -----------------------------------------------------------------------------
// PyBind11 Module Registration
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Watch Framework 2D Grid C++/OpenMP CSA Acceleration Engine";
    m.def("csa_gemm", &watch_grid_csa_gemm, "Bitwise CSA GEMM Kernel (OpenMP)",
          py::arg("x_q"), py::arg("w_q"));
    m.def("forward", &watch_grid_forward, "Watch Grid Forward Pass",
          py::arg("x"), py::arg("weight"), py::arg("bias") = c10::nullopt);
    m.def("backward", &watch_grid_backward, "Watch Grid Backward Pass",
          py::arg("grad_output"), py::arg("x"), py::arg("weight"),
          py::arg("x_q"), py::arg("w_q"), py::arg("alpha"), py::arg("has_bias"));
    m.def("apply", &watch_grid_autograd_apply, "Watch Grid Autograd Apply",
          py::arg("x"), py::arg("weight"), py::arg("bias") = c10::nullopt);
}

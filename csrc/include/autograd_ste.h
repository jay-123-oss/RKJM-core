#pragma once

#include <torch/extension.h>

namespace rkmj {
namespace autograd {

// Forward evaluation with discrete ternary weights W_quant in {-1, 0, +1}
// Backward gradient propagation directly to W_latent bounded by indicator |W_latent| <= 1.0
torch::Tensor csa_linear_ste_forward(
    torch::Tensor x,
    torch::Tensor latent_weight,
    torch::Tensor alpha,
    c10::optional<torch::Tensor> bias
);

std::vector<torch::Tensor> csa_linear_ste_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor latent_weight,
    torch::Tensor alpha,
    bool has_bias
);

} // namespace autograd
} // namespace rkmj

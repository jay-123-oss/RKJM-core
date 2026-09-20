#pragma once

#include <torch/extension.h>
#include <torch/autograd.h>
#include <c10/util/Optional.h>

namespace rkmj {
namespace autograd {

// -----------------------------------------------------------------------------
// CSALinearFunction: PyTorch Autograd Function Interface with STE
// -----------------------------------------------------------------------------

class CSALinearFunction : public torch::autograd::Function<CSALinearFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x,
        torch::Tensor latent_weight,
        torch::Tensor alpha,
        c10::optional<torch::Tensor> bias = c10::nullopt
    );

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    );
};

torch::Tensor csa_linear_forward_autograd(
    torch::Tensor x,
    torch::Tensor latent_weight,
    torch::Tensor alpha,
    c10::optional<torch::Tensor> bias = c10::nullopt
);

} // namespace autograd
} // namespace rkmj

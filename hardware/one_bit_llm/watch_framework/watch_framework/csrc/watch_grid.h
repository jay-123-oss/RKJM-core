#pragma once

#include <torch/extension.h>
#include <vector>

// High performance 1-bit GEMM with Carry-Save Addition (CSA) & OpenMP acceleration
torch::Tensor watch_grid_csa_gemm(
    const torch::Tensor& x_q,
    const torch::Tensor& w_q
);

// Forward pass: dynamic alpha scaling, sign quantization, bitwise OpenMP GEMM
std::vector<torch::Tensor> watch_grid_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias = c10::nullopt
);

// Backward pass: STE gradients with Indicator(|W| <= 1.0)
std::vector<torch::Tensor> watch_grid_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& alpha,
    bool has_bias
);

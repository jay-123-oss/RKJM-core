#include "include/csa_common.h"
#include "include/csa_autograd.h"
#include <torch/extension.h>

// Forward declarations of CPU functions
namespace rkmj {
namespace csa {
    torch::Tensor csa_linear_forward(
        torch::Tensor x,
        torch::Tensor w_packed,
        torch::Tensor alpha,
        c10::optional<torch::Tensor> bias
    );
    torch::Tensor pack_weights_cpu(torch::Tensor w_ternary);
    torch::Tensor unpack_weights_cpu(torch::Tensor w_packed, int64_t K);
}
namespace cuda {
    void bitwise_csa_forward(
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor Sum,
        torch::Tensor Carry
    );
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RKMJ-Core C++/CUDA Hardware Engine (Carry-Save Addition Popcount & STE Autograd)";

    // Training / Dynamic Autograd entry point
    m.def(
        "csa_linear",
        &rkmj::autograd::csa_linear_forward_autograd,
        "1.58-bit CSA Linear Forward + STE Backward Autograd (CPU OpenMP)",
        py::arg("x"),
        py::arg("latent_weight"),
        py::arg("alpha"),
        py::arg("bias") = py::none()
    );

    // Ultra-fast frozen inference entry point
    m.def(
        "csa_forward",
        &rkmj::csa::csa_linear_forward,
        "1.58-bit CSA Linear Forward on 2-bit Packed Weights (CPU OpenMP Popcount)",
        py::arg("x"),
        py::arg("w_packed"),
        py::arg("alpha"),
        py::arg("bias") = py::none()
    );

    // Bit-packing routines
    m.def(
        "pack_weights",
        &rkmj::csa::pack_weights_cpu,
        "Pack ternary weights [-1, 0, +1] into uint32 bitfields (16 weights/word)",
        py::arg("w_ternary")
    );

    m.def(
        "unpack_weights",
        &rkmj::csa::unpack_weights_cpu,
        "Unpack uint32 bitfields into FP32 ternary weights",
        py::arg("w_packed"),
        py::arg("K")
    );

    // Bitwise CSA hardware kernel (CPU & CUDA)
    m.def(
        "bitwise_csa_forward",
        &rkmj::cuda::bitwise_csa_forward,
        "2D Bitwise Carry-Save Addition Forward (CPU OpenMP & CUDA)",
        py::arg("A"),
        py::arg("B"),
        py::arg("Sum"),
        py::arg("Carry")
    );
}

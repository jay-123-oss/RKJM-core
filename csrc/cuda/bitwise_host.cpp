#include <torch/extension.h>
#include <cstdint>

#if defined(WITH_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

void launch_bitwise_csa_kernel(
    const uint32_t* A,
    const uint32_t* B,
    uint32_t* Sum,
    uint32_t* Carry,
    int num_elements,
    cudaStream_t stream);
#endif

namespace rkmj {
namespace cuda {

// High performance CPU fallback implementation
void bitwise_csa_cpu(
    const uint32_t* A,
    const uint32_t* B,
    uint32_t* Sum,
    uint32_t* Carry,
    int num_elements)
{
    #pragma omp parallel for
    for (int i = 0; i < num_elements; i++) {
        uint32_t a_val = A[i];
        uint32_t b_val = B[i];
        uint32_t s_val = Sum[i];
        uint32_t c_val = Carry[i];

        uint32_t product = a_val & b_val;
        uint32_t new_sum = product ^ s_val ^ c_val;
        uint32_t new_carry = ((product & s_val) | (s_val & c_val) | (c_val & product)) << 1;

        Sum[i] = new_sum;
        Carry[i] = new_carry;
    }
}

// Host Function with Tensor Checks
void bitwise_csa_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Sum,
    torch::Tensor Carry) 
{
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Tensor B must be contiguous");
    TORCH_CHECK(Sum.is_contiguous(), "Tensor Sum must be contiguous");
    TORCH_CHECK(Carry.is_contiguous(), "Tensor Carry must be contiguous");

    TORCH_CHECK(A.scalar_type() == torch::kInt32, "Tensor A must be Int32");
    TORCH_CHECK(B.scalar_type() == torch::kInt32, "Tensor B must be Int32");

    int num_elements = A.numel();

    if (A.is_cuda()) {
#if defined(WITH_CUDA) || defined(__CUDACC__)
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        launch_bitwise_csa_kernel(
            reinterpret_cast<const uint32_t*>(A.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(B.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(Sum.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(Carry.data_ptr<int32_t>()),
            num_elements,
            stream
        );
#else
        TORCH_CHECK(false, "Extension was compiled without CUDA support");
#endif
    } else {
        bitwise_csa_cpu(
            reinterpret_cast<const uint32_t*>(A.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(B.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(Sum.data_ptr<int32_t>()),
            reinterpret_cast<uint32_t*>(Carry.data_ptr<int32_t>()),
            num_elements
        );
    }
}

} // namespace cuda
} // namespace rkmj
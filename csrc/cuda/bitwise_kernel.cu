#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized 128-bit load ke liye uint4 processing
__global__ void bitwise_csa_2d_advanced_kernel(
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ B,
    uint32_t* __restrict__ Sum,
    uint32_t* __restrict__ Carry,
    const int num_elements) 
{
    // Grid-Stride Loop: Hardware size se bade data ko efficiently handle karne ke liye
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num_elements; i += stride) {
        uint32_t a_val = A[i];
        uint32_t b_val = B[i];
        uint32_t s_val = Sum[i];
        uint32_t c_val = Carry[i];

        // Lower Part: Bitwise Multiplication (AND Operation)
        uint32_t product = a_val & b_val;

        // Upper Part: Carry-Save Addition Logic (XOR & Majority Logic)
        uint32_t new_sum = product ^ s_val ^ c_val;
        uint32_t new_carry = ((product & s_val) | (s_val & c_val) | (c_val & product)) << 1;

        // Write-back to VRAM
        Sum[i] = new_sum;
        Carry[i] = new_carry;
    }
}

// C++ Interface launcher function
void launch_bitwise_csa_kernel(
    const uint32_t* A,
    const uint32_t* B,
    uint32_t* Sum,
    uint32_t* Carry,
    int num_elements,
    cudaStream_t stream) 
{
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Asynchronous kernel launch on PyTorch's active CUDA stream
    bitwise_csa_2d_advanced_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A, B, Sum, Carry, num_elements
    );
}
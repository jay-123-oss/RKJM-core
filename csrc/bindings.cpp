#include "include/csa_common.h"
#include "include/csa_autograd.h"
#include "include/kv_cache_ops.hpp"
#include "include/mmap_streamer.hpp"
#include "include/allocator.h"
#include "include/pack.h"
#include "include/kernels_avx2.h"
#include "include/scheduler.h"
#include "include/autograd_ste.h"
#include <torch/extension.h>
#include <string>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// Forward declarations of legacy CPU functions
namespace rkmj {
namespace csa {
    torch::Tensor csa_linear_forward(
        torch::Tensor x,
        torch::Tensor w_packed,
        torch::Tensor alpha,
        c10::optional<torch::Tensor> bias
    );
    torch::Tensor fused_rmsnorm_csa_forward(
        torch::Tensor x,
        torch::Tensor rmsnorm_weight,
        double eps,
        torch::Tensor w_packed,
        torch::Tensor alpha,
        c10::optional<torch::Tensor> bias
    );
    torch::Tensor pack_weights_cpu(torch::Tensor w_ternary);
    torch::Tensor unpack_weights_cpu(torch::Tensor w_packed, int64_t K);
    std::string get_simd_backend_name();
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
    m.doc() = "RKMJ-Core C++/CUDA Hardware Engine (Multi-ISA Popcount, Fused Ops, STE Autograd & Dynamic Memory Arena)";

    // ==========================================
    // 1. Core Memory Pool & Allocator
    // ==========================================
    m.def(
        "recycle_buffer",
        [](torch::Tensor t) {
            rkmj::core::get_global_pool().recycle(t);
        },
        "Recycle activation tensor into segregated memory pool",
        py::arg("tensor")
    );

    m.def(
        "cached_buffers_count",
        []() -> size_t {
            return rkmj::core::get_global_pool().cached_tensors_count();
        },
        "Number of cached reusable activation buffers in pool"
    );

    m.def(
        "clear_memory_pool",
        []() {
            rkmj::core::get_global_pool().clear();
        },
        "Flush segregated memory pool"
    );

    m.def(
        "force_heap_trim",
        &rkmj::core::force_heap_trim,
        "Force glibc to immediately return free heap pages to the OS via malloc_trim(0)"
    );

    m.def(
        "madvise_dontneed_buffer",
        [](py::buffer b) {
#if defined(__linux__)
            py::buffer_info info = b.request();
            if (info.ptr != nullptr && info.size > 0) {
                size_t bytes = info.size * info.itemsize;
                uintptr_t addr = reinterpret_cast<uintptr_t>(info.ptr);
                size_t page_size = sysconf(_SC_PAGESIZE);
                uintptr_t page_start = addr & ~(page_size - 1);
                size_t len = (addr + bytes) - page_start;
                madvise(reinterpret_cast<void*>(page_start), len, MADV_DONTNEED);
            }
#endif
        },
        "Evict physical RAM pages for any buffer (e.g. mmap) using madvise(MADV_DONTNEED)"
    );

    // ==========================================
    // 2. Numerically Stable 1.58-bit Quantizer & Bit-Packing
    // ==========================================
    m.def(
        "pack_weights_2bit",
        &rkmj::quant::pack_ternary_2bit_cpu,
        "Pack ternary weights [-1, 0, +1] into 2-bit aligned uint32 bitfield (16 weights/word)",
        py::arg("w_ternary")
    );

    m.def(
        "unpack_weights_2bit",
        &rkmj::quant::unpack_ternary_2bit_cpu,
        "Unpack 2-bit aligned uint32 bitfield into FP32 ternary weights with exact arithmetic shifts",
        py::arg("w_packed"),
        py::arg("K")
    );

    m.def(
        "quantize_grouped",
        &rkmj::quant::quantize_grouped_cpu,
        "Per-Group Dynamic Scaling Quantizer: (w_packed, scales) = quantize_grouped(weight, group_size)",
        py::arg("weight"),
        py::arg("group_size") = 64
    );

    m.def(
        "dequantize_grouped",
        &rkmj::quant::dequantize_grouped_cpu,
        "Per-Group Dynamic Scaling Dequantizer",
        py::arg("w_packed"),
        py::arg("scales"),
        py::arg("in_features"),
        py::arg("group_size") = 64
    );

    // Legacy bit-packing routines (compatible with earlier models)
    m.def(
        "pack_weights",
        &rkmj::csa::pack_weights_cpu,
        "Pack ternary weights [-1, 0, +1] into uint32 bitfields",
        py::arg("w_ternary")
    );

    m.def(
        "unpack_weights",
        &rkmj::csa::unpack_weights_cpu,
        "Unpack uint32 bitfields into FP32 ternary weights",
        py::arg("w_packed"),
        py::arg("K")
    );

    // ==========================================
    // 3. High-Performance CPU Execution Engine
    // ==========================================
    m.def(
        "gemv_tiled_csa",
        &rkmj::cpu::gemv_tiled_csa_cpu,
        "Cache-Aware Tiled GEMV on 2-bit Packed Ternary Weights (AVX2/FMA/BMI2)",
        py::arg("x"),
        py::arg("w_packed"),
        py::arg("scales"),
        py::arg("bias") = py::none(),
        py::arg("group_size") = 64
    );

    m.def(
        "gemm_tiled_csa",
        &rkmj::cpu::gemm_tiled_csa_cpu,
        "Cache-Aware Tiled GEMM on 2-bit Packed Ternary Weights (AVX2/FMA/BMI2)",
        py::arg("x"),
        py::arg("w_packed"),
        py::arg("scales"),
        py::arg("bias") = py::none(),
        py::arg("group_size") = 64
    );

    // Frozen inference popcount engine
    m.def(
        "csa_forward",
        &rkmj::csa::csa_linear_forward,
        "1.58-bit CSA Linear Forward on 2-bit Packed Weights (Multi-ISA CPU OpenMP Popcount)",
        py::arg("x"),
        py::arg("w_packed"),
        py::arg("alpha"),
        py::arg("bias") = py::none()
    );

    // Operator Fusion entry point (RMSNorm + Activation Quantization + CSA Popcount)
    m.def(
        "fused_rmsnorm_csa_forward",
        &rkmj::csa::fused_rmsnorm_csa_forward,
        "Fused RMSNorm + Activation Quantization + 1.58-bit CSA Linear Forward",
        py::arg("x"),
        py::arg("rmsnorm_weight"),
        py::arg("eps"),
        py::arg("w_packed"),
        py::arg("alpha"),
        py::arg("bias") = py::none()
    );

    // SIMD Backend and CPU Thread Affinity
    m.def(
        "get_simd_backend_name",
        &rkmj::csa::get_simd_backend_name,
        "Returns active multi-ISA SIMD acceleration backend name"
    );

    m.def(
        "set_thread_affinity",
        &rkmj::scheduler::set_thread_affinity,
        "Bind worker threads to physical CPU cores to eliminate NUMA thrashing",
        py::arg("num_threads") = -1
    );

    m.def(
        "get_num_physical_cores",
        &rkmj::scheduler::get_num_physical_cores,
        "Returns count of detected physical CPU cores"
    );

    // ==========================================
    // 4. Straight-Through Estimator (STE) Autograd Engine
    // ==========================================
    m.def(
        "csa_ste_forward",
        &rkmj::autograd::csa_linear_ste_forward,
        "1.58-bit CSA Linear STE Forward",
        py::arg("x"),
        py::arg("latent_weight"),
        py::arg("alpha"),
        py::arg("bias") = py::none()
    );

    m.def(
        "csa_ste_backward",
        &rkmj::autograd::csa_linear_ste_backward,
        "1.58-bit CSA Linear STE Backward (Returns grad_x, grad_latent_weight, grad_alpha, grad_bias)",
        py::arg("grad_output"),
        py::arg("x"),
        py::arg("latent_weight"),
        py::arg("alpha"),
        py::arg("has_bias")
    );

    // Legacy autograd entry point
    m.def(
        "csa_linear",
        &rkmj::autograd::csa_linear_forward_autograd,
        "1.58-bit CSA Linear Forward + STE Backward Autograd (CPU OpenMP)",
        py::arg("x"),
        py::arg("latent_weight"),
        py::arg("alpha"),
        py::arg("bias") = py::none()
    );

    // ==========================================
    // 5. KV Cache & I/O Streamer
    // ==========================================
    m.def(
        "kv_cache_update",
        &rkmj::cache::kv_cache_update_cpu,
        "Contiguous Pointer-based Zero-Copy KV Cache Update (CPU)",
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("k_state"),
        py::arg("v_state"),
        py::arg("layer_idx"),
        py::arg("start_pos")
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

    // Asynchronous Double-Buffered Ring Prefetcher
    py::class_<rkmj::io::AsyncRingStreamer>(m, "AsyncRingStreamer")
        .def(py::init<const std::string&, const std::vector<int64_t>&, const std::vector<int64_t>&>(),
             py::arg("filepath"), py::arg("layer_offsets"), py::arg("layer_sizes"))
        .def("start", &rkmj::io::AsyncRingStreamer::start, py::arg("start_layer_idx") = 0)
        .def("acquire_compute_layer", &rkmj::io::AsyncRingStreamer::acquire_compute_layer, py::arg("layer_idx"))
        .def("release_and_prefetch_next", &rkmj::io::AsyncRingStreamer::release_and_prefetch_next)
        .def("stop", &rkmj::io::AsyncRingStreamer::stop)
        .def("get_buffer_size_bytes", &rkmj::io::AsyncRingStreamer::get_buffer_size_bytes)
        .def("get_num_layers", &rkmj::io::AsyncRingStreamer::get_num_layers);
}

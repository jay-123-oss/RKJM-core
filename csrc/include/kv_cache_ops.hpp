#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace rkmj {
namespace cache {

// -----------------------------------------------------------------------------
// Contiguous C++ Zero-Copy Pointer-Based KV Cache Update
//
// Layout:
//   k_cache, v_cache: [n_layers, max_batch, n_kv_heads, max_seq_len, head_dim]
//   k_state, v_state: [batch_size, n_kv_heads, seq_len, head_dim]
//
// Replaces PyTorch Python-level tensor slice copy operations with direct
// pointer arithmetic and SIMD/memcpy zero-heap block transfer.
// -----------------------------------------------------------------------------

inline void kv_cache_update_cpu(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_state,
    torch::Tensor v_state,
    int64_t layer_idx,
    int64_t start_pos
) {
    TORCH_CHECK(k_cache.is_cpu() && v_cache.is_cpu(), "Caches must be CPU tensors");
    TORCH_CHECK(k_state.is_cpu() && v_state.is_cpu(), "States must be CPU tensors");
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(), "Caches must be contiguous");
    TORCH_CHECK(k_state.is_contiguous() && v_state.is_contiguous(), "States must be contiguous");

    TORCH_CHECK(k_cache.dim() == 5, "k_cache must be 5D [layers, batch, heads, seq, dim]");
    TORCH_CHECK(k_state.dim() == 4, "k_state must be 4D [batch, heads, seq, dim]");

    const int64_t max_batch = k_cache.size(1);
    const int64_t max_heads = k_cache.size(2);
    const int64_t max_seq = k_cache.size(3);
    const int64_t head_dim = k_cache.size(4);

    const int64_t batch_size = k_state.size(0);
    const int64_t n_kv_heads = k_state.size(1);
    const int64_t seq_len = k_state.size(2);

    TORCH_CHECK(batch_size <= max_batch, "batch_size exceeds max_batch in cache");
    TORCH_CHECK(n_kv_heads <= max_heads, "n_kv_heads exceeds max_heads in cache");
    TORCH_CHECK(start_pos + seq_len <= max_seq, "Sequence position exceeds cache capacity");

    float* k_cache_ptr = k_cache.data_ptr<float>();
    float* v_cache_ptr = v_cache.data_ptr<float>();
    const float* k_src_ptr = k_state.data_ptr<float>();
    const float* v_src_ptr = v_state.data_ptr<float>();

    const int64_t layer_stride = max_batch * max_heads * max_seq * head_dim;
    const int64_t batch_stride = max_heads * max_seq * head_dim;
    const int64_t head_stride = max_seq * head_dim;

    const int64_t src_batch_stride = n_kv_heads * seq_len * head_dim;
    const int64_t src_head_stride = seq_len * head_dim;

    const size_t copy_bytes = static_cast<size_t>(seq_len * head_dim * sizeof(float));

    float* k_layer_base = k_cache_ptr + layer_idx * layer_stride;
    float* v_layer_base = v_cache_ptr + layer_idx * layer_stride;

    for (int64_t b = 0; b < batch_size; ++b) {
        float* k_b_base = k_layer_base + b * batch_stride;
        float* v_b_base = v_layer_base + b * batch_stride;

        const float* k_b_src = k_src_ptr + b * src_batch_stride;
        const float* v_b_src = v_src_ptr + b * src_batch_stride;

        for (int64_t h = 0; h < n_kv_heads; ++h) {
            float* k_dst = k_b_base + h * head_stride + start_pos * head_dim;
            float* v_dst = v_b_base + h * head_stride + start_pos * head_dim;

            const float* k_src = k_b_src + h * src_head_stride;
            const float* v_src = v_b_src + h * src_head_stride;

            std::memcpy(k_dst, k_src, copy_bytes);
            std::memcpy(v_dst, v_src, copy_bytes);
        }
    }
}

} // namespace cache
} // namespace rkmj

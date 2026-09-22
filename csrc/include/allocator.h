#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <torch/extension.h>

namespace rkmj {
namespace core {

class VirtualMemoryArena {
public:
    explicit VirtualMemoryArena(size_t reserve_bytes = 4ULL * 1024 * 1024 * 1024);
    ~VirtualMemoryArena();

    VirtualMemoryArena(const VirtualMemoryArena&) = delete;
    VirtualMemoryArena& operator=(const VirtualMemoryArena&) = delete;

    uint8_t* allocate(size_t bytes, size_t alignment = 64);
    void reset();

    size_t allocated_bytes() const;
    size_t total_reserved() const;

private:
    size_t total_reserved_;
    size_t allocated_bytes_;
    void* base_ptr_;
    mutable std::mutex mutex_;
};

class MemoryPool {
public:
    MemoryPool() = default;
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    torch::Tensor get_buffer(const std::vector<int64_t>& shape, torch::Dtype dtype, torch::Device device);
    void recycle(torch::Tensor tensor);
    void clear();
    size_t cached_tensors_count() const;

private:
    static size_t round_to_bucket(size_t bytes);
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<torch::Tensor>> free_pools_;
};

// Global singleton accessors
VirtualMemoryArena& get_global_arena();
MemoryPool& get_global_pool();

// Force OS heap trimming via malloc_trim
void force_heap_trim();

} // namespace core
} // namespace rkmj

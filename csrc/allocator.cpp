#include "include/allocator.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <malloc.h>
#endif

namespace rkmj {
namespace core {

VirtualMemoryArena::VirtualMemoryArena(size_t reserve_bytes)
    : total_reserved_(reserve_bytes), allocated_bytes_(0), base_ptr_(nullptr) {
#if defined(__linux__)
    base_ptr_ = mmap(
        nullptr,
        total_reserved_,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
        -1,
        0
    );
    if (base_ptr_ == MAP_FAILED) {
        base_ptr_ = nullptr;
        throw std::runtime_error("VirtualMemoryArena: mmap failed to reserve address space.");
    }
#else
    base_ptr_ = std::malloc(total_reserved_);
    if (!base_ptr_) {
        throw std::runtime_error("VirtualMemoryArena: std::malloc failed.");
    }
#endif
}

VirtualMemoryArena::~VirtualMemoryArena() {
    if (base_ptr_) {
#if defined(__linux__)
        munmap(base_ptr_, total_reserved_);
#else
        std::free(base_ptr_);
#endif
        base_ptr_ = nullptr;
    }
}

uint8_t* VirtualMemoryArena::allocate(size_t bytes, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t current_addr = reinterpret_cast<size_t>(base_ptr_) + allocated_bytes_;
    size_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
    size_t offset = aligned_addr - reinterpret_cast<size_t>(base_ptr_);

    if (offset + bytes > total_reserved_) {
        throw std::runtime_error("VirtualMemoryArena: Out of reserved virtual memory.");
    }

    allocated_bytes_ = offset + bytes;
    return reinterpret_cast<uint8_t*>(aligned_addr);
}

void VirtualMemoryArena::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(__linux__)
    if (base_ptr_ && allocated_bytes_ > 0) {
        madvise(base_ptr_, allocated_bytes_, MADV_DONTNEED);
    }
#endif
    allocated_bytes_ = 0;
}

size_t VirtualMemoryArena::allocated_bytes() const {
    return allocated_bytes_;
}

size_t VirtualMemoryArena::total_reserved() const {
    return total_reserved_;
}

// MemoryPool implementation
MemoryPool::~MemoryPool() {
    clear();
}

size_t MemoryPool::round_to_bucket(size_t bytes) {
    static const std::vector<size_t> buckets = {
        4ULL * 1024,
        64ULL * 1024,
        256ULL * 1024,
        1ULL * 1024 * 1024,
        4ULL * 1024 * 1024,
        16ULL * 1024 * 1024,
        64ULL * 1024 * 1024,
        256ULL * 1024 * 1024
    };
    for (size_t b : buckets) {
        if (bytes <= b) return b;
    }
    return ((bytes + 64ULL * 1024 * 1024 - 1) / (64ULL * 1024 * 1024)) * (64ULL * 1024 * 1024);
}

torch::Tensor MemoryPool::get_buffer(const std::vector<int64_t>& shape, torch::Dtype dtype, torch::Device device) {
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    size_t element_size = torch::elementSize(dtype);
    size_t total_bytes = num_elements * element_size;
    size_t bucket_size = round_to_bucket(total_bytes);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = free_pools_.find(bucket_size);
    if (it != free_pools_.end() && !it->second.empty()) {
        torch::Tensor cached = it->second.back();
        it->second.pop_back();
        if (cached.dtype() == dtype && cached.device() == device) {
            return cached.view(shape);
        }
    }

    int64_t bucket_elements = static_cast<int64_t>(bucket_size / element_size);
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    torch::Tensor new_tensor = torch::empty({bucket_elements}, options);
    return new_tensor.view(shape);
}

void MemoryPool::recycle(torch::Tensor tensor) {
    if (!tensor.is_contiguous()) {
        return;
    }
    size_t total_bytes = tensor.nbytes();
    size_t bucket_size = round_to_bucket(total_bytes);

    std::lock_guard<std::mutex> lock(mutex_);
    free_pools_[bucket_size].push_back(tensor.flatten());
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_pools_.clear();
}

size_t MemoryPool::cached_tensors_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& kv : free_pools_) {
        total += kv.second.size();
    }
    return total;
}

VirtualMemoryArena& get_global_arena() {
    static VirtualMemoryArena global_arena(8ULL * 1024 * 1024 * 1024); // 8GB virtual headroom
    return global_arena;
}

MemoryPool& get_global_pool() {
    static MemoryPool global_pool;
    return global_pool;
}

void force_heap_trim() {
#if defined(__linux__)
    malloc_trim(0);
#endif
}

} // namespace core
} // namespace rkmj

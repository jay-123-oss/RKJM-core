#pragma once

#include <torch/extension.h>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>

namespace rkmj {
namespace io {

/**
 * High-Throughput Asynchronous Double-Buffered Ring Prefetcher.
 * Overlaps background NVMe / SSD disk I/O (Layer N+1) with active CPU compute (Layer N)
 * using two 64-byte aligned DRAM buffers and POSIX asynchronous I/O advice.
 */
class AsyncRingStreamer {
public:
    AsyncRingStreamer(
        const std::string& filepath,
        const std::vector<int64_t>& layer_offsets,
        const std::vector<int64_t>& layer_sizes
    );
    ~AsyncRingStreamer();

    // Start background prefetcher thread
    void start(int64_t start_layer_idx = 0);

    // Acquire current compute buffer for layer_idx (blocks until I/O finishes)
    torch::Tensor acquire_compute_layer(int64_t layer_idx);

    // Signal compute completion and trigger prefetch for next layer
    void release_and_prefetch_next();

    // Stop background worker thread and clean up
    void stop();

    // Capacity queries
    int64_t get_buffer_size_bytes() const { return max_layer_size_; }
    int64_t get_num_layers() const { return num_layers_; }

private:
    void io_worker_loop();

    std::string filepath_;
    int fd_;
    std::vector<int64_t> layer_offsets_;
    std::vector<int64_t> layer_sizes_;
    int64_t num_layers_;
    int64_t max_layer_size_;

    // Contiguous 64-byte aligned buffers
    uint8_t* buffer_a_;
    uint8_t* buffer_b_;

    // Pointers representing roles
    uint8_t* compute_buf_;
    uint8_t* prefetch_buf_;

    int64_t current_compute_layer_;
    int64_t current_prefetch_layer_;

    // Thread synchronization
    std::thread io_thread_;
    std::mutex mtx_;
    std::condition_variable cv_compute_;
    std::condition_variable cv_io_;

    std::atomic<bool> is_running_;
    bool prefetch_ready_;
    bool prefetch_requested_;
};

} // namespace io
} // namespace rkmj

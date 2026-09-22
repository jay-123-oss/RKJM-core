#include "mmap_streamer.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace rkmj {
namespace io {

AsyncRingStreamer::AsyncRingStreamer(
    const std::string& filepath,
    const std::vector<int64_t>& layer_offsets,
    const std::vector<int64_t>& layer_sizes
) : filepath_(filepath),
    layer_offsets_(layer_offsets),
    layer_sizes_(layer_sizes),
    num_layers_(layer_offsets.size()),
    max_layer_size_(0),
    buffer_a_(nullptr),
    buffer_b_(nullptr),
    compute_buf_(nullptr),
    prefetch_buf_(nullptr),
    current_compute_layer_(-1),
    current_prefetch_layer_(-1),
    is_running_(false),
    prefetch_ready_(false),
    prefetch_requested_(false)
{
    if (layer_offsets.size() != layer_sizes.size()) {
        throw std::invalid_argument("layer_offsets and layer_sizes must have matching lengths");
    }
    if (num_layers_ == 0) {
        throw std::invalid_argument("Cannot initialize AsyncRingStreamer with 0 layers");
    }

    fd_ = open(filepath_.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open file for AsyncRingStreamer: " + filepath_);
    }

    // Determine peak layer memory requirement
    for (int64_t sz : layer_sizes_) {
        if (sz > max_layer_size_) {
            max_layer_size_ = sz;
        }
    }
    // Round to 64-byte alignment
    max_layer_size_ = ((max_layer_size_ + 63) / 64) * 64;

    // Allocate 64-byte aligned double buffers
    if (posix_memalign((void**)&buffer_a_, 64, max_layer_size_) != 0) {
        close(fd_);
        throw std::bad_alloc();
    }
    if (posix_memalign((void**)&buffer_b_, 64, max_layer_size_) != 0) {
        free(buffer_a_);
        close(fd_);
        throw std::bad_alloc();
    }

    compute_buf_ = buffer_a_;
    prefetch_buf_ = buffer_b_;
}

AsyncRingStreamer::~AsyncRingStreamer() {
    stop();
    if (buffer_a_) {
        free(buffer_a_);
        buffer_a_ = nullptr;
    }
    if (buffer_b_) {
        free(buffer_b_);
        buffer_b_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void AsyncRingStreamer::start(int64_t start_layer_idx) {
    if (is_running_) {
        return;
    }

    is_running_ = true;
    current_prefetch_layer_ = start_layer_idx;
    prefetch_ready_ = false;
    prefetch_requested_ = true;

    io_thread_ = std::thread(&AsyncRingStreamer::io_worker_loop, this);
}

void AsyncRingStreamer::stop() {
    if (!is_running_) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        is_running_ = false;
        prefetch_requested_ = false;
        cv_io_.notify_all();
        cv_compute_.notify_all();
    }

    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

void AsyncRingStreamer::io_worker_loop() {
    while (is_running_) {
        int64_t layer_to_load = -1;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_io_.wait(lock, [this]() {
                return !is_running_ || prefetch_requested_;
            });

            if (!is_running_) {
                break;
            }

            layer_to_load = current_prefetch_layer_;
            prefetch_requested_ = false;
        }

        if (layer_to_load >= 0 && layer_to_load < num_layers_) {
            int64_t offset = layer_offsets_[layer_to_load];
            int64_t size = layer_sizes_[layer_to_load];

            // Provide kernel page cache hint
            posix_fadvise(fd_, offset, size, POSIX_FADV_WILLNEED);

            // Read sequentially from file descriptor
            int64_t bytes_read = 0;
            while (bytes_read < size && is_running_) {
                ssize_t n = pread(fd_, prefetch_buf_ + bytes_read, size - bytes_read, offset + bytes_read);
                if (n <= 0) {
                    break;
                }
                bytes_read += n;
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);
            prefetch_ready_ = true;
            cv_compute_.notify_all();
        }
    }
}

torch::Tensor AsyncRingStreamer::acquire_compute_layer(int64_t layer_idx) {
    if (!is_running_) {
        start(layer_idx);
    }

    if (current_compute_layer_ == layer_idx) {
        int64_t sz = layer_sizes_[layer_idx];
        return torch::from_blob(compute_buf_, {sz}, torch::kUInt8);
    }

    std::unique_lock<std::mutex> lock(mtx_);
    if (current_prefetch_layer_ != layer_idx) {
        current_prefetch_layer_ = layer_idx;
        prefetch_ready_ = false;
        prefetch_requested_ = true;
        cv_io_.notify_one();
    }

    cv_compute_.wait(lock, [this, layer_idx]() {
        return !is_running_ || (prefetch_ready_ && current_prefetch_layer_ == layer_idx);
    });

    if (!is_running_) {
        throw std::runtime_error("AsyncRingStreamer stopped during acquire_compute_layer");
    }

    // Double-buffer swap
    std::swap(compute_buf_, prefetch_buf_);
    current_compute_layer_ = layer_idx;
    prefetch_ready_ = false;

    int64_t sz = layer_sizes_[layer_idx];
    return torch::from_blob(compute_buf_, {sz}, torch::kUInt8);
}

void AsyncRingStreamer::release_and_prefetch_next() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (current_compute_layer_ + 1 < num_layers_) {
        current_prefetch_layer_ = current_compute_layer_ + 1;
        prefetch_ready_ = false;
        prefetch_requested_ = true;
        cv_io_.notify_one();
    }
}

} // namespace io
} // namespace rkmj

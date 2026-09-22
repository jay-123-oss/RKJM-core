#pragma once

#include <cstdint>

namespace rkmj {
namespace scheduler {

// Binds worker threads to physical CPU cores to eliminate cross-core cache thrashing.
bool set_thread_affinity(int64_t num_threads = -1);

// Returns active number of physical CPU cores detected.
int64_t get_num_physical_cores();

} // namespace scheduler
} // namespace rkmj

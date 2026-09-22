#include "include/scheduler.h"
#include <omp.h>
#include <iostream>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#endif

namespace rkmj {
namespace scheduler {

int64_t get_num_physical_cores() {
    unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) return 4;
    return static_cast<int64_t>(hardware_threads);
}

bool set_thread_affinity(int64_t num_threads) {
    if (num_threads <= 0) {
        num_threads = get_num_physical_cores();
    }
    omp_set_num_threads(static_cast<int>(num_threads));

#if defined(__linux__)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid % num_threads, &cpuset);

        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
    return true;
#else
    return false;
#endif
}

} // namespace scheduler
} // namespace rkmj

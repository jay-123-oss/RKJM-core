cmake_minimum_required(VERSION 3.18)
project(rkmj_core LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Compiler flags for maximum vectorization performance
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(SIMD_FLAGS -mavx2 -mfma -mbmi2 -mpopcnt)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(SIMD_FLAGS -march=armv8-a+simd)
endif()

add_compile_options(
    -O3
    -Wall
    -fPIC
    ${SIMD_FLAGS}
)

set(SOURCES
    csrc/bindings.cpp
    csrc/allocator.cpp
    csrc/pack.cpp
    csrc/kernels_avx2.cpp
    csrc/scheduler.cpp
    csrc/autograd_ste.cpp
    csrc/cpu/csa_linear_cpu.cpp
    csrc/cpu/csa_autograd_cpu.cpp
    csrc/cuda/bitwise_host.cpp
    csrc/io/mmap_streamer.cpp
)

add_library(rkmj_core SHARED ${SOURCES})

target_include_directories(rkmj_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(rkmj_core PUBLIC
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${Python_LIBRARIES}
)

set_target_properties(rkmj_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
)

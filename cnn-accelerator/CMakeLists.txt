cmake_minimum_required(VERSION 3.16)
project(cnn_accelerator_model CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only core: Tensor, and later CNN ops / accelerator model, all live under include/.
add_library(accel_core INTERFACE)
target_include_directories(accel_core INTERFACE include)
target_compile_options(accel_core INTERFACE -Wall -Wextra)

enable_testing()
add_subdirectory(tests)

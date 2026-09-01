cmake_minimum_required(VERSION 3.16)
project(cnn_accelerator_model CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only core: Tensor and the CNN ops (ops.hpp) are templates/inline
# functions, so they have no .cpp to compile -- this target just carries the
# include path and warning flags to anything that links it.
add_library(accel_core INTERFACE)
target_include_directories(accel_core INTERFACE include)
target_compile_options(accel_core INTERFACE -Wall -Wextra)

# Accelerator is a concrete (non-template) class, so unlike Tensor/ops.hpp its
# implementation lives in a compiled .cpp instead of being header-only.
add_library(accel_lib STATIC src/accelerator.cpp)
target_link_libraries(accel_lib PUBLIC accel_core)

enable_testing()
add_subdirectory(tests)

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

# Accelerator and WeightLoader are concrete (non-template) classes, so unlike
# Tensor/ops.hpp their implementations live in compiled .cpp files instead of
# being header-only.
add_library(accel_lib STATIC src/accelerator.cpp src/weight_loader.cpp)
target_link_libraries(accel_lib PUBLIC accel_core)

# End-to-end inference CLI: loads a trained/quantized model and runs it on
# the exported test images, through both the flat reference ops and the
# Accelerator model.
add_executable(mnist_infer src/mnist_infer.cpp)
target_link_libraries(mnist_infer PRIVATE accel_lib)

enable_testing()
add_subdirectory(tests)

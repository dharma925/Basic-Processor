add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE accel_core)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE accel_core)
add_test(NAME test_ops COMMAND test_ops)

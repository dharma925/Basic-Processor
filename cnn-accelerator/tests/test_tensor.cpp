#include "tensor.hpp"
#include "test_utils.hpp"

using accel::Tensor;

static void test_shape_and_size() {
    Tensor<int8_t> t({2, 3, 4});  // e.g. 2 channels, 3 rows, 4 cols
    CHECK_EQ(t.rank(), size_t{3});
    CHECK_EQ(t.size(), size_t{2 * 3 * 4});
    CHECK_EQ(t.shape()[0], size_t{2});
    CHECK_EQ(t.shape()[1], size_t{3});
    CHECK_EQ(t.shape()[2], size_t{4});
}

static void test_default_initialized_to_zero() {
    Tensor<int32_t> t({4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            CHECK_EQ(t(i, j), 0);
        }
    }
}

static void test_indexing_roundtrip() {
    Tensor<int32_t> t({2, 3});
    int value = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t(i, j) = value++;
        }
    }
    value = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            CHECK_EQ(t(i, j), value++);
        }
    }
}

static void test_row_major_layout() {
    // Row-major: last dimension is contiguous. For shape (2,3), element (0,1)
    // and (0,2) are adjacent in memory; (1,0) is 3 elements after (0,0).
    Tensor<int32_t> t({2, 3});
    t.data()[0] = 10;  // should be t(0,0)
    t.data()[1] = 11;  // should be t(0,1)
    t.data()[3] = 20;  // should be t(1,0)
    CHECK_EQ(t(0, 0), 10);
    CHECK_EQ(t(0, 1), 11);
    CHECK_EQ(t(1, 0), 20);
}

static void test_fill() {
    Tensor<int8_t> t({3, 3});
    t.fill(7);
    for (size_t i = 0; i < 9; ++i) {
        CHECK_EQ(t.data()[i], 7);
    }
}

static void test_out_of_bounds_throws() {
    Tensor<int8_t> t({2, 2});
    CHECK_THROWS(t.at({2, 0}));   // row index out of range
    CHECK_THROWS(t.at({0, 5}));   // col index out of range
    CHECK_THROWS(t.at({0}));      // wrong rank
}

static void test_4d_nchw_shape() {
    // A realistic activation tensor shape: batch=1, channels=3, height=8, width=8.
    Tensor<int8_t> t({1, 3, 8, 8});
    CHECK_EQ(t.size(), size_t{1 * 3 * 8 * 8});
    t(0, 2, 7, 7) = 42;
    CHECK_EQ(t(0, 2, 7, 7), 42);
}

int main() {
    test_shape_and_size();
    test_default_initialized_to_zero();
    test_indexing_roundtrip();
    test_row_major_layout();
    test_fill();
    test_out_of_bounds_throws();
    test_4d_nchw_shape();

    if (g_failures == 0) {
        std::cout << "All tensor tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " test(s) failed.\n";
    return 1;
}

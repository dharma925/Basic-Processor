#include <random>

#include "accelerator.hpp"
#include "ops.hpp"
#include "test_utils.hpp"

using namespace accel;

namespace {

Tensor<int8_t> randomTensor(std::vector<size_t> shape, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(-20, 20);
    Tensor<int8_t> t(shape);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = static_cast<int8_t>(dist(rng));
    }
    return t;
}

bool allEqual(const Tensor<int32_t>& a, const Tensor<int32_t>& b) {
    if (a.shape() != b.shape()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a.data()[i] != b.data()[i]) return false;
    }
    return true;
}

}  // namespace

// The whole point of this test file: the tiled, MAC-array-mapped execution
// path must produce results bit-identical to the flat reference ops. Same
// role as comparing a DUT against a golden model in a verification testbench
// -- different implementation, same expected numbers.

static void test_conv2d_matches_reference_small_array() {
    // Same 3x3 ramp / 2x2 kernel as test_ops.cpp's conv test, but a 2x2 MAC
    // array against a 4-pixel output -- forces the tiling loop to iterate.
    Tensor<int8_t> input({1, 3, 3});
    int8_t vals[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 9; ++i) input.data()[i] = vals[i];
    Tensor<int8_t> weight({1, 1, 2, 2});
    weight.data()[0] = 1;
    weight.data()[1] = 0;
    weight.data()[2] = 0;
    weight.data()[3] = -1;
    Tensor<int32_t> bias({1});
    bias.data()[0] = 10;

    AcceleratorConfig cfg;
    cfg.mac_rows = 2;
    cfg.mac_cols = 2;
    Accelerator acc(cfg);

    Tensor<int32_t> got = acc.conv2d(input, weight, bias);
    Tensor<int32_t> want = conv2d(input, weight, bias);
    CHECK(allEqual(got, want));
}

static void test_conv2d_matches_reference_random_with_padding() {
    std::mt19937 rng(42);
    Tensor<int8_t> input = randomTensor({3, 8, 8}, rng);
    Tensor<int8_t> weight = randomTensor({5, 3, 3, 3}, rng);
    Tensor<int32_t> bias({5});
    for (size_t i = 0; i < 5; ++i) bias.data()[i] = static_cast<int32_t>(i) - 2;

    // Deliberately small/uneven array so tiles rarely divide the workload evenly.
    AcceleratorConfig cfg;
    cfg.mac_rows = 3;
    cfg.mac_cols = 2;
    Accelerator acc(cfg);

    Tensor<int32_t> got = acc.conv2d(input, weight, bias, /*stride=*/1, /*padding=*/1);
    Tensor<int32_t> want = conv2d(input, weight, bias, /*stride=*/1, /*padding=*/1);
    CHECK(allEqual(got, want));
}

static void test_conv2d_matches_reference_stride2_no_bias() {
    std::mt19937 rng(7);
    Tensor<int8_t> input = randomTensor({2, 9, 9}, rng);
    Tensor<int8_t> weight = randomTensor({4, 2, 3, 3}, rng);

    AcceleratorConfig cfg;
    cfg.mac_rows = 4;
    cfg.mac_cols = 4;
    Accelerator acc(cfg);

    Tensor<int32_t> got = acc.conv2d(input, weight, std::nullopt, /*stride=*/2, /*padding=*/0);
    Tensor<int32_t> want = conv2d(input, weight, std::nullopt, /*stride=*/2, /*padding=*/0);
    CHECK(allEqual(got, want));
}

static void test_fully_connected_matches_reference() {
    std::mt19937 rng(99);
    Tensor<int8_t> x = randomTensor({10}, rng);
    Tensor<int8_t> weight = randomTensor({7, 10}, rng);
    Tensor<int32_t> bias({7});
    for (size_t i = 0; i < 7; ++i) bias.data()[i] = static_cast<int32_t>(i) * 3 - 5;

    AcceleratorConfig cfg;
    cfg.mac_rows = 4;  // R is always 1 for FC -- rows just needs to be >= 1
    cfg.mac_cols = 3;  // forces multiple column tiles across 7 outputs
    Accelerator acc(cfg);

    Tensor<int32_t> got = acc.fullyConnected(x, weight, bias);
    Tensor<int32_t> want = fully_connected(x, weight, bias);
    CHECK(allEqual(got, want));
}

static void test_mac_array_rejects_oversized_tile() {
    MacArray arr(2, 2);
    Tensor<int8_t> input_tile({3, 4});  // R=3 > rows=2
    Tensor<int8_t> weight_tile({4, 2});
    CHECK_THROWS(arr.computeTile(input_tile, weight_tile));
}

static void test_local_buffer_capacity_enforced() {
    LocalBuffer<int8_t> buf(4);
    Tensor<int8_t> small({2, 2});  // 4 elements, fits exactly
    Tensor<int8_t> big({3, 2});    // 6 elements, does not fit
    buf.store(small);  // should not throw
    CHECK_THROWS(buf.store(big));
}

int main() {
    test_conv2d_matches_reference_small_array();
    test_conv2d_matches_reference_random_with_padding();
    test_conv2d_matches_reference_stride2_no_bias();
    test_fully_connected_matches_reference();
    test_mac_array_rejects_oversized_tile();
    test_local_buffer_capacity_enforced();

    if (g_failures == 0) {
        std::cout << "All accelerator tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " test(s) failed.\n";
    return 1;
}

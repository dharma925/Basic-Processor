#include <cmath>
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

// Regression test for a real bug found while running mnist_infer on the
// actual trained model: its FC layer has K=1568 (8*14*14 flattened), and a
// weight tile is (K, c_tile) elements -- for a 4x4 array that's up to
// 1568*4=6272 elements, comfortably bigger than the 4096-element default
// buffer used by the small examples elsewhere in this file. LocalBuffer
// correctly throws in that situation; this test pins that behavior down
// deliberately instead of leaving it to be rediscovered by accident.
static void test_fully_connected_throws_when_weight_buffer_too_small() {
    Tensor<int8_t> x({100});
    Tensor<int8_t> weight({8, 100});  // K=100, N=8

    AcceleratorConfig cfg;
    cfg.mac_cols = 4;                  // weight tile = (100, 4) = 400 elements
    cfg.weight_buffer_elems = 50;      // deliberately too small
    Accelerator acc(cfg);
    CHECK_THROWS(acc.fullyConnected(x, weight));
}

static void test_fully_connected_succeeds_when_buffer_sized_for_workload() {
    Tensor<int8_t> x({100});
    Tensor<int8_t> weight({8, 100});

    AcceleratorConfig cfg;
    cfg.mac_cols = 4;
    cfg.weight_buffer_elems = 400;  // exactly large enough
    Accelerator acc(cfg);
    Tensor<int32_t> out = acc.fullyConnected(x, weight);  // should not throw
    CHECK_EQ(out.shape()[0], size_t{8});
}

// --- Performance counters: every number below is hand-computed in the
// comments from the cycle model in docs/performance_model.md, the same way
// the ops.hpp/tensor.hpp tests use hand-computed correctness vectors.

static void test_conv2d_perf_counters() {
    // Same 3x3 ramp / 2x2 kernel as the correctness tests above.
    // input (1,3,3), weight (1,1,2,2) -> output (1,2,2): p_total=4, c_out=1, k=1*2*2=4.
    Tensor<int8_t> input({1, 3, 3});
    int8_t vals[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 9; ++i) input.data()[i] = vals[i];
    Tensor<int8_t> weight({1, 1, 2, 2});
    weight.data()[0] = 1;
    weight.data()[1] = 0;
    weight.data()[2] = 0;
    weight.data()[3] = -1;

    // mac_rows=2, mac_cols=2 -> tiles of (r_tile=2, c_tile=1), two tiles total
    // (p_total=4 splits into two row-tiles of 2; c_out=1 fits in one column-tile).
    // buffer_bandwidth=4 elems/cycle, chosen so ceilDiv isn't trivial.
    AcceleratorConfig cfg;
    cfg.mac_rows = 2;
    cfg.mac_cols = 2;
    cfg.buffer_bandwidth_elems_per_cycle = 4;
    Accelerator acc(cfg);

    acc.conv2d(input, weight);
    const PerfCounters& s = acc.stats();

    // Per tile: input_tile=(r_tile=2,k=4)->8 elems, weight_tile=(k=4,c_tile=1)->4 elems.
    // load_cycles/tile = ceil(8/4) + ceil(4/4) = 2 + 1 = 3; two tiles -> 6.
    CHECK_EQ(s.load_cycles, uint64_t{6});
    // compute_cycles/tile = k = 4; two tiles -> 8.
    CHECK_EQ(s.compute_cycles, uint64_t{8});
    CHECK_EQ(s.totalCycles(), uint64_t{14});
    // mac_ops/tile = r_tile*c_tile*k = 2*1*4 = 8; two tiles -> 16.
    CHECK_EQ(s.mac_ops, uint64_t{16});
    // mac_capacity_ops/tile = mac_rows*mac_cols*k = 2*2*4 = 16; two tiles -> 32.
    CHECK_EQ(s.mac_capacity_ops, uint64_t{32});
    CHECK(std::abs(s.macUtilization() - 0.5) < 1e-12);  // 16/32
    CHECK_EQ(s.activation_bytes_loaded, uint64_t{16});  // 8 elems * 2 tiles
    CHECK_EQ(s.weight_bytes_loaded, uint64_t{8});        // 4 elems * 2 tiles
    CHECK_EQ(s.tile_count, uint64_t{2});
}

static void test_fully_connected_perf_counters() {
    // K=3, N=5, mac_rows=4 (irrelevant to FC's utilization -- R is always 1),
    // mac_cols=2 -> column tiles of size 2,2,1; bandwidth=2 elems/cycle.
    Tensor<int8_t> x({3});
    x.data()[0] = 1;
    x.data()[1] = 2;
    x.data()[2] = 3;
    Tensor<int8_t> weight({5, 3});
    for (size_t i = 0; i < 15; ++i) weight.data()[i] = 1;

    AcceleratorConfig cfg;
    cfg.mac_rows = 4;
    cfg.mac_cols = 2;
    cfg.buffer_bandwidth_elems_per_cycle = 2;
    Accelerator acc(cfg);

    acc.fullyConnected(x, weight);
    const PerfCounters& s = acc.stats();

    // Activation (x) is staged ONCE and reused across all 3 column tiles:
    // load_cycles += ceil(3/2) = 2; activation_bytes_loaded = 3.
    // Weight tiles: c_tile in {2,2,1}, each weight_tile size = k*c_tile = 3*c_tile.
    //   tile1: size=6, load_cycles+=ceil(6/2)=3, mac_ops+=2*3=6, mac_capacity+=4*2*3=24, compute_cycles+=3
    //   tile2: same as tile1
    //   tile3: size=3, load_cycles+=ceil(3/2)=2, mac_ops+=1*3=3, mac_capacity+=24, compute_cycles+=3
    // load_cycles total = 2 + 3 + 3 + 2 = 10
    CHECK_EQ(s.load_cycles, uint64_t{10});
    // compute_cycles = 3+3+3 = 9
    CHECK_EQ(s.compute_cycles, uint64_t{9});
    CHECK_EQ(s.totalCycles(), uint64_t{19});
    // mac_ops = 6+6+3 = 15
    CHECK_EQ(s.mac_ops, uint64_t{15});
    // mac_capacity_ops = 24*3 = 72
    CHECK_EQ(s.mac_capacity_ops, uint64_t{72});
    CHECK(std::abs(s.macUtilization() - (15.0 / 72.0)) < 1e-12);
    CHECK_EQ(s.activation_bytes_loaded, uint64_t{3});
    CHECK_EQ(s.weight_bytes_loaded, uint64_t{15});  // 6+6+3
    CHECK_EQ(s.tile_count, uint64_t{3});
}

static void test_stats_accumulate_and_reset() {
    Tensor<int8_t> x({3});
    x.data()[0] = 1;
    x.data()[1] = 2;
    x.data()[2] = 3;
    Tensor<int8_t> weight({2, 3});
    for (size_t i = 0; i < 6; ++i) weight.data()[i] = 1;

    AcceleratorConfig cfg;
    Accelerator acc(cfg);

    acc.fullyConnected(x, weight);
    const uint64_t single_call_mac_ops = acc.stats().mac_ops;
    CHECK(single_call_mac_ops > 0);

    // A second call without resetStats() should ACCUMULATE, not overwrite.
    acc.fullyConnected(x, weight);
    CHECK_EQ(acc.stats().mac_ops, single_call_mac_ops * 2);

    // resetStats() should bring every counter back to zero.
    acc.resetStats();
    CHECK_EQ(acc.stats().mac_ops, uint64_t{0});
    CHECK_EQ(acc.stats().tile_count, uint64_t{0});
    CHECK_EQ(acc.stats().totalCycles(), uint64_t{0});

    // A fresh call after reset should match the very first call exactly.
    acc.fullyConnected(x, weight);
    CHECK_EQ(acc.stats().mac_ops, single_call_mac_ops);
}

int main() {
    test_conv2d_matches_reference_small_array();
    test_conv2d_matches_reference_random_with_padding();
    test_conv2d_matches_reference_stride2_no_bias();
    test_fully_connected_matches_reference();
    test_mac_array_rejects_oversized_tile();
    test_local_buffer_capacity_enforced();
    test_fully_connected_throws_when_weight_buffer_too_small();
    test_fully_connected_succeeds_when_buffer_sized_for_workload();
    test_conv2d_perf_counters();
    test_fully_connected_perf_counters();
    test_stats_accumulate_and_reset();

    if (g_failures == 0) {
        std::cout << "All accelerator tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " test(s) failed.\n";
    return 1;
}

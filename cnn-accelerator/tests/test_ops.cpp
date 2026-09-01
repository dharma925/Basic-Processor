#include "ops.hpp"
#include "test_utils.hpp"

using namespace accel;

// Every test here uses small values that are hand-computed in the comments —
// these are meant to be readable as verification test vectors, not just
// "trust the code that generated them."

static void test_conv2d_no_padding() {
    // input (1,3,3): a 3x3 ramp
    // 1 2 3
    // 4 5 6
    // 7 8 9
    Tensor<int8_t> input({1, 3, 3});
    int8_t vals[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 9; ++i) input.data()[i] = vals[i];

    // weight (1,1,2,2): [[1,0],[0,-1]]
    Tensor<int8_t> weight({1, 1, 2, 2});
    weight.data()[0] = 1;
    weight.data()[1] = 0;
    weight.data()[2] = 0;
    weight.data()[3] = -1;

    Tensor<int32_t> out = conv2d(input, weight);
    CHECK_EQ(out.shape()[0], size_t{1});
    CHECK_EQ(out.shape()[1], size_t{2});
    CHECK_EQ(out.shape()[2], size_t{2});
    // Each 2x2 window over the ramp gives top-left - bottom-right = -4 everywhere.
    CHECK_EQ(out(0, 0, 0), -4);
    CHECK_EQ(out(0, 0, 1), -4);
    CHECK_EQ(out(0, 1, 0), -4);
    CHECK_EQ(out(0, 1, 1), -4);
}

static void test_conv2d_with_bias() {
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

    Tensor<int32_t> out = conv2d(input, weight, bias);
    CHECK_EQ(out(0, 0, 0), 6);  // -4 + 10
    CHECK_EQ(out(0, 1, 1), 6);
}

static void test_conv2d_with_padding() {
    Tensor<int8_t> input({1, 3, 3});
    int8_t vals[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 9; ++i) input.data()[i] = vals[i];
    Tensor<int8_t> weight({1, 1, 2, 2});
    weight.data()[0] = 1;
    weight.data()[1] = 0;
    weight.data()[2] = 0;
    weight.data()[3] = -1;

    Tensor<int32_t> out = conv2d(input, weight, std::nullopt, /*stride=*/1, /*padding=*/1);
    // padding=1, kernel=2 -> H_out = W_out = 4
    CHECK_EQ(out.shape()[1], size_t{4});
    CHECK_EQ(out.shape()[2], size_t{4});
    // Top-left output: only the (r=1,c=1) tap (weight -1) lands on a real pixel
    // (input(0,0)=1); every other tap falls in the zero-padded border.
    CHECK_EQ(out(0, 0, 0), -1);
}

static void test_matmul() {
    Tensor<int8_t> a({2, 3});
    int8_t avals[6] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; ++i) a.data()[i] = avals[i];

    Tensor<int8_t> b({3, 2});
    int8_t bvals[6] = {1, 0, 0, 1, 1, 1};
    for (int i = 0; i < 6; ++i) b.data()[i] = bvals[i];

    Tensor<int32_t> out = matmul(a, b);
    CHECK_EQ(out(0, 0), 4);
    CHECK_EQ(out(0, 1), 5);
    CHECK_EQ(out(1, 0), 10);
    CHECK_EQ(out(1, 1), 11);
}

static void test_fully_connected() {
    Tensor<int8_t> x({3});
    x.data()[0] = 1;
    x.data()[1] = 2;
    x.data()[2] = 3;

    Tensor<int8_t> weight({2, 3});
    int8_t wvals[6] = {1, 0, 1, 0, 1, 1};
    for (int i = 0; i < 6; ++i) weight.data()[i] = wvals[i];

    Tensor<int32_t> bias({2});
    bias.data()[0] = 10;
    bias.data()[1] = 20;

    Tensor<int32_t> out = fully_connected(x, weight, bias);
    CHECK_EQ(out(0), 14);
    CHECK_EQ(out(1), 25);
}

static void test_relu_inplace() {
    Tensor<int32_t> t({5});
    int32_t vals[5] = {-3, -1, 0, 2, 5};
    for (int i = 0; i < 5; ++i) t.data()[i] = vals[i];
    relu_inplace(t);
    CHECK_EQ(t(0), 0);
    CHECK_EQ(t(1), 0);
    CHECK_EQ(t(2), 0);
    CHECK_EQ(t(3), 2);
    CHECK_EQ(t(4), 5);
}

static void test_maxpool2d() {
    Tensor<int32_t> input({1, 4, 4});
    int32_t vals[16] = {1, 3, 2, 4, 5, 6, 1, 2, 7, 8, 9, 0, 1, 2, 3, 4};
    for (int i = 0; i < 16; ++i) input.data()[i] = vals[i];

    Tensor<int32_t> out = maxpool2d(input, /*kernel=*/2, /*stride=*/2);
    CHECK_EQ(out.shape()[1], size_t{2});
    CHECK_EQ(out.shape()[2], size_t{2});
    CHECK_EQ(out(0, 0, 0), 6);
    CHECK_EQ(out(0, 0, 1), 4);
    CHECK_EQ(out(0, 1, 0), 8);
    CHECK_EQ(out(0, 1, 1), 9);
}

static void test_requantize() {
    Tensor<int32_t> acc({8});
    int32_t vals[8] = {10, -10, 254, 256, -256, -258, 7, -7};
    for (int i = 0; i < 8; ++i) acc.data()[i] = vals[i];

    Tensor<int8_t> out = requantize(acc, /*shift=*/1);
    CHECK_EQ(static_cast<int>(out(0)), 5);     // 10/2
    CHECK_EQ(static_cast<int>(out(1)), -5);    // -10/2
    CHECK_EQ(static_cast<int>(out(2)), 127);   // 254/2 = 127, within range
    CHECK_EQ(static_cast<int>(out(3)), 127);   // 256/2 = 128, saturates to 127
    CHECK_EQ(static_cast<int>(out(4)), -128);  // -256/2 = -128, boundary, in range
    CHECK_EQ(static_cast<int>(out(5)), -128);  // -258/2 = -129, saturates to -128
    CHECK_EQ(static_cast<int>(out(6)), 3);     // 7/2 truncates to 3
    CHECK_EQ(static_cast<int>(out(7)), -3);    // -7/2 truncates to -3
}

static void test_requantize_float_scale() {
    // scale = 0.5, same effective divisor as the shift=1 test above, but via
    // the calibrated-scale overload -- checks round-half-away-from-zero and
    // saturation behavior distinct from the truncating shift/divide overload.
    Tensor<int32_t> acc({6});
    int32_t vals[6] = {10, -10, 5, -5, 300, -300};  // 5/0.5=2.5, -5/0.5=-2.5 -> round-half-away-from-zero
    for (int i = 0; i < 6; ++i) acc.data()[i] = vals[i];

    Tensor<int8_t> out = requantize(acc, 0.5);
    CHECK_EQ(static_cast<int>(out(0)), 5);     // 10*0.5=5
    CHECK_EQ(static_cast<int>(out(1)), -5);    // -10*0.5=-5
    CHECK_EQ(static_cast<int>(out(2)), 3);     // 5*0.5=2.5 -> rounds away from zero to 3
    CHECK_EQ(static_cast<int>(out(3)), -3);    // -5*0.5=-2.5 -> rounds away from zero to -3
    CHECK_EQ(static_cast<int>(out(4)), 127);   // 300*0.5=150 -> saturates to 127
    CHECK_EQ(static_cast<int>(out(5)), -128);  // -300*0.5=-150 -> saturates to -128
}

int main() {
    test_conv2d_no_padding();
    test_conv2d_with_bias();
    test_conv2d_with_padding();
    test_matmul();
    test_fully_connected();
    test_relu_inplace();
    test_maxpool2d();
    test_requantize();
    test_requantize_float_scale();

    if (g_failures == 0) {
        std::cout << "All ops tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " test(s) failed.\n";
    return 1;
}

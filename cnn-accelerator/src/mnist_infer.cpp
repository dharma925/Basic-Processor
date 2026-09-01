// End-to-end INT8 MNIST inference on real trained weights: loads the model
// exported by scripts/export_int8.py and runs conv1 -> relu -> requantize ->
// maxpool -> flatten -> fc on each exported test image.
//
// Runs the *same* pipeline through two different conv implementations --
// the flat reference (accel::conv2d, ops.hpp) and the tiled MAC-array model
// (Accelerator::conv2d) -- and checks they agree on real data, not just the
// synthetic random tensors in test_accelerator.cpp. Writes results/
// cpp_predictions.csv, which scripts/compare_cpp_python.py then diffs
// against the independent numpy reference (quantized_reference.py) -- the
// actual "Python result vs C++ functional model" comparison the project
// spec asks for.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "accelerator.hpp"
#include "ops.hpp"
#include "weight_loader.hpp"

using namespace accel;

namespace {

// Runs the shared forward pass through whichever conv implementation is
// passed in. Templated on the callable type so the exact same pipeline code
// drives both accel::conv2d (a free function) and Accelerator::conv2d (a
// bound member function, passed in via a lambda) without duplicating the
// four lines in between.
template <typename ConvFn>
Tensor<int32_t> forward(ConvFn conv, const Tensor<int8_t>& image, const WeightLoader& weights,
                         float requant_scale_conv1) {
    Tensor<int32_t> acc = conv(image, weights.int8("conv1_weight"), weights.int32("conv1_bias"),
                                /*stride=*/1, /*padding=*/1);
    relu_inplace(acc);
    Tensor<int8_t> activ = requantize(acc, static_cast<double>(requant_scale_conv1));
    Tensor<int8_t> pooled = maxpool2d(activ, /*kernel=*/2, /*stride=*/2);
    Tensor<int8_t> flat = pooled.reshape({pooled.size()});
    return fully_connected(flat, weights.int8("fc_weight"), weights.int32("fc_bias"));
}

size_t argmax(const Tensor<int32_t>& logits) {
    size_t best = 0;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits(i) > logits(best)) best = i;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string weights_path = argc > 1 ? argv[1] : "models/mnist_int8_model.bin";
    const std::string output_path = argc > 2 ? argv[2] : "results/cpp_predictions.csv";

    WeightLoader weights(weights_path);
    const Tensor<int8_t>& test_images = weights.int8("test_images");     // (N,1,28,28)
    const Tensor<int32_t>& test_labels = weights.int32("test_labels");   // (N,)
    const float requant_scale_conv1 = weights.scalarFloat("requant_scale_conv1");

    const size_t n = test_images.shape()[0];
    const size_t image_c = test_images.shape()[1];
    const size_t image_h = test_images.shape()[2];
    const size_t image_w = test_images.shape()[3];
    const size_t image_elems = image_c * image_h * image_w;

    AcceleratorConfig cfg;  // defaults: 4x4 MAC array
    Accelerator accel_model(cfg);

    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    std::ofstream out(output_path);
    out << "image,predicted,true,match";
    for (size_t j = 0; j < 10; ++j) out << ",logit" << j;
    out << "\n";

    size_t correct = 0;
    size_t accelerator_mismatches = 0;

    for (size_t i = 0; i < n; ++i) {
        // test_images has shape (N, C, H, W) and Tensor's storage is row-major,
        // so image i's C*H*W elements are a contiguous slice starting at
        // offset i*image_elems in the flat buffer -- a direct pointer copy.
        Tensor<int8_t> image({image_c, image_h, image_w});
        std::copy(test_images.data() + i * image_elems, test_images.data() + (i + 1) * image_elems,
                  image.data());

        Tensor<int32_t> logits_ref = forward(
            [](const Tensor<int8_t>& in, const Tensor<int8_t>& w, const Tensor<int32_t>& b,
               size_t stride, size_t padding) { return conv2d(in, w, b, stride, padding); },
            image, weights, requant_scale_conv1);

        Tensor<int32_t> logits_acc = forward(
            [&accel_model](const Tensor<int8_t>& in, const Tensor<int8_t>& w, const Tensor<int32_t>& b,
                            size_t stride, size_t padding) {
                return accel_model.conv2d(in, w, b, stride, padding);
            },
            image, weights, requant_scale_conv1);

        bool matches_accel = true;
        for (size_t j = 0; j < logits_ref.size(); ++j) {
            if (logits_ref(j) != logits_acc(j)) {
                matches_accel = false;
                break;
            }
        }
        if (!matches_accel) ++accelerator_mismatches;

        const size_t predicted = argmax(logits_ref);
        const int32_t true_label = test_labels(i);
        const bool correct_pred = (static_cast<int32_t>(predicted) == true_label);
        correct += correct_pred ? 1 : 0;

        out << i << "," << predicted << "," << true_label << "," << (correct_pred ? 1 : 0);
        for (size_t j = 0; j < logits_ref.size(); ++j) out << "," << logits_ref(j);
        out << "\n";
    }
    out.close();

    std::cout << correct << "/" << n << " correct (flat reference)\n";
    std::cout << accelerator_mismatches
              << " image(s) where Accelerator's tiled path disagreed with the flat reference "
                 "(expect 0)\n";
    std::cout << "wrote " << output_path << "\n";
    return 0;
}

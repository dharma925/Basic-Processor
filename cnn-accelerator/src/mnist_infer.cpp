// End-to-end INT8 MNIST inference on real trained weights: loads the model
// exported by scripts/export_int8.py and runs conv1 -> relu -> requantize ->
// maxpool -> flatten -> fc on each exported test image.
//
// Runs the *same* pipeline through two full implementations end to end --
// the flat reference (accel::conv2d + accel::fully_connected, ops.hpp) and
// the tiled MAC-array model (Accelerator::conv2d + Accelerator::fullyConnected)
// -- and checks they agree on real data, not just the synthetic random
// tensors in test_accelerator.cpp. Writes results/
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

// Runs the shared forward pass through whichever conv/fc implementations are
// passed in. Templated on both callable types so the exact same pipeline
// code drives either the flat reference (accel::conv2d + accel::fully_connected)
// or the tiled MAC-array model (Accelerator::conv2d + Accelerator::fullyConnected)
// end to end, without duplicating the four lines in between. Earlier this
// only templated on the conv step and always called the flat fully_connected
// -- which meant Accelerator::fullyConnected was never actually exercised on
// real data. Fixed here so "Accelerator matches the flat reference" is a
// claim about the whole model, not just the conv layer.
template <typename ConvFn, typename FcFn>
Tensor<int32_t> forward(ConvFn conv, FcFn fc, const Tensor<int8_t>& image, const WeightLoader& weights,
                         float requant_scale_conv1) {
    Tensor<int32_t> acc = conv(image, weights.int8("conv1_weight"), weights.int32("conv1_bias"),
                                /*stride=*/1, /*padding=*/1);
    relu_inplace(acc);
    Tensor<int8_t> activ = requantize(acc, static_cast<double>(requant_scale_conv1));
    Tensor<int8_t> pooled = maxpool2d(activ, /*kernel=*/2, /*stride=*/2);
    Tensor<int8_t> flat = pooled.reshape({pooled.size()});
    return fc(flat, weights.int8("fc_weight"), weights.int32("fc_bias"));
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

    AcceleratorConfig cfg;  // 4x4 MAC array (default)
    // The default 4096-element buffer (sized for the small examples in
    // test_accelerator.cpp) is too small for this model's real FC layer: its
    // weight tile is (K=1568, c_tile<=mac_cols) elements, i.e. up to 1568*4 =
    // 6272 for the default 4x4 array -- bigger than 4096. Sized explicitly
    // here to fit that real workload; discovered by actually running this
    // program and hitting LocalBuffer's capacity check, not chosen in advance.
    cfg.weight_buffer_elems = 8192;
    cfg.activation_buffer_elems = 8192;
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
            [](const Tensor<int8_t>& xx, const Tensor<int8_t>& w, const Tensor<int32_t>& b) {
                return fully_connected(xx, w, b);
            },
            image, weights, requant_scale_conv1);

        Tensor<int32_t> logits_acc = forward(
            [&accel_model](const Tensor<int8_t>& in, const Tensor<int8_t>& w, const Tensor<int32_t>& b,
                            size_t stride, size_t padding) {
                return accel_model.conv2d(in, w, b, stride, padding);
            },
            [&accel_model](const Tensor<int8_t>& xx, const Tensor<int8_t>& w, const Tensor<int32_t>& b) {
                return accel_model.fullyConnected(xx, w, b);
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

    // Performance characterization: only the Accelerator path has anything
    // to report here -- accel::conv2d (the flat reference) has no notion of
    // tiling or cycles, which is exactly why Accelerator exists. Cycle/
    // utilization counts depend only on tensor *shapes* and AcceleratorConfig,
    // not on pixel values, so profiling one image characterizes the whole
    // model on this hardware config -- every other image would report the
    // same numbers.
    {
        Tensor<int8_t> image0({image_c, image_h, image_w});
        std::copy(test_images.data(), test_images.data() + image_elems, image0.data());

        accel_model.resetStats();
        Tensor<int32_t> conv_acc = accel_model.conv2d(image0, weights.int8("conv1_weight"),
                                                        weights.int32("conv1_bias"),
                                                        /*stride=*/1, /*padding=*/1);
        PerfCounters conv_stats = accel_model.stats();

        relu_inplace(conv_acc);
        Tensor<int8_t> activ = requantize(conv_acc, static_cast<double>(requant_scale_conv1));
        Tensor<int8_t> pooled = maxpool2d(activ, /*kernel=*/2, /*stride=*/2);
        Tensor<int8_t> flat = pooled.reshape({pooled.size()});

        accel_model.resetStats();
        accel_model.fullyConnected(flat, weights.int8("fc_weight"), weights.int32("fc_bias"));
        PerfCounters fc_stats = accel_model.stats();

        PerfCounters total = conv_stats;
        total += fc_stats;

        auto printLayer = [](const std::string& name, const PerfCounters& s) {
            std::cout << name << ": tiles=" << s.tile_count << " mac_ops=" << s.mac_ops
                      << " mac_capacity_ops=" << s.mac_capacity_ops << " utilization="
                      << (s.macUtilization() * 100.0) << "% compute_cycles=" << s.compute_cycles
                      << " load_cycles=" << s.load_cycles << " total_cycles=" << s.totalCycles()
                      << " activation_bytes=" << s.activation_bytes_loaded
                      << " weight_bytes=" << s.weight_bytes_loaded << "\n";
        };

        std::cout << "\n--- performance (Accelerator, " << cfg.mac_rows << "x" << cfg.mac_cols
                  << " MAC array, " << cfg.buffer_bandwidth_elems_per_cycle
                  << " elems/cycle buffer bandwidth) ---\n";
        printLayer("conv1", conv_stats);
        printLayer("fc   ", fc_stats);
        printLayer("total", total);
    }

    return 0;
}

#include "accelerator.hpp"

#include <algorithm>
#include <array>

namespace accel {

Accelerator::Accelerator(AcceleratorConfig config)
    : config_(config),
      mac_array_(config.mac_rows, config.mac_cols),
      activation_buffer_(config.activation_buffer_elems),
      weight_buffer_(config.weight_buffer_elems) {}

// Maps convolution onto the MAC array via an im2col-style lowering, done
// tile-by-tile rather than by materializing one giant im2col matrix:
//
//   - The K (reduction) dimension is c_in * kh * kw, flattened.
//   - Output pixels (oh, ow) are flattened to a single "row" index p, and
//     tiled in groups of at most mac_rows.
//   - Output channels are tiled in groups of at most mac_cols.
//   - For each (pixel tile, channel tile), an (R, K) activation tile and a
//     (K, C) weight tile are gathered, staged into the local buffers (this
//     is the "load into on-chip SRAM before compute" step), then handed to
//     MacArray::computeTile() as one GEMM tile.
//
// Because this performs the exact same multiply-accumulates as accel::conv2d
// in ops.hpp -- just grouped into tiles instead of one flat loop, and integer
// addition is exactly associative/commutative (no floating-point reordering
// error to worry about) -- the two must agree bit-for-bit. That equivalence
// is the correctness check in tests/test_accelerator.cpp.
Tensor<int32_t> Accelerator::conv2d(const Tensor<int8_t>& input, const Tensor<int8_t>& weight,
                                     const std::optional<Tensor<int32_t>>& bias, size_t stride,
                                     size_t padding) {
    if (input.rank() != 3 || weight.rank() != 4) {
        throw std::invalid_argument("Accelerator::conv2d: input must be rank 3, weight rank 4");
    }
    const size_t c_in = input.shape()[0];
    const size_t h = input.shape()[1];
    const size_t w = input.shape()[2];
    const size_t c_out = weight.shape()[0];
    const size_t kh = weight.shape()[2];
    const size_t kw = weight.shape()[3];
    if (weight.shape()[1] != c_in) {
        throw std::invalid_argument("Accelerator::conv2d: weight C_in does not match input channels");
    }

    const size_t h_out = (h + 2 * padding - kh) / stride + 1;
    const size_t w_out = (w + 2 * padding - kw) / stride + 1;
    const size_t k = c_in * kh * kw;
    const size_t p_total = h_out * w_out;

    Tensor<int32_t> output({c_out, h_out, w_out});

    // Decode a flattened K-index back into (channel, kernel row, kernel col).
    auto decode_k = [&](size_t kidx) {
        const size_t ci = kidx / (kh * kw);
        const size_t rem = kidx % (kh * kw);
        const size_t kr = rem / kw;
        const size_t kc = rem % kw;
        return std::array<size_t, 3>{ci, kr, kc};
    };

    for (size_t ps = 0; ps < p_total; ps += config_.mac_rows) {
        const size_t r_tile = std::min(config_.mac_rows, p_total - ps);
        for (size_t cs = 0; cs < c_out; cs += config_.mac_cols) {
            const size_t c_tile = std::min(config_.mac_cols, c_out - cs);

            Tensor<int8_t> input_tile({r_tile, k});
            for (size_t r = 0; r < r_tile; ++r) {
                const size_t p = ps + r;
                const size_t oh = p / w_out;
                const size_t ow = p % w_out;
                for (size_t kidx = 0; kidx < k; ++kidx) {
                    const auto [ci, kr, kc] = decode_k(kidx);
                    const long ih = static_cast<long>(oh * stride + kr) - static_cast<long>(padding);
                    const long iw = static_cast<long>(ow * stride + kc) - static_cast<long>(padding);
                    int8_t v = 0;
                    if (ih >= 0 && iw >= 0 && ih < static_cast<long>(h) && iw < static_cast<long>(w)) {
                        v = input(ci, static_cast<size_t>(ih), static_cast<size_t>(iw));
                    }
                    input_tile(r, kidx) = v;
                }
            }

            Tensor<int8_t> weight_tile({k, c_tile});
            for (size_t kidx = 0; kidx < k; ++kidx) {
                const auto [ci, kr, kc] = decode_k(kidx);
                for (size_t c = 0; c < c_tile; ++c) {
                    weight_tile(kidx, c) = weight(cs + c, ci, kr, kc);
                }
            }

            activation_buffer_.store(input_tile);
            weight_buffer_.store(weight_tile);
            Tensor<int32_t> partial =
                mac_array_.computeTile(activation_buffer_.contents(), weight_buffer_.contents());

            for (size_t r = 0; r < r_tile; ++r) {
                const size_t p = ps + r;
                const size_t oh = p / w_out;
                const size_t ow = p % w_out;
                for (size_t c = 0; c < c_tile; ++c) {
                    const size_t co = cs + c;
                    int32_t val = partial(r, c);
                    if (bias) {
                        val += bias->at({co});
                    }
                    output(co, oh, ow) = val;
                }
            }
        }
    }
    return output;
}

// Maps a fully-connected layer onto the same MAC array machinery as a GEMM
// with a single "row" (R = 1): x is (1, K), weight is tiled to (K, C) slices
// of the (N, K) weight matrix (transposed on the fly while gathering).
//
// Worth noticing now, ahead of Phase 3: with R fixed at 1, only one row of
// the mac_rows x mac_cols array is ever active for an FC layer executed this
// way -- the other (mac_rows - 1) rows of PEs sit idle every tile. That's a
// concrete, early example of "more MACs doesn't automatically mean more
// throughput" -- utilization depends on whether the *workload shape* can
// actually fill the array, not just on array size.
Tensor<int32_t> Accelerator::fullyConnected(const Tensor<int8_t>& x, const Tensor<int8_t>& weight,
                                             const std::optional<Tensor<int32_t>>& bias) {
    if (x.rank() != 1 || weight.rank() != 2) {
        throw std::invalid_argument("Accelerator::fullyConnected: x must be rank 1, weight rank 2");
    }
    const size_t k = x.shape()[0];
    const size_t n = weight.shape()[0];
    if (weight.shape()[1] != k) {
        throw std::invalid_argument("Accelerator::fullyConnected: weight in_features must match x size");
    }

    Tensor<int32_t> output({n});

    Tensor<int8_t> input_tile({size_t{1}, k});
    for (size_t i = 0; i < k; ++i) {
        input_tile(0, i) = x(i);
    }
    activation_buffer_.store(input_tile);

    for (size_t cs = 0; cs < n; cs += config_.mac_cols) {
        const size_t c_tile = std::min(config_.mac_cols, n - cs);

        Tensor<int8_t> weight_tile({k, c_tile});
        for (size_t i = 0; i < k; ++i) {
            for (size_t c = 0; c < c_tile; ++c) {
                weight_tile(i, c) = weight(cs + c, i);
            }
        }
        weight_buffer_.store(weight_tile);

        Tensor<int32_t> partial =
            mac_array_.computeTile(activation_buffer_.contents(), weight_buffer_.contents());
        for (size_t c = 0; c < c_tile; ++c) {
            int32_t val = partial(0, c);
            if (bias) {
                val += bias->at({cs + c});
            }
            output(cs + c) = val;
        }
    }
    return output;
}

}  // namespace accel

// CNN operations as free functions over Tensor<T>.
//
// These are pure functional references: correctness only, no notion of cycles,
// parallelism, or memory traffic yet (that's the accelerator/performance model,
// built on top of these). Every nested loop here is written the "obvious" CPU
// way on purpose — the loop *structure* (especially conv2d's 6 nested loops) is
// exactly what later gets tiled and mapped onto the MAC array, so it's worth
// reading closely now.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#include "tensor.hpp"

namespace accel {

// ---------------------------------------------------------------------------
// convolution
// ---------------------------------------------------------------------------
// input:  (C_in, H, W)                  — INT8 activations
// weight: (C_out, C_in, KH, KW)          — INT8 weights
// bias:   optional (C_out,)              — INT32, one accumulator seed per output channel
// output: (C_out, H_out, W_out)          — INT32 raw accumulator values (not yet
//                                          requantized back to INT8 — see requantize())
//
// std::optional<Tensor<int32_t>> models an optional hardware feature: a real MAC
// array either has a bias-add stage wired into the accumulator path or it doesn't.
// Passing std::nullopt is like tying that stage's enable signal low — the code
// that would use it is simply skipped, with the compiler forcing us to check
// bias.has_value() (or use bias->...) before touching it, the same discipline a
// "valid" bit demands before you read the data behind it.
inline Tensor<int32_t> conv2d(const Tensor<int8_t>& input,
                               const Tensor<int8_t>& weight,
                               const std::optional<Tensor<int32_t>>& bias = std::nullopt,
                               size_t stride = 1,
                               size_t padding = 0) {
    if (input.rank() != 3 || weight.rank() != 4) {
        throw std::invalid_argument("conv2d: input must be rank 3 (C,H,W), weight rank 4 (Cout,Cin,KH,KW)");
    }
    const size_t c_in = input.shape()[0];
    const size_t h = input.shape()[1];
    const size_t w = input.shape()[2];
    const size_t c_out = weight.shape()[0];
    const size_t kh = weight.shape()[2];
    const size_t kw = weight.shape()[3];

    if (weight.shape()[1] != c_in) {
        throw std::invalid_argument("conv2d: weight C_in does not match input channels");
    }
    if (bias && bias->shape() != std::vector<size_t>{c_out}) {
        throw std::invalid_argument("conv2d: bias shape must be (C_out,)");
    }

    const size_t h_out = (h + 2 * padding - kh) / stride + 1;
    const size_t w_out = (w + 2 * padding - kw) / stride + 1;

    Tensor<int32_t> output({c_out, h_out, w_out});

    for (size_t co = 0; co < c_out; ++co) {
        const int32_t bias_val = bias ? bias->at({co}) : 0;
        for (size_t oh = 0; oh < h_out; ++oh) {
            for (size_t ow = 0; ow < w_out; ++ow) {
                int32_t acc = bias_val;
                // This is the MAC-heavy inner loop: c_in * kh * kw multiply-accumulates
                // per output element. In the accelerator model, this is the work that
                // gets spread across the MAC array instead of executed one at a time.
                for (size_t ci = 0; ci < c_in; ++ci) {
                    for (size_t r = 0; r < kh; ++r) {
                        for (size_t c = 0; c < kw; ++c) {
                            // signed index math to support padding: a padded position
                            // that falls outside the real input is treated as zero,
                            // same as a zero-padded activation buffer in hardware.
                            const long ih = static_cast<long>(oh * stride + r) - static_cast<long>(padding);
                            const long iw = static_cast<long>(ow * stride + c) - static_cast<long>(padding);
                            if (ih < 0 || iw < 0 || ih >= static_cast<long>(h) || iw >= static_cast<long>(w)) {
                                continue;  // zero-padding: contributes 0, so skip
                            }
                            const int32_t a = input(ci, static_cast<size_t>(ih), static_cast<size_t>(iw));
                            const int32_t wv = weight(co, ci, r, c);
                            acc += a * wv;
                        }
                    }
                }
                output(co, oh, ow) = acc;
            }
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// matmul: general (M,K) x (K,N) -> (M,N), INT8 operands, INT32 accumulation.
// This is the same primitive a systolic MAC array is fundamentally built to
// execute; a fully-connected layer (below) is a special case of it.
// ---------------------------------------------------------------------------
inline Tensor<int32_t> matmul(const Tensor<int8_t>& a, const Tensor<int8_t>& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        throw std::invalid_argument("matmul: both operands must be rank 2");
    }
    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];
    if (b.shape()[0] != k) {
        throw std::invalid_argument("matmul: inner dimensions must match");
    }
    const size_t n = b.shape()[1];

    Tensor<int32_t> out({m, n});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (size_t p = 0; p < k; ++p) {
                acc += static_cast<int32_t>(a(i, p)) * static_cast<int32_t>(b(p, j));
            }
            out(i, j) = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// fully connected: x (K,) times weight (N,K) [PyTorch's nn.Linear layout:
// out_features x in_features] plus optional bias (N,) -> output (N,).
// Written directly (not via matmul) because this is the exact shape real
// exported FC weights arrive in, and because a matvec is the concrete unit of
// work later mapped one row at a time onto the MAC array.
// ---------------------------------------------------------------------------
inline Tensor<int32_t> fully_connected(const Tensor<int8_t>& x,
                                        const Tensor<int8_t>& weight,
                                        const std::optional<Tensor<int32_t>>& bias = std::nullopt) {
    if (x.rank() != 1 || weight.rank() != 2) {
        throw std::invalid_argument("fully_connected: x must be rank 1, weight rank 2 (N,K)");
    }
    const size_t k = x.shape()[0];
    const size_t n = weight.shape()[0];
    if (weight.shape()[1] != k) {
        throw std::invalid_argument("fully_connected: weight in_features must match x size");
    }
    if (bias && bias->shape() != std::vector<size_t>{n}) {
        throw std::invalid_argument("fully_connected: bias shape must be (N,)");
    }

    Tensor<int32_t> out({n});
    for (size_t j = 0; j < n; ++j) {
        int32_t acc = bias ? bias->at({j}) : 0;
        for (size_t i = 0; i < k; ++i) {
            acc += static_cast<int32_t>(x(i)) * static_cast<int32_t>(weight(j, i));
        }
        out(j) = acc;
    }
    return out;
}

// ---------------------------------------------------------------------------
// ReLU: elementwise max(0, x), in place. Templated so it works whether it's
// applied to raw INT32 accumulator output or already-requantized INT8 data —
// it's the same activation-unit logic either way, just a different datapath width.
// ---------------------------------------------------------------------------
template <typename T>
void relu_inplace(Tensor<T>& t) {
    T* d = t.data();
    for (size_t i = 0; i < t.size(); ++i) {
        d[i] = std::max(d[i], T{0});
    }
}

// ---------------------------------------------------------------------------
// max pooling: input (C, H, W) -> output (C, H_out, W_out), no padding.
// Templated on T for the same reason as relu_inplace above.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> maxpool2d(const Tensor<T>& input, size_t kernel, size_t stride) {
    if (input.rank() != 3) {
        throw std::invalid_argument("maxpool2d: input must be rank 3 (C,H,W)");
    }
    const size_t c = input.shape()[0];
    const size_t h = input.shape()[1];
    const size_t w = input.shape()[2];
    const size_t h_out = (h - kernel) / stride + 1;
    const size_t w_out = (w - kernel) / stride + 1;

    Tensor<T> out({c, h_out, w_out});
    for (size_t ch = 0; ch < c; ++ch) {
        for (size_t oh = 0; oh < h_out; ++oh) {
            for (size_t ow = 0; ow < w_out; ++ow) {
                T best = std::numeric_limits<T>::lowest();
                for (size_t r = 0; r < kernel; ++r) {
                    for (size_t col = 0; col < kernel; ++col) {
                        best = std::max(best, input(ch, oh * stride + r, ow * stride + col));
                    }
                }
                out(ch, oh, ow) = best;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// requantize: INT32 accumulator -> INT8, via a power-of-two right shift
// (an approximation of the per-layer scale factor a real quantized model
// applies) followed by saturation to the INT8 range [-128, 127].
//
// Hardware framing: a real accelerator can't keep accumulating in wide INT32
// forever between layers — it writes results back to the (narrow) activation
// buffer at reduced precision so the *next* layer's MAC array can consume them
// as INT8 again. The shift+clamp here is that narrowing step. Saturation
// (clamping instead of wrapping/overflowing) mirrors how fixed-point DSP/MAC
// hardware is normally built, rather than silently wrapping on overflow.
//
// Caveat (documented, not hidden): we implement the shift via integer division
// by a power of two rather than a raw `>>`, because right-shift of a negative
// signed integer is only guaranteed to be arithmetic (floor) as of C++20 — on
// C++17 it's implementation-defined, even though every mainstream compiler
// does the arithmetic-shift thing anyway. Division truncates toward zero
// instead of flooring, which differs from a true hardware arithmetic shifter
// by at most 1 LSB for negative values. That's a known, small deviation from
// real HW behavior, not something this model claims to get bit-exact.
inline Tensor<int8_t> requantize(const Tensor<int32_t>& acc, int shift) {
    Tensor<int8_t> out(acc.shape());
    const int32_t divisor = int32_t{1} << shift;
    int8_t* dst = out.data();
    const int32_t* src = acc.data();
    for (size_t i = 0; i < acc.size(); ++i) {
        int32_t v = src[i] / divisor;
        v = std::clamp(v, static_cast<int32_t>(std::numeric_limits<int8_t>::min()),
                        static_cast<int32_t>(std::numeric_limits<int8_t>::max()));
        dst[i] = static_cast<int8_t>(v);
    }
    return out;
}

// ---------------------------------------------------------------------------
// requantize (calibrated-scale overload): INT32 -> INT8 using a single
// per-layer float scale computed offline (see scripts/export_int8.py's
// requant_scale_conv1), rather than the power-of-two shift above. This is
// what a real post-training-quantization pipeline actually produces, and
// it's the overload used to run this model on real trained weights.
//
// Rounding is round-half-away-from-zero (std::lround's documented behavior),
// chosen deliberately to match scripts/quantized_reference.py's requantize()
// bit-for-bit -- that script avoids numpy's default round-half-to-even for
// exactly this reason. Two independently-written implementations agreeing
// exactly, not just approximately, is the whole point of the Python-vs-C++
// comparison this overload exists for.
//
// Documented simplification: real INT8 accelerator hardware implements a
// per-tensor scale as an integer fixed-point multiply + shift, specifically
// to avoid floating point in the datapath. This overload does a plain
// double-precision multiply instead -- numerically fine for a functional
// model, but not a model of that specific multiplier circuit.
inline Tensor<int8_t> requantize(const Tensor<int32_t>& acc, double scale) {
    Tensor<int8_t> out(acc.shape());
    const int32_t* src = acc.data();
    int8_t* dst = out.data();
    for (size_t i = 0; i < acc.size(); ++i) {
        long v = std::lround(static_cast<double>(src[i]) * scale);
        v = std::clamp<long>(v, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
        dst[i] = static_cast<int8_t>(v);
    }
    return out;
}

}  // namespace accel

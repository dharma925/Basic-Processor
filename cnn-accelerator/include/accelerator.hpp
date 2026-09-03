// The accelerator abstraction: MAC units -> MAC array -> local buffer/SRAM
// -> a tiling controller (Accelerator) that maps conv2d/fully_connected onto
// them, instrumented with PerfCounters (cycles, MAC utilization, memory
// traffic) -- see docs/performance_model.md for the exact cycle model and,
// importantly, what it does and doesn't claim to model accurately. This is
// explicitly a functional/performance model, not RTL-cycle-accurate.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

namespace accel {

// One processing element (PE): a multiply-accumulate unit plus the single
// accumulator register it owns. This is the atomic building block the MAC
// array below instantiates rows*cols times -- the software analogue of a
// `generate for` loop instantiating ROWS*COLS copies of a `mac_pe` module in
// SystemVerilog.
class MacUnit {
public:
    void reset() { acc_ = 0; }

    // One multiply-accumulate: acc += a * b. This single call is the
    // fundamental unit of work whose *count*, in Phase 2, becomes the basis
    // for cycle estimation (N MACs at some issue rate -> N/rate cycles).
    void mac(int8_t a, int8_t b) {
        acc_ += static_cast<int32_t>(a) * static_cast<int32_t>(b);
    }

    int32_t value() const { return acc_; }

private:
    int32_t acc_ = 0;
};

// A configurable rows x cols grid of MacUnits: the accelerator's spatial
// compute parallelism. PE(r,c) computes one output element as a K-deep dot
// product: out(r,c) = sum_k input_tile(r,k) * weight_tile(k,c). That's the
// GEMM view both convolution (via an im2col-style mapping) and a
// fully-connected layer get lowered to -- Accelerator, below, does that
// lowering and calls computeTile() once per output tile.
//
// What this deliberately does NOT model (a documented limitation, not a
// hidden one): true systolic dataflow, where operands or partial sums are
// forwarded PE-to-PE across cycles. Here every PE independently walks its
// own K-deep reduction to completion inside one computeTile() call. Per-
// cycle dataflow and PE utilization over time are Phase 2/3 concerns; Phase 1
// only needs the tile *result* to be numerically correct.
class MacArray {
public:
    MacArray(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        if (rows_ == 0 || cols_ == 0) {
            throw std::invalid_argument("MacArray: rows and cols must be > 0");
        }
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // input_tile: (R, K), weight_tile: (K, C), with R <= rows() and C <= cols().
    // Returns the (R, C) INT32 partial-sum tile.
    Tensor<int32_t> computeTile(const Tensor<int8_t>& input_tile,
                                 const Tensor<int8_t>& weight_tile) const {
        if (input_tile.rank() != 2 || weight_tile.rank() != 2) {
            throw std::invalid_argument("MacArray::computeTile: tiles must be rank 2");
        }
        const size_t r = input_tile.shape()[0];
        const size_t k = input_tile.shape()[1];
        const size_t c = weight_tile.shape()[1];
        if (weight_tile.shape()[0] != k) {
            throw std::invalid_argument("MacArray::computeTile: K dimension mismatch");
        }
        if (r > rows_ || c > cols_) {
            throw std::invalid_argument("MacArray::computeTile: tile exceeds array dimensions");
        }

        std::vector<MacUnit> pes(r * c);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                MacUnit& pe = pes[i * c + j];
                pe.reset();
                for (size_t p = 0; p < k; ++p) {
                    pe.mac(input_tile(i, p), weight_tile(p, j));
                }
            }
        }

        Tensor<int32_t> out({r, c});
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                out(i, j) = pes[i * c + j].value();
            }
        }
        return out;
    }

private:
    size_t rows_;
    size_t cols_;
};

// Local on-chip buffer / SRAM abstraction: bounded storage that a tile must
// be staged into before the MAC array can consume it. Capacity is in
// elements of type T (e.g. an INT8 activation buffer sized like a real SRAM
// macro would be, in bytes-equivalent). Templated so the same class models
// both the activation buffer (T=int8_t) and the weight buffer (T=int8_t
// here too, but kept as a separate instance/instantiation since in real
// hardware they're physically separate SRAMs with independent capacity).
template <typename T>
class LocalBuffer {
public:
    explicit LocalBuffer(size_t capacity_elems) : capacity_(capacity_elems) {}

    size_t capacity() const { return capacity_; }

    // Stage a tile into the buffer. Throws if it doesn't fit: a real
    // accelerator's tiling controller must pick tile sizes that respect this
    // limit, and this is where that constraint is actually enforced here.
    void store(const Tensor<T>& tile) {
        if (tile.size() > capacity_) {
            throw std::runtime_error("LocalBuffer: tile does not fit in buffer capacity");
        }
        contents_ = tile;
    }

    const Tensor<T>& contents() const { return contents_; }

private:
    size_t capacity_;
    Tensor<T> contents_{std::vector<size_t>{0}};
};

// Configuration for an Accelerator instance: MAC array shape, buffer sizes,
// and the one timing parameter this model has -- how fast a tile can be
// staged into the local buffers. A plain struct (aggregate), not a class --
// there's no behavior to encapsulate, just a parameter record, the software
// equivalent of a hardware module's `parameter`/`localparam` block at
// instantiation time.
struct AcceleratorConfig {
    size_t mac_rows = 4;
    size_t mac_cols = 4;
    size_t activation_buffer_elems = 4096;
    size_t weight_buffer_elems = 4096;

    // Elements-per-cycle the local buffers can be loaded at. Since every
    // buffer in this model stores int8_t, "elements" and "bytes" coincide
    // here -- a wider element type would need this split into separate
    // elements/bytes handling, which this model doesn't do. This number is
    // an illustrative, configurable knob, not derived from a real SRAM/bus
    // datasheet -- see docs/performance_model.md.
    size_t buffer_bandwidth_elems_per_cycle = 16;
};

// Performance counters accumulated by an Accelerator as it runs. The same
// idea as a hardware performance-monitoring unit (PMU): counters that
// accumulate during execution and that a caller reads and optionally resets
// between measurement windows (e.g. once per layer) rather than a single
// end-of-run summary. See docs/performance_model.md for exactly how each
// field is computed and what it does/doesn't account for.
struct PerfCounters {
    uint64_t mac_ops = 0;           // utilized multiply-accumulates actually performed
    uint64_t mac_capacity_ops = 0;  // rows*cols*compute_cycles: the array's ideal max over the same cycles
    uint64_t compute_cycles = 0;    // cycles spent doing MAC-array compute (sum of K per tile)
    uint64_t load_cycles = 0;       // cycles spent staging tiles into the local buffers
    uint64_t tile_count = 0;        // number of MAC-array tiles processed
    uint64_t activation_bytes_loaded = 0;  // bytes written into the activation buffer
    uint64_t weight_bytes_loaded = 0;      // bytes written into the weight buffer

    uint64_t totalCycles() const { return compute_cycles + load_cycles; }

    // Fraction of the array's PEs doing useful work, averaged over compute
    // cycles (not blended with load-cycle idle time -- see performance_model.md
    // for why that split is drawn where it is).
    double macUtilization() const {
        return mac_capacity_ops == 0 ? 0.0
                                      : static_cast<double>(mac_ops) / static_cast<double>(mac_capacity_ops);
    }

    void reset() { *this = PerfCounters{}; }

    PerfCounters& operator+=(const PerfCounters& other) {
        mac_ops += other.mac_ops;
        mac_capacity_ops += other.mac_capacity_ops;
        compute_cycles += other.compute_cycles;
        load_cycles += other.load_cycles;
        tile_count += other.tile_count;
        activation_bytes_loaded += other.activation_bytes_loaded;
        weight_bytes_loaded += other.weight_bytes_loaded;
        return *this;
    }
};

// Ties MacArray + LocalBuffers together into the "tiled execution" stage of
// the target pipeline: it takes a full conv2d/fully_connected workload,
// partitions it into tiles that fit the configured MAC array, stages each
// tile through the local buffers, and assembles the tile results back into a
// full output tensor.
//
// Correctness contract: Accelerator::conv2d must produce results identical
// to accel::conv2d (ops.hpp) for the same inputs -- same idea as checking a
// DUT against a golden reference model in a verification testbench. Verified
// in tests/test_accelerator.cpp.
//
// Note this class is declared here but *defined* in src/accelerator.cpp --
// unlike Tensor<T> or MacArray, Accelerator is not a template, so the
// compiler doesn't need its full body at every call site. This is the
// interface/implementation split modern C++ allows for non-generic code,
// loosely analogous to separating a module's port list from its internal
// behavioral logic.
class Accelerator {
public:
    explicit Accelerator(AcceleratorConfig config);

    const AcceleratorConfig& config() const { return config_; }

    Tensor<int32_t> conv2d(const Tensor<int8_t>& input, const Tensor<int8_t>& weight,
                            const std::optional<Tensor<int32_t>>& bias = std::nullopt,
                            size_t stride = 1, size_t padding = 0);

    Tensor<int32_t> fullyConnected(const Tensor<int8_t>& x, const Tensor<int8_t>& weight,
                                    const std::optional<Tensor<int32_t>>& bias = std::nullopt);

    // Performance counters accumulated across every conv2d/fullyConnected
    // call since construction or the last resetStats(). Call resetStats()
    // before a layer to get that layer's stats in isolation (layer-by-layer
    // profiling), or leave it running to get a whole-run total.
    const PerfCounters& stats() const { return stats_; }
    void resetStats() { stats_.reset(); }

private:
    AcceleratorConfig config_;
    MacArray mac_array_;
    LocalBuffer<int8_t> activation_buffer_;
    LocalBuffer<int8_t> weight_buffer_;
    PerfCounters stats_;
};

}  // namespace accel

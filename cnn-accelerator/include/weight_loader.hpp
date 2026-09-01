// Loads the binary tensor container written by scripts/model_io.py --
// the trained, quantized weights (and a few scalars/test tensors) the C++
// model runs on. Same relationship as a DV testbench loading a .mem/.hex
// file into a simulated memory: a small, documented raw layout, no
// third-party serialization library on either side.
//
// File layout (little-endian; see scripts/model_io.py for the authoritative
// description and the Python-side writer):
//   magic:        4 bytes, ASCII "CACM"
//   version:      uint32
//   num_tensors:  uint32
//   per tensor:   name_len(uint32) + name (name_len bytes) + dtype(uint8)
//                 + rank(uint32) + dims(rank * uint64) + raw data
//
// Documented limitation: this reader assumes a little-endian host (true for
// the x86_64 platform this project targets) and does no byte-swapping. A
// genuinely portable loader would need to -- that's a real gap, not hidden.
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>

#include "tensor.hpp"

namespace accel {

// Mirrors scripts/model_io.py's dtype tag values exactly -- this enum *is*
// the file format's dtype field, not just a local convenience. Compare with
// a hardware "opcode"/"type" field on a tagged bus: the numeric value has an
// externally-fixed meaning, not one this program is free to renumber.
enum class DType : uint8_t {
    Int8 = 0,
    Int32 = 1,
    Float32 = 2,
};

// A loaded tensor can be one of three concrete types; which one depends on
// data read from the file at runtime, not on anything known at compile time.
// std::variant is a type-safe tagged union for exactly this situation: it
// can only ever hold one of these three alternatives, and holds a tag saying
// which -- so retrieving the wrong type (e.g. asking for Tensor<int32_t>
// when the stored tensor is Tensor<int8_t>) throws std::bad_variant_access
// instead of silently reinterpreting bytes, the way a raw C union would.
using TensorVariant = std::variant<Tensor<int8_t>, Tensor<int32_t>, Tensor<float>>;

class WeightLoader {
public:
    explicit WeightLoader(const std::string& path);

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }

    const Tensor<int8_t>& int8(const std::string& name) const;
    const Tensor<int32_t>& int32(const std::string& name) const;

    // Convenience for a rank-1, size-1 float32 tensor used as a scalar
    // (e.g. "requant_scale_conv1").
    float scalarFloat(const std::string& name) const;

private:
    std::map<std::string, TensorVariant> tensors_;
};

}  // namespace accel

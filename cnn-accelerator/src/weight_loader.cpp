#include "weight_loader.hpp"

#include <array>
#include <cstdint>
#include <fstream>

namespace accel {

namespace {

constexpr std::array<char, 4> kMagic = {'C', 'A', 'C', 'M'};
constexpr uint32_t kVersion = 1;

// Small helpers so the parsing logic below reads as "read a uint32, read a
// name, read a tensor" instead of a wall of raw .read() calls. Each throws
// on a short/failed read rather than leaving the caller to notice a
// half-populated Tensor -- the file is either well-formed or this loader
// refuses it outright.
uint32_t readU32(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f) throw std::runtime_error("WeightLoader: unexpected end of file reading uint32");
    return v;
}

uint64_t readU64(std::ifstream& f) {
    uint64_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f) throw std::runtime_error("WeightLoader: unexpected end of file reading uint64");
    return v;
}

uint8_t readU8(std::ifstream& f) {
    uint8_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f) throw std::runtime_error("WeightLoader: unexpected end of file reading uint8");
    return v;
}

std::string readName(std::ifstream& f) {
    const uint32_t len = readU32(f);
    std::string name(len, '\0');
    f.read(name.data(), static_cast<std::streamsize>(len));
    if (!f) throw std::runtime_error("WeightLoader: unexpected end of file reading tensor name");
    return name;
}

std::vector<size_t> readDims(std::ifstream& f, uint32_t rank) {
    std::vector<size_t> dims(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        dims[i] = static_cast<size_t>(readU64(f));
    }
    return dims;
}

// Bulk-reads a tensor's raw payload directly into its contiguous storage via
// data() -- no per-element parsing needed, since the file's byte layout for
// a dtype already matches this (little-endian x86_64) host's in-memory
// layout for that same type. See the "documented limitation" note in
// weight_loader.hpp about why that assumption is safe here but not portable.
template <typename T>
Tensor<T> readTensorData(std::ifstream& f, std::vector<size_t> dims) {
    Tensor<T> t(std::move(dims));
    f.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(T)));
    if (!f) throw std::runtime_error("WeightLoader: unexpected end of file reading tensor data");
    return t;
}

}  // namespace

WeightLoader::WeightLoader(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("WeightLoader: could not open '" + path + "'");
    }

    std::array<char, 4> magic{};
    f.read(magic.data(), magic.size());
    if (!f || magic != kMagic) {
        throw std::runtime_error("WeightLoader: '" + path + "' is not a valid CACM tensor container");
    }
    const uint32_t version = readU32(f);
    if (version != kVersion) {
        throw std::runtime_error("WeightLoader: unsupported container version " + std::to_string(version));
    }

    const uint32_t num_tensors = readU32(f);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        std::string name = readName(f);
        const auto dtype = static_cast<DType>(readU8(f));
        const uint32_t rank = readU32(f);
        std::vector<size_t> dims = readDims(f, rank);

        switch (dtype) {
            case DType::Int8:
                tensors_.emplace(std::move(name), readTensorData<int8_t>(f, std::move(dims)));
                break;
            case DType::Int32:
                tensors_.emplace(std::move(name), readTensorData<int32_t>(f, std::move(dims)));
                break;
            case DType::Float32:
                tensors_.emplace(std::move(name), readTensorData<float>(f, std::move(dims)));
                break;
            default:
                throw std::runtime_error("WeightLoader: unknown dtype tag in '" + path + "'");
        }
    }
}

const Tensor<int8_t>& WeightLoader::int8(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("WeightLoader: no tensor named '" + name + "'");
    }
    try {
        return std::get<Tensor<int8_t>>(it->second);
    } catch (const std::bad_variant_access&) {
        throw std::runtime_error("WeightLoader: tensor '" + name + "' is not int8");
    }
}

const Tensor<int32_t>& WeightLoader::int32(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("WeightLoader: no tensor named '" + name + "'");
    }
    try {
        return std::get<Tensor<int32_t>>(it->second);
    } catch (const std::bad_variant_access&) {
        throw std::runtime_error("WeightLoader: tensor '" + name + "' is not int32");
    }
}

float WeightLoader::scalarFloat(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("WeightLoader: no tensor named '" + name + "'");
    }
    const Tensor<float>* t = nullptr;
    try {
        t = &std::get<Tensor<float>>(it->second);
    } catch (const std::bad_variant_access&) {
        throw std::runtime_error("WeightLoader: tensor '" + name + "' is not float32");
    }
    if (t->size() != 1) {
        throw std::runtime_error("WeightLoader: tensor '" + name + "' is not a scalar (size 1)");
    }
    return t->data()[0];
}

}  // namespace accel

// Tensor: the storage/indexing abstraction everything else in this project builds on.
//
// Hardware framing: a Tensor models the *contents and layout* of data sitting in an
// on-chip buffer/SRAM (or, for the input image, off-chip DRAM before it's staged on-chip).
// shape_ + strides_ is a software stand-in for an address-generation unit: given
// logical indices (n, c, h, w), it computes a single flat offset the same way an AGU
// in RTL turns loop-nest indices into a memory address.
#pragma once

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace accel {

// Row-major (C-order) dense tensor. Template parameter T is the element type —
// e.g. Tensor<int8_t> for INT8 activations/weights, Tensor<int32_t> for accumulators
// that need extra headroom to avoid overflow while summing many INT8 products.
template <typename T>
class Tensor {
public:
    explicit Tensor(std::vector<size_t> shape)
        : shape_(std::move(shape)),
          strides_(computeStrides(shape_)),
          data_(numElements(shape_), T{}) {}

    // Rank: number of dimensions (e.g. 4 for NCHW).
    size_t rank() const { return shape_.size(); }

    // Total element count across all dimensions.
    size_t size() const { return data_.size(); }

    const std::vector<size_t>& shape() const { return shape_; }

    // General indexing: works for any rank. Bounds-checked (throws on misuse) —
    // in RTL this would be the equivalent of an address-range assertion.
    T& at(const std::vector<size_t>& idx) { return data_[flatten(idx)]; }
    const T& at(const std::vector<size_t>& idx) const { return data_[flatten(idx)]; }

    // Convenience call-operator: t(n, c, h, w) instead of t.at({n, c, h, w}).
    // Idx... is a variadic template (parameter pack) — it accepts any number of
    // index arguments and forwards them as a vector to at().
    template <typename... Idx>
    T& operator()(Idx... idx) {
        return at(std::vector<size_t>{static_cast<size_t>(idx)...});
    }
    template <typename... Idx>
    const T& operator()(Idx... idx) const {
        return at(std::vector<size_t>{static_cast<size_t>(idx)...});
    }

    void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

private:
    static size_t numElements(const std::vector<size_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
    }

    // strides[i] = number of elements to skip to advance index i by 1.
    // Standard row-major layout, same convention NumPy/PyTorch use by default.
    static std::vector<size_t> computeStrides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i-- > 1;) {
            strides[i - 1] = strides[i] * shape[i];
        }
        return strides;
    }

    size_t flatten(const std::vector<size_t>& idx) const {
        if (idx.size() != shape_.size()) {
            throw std::out_of_range("Tensor::at: index rank does not match tensor rank");
        }
        size_t offset = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Tensor::at: index out of bounds");
            }
            offset += idx[i] * strides_[i];
        }
        return offset;
    }

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<T> data_;
};

}  // namespace accel

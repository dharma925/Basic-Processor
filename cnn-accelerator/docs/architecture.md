# Architecture Notes

Living document — updated as each component lands. Right now this covers only the
Tensor abstraction (the first piece); CNN ops, the accelerator model, and the
performance model will each get their own section as they're implemented.

## Tensor (`include/tensor.hpp`)

`Tensor<T>` is a dense, row-major, N-dimensional array: a shape (`std::vector<size_t>`),
computed strides, and a flat `std::vector<T>` backing store.

Hardware framing: a `Tensor` is a software stand-in for *what's sitting in an on-chip
buffer or off-chip DRAM* — a block of memory plus a way to compute an address from
logical indices. `shape_`/`strides_` + `flatten()` play the role of an address
generation unit (AGU): given loop-nest indices like `(n, c, h, w)`, produce one flat
memory offset, the same job an AGU does in RTL when it turns nested-loop counters
into a memory address.

Design choices and why:

- **Template on element type** (`Tensor<int8_t>`, `Tensor<int32_t>`): activations and
  weights are INT8 (matches the target accelerator's fixed-point representation), but
  accumulator results need more headroom — summing many INT8xINT8 products can exceed
  an 8-bit range — so accumulator tensors use `int32_t`. One `Tensor` implementation,
  two element types, chosen at compile time. This mirrors the DUT-side truth: an
  accelerator's activation SRAM is typically narrower than its accumulator registers.
- **Row-major layout**: last dimension is contiguous in memory (matches NumPy/PyTorch
  default), so exported weights need no re-layout to be read into a `Tensor`.
- **Bounds-checked indexing** (`at()` throws `std::out_of_range`): during model
  development, an out-of-bounds access is a bug in the model, not a legal hardware
  event, so it should fail loudly and immediately — analogous to an address-range
  assertion firing in simulation rather than corrupting silently.
- **Header-only**: `Tensor` is a class template, and templates must be fully visible
  at every point they're instantiated (the compiler generates code per concrete type
  at compile time, unlike a parameterized SystemVerilog module which is elaborated
  once per parameter set at elaboration time but from a single compiled description).
  That's why there's no `tensor.cpp` — the definition has to live in the header.

## Next

CNN operations (convolution, matmul, ReLU, pooling, fully connected) will be added as
free functions over `Tensor<T>` in `include/`, each documented with its own hardware
framing (e.g. convolution's nested loop structure directly foreshadows the tiling and
MAC-array mapping done in the accelerator model).

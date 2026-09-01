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

## CNN operations (`include/ops.hpp`)

Free functions over `Tensor<T>`: `conv2d`, `matmul`, `fully_connected`, `relu_inplace`,
`maxpool2d`, `requantize`. Pure functional references — correctness only, still no
notion of cycles or parallel execution. That comes next, in the accelerator model,
which will *reuse* this same math but execute it through a MAC-array/buffer
abstraction instead of a flat CPU loop.

**Why `conv2d`'s loop nest matters beyond correctness.** It's 6 loops deep:
`(c_out, h_out, w_out)` for the output grid, `(c_in, kh, kw)` for the MAC-heavy inner
reduction. That inner triple loop — `c_in * kh * kw` multiply-accumulates per output
pixel — is *exactly* the work the accelerator model will later tile across a
configurable MAC array. Reading this loop now is reading the workload the hardware
has to execute; nothing changes about the math later, only how (and how fast) it's
carried out.

**`std::optional<Tensor<int32_t>>` for bias.** Bias-add is a real but optional
hardware feature — some accelerator datapaths have a bias-add stage wired into the
accumulator, some don't. `std::optional` forces the caller to explicitly check
`bias.has_value()` (or use `bias->...`) before using it, the same discipline a
"valid" bit demands before you trust the data behind it — the compiler won't let you
silently read a bias that was never provided.

**Padding via signed index arithmetic, not a padded copy.** `conv2d` computes
`ih`/`iw` as `long` (can go negative) and skips MAC taps that land outside the real
input — that's zero-padding without materializing a padded buffer. This mirrors how
an accelerator's address generator would gate accesses at the buffer boundary rather
than allocating extra zero-filled SRAM.

**Two matrix-multiply-shaped ops, on purpose.** `matmul` is the general `(M,K)x(K,N)`
primitive; `fully_connected` hand-rolls the specific `(K,)x(N,K)+bias -> (N,)` case
because that's the literal shape PyTorch's `nn.Linear` weights export in
(`out_features, in_features`), and because a matvec is the concrete unit of work that
later gets mapped one row at a time onto the MAC array.

**`relu_inplace` and `maxpool2d` are templated on `T`.** Both need to run on
whatever precision the data is currently at — raw INT32 accumulator output right
after `conv2d`, or INT8 after `requantize()` — without duplicating the function per
type.

**`requantize`: INT32 accumulator -> INT8, via shift + saturate.** A real
accelerator can't accumulate in wide INT32 forever between layers; results get
written back to the (narrow) activation buffer at reduced precision so the *next*
layer's MAC array can consume INT8 operands again. The right-shift approximates
dividing by a per-layer power-of-two scale factor; clamping to `[-128, 127]` instead
of wrapping on overflow mirrors how fixed-point DSP/MAC hardware is normally built
(saturating arithmetic, not silent wraparound).

*Documented deviation from true HW behavior*: the shift is implemented as integer
division rather than a raw `>>`, because right-shift of a negative signed integer is
only guaranteed to be an arithmetic (floor) shift as of C++20 — under C++17 it's
implementation-defined (even though every mainstream compiler does the obvious
two's-complement thing). Division truncates toward zero instead of flooring, which
can differ from a true hardware arithmetic shifter by at most 1 LSB on negative
values. Called out here rather than left implicit, per this project's rule: never
claim behavior the model doesn't actually have.

## Accelerator abstraction (`include/accelerator.hpp`, `src/accelerator.cpp`)

Four pieces, composed bottom-up:

- **`MacUnit`** — one multiply-accumulate + its accumulator register. The atomic PE
  (processing element), analogous to a single `mac_pe` module instantiated many times
  in RTL via a `generate` block.
- **`MacArray`** — a configurable `rows x cols` grid of `MacUnit`s. Given an `(R,K)`
  activation tile and a `(K,C)` weight tile (`R<=rows`, `C<=cols`), `computeTile()`
  returns the `(R,C)` output tile: `out(r,c) = sum_k input(r,k) * weight(k,c)`. This is
  the GEMM shape both convolution (via an im2col-style mapping) and a fully-connected
  layer get reduced to.
- **`LocalBuffer<T>`** — bounded on-chip storage (capacity in elements) that a tile
  must be staged into before the array can consume it; `store()` throws if the tile
  doesn't fit. Models an SRAM macro's fixed capacity as an actual constraint, not just
  a comment.
- **`Accelerator`** — the tiling controller. Owns one `MacArray` and two
  `LocalBuffer<int8_t>` instances (activation, weight) and implements `conv2d`/
  `fullyConnected` by partitioning the workload into `<=rows x <=cols` tiles, staging
  each into the buffers, and calling `computeTile()`.

**Correctness contract.** `Accelerator::conv2d`/`fullyConnected` must produce results
identical to the flat reference ops (`ops.hpp`) for the same inputs — same idea as
comparing a DUT against a golden reference model in a verification testbench, just
with the "DUT" being a differently-structured but equivalent implementation.
`tests/test_accelerator.cpp` checks this against random inputs, uneven tile
boundaries (array size that doesn't evenly divide the workload), padding, and
stride > 1. It holds bit-for-bit, not just "close," because integer addition is
exactly associative/commutative — there's no floating-point reordering error to
worry about when re-grouping the same multiply-accumulates into tiles.

**What `MacArray` does *not* model, on purpose.** No PE-to-PE forwarding, no
systolic dataflow — each PE independently completes its full K-deep reduction inside
one `computeTile()` call. That is a real simplification versus how, say, a
weight-stationary systolic array actually moves data cycle-by-cycle; it's called out
explicitly here rather than left implicit, because Phase 2/3 will start reasoning
about cycles and dataflow, and it matters that the model's limits are known before
numbers get attached to them.

**An early utilization observation (real, not seeded for later).**
`Accelerator::fullyConnected` maps `x` as a single row (`R=1`) against the array —
which means only 1 of `mac_rows` rows of PEs is ever active for an FC layer executed
this way, no matter how large the array is configured. That's already visible in
Phase 1, before any cycle counting exists: a workload's *shape* can fail to fill an
array regardless of the array's size. Phase 3 will quantify this as MAC utilization
and connect it to "why adding more MACs can fail to improve performance."

**Header/source split.** `Tensor<T>` and everything in `ops.hpp` are templates or
inline functions, so they live entirely in headers — the compiler needs their full
body at every instantiation site. `Accelerator` is a concrete class (no template
parameter), so unlike those, its member function bodies live in `src/accelerator.cpp`
and get compiled once into a real `.o`/static library (`accel_lib`) that other files
just declare against via the header. This is the first real use of `src/` in this
project, and the split is deliberate: it's the same declaration/implementation
separation as a module's port list versus its internal behavioral logic — the caller
only needs to know the interface, not the body, to link against it.

## Next

Train a small CNN on MNIST in PyTorch, quantize its weights to INT8, and export them
in a format the C++ model can load — then run the same image through the Python
reference and both C++ paths (`ops::conv2d` and `Accelerator::conv2d`) and compare.

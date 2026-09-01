# CNN Accelerator — Functional + Performance Model (C++)

**Status: work in progress, built incrementally.** This section will grow into the
full project write-up (architecture, results, optimization) as each phase lands.
See `docs/` for living design notes.

## What this project is

A C++ model of a small, configurable CNN accelerator: not "run a CNN in C++," but
"model a piece of ML accelerator *hardware* — MAC array, accumulator, on-chip
buffer — executing a CNN, in both function (does it compute the right answer) and
performance (how many cycles / how much memory traffic does it cost)."

Target pipeline being modeled:

```
Input image (INT8)
   |
   v
CNN layers: Conv -> ReLU -> Pool -> Fully Connected
   |
   v
Tiled execution (workload partitioned to fit on-chip buffer)
   |
   v
Configurable MAC array  ---> Accumulator ---> Output / activation ---> Prediction
   |
   v
Local buffer / SRAM abstraction (feeds the MAC array, holds partial results)
```

## Why two models

- **Functional model**: verifies correctness only — no notion of time. Same role as
  a golden reference model in a verification flow.
- **Performance model**: layers cycle/traffic *estimates* on top of the same
  execution, driven by configurable architecture parameters (MAC array size, buffer
  size, dataflow). This is explicitly **not RTL-cycle-accurate** — it's the kind of
  fast architectural model a team builds *before* RTL exists, to explore design
  tradeoffs cheaply.

## Status

- [x] Tensor abstraction (shape, storage, indexing) — `include/tensor.hpp`
- [ ] CNN ops (conv, matmul, ReLU, pool, FC)
- [ ] Accelerator abstraction (MAC array, accumulator, buffer model)
- [ ] Trained weights from a real small CNN (PyTorch -> exported format)
- [ ] Python vs C++ functional correctness comparison
- [ ] Performance model (cycles, MAC utilization, memory traffic)
- [ ] Architectural experiments (4x4 / 8x8 / 16x16 MAC array sweep)
- [ ] One optimization with before/after comparison

## Build & test

```sh
cd cnn-accelerator
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Layout

```
include/    header-only C++ core (Tensor, CNN ops, accelerator model)
src/        non-header-only sources (added as needed)
tests/      unit tests, one file per component
config/     accelerator configuration files (MAC array size, buffer size, ...)
scripts/    Python: training, weight export, reference inference
models/     exported trained weights, in a format the C++ model reads
results/    experiment output (tables, CSVs) — generated, not hand-written
docs/       architecture.md, performance_model.md, experiments.md
```

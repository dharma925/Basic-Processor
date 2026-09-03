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
- [x] CNN ops (conv, matmul, ReLU, pool, FC) — `include/ops.hpp`
- [x] Accelerator abstraction (MAC array, accumulator, buffer model) — `include/accelerator.hpp`, `src/accelerator.cpp`
- [x] Trained weights from a real small CNN (PyTorch -> exported format) — `scripts/train.py`, `scripts/export_int8.py`
- [x] Python vs C++ functional correctness comparison — `src/mnist_infer.cpp`, `scripts/compare_cpp_python.py`: **bit-exact match**, 0 diff on every logit, every image
- [x] Performance model (cycles, MAC utilization, memory traffic) — `PerfCounters` in `include/accelerator.hpp`; see `docs/performance_model.md`
- [ ] Architectural experiments (4x4 / 8x8 / 16x16 MAC array sweep)
- [ ] One optimization with before/after comparison

## Trained model

`Conv2d(1->8,3x3,pad=1) -> ReLU -> MaxPool(2x2) -> Flatten -> Linear(1568->10)`,
trained on MNIST (3 epochs, plain PyTorch), then post-training quantized to INT8
(per-tensor symmetric). See `docs/architecture.md` for the full quantization scheme.

Measured on the full 10,000-image MNIST test set:

| | accuracy |
|---|---|
| float32 | 96.46% |
| INT8 (this project's quantized model) | 96.43% |

```sh
cd cnn-accelerator/scripts
pip install -r requirements.txt
python3 train.py            # trains, saves models/mnist_fp32.pt
python3 export_int8.py      # quantizes + exports models/mnist_int8_model.bin
python3 quantized_reference.py       # sanity-checks the export on 20 images
python3 eval_quantized_accuracy.py   # float vs INT8 accuracy, full test set
```

## Build, test, and run on real weights

```sh
cd cnn-accelerator
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/mnist_infer                          # runs the trained model, writes results/cpp_predictions.csv
python3 scripts/compare_cpp_python.py        # diffs it against the independent numpy reference
```

The Python-vs-C++ comparison currently reports a bit-exact match — `max |python_logit
- cpp_logit| = 0` across all 20 exported test images and all 10 output classes each.

`mnist_infer` also prints a performance characterization from the model's
`PerfCounters` (default 4x4 MAC array, 16 elements/cycle buffer bandwidth):

| layer | mac_ops | utilization | total_cycles |
|---|---|---|---|
| conv1 | 56,448 | 100.0% | 5,880 |
| fc    | 15,680 | 20.8%  | 5,782 |
| **total** | 72,128 | 54.8% | **11,662** |

The full cycle model (exact formulas, and what it deliberately does not claim to
model) is in `docs/performance_model.md`.

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

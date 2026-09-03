# Performance Model

This document defines the cycle/traffic model implemented by `PerfCounters` and
instrumented into `Accelerator::conv2d`/`Accelerator::fullyConnected`
(`include/accelerator.hpp`, `src/accelerator.cpp`). Read this before trusting any
number this project reports about cycles, utilization, or throughput.

**This is a functional/performance model, not an RTL-cycle-accurate one.** It counts
real quantities (multiply-accumulates, tile loads, bytes moved) using a documented,
simple set of rules, and turns them into a cycle *estimate*. It does not model
pipeline stalls, clock-domain crossings, arbitration, DRAM refresh, instruction/
control overhead, or any of the dozens of effects an actual RTL simulation captures.
That's a deliberate scope choice, not an oversight — the same reason architecture
teams build models like this *before* RTL exists: to get fast, relative, "is this
design direction reasonable" answers, cheaply, before paying the (much higher) cost
of writing and simulating RTL. Absolute cycle counts from this model should not be
quoted as if they were silicon measurements; relative comparisons *between
configurations of this same model* are the thing it's actually good for (that's what
Phase 3's experiments use it for).

## What one "tile" is

`Accelerator` lowers both `conv2d` (via an im2col-style mapping) and `fullyConnected`
onto the same primitive: a GEMM tile of shape `(R, K) x (K, C) -> (R, C)`, where
`R <= mac_rows`, `C <= mac_cols`, computed by `MacArray::computeTile()`. Every cycle
number in this model is derived from the sequence of tiles a layer gets split into —
see `docs/architecture.md` for how conv2d/fullyConnected each construct that
sequence.

## The cycle model, per tile

For one tile with dimensions `(R, K, C)`:

```
input_tile_elems  = R * K
weight_tile_elems = K * C

load_cycles  = ceil(input_tile_elems  / buffer_bandwidth_elems_per_cycle)
             + ceil(weight_tile_elems / buffer_bandwidth_elems_per_cycle)

compute_cycles = K

tile_cycles = load_cycles + compute_cycles
```

- **`compute_cycles = K`**: each of the `R*C` active PEs performs exactly one
  multiply-accumulate per cycle, for `K` cycles, to complete its dot product. This
  holds regardless of `R`/`C` (as long as they fit in the array, which they do by
  construction of the tiling loop) — a tile with more active PEs doesn't take more
  compute cycles, it does more *useful work per cycle*. That is the whole reason MAC
  utilization matters: two tiles that take the same number of compute cycles can
  differ hugely in how many of those cycles were spent on useful multiply-accumulates
  versus idle PEs.
- **`load_cycles`**: staging the input tile and the weight tile into their local
  buffers, modeled as two *sequential* transfers over one shared
  `buffer_bandwidth_elems_per_cycle` budget (`AcceleratorConfig`). Elements here are
  `int8_t`, so elements and bytes coincide in this model.
- **Loads and compute do not overlap.** A tile's `load_cycles` and `compute_cycles`
  are simply added. Real accelerators commonly hide load latency behind the previous
  tile's compute via double-buffering — this model deliberately does not do that, so
  that "eliminate load/compute overlap" is available as a concrete, honest
  before/after optimization for Phase 4 rather than something silently assumed away.

**Total cycles for a layer** = sum of `tile_cycles` over every tile the layer's
workload was split into. **Total inference cycles** = sum over every layer's total.

## MAC utilization

```
mac_ops          = sum over tiles of (R * C * K)              -- utilized MACs
mac_capacity_ops  = sum over tiles of (mac_rows * mac_cols * K) -- array's ideal max over those same compute cycles

mac_utilization = mac_ops / mac_capacity_ops
```

This is *spatial* utilization — the fraction of the array's PEs doing useful work,
averaged over compute cycles — not blended with the idle time spent in `load_cycles`.
Drawing the line there is deliberate: it isolates "does the workload's shape fill the
array" (a property of `R`, `C` versus `mac_rows`, `mac_cols`) from "is the array
waiting on memory" (a property of `load_cycles` versus `compute_cycles`), which are
two different bottleneck stories Phase 3 needs to be able to tell apart.

## Memory traffic

`activation_bytes_loaded` and `weight_bytes_loaded` count every element written into
the respective `LocalBuffer` — i.e. every time a tile is staged from the layer's
source tensor into on-chip storage. This is meant to model **on-chip buffer traffic**
(how much data moves from wherever it's held into the buffers the MAC array reads
from), not off-chip DRAM traffic specifically — this model doesn't distinguish
"already on chip" from "came from DRAM"; everything an `Accelerator` call touches is
treated as if it needs staging.

One asymmetry worth knowing, because it's real and it's already visible without any
Phase 4 optimization: `fullyConnected` stages its activation tile (`x`) into the
buffer **once**, before the column-tile loop, and reuses it for every output-channel
tile — while `conv2d` gathers and restages a fresh input tile for every
`(pixel-tile, channel-tile)` pair. That difference is *inherent to how each layer's
loop is structured*, not something engineered in for this document; it falls directly
out of `x` not changing across FC's column tiles while conv2d's im2col-gathered input
tile does change across every tile. It is, in miniature, exactly the kind of data
reuse a real dataflow choice (weight-stationary vs. output-stationary vs.
input-stationary) is about — see `docs/experiments.md` (Phase 4) for a deliberate,
measured version of the same idea applied to weights.

## Performance counters as a PMU

`PerfCounters` accumulates across calls to `conv2d`/`fullyConnected` until
`Accelerator::resetStats()` is called — the same usage pattern as a hardware
performance-monitoring unit (PMU): counters that free-run during execution, read and
optionally reset at whatever measurement boundary you care about (here, typically
once per layer). `mnist_infer` uses exactly this pattern to report conv1 and fc as
separate rows.

## Measured, not assumed: one real run

From `./build/mnist_infer`, on the actual trained model (`Conv(1->8,3x3,pad=1)` then
`FC(1568->10)`), default `4x4` MAC array, `16` elements/cycle buffer bandwidth:

| layer | tiles | mac_ops | mac_capacity_ops | utilization | compute_cycles | load_cycles | total_cycles |
|---|---|---|---|---|---|---|---|
| conv1 | 392 | 56,448 | 56,448 | **100.0%** | 3,528 | 2,352 | 5,880 |
| fc    | 3   | 15,680 | 75,264 | **20.8%**  | 4,704 | 1,078 | 5,782 |
| **total** | 395 | 72,128 | 131,712 | **54.8%** | 8,232 | 3,430 | **11,662** |

Two real observations already visible from this one run, before any Phase 3 sweep:

1. **conv1 hits exactly 100% utilization at this array size** — its workload shape
   (`p_total=784` pixels, `c_out=8` channels) divides evenly into `4x4` tiles with no
   remainder. That's a property of *this specific array size against this specific
   layer shape*, not a general property of convolution — Phase 3 checks whether it
   still holds at `8x8` or `16x16`.
2. **fc's utilization (20.8%) is exactly the `R=1`-out-of-`mac_rows=4` effect
   predicted in `docs/architecture.md`**, compounded by its last column tile being
   partial (`n=10` doesn't divide evenly by `mac_cols=4`): `10/48` active-PE-cycles
   out of the array's capacity over the same compute cycles.

Despite conv1 doing more total work, its 5,880 cycles and fc's 5,782 cycles are
nearly equal here — fc gets *almost as many total cycles as conv1* while doing
roughly a quarter the useful work per cycle on average. That is what "poor
utilization" costs concretely, in cycles, not just as a percentage.

## What this model does not claim

- No pipeline stalls, no control/instruction overhead, no arbitration between the
  activation and weight buffer ports (they're modeled as one shared bandwidth budget).
- No load/compute overlap (double buffering) — see above.
- No off-chip-vs-on-chip distinction in memory traffic.
- `buffer_bandwidth_elems_per_cycle` is a configurable, illustrative number, not
  derived from a real SRAM or bus datasheet.
- Values here depend only on tensor **shapes** and `AcceleratorConfig`, not on data
  values — this model has no notion of value-dependent effects a real accelerator
  might exploit (e.g. skipping multiplies against zero-valued weights/activations,
  sparsity acceleration). Every image of the same shape produces identical
  performance counters, which is why `mnist_infer` profiles just one image to
  characterize the whole model.

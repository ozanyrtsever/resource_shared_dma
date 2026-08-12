# Update — 2D single-transfer GEMV + full cycle-budget decomposition

This update adds two things on top of the shared-FMA coprocessor: (1) a **2D single-transfer** feed
that sends a whole weight matrix in one DMA descriptor, and (2) a **full cycle-budget decomposition**
of an accelerated inference — measured entirely in software, with nothing excluded — plus the
accelerator- and CPU-side **contention** split under co-execution. No RTL and no area change: both are
software/measurement work on the existing design.

---

## 1. 2D single-transfer GEMV (removes row-chunking)

**Problem.** The DMA's 1-D transfer size (`SIZE_D1`) is 16-bit → **max 65 535 elements per transfer**.
A LeNet layer-1 weight matrix is 784×300 = 235 200 words, so it had to be **chunked by whole rows**
(≈4 transfers, each re-loading the input vector into the coprocessor's `x`-buffer).

**Fix.** The X-HEEP DMA already supports 2-D transfers, and **both** 2-D size fields are 16-bit, so a
matrix streams in one descriptor as `size_d1 = N` (inner, contiguous) × `size_d2 = M` (outer), unit
stride on both. The input vector is LOADed **once** per layer (1-D), then the entire M×N weight matrix
flows in a **single 2-D transfer**. Because it is one transfer, the coprocessor sees **one flush + one
uninterrupted stream** of M·N weights and processes it as M rows exactly as before; the accelerator RTL
is unchanged. The write side follows the accelerator's `done` (M results, written linearly to
`out[0..M-1]`), so the read≠write count of the reducing accelerator is handled as in the chunked path.

**Validation** — `sw/applications/ml_rt_gemv_2d/`: a whole matrix of **65 792 > 65 535** elements
(N=256, M=257) sent in one 2-D transfer, **bit-exact** vs a CPU fused golden on every spot-checked row,
**including the last row** (index 256, which crosses the destination's 2-D row boundary since M>N) —
confirming the linear destination addressing is correct for M>N. Then re-verified on the real network:
LeNet layer-1 (235 200 words) in one transfer, **bit-exact 5/5** on MNIST.

**Cost/benefit.** Purely a programming change — the 2-D datapath is already synthesized in the X-HEEP
DMA, so **zero RTL / zero area**. Benefit: the input buffer is filled once per layer instead of once
per chunk, and the per-transfer setup drops from ~4 to 1 per layer; LeNet forward went from 556 318 to
**552 228 cycles** (small — the compute dominates), and the driver code is simpler.

---

## 2. Full cycle-budget decomposition (where every cycle goes)

**Goal.** Account for **every** cycle of one accelerated inference, split into disjoint phases, with no
part excluded. Done in software with the cycle counter (`mcycle`) bracketing each phase, so the phases
**sum to** the measured accelerator forward — no RTL performance counters were added (they would sit
inside the accelerator, which *is* a Design-Compiler synthesis target, and would cost area).

**Phases** (`ml_lenet/main.c`): **setup** = DMA descriptor validate + register program (per transfer);
**load** = LOAD-transfer wall time (input vector → `x`-buffer); **gemv** = GEMV-transfer wall time
(weights streamed + FMA + writeback, which overlap in the streaming pipeline and are therefore not
separable in software); **act** = ReLU + bias (CPU). The contention and FMA-latency terms are the
*separately-attributed adders* below, so nothing is double-counted.

**Result — LeNet-300-100, one MNIST digit, `FPU_ADDMUL_LAT = 0`:**

| Phase | Cycles | Share |
|---|---:|---:|
| setup (DMA program) | 2 688 | 0.5 % |
| load (x stream) | 1 833 | 0.3 % |
| **gemv (stream + FMA + writeback)** | **533 267** | **96.6 %** |
| act (bias + ReLU, CPU) | 14 440 | 2.6 % |
| **total (accelerator forward)** | **552 228** | 100 % |

The coprocessor is **compute-bound**: 96.6 % is the weight-streaming dot-product phase, and the whole
software overhead of driving the DMA (setup + load) is **under 1 %**. The `gemv` phase is
**533 267 / 266 200 MACs = 2.00 cycles per fused multiply-add** — the floor for a single-issue,
single-FMA design at zero latency (one cycle to accept each weight, one to issue the FMA and latch the
result, no wait state).

---

## 3. Contention split under co-execution

`ml_coexec/main.c` runs the accelerated LeNet forward **twice** — alone, then while the CPU runs a
256-tap FIR filter (a different FP job that also reads memory) contending until each GEMV is ready — and
brackets the GEMV both times. The difference is the accelerator's slow-down from co-running.

**Result (`L = 0`):** the `gemv` phase stretches 533 287 → 609 462 cycles under contention =
**+14.3 % (76 175-cycle) memory-and-drain term**. Because a combinational FMA has essentially no drain
at `L = 0`, this is **almost entirely operand-bus** contention on the shared SRAM. The CPU's own FIR
slows a comparable **+3.9 %** (bus, not FMA starvation → the CPU-priority guarantee holds on live
traffic), and the two jobs finish in **1.20×** the time of running them sequentially — the inference
riding in the FIR's idle FMA cycles. Both remain bit-exact. The term will grow with FMA latency, where
the drain component stops being negligible (the latency sweep is next).

---

## Files

| File | Change |
|---|---|
| `x-heep/sw/applications/ml_rt_gemv_2d/main.c` | **new** — 2-D single-transfer GEMV validation (>65 535 in one transfer) |
| `x-heep/sw/applications/ml_lenet/main.c` | 2-D single-transfer per layer + per-phase cycle-budget decomposition |
| `x-heep/sw/applications/ml_coexec/main.c` | 2-D + accelerator/CPU contention split (gemv dual vs alone) |
| `thesis_progress.md` | §11 new subsection "The accelerator's cycle budget", co-execution table refreshed |

No RTL, template, or config files changed — `CHANGES-to-xheep.patch` is unaffected.

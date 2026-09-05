# Shared-FMA DMA Coprocessor on CV32E40P (X-HEEP) — paper materials

A memory-mapped, DMA-programmed **reduction coprocessor** that **time-shares the CPU's single existing
FPnew fused multiply-add (FMA)** through the APU interface, arbitrated with **strict CPU priority** —
accelerating floating-point ML inference on an MCU-class core **without adding a second floating-point
unit**. The final design is a **pipelined single coprocessor**: for a batched matrix product it issues
the `B` independent batch-MACs of each streamed weight back-to-back, so one unit keeps the shared FMA
full regardless of its pipeline latency — retiring an earlier dual-coprocessor variant.

This repository is a **curated subset** of a working X-HEEP fork: only the files this work **created or
modified** (the contribution), plus the full progress report, the companion measurement reports, and
the model-export scripts. The unmodified remainder of X-HEEP is upstream
(https://github.com/esl-epfl/x-heep) and the CV32E40P core is published; `CHANGES-to-xheep.patch` is
the exact diff applied to the vendored/template files.

## Read this first (the final design)
- **`PAPER_NOTES.md`** — compact **technical dossier** to write the paper from: the system in one
  sentence, all components, how it works (DMA hw_fifo protocol, arbiter internals, pipelined engine,
  latency-hiding math), the design evolution, what's verified, the results, parameters, and a file map.
- **`thesis_progress.md`** — the full narrative: problem, background (CV32E40P/APU/FMA, X-HEEP, DMA
  hw_fifo), the arbiter and pipelined-coprocessor design, the evaluation, and every result. Appendix A
  is the reproducibility recipe; Appendix C is a file-by-file change-log (Part IV = the final design).
- **Companion measurement reports** (self-contained, every number explained):
  - `PERF_BENCH_PIPE_SWEEP.md` — toy-MLP L×policy sweep (pipelined single).
  - `PERF_LENET_PIPE_SWEEP.md` — real LeNet-300-100 / MNIST L×policy sweep (+ accuracy, + real time).
  - `DC_STUDY_OPERATING.md` — DC-NXT area & Fmax at the converged **260 MHz** operating point, with a
    per-component (CPU / FPU / FMA / arbiter) breakdown and the "no second FPU" derivation.
  - `COMPARISON_vs_tanase2026.md` — closest prior art, advantages/gaps.
- **`CHANGES-to-xheep.patch`** — precise edits to X-HEEP's vendored RTL / templates (the arbiter wiring
  threaded through `cv32e40px_top.sv` and the `*.sv.tpl` files). Full copies are also at their real
  paths under `x-heep/`.

## Key results (all bit-exact vs a CPU/NumPy reference; TSMC 40 nm G, DC-NXT for area/Fmax)
| Result | Number |
|---|---|
| **Performance — `cyc/MAC` flat in FMA latency** (pipelining hides L) | toy MLP **1.14**, LeNet **1.12**, constant across L = 0..5 (a serial unit grows to 1.14 + L) |
| **Speedup vs an optimized CPU baseline** (weight-reuse + 8-way ILP, disassembly-verified; bit-exact, LeNet 8/8) | **rises with FMA latency**: toy MLP **2.5× (L=0) → 5.2× (L=5)**, LeNet **3.4× → 7.0×**. The coprocessor fills the pipeline the scalar CPU (~2 outstanding FP) cannot; gap widens with depth. |
| **Real time @ 260 MHz** (cycle × 3.845 ns, L=0) | MLP coproc **62.1 µs/img** (vs optimized CPU 154 µs), LeNet **1.22 ms/img** (vs 4.10 ms) |
| **Co-execution** (CPU FP FIR ∥ coproc inference, one FMA) | coproc +0.12 % (LeNet) / +2.7 % (MLP); CPU FIR +5.4 % / +1.9 % under CPU-strict — both bit-exact |
| **Area — "no second FPU"** (260 MHz, L0) | core 88.0 k µm² = CPU 52.9 k + FPU 33.2 k (**FMA 21.5 k**) + **arbiter 1.8 k**; arbiter = **8.4 % of one FMA / 2.0 % of core**; coproc logic 4.9 k (buffer = SRAM macro). Per accelerator **3.9× less added area** than a dedicated FMA. |
| **Frequency** | shared core closes at **260 MHz** (3.845 ns, WNS→0 via `converge.sh`); L0 binding path is the CPU load-store pipeline, not the FMA |
| **Arbiter policy** | P0 CPU-strict / P1 round-robin / P2 QoS-weighted (elaboration-time; only the selected one synthesizes); area P1 < P0 < P2, all ≤ ~10 % of one FMA |

*(The evaluation journey behind these — interleaved-pair dot product, serial GEMV break-even, dense
layer 2.6×, StarDist conv 1.97×, serial LeNet 4.8×, the two-coprocessor experiment — is documented in
`thesis_progress.md` §5 and §11, and its source apps are under `x-heep/sw/applications/` as history.)*

## Repository map
```
PAPER_NOTES.md                  technical dossier — WRITE THE PAPER FROM THIS
thesis_progress.md              full report — narrative + all results + change-log (App C)
PERF_BENCH_PIPE_SWEEP.md        toy-MLP performance sweep (final, pipelined)
PERF_LENET_PIPE_SWEEP.md        real-LeNet performance sweep (final, pipelined)
DC_STUDY_OPERATING.md           DC-NXT area & Fmax at 260 MHz + component breakdown
COMPARISON_vs_tanase2026.md     closest prior art, advantages/gaps
CHANGES-to-xheep.patch          exact diff to X-HEEP vendored/template files
example_model/                  model-export scripts (gen_lenet_mnist.py trains+exports LeNet/MNIST)
x-heep/
  hw/vendor/xheep/cv32e40px/rtl/
    dma_apu_arbiter.sv          *** THE contribution: FMA-sharing arbiter — 3-requestor (CPU+acc0+acc1),
                                    2-bit owner tag, NO drain (grants every cycle), swappable policy ***
    cv32e40px_top.sv            arbiter instantiated + wired between core APU and shared fp_wrapper
                                    (behind `COPROC_FPU_SHARE`)
  tb/
    dma_fp_dot_accel_pipe.sv    *** THE final coprocessor: PIPELINED batched GEMM — decoupled issue/collect,
                                    up to L+1 MACs in flight, interlock inflight<B -> cyc/MAC flat in L ***
    dma_fp_dot_accel_is.sv      serial batched-GEMM coprocessor (baseline, kept for the L comparison)
    xbuf_ram.sv                 the input buffer as a 1R1W submodule (SRAM macro in silicon; black-boxed for area)
    testharness.sv.tpl          coprocessor on DMA hw_fifo + arbiter APU ports (behind `COPROC_FPU_SHARE`);
                                    `COPROC_PIPE`/`COPROC_SERIAL` knob selects pipelined vs serial (acc1 idle)
    tb_top.cpp                  Verilator TB; FST tracing guarded by `#if VM_TRACE` (trace-off = fast)
  hw/core-v-mini-mcu/*.tpl      APU-port threading through the SoC hierarchy
  configs/cv32e40px_fpu_dma.hjson   cv32e40px+FPU, DMA hw_fifo, SRAM enlarged to 2 MB
  core-v-mini-mcu.core          `+define+COPROC_FPU_SHARE` master toggle (single-define clean revert)
  sweep.sh                      L×policy×QoS sweep driver (PROJECT/OUTDIR env; resumable)
  util/xheep_gen/load_config.py fpu_addmul_lat forwarding fix (enables the FMA-latency sweep)
  dc_scripts/                   Design Compiler NXT flow: converge.sh (find 260 MHz), study.sh + study.tcl
                                    (hierarchy-preserved area/Fmax, per-component extraction), rtl_*.f
  sw/applications/
    perf_bench_pipe/  *** FINAL: pipelined coprocessor on the toy MLP — cyc/MAC flat 1.14, 2.5x->5.2x vs CPU ***
    perf_lenet_pipe/  *** FINAL: pipelined coprocessor on REAL LeNet — cyc/MAC flat 1.12, 3.4x->7.0x, 8/8 correct ***
    perf_lenet/       LeNet MNIST data location (lenet_mnist_data.h, regenerated) used by perf_lenet_pipe
    ml_lenet, ml_coexec, ml_dual, ml_conv1[_full], ml_rt_gemv[_2d], ml_fc_bringup
                      evaluation-journey demos (§11): dense layer, conv, LeNet, co-exec, dual — history
    example_matfloat  Y0 CPU baseline (303-cycle dot product)
    y1_sum_test, y2_fpdotp_test, y3_{sweep,contention,independent}
                      X-HEEP bring-up + first shared-FMA dot product + Y3 measurement kernels — history
```
(Large generated weight headers `lenet_mnist_data.h` are omitted; regenerate with
`example_model/gen_lenet_mnist.py` into `sw/applications/perf_lenet/`.)

## One-line thesis
Because an MCU's FPU already contains an FMA that a scalar CPU leaves idle most cycles, a DMA-fed
coprocessor can borrow it — under a small CPU-priority arbiter — to run floating-point ML inference
(and to co-run alongside the CPU's own FP work), bit-exactly and several-fold faster; pipelining the
coprocessor lets a **single** unit hide the FMA's latency, so the whole cost of acceleration is a
**~1.8 k µm² arbiter (8.4 % of one FMA)** instead of a second FMA.

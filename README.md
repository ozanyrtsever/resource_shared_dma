# Shared-FMA DMA Coprocessor on CV32E40P (X-HEEP) — paper materials

A memory-mapped, DMA-programmed **dot-product / GEMV coprocessor** that **time-shares the CPU's single
existing FPnew fused multiply-add (FMA)** through the APU interface, arbitrated with **strict CPU
priority** — accelerating floating-point ML inference on an MCU-class core **without adding a second
floating-point unit**.

This repository is a **curated subset** of a working X-HEEP fork: it contains only the files this work
**created or modified** (the contribution), plus the full progress report and the model-export scripts.
The unmodified remainder of X-HEEP is upstream (https://github.com/esl-epfl/x-heep) and the CV32E40P
core is published; `CHANGES-to-xheep.patch` is the exact diff applied to the vendored/template files.

## Read this first
- **`thesis_progress.md`** — the full narrative: problem, background (CV32E40P/APU/FMA, X-HEEP, DMA
  hw_fifo), the arbiter design, the measurement study, and every result. **This is the primary source
  for the paper.** Appendix C is a file-by-file change-log; Appendix A is the reproducibility recipe.
- **`CHANGES-to-xheep.patch`** — precise edits to X-HEEP's vendored RTL / templates (e.g. the arbiter
  wiring threaded through `cv32e40px_top.sv` and the `*.sv.tpl` files). Full copies of those files are
  also included at their real paths under `x-heep/`.
- **Analysis reports** (self-contained, every number explained explicitly): `PERF_SWEEP_REPORT.md`
  (toy-MLP L×policy sweep), `LENET_SWEEP_REPORT.md` (real LeNet-300-100 L×policy sweep),
  `AREA_REPORT.md` (DC-NXT area study: FPU/FMA L-sweep + 3 arbiter policies + accelerator logic/SRAM),
  `COMPARISON_vs_tanase2026.md` (closest prior art, advantages/gaps).

## Key results (all bit-exact vs a CPU/NumPy reference; TSMC 40 nm for area)
| Result | Number |
|---|---|
| On-SoC shared-FMA dot product (32-elem) | **208 cyc** vs 303 CPU baseline, **no added FP datapath** |
| Length scaling | break-even N ≈ 16, asymptote ≈ 3.5× |
| CPU-priority under FMA-latency 0–5 | CPU slow-down bounded **≤ 16 %** |
| **Area (TSMC 40 nm G, DC-NXT):** avoided FMA vs added arbiter | one FMA (`fpnew_fma_multi`) **5 789 µm²** (full FPU 12 455) vs CPU-strict arbiter **453 µm²** = **7.8 % of one FMA** (all-RR 465; QoS 928 ≈2×); coproc logic 4 461 µm² + a 32 KB input SRAM; **saving 5 336 µm²/accel, 11 125 µm² for the dual**. Arbiter is L-independent while the FMA grows +30 % over L=0..5 → sharing is even better at higher latency. See `AREA_REPORT.md`. |
| Dense layer (GEMV) | **2.6×** |
| Convolution (StarDist conv1, im2col→GEMV) | bit-exact, **1.97×** |
| **Full network: LeNet-300-100 on MNIST** (single image, 2D transfer) | bit-exact vs CPU & NumPy, **4.80×** (552 228 vs 2 654 927 cyc, whole matrix in one 2D transfer), model ~98 % |
| **Co-execution (ML ∥ CPU FIR-DSP), shared FMA** | both bit-exact; under contention accel GEMV +14.3 % / CPU +3.9 % (bus, not FMA-starvation); inference hidden in idle FMA cycles → **1.20×** |
| **Batched GEMM (memory→compute), L=0** | batching flips memory-bound GEMV → compute-bound GEMM: toy MLP **1.14 cyc/MAC, 5.46×**; real **LeNet-300-100: 1.12 cyc/MAC, 88 % FMA-util, 8.34×** (single) — all bit-exact, LeNet **8/8 MNIST correct** |
| **Two coprocessors on one FMA** (3-requestor arbiter, batched) | dual GEMM **~1.99× vs single at L≥2** (fills the FMA pipeline-wait gaps; 1.12× at L=0 where one accel already saturates the FMA), channels balanced; **LeNet dual 9.06× vs CPU**; the second accelerator's value *grows* with FMA latency — **still no 2nd FPU** |
| **Arbiter policy / QoS** (compute-bound, 3-way) | CPU-strict keeps the CPU's own job within **≤9 %** under two accelerators at every L; the QoS weights dial the CPU's slow-down **+13 %…+103 %** and the acc0/acc1 balance (imbalance 223 → 762 k) |
| **Full L×policy sweeps (21 configs each)** | toy MLP + real LeNet, all bit-exact — see `PERF_SWEEP_REPORT.md`, `LENET_SWEEP_REPORT.md`; closest prior art in `COMPARISON_vs_tanase2026.md` |

## Repository map
```
thesis_progress.md              full report (READ FIRST) — narrative + all results + change-log
CHANGES-to-xheep.patch          exact diff to X-HEEP vendored/template files
example_model/                  model-export scripts (gen_lenet_mnist.py trains+exports LeNet/MNIST)
x-heep/
  hw/vendor/xheep/cv32e40px/rtl/
    dma_apu_arbiter.sv          *** THE contribution: FMA-sharing arbiter — parametric 3-requestor (CPU+acc0+acc1), swappable policy (CPU-strict / all-RR / QoS) ***
    cv32e40px_top.sv            arbiter instantiated + wired between core APU and shared fp_wrapper
  tb/
    dma_fp_dot_accel_is.sv      *** batched-GEMM coprocessor: runtime N/M/B, input-stationary, weight-reuse (B MACs/weight); input buffer factored into u_xbuf ***
    xbuf_ram.sv                 the input buffer as a 1R1W submodule (an SRAM macro in silicon; black-boxed to report logic-only area)
    dma_fp_dot_accel.sv         earlier pair-streaming dot-product coprocessor
    dma_sum_accel.sv            FP-free integer-sum accelerator (Y1 bring-up)
    tb_top.cpp                  Verilator TB; FST tracing guarded by `#if VM_TRACE` -> trace-off builds run fast (needed for the large LeNet sweep)
    testharness.sv.tpl          two coprocessors on DMA ch1/ch2 hw_fifo + arbiter APU ports (behind `COPROC_FPU_SHARE`)
  hw/core-v-mini-mcu/*.tpl      APU-port threading through the SoC hierarchy
  configs/cv32e40px_fpu_dma.hjson   cv32e40px+FPU, DMA hw_fifo, SRAM enlarged to 2 MB
  core-v-mini-mcu.core          `+define+COPROC_FPU_SHARE` master toggle (single-define clean revert)
  sweep.sh                      parametric L×policy×QoS sweep driver (PROJECT/OUTDIR/RUN_TIMEOUT env; resumable)
  util/xheep_gen/load_config.py fpu_addmul_lat forwarding fix (FMA-latency sweep)
  dc/                           Design Compiler NXT area flow (run_area.sh + synth_area.tcl); rtl_accel_is.f black-boxes the buffer; reports/ = FPU/FMA L-sweep + 3 arbiter policies + accel logic
  sw/applications/
    ml_lenet/       full LeNet-300-100 forward (2D single-transfer) + cycle-budget decomposition (4.80×)
    perf_bench/     *** batched end-to-end benchmark on a toy MLP: single/contention/DUAL/3-way, memory->compute, all bit-exact ***
    perf_lenet/     *** the same benchmark on the REAL LeNet-300-100/MNIST (batched, 2D weights): single 8.34x, dual 9.06x, 8/8 correct ***
    ml_coexec/      *** co-execution: accel ML || CPU FIR, 2D + accel/CPU contention decomposition ***
    ml_dual/        dual-stream: CPU classifies one digit while accel classifies another
    ml_rt_gemv/     runtime-N/M bring-up
    ml_rt_gemv_2d/  2D single-transfer GEMV validation (whole M×N > 65535 in ONE 2D DMA transfer)
    ml_conv1[_full]/ convolution via im2col→GEMV
    y2_fpdotp_test/ Y2: first on-SoC shared-FMA dot product
    y3_{sweep,contention,independent}/  Y3 measurement kernels (length, concurrency, contention)
```
(Large generated weight headers `lenet_mnist_data.h` are omitted; regenerate with
`example_model/gen_lenet_mnist.py`. Build/run flow is in `x-heep/DEV_SETUP.md`.)

## One-line thesis
Because an MCU's FPU already contains an FMA that a scalar CPU leaves idle most cycles, a DMA-fed
coprocessor can borrow it — under a small CPU-priority arbiter — to run floating-point ML inference
(and to co-run alongside the CPU's own FP work), bit-exactly and several-fold faster, at the cost of a
~453 µm² arbiter instead of a second ~5 800 µm² FMA (≈ 7.8 % of one FMA).

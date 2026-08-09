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

## Key results (all bit-exact vs a CPU/NumPy reference; TSMC 40 nm for area)
| Result | Number |
|---|---|
| On-SoC shared-FMA dot product (32-elem) | **208 cyc** vs 303 CPU baseline, **no added FP datapath** |
| Length scaling | break-even N ≈ 16, asymptote ≈ 3.5× |
| CPU-priority under FMA-latency 0–5 | CPU slow-down bounded **≤ 16 %** |
| **Area:** avoided FMA (`fp_wrapper`) vs added arbiter | **12 812 µm² vs 321 µm²** (arbiter = **2.5 %** of one FMA) |
| Dense layer (GEMV) | **2.6×** |
| Convolution (StarDist conv1, im2col→GEMV) | bit-exact, **1.97×** |
| **Full network: LeNet-300-100 on MNIST** | bit-exact vs CPU & NumPy, **4.79×** (556 318 vs 2 664 446 cyc), model ~98 % |
| **Co-execution (ML ∥ CPU FIR-DSP), shared FMA** | both bit-exact; CPU +7.9 % (bus, not FMA-starvation); inference hidden in idle FMA cycles → **1.15×** |

## Repository map
```
thesis_progress.md              full report (READ FIRST) — narrative + all results + change-log
CHANGES-to-xheep.patch          exact diff to X-HEEP vendored/template files
example_model/                  model-export scripts (gen_lenet_mnist.py trains+exports LeNet/MNIST)
x-heep/
  hw/vendor/xheep/cv32e40px/rtl/
    dma_apu_arbiter.sv          *** THE contribution: CPU-priority FMA-sharing arbiter ***
    cv32e40px_top.sv            arbiter instantiated + wired between core APU and shared fp_wrapper
  tb/
    dma_fp_dot_accel_is.sv      runtime-programmable GEMV coprocessor (reads N,M from a stream header)
    dma_fp_dot_accel.sv         earlier pair-streaming dot-product coprocessor
    dma_sum_accel.sv            FP-free integer-sum accelerator (Y1 bring-up)
    testharness.sv.tpl          coprocessor on DMA ch1 hw_fifo + arbiter APU port (behind `COPROC_FPU_SHARE`)
  hw/core-v-mini-mcu/*.tpl      APU-port threading through the SoC hierarchy
  configs/cv32e40px_fpu_dma.hjson   cv32e40px+FPU, DMA hw_fifo, SRAM enlarged to 2 MB
  core-v-mini-mcu.core          `+define+COPROC_FPU_SHARE` master toggle (single-define clean revert)
  util/xheep_gen/load_config.py fpu_addmul_lat forwarding fix (FMA-latency sweep)
  dc/                           Design Compiler area flow + the synthesis reports behind the area table
  sw/applications/
    ml_lenet/       full LeNet-300-100 forward pass on the coprocessor (4.79×)
    ml_coexec/      *** co-execution: accel ML || CPU FIR filter, sharing one FMA ***
    ml_dual/        dual-stream: CPU classifies one digit while accel classifies another
    ml_rt_gemv/     runtime-N/M bring-up
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
~321 µm² arbiter instead of a second ~12 800 µm² FMA.

# Update — Batched GEMM, full L×policy sweeps (toy MLP + real LeNet), and the rebuilt area study

This update turns the earlier point results into **complete, swept measurements** and **rebuilds the area
study** with the current 3-requestor design. Three self-contained reports capture every number:
`PERF_SWEEP_REPORT.md` (toy MLP), `LENET_SWEEP_REPORT.md` (real LeNet-300-100), `AREA_REPORT.md`
(Design Compiler NXT). A fourth, `COMPARISON_vs_tanase2026.md`, positions the closest prior art.
Still **no second FPU**.

---

## 1. Batched GEMM makes the sharing compute-bound

The coprocessor (`dma_fp_dot_accel_is`) now runs **batched** (B inputs buffered once; each streamed weight
is reused for B multiply-accumulates → weight-reuse without weight-stationary). At B = 8 the **shared FMA**
becomes the bottleneck instead of the operand bus, so the arbiter policy actually matters.

- **Memory → compute (L = 0):** cyc/MAC drops from ~2.0 (B = 1, memory-bound) to **~1.12 (B = 8)**, shared-FMA
  utilisation **~49 % → ~88 %**.
- A single serial coprocessor uses ≈ **1 / (1 + L)** of the FMA (measured utilisation matches to the point).

## 2. Full L (0..5) × policy (0/1/2) × QoS-weight sweeps

`sweep.sh` is now **parametric** (`PROJECT` / `OUTDIR` / `RUN_TIMEOUT` env vars, resumable) so one driver
runs both benchmarks. **21 configurations each, all bit-exact.**

**Toy MLP (`perf_bench`, 128-64-32-16)** and **real LeNet-300-100 (`perf_lenet`, 784-300-100-10, real
MNIST)** give the same trends; LeNet's headline is higher:

| Metric (L = 0) | toy MLP | **LeNet-300-100 (real)** |
|---|---:|---:|
| B = 8 cyc/MAC · FMA-util | 1.14 · 87 % | **1.12 · 88 %** |
| single-coprocessor speedup vs CPU | 5.46× | **8.34×** |
| dual (acc0+acc1) speedup vs CPU | 5.27× | **9.06×** |
| LeNet accuracy | — | **8/8 MNIST correct** |

- **The second coprocessor is a latency-hiding device:** dual gives only ~1.12× at L = 0 (one accel already
  fills the FMA) but **~1.99× at L ≥ 2** (it slots into the pipeline-wait gaps the serial accel leaves). Two
  coprocessors stay **3.1×–9.1× vs the CPU** at every latency, while one falls to 1.6× at L = 5.
- **Arbiter policy (3-way, compute-bound):** CPU-strict keeps the CPU's own FIR job within **≤ 9 %** even
  under two accelerators at every L; full round-robin costs it more; the **QoS weights dial the CPU
  slow-down +13 %…+103 %** and the acc0/acc1 balance (channel imbalance 223 → ~762 k cycles). Policy matters
  where the FMA is saturated (compute-bound); at high L the accelerators are latency-bound and contention
  ≈ 0 regardless of policy.

## 3. Rebuilt area study (Design Compiler NXT, TSMC 40 nm, 100 MHz) — supersedes the old point numbers

Every module re-synthesized in one consistent DC-NXT run of the current 3-requestor design. FPU and FMA
swept over the ADD/MUL pipeline latency L; the arbiter synthesized once per policy; the accelerator's 32 KB
input buffer factored into `u_xbuf` and **black-boxed** so the reported accelerator area is logic-only (the
buffer is an SRAM macro, reported separately — never flip-flops).

| Block | area (µm²) | note |
|---|---:|---|
| full FPU `cv32e40px_fp_wrapper` (L = 0 → 5) | 12 455 → 14 337 | registers grow +30 % with L |
| **one FMA `fpnew_fma_multi`** (L = 0 → 5) | **5 789 → 7 521** | ≈ half the FPU; the honest "avoided" unit |
| arbiter **P0 CPU-strict** | **452.94** | **7.8 % of one FMA** (3.6 % of the FPU) |
| arbiter P1 full round-robin | 465.07 | +2.7 % vs P0 |
| arbiter P2 QoS (4:2:1) | 927.60 | ≈ 2× (credit counters + argmax) |
| accelerator **logic** (buffer black-boxed) | 4 460.82 | incl. the eight batch accumulators |
| input buffer `x_buf` | 32 KB SRAM/accel | memory compiler (~0.11 mm² est.), not flops |

**"No second FPU":** sharing adds a **453 µm² arbiter** instead of a **5 789 µm² FMA** → **saves 5 336 µm²
per coprocessor**, **11 125 µm² for the dual** (one arbiter vs two FMAs). The arbiter is L-independent while
the FMA grows +30 % over L = 0..5, so sharing is *more* favourable at deeper pipelines (arbiter 7.8 % → 6.0 %
of the FMA).

## 4. Supporting changes

- **`u_xbuf` refactor** — the input buffer moved into a `xbuf_ram` submodule (1 write / 1 combinational-read
  port), functionally identical to the old inline array (bit-exact), so synthesis can treat it as an SRAM
  macro (`dc/rtl_accel_is.f` omits it → black box → logic-only area).
- **`tb_top.cpp` FST tracing guarded by `#if VM_TRACE`** — trace-off builds now link and run several× faster
  (essential for the LeNet sweep: trace-on wrote a 777 MB waveform and a single config timed out at 4 h;
  trace-off ~10 min/config).
- **DC flow generalised** (`dc/synth_area.tcl` + `dc/run_area.sh`): per-run env (`L`/`POLICY`/`MAXN`/`MAXB`),
  per-run report suffixes, robust area read from the report, and the `dcnxt_shell` binary.

---

## Files

| File | Change |
|---|---|
| `x-heep/tb/dma_fp_dot_accel_is.sv` | batched GEMM (runtime N/M/B, weight-reuse); input buffer factored into `u_xbuf`; fully commented |
| `x-heep/tb/xbuf_ram.sv` | **new** — input buffer as a 1R1W submodule (SRAM macro in silicon) |
| `x-heep/tb/tb_top.cpp` | **new** — Verilator TB with FST tracing behind `#if VM_TRACE` (fast trace-off sweeps) |
| `x-heep/tb/x-heep-tb-utils.core` | add `xbuf_ram.sv` to the tb fileset |
| `x-heep/sw/applications/perf_bench/main.c` | batched (B=1/B=8), per-phase decomposition, contention, dual, 3-way |
| `x-heep/sw/applications/perf_lenet/main.c` | **new** — same benchmark on real LeNet-300-100/MNIST (2D weight streaming) |
| `x-heep/sweep.sh` | **new** — parametric resumable L×policy×QoS driver (`PROJECT`/`OUTDIR`/`RUN_TIMEOUT`) |
| `x-heep/dc/synth_area.tcl` | env-driven L/POLICY/MAXN/MAXB, per-run suffixes, report-parsed area, `dcnxt_shell` |
| `x-heep/dc/run_area.sh` | 10-run matrix (FPU L-sweep + 3 policies + accel logic) + summary; `DCSH` override |
| `x-heep/dc/rtl_accel_is.f` | **new** — accel filelist that black-boxes `xbuf_ram` for logic-only area |
| `x-heep/dc/reports/area_*_L[0-5]*.rpt`, `*_P[0-2]*.rpt`, `*accel_is*.rpt` | **new** — DC-NXT area reports |
| `PERF_SWEEP_REPORT.md`, `LENET_SWEEP_REPORT.md`, `AREA_REPORT.md`, `COMPARISON_vs_tanase2026.md` | **new** — self-contained analysis reports |
| `README.md` | key-results table updated (area, batched, LeNet, policy); repo map + report pointers |

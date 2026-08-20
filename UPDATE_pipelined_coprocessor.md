# Update — Pipelined single coprocessor (latency hidden by ONE unit; the dual is retired)

This update replaces the *serial + dual* framing with a single **pipelined** coprocessor. The earlier
serial accelerator issued one MAC and **waited** `1+L` cycles for its result before issuing the next, so
its throughput was `cyc/MAC = 1.14 + L` — the FMA's pipeline latency `L` was fully exposed. A second
coprocessor was needed only to fill that idle. The new coprocessor **issues the B independent batch-MACs
of a weight back-to-back** and collects results as they retire, keeping the shared FMA full, so
`cyc/MAC` is **flat across all L** — one unit does what the dual did, at half the hardware. Still **no
second FPU**. Two self-contained reports capture every number: `PIPE_BENCH_REPORT.md` (toy MLP),
`PIPE_LENET_REPORT.md` (real LeNet-300-100).

---

## 1. What changed in the hardware

New module **`tb/dma_fp_dot_accel_pipe.sv`** — a drop-in twin of `dma_fp_dot_accel_is.sv` (identical
ports/protocol) with a **pipelined** compute engine:

- **Decoupled issue/collect.** An `iss_b` pointer issues one MAC per granted cycle across the B batch
  lanes; a `col_b` pointer writes each returning result to its lane. The shared FMA may hold up to
  `L+1` of the accelerator's MACs in flight at once (the arbiter already round-trips a 2-bit owner tag,
  so this is out-of-order safe — no arbiter change was needed).
- **Batch-interleave hides latency.** For a fixed weight the B lanes are independent, so they are issued
  back-to-back; a lane is re-issued only every B cycles, by which time its previous result has retired
  (as long as `B ≥ L+1`). A hazard interlock `inflight < B` stalls otherwise (the true pipeline limit).
- **Bit-exact preserved.** The interleaving is *across* lanes; each dot-product's summation order
  (k = 0..N−1) is unchanged, so the result is bit-identical to the serial unit and to the CPU. At
  runtime `B=1` the interlock serializes issue, reproducing the old serial behaviour exactly.

`tb/dma_fp_dot_accel_is.sv` (serial) is **kept as the baseline** and the `GEMV_ONLY` parameter is removed
(plain batched GEMM). `testharness.sv.tpl` gains a **`COPROC_PIPE`** knob that selects the pipelined vs
serial coprocessor for acc0; acc1 is left instantiated but **idle** for the single-coprocessor runs
(never programmed, so it consumes no cycles). New benchmarks `sw/applications/perf_bench_pipe/` and
`perf_lenet_pipe/` are the single-coprocessor (no-dual) versions of `perf_bench` / `perf_lenet`.

## 2. Latency is hidden — `cyc/MAC` flat across L

Full L = 0..5 × policy sweeps, **21 configs each, all bit-exact** (LeNet also **8/8 MNIST correct**):

| `cyc/MAC` (single coprocessor, B = 8) | L=0 | L=1 | L=2 | L=3 | L=4 | L=5 |
|---|---:|---:|---:|---:|---:|---:|
| serial (`dma_fp_dot_accel_is`) | 1.14 | 2.14 | 3.14 | 4.14 | 5.14 | 6.14 |
| **pipelined (`dma_fp_dot_accel_pipe`), toy MLP** | **1.14** | **1.14** | **1.14** | **1.14** | **1.14** | **1.14** |
| **pipelined, real LeNet-300-100** | **1.12** | **1.12** | **1.12** | **1.12** | **1.12** | **1.12** |

The pipelined compute cycles are essentially constant (toy MLP 98 462→98 917, +0.5 %; LeNet
2 400 642→2 402 427, +0.07 % across the whole L range). FMA utilisation holds **~87 % (toy) / ~88 %
(LeNet)** at every latency.

## 3. One pipelined unit ≥ two serial units

| At L = 5 | toy MLP | LeNet-300-100 |
|---|---:|---:|
| pipelined single vs **serial single** | **4.22×** | **5.18×** |
| pipelined single vs **serial DUAL** (2 coprocessors), compute | **2.69×** | **2.72×** |
| speedup vs CPU (held flat across L; serial collapsed) | **5.30×** (serial 1.25×) | **8.34×** (serial 1.60×) |

At L = 0 (no latency to hide) the pipelined single equals the serial single, and the serial **dual**
edges it by ~10 % on pure compute (two memory channels). Everywhere else **one pipelined coprocessor
beats two serial ones**, at half the coprocessors and half the memory bandwidth — so the dual design is
**retired** as the headline. Contention with a concurrent CPU FP FIR stays **~3 % (toy) / ~0 % (LeNet)**;
arbiter policy remains a weak lever for these workloads.

## 4. Area (staged)

`dc/rtl_accel_pipe.f` + a `dma_fp_dot_accel_pipe` target in `dc/synth_area.tcl`, run by `dc/run_area.sh`
(which now synthesises **both** the pipelined and the serial accelerator, LOGIC-only with the buffer
black-boxed, and prints the pipelining area cost in µm² and GE). The pipelined engine adds a small amount
of control logic (the `inflight`/`col_b` counters and the decoupled issue/collect) over the serial unit;
the number is filled in once the Design Compiler run completes on the synthesis server.

---

## Files

| File | What |
|---|---|
| `x-heep/tb/dma_fp_dot_accel_pipe.sv` | **new** — pipelined batched-GEMM coprocessor |
| `x-heep/tb/dma_fp_dot_accel_is.sv` | serial baseline; `GEMV_ONLY` removed (plain batched GEMM) |
| `x-heep/tb/testharness.sv.tpl` | `COPROC_PIPE` knob (pipelined ↔ serial); acc1 left idle |
| `x-heep/tb/x-heep-tb-utils.core` | pipelined accelerator added to the fileset |
| `x-heep/sw/applications/perf_bench_pipe/` | **new** — single-coprocessor toy-MLP benchmark |
| `x-heep/sw/applications/perf_lenet_pipe/` | **new** — single-coprocessor real-LeNet benchmark |
| `x-heep/dc/rtl_accel_pipe.f`, `synth_area.tcl`, `run_area.sh` | area flow extended to the pipelined accel |
| `PIPE_BENCH_REPORT.md`, `PIPE_LENET_REPORT.md` | **new** — the full pipelined-design reports |

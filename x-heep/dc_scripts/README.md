# Area synthesis for the shared-FMA thesis (Design Compiler)

Goal: quantify the **"no second FPU"** claim — show the hardware we **add**
(the CPU-priority arbiter + the accelerator FSM) is far smaller than the
hardware we **avoid** (a second FMA, i.e. a second `fp_wrapper`).

Everything is module-level, so no full SoC is needed. Run from the **x-heep root**.

## Files
- `synth_area.tcl` — the flow (analyze → elaborate → compile → `report_area -hierarchy`).
- `rtl_core.f`  — `cv32e40px_top` subtree (CPU core + fpnew FMA + **our arbiter**), 152 files, packages first.
- `rtl_accel.f` — the accelerator (`dma_fp_dot_accel` + `fifo_pkg`).

## One-time setup
Edit the 3 USER SETTINGS at the top of `synth_area.tcl`:
- `LIB_DB`  → your standard-cell `.db` (any tech; use the same one for every run).
- `TOP`     → which module to synthesize (see runs below).
- `CLK_NS`  → clock period (keep the **same** value across runs for a fair compare).

## Runs (standalone modules — the faithful numbers)

Each of these tops has **all ports at the top level**, so DC keeps every gate
(nothing is optimised away). Set `TOP` in `synth_area.tcl` and run; read
`Total cell area` from the summary report.

```
# 1) the FMA we AVOID duplicating (a second FPU):
#    TOP = cv32e40px_fp_wrapper   -> dc/reports/area_cv32e40px_fp_wrapper.rpt
# 2) the arbiter we ADD:
#    TOP = dma_apu_arbiter        -> dc/reports/area_dma_apu_arbiter.rpt
# 3) the accelerator FSM we ADD:
#    TOP = dma_fp_dot_accel       -> dc/reports/area_dma_fp_dot_accel.rpt
dcnxt_shell -f dc/synth_area.tcl | tee dc/reports/log_<TOP>.txt
grep -i "Total cell area" dc/reports/area_<TOP>.rpt
```

**Why not just synthesize `cv32e40px_top`?** Its `FPU` parameter defaults to 0,
which drops the whole `fpu_gen` block (both the FMA *and* the arbiter). Forcing
`FPU=1` brings them back, but if the elaborated core doesn't actually drive the
FPU, DC may optimise parts of it away — so the standalone runs above are the
trustworthy areas. `TOP = cv32e40px_top` (with `FPU=1`) is still available if you
want the full-core total for percentage context.

## The claim
```
ADDED    = area(dma_apu_arbiter) + area(dma_fp_dot_accel)
AVOIDED  = area(cv32e40px_fp_wrapper)     # one full FMA (fpnew)
saving   = AVOIDED - ADDED                # expected: strongly positive
```
i.e. the coprocessor + arbiter cost a small fraction of one FMA, so sharing the
CPU's existing FMA is far cheaper than instantiating a dedicated one — with the
cycle/latency behaviour characterised in `thesis_progress.md` §11.

## Troubleshooting
- **`analyze` complains a package is undefined** → move that `*_pkg.sv` line up in
  `rtl_core.f` (packages are already hoisted to the top; a stray one can be nudged),
  or switch the read to `read_file -autoread -top $TOP` which auto-orders.
- **`compile_ultra` license error** → the script auto-falls back to `compile`.
- **`fp_wrapper_i` instance name differs** → grep the hier report for `fp_wrapper`
  (it lives under the `fpu_gen`/`genblk` generate scope) and for `arb_i`.
- Keep `LIB_DB` and `CLK_NS` identical across all runs so areas are comparable.

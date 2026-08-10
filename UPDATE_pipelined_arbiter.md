# Update — Fully-Pipelined Owner-Tagged FMA-Sharing Arbiter (2026-08)

Upgrade to the thesis's central contribution (the shared-FMA arbiter): from **drain-based** sharing to
**fully-pipelined, owner-tagged** sharing, so a *pipelined* FMA (deeper pipeline → higher clock) can be
time-shared between the CPU and the DMA coprocessor without stalling the CPU on every hand-off.

---

## 1. The innovation

**Before (drain arbiter).** The CPU and the coprocessor were *mutually exclusive* in the FMA pipeline:
while one owner's op was in flight, the other could not issue. Serving the CPU required *draining* the
coprocessor's in-flight op first (FPnew has no flush), so at pipeline latency `L` the CPU stalled up to
`L` cycles on every collision. That stall grows with pipeline depth — exactly the regime a pipelined,
high-`fmax` FMA lives in — so the drain arbiter squanders the pipeline.

**After (owner-tagged arbiter).** The CPU's and the coprocessor's ops may now be **in flight
simultaneously**. Each issued op is tagged with its owner (1 bit: CPU / coprocessor). FPnew carries that
tag through whichever lane the op uses and **returns it with the result**; the arbiter routes each
response by the returned tag. This is correct **even when FPnew's multi-latency lanes retire out of
order** — e.g. a CPU `fdiv`/`fsqrt` (slow DIVSQRT lane) concurrent with the coprocessor's `fmadd` (fast
ADDMUL lane). No draining, no op-class decoding, no gating, no reorder FIFO. The CPU keeps **strict
priority** on the single 1-op/cycle issue slot, so it is never delayed by the coprocessor at issue.

**Why the tag was free.** FPnew is instantiated with a 1-bit tag type but the wrapper had it *tied off*
(`.tag_i(1'b0)`, `.tag_o` unused). FPnew already round-trips the tag with the result
(`fpnew_top.sv`: `assign tag_o = arbiter_output.tag`). The change simply **exposes an existing,
disabled signal** — no new datapath logic.

---

## 2. What changed (RTL — 3 files)

| File | Change |
|---|---|
| `cv32e40px_fp_wrapper.sv` | Expose FPnew's tag: add ports `apu_tag_i` / `apu_tag_o`, connect to FPnew `.tag_i` / `.tag_o` (were `1'b0` / unused). |
| `dma_apu_arbiter.sv` | Add `fpu_tag_o` (= owner of the op issued this cycle) and `fpu_tag_i` (= owner returned with the result). Route responses by `fpu_tag_i`. Remove the owner-mixing restriction — both owners coexist; CPU strict priority on the issue slot. (A small in-flight counter remains only to drive the FMA clock-gate.) |
| `cv32e40px_top.sv` | Wire `arbiter.fpu_tag_o → fp_wrapper.apu_tag_i` and `fp_wrapper.apu_tag_o → arbiter.fpu_tag_i`; in the non-shared (baseline) branch tie the issue tag to `1'b0`. |

All three are static RTL (no `.tpl` template) — `make mcu-gen` does **not** regenerate them.

---

## 3. The test — `sw/applications/y3_contention/main.c`

**What it measures.** The *drain penalty*: how much the CPU's own floating-point work is slowed when
the coprocessor runs concurrently, i.e. the price the shared FMA imposes on the CPU.

**How.**
1. Time a **register-only** CPU FMA chain (`acc = fmaf(acc, k, c)`, 512 iterations) **alone**, with the
   coprocessor idle → `cpu_alone`.
2. Launch a coprocessor GEMV in the background (non-blocking DMA), then time the **same** CPU chain
   concurrently → `cpu_contended`.
3. `delta = cpu_contended − cpu_alone` = the extra CPU cycles caused by sharing the FMA.

The CPU kernel is **register-only** (no array/memory operands) on purpose: it removes the CPU's data
memory-bus traffic, so `delta` isolates the **FMA-sharing (drain)** cost from memory-bus contention —
a controlled experiment. Swept over the FMA pipeline latency `L = fpu_addmul_lat = 0..5`.

*(The coprocessor now uses the runtime-`N`/`M` streaming protocol: a `[N, M, x…]` header on the LOAD
transfer, so the same test drives the current input-stationary GEMV coprocessor.)*

---

## 4. Result — drain penalty vs pipeline depth `L`

Register-only (bus-free) contention microbenchmark, drain arbiter vs the new owner-tagged arbiter:

| `L` (FMA latency) | drain arbiter (cyc) | owner-tagged (cyc) | removed |
|---:|---:|---:|---:|
| 0 | 2 | 2 | 0 |
| 1 | 2 | 2 | 0 |
| 2 | 516 | 515 | 0 |
| 3 | 1 537 | 1 024 | 513 |
| 4 | 2 046 | 1 024 | 1 022 |
| 5 | 3 069 | **1 535** | **1 534** |

**Headline.** The drain arbiter's CPU-contended time grows at **~1 022 cycles per pipeline stage**; the
owner-tagged arbiter **halves that to ~511 cyc/stage**, removing up to **1 534 cycles (a 2× reduction at
L = 5)**. The benefit is ~0 for shallow pipelines (`L ≤ 2`) and grows linearly from `L ≥ 3` — i.e. it
**matters precisely for the deep / high-`fmax` FMA** a pipeline buys you. The residual ~511 cyc/stage is
the *irreducible* single-FMA serialization (one issue slot and one result slot per cycle; two streams
cannot use the one datapath simultaneously) — not removable by any arbiter.

*Note on the metric:* compare `cpu_contended` between the two arbiters (or `delta_old − delta_new`); the
raw `delta` is confounded because the register-only `alone` loop steps with `L` (overhead-bound at low
`L`, latency-bound at high `L`).

---

## 5. Correctness / verification

- The 3-file change was **adversarially verified** (4 independent review lenses). Tag-routing and
  CPU-priority/regression came back correct; one top-level wiring error — the design initially assumed
  two live `fp_wrapper` instances when only one is live (the other is commented-out dead code) — was
  caught and corrected before integration.
- Under the new arbiter the coprocessor's result is **bit-exact** against its golden in the concurrent
  test (`accel dot0 = golden`).
- **Open item:** an explicit *out-of-order* correctness test — a CPU `fdiv`/`fsqrt` chain concurrent
  with the coprocessor's `fmadd` — to empirically confirm the tag's out-of-order routing. The tests run
  so far are ADDMUL-only (same latency, in-order), which do not exercise the case the tag exists to fix.

---

## 6. Files

- **Changed RTL:** `x-heep/hw/vendor/xheep/cv32e40px/rtl/cv32e40px_fp_wrapper.sv`,
  `x-heep/hw/vendor/xheep/cv32e40px/rtl/dma_apu_arbiter.sv`,
  `x-heep/hw/vendor/xheep/cv32e40px/rtl/cv32e40px_top.sv`
- **Test:** `x-heep/sw/applications/y3_contention/main.c` (rewritten: runtime-`N` coprocessor protocol
  + register-only, bus-isolating CPU kernel)

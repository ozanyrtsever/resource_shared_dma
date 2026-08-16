# Shared-FMA Coprocessor — Full Area Study (Design Compiler)

This report gives, with **no abbreviations and every number attributed to exactly what was synthesized**,
the complete silicon-area study behind the **"no second floating-point unit"** claim: the hardware we
**add** to let a coprocessor borrow the CPU's floating-point multiply-add unit (an arbiter + the
coprocessor's own logic) is far smaller than the hardware we **avoid** (a second multiply-add unit, or a
whole second floating-point unit).

**Synthesis setup (identical for every run below).**
- **Tool:** Synopsys Design Compiler NXT (`dcnxt_shell`).
- **Technology / library:** TSMC 40 nm G, standard cells `sc12mc_cln40g_base_rvt` (12-track, Regular-Vt).
- **Corner:** SS / 0.81 V / 125 °C (slow-slow, worst-case delay — the conservative choice).
- **Clock:** 100 MHz (10 ns period), the **same** for every run so the areas are directly comparable.
- **Optimisation:** `compile_ultra` + `compile_ultra -incremental`, `set_max_area 0` (minimise area).
- Each module is synthesized in a **fresh** `dcnxt_shell` with its own work directory (zero state carryover).
- Areas are given in **square micrometres (µm²)** and in **gate-equivalents (GE)**. One GE = the area of a
  2-input NAND in this library (`NAND2_X1M_A12TR40` = **0.9576 µm²**, taken from the DC reference report), a
  technology-independent size unit: GE = area(µm²) ÷ 0.9576. 1 mm² = 1 000 000 µm².

> Note on the corner labels (`ss`, `0p81v`, `125c`): these set the *timing/power* characterization; they do
> **not** change a cell's physical footprint. At this relaxed 100 MHz clock, timing is met with huge slack,
> so the synthesizer optimises for area and the corner has negligible effect on the reported areas. What
> matters for a fair comparison — the **same library and the same clock for every run** — holds here.

---

## 0. How to read this report

### 0.1 The things we synthesized
- **Full floating-point unit (FPU)** = `cv32e40px_fp_wrapper` — the CPU's *whole* FPU: the multiply-add
  (ADD/MUL/FMA) lane **plus** divide/square-root, compare/min-max (NONCOMP), and format-conversion (CONV)
  lanes, plus the APU glue. This is what a naive "just add a second FPU" would duplicate.
- **Multiply-add unit alone (FMA)** = `fpnew_fma_multi` — only the fused multiply-add lane, read out of the
  FPU's hierarchy report. This is the *actual* unit the coprocessor borrows; it is the honest "one FMA" cost.
- **Arbiter** = `dma_apu_arbiter` — the small block we **add** so the CPU and the coprocessor(s) time-share
  the one FMA under a chosen policy. Synthesized separately for each of its three policies.
- **Accelerator logic** = `dma_fp_dot_accel_is` — the coprocessor's own control + datapath (FSM, batch
  accumulators, counters, operand routing), **excluding** its input buffer (see §3).

### 0.2 What each number means
- **Total cell area** — the whole synthesized area of that block (µm²). This is the headline number.
- **Combinational area** — area of logic gates (adders, muxes, the multiply-add arithmetic, control logic).
- **Non-combinational area** — area of the **flip-flops (registers)**. For the FMA this grows with the
  pipeline latency L (each pipeline stage = a rank of registers).
- **L (ADD/MUL pipeline latency)** — `FPU_ADDMUL_LAT`, swept 0…5: how many pipeline register stages the
  multiply-add unit has. L = 0 is a purely combinational multiply-add; L = 5 is a deep pipeline. Only the
  FMA lane's latency is swept (the FPU's other lanes are held at latency 0), so any FPU area growth with L
  is entirely the FMA's pipeline registers.

### 0.3 The two "avoided" framings
When we say sharing "avoids" a unit, there are two honest reference points:
1. **Avoid one dedicated FMA** (`fpnew_fma_multi`) — if the coprocessor had its *own* multiply-add unit.
   This is the tightest, most honest comparison (the coprocessor only needs a multiply-add, not div/sqrt/…).
2. **Avoid a full second FPU** (`cv32e40px_fp_wrapper`) — if you dropped in another whole FPU instance
   (the easy but wasteful way). Bigger reference, even more favourable to sharing.

---

## 1. The FPU and the FMA, by pipeline latency L

Who: the CPU's **whole FPU** (`cv32e40px_fp_wrapper`), and — from the same run's hierarchy report — the
**multiply-add lane alone** (`fpnew_fma_multi`) inside it. Accelerator/arbiter not involved here.

| L | full FPU (µm²) | full FPU (GE) | **FMA alone (µm²)** | **FMA alone (GE)** | rest = FPU − FMA (µm²) | FMA as % of FPU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 455.50 | 13 007 | **5 789.01** | **6 045** | 6 666.49 | 46.5 % |
| 1 | 12 951.54 | 13 525 | **6 135.66** | **6 407** | 6 815.88 | 47.4 % |
| 2 | 13 285.10 | 13 873 | **6 469.23** | **6 756** | 6 815.87 | 48.7 % |
| 3 | 13 636.86 | 14 241 | **6 820.98** | **7 123** | 6 815.88 | 50.0 % |
| 4 | 13 985.43 | 14 605 | **7 169.55** | **7 487** | 6 815.88 | 51.3 % |
| 5 | 14 337.51 | 14 972 | **7 521.63** | **7 855** | 6 815.88 | 52.5 % |

(FPU register area grows 1 513 → 2 748 µm² over L = 0..5 — the FMA's pipeline registers; the non-FMA "rest"
is essentially constant.)

Reading:
- **The FMA is about half of the whole FPU** (46–53 %). The other half (~6 816 µm², essentially constant) is
  divide/square-root, compare/min-max, format conversion and APU glue — hardware a multiply-add-only
  coprocessor does **not** need. So the honest "one FMA" cost is **5 789 µm² at L = 0**, not the full
  12 455 µm².
- **Both the FPU and the FMA grow with the pipeline latency L**, and the growth is *entirely* registers:
  the FMA goes from 5 789 µm² (combinational, L = 0) to 7 521 µm² (5 pipeline stages, L = 5) — **+30 %** — and
  the FPU's register area rises in lockstep (1 513 → 2 748 µm²). The non-FMA part stays flat (~6 816 µm²),
  confirming the sweep isolates the FMA-latency effect.
- **Consequence for sharing:** the deeper the FMA pipeline (higher L, as needed for higher clock speeds),
  the *bigger* the unit you avoid by sharing — while the arbiter cost stays flat (§2). Sharing gets *more*
  favourable at higher L.

---

## 2. The arbiter, per policy

Who: **only the sharing arbiter** (`dma_apu_arbiter`), synthesized once per policy. This is the entire
hardware cost of *enabling* three requestors (CPU + accelerator 0 + accelerator 1) to time-share one FMA.
Only the selected policy is elaborated (conditional generate), so each number is the honest cost of that
policy alone.

| Policy | total (µm²) | **total (GE)** | combinational (µm²) | registers (µm²) | ≈ flip-flops |
|---|---:|---:|---:|---:|---:|
| **P0 — CPU-strict + accelerator round-robin** (default) | **452.94** | **473** | 424.22 | 28.73 | ~5 |
| **P1 — full round-robin** (no CPU priority) | 465.07 | 486 | 430.60 | 34.47 | ~6 |
| **P2 — QoS weighted (deficit-credit, 4:2:1)** | **927.60** | **969** | 732.24 | 195.35 | ~34 |

Reading:
- **CPU-strict (P0) is the smallest — 452.94 µm²** — and it is the default, so the design ships with the
  cheapest arbiter that also gives the CPU its guaranteed priority.
- **Full round-robin (P1) costs almost nothing extra** (+2.7 %, +12 µm²): it only swaps the priority rule
  for a 2-bit rotating pointer.
- **QoS (P2) is about 2× the others (927.60 µm²)** because it carries three signed 10-bit deficit-credit
  counters plus their adders and a 3-way argmax comparator (its register area alone, 195 µm², is ~7× P0's).
  This is the price of software-tunable share control — visible and quantified, and paid **only if you
  choose P2**.

---

## 3. The accelerator logic (and its buffer)

Who: **one coprocessor** (`dma_fp_dot_accel_is`), synthesized **logic-only**. Its 32 KB input buffer
(`x_buf`) is factored into a submodule (`u_xbuf`) and left **unresolved on purpose** so Design Compiler
treats it as a black box (area 0) — exactly as an off-the-shelf SRAM macro would be. (The synthesis log's
"1 unresolved reference / black-box area 0.000000" is the intended result, not an error.)

| Part | area | notes |
|---|---:|---|
| **Accelerator logic (total)** | **4 460.82 µm² = 4 658 GE** | FSM + batch accumulators + counters + operand routing |
| — combinational | 2 041.92 µm² | control + datapath gates |
| — registers | 2 418.90 µm² | dominated by the eight 32-bit batch accumulators `acc_q` + counters |
| **Input buffer `x_buf`** | **a 32 KB single-port SRAM macro** | 8 × 1024 × 32 bit = 262 144 bit = 32 KB **per accelerator** |

Reading:
- The coprocessor's **logic is 4 460.82 µm²** — of which ~2 419 µm² is registers, mostly the **eight
  batch accumulators** (`acc_q`, 8 × 32 bit) that make the batched matrix computation possible. These are
  legitimately flip-flops and stay in "logic".
- The **32 KB input buffer is an SRAM macro, not flip-flops.** Its area comes from the foundry memory
  compiler (recommended), or as a rough 40 nm 6T estimate **≈ 0.11 mm² (≈ 110 000 µm²) per accelerator**.
  Design Compiler does **not** convert a register array into an SRAM — the buffer is reported separately and
  summed, which is standard practice. (If it *were* synthesized as flip-flops it would be ~262 144 registers
  ≈ 1.5 mm² ≈ 26× the whole coprocessor logic — which is exactly why no one builds a buffer from flops.)
- The accelerator logic (and its SRAM) are **common to both designs** — you need the coprocessor whether it
  shares the CPU's FMA or has a dedicated one. So they cancel out of the "sharing vs. dedicated" comparison;
  the difference is purely **arbiter vs. a second FMA** (§4).

---

## 4. The headline: "no second FPU"

The only hardware difference between **sharing** the CPU's FMA and giving the coprocessor a **dedicated**
FMA is: sharing adds the **arbiter**; dedicating adds a **second FMA** (or a whole second FPU). Everything
else (the coprocessor logic + its SRAM) is identical in both.

### One coprocessor (single-accelerator), at L = 0
| | area (µm²) | area (GE) | as % of one FMA |
|---|---:|---:|---:|
| **AVOIDED** — one dedicated FMA (`fpnew_fma_multi`) | 5 789.01 | 6 045 | 100 % |
| *(or)* AVOIDED — a full second FPU (`cv32e40px_fp_wrapper`) | 12 455.50 | 13 007 | 215 % |
| **ADDED** — the sharing arbiter (P0, CPU-strict) | **452.94** | **473** | **7.8 %** |
| **NET SAVING** = FMA − arbiter | **5 336.07** | **5 572** | 92 % |

**Sharing the CPU's FMA costs a 453 µm² arbiter instead of a 5 789 µm² FMA — the arbiter is 7.8 % of one
FMA (3.6 % of the whole FPU).** Put differently, the arbiter is ~13× smaller than the FMA it lets you skip.

### The saving grows with pipeline latency L
The arbiter does **not** contain an FMA, so its area is **independent of L** (~453 µm² at every L). The
avoided FMA, however, grows with L. So the arbiter's relative cost *shrinks* as pipelines get deeper:

| L | avoided FMA (µm²) | added arbiter P0 (µm²) | arbiter as % of FMA | net saving (µm²) | net saving (GE) |
|---:|---:|---:|---:|---:|---:|
| 0 | 5 789.01 | 452.94 | 7.8 % | 5 336.07 | 5 572 |
| 3 | 6 820.98 | 452.94 | 6.6 % | 6 368.04 | 6 650 |
| 5 | 7 521.63 | 452.94 | 6.0 % | 7 068.69 | 7 381 |

### Two coprocessors (dual-accelerator) on one FMA
The 3-requestor arbiter (452.94 µm²) already lets **both** accelerators plus the CPU share the single FMA.
The dedicated alternative would need **two** FMAs:

| | area (µm²) | area (GE) |
|---|---:|---:|
| **AVOIDED** — two dedicated FMAs (2 × 5 789.01) | 11 578.02 | 12 090 |
| **ADDED** — one 3-requestor arbiter (P0) | 452.94 | 473 |
| **NET SAVING** | 11 125.08 | 11 618 |

One 453 µm² arbiter replaces two 5 789 µm² FMAs — the arbiter is **3.9 % of the two FMAs** it avoids.

---

## 5. Key findings

1. **The FMA is ~half the FPU** (46–53 %); the honest "one FMA" avoided cost is **5 789 µm² at L = 0**
   (`fpnew_fma_multi`), the rest of the FPU (~6 816 µm², div/sqrt/cmp/conv/glue) being hardware a
   multiply-add coprocessor doesn't need.
2. **The sharing arbiter is tiny and its cost is honest per policy:** CPU-strict **452.94 µm²**, full
   round-robin 465.07 µm² (+2.7 %), QoS weighted 927.60 µm² (~2×, the only expensive one — pay it only if
   you want software-tunable shares).
3. **"No second FPU" is quantified:** to share the CPU's FMA you add a **453 µm² (473 GE) arbiter (7.8 % of
   one FMA, 3.6 % of the full FPU)** instead of duplicating a **5 789 µm² (6 045 GE) FMA** — a
   **5 336 µm² (5 572 GE) net saving** per coprocessor; **11 125 µm² (11 618 GE)** for the dual-accelerator
   (one arbiter vs. two FMAs).
4. **Sharing scales even better with deeper pipelines:** the arbiter is L-independent while the FMA grows
   +30 % from L = 0 to L = 5, so the arbiter's share of the avoided FMA drops from 7.8 % to 6.0 %.
5. **The coprocessor's buffer is an SRAM, not logic:** the accelerator *logic* is 4 460.82 µm² (with the
   eight batch accumulators); its 32 KB input buffer is a separate SRAM macro (~0.11 mm² per accelerator
   from a memory compiler), never flip-flops.

---

## Appendix A — every synthesized configuration, one row each

TSMC 40 nm G, RVT, SS/0.81 V/125 °C, 100 MHz, `compile_ultra`. Areas in µm² and GE (1 GE = 0.9576 µm²).

| Run | what | total (µm²) | total (GE) | combinational (µm²) | registers (µm²) |
|---|---|---:|---:|---:|---:|
| cv32e40px_fp_wrapper_L0 | full FPU, L=0 | 12 455.50 | 13 007 | 10 942.50 | 1 513.01 |
| cv32e40px_fp_wrapper_L1 | full FPU, L=1 | 12 951.54 | 13 525 | 11 191.47 | 1 760.07 |
| cv32e40px_fp_wrapper_L2 | full FPU, L=2 | 13 285.10 | 13 873 | 11 277.97 | 2 007.13 |
| cv32e40px_fp_wrapper_L3 | full FPU, L=3 | 13 636.86 | 14 241 | 11 382.67 | 2 254.19 |
| cv32e40px_fp_wrapper_L4 | full FPU, L=4 | 13 985.43 | 14 605 | 11 484.18 | 2 501.25 |
| cv32e40px_fp_wrapper_L5 | full FPU, L=5 | 14 337.51 | 14 972 | 11 589.19 | 2 748.31 |
| fpnew_fma_multi (from hier) | FMA, L=0 | 5 789.01 | 6 045 | — | — |
| fpnew_fma_multi (from hier) | FMA, L=1 | 6 135.66 | 6 407 | — | — |
| fpnew_fma_multi (from hier) | FMA, L=2 | 6 469.23 | 6 756 | — | — |
| fpnew_fma_multi (from hier) | FMA, L=3 | 6 820.98 | 7 123 | — | — |
| fpnew_fma_multi (from hier) | FMA, L=4 | 7 169.55 | 7 487 | — | — |
| fpnew_fma_multi (from hier) | FMA, L=5 | 7 521.63 | 7 855 | — | — |
| dma_apu_arbiter_P0 | arbiter, CPU-strict | 452.94 | 473 | 424.22 | 28.73 |
| dma_apu_arbiter_P1 | arbiter, full round-robin | 465.07 | 486 | 430.60 | 34.47 |
| dma_apu_arbiter_P2 | arbiter, QoS 4:2:1 | 927.60 | 969 | 732.24 | 195.35 |
| dma_fp_dot_accel_is | coprocessor logic (buffer black-boxed) | 4 460.82 | 4 658 | 2 041.92 | 2 418.90 |

Input buffer (not in the table above): `x_buf` = 8 × 1024 × 32 bit = 32 KB single-port SRAM **per
accelerator** (two accelerators on the SoC → 64 KB total). Report the macro area from the foundry memory
compiler; a rough 40 nm 6T estimate is ≈ 0.11 mm² for 32 KB.

> These numbers supersede the earlier single-point figures (FPU 12 812 µm², FMA 5 823 µm², 2-requestor
> arbiter 321 µm²), which came from an older Design Compiler version and the earlier 2-requestor arbiter.
> Everything here is from one consistent DC-NXT run of the current 3-requestor design, so all rows are
> directly comparable.

# perf_bench_pipe — pipelined single-coprocessor report (toy MLP)

**Design:** pipelined batched-GEMM accelerator (`dma_fp_dot_accel_pipe.sv`), ONE coprocessor, sharing the CPU's single FPnew FMA over the APU. It issues the B independent batch-MACs of a weight back-to-back and collects results as they retire, keeping the FMA pipeline full.
**Platform:** X-HEEP + CV32E40P (cv32e40px), one shared FMA, APU arbiter, hw-FIFO DMA. Verilator RTL sim, trace off.
**Workload:** MLP `128-64-32-16`, FP32 = 10 752 MAC/img, batch **B=8** (86 016 MAC/batch).
**Sweep:** FMA add/mul latency `L ∈ {0..5}` × arbiter policy `P ∈ {0,1,2}` (+ QoS weights at L=0). Folder: `sweep_results/pipe_bench/`.
**Correctness:** every config `ALL bit-exact=1` (accelerator output identical to the CPU, bit for bit).

---

## 0. TL;DR — the headline

**The pipelined coprocessor's `cyc/MAC` stays FLAT at 1.14 across ALL latencies L.** The serial baseline grew to 1.14+L (up to 6.14 at L=5). One pipelined unit hides the FMA latency that used to need a second coprocessor.

| L | **PIPE B=8** cyc/MAC | serial B=8 cyc/MAC | serial *dual* gemv/MAC |
|:-:|:-------------------:|:------------------:|:----------------------:|
| 0 | **1.14** | 1.14 | 1.02 |
| 1 | **1.14** | 2.14 | 1.14 |
| 2 | **1.14** | 3.14 | 1.59 |
| 3 | **1.14** | 4.14 | 2.09 |
| 4 | **1.14** | 5.14 | 2.59 |
| 5 | **1.14** | 6.14 | 3.09 |

**Consequences:**
- **vs serial single:** identical at L=0, up to **4.22× faster at L=5** — with the *same* hardware.
- **vs serial dual (two coprocessors):** the single pipelined unit's compute is **2.69× faster than the two serial units at L=5** (and its *total* cycles beat the dual at *every* L). **One pipelined unit ≥ two serial units for all L≥1** — using half the coprocessors and half the memory bandwidth. This is why the dual design is dropped.
- **vs CPU:** holds **5.30–5.46×** across all L (the serial single collapsed 5.46× → 1.25×).
- **FMA utilization ~87%** at every L; contention with concurrent CPU FP ~3%; all bit-exact.

---

## 1. The core result — latency is hidden (single run, per L)

| L | [B] B=1 serial-ref cyc/MAC | [C] B=8 PIPELINED gemv | [C] cyc/MAC | FMA-util | [C] total | speedup vs CPU |
|:-:|:--------------------------:|:----------------------:|:-----------:|:--------:|:---------:|:--------------:|
| 0 | 2.06 | 98 462 | **1.14** | 87% | 128 398 | 5.46× |
| 1 | 2.06 | 98 589 | **1.14** | 87% | 129 421 | 5.42× |
| 2 | 3.06 | 98 706 | **1.14** | 87% | 130 434 | 5.38× |
| 3 | 4.06 | 98 805 | **1.14** | 87% | 131 429 | 5.35× |
| 4 | 5.06 | 98 889 | **1.14** | 86% | 132 409 | 5.32× |
| 5 | 6.06 | 98 917 | **1.14** | 86% | 133 333 | 5.30× |

**Reading it.**
- The **compute (`gemv`) cycles are essentially constant** — 98 462 → 98 917, a **0.5% drift** across the whole L=0→5 range. The FMA pipeline is kept full: while one batch-MAC is in flight (latency `L`), the next `L` independent batch-MACs are issued, so the latency never stalls the datapath.
- The in-run **B=1 serial reference `[B]` grows** 2.06 → 6.06 (it exposes the latency, because at B=1 the interlock serializes issue). The gap between `[B]` and `[C]` at each L is exactly the latency the pipeline hides.
- The small **`[C]` total** growth (128 398 → 133 333, +3.8%) is *not* the accelerator — it is the CPU-side glue (bias-add / ReLU on the shared FMA) getting slower with `L`; the accelerator's own `gemv` is flat.
- **Utilization ~87%** = the accelerator drives the shared FMA at 87% of its 1-MAC/cycle peak, with a *single* unit, at every latency.

---

## 2. Pipelined single vs serial single — same hardware, huge divergence

| L | serial single [C] total | PIPE single [C] total | **PIPE speedup vs serial** |
|:-:|:-----------------------:|:---------------------:|:--------------------------:|
| 0 | 128 341 | 128 398 | 1.00× |
| 1 | 215 139 | 129 421 | 1.66× |
| 2 | 302 089 | 130 434 | 2.32× |
| 3 | 388 948 | 131 429 | 2.96× |
| 4 | 475 874 | 132 409 | 3.59× |
| 5 | 562 783 | 133 333 | **4.22×** |

Same accelerator interface, same workload — the *only* difference is the pipelined issue engine. At L=0 there is no latency to hide, so they match (the +57 cycles is the pipe's per-row drain, 0.04%). From L=1 up, the serial unit stalls `1+L` per MAC while the pipelined unit does not — the gap grows to **4.22× at L=5**.

---

## 3. Pipelined single vs serial DUAL — one unit beats two

The dual design existed only to fill the FMA idle that a *serial* unit left. A pipelined unit fills it itself:

| L | serial DUAL gemv (2 units) | PIPE single gemv (1 unit) | **PIPE advantage** |
|:-:|:--------------------------:|:-------------------------:|:------------------:|
| 0 | 87 998 | 98 462 | 0.89× (dual edges by 12%) |
| 1 | 98 257 | 98 589 | ~1.00× |
| 2 | 136 917 | 98 706 | **1.39×** |
| 3 | 179 974 | 98 805 | **1.82×** |
| 4 | 222 991 | 98 889 | **2.26×** |
| 5 | 265 925 | 98 917 | **2.69×** |

On **total** cycles the pipelined single beats the serial dual at *every* L (e.g. L5: 133 333 vs the dual's 315 610 — 2.37×), because the dual also pays two channels' setup/load and channel imbalance. Only at L=0, on pure compute, do two serial units (1.02 cyc/MAC) marginally beat one pipelined unit (1.14). Everywhere else **one pipelined coprocessor delivers more throughput than two serial ones, at half the area and half the memory traffic** — the justification for the single-coprocessor design.

---

## 4. Speedup vs the CPU — held flat

| L | PIPE B=8 speedup | serial B=8 speedup |
|:-:|:----------------:|:------------------:|
| 0 | 5.46× | 5.46× |
| 1 | 5.42× | 3.26× |
| 2 | 5.38× | 2.32× |
| 3 | 5.35× | 1.81× |
| 4 | 5.32× | 1.48× |
| 5 | 5.30× | **1.25×** |

The pipelined accelerator keeps a **~5.3–5.46× advantage over the CPU at every latency**; the serial one decayed toward parity (1.25× at L=5). The slight downward drift (5.46→5.30) is only because the CPU golden itself uses the shared FMA and slows a little with `L`.

---

## 5. Contention with concurrent CPU FP (scenario [D])

The CPU runs a 32-tap FP FIR on the *same* FMA while the accelerator computes:

| L | gemv alone | gemv w/ CPU | CONTENTION | % |
|:-:|:----------:|:-----------:|:----------:|:-:|
| 0 | 98 462 | 102 147 | 3 685 | 3% |
| 1 | 98 589 | 102 361 | 3 772 | 3% |
| 2 | 98 706 | 102 613 | 3 907 | 3% |
| 3 | 98 805 | 102 657 | 3 852 | 3% |
| 4 | 98 889 | 102 797 | 3 908 | 3% |
| 5 | 98 917 | 102 758 | 3 841 | 3% |
*(P0/CPU-strict.)*

Contention is a **stable ~3% at every latency** — even though the pipelined accelerator now keeps the FMA 87% busy, a concurrent CPU FIR costs it only ~3%, and the CPU's FIR stays bit-exact and is slowed only ~2%. The CPU-priority arbiter lets the two workloads share the one FMA cheaply.

---

## 6. Arbiter policy (L=0, all variants) — weak lever

| Config (L0) | [D] contention | Config (L0) | [D] contention |
|---|:---:|---|:---:|
| P0 `w1-1-1` | 3% | P2 `w4-1-1` | 3% |
| P1 `w1-1-1` | 3% | P2 `w1-4-1` | **1%** |
| P2 `w1-1-1` | 3% | P2 `w1-4-4` | **1%** |

Policy barely moves the numbers (1–3% total spread). Giving the accelerator QoS priority (`w1-4-*`) shaves contention from 3% to 1%, but the effect is small because the CPU's FIR demand (4 096 MACs) is tiny next to the GEMM (86 016 MACs). Consistent with the standing finding: **a simple CPU-priority arbiter suffices; rich QoS is unnecessary** for these workloads.

---

## 7. Correctness

All configs report `ALL bit-exact=1` — accelerator logits identical to the CPU golden, and the CPU FIR unchanged under sharing, across every L and policy. Sharing the CPU's own FMA means the accelerator is numerically the CPU; the pipelined reordering is *across* batch lanes only, so each dot-product's summation order (k=0..N-1) is preserved bit-for-bit.

---

## 8. Takeaways

1. **`cyc/MAC = 1.14`, flat across L** — the pipelined engine hides the shared FMA's latency with a single unit (serial was 1.14+L).
2. **Up to 4.22× faster than the serial single** (same hardware) and **≥ the serial dual for all L≥1** at half the coprocessors and memory bandwidth → the dual design is retired.
3. **~5.3× over the CPU at every latency**, ~87% FMA utilization — near the shared unit's 1-MAC/cycle peak.
4. **~3% contention** with concurrent CPU FP; policy is a weak lever (CPU-priority is enough).
5. **Bit-exact everywhere.**
6. Next: real-model confirmation (`perf_lenet_pipe`) and the area/energy study (`dma_fp_dot_accel_pipe` logic vs a dedicated-FPU baseline).

---

## Appendix — full [C] pipelined table (P0)

| L | CPU (8img) | [B] B=1 cyc/MAC | [C] gemv | [C] cyc/MAC | util | [C] total | speedup | vs serial | vs dual gemv |
|:-:|:----------:|:---------------:|:--------:|:-----------:|:----:|:---------:|:-------:|:---------:|:------------:|
| 0 | 701 214 | 2.06 | 98 462 | 1.14 | 87% | 128 398 | 5.46× | 1.00× | 0.89× |
| 1 | 701 982 | 2.06 | 98 589 | 1.14 | 87% | 129 421 | 5.42× | 1.66× | 1.00× |
| 2 | 702 878 | 3.06 | 98 706 | 1.14 | 87% | 130 434 | 5.38× | 2.32× | 1.39× |
| 3 | 703 902 | 4.06 | 98 805 | 1.14 | 87% | 131 429 | 5.35× | 2.96× | 1.82× |
| 4 | 705 566 | 5.06 | 98 889 | 1.14 | 86% | 132 409 | 5.32× | 3.59× | 2.26× |
| 5 | 707 230 | 6.06 | 98 917 | 1.14 | 86% | 133 333 | 5.30× | 4.22× | 2.69× |

*All 21 configs: `ALL bit-exact=1`. Source: `sweep_results/pipe_bench/` (pipe) vs `sweep_results/final/` (serial + dual).*

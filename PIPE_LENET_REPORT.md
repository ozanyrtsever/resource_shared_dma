# perf_lenet_pipe — pipelined single-coprocessor report (real LeNet-300-100 / MNIST)

**Design:** pipelined batched-GEMM accelerator (`dma_fp_dot_accel_pipe.sv`), ONE coprocessor, sharing the CPU's single FPnew FMA over the APU. Real trained **LeNet-300-100** weights, real **MNIST** test images.
**Platform:** X-HEEP + CV32E40P (cv32e40px), one shared FMA, APU arbiter, hw-FIFO DMA. Verilator RTL sim, trace off.
**Workload:** LeNet `784-300-100-10`, FP32 = **266 200 MAC/img**, batch **B=8** (2 129 600 MAC/batch).
**Sweep:** FMA add/mul latency `L ∈ {0..5}` × arbiter policy `P ∈ {0,1,2}` (+ QoS weights at L=0). Folder: `sweep_results/pipe_lenet/`. Serial baseline (batched GEMM, B=8): `sweep_results/lenet/`.
**Correctness:** every config `ALL bit-exact=1` (accelerator logits identical to the CPU) **and** `correct=8/8, match_ref=8/8` (predictions match the MNIST labels and the offline reference).

---

## 0. TL;DR — the real-model confirmation

The perf_bench result holds on a real network, **more strongly**: the pipelined coprocessor's `cyc/MAC` is **flat at 1.12 across all latencies L**, while the serial baseline grew to 1.12+L (6.12 at L=5).

| L | **PIPE B=8** cyc/MAC | serial B=8 cyc/MAC | serial *dual* gemv/MAC |
|:-:|:-------------------:|:------------------:|:----------------------:|
| 0 | **1.12** | 1.12 | 1.00 |
| 1 | **1.12** | 2.12 | 1.12 |
| 2 | **1.12** | 3.12 | 1.56 |
| 3 | **1.12** | 4.12 | 2.06 |
| 4 | **1.12** | 5.12 | 2.56 |
| 5 | **1.12** | 6.12 | 3.06 |

**Consequences (real model):**
- **vs serial single:** identical at L=0, up to **5.18× faster at L=5** — same hardware.
- **vs serial dual (two coprocessors):** the single pipelined unit's compute is **2.72× faster than two serial units at L=5**; one pipelined unit ≥ two serial units for all L≥1, at half the coprocessors and memory bandwidth.
- **vs CPU:** holds **8.34–8.39×** across all L (the serial single collapsed 8.34× → 1.60×). LeNet's larger size amortizes overhead better than the toy MLP (5.3×), so the shared-FMA advantage is bigger on the real model.
- **Accuracy 8/8** at every L — the real network runs correctly on the accelerator.
- **Contention ~0%** with concurrent CPU FP; **FMA utilization ~88%**; all bit-exact.

---

## 1. The core result — latency hidden on the real model

| L | [B] B=1 serial-ref cyc/MAC | [C] B=8 PIPELINED gemv | [C] cyc/MAC | FMA-util | [C] total | speedup vs CPU | acc |
|:-:|:--------------------------:|:----------------------:|:-----------:|:--------:|:---------:|:--------------:|:---:|
| 0 | 2.00 | 2 400 642 | **1.12** | 88% | 2 527 073 | 8.39× | 8/8 |
| 1 | 2.00 | 2 400 997 | **1.12** | 88% | 2 530 708 | 8.38× | 8/8 |
| 2 | 3.00 | 2 401 355 | **1.12** | 88% | 2 534 346 | 8.37× | 8/8 |
| 3 | 4.00 | 2 401 720 | **1.12** | 88% | 2 537 991 | 8.36× | 8/8 |
| 4 | 5.00 | 2 402 081 | **1.12** | 88% | 2 541 632 | 8.35× | 8/8 |
| 5 | 6.00 | 2 402 427 | **1.12** | 88% | 2 545 258 | 8.34× | 8/8 |

**The compute (`gemv`) cycles are essentially constant** — 2 400 642 → 2 402 427, a **0.07% drift** across the whole L=0→5 range. The pipeline stays full at every latency; `L` never stalls the datapath. The in-run B=1 serial reference `[B]` grows 2.00 → 6.00 (it exposes the latency the pipeline hides). FMA utilization holds **88%** — the single unit drives the shared FMA at 88% of its 1-MAC/cycle peak, at every latency.

---

## 2. Pipelined single vs serial single — same hardware

| L | serial single [C] total | PIPE single [C] total | **PIPE speedup vs serial** |
|:-:|:-----------------------:|:---------------------:|:--------------------------:|
| 0 | 2 529 812 | 2 527 073 | 1.00× |
| 1 | 4 662 806 | 2 530 708 | 1.84× |
| 2 | 6 795 477 | 2 534 346 | 2.68× |
| 3 | 8 928 425 | 2 537 991 | 3.52× |
| 4 | 11 061 333 | 2 541 632 | 4.35× |
| 5 | 13 194 581 | 2 545 258 | **5.18×** |

Identical at L=0 (no latency to hide). From L=1 the serial unit stalls `1+L` per MAC while the pipelined unit does not — the gap grows to **5.18× at L=5** on the real model, with the *same* accelerator interface.

---

## 3. Pipelined single vs serial DUAL — one unit beats two

| L | serial DUAL gemv (2 units) | PIPE single gemv (1 unit) | **PIPE advantage** |
|:-:|:--------------------------:|:-------------------------:|:------------------:|
| 0 | 2 130 537 | 2 400 642 | 0.89× (dual edges by 13%) |
| 1 | 2 397 142 | 2 400 997 | ~1.00× |
| 2 | 3 330 149 | 2 401 355 | **1.39×** |
| 3 | 4 395 072 | 2 401 720 | **1.83×** |
| 4 | 5 459 828 | 2 402 081 | **2.27×** |
| 5 | 6 524 595 | 2 402 427 | **2.72×** |

On **total** cycles the pipelined single beats the serial dual at every L≥1 as well (L5: 2 545 258 vs the dual's 6 739 959 — 2.65×). Only at L=0, on pure compute, do two serial units (1.00 cyc/MAC) marginally beat one pipelined unit (1.12). Everywhere else **one pipelined coprocessor delivers more throughput than two serial ones, at half the area and half the memory traffic** — the real-model justification for the single-coprocessor design.

---

## 4. Speedup vs the CPU — held flat

| L | PIPE B=8 speedup | serial B=8 speedup |
|:-:|:----------------:|:------------------:|
| 0 | 8.39× | 8.34× |
| 1 | 8.38× | 4.52× |
| 2 | 8.37× | 3.10× |
| 3 | 8.36× | 2.36× |
| 4 | 8.35× | 1.90× |
| 5 | 8.34× | **1.60×** |

The pipelined accelerator keeps a **~8.3–8.4× advantage over the CPU at every latency**; the serial one decayed toward parity (1.60× at L=5). The advantage is larger than the toy MLP's 5.3× because LeNet's 266 200 MAC/img amortizes the fixed setup/load overhead better.

---

## 5. Accuracy — the real network runs correctly

Every config: **`correct=8/8`** (predictions match the MNIST labels) and **`match_ref=8/8`** (predictions match the offline reference), at all L and all policies. Combined with `bit-exact=1`, this proves the accelerator does not merely produce plausible numbers — it runs the real LeNet-300-100 to the exact same logits and the same classifications as the CPU, at every FMA latency.

---

## 6. Contention with concurrent CPU FP (scenario [D])

| L | gemv alone | gemv w/ CPU | CONTENTION | % |
|:-:|:----------:|:-----------:|:----------:|:-:|
| 0 | 2 400 642 | 2 404 151 | 3 509 | 0% |
| 1 | 2 400 997 | 2 404 738 | 3 741 | 0% |
| 2 | 2 401 355 | 2 405 368 | 4 013 | 0% |
| 3 | 2 401 720 | 2 405 635 | 3 915 | 0% |
| 4 | 2 402 081 | 2 406 126 | 4 045 | 0% |
| 5 | 2 402 427 | 2 406 334 | 3 907 | 0% |

Contention is **~0% at every latency**: the CPU's 32-tap FIR (4 096 MACs) is negligible next to LeNet's 2.1 M-MAC batch, so a concurrent CPU FP job costs the accelerator essentially nothing (~4 000 cycles out of 2.4 M). Arbiter policy makes no meaningful difference here — the FMA is not contended by the tiny CPU job. (Where the CPU FP load is heavier, the perf_bench report shows the contention/policy behavior.)

---

## 7. Takeaways

1. **Real-model confirmation:** `cyc/MAC = 1.12`, flat across L — the pipelined engine hides the shared FMA's latency on LeNet-300-100 with a single unit (serial was 1.12+L).
2. **Up to 5.18× faster than the serial single** (same hardware) and **≥ the serial dual for all L≥1** (2.72× at L=5) at half the coprocessors/bandwidth → the dual design is retired.
3. **~8.3× over the CPU at every latency**, ~88% FMA utilization; the advantage exceeds the toy MLP's because the larger model amortizes overhead better.
4. **8/8 accuracy + bit-exact everywhere** — the accelerator is numerically and functionally the CPU.
5. **~0% contention** with concurrent CPU FP.
6. Next: the area/energy study (`dma_fp_dot_accel_pipe` logic vs a dedicated-FPU baseline) already staged in `dc/`.

---

## Appendix — full [C] pipelined table (P0)

| L | CPU (8img) | [C] gemv | cyc/MAC | util | [C] total | cyc/img | speedup | vs serial | vs dual gemv | acc |
|:-:|:----------:|:--------:|:-------:|:----:|:---------:|:-------:|:-------:|:---------:|:------------:|:---:|
| 0 | 21 219 682 | 2 400 642 | 1.12 | 88% | 2 527 073 | 315 884 | 8.39× | 1.00× | 0.89× | 8/8 |
| 1 | 21 222 962 | 2 400 997 | 1.12 | 88% | 2 530 708 | 316 339 | 8.38× | 1.84× | 1.00× | 8/8 |
| 2 | 21 226 242 | 2 401 355 | 1.12 | 88% | 2 534 346 | 316 793 | 8.37× | 2.68× | 1.39× | 8/8 |
| 3 | 21 229 522 | 2 401 720 | 1.12 | 88% | 2 537 991 | 317 249 | 8.36× | 3.52× | 1.83× | 8/8 |
| 4 | 21 232 882 | 2 402 081 | 1.12 | 88% | 2 541 632 | 317 704 | 8.35× | 4.35× | 2.27× | 8/8 |
| 5 | 21 239 362 | 2 402 427 | 1.12 | 88% | 2 545 258 | 318 157 | 8.34× | 5.18× | 2.72× | 8/8 |

*All 21 configs: `ALL bit-exact=1`, `correct=8/8`, `match_ref=8/8`. Source: `sweep_results/pipe_lenet/` (pipe) vs `sweep_results/lenet/` (serial batched GEMM + dual).*

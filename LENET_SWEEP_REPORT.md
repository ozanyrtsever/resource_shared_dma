# Shared-FMA Coprocessor — LeNet-300-100 / MNIST Performance Sweep (full)

Same benchmark and same six scenarios as the toy-MLP study (`PERF_SWEEP_REPORT.md`), but run on the **real
LeNet-300-100 network** (784 → 300 → 100 → 10) with **real trained weights** and **real MNIST test images**.
It performs **266 200 multiply-accumulate operations per image** (≈ 25× the toy MLP's 10 752). Run at batch
**B = 1** (one image, memory-bound) and **B = 8** (eight images at once, compute-bound). All cycle counts from
`mcycle`. Verilator RTL simulation of the full X-HEEP SoC, `cv32e40px` + FPnew, DMA hardware-FIFO.

**Every one of the 21 configurations is bit-exact against the CPU, and the accelerator classifies all 8
MNIST images correctly (8/8, matching the reference predictions).**

Swept: `L` = FMA pipeline latency (`fpu_addmul_lat`, 0…5); `policy` = arbiter grant policy
(0 = CPU-strict + accelerator round-robin, 1 = full round-robin, 2 = QoS weighted); QoS weights = CPU:acc0:acc1.

---

## 0. How to read this report

### 0.1 The five ways the network is run (each number belongs to exactly one)

| Tag | Who does the work | The CPU meanwhile | Coprocessors | Batch |
|---|---|---|---|---|
| **[A] CPU-only** | the **CPU alone**, its own FMA | is the one working | 0 | 8 images |
| **[B] one coprocessor, one image** | **one coprocessor** borrows the FMA | idle (launches DMA, waits) | 1 | 1 image |
| **[C] one coprocessor, eight images** | **one coprocessor**, batched | idle | 1 | 8 images |
| **[D] one coprocessor + CPU** | **one coprocessor** *and* the CPU's own FIR filter, together on the one FMA | busy with its FIR | 1 + CPU | 8 images |
| **[E] two coprocessors** | **acc0 + acc1** split the rows, run together | idle | 2 | 8 images |
| **[F] two coprocessors + CPU** | **acc0 + acc1 + the CPU's FIR**, three users of the one FMA | busy with its FIR | 2 + CPU | 8 images |

There is **one** floating-point multiply-add unit in the whole chip; the coprocessor(s) borrow the CPU's,
arbitrated. **No second floating-point unit is added.**

### 0.2 What each number means
- **total cycles** — whole clock-cycle count for that scenario.
- **compute phase** — cycles the coprocessor is streaming weights and issuing MACs (labelled `gemv` in the
  raw log — just the timer's name; at B = 8 it is the batched matrix computation).
- **cycles per multiply-accumulate (cyc/MAC)** — compute-phase cycles ÷ number of MACs. Core efficiency
  number (1 image = 266 200 MACs; 8 images = 2 129 600).
- **shared-unit utilisation** — fraction of compute cycles the one FMA issues an op = 1 ÷ cyc/MAC.
- **setup / load** — DMA descriptor programming / streaming the input vectors into the coprocessor.
- **speedup vs CPU** — CPU-only 8-image time ÷ coprocessor 8-image time.
- **CPU FIR slow-down** — how much the CPU's *own* FIR job slows because the coprocessor(s) also use the FMA
  (the price the CPU pays for lending it; kept small by CPU-priority).
- **channel imbalance** — in the two-coprocessor runs, the cycle gap between acc0 finishing and acc1
  finishing (small = balanced).
- **bit-exact** — 1 means the coprocessor output equalled the CPU output bit-for-bit. All runs = 1.

### 0.3 The knob: L = FMA pipeline latency (`fpu_addmul_lat`, 0…5)
How many pipeline stages the multiply-add unit has. L = 0 combinational, L = 5 deep pipeline.

---

## 1. Correctness (all 21 configurations)
**Every configuration is bit-exact against the CPU**, and the batched accelerator run classifies **8/8**
MNIST images correctly, matching the reference predictions — batched GEMM, dual accelerators, three-way
CPU+acc0+acc1 contention, every policy and weight. Sharing the FMA never changes the numerical result on the
real network.

---

## 2. Scenario [A] — CPU alone, 8 images (the baseline)
Who: **CPU only**. Batch: 8 images.

| L | CPU-only time, 8 images (cycles) |
|---:|---:|
| 0 | 21 108 732 |
| 1 | 21 112 012 |
| 2 | 21 115 292 |
| 3 | 21 118 572 |
| 4 | 21 121 932 |
| 5 | 21 128 412 |

The CPU-only time barely moves with L (+0.09 % across L = 0…5). Each speedup below is against the same-L CPU.

---

## 3. Scenario [B] — one coprocessor, ONE image (memory-bound)
Who: **one coprocessor**, CPU idle. Batch: **1 image** (266 200 MACs).

| L | total (cyc) | setup | load | compute | **cyc/MAC** | shared-unit util |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 555 719 | 1 641 | 1 833 | 533 308 | **2.00** | 49 % |
| 1 | 822 330 | 1 641 | 1 833 | 799 509 | **3.00** | 33 % |
| 2 | 1 089 018 | 1 641 | 1 833 | 1 065 787 | **4.00** | 24 % |
| 3 | 1 355 468 | 1 641 | 1 833 | 1 331 827 | **5.00** | 19 % |
| 4 | 1 622 148 | 1 641 | 1 833 | 1 598 097 | **6.00** | 16 % |
| 5 | 1 888 768 | 1 641 | 1 833 | 1 864 307 | **7.00** | 14 % |

One image, one coprocessor: each MAC costs **exactly 2 + L** cycles (the "+L" is the serial wait for the
pipeline result — one operation in flight). Memory-bound: the shared unit sits idle most cycles.

---

## 4. Scenario [C] — one coprocessor, EIGHT images batched (compute-bound)
Who: **one coprocessor**, CPU idle. Batch: **8 images** (2 129 600 MACs). Each weight is fetched once and
reused across all eight images.

| L | total (cyc) | setup | load | compute | **cyc/MAC** | shared-unit util | **speedup vs CPU** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 529 812 | 1 641 | 12 183 | 2 400 279 | **1.12** | 88 % | **8.34×** |
| 1 | 4 662 806 | 1 641 | 12 183 | 4 529 993 | **2.12** | 47 % | 4.52× |
| 2 | 6 795 477 | 1 641 | 12 183 | 6 659 384 | **3.12** | 31 % | 3.10× |
| 3 | 8 928 425 | 1 641 | 12 183 | 8 789 052 | **4.12** | 24 % | 2.36× |
| 4 | 11 061 333 | 1 641 | 12 183 | 10 918 680 | **5.12** | 19 % | 1.90× |
| 5 | 13 194 581 | 1 641 | 12 183 | 13 048 648 | **6.12** | 16 % | 1.60× |

Batching removes a fixed **0.88 cyc/MAC** at every L (2.00 → 1.12, …, 7.00 → 6.12): the per-weight overhead
now amortised over eight images. At **L = 0 this is decisive** — 1.12 cyc/MAC, the shared unit **88 %** busy,
and **8.34× faster than the CPU** on a real network. At higher L each MAC still pays the "+L" serial wait, so
one coprocessor's advantage falls to 1.60× at L = 5 — recovered by a second coprocessor (§5).

---

## 5. Scenario [E] — TWO coprocessors, eight images (the headline)
Who: **acc0 + acc1** together (rows split in half), CPU idle. Batch: 8 images.

| L | total (cyc) | compute | non-compute overhead | **cyc/MAC** | **speedup of two vs one (compute)** | imbalance | **speedup vs CPU** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 329 501 | 2 130 537 | 198 964 | 1.00 | **1.12×** | 223 | 9.06× |
| 1 | 2 599 386 | 2 397 142 | 202 244 | 1.12 | **1.88×** | 259 | 8.12× |
| 2 | 3 535 673 | 3 330 149 | 205 524 | 1.56 | **1.99×** | 220 | 5.97× |
| 3 | 4 603 876 | 4 395 072 | 208 804 | 2.06 | **1.99×** | 224 | 4.59× |
| 4 | 5 671 912 | 5 459 828 | 212 084 | 2.56 | **1.99×** | 222 | 3.72× |
| 5 | 6 739 959 | 6 524 595 | 215 364 | 3.06 | **1.99×** | 180 | 3.13× |

The key finding, confirmed on the real network:
- **The second coprocessor is nearly useless at L = 0 (1.12×) but nearly doubles throughput at L ≥ 2
  (~1.99×).** A single serial coprocessor uses ≈ 1 / (1 + L) of the shared unit (measured: 88/47/31/24/19/16 %
  ≈ 100/50/33/25/20/17 %), so at L = 0 one already fills the unit (a second finds no room), while at L ≥ 2 one
  leaves most of the unit idle and the second slots straight into those gaps.
- Unlike the toy MLP (where the dual was slightly *slower* in total at L = 0), on LeNet **the dual is already
  a bit faster at L = 0** (2 329 501 vs 2 529 812) and **9.06× vs the CPU** — because LeNet's compute is large
  enough that even a 12 % compute saving outweighs the second channel's fixed setup/load overhead.
- **Two coprocessors stay 3.13×–9.06× faster than the CPU at every latency**, versus one coprocessor which
  falls to 1.60× at L = 5. Channels stay balanced (imbalance ≤ 259 cycles, < 0.01 %).

---

## 6. Scenarios [D] and [F] — coprocessor(s) AND the CPU together (the arbiter)
The coprocessor(s) run the 8-image batch **while the CPU runs its own FIR filter**, so they truly contend for
the one FMA. The number that matters is **CPU FIR slow-down** — the CPU's price for lending its unit.

### 6.1 CPU FIR slow-down with ONE coprocessor + CPU ([D])
| L | P0 CPU-strict | P1 round-robin | P2 QoS (weights) |
|---:|---:|---:|---|
| 0 | 7 % | 7 % | 18 % (1:1:1) · 10 % (4:1:1) · 38 % (1:4:4) · 38 % (1:4:1) |
| 1 | 4 % | 4 % | 7 % (4:2:1) |
| 2 | 3 % | 3 % | 4 % (4:2:1) |
| 3 | 3 % | 3 % | 3 % (4:2:1) |
| 4 | 2 % | 2 % | 3 % (4:2:1) |
| 5 | 1 % | 1 % | 2 % (4:2:1) |

With one coprocessor, CPU-strict and round-robin behave the same (a lone coprocessor + the sparse FIR rarely
collide); the QoS weight is a tunable knob (CPU-favoured 10 % → coprocessor-favoured 38 % at L = 0). The
numbers shrink with L, because at higher L the coprocessor leaves the unit idle and the CPU takes it freely.

### 6.2 CPU FIR slow-down with TWO coprocessors + CPU — three-way ([F])
| L | P0 CPU-strict | P1 round-robin | P2 QoS (weights) |
|---:|---:|---:|---|
| 0 | **9 %** | 13 % | 29 % (1:1:1) · 13 % (4:1:1) · **103 %** (1:4:4) · 53 % (1:4:1) |
| 1 | **8 %** | 16 % | 12 % (4:2:1) |
| 2 | **5 %** | 8 % | 8 % (4:2:1) |
| 3 | **4 %** | 5 % | 6 % (4:2:1) |
| 4 | **4 %** | 6 % | 5 % (4:2:1) |
| 5 | **3 %** | 6 % | 4 % (4:2:1) |

Reading:
- **CPU-strict (P0) protects the CPU best at every latency** (9 % → 3 %), always below full round-robin.
- **Full round-robin (P1) costs the CPU more** (13 % → 6 %): removing CPU priority has a real price.
- **QoS (P2) is a genuine knob:** at L = 0, favouring the CPU (4:1:1) holds the CPU slow-down to 13 %; equal
  shares give 29 %; favouring the coprocessors (1:4:4) makes the CPU's own job take **more than twice as long
  (103 %)**. The whole span is dialled by three weights.
- The coprocessors themselves are barely slowed in [D]/[F] (contention ≈ 0 %); at L ≥ 1 the accelerators are
  latency-bound (the FMA has spare capacity), so the CPU's FIR causes essentially no accelerator slowdown —
  a couple of runs even measure a tiny negative value, i.e. arbitration jitter, not a real effect.

### 6.3 The other QoS knob: balancing the two coprocessors (dual [E], L = 0)
| weights (CPU:acc0:acc1) | channel imbalance (cyc) | meaning |
|---|---:|---|
| 1:1:1 (equal) | 223 | the two coprocessors finish together |
| 1:4:4 (both favoured equally) | 224 | still balanced |
| **1:4:1 (acc0 favoured 4-to-1 over acc1)** | **761 975** | acc0 races ahead; acc1 is starved and finishes far later |

The same weighted policy either protects the CPU or deliberately prioritises one coprocessor over the other:
weighting acc0 4-to-1 blows the channel imbalance from ~223 to **761 975 cycles**.

---

## 7. Toy MLP vs. real LeNet — the effects hold, and the win is bigger
| Metric (L = 0) | toy MLP (128-64-32-16) | **LeNet-300-100 (real)** |
|---|---:|---:|
| B = 1 cyc/MAC | 2.05 | 2.00 |
| B = 8 cyc/MAC | 1.14 | 1.12 |
| shared-unit util (B = 8) | 87 % | 88 % |
| **single-coprocessor speedup vs CPU** | 5.46× | **8.34×** |
| **dual speedup vs CPU** | 5.27× | **9.06×** |
| dual vs single (compute) | 1.11× | 1.12× |
| [F] CPU slow-down P0 / P1 / QoS-equal | 2 / 12 / 25 % | 9 / 13 / 29 % |
| dual speedup at L ≥ 2 | ~1.98× | ~1.99× |
| bit-exact / accuracy | 1 / — | 1 / **8-of-8 correct** |

Every trend from the toy MLP reproduces on the real network. Two things are *better* on LeNet: the
single-coprocessor speedup is higher (8.34× vs 5.46×, because the real network's CPU is relatively slower and
the offload overhead is better amortised), and the dual is already worthwhile at L = 0 (9.06× vs CPU). The
CPU FIR slow-downs are larger in absolute terms because LeNet's accelerator busy-window is ~25× longer, so
the CPU's filter overlaps a much longer contended period — but the **policy ordering is identical**
(CPU-strict < round-robin < QoS-equal < QoS-accelerator-favoured).

---

## 8. Key findings
1. **Correct on a real network:** all 21 configurations bit-exact, and 8/8 MNIST images classified correctly.
2. **Batching turns the memory-bound single-image problem compute-bound:** L = 0, 2.00 → 1.12 cyc/MAC, 88 %
   shared-unit utilisation, **8.34× faster than the CPU** — no second FMA, only a larger input buffer.
3. **The second coprocessor is a latency-hiding device** whose value grows with FMA latency: 1.12× at L = 0
   but **~1.99× at L ≥ 2**; two coprocessors stay **3.13×–9.06× vs the CPU** at every latency.
4. **CPU priority is cheap and the QoS policy is a real knob:** CPU-strict keeps the CPU's own job within
   3–9 % even under two coprocessors; the weighted policy dials the CPU's slow-down from 13 % to 103 %, and
   balances or unbalances the two coprocessors (imbalance 223 → 761 975 cycles).
5. **Every toy-MLP conclusion reproduces on the real LeNet**, with an even higher headline speedup.

---

## Appendix A — every configuration (21 runs)

cyc/MAC = compute-phase cycles per multiply-accumulate. "vs CPU" = CPU-only 8-image time ÷ this run.
Rows sharing an L have identical [A]/[B]/[C]/[E] (those are policy-independent); only [D]/[F] change with policy.

| config | L | policy | weights | [A] CPU 8-img | [B] 1-img cyc/MAC | [C] 8-img cyc/MAC | [C] vs CPU | [E] dual vs 1 | [E] dual vs CPU | [E] imbalance | [D] CPU-slow | [F] CPU-slow | bit-exact / preds |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| L0_P0 | 0 | 0 | 1:1:1 | 21 108 732 | 2.00 | 1.12 | 8.34× | 1.12× | 9.06× | 223 | 7 % | 9 % | ✓ 8/8 |
| L0_P1 | 0 | 1 | 1:1:1 | 21 108 732 | 2.00 | 1.12 | 8.34× | 1.12× | 9.06× | 223 | 7 % | 13 % | ✓ 8/8 |
| L0_P2 | 0 | 2 | 1:1:1 | 21 108 732 | 2.00 | 1.12 | 8.34× | 1.12× | 9.06× | 224 | 18 % | 29 % | ✓ 8/8 |
| L0_P2 | 0 | 2 | 4:1:1 | 21 108 732 | 2.00 | 1.12 | 8.34× | 1.12× | 9.06× | 224 | 10 % | 13 % | ✓ 8/8 |
| L0_P2 | 0 | 2 | 1:4:4 | 21 108 732 | 2.00 | 1.12 | 8.34× | 1.12× | 9.06× | 224 | 38 % | **103 %** | ✓ 8/8 |
| L0_P2 | 0 | 2 | 1:4:1 | 21 108 732 | 2.00 | 1.12 | 8.34× | 1.08× | 8.74× | **761 975** | 38 % | 53 % | ✓ 8/8 |
| L1_P0 | 1 | 0 | 1:1:1 | 21 112 012 | 3.00 | 2.12 | 4.52× | 1.88× | 8.12× | 259 | 4 % | 8 % | ✓ 8/8 |
| L1_P1 | 1 | 1 | 1:1:1 | 21 112 012 | 3.00 | 2.12 | 4.52× | 1.88× | 8.12× | 259 | 4 % | 16 % | ✓ 8/8 |
| L1_P2 | 1 | 2 | 4:2:1 | 21 112 012 | 3.00 | 2.12 | 4.52× | 1.99× | 8.53× | 3 667 | 7 % | 12 % | ✓ 8/8 |
| L2_P0 | 2 | 0 | 1:1:1 | 21 115 292 | 4.00 | 3.12 | 3.10× | 1.99× | 5.97× | 220 | 3 % | 5 % | ✓ 8/8 |
| L2_P1 | 2 | 1 | 1:1:1 | 21 115 292 | 4.00 | 3.12 | 3.10× | 1.99× | 5.97× | 220 | 3 % | 8 % | ✓ 8/8 |
| L2_P2 | 2 | 2 | 4:2:1 | 21 115 292 | 4.00 | 3.12 | 3.10× | 1.99× | 5.97× | 180 | 4 % | 8 % | ✓ 8/8 |
| L3_P0 | 3 | 0 | 1:1:1 | 21 118 572 | 5.00 | 4.12 | 2.36× | 1.99× | 4.59× | 224 | 3 % | 4 % | ✓ 8/8 |
| L3_P1 | 3 | 1 | 1:1:1 | 21 118 572 | 5.00 | 4.12 | 2.36× | 1.99× | 4.59× | 224 | 3 % | 5 % | ✓ 8/8 |
| L3_P2 | 3 | 2 | 4:2:1 | 21 118 572 | 5.00 | 4.12 | 2.36× | 1.99× | 4.59× | 222 | 3 % | 6 % | ✓ 8/8 |
| L4_P0 | 4 | 0 | 1:1:1 | 21 121 932 | 6.00 | 5.12 | 1.90× | 1.99× | 3.72× | 222 | 2 % | 4 % | ✓ 8/8 |
| L4_P1 | 4 | 1 | 1:1:1 | 21 121 932 | 6.00 | 5.12 | 1.90× | 1.99× | 3.72× | 222 | 2 % | 6 % | ✓ 8/8 |
| L4_P2 | 4 | 2 | 4:2:1 | 21 121 932 | 6.00 | 5.12 | 1.90× | 1.99× | 3.72× | 221 | 3 % | 5 % | ✓ 8/8 |
| L5_P0 | 5 | 0 | 1:1:1 | 21 128 412 | 7.00 | 6.12 | 1.60× | 1.99× | 3.13× | 180 | 1 % | 3 % | ✓ 8/8 |
| L5_P1 | 5 | 1 | 1:1:1 | 21 128 412 | 7.00 | 6.12 | 1.60× | 1.99× | 3.13× | 180 | 1 % | 6 % | ✓ 8/8 |
| L5_P2 | 5 | 2 | 4:2:1 | 21 128 412 | 7.00 | 6.12 | 1.60× | 1.99× | 3.13× | 141 | 2 % | 4 % | ✓ 8/8 |

Constants: 1 image = 266 200 MACs, 8 images = 2 129 600. Per-phase (L = 0, B = 8): setup 1 641, load 12 183,
compute 2 400 279 cycles (compute ≈ 95 % of the 8-image accelerator forward). Dual [E] non-compute overhead
is a near-constant ~+70 000 cycles over single [C] (the second DMA channel's setup + doubled input load).

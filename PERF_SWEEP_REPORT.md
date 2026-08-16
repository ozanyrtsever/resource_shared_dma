# Shared-FMA Coprocessor — Full Performance Sweep Report

This report explains, with **no abbreviations and every number attributed to exactly who was running**,
the complete performance sweep of the shared-FMA coprocessor. Every configuration below completed and is
**bit-exact** against the CPU reference (the coprocessor's floating-point result matched the CPU's result
bit for bit).

The workload is one small neural network: a 3-layer fully-connected classifier (sizes 128 → 64 → 32 → 16),
which performs **10 752 multiply-accumulate operations per input image**. We run it either for **one image**
or for **eight images at once (a batch of eight)**.

---

## 0. How to read this report (read this first)

### 0.1 The five ways the network can be run

Throughout the report, each measurement is one of these five scenarios. **Whenever you see a number, it
belongs to exactly one of these — the scenario is always named.**

| Tag | Who is doing the work | The CPU meanwhile | How many coprocessors | Batch |
|---|---|---|---|---|
| **[A] CPU-only** | The **CPU alone** runs the whole network, using its **own** fused multiply-add unit. | is the one doing everything | **zero** coprocessors | 8 images |
| **[B] one coprocessor, one image** | **One coprocessor** runs the network for a **single image**, borrowing the shared multiply-add unit. | is **idle** (only starts the DMA and waits) | **one** coprocessor | 1 image |
| **[C] one coprocessor, eight images** | **One coprocessor** runs the network for **eight images in one batched pass**. | is **idle** | **one** coprocessor | 8 images |
| **[D] one coprocessor + the CPU together** | **One coprocessor** runs the eight-image batch **at the same time as** the CPU runs its **own** FIR-filter job. Both compete for the **one** shared multiply-add unit. | is **busy** with its own FIR filter | **one** coprocessor + CPU | 8 images |
| **[E] two coprocessors, eight images** | **Two coprocessors** (called acc0 and acc1) split the work by output rows and run **at the same time**. | is **idle** | **two** coprocessors | 8 images |
| **[F] two coprocessors + the CPU together** | **Two coprocessors** run the eight-image batch **at the same time as** the CPU runs its FIR filter. **Three** users (acc0, acc1, CPU) compete for the **one** shared multiply-add unit. | is **busy** with its own FIR filter | **two** coprocessors + CPU | 8 images |

There is **only one** hardware floating-point multiply-add unit in the whole chip. The CPU normally owns it;
the coprocessor(s) borrow it through a small arbiter. **No second floating-point unit is ever added** — that
is the entire point of the design.

### 0.2 What each measured number means

- **total cycles** — the whole clock-cycle count for that scenario, measured with the CPU's cycle counter
  (`mcycle`). Smaller is faster.
- **compute-phase cycles** — the part of the run in which the coprocessor is actually streaming weights and
  issuing multiply-accumulate operations. *In the raw log this counter is labelled `gemv` — that is only the
  name of the timer, left over from when the batch was one image. At a batch of eight it times the batched
  matrix–matrix computation (GEMM). The operation is correct; only the label name is old.*
- **cycles per multiply-accumulate (cyc/MAC)** — compute-phase cycles divided by the number of
  multiply-accumulate operations. This is the core efficiency number: **how many clock cycles each
  multiply-accumulate costs.** Smaller is better. (One image = 10 752 operations; eight images = 86 016.)
- **shared-unit utilisation** — the fraction of compute-phase cycles in which the one shared multiply-add
  unit actually starts a new operation. It equals 1 ÷ (cycles per multiply-accumulate). Higher means the
  unit is kept busier.
- **setup cycles** — time spent programming the DMA engine's descriptors.
- **load cycles** — time spent streaming the input vector(s) into the coprocessor before computing.
- **speedup versus CPU** — the CPU-only time for the *same eight images* divided by the coprocessor time.
  "3.26×" means 3.26 times faster than the CPU doing the same work.
- **coprocessor slow-down under contention** — in scenarios [D] and [F], how many extra compute-phase cycles
  the coprocessor needs *because the CPU is also using the shared unit at the same time*. Reported as extra
  cycles and as a percentage of the coprocessor's alone-time.
- **CPU FIR slow-down** — in scenarios [D] and [F], how much **the CPU's own FIR-filter job** slowed down
  *because the coprocessor(s) were also using the shared unit*. **This is the price the CPU pays for
  lending its multiply-add unit.** Keeping this small is the job of the CPU-priority arbiter.
- **channel imbalance** — in scenarios [E] and [F] (two coprocessors), the gap in cycles between when acc0
  finished and when acc1 finished. Small means the two halves of the work were well balanced.
- **bit-exact** — 1 means the coprocessor's floating-point output equalled the CPU's output exactly, bit
  for bit. Every scenario in this report is bit-exact.

### 0.3 The three knobs we swept

- **L = floating-point multiply-add latency** (`fpu_addmul_lat`, values 0…5): how many pipeline cycles the
  multiply-add unit takes to produce a result. L = 0 is a fully combinational unit; L = 5 is a deep
  pipeline. This is a **hardware** parameter of the multiply-add unit, not of our design.
- **arbiter policy** (0, 1, 2): the rule the arbiter uses to decide who gets the shared unit next.
  - **policy 0 = CPU-strict**: the CPU always wins; the two coprocessors take turns among themselves for
    what is left. This is the default and it is what protects the CPU.
  - **policy 1 = full round-robin**: the CPU has **no** priority; CPU, acc0 and acc1 all take equal turns.
  - **policy 2 = weighted (Quality-of-Service)**: each requester gets a share proportional to a weight.
    Weights are written as **CPU : acc0 : acc1**, e.g. `4:1:1` favours the CPU, `1:4:4` favours the
    coprocessors, `1:4:1` favours acc0 over acc1.
- **QoS weights** (only meaningful under policy 2): the CPU : acc0 : acc1 numbers just described.

---

## 1. Correctness (every single configuration)

**All 21 configurations are bit-exact against the CPU.** This includes: the batched matrix computation, two
coprocessors running together, three users (two coprocessors plus the CPU) fighting over one multiply-add
unit, and every arbiter policy and weight combination. Sharing the unit — in any policy, at any latency,
with one or two coprocessors, with or without the CPU running concurrently — **never changes the numerical
result.**

---

## 2. Scenario [A] — the CPU alone, eight images (the baseline everything is compared to)

Who: **CPU only, no coprocessor.** Batch: 8 images.

| L (unit latency) | CPU-only time for 8 images (cycles) |
|---:|---:|
| 0 | 701 463 |
| 1 | 702 231 |
| 2 | 703 127 |
| 3 | 704 151 |
| 4 | 705 815 |
| 5 | 707 479 |

Note: the CPU-only time rises very slightly with L (about +0.9 % from L = 0 to L = 5) because the CPU also
uses the same multiply-add unit, so a deeper pipeline costs the CPU a little too. Each speedup below is
computed against the CPU-only time **at the same L**.

---

## 3. Scenario [B] — one coprocessor, ONE image (memory-bound baseline)

Who: **one coprocessor**, CPU idle. Batch: **1 image** (10 752 multiply-accumulate operations).

| L | total time (cycles) | setup | load | compute phase | **cycles per multiply-accumulate** | shared-unit utilisation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28 534 | 1 362 | 657 | 22 042 | **2.05** | 48 % |
| 1 | 39 370 | 1 362 | 657 | 32 766 | **3.04** | 32 % |
| 2 | 50 265 | 1 362 | 657 | 43 549 | **4.05** | 24 % |
| 3 | 61 062 | 1 362 | 657 | 54 234 | **5.04** | 19 % |
| 4 | 71 938 | 1 362 | 657 | 64 998 | **6.04** | 16 % |
| 5 | 82 821 | 1 362 | 657 | 75 769 | **7.04** | 14 % |

Reading: with one image and one coprocessor, each multiply-accumulate costs about **(2 + L)** cycles. The
"+L" is the coprocessor **waiting** for the pipeline result before issuing the next operation — it keeps
only one operation in flight (it is a **serial** coprocessor). This is the *memory-bound / latency-bound*
baseline: the shared unit sits idle most of the time (utilisation drops from 48 % at L = 0 to 14 % at L = 5).

---

## 4. Scenario [C] — one coprocessor, EIGHT images batched (the batching effect)

Who: **one coprocessor**, CPU idle. Batch: **8 images** (86 016 multiply-accumulate operations). Each weight
is fetched **once** and reused across all eight images — this is what makes it a batched matrix computation.

| L | total time (cycles) | setup | load | compute phase | **cycles per multiply-accumulate** | shared-unit utilisation | **speedup vs CPU-only [A]** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 341 | 1 362 | 2 559 | 98 319 | **1.14** | 87 % | **5.46×** |
| 1 | 215 139 | 1 362 | 2 559 | 184 221 | **2.14** | 46 % | 3.26× |
| 2 | 302 089 | 1 362 | 2 559 | 270 275 | **3.14** | 31 % | 2.32× |
| 3 | 388 948 | 1 362 | 2 559 | 356 238 | **4.14** | 24 % | 1.81× |
| 4 | 475 874 | 1 362 | 2 559 | 442 268 | **5.14** | 19 % | 1.48× |
| 5 | 562 783 | 1 362 | 2 559 | 528 281 | **6.14** | 16 % | 1.25× |

Reading — compare each row here (**one coprocessor, eight images**) with the same L row in §3 (**one
coprocessor, one image**):

- Batching **removes a fixed 0.9 cycles per multiply-accumulate** at every L (2.05 → 1.14, 3.04 → 2.14, …,
  7.04 → 6.14). That saved 0.9 is the per-weight fetch/overhead now spread over eight images instead of one.
- At **L = 0** this is decisive: cost drops to **1.14** cycles per operation and the shared unit is **87 %**
  busy — the computation is now limited by the multiply-add unit itself, not by memory. Result: **5.46×
  faster than the CPU.**
- At **higher L** batching helps less, because each operation still pays the "+L" serial wait, which
  batching does **not** remove. By L = 5 one coprocessor is only 1.25× the CPU. **Batching alone cannot hide
  the pipeline latency — that requires a second coprocessor (§5).**

---

## 5. Scenario [E] — TWO coprocessors, eight images (the headline result)

Who: **two coprocessors, acc0 and acc1**, running at the same time; CPU idle. Batch: 8 images. The network's
output rows are split in half: acc0 computes the first half, acc1 the second half (on separate DMA channels
on separate bus ports).

| L | total time (cycles) | compute phase | **non-compute overhead** (setup + load + activation) | **cycles per multiply-accumulate** | **speedup of two coprocessors vs one (§4 compute)** | channel imbalance | **speedup vs CPU-only [A]** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 133 203 | 87 998 | 45 205 | 1.02 | **1.11×** | 223 | 5.27× |
| 1 | 144 358 | 98 257 | 46 101 | 1.14 | **1.87×** | 140 | 4.86× |
| 2 | 183 914 | 136 917 | 46 997 | 1.59 | **1.97×** | 179 | 3.82× |
| 3 | 227 867 | 179 974 | 47 893 | 2.09 | **1.97×** | 220 | 3.09× |
| 4 | 271 780 | 222 991 | 48 789 | 2.59 | **1.98×** | 138 | 2.60× |
| 5 | 315 610 | 265 925 | 49 685 | 3.09 | **1.98×** | 222 | 2.24× |

**About the "non-compute overhead" column — where the setup and load cycles are.** The two coprocessors'
setup and input-load cycles **are** measured — the two-coprocessor code brackets them exactly as the
single-coprocessor code does (setup = programming both DMA channels' descriptors; load = streaming the eight
input images into **both** coprocessors) — and they **are** included in the total time. The only thing the
raw simulator log does not do is print them separately on the [E] line (it prints total, compute phase, and
channel imbalance), so this column is obtained as **total − compute phase**; it is the sum of DMA-descriptor
setup, input streaming, and the between-layer activation.

**The key point this column reveals:** the two-coprocessor run carries a **constant extra +15 183 cycles of
non-compute overhead versus the one-coprocessor run, at every latency** (dual 45 205 vs single 30 022 at
L = 0; dual 49 685 vs single 34 502 at L = 5 — the gap is exactly 15 183 in every row). That fixed extra is
the **second DMA channel's descriptor setup plus the doubled input load** (both independent of the
multiply-add latency L). **This is precisely why two coprocessors are slightly slower in total than one at
L = 0:** the compute saving there is only 98 319 − 87 998 = 10 321 cycles, which the +15 183 overhead
outweighs (net +4 862 cycles: total 133 203 vs 128 341). From L = 1 onward the compute saving is large
enough to dwarf this fixed overhead, so the two-coprocessor total wins comfortably.

Reading — the key finding of the whole study:

- **The second coprocessor is nearly useless at L = 0 (only 1.11×) but nearly doubles throughput at L ≥ 2
  (up to 1.98×).**
- Why, exactly: a single **serial** coprocessor issues one operation about every **(1 + L)** cycles, so it
  can only use about **1 / (1 + L)** of the shared unit. The measured single-coprocessor utilisation from §4
  (87 %, 46 %, 31 %, 24 %, 19 %, 16 % for L = 0…5) matches 1 / (1 + L) almost exactly (100 %, 50 %, 33 %,
  25 %, 20 %, 17 %).
  - At **L = 0** one coprocessor already fills the unit (≈ 87 %), so a second one finds almost no free
    capacity → only 1.11×, and its extra setup/load overhead even makes the *total* slightly slower than one
    coprocessor (133 203 vs 128 341).
  - At **L = 5** one coprocessor uses only ~1/6 of the unit, leaving **five-sixths idle**, so the second
    coprocessor slots straight into those idle cycles → the compute phase **halves** and throughput doubles.
- **So the value of a second (shared-unit) coprocessor grows with the multiply-add latency.** It is a
  latency-hiding device. The two channels stay well balanced (imbalance ≤ 223 cycles, under 0.2 %).
- Compared to the CPU: two coprocessors are **2.24× to 5.27×** faster than the CPU across all latencies —
  and unlike one coprocessor (which falls to 1.25× at L = 5), two coprocessors stay above **2× even at the
  deepest pipeline.**

---

## 6. Scenarios [D] and [F] — running the coprocessor(s) AND the CPU at the same time (the arbiter)

Here the coprocessor(s) run the eight-image batch **while the CPU runs its own FIR-filter job**, so they
genuinely fight over the one shared multiply-add unit. This is where the arbiter policy matters. The number
we care about most is the **CPU FIR slow-down** — the price the CPU pays for lending its unit.

### 6.1 CPU FIR slow-down with ONE coprocessor + the CPU (scenario [D])

Who: **one coprocessor + the CPU**, together.

| L | policy 0 (CPU-strict) | policy 1 (full round-robin) | policy 2 (weighted, weights shown) |
|---:|---:|---:|---|
| 0 | **1 %** | 1 % | 12 % (1:1:1) · 4 % (4:1:1) · 15 % (1:4:4) · 15 % (1:4:1) |
| 1 | **1 %** | 1 % | 3 % (4:2:1) |
| 2 | **1 %** | 1 % | 3 % (4:2:1) |
| 3 | **1 %** | 1 % | 1 % (4:2:1) |
| 4 | **1 %** | 1 % | 2 % (4:2:1) |
| 5 | **0 %** | 0 % | 1 % (4:2:1) |

With only one coprocessor, the CPU is barely affected under any policy (≤ 3 %, usually ~1 %): a single serial
coprocessor leaves the shared unit idle enough that the CPU gets it almost for free. The coprocessor's own
slow-down in this scenario is also small everywhere (0–3 %).

### 6.2 CPU FIR slow-down with TWO coprocessors + the CPU — three-way contention (scenario [F])

Who: **two coprocessors + the CPU**, all three together. This is the hardest test — two coprocessors keep
the unit much busier, so now the policy really decides how much the CPU suffers.

| L | policy 0 (CPU-strict) | policy 1 (full round-robin) | policy 2 (weighted, weights shown) |
|---:|---:|---:|---|
| 0 | **2 %** | 12 % | 25 % (1:1:1) · 8 % (4:1:1) · **100 %** (1:4:4) · 48 % (1:4:1) |
| 1 | **1 %** | 8 % | 6 % (4:2:1) |
| 2 | **1 %** | 8 % | 4 % (4:2:1) |
| 3 | **1 %** | 4 % | 3 % (4:2:1) |
| 4 | **2 %** | 3 % | 3 % (4:2:1) |
| 5 | **1 %** | 3 % | 2 % (4:2:1) |

Reading:

- **Policy 0 (CPU-strict) keeps the CPU almost untouched — at most 2 % slow-down — even with two
  coprocessors hammering the unit at every latency.** This is the design guarantee, and it holds across the
  whole sweep.
- **Policy 1 (full round-robin) makes the CPU pay**: 3 % to 12 % slow-down, worst at low latency (12 % at
  L = 0) where the unit is most contended. Removing the CPU's priority has a real, measurable cost to the CPU.
- **Policy 2 (weighted) is a genuine, adjustable knob.** At L = 0, look along the row: favouring the CPU
  (`4:1:1`) holds the CPU slow-down to **8 %**; equal shares (`1:1:1`) give **25 %**; favouring the
  coprocessors (`1:4:4`) makes the CPU's own job take **twice as long (100 %)**. The weight directly sets how
  much of the shared unit the CPU keeps.

### 6.3 The other weighted knob: balancing the two coprocessors against each other

Under policy 2, the acc0 : acc1 weights also control how the **two coprocessors** share the unit *with each
other*. In the two-coprocessor run (scenario [E]) at L = 0:

| weights (CPU:acc0:acc1) | channel imbalance (cycles) | meaning |
|---|---:|---|
| 1:1:1 (equal) | 138 | the two coprocessors finish together |
| 1:4:4 (both favoured equally) | 138 | still balanced |
| **1:4:1 (acc0 favoured 4-to-1 over acc1)** | **30 615** | acc0 races ahead, acc1 is starved and finishes far later |

So the same weighted policy can be used either to protect the CPU, or to deliberately prioritise one
coprocessor over the other — the imbalance jumps from ~138 cycles to **30 615 cycles** when acc0 is favoured
4-to-1.

### 6.4 When does the policy matter?

- **The policy matters most when the shared unit is contended.** That happens (a) at low latency (L = 0),
  where even one coprocessor nearly fills the unit, and (b) whenever **two** coprocessors run with the CPU
  (scenario [F]), which keeps the unit busy even at higher latency.
- **With only one coprocessor at higher latency, the policy barely matters** (§6.1: ~1 % under every policy),
  because that lone serial coprocessor leaves the unit idle and the CPU takes the free cycles.
- In every case, **CPU-strict (policy 0) is essentially free for the CPU (≤ 2 %)**, so it is a safe default;
  the other policies exist for when you deliberately want to redistribute the shared unit.

---

## 7. Key findings

1. **Correctness is unconditional.** All 21 configurations — one or two coprocessors, with or without the
   CPU running concurrently, every policy and weight — are bit-exact against the CPU.
2. **Batching turns the single-image, memory-bound problem into a compute-bound one.** With one coprocessor
   at L = 0, eight-image batching cuts the cost from 2.05 to **1.14 cycles per multiply-accumulate**, fills
   the shared unit to **87 %**, and runs **5.46× faster than the CPU** — using no second floating-point unit,
   only a larger input buffer.
3. **A single serial coprocessor uses about 1 / (1 + L) of the shared unit** (measured utilisation 87 / 46 /
   31 / 24 / 19 / 16 % for L = 0…5). This one relationship explains the entire latency behaviour.
4. **The second coprocessor is a latency-hiding device whose value grows with latency**: 1.11× at L = 0 (the
   unit is already full) but **~1.98× at L ≥ 2** (it fills the idle cycles the first one leaves). Two
   coprocessors stay **2.24×–5.27× faster than the CPU** at every latency, whereas one coprocessor collapses
   to 1.25× at L = 5.
5. **CPU priority is essentially free, and the weighted policy is a real knob.** CPU-strict keeps the CPU's
   own job within **2 %** even under two coprocessors at every latency. The weighted policy can be dialled
   from protecting the CPU (8 % slow-down) to starving it (100 % slow-down), and can likewise balance or
   deliberately unbalance the two coprocessors (imbalance 138 → 30 615 cycles).

---

## 8. Notes and scope

- **Latency sweep is complete for L = 0 through 5.** The full-detail policy/weight sweep (six weight
  combinations) was run at L = 0, where contention is strongest and the policy differences are largest;
  L = 1…5 were run with the three main policies (CPU-strict, round-robin, one weighted setting 4:2:1).
- The label `gemv` in the raw simulator logs is only the name of the compute-phase timer; at a batch of
  eight the operation is a batched matrix computation (GEMM), and the reported cycles-per-operation and
  speedups are correct for that.
- The coprocessor is **serial** (one operation in flight, so each costs about 1 + L cycles). A pipelined
  coprocessor would stay compute-bound at higher latency and would change §4/§5 (batching alone would then
  hide latency, and one coprocessor could fill the unit at any L). This is the main lever for future work.

---

## Appendix A — every configuration, one row each

The columns [A]–[F] name exactly who was running (see §0.1). "cyc/MAC" = cycles per multiply-accumulate.
Blank cells under [C]/[E] repeat the value above them (those scenarios do not depend on the arbiter policy).

| Configuration | L | policy | weights (CPU:acc0:acc1) | [A] CPU-only 8 img (cyc) | [B] 1 coproc, 1 img: cyc/MAC | [C] 1 coproc, 8 img: cyc/MAC | [C] speedup vs CPU | [E] 2 coproc, 8 img: speedup vs 1 | [E] 2 coproc vs CPU | [E] imbalance | [D] 1 coproc+CPU: CPU slow | [F] 2 coproc+CPU: CPU slow | bit-exact |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| L0_P0 | 0 | 0 CPU-strict | 1:1:1 | 701 463 | 2.05 | 1.14 | 5.46× | 1.11× | 5.27× | 223 | 1 % | 2 % | ✓ |
| L0_P1 | 0 | 1 round-robin | 1:1:1 | 701 463 | 2.05 | 1.14 | 5.46× | 1.11× | 5.27× | 223 | 1 % | 12 % | ✓ |
| L0_P2 | 0 | 2 weighted | 1:1:1 | 701 463 | 2.05 | 1.14 | 5.46× | 1.11× | 5.27× | 138 | 12 % | 25 % | ✓ |
| L0_P2 | 0 | 2 weighted | 4:1:1 (CPU-fav) | 701 463 | 2.05 | 1.14 | 5.46× | 1.11× | 5.27× | 141 | 4 % | 8 % | ✓ |
| L0_P2 | 0 | 2 weighted | 1:4:4 (coproc-fav) | 701 463 | 2.05 | 1.14 | 5.46× | 1.11× | 5.27× | 138 | 15 % | **100 %** | ✓ |
| L0_P2 | 0 | 2 weighted | 1:4:1 (acc0-fav) | 701 463 | 2.05 | 1.14 | 5.46× | 1.07× | 5.13× | **30 615** | 15 % | 48 % | ✓ |
| L1_P0 | 1 | 0 CPU-strict | 1:1:1 | 702 231 | 3.04 | 2.14 | 3.26× | 1.87× | 4.86× | 140 | 1 % | 1 % | ✓ |
| L1_P1 | 1 | 1 round-robin | 1:1:1 | 702 231 | 3.04 | 2.14 | 3.26× | 1.87× | 4.86× | 140 | 1 % | 8 % | ✓ |
| L1_P2 | 1 | 2 weighted | 4:2:1 | 702 231 | 3.04 | 2.14 | 3.26× | 1.93× | 4.96× | 895 | 3 % | 6 % | ✓ |
| L2_P0 | 2 | 0 CPU-strict | 1:1:1 | 703 127 | 4.05 | 3.14 | 2.32× | 1.97× | 3.82× | 179 | 1 % | 1 % | ✓ |
| L2_P1 | 2 | 1 round-robin | 1:1:1 | 703 127 | 4.05 | 3.14 | 2.32× | 1.97× | 3.82× | 179 | 1 % | 8 % | ✓ |
| L2_P2 | 2 | 2 weighted | 4:2:1 | 703 127 | 4.05 | 3.14 | 2.32× | 1.97× | 3.82× | 222 | 3 % | 4 % | ✓ |
| L3_P0 | 3 | 0 CPU-strict | 1:1:1 | 704 151 | 5.04 | 4.14 | 1.81× | 1.97× | 3.09× | 220 | 1 % | 1 % | ✓ |
| L3_P1 | 3 | 1 round-robin | 1:1:1 | 704 151 | 5.04 | 4.14 | 1.81× | 1.97× | 3.09× | 220 | 1 % | 4 % | ✓ |
| L3_P2 | 3 | 2 weighted | 4:2:1 | 704 151 | 5.04 | 4.14 | 1.81× | 1.97× | 3.09× | 266 | 1 % | 3 % | ✓ |
| L4_P0 | 4 | 0 CPU-strict | 1:1:1 | 705 815 | 6.04 | 5.14 | 1.48× | 1.98× | 2.60× | 138 | 1 % | 2 % | ✓ |
| L4_P1 | 4 | 1 round-robin | 1:1:1 | 705 815 | 6.04 | 5.14 | 1.48× | 1.98× | 2.60× | 138 | 1 % | 3 % | ✓ |
| L4_P2 | 4 | 2 weighted | 4:2:1 | 705 815 | 6.04 | 5.14 | 1.48× | 1.98× | 2.60× | 138 | 2 % | 3 % | ✓ |
| L5_P0 | 5 | 0 CPU-strict | 1:1:1 | 707 479 | 7.04 | 6.14 | 1.25× | 1.98× | 2.24× | 222 | 0 % | 1 % | ✓ |
| L5_P1 | 5 | 1 round-robin | 1:1:1 | 707 479 | 7.04 | 6.14 | 1.25× | 1.98× | 2.24× | 222 | 0 % | 3 % | ✓ |
| L5_P2 | 5 | 2 weighted | 4:2:1 | 707 479 | 7.04 | 6.14 | 1.25× | 1.98× | 2.24× | 181 | 1 % | 2 % | ✓ |

Reference constants: one image = 10 752 multiply-accumulate operations; eight images = 86 016. Per-phase
breakdown of one coprocessor on eight images at L = 0: setup 1 362, load 2 559, compute 98 319 cycles (the
compute phase is ~96 % of the coprocessor's total time — the offload overhead is under 4 %).

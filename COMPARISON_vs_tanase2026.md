# Head-to-head: this work vs. Tanase 2026 (the closest prior art)

**The paper.** C. A. Tanase, *"Energy-Efficient Dual-Core RISC-V Architecture for Edge AI Acceleration
with Dynamic MAC Unit Reuse,"* Computers 2026, 15(4), 219 (MDPI). Two **full RISC-V cores** (CPU0, CPU1,
based on the HL5 model) plus an **opportunistic NPU** (CONV/GEMM/POOL tiles) all share **one scalar
Multiply–Accumulate (MAC) unit** and a shared memory, through a CPU-priority FIFO arbiter, with three-level
dynamic frequency scaling (DFS, 100/200/400 MHz) on the MAC. It is the **closest prior art in concept** —
"reuse the existing MAC in the pipeline instead of duplicating it, under a CPU-priority arbiter, with
opportunistic background AI." So the comparison matters.

**But the foundations differ sharply**, and that is where most of our advantage comes from.

---

## 1. What each project actually is (side by side)

| Aspect | Tanase 2026 | This work |
|---|---|---|
| Shared arithmetic unit | a **scalar MAC**, 32×32, **single-cycle**, integer-centric (an FP channel exists but the tiles/benchmarks are integer) | the CPU's **real IEEE-754 floating-point FMA** inside FPnew (the actual `fmadd`), shared through the APU interface |
| Latency of the shared unit | **assumed 1 cycle** (no latency modeling) | **swept L = 0…5** (real `fpu_addmul_lat`), with a measured 1/(1+L) utilisation law |
| Who borrows the unit | a **second full CPU core** + an NPU | a **tiny DMA-fed coprocessor** (now two), **no second core, no second register file, no second pipeline** |
| Implementation | **cycle-accurate SystemC** model (SystemC 2.3.3, g++) | **real RTL**: cv32e40px core + FPnew + the X-HEEP SoC, simulated in **Verilator** (the actual hardware) |
| Area evidence | **post-synthesis FPGA estimate, extrapolated** from a 1×CPU reference to 2×CPU+1×NPU | **real Design Compiler synthesis, TSMC 40 nm** (measured gate-level) |
| Silicon/FPGA realisation | **none** — "FPGA deployment and measurement are left for future work" | RTL is **synthesizable and DC-synthesized**; runs as a full SoC in RTL simulation |
| Numerical correctness | integer tiles; **no bit-exactness / no accuracy** reported | **bit-exact vs CPU and NumPy** on a real network (LeNet-300-100 / MNIST), plus convolution and dense layers |
| Arbiter policies | **one** (CPU-priority FIFO), compared only to round-robin | **three, swappable at build time** (CPU-strict / full round-robin / weighted QoS), with a full L×policy×weight sweep |
| Out-of-order responses | not an issue (1-cycle MAC) | handled: **2-bit owner-tag routed through the FPnew pipeline** for a multi-cycle unit |
| Energy / power / DFS | **yes** — 3-level DFS, 70 % power reduction, energy table | **not studied (our gap)** |
| Dual-core general-purpose | **yes** — 1.87× parallel speedup | single-core CPU (out of scope) |
| Memory/cache study | cache-miss and memory-arbitration tables | bus contention noted, less detailed |

---

## 2. Where WE have the advantage

### 2.1 Real hardware evidence vs. a software model — our single biggest edge
Their entire system is a **SystemC C++ model**; it was "validated for hardware compatibility using Vivado
HLS 2019," but **never deployed to FPGA or silicon**, and its area/power numbers are **extrapolated** from a
one-core reference ("we extrapolate an equivalent 2×CPU+1×NPU configuration"; power is "post-synthesis
gate-level estimation … post-implementation may yield different absolute values"). Our results come from the
**published cv32e40px RTL + real FPnew + the X-HEEP SoC**, run in **Verilator** (RTL cycle-accurate) and
synthesized in **Design Compiler (TSMC 40 nm)**. Every number in our sweep is from the actual hardware
description, not a behavioural estimate. **This is the strongest differentiator: they proposed and modelled
it; we built, ran, and synthesized it.**

### 2.2 We share a real floating-point FMA; they share a single-cycle integer MAC
Edge-AI inference is floating-point in practice. We time-share the CPU's **real IEEE-754 fused
multiply-add**, prove the shared result is **bit-exact** with the CPU and with NumPy, and handle the hard
parts a real FP unit forces on you (multi-cycle latency, in-flight ordering). Their MAC is **single-cycle
and their tiles are integer** (Fibonacci, 10×10 integer matmul, integer CONV/GEMM/POOL), so the FP sharing
problem — the one that actually appears on an MCU FPU — is never confronted.

### 2.3 Honest latency modeling → a real finding they cannot produce
Because their MAC is one cycle, latency-hiding is a non-question. We sweep the FMA latency **L = 0…5** and
show a clean law: **a serial coprocessor uses ≈ 1/(1+L) of the shared unit** (measured utilisation
87/46/31/24/19/16 % for L = 0…5, matching 100/50/33/25/20/17 %). From this comes our headline result: **a
second coprocessor is nearly useless at L = 0 (1.11×) but nearly doubles throughput at L ≥ 2 (~1.98×)**,
because it fills the idle cycles the first one leaves. This latency-scaling story is impossible in a
single-cycle-MAC model.

### 2.4 We reuse the unit without paying for a second core
Their "MAC reuse" still **duplicates the entire CPU pipeline**: each core is 35,719 LUTs, so the two cores
are ~71,400 LUTs — about **90 % of their whole design** — while they save only the second MAC. To avoid one
MAC they add a whole CPU. We add a **tiny DMA coprocessor and a small arbiter** (DC: arbiter ≈ **321 µm²**,
i.e. **2.5 % of one FMA** at 12,812 µm²) and reuse the CPU's already-idle FMA cycles. **Ours is the more
area-efficient form of the same idea.**

### 2.5 A richer, quantified arbiter study
They have one policy (CPU-priority FIFO) and one comparison (vs round-robin). We provide **three swappable
policies** and quantify the cost of each across latency and weights: **CPU-strict keeps the CPU's own job
within ≤ 2 %** at every latency even with two coprocessors; full round-robin costs the CPU up to 12 %; and
the weighted policy is a real knob, dialling CPU slow-down from **+8 % to +100 %**, and balancing (or
deliberately unbalancing) the two coprocessors (channel imbalance 138 → 30,615 cycles).

### 2.6 End-to-end, bit-exact ML — not demonstrative tiles
We run a **complete network end-to-end** (LeNet-300-100 on MNIST, **4.80×**, bit-exact vs CPU and NumPy),
plus a real convolution (im2col→GEMV, 1.97×) and dense layers (2.6×; batched GEMM **5.46×** at L = 0). Their
AI evaluation is **demonstrative tile sequences** (CONV→POOL→CONV→POOL→GEMM) with latency in cycles and no
network-level accuracy or bit-exactness.

---

## 3. Where THEY have the advantage (our honest gaps)

### 3.1 Energy / power / DFS — a whole dimension we do not cover
They apply three-level DFS to the shared unit and report energy and power: **70 % NPU power reduction** at
100 MHz vs 400 MHz, total system energy 145→230 mJ across frequencies, adding the NPU costs only +17 %
energy. **We have no power, energy, or frequency-scaling analysis.** (Caveat: their power is a post-synthesis
*estimate*, and FPGA "DFS" is only clock division — but the dimension is real and we lack it.)

### 3.2 Dual-core general-purpose parallelism
Their 1.87× headline is two cores running general-purpose parallel code (Fibonacci + integer matmul) while
the AI runs opportunistically. We accelerate **ML offload** with a single-core CPU; we do not address
two-core general-purpose speedup. (One can argue "two cores ≈ 2×" is a weak result, but it is a scenario we
simply do not cover.)

### 3.3 Memory / cache contention analysis
They report cache-miss rates and memory-arbitration behaviour under stress (per-requestor hit/miss, average
and worst-case latency). Our design is DMA-streamed rather than cache-based, and our memory-contention
treatment is lighter (a bus-contention note, not a dedicated study).

### 3.4 A latency-distribution / predictability analysis
They give a full CPU MAC-latency distribution (72.3 % served in 1 cycle, 99th percentile 5 cycles, max 8)
framed for soft-real-time. We report contention as percentages, without a comparable distribution/percentile
analysis.

---

## 4. The few directly comparable numbers

Most metrics are not apples-to-apples (their MAC is single-cycle integer in a model; ours is a multi-cycle FP
FMA in RTL). The genuinely comparable ones:

| Question | Tanase 2026 | This work |
|---|---|---|
| Cost to the CPU of lending its unit | CPU MAC-request latency **1.0 → 1.2 cyc (+20 %)** with AI active; dual-core arbitration overhead **0.8 %** | CPU's concurrent job slows **≤ 2 %** under CPU-strict, at **every** latency L = 0…5, with **one or two** coprocessors |
| Share the AI gets under CPU contention | NPU **18 %** of MAC slots (two busy cores) | coprocessor loses only **0–5 %** of its compute vs. alone (one CPU running a sparse FIR) — different contention scenario |
| Arbitration overhead | 0.8 % (dual-core) | 0–5 % accelerator slow-down; ≤ 2 % CPU slow-down (CPU-strict) |
| "Speedup" headline | **1.87×** — *dual-core general-purpose* parallelism | **5.46×** batched dense / **2.6×** dense / **4.80×** LeNet — *accelerator-vs-CPU ML*, all bit-exact |
| Area of the sharing logic | shared MUL+NPU engine **3019 LUTs, 13 DSP** (extrapolated FPGA) | arbiter **321 µm², = 2.5 % of one FMA** (real DC, TSMC 40 nm) |

Note the two "speedup" figures measure different things: theirs is "two cores ≈ 1.87× one core"; ours is
"the shared-FMA accelerator vs. the CPU doing the same ML." They are not competing numbers.

---

## 5. Bottom line (for the related-work section)

Tanase 2026 and this work share the **same core idea** — reuse the CPU's existing arithmetic unit for
opportunistic AI under a CPU-priority arbiter, instead of adding a dedicated one. The differences are what
make our contribution stand:

- **They model; we build.** Theirs is a SystemC behavioural model with extrapolated area/power and no
  FPGA/silicon; ours is real cv32e40px+FPnew RTL, simulated in Verilator and synthesized in Design Compiler.
- **They share a single-cycle integer MAC; we share the real multi-cycle FP FMA** and prove bit-exact
  floating-point inference on a real network — and we quantify the FMA-latency effect (the 1/(1+L) law) that
  a single-cycle model cannot even express.
- **They pay for a whole second CPU core to reuse a MAC; we add only a DMA coprocessor and a 321 µm²
  arbiter** — the more area-efficient realisation of the shared idea.
- Our arbiter study is broader (three swappable policies, full sweep) and our correctness guarantee is
  stronger (bit-exact, every configuration).

**Our honest gaps** relative to them are **energy/power/DFS**, **dual-core general-purpose parallelism**, and
a **detailed memory/cache and latency-distribution analysis**. The first (an energy/DFS study of the shared
FMA) is the most worthwhile to add, and would directly close the main dimension where they are ahead.

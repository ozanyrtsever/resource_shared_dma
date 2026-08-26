# A DMA Dot-Product Coprocessor that Time-Shares the CPU's Floating-Point Unit on CV32E40P / X-HEEP

**Master's Thesis — Design and Implementation Progress Report**

*Last updated: 2026-08 — final design complete. The coprocessor is a **pipelined single unit** that
time-shares the CPU's one FMA through a no-drain, owner-tagged, CPU-priority APU arbiter; performance,
area, and Fmax are measured. Earlier variants (the interleaved-pair dot product, the serial GEMV, the
dual coprocessor) are retained only as design-evolution history (§5, §11).*

---

## Abstract

This work presents the design, implementation, and evaluation of a memory-mapped, DMA-programmed
**reduction coprocessor** for the OpenHW **CV32E40P** RISC-V core, integrated into the EPFL **X-HEEP**
microcontroller. Its distinguishing feature is that the coprocessor instantiates **no floating-point
arithmetic of its own**: every multiply-accumulate is issued to the **CPU's single existing
fused-multiply-add (FMA)** inside the FPnew (CVFPU) unit, over the same Auxiliary Processing Unit (APU)
port the core itself uses. A small **CPU-priority arbiter**, inserted at the core's APU boundary,
time-multiplexes the FMA between the CPU and the coprocessor; it round-trips a 2-bit **owner tag** with
each in-flight operation so results route back correctly even when they retire out of order, which lets
it grant every cycle **without a drain** and never starve the CPU's own floating-point work. Because no
second floating-point datapath is added, the design delivers data-movement-overlapped reduction
throughput at the cost of a small arbiter rather than a whole FMA — the area/throughput trade-off that
is the quantitative claim of the thesis.

A single design subtlety shapes the final architecture. A pipelined FMA has latency *L*; a coprocessor
that issues one MAC and then waits for its result exposes that latency in full (`cyc/MAC = 1.14 + L`).
But a **batched** matrix product carries, per streamed weight, *B* **independent** multiply-accumulates
— one per batch lane — and issuing those back-to-back keeps the pipelined FMA full from a single
requestor. The final coprocessor (`dma_fp_dot_accel_pipe`) does exactly this, decoupling issue from
result-collection so up to *L*+1 of its MACs are in flight at once; its per-MAC cost is therefore
**independent of the FMA latency** (flat ≈ 1.12–1.14 cyc/MAC across *L* = 0..5), and **one** pipelined
unit saturates the FMA — retiring an earlier dual-coprocessor variant. On the X-HEEP SoC (Verilator),
the design runs a toy MLP at **5.45×** and real **LeNet-300-100 / MNIST** at **8.37×** over the CPU,
**bit-exact** with the CPU (it uses the CPU's own FMA) at **8/8** MNIST accuracy; a CPU floating-point
DSP kernel can run **concurrently** on the shared FMA at negligible cost. Synthesized in TSMC 40 nm at
its **260 MHz** operating point (Synopsys DC-NXT), the sharing mechanism costs a **1.8 k µm² arbiter —
8.4 % of one FMA, 2.0 % of the core** — i.e. per accelerator **3.9× less added area** than giving it a
dedicated FMA. The report documents the full path to this result across three parts. **Part I
(standalone feasibility, X0–X2)** brought up the DMA substrate and drove a dot product on the *real*
FPnew FMA; **Part II (on-SoC, Y0–Y1)** built a full FPU+DMA X-HEEP MCU — establishing a **303-cycle**
CPU dot-product baseline — and proved the accelerator tap from C; **Part III (Y2–Y3)** built the
CPU-priority sharing arbiter and the pipelined coprocessor and measured performance, area, and Fmax.
Throughout, the boundary between the design's evolution and its final form is stated explicitly.

---

## 1. Introduction and Motivation

### 1.1 Problem statement

Embedded RISC-V cores such as the CV32E40P are frequently deployed in signal-processing and
machine-learning-inference workloads whose inner loops are dominated by **dot products** and
multiply-accumulate reductions over arrays in memory. Two costs dominate such loops on a scalar,
in-order core:

1. **Instruction and load/store overhead** — every operand pair must be explicitly loaded by the
   CPU, multiplied, and accumulated, with the core stalling on memory latency and spending most of
   its issue slots on address generation and loop control rather than on arithmetic.
2. **Floating-point throughput** — the single scalar FMA processes one pair at a time, and its
   result feeds directly back into the next accumulation, so the loop is fundamentally
   latency-bound on the FMA's dependency chain.

The conventional remedy is a dedicated floating-point accelerator with its own multiply-add datapath
and its own data-movement engine. In an area-constrained embedded SoC, however, a second
floating-point datapath is expensive: the FMA, with its wide mantissa multiplier and alignment /
normalization logic, is one of the largest combinational blocks in the core.

### 1.2 Key idea and contribution

The central observation of this thesis is that, during data-movement-bound reduction kernels, the
CPU's own floating-point unit is **largely idle** — the core spends its time issuing loads and
managing the loop, not executing back-to-back FMAs. This thesis exploits that idle time:

> A DMA-programmed coprocessor streams operands from memory and feeds every multiply-accumulate into
> the **CPU's existing FMA** through the APU port, while a CPU-priority arbiter — routing each result
> by an owner tag so it never has to drain the pipeline — keeps the CPU's own floating-point
> instructions first in line. The coprocessor is **pipelined**: for a batched matrix product it issues
> the batch's independent MACs back-to-back, so a single unit keeps the shared FMA full regardless of
> its latency.

The contributions of the work are:

- **FMA time-sharing without a second FPU:** a reduction coprocessor with **zero added floating-point
  datapath**, multiplexing its operands onto the core's APU/FPnew interface.
- A **CPU-priority, owner-tagged, no-drain arbiter** that keeps the CPU's floating-point path first,
  routes out-of-order results to their correct owner, and preserves the CPU's architectural
  floating-point state bit-exactly.
- A **pipelined batched-reduction coprocessor** that issues *B* independent batch-MACs back-to-back to
  hide the FMA pipeline latency, so `cyc/MAC` is independent of *L* and one unit saturates the FMA —
  making a second coprocessor unnecessary.
- **Bit-exactness by construction** — using the CPU's own FMA and preserving each dot product's
  summation order — verified across every latency, policy, and co-execution configuration.
- A **quantified area / frequency study** (silicon-representative synthesis) showing the sharing costs
  an arbiter, not an FMA, together with a **co-execution** demonstration of the CPU and coprocessor
  sharing the one FMA concurrently and bit-exactly.

### 1.3 Scope of this report

This report documents the completed work end to end: the standalone-feasibility phases (X0–X2), the
on-SoC bring-up and accelerator tap (Y0–Y1), and the contribution itself — the CPU-priority sharing
arbiter and the pipelined coprocessor (Y2–Y3) — together with the performance, area, and frequency
measurements. Earlier design points (the interleaved-pair dot product of Phase X2, the serial
single-image GEMV, and the two-coprocessor variant) are described where they illuminate the final
design's rationale, and are explicitly marked as superseded so the boundary between the design's
evolution and its final form is unambiguous. Full measurement tables live in the companion reports
(Appendix A).

---

## 2. Background

### 2.1 The CV32E40P core, FPnew, and the APU interface

CV32E40P is a 4-stage, in-order, 32-bit RISC-V core (RV32IMFC) maintained by the OpenHW Group.
When configured with floating-point support, the core does **not** embed the FPU in its pipeline.
Instead, the FPU — an instance of the parameterizable **FPnew / CVFPU** floating-point unit — is
attached as an external functional unit through the **Auxiliary Processing Unit (APU)** interface.
In X-HEEP this FPU-capable variant of the core is called **`cv32e40px`**; its floating-point wrapper
(`cv32e40px_fp_wrapper` → `fpnew_top`) and APU connection are structurally identical to the
standalone `cv32e40p_fp_wrapper` used in Part I of this work.

The APU interface is a latency-insensitive, handshake-based port:

- The core drives a **request** (`apu_req`) together with the **operands** (read from the register
  file), the **operation** code (`apu_op`), and **format/rounding flags** (`apu_flags`).
- The FPU returns a **grant** (`apu_gnt`) to accept the request, and later a **response**
  (`apu_rvalid`) carrying the **result** (`apu_rdata`) and the **floating-point status flags**
  (`apu_rflags`, i.e. `fflags`).

Crucially, the operand source is the **register file**, and the FPnew FMA belongs to the **ADDMUL**
operation group. The precise encoding of a single-precision fused multiply-add, established
experimentally in Part I, is `apu_op = 0`, `apu_flags = 0` (FP32, round-to-nearest-even), with the
three operands supplied as `{acc, b, a}` to compute `a·b + acc`. The FMA latency is a synthesis-time
parameter (`FPU_ADDMUL_LAT`); at latency 0 the FMA is purely combinational, while at latency ≥ 1 it
is pipelined and the response arrives one or more cycles after the request is granted. This
parameter is central to the sharing design: the drain logic is only meaningfully exercised when the
FMA is pipelined.

A small but consequential implementation detail of the CV32E40P floating-point wrapper was verified
directly in the RTL: the wrapper hard-ties the FPnew `out_ready_i` input to `1`, ties `flush_i` to
`0`, and leaves the `busy_o` output unconnected, and its clock is gated by
`apu_clk_en = apu_req | apu_busy`. These tie-offs and the clock gate must be revisited when the FMA
is shared, because the sharing arbiter needs back-pressure and busy information that the stock
wrapper discards, and the FMA must remain clocked during coprocessor-only activity.

### 2.2 The OBI memory bus

Data movement in this SoC family uses the **Open Bus Interface (OBI)**, a lightweight RISC-V memory
protocol with two decoupled phases: an **address phase** (`req`/`gnt` handshake carrying
`addr`/`we`/`be`/`wdata`) and a **response phase** (`rvalid`/`rdata`, with no back-handshake — the
master must always accept the response). The two phases are independent, which permits outstanding
(pipelined) transactions. The DMA used here exposes read and write masters speaking OBI, which the
X-HEEP crossbar merges with the CPU's instruction and data ports.

### 2.3 DMA engines and the descriptor model

A DMA engine moves blocks of data between memory regions without CPU intervention. The CPU programs a
**descriptor** — source pointer, destination pointer, transfer size, strides, data type — into the
DMA's memory-mapped registers and then triggers the transfer; the DMA autonomously issues the bus
transactions and signals completion. This thesis builds the dot-product coprocessor on top of such an
engine so that operand streaming, address generation, and buffering are reused rather than
re-implemented, and so that the compute datapath is the only genuinely new block on the data path.
This is deliberate: rather than building a second, complete data-movement engine (as a stand-alone
accelerator would), the design reuses the SoC's existing DMA — mirroring, on the data side, the same
"don't duplicate expensive infrastructure" philosophy that the FPU-sharing applies on the compute
side.

### 2.4 X-HEEP and the hardware-FIFO accelerator interface

The integration platform is **X-HEEP** (eXtendable Heterogeneous Energy-Efficient Platform, EPFL): a
configurable, open-source RISC-V microcontroller that generates a complete SoC — core, memory,
crossbar interconnect, peripherals, DMA, boot ROM, and simulation/FPGA/ASIC flows — from a single
configuration file. X-HEEP was chosen because it is the **native home of the DMA used here** and it
provides the entire surrounding SoC "for free," letting the thesis focus on the compute datapath and
the sharing arbiter rather than on re-plumbing a system.

X-HEEP's DMA ships with a **hardware-FIFO (HW-FIFO) accelerator interface** intended for exactly the
use case of this thesis — *compute-in-DMA*. When a transfer is launched with the run-time flag
`hw_fifo_en` set, the engine **replaces its internal write FIFO with an external accelerator**: the
data that would have been pushed into the write FIFO is instead emitted on `hw_fifo_req_o` (a
`pop/push/flush/data` request), and the accelerator's responses (`empty/full/alm_full/data`) are
returned on `hw_fifo_resp_i` in place of the write FIFO's status. From the engine's perspective the
accelerator *is* the write FIFO; from the accelerator's perspective it receives a stream of input
words, performs an arbitrary transformation, and presents output words to be drained by the write
master and stored to the destination pointer. A dedicated input, `hw_fifo_done_i`, lets the
accelerator signal end-of-output; together with an empty write buffer it terminates the transfer
(`dma_write_done_override = write_buffer_empty & hw_fifo_done_i & hw_fifo_mode`). This done signal is
mandatory — without it the DMA would wait indefinitely — and it is precisely what allows an
accelerator to consume *N* input words but emit a *different* number of output words (e.g. a single
scalar reduction result). The interface is exposed per DMA channel at the SoC boundary as
`hw_fifo_req_o / hw_fifo_resp_i / hw_fifo_done_i [DMA_CH_NUM-1:0]`, and the FIFO structures are
defined once in `hw/core-v-mini-mcu/include/fifo_pkg.sv`.

A second, equally important property of this interface is that the accelerator's `full` / `alm_full`
status propagates as **back-pressure** all the way to the read master: when the accelerator cannot
accept data, the write-FIFO push stalls, the read FIFO fills, and the read master stops issuing
requests. This natural back-pressure lets the coprocessor stall its own operand stream whenever the
shared FMA is momentarily unavailable, so no in-flight data is ever lost — and, as the final design
shows, the owner-tagged arbiter reclaims the FMA for the CPU without having to drain the pipeline at
all (Phase Y2).

---

## 3. System Architecture

### 3.1 Overview

```
   CV32E40P (cv32e40px) core ──(APU: regfile operands, tag=CPU)──┐
                                                                  ├─►[dma_apu_arbiter]─► FPnew FMA ─► result
   DMA + pipelined coprocessor ──(fmadd operands, tag=acc)────────┘   ▲ CPU priority,        │  (+ owner tag)
        │                                                             │ no drain, grants     │
        │  DMA read master streams the weight matrix from SRC_PTR      │ every cycle          │
        │  HW-FIFO: read data → coprocessor (issue B MACs/weight to    └── tag-routed result ─┘
        │           the shared FMA, collect) → results → write master → DST_PTR
   [X-HEEP crossbar, CPU-priority] ── shared memory ── on-chip RAM
```

Two resources are shared between the CPU and the coprocessor, both arbitrated with **CPU priority**:

1. The **FPnew FMA**, shared through the `dma_apu_arbiter` inserted at the core's APU boundary inside
   `cv32e40px_top` — the contribution of the thesis (Phase Y2). The arbiter round-trips a **2-bit owner
   tag** with every operation, so a result returns to its correct requester even when operations retire
   out of order; this is what lets it **grant every cycle with no drain** — it never stalls the pipeline
   to switch owners, which is exactly what makes the coprocessor's pipelined multi-issue possible with
   no arbiter change. The grant policy is an elaboration-time parameter (only the selected one
   synthesizes): **P0** CPU-strict (default — the CPU always wins, the coprocessor takes the cycles it
   leaves), **P1** round-robin, **P2** QoS-weighted.
2. The **memory bus**, already shared through the X-HEEP crossbar, which merges the DMA's masters with
   the CPU's ports. (This required no new work — a benefit of building on X-HEEP.)

### 3.2 The layer protocol and data flow

The final coprocessor computes a **batched matrix product** `Y[M][B] = W[M][N] · X[N][B]` (`B = 1` is a
matrix-vector product). It is **input-stationary**: the `B` input vectors are buffered once, and each
streamed weight is reused across all `B` batch lanes (weight-reuse without weight-stationary storage).
One layer is two back-to-back DMA transfers on the coprocessor's HW-FIFO channel:

1. **LOAD** — a header `[N, M, B]` (dot length, output rows, batch), then the `B` input vectors, latched
   into the coprocessor's input buffer.
2. **WEIGHT** — the whole `M×N` weight matrix streamed once. A FIFO **flush** between the two transfers
   flips the coprocessor into its compute phase; per streamed weight it issues `B` MACs to the shared
   FMA (one per lane) and, after the row's `N` weights, emits the `M×B` results back over the same
   channel, terminating with `hw_fifo_done`. (LeNet layer-1 has `M·N = 235 200 > 65 535`, the DMA's
   16-bit size limit, so its weights travel as one 2-D transfer.)

This input-stationary, whole-matrix feed is the dataflow that makes the accelerator win: an earlier
interleaved-pair layout (Phase X2, §5.3) and per-neuron DMA reprogramming were *slower* than the CPU
because the host marshalling cost as much as the dot product itself; streaming the weights directly
from their natural contiguous layout removes that overhead entirely (§11).

### 3.3 The pipelined engine

The coprocessor (`dma_fp_dot_accel_pipe`) has **no local multiplier** — every MAC is one `fmadd`
issued to the shared FMA over the APU. Its engine is pipelined via a decoupled **issue / collect**
structure: an **issue pointer** offers one MAC per granted cycle across the `B` lanes for the current
weight (operands `{acc_q[iss_b], w_q, x[iss_b][i]}`); a **collect pointer** writes each returning
result to its lane. Because FPnew's ADDMUL lane is an in-order pipeline and the coprocessor's in-flight
MACs are all the same op and latency, they retire in issue order, so a simple round-robin collect
counter suffices — no per-op tag inside the block. Up to `L+1` MACs are in flight at once, keeping the
FMA full; a one-line hazard interlock (`inflight < B`) stalls issue only when the batch cannot cover
the latency (`B < L+1`), and for `B ≥ L+1` never fires. The input buffer is a 1R1W submodule
(`xbuf_ram`, an SRAM macro in silicon).

**Bit-exactness is structural.** Each output `Y[r][b] = Σ_{k} W[r][k]·X[k][b]` accumulates in order
`k = 0..N−1` per lane; the pipeline interleaves *across* lanes (independent sums), never within a
single dot product, so every summation order is unchanged — identical to a serial unit and, since the
coprocessor uses the CPU's own FMA, to the CPU, at every latency and under concurrent CPU FP traffic.
At runtime `B = 1` the interlock serializes issue, reproducing exact memory-bound behavior from one
RTL and one correctness argument.

### 3.4 Where the accelerator lives

X-HEEP deliberately exposes the HW-FIFO ports at the microcontroller boundary so that an accelerator
attaches **at the system/top level, outside the CPU core** — the platform's example accelerators do
exactly this. The coprocessor of this thesis is therefore an on-chip block that taps the DMA stream
and (in Phase Y2) reaches back into the core for the FMA. For the current simulation-based
verification the accelerator is instantiated in the generated simulation top (`testharness`), which
mirrors precisely where it would attach on a real chip; for the area study (Phase Y3) it will be
placed in the synthesizable system so that it is counted by the synthesis flow.

---

## 4. Methodology: a risk-deferred, two-part plan

The implementation is organized so that the hardest, most novel work — the floating-point sharing
arbiter — is undertaken **last**, only after every supporting layer has been independently verified.
Each phase ends with a concrete, self-checking pass criterion. The work naturally divides into a
standalone-feasibility part (Phases X, verifying the compute and DMA mechanisms in isolation) and an
on-SoC-integration part (Phases Y, inside the full X-HEEP system).

| Phase | Goal | Status |
|------|------|--------|
| **X0** | Bring up the bare DMA standalone: configure, generate the register file, resolve bus types, elaborate cleanly | **Complete** |
| **X1** | Standalone functional verification: prove the DMA moves data, including through the HW-FIFO accelerator path | **Complete** |
| **X2** | Floating-point dot-product accelerator on the HW-FIFO port, driven by the *real* CV32E40P FP wrapper | **Complete** |
| **— pivot —** | Move integration from a bespoke core testbench to the X-HEEP platform (rationale in §5.4) | **Done** |
| **Y0** | Generate a minimal FPU + DMA X-HEEP MCU, build it on Verilator, and establish the CPU-only dot-product baseline | **Complete** |
| **Y1** | Attach an FP-free reduction accelerator to the DMA HW-FIFO and prove the tap on the real SoC from C | **Complete** |
| **Y2** | **The contribution:** replace the accelerator's local arithmetic with the shared CPU FPnew via the APU arbiter (CPU-priority, owner-tagged, no-drain); evolve the datapath to the input-stationary batched-GEMM and finally the **pipelined single coprocessor** that hides FMA latency | **Complete** |
| **Y3** | Measurement: performance sweeps (L × policy, bit-exact) on toy MLP and real LeNet, co-execution, and the shared-vs-dedicated **area & Fmax** study at the 260 MHz operating point (no second FPU) | **Complete** |

---

## 5. Implementation Progress — Part I: Standalone feasibility (X0–X2)

Part I verified the two mechanisms the thesis depends on — the DMA's accelerator-injection path and
the floating-point dot-product datapath driving the real FMA — in a fast, isolated Verilator
harness, before committing to full-SoC integration.

### 5.1 Phase X0 — Standalone bring-up of the DMA

*What & why.* The DMA (`xheep_dma`, the DMA channel extracted from X-HEEP) is distributed as
templates parameterized over the concrete bus struct types. Phase X0 produced a configuration in
which the bare DMA elaborates cleanly with the HW-FIFO interface enabled, establishing a known-good
build from which all later work proceeds.

*How.* A configuration header compiled in only the HW-FIFO mode (`HW_FIFO_MODE_EN`), minimizing the
elaborated logic. The memory-mapped register block was generated from a hand-written register
description (`dma.hjson`) with the OpenTitan `reggen`/`regtool` flow, producing typed register
structures and the register file itself (built on `prim_subreg` primitives); writing the `SIZE_D1`
register was confirmed to raise the transfer trigger, and `STATUS.ready` to report idle/busy. The
`dma` module's six struct-type parameters were supplied through two small packages: a `fifo_pkg`
defining `fifo_req_t`/`fifo_resp_t`, and a user package defining the flattened OBI and register-bus
request/response structs (written explicitly to remove a fragile include-path dependency).

*Result & contribution.* The bare DMA elaborates with zero errors under the Verilator lint
front-end. The report notes, for honesty, that this artifact is a *reconstruction* (regenerated
register file, hand-authored bus types) whose upstream test pedigree does not transfer — motivating
the dedicated standalone verification of Phase X1.

### 5.2 Phase X1 — Standalone functional verification of the DMA

*What & why.* A compact, self-checking SystemVerilog testbench (`tb_dma_copy.sv`) verified the
reconstructed DMA in isolation — in seconds under Verilator rather than minutes in a full core
environment — for both a plain copy and, critically, the HW-FIFO accelerator path on which the whole
thesis rests.

*How.* The testbench drives the `dma` module directly with a clock/reset, two behavioral OBI
subordinate memories (one per master), a register-bus driver task that programs the descriptor and
polls status, and a golden comparison. Bring-up subtleties were captured: the clock-gate enable must
be driven high or the device receives no clock, and the trigger-slot input requires a non-zero slot
count (set to one, tied off).

*Result & contribution.* **X1a (plain copy, `HW_FIFO_EN = 0`)** passes — validating in one stroke the
hand-authored OBI struct field ordering, the descriptor programming sequence, the size-write
trigger, the completion/ready path, and both OBI handshakes. **X1b (HW-FIFO path, `HW_FIFO_EN = 1`)**
attaches a pass-through accelerator (a transparent FIFO) to `hw_fifo_req_o`/`hw_fifo_resp_i`; because
the engine provably bypasses its internal write FIFO in this mode, a correct copy is direct evidence
that every word traversed the accelerator-injection datapath. Both tests pass, establishing that the
compute-injection interface works and providing the structural scaffold for the dot-product
accelerator.

### 5.3 Phase X2 — Floating-point dot-product accelerator on the real FP wrapper

*What & why.* Phase X2 replaced the pass-through FIFO with the multiply-accumulate datapath and — the
decisive step — drove it with the **real `cv32e40p_fp_wrapper`** (FPnew), not a behavioral stub. This
proves the accelerator's protocol and arithmetic against the exact floating-point unit it will later
share, eliminating the FMA implementation as a variable before any sharing is attempted.

*How.* The accelerator (`dma_fp_dot_accel.sv`) is a hardware-FIFO block that pairs interleaved input
pushes into `(a, b)`, issues an APU-style FMA request to the FP wrapper, accumulates the result, and
after the programmed number of pairs presents the scalar accumulator as a single output word and
asserts `hw_fifo_done`. Its state machine (RECV_A → RECV_B → FMA_REQ → FMA_WAIT → RESULT → DONE)
tolerates both combinational (latency-0) and pipelined FMA. The FMA encoding was determined here:
`apu_op = 0`, `apu_flags = 0` (FP32, RNE), operands `{acc, b, a}`. Because Verilator has no
`shortreal`, hardcoded IEEE-754 bit patterns were used, and FPnew required a set of Verilator lint
waivers (`-Wno-BLKANDNBLK`, `-Wno-WIDTHCONCAT`, `-Wno-UNOPTFLAT`) — waivers that reappear, already
packaged, in the X-HEEP build.

*Result & contribution.* The accelerator computes a dot product of **18.0** (`0x41900000`) correctly,
and does so **latency-robustly at `FPU_ADDMUL_LAT` = 0 and 5** (the 5-cycle latency verified with an
accept→rvalid monitor). This establishes that the full compute datapath — pairing, FMA hand-shaking,
accumulation, termination — is correct against the genuine FPnew FMA, which is the exact unit the
sharing arbiter will later multiplex.

### 5.4 The pivot to X-HEEP

*What & why.* Part I verified the mechanisms standalone; the next step was integration into a full
system with a real CPU. The originally-planned route — a bespoke integration into the CV32E40P
`core-v-verif` UVM environment (and the unmaintained `example_tb`) — proved painful and
low-leverage: it would have required hand-building an SoC, a memory system, a bus arbiter, and
firmware plumbing, none of which is the thesis's contribution.

*Decision.* Integration was moved to **X-HEEP**, which is the DMA's native platform and supplies the
entire SoC (core, DMA, crossbar, memory, software stack, and a Verilator simulation flow) out of the
box. A multi-agent survey of X-HEEP confirmed that (i) the FPU is attached via the internal APU
exactly as in the standalone wrapper, (ii) the HW-FIFO accelerator ports are exposed at the SoC
boundary with the same `fifo_pkg` structure, and (iii) the X2 accelerator and the Y2 arbiter plan
port over directly. `core-v-verif` is retained only as an optional lock-step co-simulation
environment for later verification. This pivot re-scoped the phases from X3–X5 to Y0–Y3 but preserved
the plan's substance: bring up the system, prove the tap, then share the FMA, then measure.

---

## 6. Implementation Progress — Part II: On-SoC integration (Y0–Y1)

### 6.1 Phase Y0 — Full FPU + DMA MCU bring-up and the CPU baseline

*What & why.* Y0 had two goals: (1) resolve the one genuine unknown — whether the full SoC, including
the FPnew FMA, elaborates and *simulates correctly on Verilator* (open-source, the only simulator
available here) — and (2) establish the **CPU-only dot-product baseline** against which the
coprocessor will be measured.

*How.* No stock X-HEEP configuration enables both the FPU-capable core and the DMA's hardware-FIFO
mode, so a configuration (`configs/cv32e40px_fpu_dma.hjson`) was authored: the FPU-capable
`cv32e40px` core with `cpu_features: { fpu: true }`, together with the DMA in `hw_fifo_mode_en`. From
this, `make mcu-gen` generates the SoC RTL; `make verilator-build` builds the cycle-accurate model;
and the bundled `example_matfloat` application — a 32-element single-precision vector-add and dot
product with cycle counting via the `mcycle` CSR — serves as the baseline vehicle. A subtlety was
identified by reading the application: as shipped it times only the vector-add and prints nothing in
simulation, so a small edit brackets the dot-product loop with `mcycle` and prints the isolated
count. Because the default compiler target omits the `F` extension (which would silently fall back to
software floating point and never exercise the FMA), the application was deliberately built for the
hard-float architecture `rv32imfc_zicsr`.

*Result & contribution.* The simulation reports **`Program Finished with value 0`** — the application
passes, meaning the full SoC (`cv32e40px` + FPnew + DMA + crossbar) **elaborates and simulates
correctly on Verilator**, and the dot product is computed correctly on the hardware FPU. This
resolves the thesis's single real feasibility unknown. Disassembly of the kernel confirms the
compiler emits a **single hardware `fmadd.s` per element** (`res = A[i]·B[i] + res`) surrounded by
two loads, two address increments, and a branch — i.e. one fused multiply-add of genuine arithmetic
against roughly five instructions of loop/addressing overhead, with the accumulation serialized on
the FMA's dependency chain. The measured **CPU baseline is 303 cycles** for the 32-element
single-precision dot product (≈ 9.5 cycles/element). This number both quantifies the overhead the
DMA coprocessor targets (the loads, addressing, and branch that a streamed operand feed eliminates)
and defines the apples-to-apples comparison for the shared design, since the coprocessor will perform
the identical one-FMA-per-element arithmetic on the identical FMA.

*Reproducibility note.* Standing up the open-source toolchain required several documented, deliberate
deviations from a stock install, recorded here for the thesis's reproducibility appendix: the X-HEEP
Python environment was built with Python 3.10 and its unused `yamlfmt` dependency (which pins an
ancient `ruamel.yaml` that will not compile on modern Python) removed; `setuptools` was pinned below
82 (which removed `pkg_resources`, needed by the register generator); the RISC-V toolchain root was
pointed at the installed `corev` toolchain; and Verilator was built from source at version **5.040**
(the version X-HEEP officially targets) after the distribution's 5.020 hit a trace-code-generation
bug on the full design. These are environment concerns, not design changes, but they are part of
reproducing the result.

### 6.2 Phase Y1 — Proving the accelerator tap on the real SoC

*What & why.* Before sharing the FMA (Y2), Y1 proved — on the actual SoC, driven from C — that a
custom accelerator can be tapped into the DMA's hardware-FIFO datapath and correctly receive the
streamed data, process it, return a result to memory, and terminate the transfer. To isolate this
integration question from the floating-point-sharing question, the Y1 accelerator is deliberately
**floating-point-free**: an integer sum-reduction. Crucially, this is not throwaway scaffolding — the
integer reduction exercises the *exact* control structure the dot product needs (consume *N* words,
reduce, emit one result, assert done, reconcile the *N*-in/1-out length mismatch); Phase Y2 will
replace only its arithmetic (integer add → shared-FMA), leaving the surrounding integration
untouched.

*How.* A minimal accelerator (`dma_sum_accel.sv`) consumes *N* 32-bit words from the HW-FIFO,
accumulates their integer sum, presents the sum as one output word, and asserts `done`. It imports
X-HEEP's `fifo_pkg` — a necessary detail, since the standalone `xheep_dma` copy of that package
declares the `push`/`pop` and `full`/`empty` fields in the opposite order, so the accelerator must
reference fields by name through the platform's own package to be wired correctly. Its state machine
(ACCUM → RESULT → DONE) uses the FIFO status the DMA reads as its write-FIFO: it accepts pushes while
accumulating (`full = 0`, `empty = 1`), and once the *N*-th word arrives it presents the sum
(`empty = 0`) for the DMA to pop, then asserts `done` so the transfer terminates via the
`hw_fifo_done` override. The accelerator was attached on **DMA channel 1** (leaving the platform's
example accelerator on channel 0 untouched) by editing the simulation-top *template*
(`tb/testharness.sv.tpl`, since the elaborated `testharness.sv` is generated) and adding the source
to the testbench file list. A C application (`sw/applications/y1_sum_test`) programs a HW-FIFO DMA
transfer on channel 1 (`hw_fifo_en = 1`, `channel = 1`), polls for completion, and checks the result
against a software golden sum.

*Result & contribution.* The simulation prints **`dst[0] = 136 (golden = 136)`, `Y1 HW-FIFO tap:
PASS`**, exit code 0. The DMA streamed sixteen words `1…16` from source memory into the accelerator,
which reduced them to their sum (136), which the DMA then wrote to the destination and completed on
the accelerator's `done`. This proves, end to end on the real SoC in Verilator, the complete
accelerator-injection datapath — DMA → accelerator → DMA → memory — together with the `push`/`pop`,
`empty`/`full` back-pressure, and `done` handshake. Every piece of on-SoC plumbing the contribution
needs (the top-level wiring, the file-list integration, the FIFO-package convention, the C driver
sequence, and the *N*-to-1 reduction with `done`-based termination) is now verified and carries
directly into Phase Y2.

---

## 7. Design Decisions and Rationale

- **Reuse the SoC's DMA rather than build one; share the CPU's FMA rather than add one.** The two
  central design choices are the same idea applied to data movement and to compute: do not duplicate
  expensive, already-present infrastructure. This is what makes the coprocessor cheap.
- **Compute tap = HW-FIFO mode.** The accelerator is attached through the interface the DMA
  explicitly provides for accelerators, as an external block with its own state, rather than by
  modifying the engine's internal stages. This keeps the DMA RTL untouched and isolates the
  contribution.
- **Operand sourcing = input-stationary batched GEMM.** The `B` inputs are buffered once and each
  streamed weight is reused across all lanes, so the DMA delivers a single contiguous weight stream
  from its natural layout — no second read master, no host-side operand marshalling. (The earlier
  interleaved-pair layout of Phase X2 was retired precisely because that marshalling cost as much as
  the dot product; §11.)
- **Result returned to memory, termination by `done`.** The scalar result is written back through the
  write master and read by the CPU; the *N*-in/1-out mismatch is handled by `hw_fifo_done_i`, not by
  the write counter — validated in Y1.
- **Sharing arbiter at the core's APU boundary (`cv32e40px_top`).** The floating-point arbiter is
  inserted at the APU seam inside the core wrapper, not inside the DMA, keeping the sharing mechanism
  independent of the DMA choice and confining the core modification to a single, well-defined,
  documented fork of vendor RTL.
- **Floating-point-free first (Y1), shared-FMA second (Y2).** Proving the on-SoC tap with an integer
  reduction isolates the integration from the arithmetic, so that when the FMA is shared every other
  variable has already been eliminated.
- **Risk deferred to the end.** Everything except the sharing arbiter is verified first; the arbiter
  is the last thing brought up.

---

## 8. Verification Methodology

Verification proceeds bottom-up, one self-checking layer at a time:

1. **Elaboration / lint** (Verilator front-end) establishes structural soundness and type
   resolution. *Done, standalone and on-SoC.*
2. **Standalone unit testbenches** (Verilator, behavioral OBI memories + software golden model) check
   each datapath in isolation and in seconds. *Done for the DMA copy, the HW-FIFO tap, and the
   dot-product accelerator against the real FMA (X1–X2).*
3. **On-SoC C tests** program the descriptor from firmware, trigger the transfer, read back the
   result, and self-check against a CPU-computed reference through a memory-mapped exit code. *Done
   for the CPU baseline (Y0) and the accelerator tap (Y1); will be extended to the shared-FMA dot
   product in Y2.*
4. **Latency × policy sweep.** The FMA latency (`FPU_ADDMUL_LAT` = 0..5) and arbiter policy (P0/P1/P2)
   are swept — 21 configurations per benchmark — each checked **bit-exact** against a CPU golden
   forward pass. This both exercises the owner-tagged routing at every latency (where a naïve design
   would hang or corrupt state) and demonstrates the pipelining result (`cyc/MAC` flat in *L*). *Done,
   toy MLP and real LeNet.*
5. **End-to-end model correctness.** The real LeNet-300-100 predictions are checked against the MNIST
   labels and an offline NumPy reference — **8/8** correct across every configuration — so bit-exactness
   is confirmed not just against the CPU's own loop but against an independent float32 model. *Done.*

All five layers are complete; the design is verified bit-exact end to end across the full latency and
policy sweep on both a toy MLP and a real published network.

---

## 9. Current Status and Next Steps

**Completed — the whole design and its evaluation.** The pipelined single coprocessor and the
CPU-priority, owner-tagged, no-drain APU arbiter are integrated on X-HEEP and verified end to end. The
coprocessor issues every multiply-accumulate to the CPU's own FPnew FMA; a full `L × policy` sweep
(21 configurations each) on a toy MLP and on real LeNet-300-100 is **bit-exact** at every point, with
LeNet **8/8** MNIST-correct. Performance shows `cyc/MAC` **flat in L** (≈ 1.14 / 1.12), **5.49× / 8.44×**
over the CPU, ~87–88 % FMA utilization; co-execution has the CPU's own FP DSP kernel running
concurrently on the shared FMA at negligible cost, and a pure-FMA stress kernel (§11.11) makes the
CPU-priority policy separate sharply from round-robin under genuine per-cycle contention (CPU slowed
+6–11 % under P0 vs +64 % under P1) while exposing a CV32E40P limit of ≈ 2 outstanding FP ops; and
Synopsys DC-NXT (TSMC 40 nm) gives the area and
Fmax at the converged **260 MHz** operating point with a per-component breakdown, from which the "no
second FPU" claim is quantified (arbiter = 8.4 % of one FMA; 3.9× less added area per accelerator).
Verilator cycle counts convert to wall-clock at that operating point (× 3.845 ns). Full tables are in
the companion reports (Appendix A).

**Open / optional (not blocking).** (i) FMA-specific critical path — `report_timing -through` is wired
into the flow; a 0.1 ns re-run yields the FMA's own fmax versus *L*. (ii) Full-SoC synthesis for the
absolute chip clock and full-SoC wall-clock (the 260 MHz figure is the *core*'s fmax). (iii) Energy per
MAC via DC power (shared vs dedicated — the strongest "no second FPU" argument). (iv) A `base`
(no-arbiter) core run to express the arbiter's clock cost as a delta. (v) Verify the related-work
citations (§10). The core measurement and area programme is otherwise complete.

---

## 10. Related Work and Novelty Positioning

A literature survey places this work at the intersection of three lines of research; the specific
combination it embodies was not found in the published literature. *(Citations marked "verify" should
be confirmed before inclusion in the final bibliography.)*

**(A) Streaming operands to keep the FMA utilized.** The premise — that the FMA is the costly
resource and the load/address/loop overhead around it is what wastes it — is shared with **Snitch**
(Zaruba et al., 2020; Stream Semantic Registers + FREP feed a *dedicated* FPU by eliding explicit
loads via ISA extensions) and **NTX** (Schuiki et al., DATE 2019; a floating-point *reduction*
streaming co-processor loosely coupled to a RISC-V core). Both validate the motivation but attach
their *own* FPU. This thesis pursues the same goal — stream operands to keep the FMA busy — but does
so with a DMA and, crucially, **shares the host core's existing FMA** rather than adding one.

**(B) Shared / time-multiplexed floating-point units.** Sharing an FPU is established for *peer
cores* with *fair* arbitration: US Patent 6,148,395 (multiple CPUs share one FPU via a dispatch
arbiter, result steering); the "Florian" shared-FPU architecture (IJERT 2025, *verify*) and
many-soft-core shared FPUs (round-robin FIFO); and — the closest prior art — Tanase's energy-efficient
dual-core RISC-V that shares a *MAC* unit with CPU-priority arbitration for an opportunistic NPU
(C. A. Tanase, *Computers* 2026, 15(4), 219) — but that MAC is a *single-cycle integer* unit, the sharers
are two *peer cores* plus the NPU, and the whole system is a *SystemC model* (no FPGA/silicon; area/power
extrapolated). A full advantages/gaps comparison is in `COMPARISON_vs_tanase2026.md`. Related patents cover shared MAC units (US 6,223,196) and
priority-toggled shared functional units (US 7,533,248). This thesis differs in being **asymmetric
(strict CPU priority with an explicit drain/resume), between a CPU and a DMA coprocessor (not peer
cores), on the floating-point FMA (not an integer MAC)**.

**(C) Compute-in-DMA.** Performing computation inside the data-movement engine is an accepted
technique — e.g. the **OpenTitan DMA** computes inline SHA-256 over the stream it moves — establishing
that "compute-in-DMA" is real. Such engines, however, carry their *own* compute datapath; none reuse
the host's FPU.

**Contrast with RISC-V FP coprocessors.** The dominant pattern *adds* a floating-point datapath:
`fpu_ss` / F-HEEP (a CV-X-IF FP subsystem with its own FP register file), RedMulE (a dedicated FP16
GEMM engine), Quadrilatero (a CORE-V-XIF matrix coprocessor with its own FPU), and the PULP HWPE
family. All instantiate new floating-point hardware; this thesis deliberately does not.

**Novelty statement.** *Unlike prior shared-FPU designs, which multiplex a floating-point unit among
peer cores with fair (round-robin) arbitration, and unlike streaming FP accelerators such as Snitch
and NTX, which feed a dedicated FPU, we time-share the host core's single existing FMA between the CPU
and a DMA-fed reduction coprocessor through the APU interface, under an asymmetric strict-CPU-priority,
owner-tagged, no-drain arbiter — adding no floating-point datapath — and we pipeline the coprocessor so
one unit hides the FMA's latency and saturates it alone.* The three closest individual works to
differentiate against are **NTX** (FP reduction + streaming, but a dedicated FPU), the **dual-core
RISC-V MAC-reuse** design (CPU-priority sharing, but an integer MAC between peer cores), and **Snitch**
(elides overhead to feed a dedicated FPU). *(Coverage note: the streaming and compute-in-DMA axes
warrant a deeper dedicated search before the final related-work chapter.)*

---

## 11. Measurement and Evaluation

> **Reading guide.** This section traces the design's *evaluation journey* — from the first serial,
> single-image measurements, through the dense-layer / convolution / LeNet demonstrations and the
> two-coprocessor experiment, to the **final pipelined single coprocessor** (§11.10) that retires them
> all. Earlier subsections' absolute numbers are therefore **superseded** by the pipelined design; they
> are kept because each established a fact the final design relies on — the input-stationary dataflow,
> bit-exactness on real models, the co-execution guarantee, and the latency-hiding diagnosis that
> motivated pipelining. **Authoritative final numbers** live in the companion reports:
> `PERF_BENCH_PIPE_SWEEP.md`, `PERF_LENET_PIPE_SWEEP.md`, and `DC_STUDY_OPERATING.md`.

### 11.1 Length scaling (serial single — superseded)

The first quantitative study sweeps the dot-product length *N* and, for each *N*, measures both the
CPU-only kernel and the shared-FMA coprocessor path on the same SoC. A single firmware
(`sw/applications/y3_sweep/`) computes a CPU reference dot product (an `mcycle`-bracketed hard-float
accumulation loop) and then the coprocessor dot product (an `mcycle`-bracketed HW-FIFO DMA on channel
1, from `dma_launch` to completion), and self-checks the coprocessor result against the CPU golden.
`num_pairs` is fixed in the elaborated design, so the model is rebuilt per length; every point was
verified functionally (the coprocessor result equalled the golden at all *N*), and both figures come
from the same firmware for consistency.

| *N* | CPU (cycles) | Coprocessor (cycles) | Speed-up |
|----:|-----:|-----:|-----:|
| 8   | 82   | 150  | 0.55× |
| 16  | 163  | 162  | ~1.00× (break-even) |
| 32  | 323  | 208  | 1.55× |
| 64  | 643  | 298  | 2.16× |
| 128 | 1282 | 482  | 2.66× |

Both costs are almost perfectly linear in *N*: **CPU ≈ 10·N cycles** (≈10 cycles per element, with
negligible fixed cost — every element pays for its two loads, two address increments, the FMA, and
the branch), and **coprocessor ≈ 116 + 2.86·N cycles** (a fixed ≈116-cycle DMA-offload setup — first
read latency, pipeline fill, result write-back — plus only ≈2.86 cycles per element, since the
operand stream from the DMA eliminates the CPU's per-element instruction overhead). Two consequences
follow, and both are central thesis results:

1. **Break-even at N ≈ 16** (`116 = (10 − 2.86)·N`). Below it the CPU wins, because the fixed offload
   cost is not amortised; above it the coprocessor wins and the margin grows with *N*. This is the
   honest operating envelope of the design — the coprocessor is worthwhile for reductions above a
   small threshold, not universally.
2. **Asymptotic speed-up ≈ 3.5×** (the ratio of per-element costs, 10 / 2.86), approached as *N* grows
   (2.66× already at N = 128), at **zero added floating-point datapath**, since the coprocessor borrows
   the CPU's FMA.

Crucially the ≈3.5× per-element efficiency comes entirely from *removing the instruction overhead
around the FMA* (loads, address generation, loop control), not from the FMA itself — the same FMA is
used either way, one operation per issue. The remaining Y3 work targets both terms of the coprocessor
cost model: a pipelined (multi-partial-accumulator) datapath to lower the per-element term toward one
cycle per element at `FPU_ADDMUL_LAT ≥ 1`, and a leaner launch to lower the fixed term (shifting
break-even left); plus the `FPU_ADDMUL_LAT` sweep (the CPU-priority drain cost) and the area
comparison (coprocessor enabled vs. disabled via `+define+COPROC_FPU_SHARE`).

### Concurrency: what the coprocessor costs an independently-running CPU

The length study measures the coprocessor in isolation; a second study asks the operationally more
important question — **while the coprocessor is streaming a dot product on the shared FMA, what
latency does it impose on the CPU running its own, unrelated code?** Two firmwares probe the two
regimes, each measuring a CPU kernel first alone (coprocessor idle) and then concurrently
(coprocessor launched in the background on channel 1), and each self-checking that both the
coprocessor result and the CPU result remain correct.

**(a) CPU also executing floating-point (`sw/applications/y3_contention/`).** A hard-float MAC loop
runs concurrently with the coprocessor. Both results are correct, and the CPU loop slows by ≈19
cycles (~6%). Because at `FPU_ADDMUL_LAT = 0` the FMA pipeline is empty almost every cycle, the
CPU-priority path grants the CPU its FMA immediately whenever it asks; the 19-cycle cost is therefore
dominated not by FMA arbitration but by **memory-bus contention** — the coprocessor's DMA read stream
and the CPU's operand loads competing on the shared crossbar.

**(b) CPU executing unrelated non-floating-point work (`sw/applications/y3_independent/`).** This is
the design's sweet spot — the CPU does control/integer work while the FMA would otherwise sit idle.
Two kernels are measured, each alone and then concurrently with the active coprocessor:

| CPU kernel (coprocessor active) | Alone | Concurrent | Added latency |
|---|---:|---:|---:|
| compute-bound (register-only integer) | 179 | 179 | **0 cycles** |
| memory-bound (integer array) | 241 | 246 | **5 cycles** |

The compute-bound kernel pays **zero** — the coprocessor's floating-point activity is completely
invisible to CPU code that does not itself issue floating-point. The memory-bound kernel pays only a
few cycles of the same crossbar contention seen in (a). The waveform confirms the mechanism directly:
throughout both coprocessor bursts the CPU's FMA-request line (`cpu_req_i` at the arbiter) is **flat
zero** — the CPU-priority stall/drain logic is never even exercised — so any residual CPU delay can
come only from the memory bus, never from FMA sharing. *(Methodological note: the compute-bound kernel
had to be seeded from a `volatile` to defeat the compiler's common-subexpression elimination, which
otherwise folded the second identical call and reported a spurious negative delta.)*

Two attributions matter for the thesis. First, **at combinational FMA latency the shared FMA imposes
essentially no latency on independent CPU code**, and this is *latency-independent* — because
non-floating-point code never requests the FMA, it pays nothing regardless of `FPU_ADDMUL_LAT`.
Second, the only residual cost is **pre-existing DMA memory-bus arbitration, not the sharing
mechanism**: the same one-cycle `data_gnt` stalls appear in the baseline with the coprocessor idle —
they are present in the "alone" measurement and cancel in the alone-vs-concurrent difference — so any
DMA moving the same data would incur them.

### The CPU-priority drain cost versus FMA latency

The concurrency study above is at `FPU_ADDMUL_LAT = 0` (combinational FMA), where the CPU-priority
path never sits on the critical timing path. To characterize the *drain cost* — the price the CPU pays
when it must wait for an in-flight coprocessor FMA to leave the shared pipeline before issuing its own
— the FMA latency was swept from 0 to 5 (the `fpu_addmul_lat` MCU-config field), re-running the
CPU-floating-point-concurrent test (`y3_contention`: 32 CPU MACs alongside a 32-pair coprocessor dot
product) at each. *(Enabling the field required forwarding it in X-HEEP's config loader, which omitted
it for the cv32e40px; see Appendix C.)*

| `FPU_ADDMUL_LAT` | CPU alone (cyc) | CPU under contention | Δ (drain) | slowdown |
|---:|---:|---:|---:|---:|
| 0 | ~317 | ~336 | 19 | 6.0% |
| 1 | 357 | 378 | 21 | 5.9% |
| 2 | 389 | 403 | 14 | 3.6% |
| 3 | 421 | 439 | 18 | 4.3% |
| 4 | 453 | 497 | 44 | 9.7% |
| 5 | 485 | 561 | 76 | 15.7% |

Two regimes are visible, with a knee near L = 4. **For L ≤ 3 the added CPU cost is a flat ≈ 14–21
cycles and *non-monotonic* in L** — the signature that at low latency the cost is dominated by the
pre-existing memory-bus arbitration (whose collision alignment jitters as L shifts the instruction
schedule), not by FMA drain: a drain-driven cost would rise monotonically. Two effects keep the drain
negligible here: the coprocessor's single-accumulator datapath serializes its own FMAs (each
`fma(a,b,acc)` waits for `acc`, so the shared pipeline is rarely full), and the CPU's own
dependency-chain stalls (waiting L cycles for each MAC result) create shadows in which the coprocessor
advances without charging the CPU. **From L = 4 the drain cost emerges** (44 cycles) and roughly
doubles by L = 5 (76 cycles, ≈16%), as an in-flight coprocessor FMA now occupies the shared pipeline
long enough to stall a CPU issue for up to L cycles. Across the entire sweep **both results remain
bit-correct** — validating the arbiter's drain/owner routing at every latency on the real SoC — and
the CPU is never starved: its worst-case slowdown is bounded and modest (≤16% even at L = 5), the
concrete form of the CPU-priority guarantee. (The CPU's own `alone` cost also rises with L, 317→485,
as each MAC pays the deeper FMA latency; the Δ isolates only the *sharing* cost, since that baseline is
present with or without the coprocessor.) In practical terms, the stock X-HEEP FPU is combinational
(L = 0) and typical pipelined FPnew configurations use L = 1–2, so the design operates in the flat,
near-free regime for realistic latencies; the deep-latency cost is characterized for completeness and
remains bounded by construction.

### A clean baseline: the entire coprocessor behind a single define

For the baseline-vs-augmented cycle comparison and the forthcoming area study, the whole contribution
must be removable by one switch. An audit confirmed that every addition — the arbiter instance and its
wrapper-side nets in `cv32e40px_top`, and the eight-signal coprocessor APU port threaded through all
six hierarchy levels — is guarded by `` `ifdef COPROC_FPU_SHARE ``, with each port declaration and its
connection removed together, and with `cv32e40px_top`'s `` `else `` branch reproducing the stock
core→FMA wiring bit-for-bit. One site was initially unguarded — the accelerator instance in the
testbench — and was corrected: it is now wrapped in the define, with an `` `else `` branch that ties
off its channel-1 HW-FIFO (`accel_done = 0`, `hw_fifo_resp[1] = '0`). With the define commented out in
`core-v-mini-mcu.core`, the design therefore elaborates as **stock X-HEEP** (cv32e40px + FPnew + DMA,
no coprocessor, no arbiter). This single-define clean revert is what makes the area comparison honest:
the same source tree yields both the augmented design and its exact baseline.

*(A clarification recorded during the audit: the accelerator taps the MCU's **integrated** DMA — the
`dma_subsystem` inside `core_v_mini_mcu`, channel 1 — on the shared system bus, which is why its reads
genuinely contend with the CPU. The separate `dma_i` visible at the testbench top level is the
unrelated off-MCU **external-DMA example** guarded by `USE_EXTERNAL_DEVICE_EXAMPLE`, idle in all of
our runs; tapping it would have been both unrepresentative — a separate bus — and non-canonical.)*

### The area cost: sharing versus a dedicated FMA

The central quantitative claim — that sharing the CPU's FMA avoids the area of a second one — was
measured in Synopsys Design Compiler NXT (TSMC 40 nm G, RVT, SS/0.81 V/125 °C corner, 100 MHz;
DesignWare arithmetic; each module synthesized **standalone** so no logic is optimized away). Every
module was re-synthesized in one consistent run of the current 3-requestor design, and the ADD/MUL
pipeline latency L was swept 0..5; at this relaxed clock all timing slacks are large, so the areas are
area-optimal rather than timing-inflated. *(These are the early standalone-per-module figures at a
relaxed 100 MHz; the **final, authoritative area** is measured differently — every component extracted
in-context from one hierarchy-preserved `cv32e40px_top` synthesis at the converged **260 MHz operating
point** — and is reported in `DC_STUDY_OPERATING.md`. There the FMA is 21.5 k µm² and the P0 arbiter
1.8 k µm² = 8.4 % of one FMA / 2.0 % of the core, giving 3.9× less added area per accelerator than a
dedicated FMA. The absolute numbers below differ from the operating-point report for exactly this
methodology reason; the **ratio** — arbiter ≪ FMA — is the invariant claim and holds in both.)*

| Block | Role | Cell area (µm², L = 0) |
|---|---|---:|
| `cv32e40px_fp_wrapper` (the whole FPU) | what a "just add a second FPU" design would duplicate | **12 455** |
| `fpnew_fma_multi` (the FMA lane alone) | the unit the coprocessor actually **reuses** | **5 789** |
| `dma_apu_arbiter`, P0 CPU-strict | the sharing logic we **add** (default) | **452.94** |
| `dma_apu_arbiter`, P1 full round-robin | (removes CPU priority) | 465.07 |
| `dma_apu_arbiter`, P2 QoS weighted | (tunable-share variant) | 927.60 |
| `dma_fp_dot_accel_is` logic | the accelerator datapath (present either way) | 4 461 |
| `x_buf` input buffer | a 32 KB SRAM macro (both designs) | ~0.11 mm² (memory compiler) |

The FMA lane is about **half of the whole FPU** (5 789 of 12 455 µm²); the other half is
divide/square-root, compare/min-max and format conversion — hardware a multiply-add coprocessor does
not need. So the honest "one FMA" the coprocessor avoids duplicating is **5 789 µm²**.

The comparison is therefore stark. A dedicated dot-product accelerator would instantiate its own FMA
(**+5 789 µm²**); our design instead adds only the CPU-priority arbiter (**+452.94 µm²**). The
accelerator logic (4 461 µm²) and its 32 KB input SRAM are common to both and cancel. The net area
saved by sharing is one whole FMA minus the arbiter — **5 336 µm² per coprocessor** — and the price of
sharing is an arbiter that is **just 7.8 % of one FMA (3.6 % of the whole FPU)**. For the two-coprocessor
design a single 3-requestor arbiter replaces **two** FMAs, saving **11 125 µm²**. Because the arbiter
contains no FMA, its area is independent of the pipeline latency, whereas the FMA grows +30 % from L = 0
to L = 5 (5 789 → 7 521 µm²) — so sharing is *more* favourable at deeper pipelines (the arbiter falls
from 7.8 % to 6.0 % of the FMA). Swapping the grant policy costs area only if you want it: CPU-strict and
full round-robin are ~453–465 µm², while the QoS-weighted policy (deficit-credit counters + an argmax) is
~928 µm², about twice the others. This is the thesis's "no second FPU" claim, quantified: the shared-FMA
coprocessor delivers the dot-product, batched-GEMM and full-network acceleration characterized above at
the cost of a ~453 µm² arbiter instead of a whole ~5 789 µm² FMA — essentially **zero added
floating-point-datapath area**.

### Real ML kernel: a dense layer, and why the feeding dataflow decides the win

Beyond microbenchmarks, the coprocessor was exercised on a real neural-network primitive — a
fully-connected (dense) layer, the dominant compute of MLPs and the FC heads of CNNs. A dense layer
`y = Wx` (M outputs, N inputs) is M dot products of the shared input vector `x` with the rows of the
weight matrix `W`. Three successive integrations, each measured on one N = M = 32 layer against the
CPU's own hard-float loop (8 580 cycles), show that the **operand-feeding dataflow — not the compute —
governs whether the accelerator helps**:

| Integration | accel (cyc) | vs CPU | Bottleneck |
|---|---:|---:|---|
| Interleaved input | ~17 200 | 2.0× slower | host marshals `[x,w]` pairs per dot (O(N) work ≈ the dot itself) |
| Input-stationary | 14 047 | 1.6× slower | `x` buffered once, `W` streamed directly — but the DMA is reprogrammed per neuron |
| **Input-stationary GEMV** | **3 266** | **2.6× faster** | one DMA transfer streams the whole `W`; the accelerator emits M results, one per N-word row |

The natural first attempt — interleave `x` and `w` into the accelerator's paired input stream — is
*slower* than the CPU, because building the interleaved operand buffer is O(N) host work comparable to
just computing the dot on the CPU (a cost the isolated microbenchmark hid by pre-building the buffer
outside the timed region — an important honesty correction: the 208-cycle single-dot figure was
compute-only). Adopting the established **input-stationary** dataflow — buffer the reused activation
`x` once, stream the weights directly from their natural contiguous layout — removes host marshalling
entirely. Folding the per-neuron DMA setup into a single **whole-matrix (GEMV) transfer**, where the
accelerator finalizes and emits one result per N words, removes the last orchestration overhead and
makes the shared-FMA coprocessor **2.6× faster than the CPU on a real dense layer** — bit-exact, no
added floating-point datapath, at N = M = 32 (the margin grows with layer size). This mirrors how
modern MCU/edge accelerators (PULP's HWPE / RedMulE) feed operands via hardware streamers from natural
memory layout — realized here with X-HEEP's DMA as the streamer and an input-stationary buffer in the
coprocessor. The coprocessor's `x`-buffer adds area, but it is common to any accelerator (shared or
dedicated) and so does not affect the "no second FPU" comparison, which remains arbiter-vs-FMA.

### From a dense layer to a real convolution — the same coprocessor, a different feed

A convolution — the workhorse of every CNN — maps onto the *same* GEMV coprocessor with no hardware
change, only a different DMA feeding pattern. Taking the first convolutional layer of a pretrained
StarDist nucleus-segmentation network (a 3×3×1 → 32-channel layer producing a 6×6 output map, 1 152
output activations), each output pixel is an `im2col` dot product: its 3×3 receptive field is
flattened into the length-9 input vector `x`, and the layer's 32 filters form the rows of the weight
matrix `W`. Per pixel the host issues a LOAD of the patch (fills the coprocessor's input buffer)
followed by one whole-matrix GEMV over the 32 filters, and the coprocessor emits the 32 channel
results in a single streamed transfer.

| Kernel | accel (cyc) | CPU (cyc) | Speedup | Correctness |
|---|---:|---:|---:|---|
| StarDist conv1 (36 px × 32 ch, `im2col`→GEMV) | 63 166 | 124 734 | **1.97×** | bit-exact vs CPU **and** NumPy float32 golden (0 / 1 152 mismatches) |

The shared-FMA coprocessor runs the full layer in **63 166 cycles versus the CPU's 124 734 — a 1.97×
speedup — and is bit-exact**: every one of the 1 152 outputs matches both the CPU's own fused
hard-float loop and an independent NumPy float32 reference to the last bit (`err = 0`), confirming that
time-sharing the CPU's single FMA introduces no numerical deviation on a genuine pretrained-model
kernel.

The convolution's 1.97× trails the dense layer's 2.6× for a structural reason worth stating: in a
dense layer the stationary operand `x` is genuinely reused across all M outputs, so it is buffered
once; in a convolution every output pixel has a *different* receptive field, so the "stationary" patch
must be reloaded each pixel — 36 extra LOAD transfers whose per-transfer setup the single GEMV cannot
amortize. The margin therefore widens with the dot length `N` (deeper layers, where each reload is
amortized over far more multiply-accumulates) rather than with the pixel count — motivating the deeper
`N = 288` conv2 layer as the next measurement.

This result also carries a measurement-integrity lesson. An initial full-layer run reported a spurious
**3.87×** that proved to be an artifact: the DMA transaction descriptor had been left partially
initialized (hardware-FIFO mode, address increment, source/destination binding, and channel all
unset), so the coprocessor path silently emitted *incomplete* results — and did so quickly, inflating
the apparent speedup. Completing the descriptor restored bit-exactness and yielded the honest 1.97×.
The episode reinforces the discipline applied throughout this report: a speedup unaccompanied by a
bit-exact correctness check can be an illusion, and every performance figure here is gated on such a
reference.

### A complete published network end-to-end: LeNet-300-100 on MNIST

The strongest demonstration is a *whole, published* neural network running end-to-end on the shared
FMA. **LeNet-300-100** (LeCun et al., 1998) — the canonical MNIST multilayer perceptron,
`784 → 300 → 100 → 10` with ReLU — was trained to its published accuracy and its **FP32** weights
(≈ 266 K parameters, ≈ 1.06 MB) exported to the SoC. Inference is three dense layers, i.e. three GEMVs,
each mapped onto the shared-FMA coprocessor exactly as the dense-layer study prescribes: buffer the
activation vector once, stream the weight matrix, emit one result per row.

Running a real multi-layer network surfaced — and the design absorbed — three concerns that a single
kernel never exercised, each a small reusable contribution:

- **Runtime-programmable shape.** The three layers have different dimensions (`N,M` = 784×300,
  300×100, 100×10), so the coprocessor cannot be parameterized at elaboration. The accelerator was
  extended to **read `N` and `M` from a two-word header at the front of each LOAD transfer**, so a
  single bitstream runs layers of arbitrary shape — the operand-feeding DMA descriptor carries the
  geometry, no re-synthesis, no per-layer wiring.
- **Model residency.** X-HEEP's default 64 KB SRAM cannot hold a 1 MB model; the platform's
  configurable memory was enlarged to **2 MB** so the weights live in fast on-chip RAM and the DMA
  streams them from there (a research-SoC configuration choice, not a datapath change).
- **Transfer-size limit.** The DMA's `SIZE_D1` field is 16 bits (≤ 65 535 elements per 1-D transfer),
  which the 235 200-element first layer exceeds; the driver **chunks the weight matrix by whole rows**
  (≤ 83 rows per transfer for `N = 784`). Because every output is an independent dot product of the
  same buffered activation, chunking is exact.

| Network | Shape | Params (FP32) | accel (cyc) | CPU (cyc) | Speedup | Correctness |
|---|---|---:|---:|---:|---:|---|
| **LeNet-300-100** (MNIST) | 784→300→100→10 | 266 K (1.06 MB) | **556 318** | 2 664 446 | **4.79×** | prediction = label; **bit-exact** vs CPU **and** NumPy |

On-device the coprocessor classifies the test digit correctly and is **bit-exact to the last bit**
against both the CPU's own fused loop and an independent NumPy reference, at **4.79× the CPU's speed**
(556 318 vs 2 664 446 cycles) — a larger margin than the convolution because the long dot products
(`N = 784`) amortize the per-transfer overhead. *(The accelerator figure is the clean 556 318-cycle
forward pass, confirmed by two independent measurements; an earlier 733 418 reading was an artifact of
debug `printf`s left inside the cycle-counter window — a reminder that instrumentation must sit outside
the measured region.)* The network itself reaches its published ~98 % accuracy
on the full MNIST test set (measured off-line); crucially, **because the shared FMA is bit-exact, the
accelerator reproduces the model's output for every input**, so that full-test accuracy transfers
verbatim to the accelerated inference — a single on-device image is therefore sufficient proof of
correctness, not a sampled estimate. This is the thesis's central claim realized at full scale: a real,
published neural network runs end-to-end on the CPU's *own* floating-point FMA, bit-exact and 4.8×
faster, with **no second floating-point datapath added**.

### Co-execution: sharing the FMA between two concurrent floating-point workloads

Every result so far offloads work *from* an idle CPU — the CPU launches the coprocessor and polls until
it finishes. That leaves the arbiter, the thesis's central contribution, unexercised by the workload:
the CPU never asks for the FMA while the coprocessor is using it. The sharpest test is the opposite
regime — the CPU running its *own* floating-point job at the same time as the coprocessor runs the
model, so that both stream FMAs into the single unit concurrently and the CPU-priority arbiter must
mediate on live traffic.

This is also the coprocessor's natural deployment: an edge SoC rarely sits idle waiting on inference.
It runs a control loop, a sensor-conditioning filter, sensor fusion — a foreground floating-point task
— and would like inference to happen *alongside* it without a second FPU. (Note this task must be
genuinely floating-point: conventional cryptography — AES, SHA, RSA, ECC — is integer arithmetic and
would never touch the FPU, so a DSP kernel is used here.) Two experiments realize it, both driving the
model's host-orchestrated forward pass in interleaved async windows (launch a GEMV transfer, advance
the CPU's own job while it streams, synchronize) so the two truly overlap:

| Co-execution | Foreground (CPU) | Background (accel) | CPU slow-down | Both correct? | 2 jobs, concurrent vs sequential |
|---|---|---|---:|---|---|
| **Dual inference** | LeNet on digit A | LeNet on digit B | +6.8 % | bit-exact ✓ | 2 868 098 vs 3 220 764 → **1.12×** |
| **ML ∥ DSP** | 256-tap FIR filter | LeNet inference | +3.9 % | bit-exact ✓ | 2 208 719 vs 2 659 133 → **1.20×** |

Three findings hold across both. **(1) Correctness under contention:** every result — the CPU's job and
the coprocessor's — is *bit-identical* to the same computation run alone, so time-sharing the FMA on
concurrent live traffic corrupts nothing; the arbiter's mediation is transparent. **(2) The
CPU is not starved:** its floating-point job runs only 3.9–6.8 % slower with the coprocessor active
concurrently, and that residual is memory-*bus* arbitration (both masters fetch operands from the same
SRAM), not FMA starvation — the CPU-priority guarantee holds on live, workload-generated contention,
consistent with the isolated Phase-Y3 measurement. **(3) The inference rides in the CPU's idle FMA
cycles:** because a CPU floating-point loop leaves the FMA idle most cycles (operand loads, indexing,
loop overhead between issues), the coprocessor slots the model's dot products into those gaps —
delivering the bulk of the inference "for free" and completing two concurrent jobs in
**1.12–1.20×** the time of one, from a **single** FMA. The throughput gain is modest here only because
both workloads are themselves memory-active, so the shared operand bus — not the shared FMA — becomes
the limiter; the correctness and CPU-priority guarantees, which are the point, are unconditional. This
is the arbiter's *raison d'être* demonstrated on a real workload: a foreground floating-point task and
a neural network share one FMA, concurrently and bit-exactly, with **no second floating-point unit**.

### The accelerator's cycle budget: where every cycle goes

The speed-up figures say *how much* faster the coprocessor is; they do not say *where* its time goes,
and a fair engineering account must leave nothing out. To decompose one inference into disjoint,
exhaustive phases, the host code brackets each phase of the accelerated forward pass with the cycle
counter (`mcycle`): the DMA descriptor programming (**setup**), the transfer that streams the input
vector into the coprocessor's buffer (**load**), the transfer that streams the weight matrix and drains
the dot products (**gemv**), and the CPU-side activation between layers (**act** — bias add and ReLU).
By construction these four sum to the measured accelerator forward pass, so no cycle is left
unattributed. The weight matrix is now sent as a **single two-dimensional DMA transfer** per layer
(`size_d1 = N` inner, `size_d2 = M` outer, unit stride) rather than the earlier row-chunked
one-dimensional transfers; this fills the input buffer once per layer instead of once per chunk, and —
because both 2-D size fields are 16-bit — a whole 235 200-element layer travels in one descriptor,
re-verified bit-exact against both the chunked path and the CPU reference.

For LeNet-300-100 on one MNIST digit, with a combinational FMA (`FPU_ADDMUL_LAT = 0`):

| Phase | Cycles | Share | What it is |
|---|---:|---:|---|
| setup (DMA program) | 2 688 | 0.5 % | descriptor validation + register programming, both transfers, all layers |
| load (x stream) | 1 833 | 0.3 % | input vector streamed into the coprocessor's buffer, once per layer |
| **gemv (stream + FMA + writeback)** | **533 267** | **96.6 %** | weights streamed, dot products computed and drained |
| act (bias + ReLU) | 14 440 | 2.6 % | CPU-side activation between layers |
| **total (accelerator forward)** | **552 228** | 100 % | CPU fused baseline 2 654 927 → **4.80×**, no 2nd FPU |

Two things stand out. First, the coprocessor is overwhelmingly **compute-bound**: 96.6 % of the time is
the weight-streaming dot-product phase, while the software overhead of driving the DMA (setup) and
loading the input (load) together cost under 1 % — the shared-FMA datapath, not the programming model,
sets the pace, which is exactly what an offload engine should do. Second, the `gemv` phase costs
**533 267 / 266 200 = 2.00 cycles per fused multiply-add** across the network's 266 200 MACs. That
number is the accelerator's steady-state throughput laid bare: at zero FMA latency the control FSM
spends one cycle accepting each weight and one cycle issuing the FMA and latching its result, with no
wait state — the floor for a single-issue, single-FMA design. The `gemv` figure necessarily fuses
operand fetch, FMA execution and result writeback, which overlap in the streaming pipeline and are not
separable in software; the components that *do* separate cleanly — the per-transfer programming
overhead here, and the memory-contention and FMA-latency adders isolated in the concurrency and latency
studies — are each reported on their own line. This table is the `FPU_ADDMUL_LAT = 0` row; sweeping the
FMA latency adds a wait-state term to `gemv` (the 2.00 cyc/MAC floor rises), and running the same
forward pass under CPU contention adds the memory-bus term measured in the co-execution study, so the
budget stays closed from end to end at every operating point.

That contention term is now measured on the very same `L = 0` forward pass. Running LeNet on the
coprocessor while the CPU runs its FIR filter — both streaming into the one FMA — stretches the `gemv`
phase from 533 287 to 609 462 cycles, a **+14.3 % (76 175-cycle) memory-and-drain term**; because a
combinational FMA has essentially no drain at `FPU_ADDMUL_LAT = 0` (the isolated drain study measures
≈ 2 cycles there), this term is almost entirely operand-**bus** contention between the DMA and the CPU
on the shared SRAM. The CPU's own FIR slows by a comparable **+3.9 %** (81 207 cycles) — again bus, not
FMA starvation, so the CPU-priority guarantee holds on live traffic — and the two jobs finish in
**1.20×** the time of running them sequentially (2 208 719 vs 2 659 133 cycles), the inference riding in
the FIR's idle FMA cycles. Both results remain bit-exact under the contention. The memory-and-drain
term will grow with FMA latency, where the drain component stops being negligible — the subject of the
latency sweep.

### Scaling the sharing: two coprocessors on one FMA, and a parametric arbiter

The cycle budget above exposes an opening: with a combinational FMA the single coprocessor spends
2.00 cycles per multiply–add — one to accept a weight, one to issue the fused op — so the shared FMA's
issue slot sits **idle every other cycle**. If one coprocessor leaves the unit half-idle, a *second*
one can fill those slots, lifting throughput **still without a second FMA**.

A second identical GEMV coprocessor (acc1) is attached to a second DMA channel — channel 2, which the
X-HEEP DMA routes through a *different* bus master port, so the two weight streams fetch operands in
parallel rather than serialising. The CPU-priority arbiter generalises from two requestors to three
(CPU, acc0, acc1): the owner tag widens from one bit to two (0 = CPU, 1 = acc0, 2 = acc1) and, exactly
as before, rides through FPnew and returns with the result, so responses route to the right owner even
when the shared unit's lanes retire out of order. A dense layer is divided by **output rows** (an
M-split): acc0 computes the first ⌈M/2⌉ rows, acc1 the rest, each an independent dot product of the
same input vector. Because the split never crosses a dot product, every row's accumulation order is
identical to the single-coprocessor and CPU orders — bit-exactness is structural, not incidental; the
only thing the two share is the FMA, which the arbiter still serialises to one issue per cycle.

The grant decision itself is made **swappable**. It is an elaboration-time policy parameter, so only
the selected policy's logic synthesises (keeping the per-policy area honest), and three are provided:
**(0)** CPU strict priority with the two coprocessors round-robining the slots the CPU leaves — the
default, which preserves the CPU-priority guarantee; **(1)** a full round-robin over all three peers,
which *removes* CPU priority so its cost can be measured directly; and **(2)** a weighted (QoS)
round-robin whose per-requestor weights are programmable. The shared datapath — operand mux, tag
generation, tag-routed response — is identical for every policy; adding one is a ~15-line block.

A compact benchmark exercises the whole design end to end on a tiny three-layer MLP (128→64→32→16,
10 752 MACs) so it runs in seconds while measuring every quantity of interest; all results are
bit-exact against the CPU (policy 0, combinational FMA):

| Scenario | GEMV cycles | cyc/MAC | Result |
|---|---:|---:|---|
| One coprocessor (acc0) | 22 042 | 2.05 | **3.22×** vs CPU (27 341 vs 88 174), bit-exact |
| acc0 ∥ CPU FIR | 24 657 | — | +11 % accelerator slow-down (bus); **CPU FIR +0 %** |
| **Two coprocessors (acc0 + acc1)** | **14 837** | **1.37** | **GEMV 1.48×**, bit-exact, channel imbalance 188 cyc (≈1 %) |
| Two coprocessors ∥ CPU FIR | 23 913 | — | +61 % accelerator slow-down; **CPU FIR +11 %** (bus) |

Three points. **The second coprocessor pays off:** it lifts GEMV throughput 1.48× (2.05 → 1.37
cyc/MAC), pushing FMA-issue utilisation from ~50 % toward ~75 %, and the two channels finish within
~1 % of each other, so the row-split is well balanced. The shortfall from an ideal 2× is operand-bus
and arbitration overhead between the two streams — not the FMA — and no floating-point datapath was
added to obtain it. **Bit-exactness survives every configuration**, including two coprocessors and the
CPU all issuing into the one unit concurrently, confirming the two-bit tag routing is correct under
genuine three-way, out-of-order traffic. **CPU priority still holds where it must:** a concurrent CPU
FIR filter slows 0 % against one coprocessor and 11 % against two — and that 11 % is operand-bus
contention from two DMA streams, never FMA starvation — while the coprocessors absorb the whole cost of
sharing (+11 % and +61 %). That asymmetry is the design intent made measurable: the CPU's own
floating-point work is protected by construction, and the coprocessors are best-effort.

Comparing the three policies (0/1/2), sweeping the FMA latency L = 0..5, and moving from this toy MLP to
the real **LeNet-300-100** network were the subsequent measurements for this **two-coprocessor**
design; their headline findings are summarized next, but note the whole dual design is **retired by the
pipelined single coprocessor of §11.10** (whose final sweeps are in `PERF_BENCH_PIPE_SWEEP.md` and
`PERF_LENET_PIPE_SWEEP.md`). The dual findings, with
the coprocessor run **batched** (B = 8) so the shared FMA — not the operand bus — is the bottleneck: **(i)**
batching flips the memory-bound single-image GEMV into a compute-bound GEMM (LeNet, L = 0: 1.12 cyc/MAC,
≈ 88 % FMA-issue utilisation, **8.34× vs the CPU**, all bit-exact, and 8/8 MNIST images classified
correctly); **(ii)** the **second coprocessor is a latency-hiding device** whose value grows with FMA
latency — only ~1.12× at L = 0 (one accelerator already fills the unit) but **~1.99× at L ≥ 2**, keeping the
dual **3.1×–9.1× vs the CPU** across the whole latency range; and **(iii)** the **arbiter policy is decisive
precisely when the FMA is saturated** (compute-bound): CPU-strict holds the CPU's own job within ≤ 9 % even
under two accelerators, full round-robin costs it more, and the QoS weights dial the CPU's slow-down from
+13 % to +103 % and the two accelerators' balance (channel imbalance 223 → ~762 k cycles). Every trend
measured on the toy MLP reproduces on the real network, with a higher headline speedup.

### 11.10 The final design — a pipelined coprocessor hides the FMA latency with a single unit

The dual-coprocessor result of the previous subsection is best read as a diagnosis rather than a
destination. The second coprocessor helped *only because the first one was serial*: the accelerator
issued one multiply-accumulate to the shared FMA and then **waited** the full `1+L` cycles for the
result before issuing the next, so a single unit's throughput was `cyc/MAC = 1.14 + L` and its FMA-issue
utilisation fell as `1/(1+L)`. All the second unit did was pour its own MACs into the idle the first
one left. But that idle is an artefact of the serial issue, not of the sharing — and a batched workload
already carries, for every streamed weight, **B independent** multiply-accumulates (one per batch lane,
each into a different accumulator). Issuing those back-to-back keeps a pipelined FMA full from a single
requestor. The final coprocessor (`dma_fp_dot_accel_pipe`) does exactly this: it **decouples issue from
collection** — an issue pointer offers one MAC per granted cycle across the B lanes while a collection
pointer writes each returning result to its lane — and lets up to `L+1` of its own MACs be in flight at
once. No arbiter change was required: the arbiter already round-trips a two-bit owner tag with each
result, so the accelerator's in-flight MACs, all of the same op and latency, retire in issue order and a
simple round-robin collection counter suffices. A one-line hazard interlock (`inflight < B`) stalls
issue only in the regime `B < L+1`, where the batch is too small to cover the latency; for `B ≥ L+1` it
never fires. Bit-exactness is untouched by construction: the interleaving is *across* lanes, so each
dot product still accumulates its terms in order `k = 0..N−1`, identical to the serial unit and to the
CPU — and at runtime `B = 1` the interlock degenerates the engine back to serial issue, so the two
designs share one RTL and one correctness argument.

The effect is that the FMA-latency term vanishes from the throughput. Swept over `L = 0..5` (21 configs,
all bit-exact; LeNet also 8/8 MNIST correct), the pipelined single coprocessor holds `cyc/MAC` **flat at
1.14 on the toy MLP and 1.12 on LeNet-300-100** — the compute cycles drift by 0.5 % and 0.07 %
respectively across the entire latency range — where the serial unit grew to `1.14 + L` (up to 6.14).
Utilisation of the shared FMA stays at ~87 % (toy) / ~88 % (LeNet) at every latency, i.e. within ~13 % of
the unit's one-MAC-per-cycle ceiling, from a single accelerator. Against the serial single this is up to
**4.2× (toy) / 5.2× (LeNet) faster at L = 5** with the *same* hardware; against the CPU it holds
**5.3× (toy) / 8.3× (LeNet) at every latency**, where the serial single had decayed toward parity
(1.25× / 1.60× at L = 5). Most tellingly, the single pipelined unit **matches or beats the two-coprocessor
dual for every `L ≥ 1`** — its compute is 2.7× faster than the two serial units at L = 5 — at half the
coprocessors and half the operand-bus traffic; only at L = 0, where there is no latency to hide, do two
serial channels edge it by ~10 % on pure compute. Contention with a concurrent CPU FP filter stays
~3 % (toy) / ~0 % (LeNet), the latter because LeNet's 2.1 M-MAC batch dwarfs the CPU's small job.

This retires the dual as the headline design and sharpens the thesis. The contribution was never a
second datapath of any kind — it is that one FMA, already present and mostly idle, can be **shared**;
the pipelined single coprocessor is the clean realisation of that idea, hiding the sharing's only real
cost (the FMA pipeline latency) without adding a second coprocessor, a second FPU, or any change to the
CPU-priority arbiter. The two designs are kept side by side (`dma_fp_dot_accel_is` serial baseline,
`dma_fp_dot_accel_pipe` pipelined) precisely so the latency term can be shown appearing and then
vanishing; the full pipelined sweeps are in `PERF_BENCH_PIPE_SWEEP.md` and `PERF_LENET_PIPE_SWEEP.md`,
and the area and Fmax of the design at its 260 MHz operating point (core with arbiter, coprocessor
separate) are in `DC_STUDY_OPERATING.md`, produced by the `dc_scripts/` flow.

### 11.11 Pure-FMA contention: when the arbiter policies actually separate

Every co-execution measurement above uses the CPU's own FIR filter as the foreground workload. The FIR
is memory-bound — each tap loads an operand — so it issues a floating-point MAC only sparsely, and it
almost never wants the shared FMA in the *same cycle* as the coprocessor. A consequence, visible
throughout the sweeps, is that the CPU-strict (P0) and round-robin (P1) policies produce **nearly
identical** results (e.g. the shared CPU-FIR at L = 0 is 34 590 cycles under P0 versus 34 585 under P1 on
the toy MLP — a 0.01 % difference). This is not a defect of the arbiter; it is that the workload never
creates the contention the arbiter exists to resolve. To exercise the arbiter under genuine per-cycle
FMA contention, a second foreground kernel was added and run concurrently in its place: a **register-only
multiply-accumulate stress kernel with eight independent accumulators**, which — like the coprocessor's
own pipelining — issues an FMA to the shared unit on nearly every cycle. (The two foreground kernels are
run in separate co-execution passes, never together.) Two results follow, and both sharpen the thesis.

**The host core sustains only about two outstanding floating-point operations.** Run alone, the stress
kernel's cost per fused multiply-add is **flat at ≈ 1.51 cycles for L = 0 and L = 1** (issue-bound — the
eight accumulators hide the pipeline latency), then rises to **≈ L + 1 for L ≥ 2** (latency-bound:
3.27, 4.14, 5.13, 6.01 at L = 2..5). Because eight independent accumulators *should* hide up to seven
cycles of latency, the fact that they stop helping past L = 1 pins a **micro-architectural limit of
CV32E40P: its APU dispatcher keeps at most ≈ 2 floating-point operations in flight**. The practical
meaning is that the CPU can *saturate* the shared FMA only at L ≤ 1; at deeper latencies it throttles
itself and cannot flood the unit however much instruction-level parallelism the code exposes.

**Under genuine contention the CPU-priority guarantee is demonstrated with teeth.** At L ≤ 1, where the
CPU does saturate the FMA, the policies diverge sharply. Strict CPU priority (P0) slows the CPU's own
stress kernel by only **+6.4 % (toy MLP) / +11.3 % (LeNet)** — the CPU wins essentially every contested
cycle and the coprocessor absorbs the contention — whereas round-robin (P1) slows it by **+63.6 % /
+64.2 %**, a **53–57 percentage-point gap** where the FIR had shown none. The asymmetry is the design
intent made measurable: P0 protects the CPU's own floating-point work by construction, at the cost of
the coprocessor (which slows more under P0 than under P1: +31 % vs +27 % on LeNet). The QoS policy dials
the full spectrum between these extremes — CPU-weighted `4:1:1` holds the CPU to +22–26 %, while
accelerator-weighted `1:4:1` protects the coprocessor (+13–16 %) and lets the CPU pay +137–141 %. For
L ≥ 2 the policies converge again (P0 ≡ P1 to within measurement noise), precisely because the
latency-bound CPU no longer issues floating-point densely enough to contend. The takeaway is a clean,
honest one: **the arbiter policy is decisive exactly when the host can saturate the shared FMA, and the
near-identical FIR result reflects the workload, not the arbiter.** Full per-latency and per-policy
tables (metrics [6]–[8]) are in `PERF_BENCH_PIPE_SWEEP.md` and `PERF_LENET_PIPE_SWEEP.md`, §2b.

---

## Appendix A — Reproducibility (current, X-HEEP)

**Platform:** X-HEEP, cloned at `x-heep/`. **Accelerator sources (Part I standalone):** `xheep_dma/`.

**MCU configuration:** `configs/cv32e40px_fpu_dma.hjson` — `cpu_type: cv32e40px`,
`cpu_features: { fpu: true }`, DMA with `hw_fifo_mode_en: "yes"` (4 channels).

**Toolchain:** RISC-V `riscv32-corev-elf-` (hard-float `rv32imfc_zicsr` for FP apps); Verilator
**5.040** built from source; X-HEEP Python venv on Python 3.10 with `yamlfmt` removed and
`setuptools < 82`; `RISCV_XHEEP` pointed at the corev toolchain; Verible installed for the mcu-gen
format step.

**Build & run (Verilator):**
`make mcu-gen X_HEEP_CFG=configs/cv32e40px_fpu_dma.hjson` → `make verilator-build` →
`make app PROJECT=<name> [ARCH=rv32imfc_zicsr]` → `make verilator-run`. UART output (including
printed results) is captured in the simulation build directory's `uart0.log`.

**Baseline (Y0):** `example_matfloat`, dot-product loop bracketed with `mcycle` → **303 cycles**
(32-element FP32), single `fmadd.s` per element.

**Accelerator tap (Y1):** `tb/dma_sum_accel.sv` on DMA channel 1 (wired via `tb/testharness.sv.tpl`
and `tb/x-heep-tb-utils.core`); C driver `sw/applications/y1_sum_test` (`hw_fifo_en = 1`,
`channel = 1`, polling) → **`dst[0] = 136 = golden`, PASS**.

**Final design (pipelined single coprocessor).**
- **RTL:** `tb/dma_fp_dot_accel_pipe.sv` (the pipelined coprocessor) + `tb/xbuf_ram.sv` (input buffer);
  `tb/dma_fp_dot_accel_is.sv` is the serial baseline kept for the latency comparison. The
  `dma_apu_arbiter` lives in `cv32e40px_top.sv` behind `` `ifdef COPROC_FPU_SHARE ``. The
  `tb/testharness.sv.tpl` `COPROC_PIPE`/`COPROC_SERIAL` knob selects which coprocessor is instantiated.
- **Benchmarks:** `sw/applications/perf_bench_pipe/` (toy MLP 128-64-32-16, B = 8) and
  `sw/applications/perf_lenet_pipe/` (real LeNet-300-100 / MNIST, weights from `example_model/`), each
  printing the five setup/load/compute/total metrics and a bit-exactness / accuracy check.
- **Performance sweep:** `./sweep.sh` runs the 21 configs (L = 0..5 × policy P0/P1/P2, + QoS weight
  variants at L0) into `sweep_results/bench_pipe/` and `sweep_results/lenet_pipe/`; every config reports
  `bit-exact = 1` (LeNet also `8/8`). Tables: `PERF_BENCH_PIPE_SWEEP.md`, `PERF_LENET_PIPE_SWEEP.md`.
- **Area & Fmax (DC-NXT):** `dc_scripts/converge.sh` finds the operating clock (`CLK += |WNS|/2` →
  3.845 ns = **260 MHz**), then `dc_scripts/study.sh` synthesizes all configs at that clock,
  hierarchy-preserved, extracting FMA / FPU / arbiter / core from one `cv32e40px_top` run and the
  coprocessor separately. TSMC 40 nm G, `sc12mc_cln40g_base_rvt`, SS/0.81 V/125 °C. Report:
  `DC_STUDY_OPERATING.md`.
- **Real time:** any cycle count × **3.845 ns** = wall-clock at the core's 260 MHz operating point.

## Appendix B — Hardware-FIFO accelerator interface (X-HEEP)

- **`fifo_req_t` (DMA → accelerator):** `pop`, `push`, `flush`, `data[31:0]`.
- **`fifo_resp_t` (accelerator → DMA):** `empty`, `full`, `alm_full`, `data[31:0]`.
- **`hw_fifo_done_i`:** accelerator end-of-output; with an empty write buffer, terminates the
  transfer. Mandatory in HW-FIFO mode.
- **Datapath:** in HW-FIFO mode the accelerator *replaces* the DMA's internal write FIFO — the DMA
  pushes read-stream words in and pops processed words out to memory.
- **APU FMA encoding (for Y2):** `apu_op = 0`, `apu_flags = 0` (FP32, RNE), operands `{acc, b, a}` →
  `a·b + acc`.

---

## Appendix C — Files created and edited (engineering change-log)

This appendix records every file we authored or modified, so the concrete artifacts can be tracked
and lifted directly into the dissertation. Legend: **N** = newly authored by us, **G** = generated by
a tool from one of our inputs, **V** = vendored upstream and adapted, **E** = existing platform file
we edited.

### Part I — standalone feasibility (directory `xheep_dma/`)

| File | Kind | What & why |
|------|:---:|------------|
| `data/dma.hjson` | N | Register-map description; input to the OpenTitan `regtool`. Defines the DMA's memory-mapped descriptor registers (`SRC_PTR`, `DST_PTR`, `SIZE_D1`, `STATUS`, `HW_FIFO_EN`, …). |
| `data/dma_conf.svh` | N | Compile-time DMA configuration; enables **only** `HW_FIFO_MODE_EN` to minimize elaborated logic to exactly what the thesis needs. |
| `rtl/dma_reg_pkg.sv` | G | Register structs and address offsets, generated from `dma.hjson` by `regtool`. |
| `rtl/dma_reg_top.sv` | G | The register-file RTL (built on `prim_subreg`), generated from `dma.hjson`. |
| `rtl/fifo_pkg.sv` | N | `fifo_req_t`/`fifo_resp_t` type definitions for the DMA's FIFO/accelerator ports. **Note:** the field order here differs from X-HEEP's package (§6.2), which is why the on-SoC accelerator must import X-HEEP's version. |
| `rtl/dma_user_pkg.sv` | N | Flattened OBI request/response and register-bus request/response structs supplied as the DMA's type parameters (written explicitly to drop a fragile include-path dependency). |
| `rtl/dma.sv`, `rtl/dma_pkg.sv` | V | The `xheep_dma` engine itself (vendored from xheep-common), instantiated unmodified. |
| `rtl/xheep_dma_wrap.sv` | N | Thin top wrapper tying off unused engine inputs so the bare DMA elaborates standalone. |
| `rtl/dma_fp_dot_accel.sv` | N | **Phase X2 accelerator.** HW-FIFO block: pairs `(a,b)`, issues an APU-style FMA, accumulates, emits the scalar dot product, asserts `done`. This is the module promoted for Phase Y2. |
| `tb/tb_dma_copy.sv` | N | **Phase X1 testbench:** behavioral OBI memories + register driver; verifies plain copy and the HW-FIFO pass-through tap. |
| `tb/tb_dma_dot.sv` | N | **Phase X2 testbench:** drives `dma_fp_dot_accel` with the **real** `cv32e40p_fp_wrapper`; checks the dot product = 18.0 at FMA latency 0 and 5. |

### Part II — X-HEEP integration (directory `x-heep/`)

| File | Kind | What & why |
|------|:---:|------------|
| `configs/cv32e40px_fpu_dma.hjson` | N | **Phase Y0 MCU config.** Authored because no stock config has both an FPU and the DMA HW-FIFO: selects `cv32e40px` with `cpu_features: { fpu: true }` and the DMA in `hw_fifo_mode_en`. Drives `make mcu-gen`. |
| `util/python-requirements.txt` | E | Commented out `yamlfmt==1.1.0` (it pins an ancient `ruamel.yaml` that will not compile on modern Python) and pinned `setuptools < 82` (82 removed `pkg_resources`, needed by the register generator). Environment fix, not a design change. |
| `sw/applications/example_matfloat/main.c` | E | **Phase Y0 baseline instrumentation.** Set `PRINTF_IN_SIM = 1` and bracketed the `dotp()` call with `mcycle` read/reset + a print, to isolate and report the CPU dot-product baseline (303 cycles). |
| `tb/tb_v5.vlt` | E→reverted | Temporarily commented the `PROCASSINIT` Verilator waiver while on Verilator 5.020 (which predates that lint rule); reverted after upgrading to Verilator 5.040. |
| `tb/dma_sum_accel.sv` | N | **Phase Y1 accelerator.** FP-free integer sum-reduction over the HW-FIFO (states ACCUM→RESULT→DONE); imports X-HEEP's `fifo_pkg`. Proves the on-SoC tap; its arithmetic is replaced by the shared FMA in Y2. |
| `tb/testharness.sv.tpl` | E | **Phase Y1 wiring** (the *template*, since `testharness.sv` is generated): declared `accel_done`, put it into `hw_fifo_done_i` bit 1, and instantiated `dma_sum_accel` on DMA **channel 1**. |
| `tb/x-heep-tb-utils.core` | E | Added `dma_sum_accel.sv` to the `tb-harness` file set so it is compiled into the simulation. |
| `sw/applications/y1_sum_test/main.c` | N | **Phase Y1 C driver.** Programs a HW-FIFO DMA transfer on channel 1 (`hw_fifo_en = 1`, `channel = 1`), polls for completion, checks `dst[0]` against a software golden sum → `PASS` (136). |

### Part III — Y2: the floating-point sharing arbiter

| File | Kind | What & why |
|------|:---:|------------|
| `hw/vendor/xheep/cv32e40px/rtl/dma_apu_arbiter.sv` | N | **The contribution.** CPU-priority shared-FPU APU arbiter: muxes the CPU's and the coprocessor's APU/FMA requests onto the one FPnew FMA; `inflight_q` counts in-flight ops, `owner_q` keeps them un-mixed and only switches when the FMA pipeline is empty, responses route to the owner, and `fma_active_o` keeps the FMA clocked during coprocessor-only activity/drain. |
| `hw/vendor/xheep/cv32e40px/rtl/cv32e40px_top.sv` | V→forked | Separated the wrapper-side APU nets (`w_apu_*`) from the core-side nets and inserted `arb_i` between them under `` `ifdef COPROC_FPU_SHARE ``; the `` `else `` branch reproduces the stock wiring bit-for-bit. Clock gate extended to `apu_req | apu_busy | fma_active`. |
| `hw/vendor/xheep_cv32e40px.core` | E | Added `dma_apu_arbiter.sv` to the cv32e40px file list so it is compiled. |
| `core-v-mini-mcu.core` | E | Added `+define+COPROC_FPU_SHARE` to the Verilator options — the single master toggle for baseline (comment out) vs augmented (coprocessor) builds, for the area/cycle comparison. |

**Y2.1 verified (2026-07):** with the arbiter inserted and its coprocessor side tied off, `example_matfloat` still reports **303 dot-product cycles** and passes — i.e. the arbiter is transparent to the CPU's floating-point path, confirming the insertion in isolation from the port-threading.

**Y2.2 threading verified (2026-07):** the coprocessor's 8-signal APU port (`apu_ext_req_i`,
`apu_ext_operands_i[3][32]`, `apu_ext_op_i[6]`, `apu_ext_flags_i[15]` → and `apu_ext_gnt_o`,
`apu_ext_rvalid_o`, `apu_ext_rdata_o[32]`, `apu_ext_rflags_o[5]` ←) was threaded, guarded by
`` `ifdef COPROC_FPU_SHARE ``, from the simulation top down to the arbiter through five levels —
`hw/core-v-mini-mcu/cv32e40px_xif_wrapper.sv`, `hw/core-v-mini-mcu/cpu_subsystem.sv.tpl`,
`hw/core-v-mini-mcu/core_v_mini_mcu.sv.tpl`, `hw/system/x_heep_system.sv.tpl`, and
`tb/testharness.sv.tpl`. With the coprocessor end tied off, `example_matfloat` still reports 303
cycles and passes, confirming the port is plumbed end-to-end without disturbing the CPU's
floating-point path.

**Y2.3 — the contribution demonstrated (2026-07).** The standalone X2 accelerator
(`dma_fp_dot_accel.sv`, brought into `tb/`) was wired on DMA channel 1 with its HW-FIFO to the DMA
and its APU/FMA port to the threaded coprocessor APU path, so that it issues fused multiply-adds on
the **CPU's own FPnew FMA** through the arbiter. A firmware test (`sw/applications/y2_fpdotp_test/`)
lays out 32 interleaved `(a,b)` operand pairs, launches a single HW-FIFO DMA on channel 1, and reads
back the scalar result. **Result: `accel dotp = 11440 = golden`, PASS — a 32-element single-precision
dot product computed correctly on the shared FMA — in 208 cycles versus the 303-cycle CPU baseline
(a ~31% / 1.46× reduction), with no added floating-point datapath.** This is the central claim of the
thesis, now demonstrated end-to-end on the real SoC in simulation. Notably the speed-up appears even
though (i) the 208-cycle figure includes the DMA orchestration overhead and (ii) this is the
single-accumulator, latency-bound accelerator; the pipelined (multi-partial-accumulator) variant is
expected to widen the gap.

*A microarchitectural subtlety surfaced and was fixed:* at `FPU_ADDMUL_LAT = 0` the FMA returns its
result in the **same cycle** the operation is issued, before the arbiter's registered `owner_q` has
updated — so routing the response by `owner_q` misdirected the same-cycle result and hung the
accelerator. The arbiter now routes the response by a combinational owner
(`(pipe_empty & fpu_req_o & fpu_gnt_i) ? issue_dma : owner_q`), correct at every FMA latency.

Change-log for Y2.3: **N** `tb/dma_fp_dot_accel.sv` (the X2 dot-product datapath, imports X-HEEP's
`fifo_pkg`, two lint fixes: removed the unused `DATA_W` parameter, waived the intentionally-unused
`fpu_rflags_i`); **E** `tb/x-heep-tb-utils.core` (added it to the file set); **E** `tb/testharness.sv.tpl`
(declared the coprocessor APU wires, connected `x_heep_system`'s ext-APU to them, replaced
`dma_sum_accel_i` with `dma_fp_dot_accel_i` on channel 1); **N** `sw/applications/y2_fpdotp_test/main.c`;
**E** `hw/vendor/xheep/cv32e40px/rtl/dma_apu_arbiter.sv` (the combinational response-owner fix).

**Change-log for Y3 (measurement, 2026-07):** **N** `sw/applications/y3_sweep/main.c` (per-*N* CPU
and coprocessor dot product, self-checked — the data behind §11's length table); **N**
`sw/applications/y3_contention/main.c` (CPU hard-float MAC concurrent with the coprocessor; both
results checked; reports the ≈19-cycle, bus-dominated CPU slow-down); **N**
`sw/applications/y3_independent/main.c` (register-only and memory-bound integer kernels measured alone
vs concurrent with the active coprocessor — compute Δ = 0, memory Δ = 5; `volatile`-seeded to defeat
compiler CSE); **E** `tb/testharness.sv.tpl` (Y3 clean-revert: wrapped the `dma_fp_dot_accel_i`
instance in `` `ifdef COPROC_FPU_SHARE `` with an `` `else `` that ties off channel-1 HW-FIFO, so the
single define fully reverts the design to stock X-HEEP); **E** `util/xheep_gen/load_config.py`
(forwarded `fpu_addmul_lat`/`fpu_others_lat` from `cpu_features` to the `cv32e40px` constructor — the
upstream loader accepted these in the CPU class but never passed them, so the config field was
silently ignored; this enables the FMA-latency sweep); **E** `hw/vendor/waiver/lint/cv32e40px.vlt`
(added file-scoped `WIDTHEXPAND`/`WIDTHTRUNC` waivers for `cv32e40px_decoder.sv` — Verilator 5.040
split the old `WIDTH` rule and changed the message text, so the vendor's existing `-rule WIDTH`
waivers no longer matched; the benign width note in `apu_lat_o` surfaces only at `fpu_addmul_lat ≥ 2`); **N** `dc/`
(Design Compiler area-estimation flow — `synth_area.tcl` per-module + `run_area.sh` one-command driver
running each module in a fresh dc_shell for zero state carryover; `rtl_core.f`/`rtl_accel.f` file lists
frozen from the Verilator `.vc`; measures `cv32e40px_fp_wrapper` = the avoided FMA vs `dma_apu_arbiter`
+ `dma_fp_dot_accel` = the added logic).

**Y3 status (2026-07).** Done: the length sweep (§11, break-even N ≈ 16, asymptote ≈ 3.5×); the
concurrency study (independent non-FP CPU work sees ≈0 added latency; the residual is baseline
memory-bus arbitration, not FMA sharing); the **FMA-latency drain sweep** (§11: flat ≈14–21-cycle,
bus-dominated cost for L ≤ 3, a knee near L = 4, rising to ≈16% at L = 5, bit-correct and bounded at
every latency — the CPU-priority guarantee across the full pipeline-depth range); the audited
single-define clean revert to stock X-HEEP; and the **area study** (§11: Design Compiler NXT, TSMC 40 nm —
sharing avoids a 5 789 µm² FMA at the cost of a 453 µm² arbiter = 7.8 % of one FMA / 3.6 % of the whole
FPU; net saving 5 336 µm² per coprocessor, 11 125 µm² for the dual). *Remaining (optional):* the pipelined multi-partial-accumulator
accelerator, a throughput enhancement to push the per-element term toward one cycle at L ≥ 1 — the core
measurement and area programme is otherwise complete.

**Change-log for the ML kernels (real-workload demonstration, 2026-08):** **N**
`tb/dma_fp_dot_accel_is.sv` (input-stationary GEMV coprocessor: an `x`-buffer holds the reused input
vector, one whole-matrix transfer streams `W`, and the FSM emits one result per `N`-word row for `M`
rows — the `num_rows_i` extension and a `loaded_q` flag that lets the buffered input persist across
the DMA `flush` between the LOAD and GEMV transfers, then clears after the last row so the next pixel
re-loads); **E** `tb/testharness.sv.tpl` / `tb/x-heep-tb-utils.core` (instantiate the GEMV accelerator
with `num_pairs_i`/`num_rows_i`; add its source to the fileset); **N**
`sw/applications/ml_fc_gemv/main.c` (dense-layer GEMV: buffer `x` once, one transfer over `W` → M
outputs; the 2.6× dense result of §"Real ML kernel"); **N** `sw/applications/ml_conv1/main.c`
(single-pixel StarDist conv1 via `im2col`, bit-exact against the NumPy golden — the correctness
bring-up); **N** `sw/applications/ml_conv1_full/main.c` (full 36-pixel conv1 layer: per-pixel `im2col`
patch LOAD + whole-filter-bank GEMV, self-checked against both the CPU fused loop and the NumPy golden
→ 0/1 152 mismatches, honest 1.97×). *Integrity fix:* the full-layer driver initially reported a
spurious 3.87× because its DMA transaction descriptor was incompletely initialized (`hw_fifo_en`,
`inc_d1_du`, `src`/`dst`, `channel` all left zero), so the coprocessor path emitted incomplete results
quickly; adding the one-time descriptor setup (`dma_init` + a fully-populated `dma_trans_t` +
`dma_validate_transaction`, mirroring the bring-up app) restored bit-exactness and the honest 1.97×.
**N** `example_model/` (pretrained StarDist 2D_versatile_fluo conv1/encoder data extracted to C
headers — `conv_test_data.h`: FP32 input, weights, biases, and a pre-activation NumPy reference — plus
the `gen_*` scripts that produced them).

**Change-log for the full network + co-execution (real published model, 2026-08):** **E**
`tb/dma_fp_dot_accel_is.sv` (made the coprocessor **runtime-programmable**: it now reads the dot length
`N` and row count `M` from a two-word header at the front of each LOAD transfer — new `LHDR_N`/`LHDR_M`
states — instead of elaboration-time parameters, so one bitstream runs every layer shape; `MAXN` 128 →
1024; the `num_pairs_i`/`num_rows_i` ports were removed from the module and its `tb/testharness.sv.tpl`
instance); **E** `configs/cv32e40px_fpu_dma.hjson` (**enlarged SRAM to 2 MB** — `ram_banks sizes:[1024]`,
`linker_sections` code region 0x180000 to hold the ≈1 MB of weights in `.rodata` — so a real model is
memory-resident and the DMA streams it from fast RAM); **N** `example_model/gen_lenet_mnist.py` (trains
LeNet-300-100 on MNIST to its published ~98 %, exports FP32 weights transposed to out-major `[M][N]`,
pre-normalized test images, and reference predictions to `lenet_mnist_data.h`); **N**
`sw/applications/ml_rt_gemv/` (runtime-`N`/`M` bring-up, bit-exact); **N** `sw/applications/ml_lenet/`
(full LeNet forward pass: three GEMV layers with **row-chunking** to the DMA's 16-bit transfer limit —
≤ 65 535 elements/transfer, 4 chunks for the 235 200-element first layer — bit-exact, 4.79×); **N**
`sw/applications/ml_dual/` (dual-stream: CPU classifies one digit while the accelerator classifies
another, interleaved async windows, bit-exact both, CPU +6.8 %, 1.12× throughput); **N**
`sw/applications/ml_coexec/` (co-execution: a 256-tap CPU FIR filter concurrent with the accelerator's
LeNet inference — the headline "two independent FP jobs share one FMA" scenario — bit-exact both, CPU
+7.9 %, 1.15×). *Debug/measurement note:* the earlier LeNet reading of 733 418 accelerator cycles was
inflated by `printf`s inside the `mcycle` window; the clean figure is 556 318 (confirmed by two
independent runs), giving the corrected 4.79× single-inference speedup.

### Part IV — the final pipelined single coprocessor (2026-08)

| File | Kind | What & why |
|------|:---:|------------|
| `tb/dma_fp_dot_accel_pipe.sv` | N | **The final coprocessor.** Pipelined batched-GEMM: decoupled issue/collect pointers (`iss_b`/`col_b`), up to `L+1` MACs in flight, one-line hazard interlock `inflight < B`, in-order collect (no per-op tag). Reads header `[N,M,B]`; input-stationary; `B = 1` degenerates to serial. `MAXN = 1024`, `MAXB = 8`. (`inflight` widened to 16 bits to satisfy a Verilator WIDTHEXPAND check.) |
| `tb/dma_fp_dot_accel_is.sv` | E | Serial baseline retained for the latency comparison; the `GEMV_ONLY` parameter removed so it is a clean batched-GEMM (`NB = MAXB`, `BEFF = bn_q`). |
| `tb/xbuf_ram.sv` | N | 1R1W input buffer submodule (an SRAM macro in silicon; black-boxed in the DC area run). |
| `dma_apu_arbiter.sv` (in `cv32e40px_top.sv`) | E | Generalized to 3 requestors (CPU + acc0 + acc1) with a **2-bit owner tag**, **no drain** — grants every cycle, so pipelined multi-issue needs no arbiter change. Swappable elaboration-time policy `ARB_POLICY_SEL` = P0 CPU-strict / P1 round-robin / P2 QoS-weighted. |
| `tb/testharness.sv.tpl` | E | `COPROC_PIPE` / `COPROC_SERIAL` knob selecting the pipelined vs serial coprocessor for acc0; acc1 left idle (no cycles). |
| `tb/x-heep-tb-utils.core` | E | Added `dma_fp_dot_accel_pipe.sv` (and `xbuf_ram.sv`) to the fileset. |
| `sw/applications/perf_bench_pipe/main.c` | N | Toy MLP 128-64-32-16, B = 8; five metrics (CPU inference, CPU FIR, coproc inference, shared CPU FIR, shared coproc) each setup/load/compute/total, + a RAPOR block; bit-exact + cyc/MAC + speedup. |
| `sw/applications/perf_lenet_pipe/main.c` | N | Same five-metric structure on the real LeNet-300-100 / MNIST (weights via `example_model/`), 2-D weight transfer for layer-1, `argmax` accuracy → `8/8`. |
| `sweep.sh` | E | Drives the 21-config L × policy sweep (+ QoS weight variants) for the pipe apps into `sweep_results/bench_pipe/` and `sweep_results/lenet_pipe/`. |
| `dc_scripts/` | N | Reorganized DC-NXT flow: `converge.sh` (`CLK += |WNS|/2` → 260 MHz operating point), `study.sh` + `study.tcl` (hierarchy-preserved area/Fmax, per-component extraction, FMA-specific `report_timing -through`), `rtl_core.f` / `rtl_accel_pipe.f` filelists. Outputs to `dc_reports/`, work in `dc_work/`. |
| `PAPER_NOTES.md`, `PERF_BENCH_PIPE_SWEEP.md`, `PERF_LENET_PIPE_SWEEP.md`, `DC_STUDY_OPERATING.md` | N | The compact technical dossier and the three companion measurement reports (Appendix A). |

**Result (final):** `cyc/MAC` flat in L (≈ 1.14 MLP / 1.12 LeNet), **5.45× / 8.37×** vs CPU, all
bit-exact (LeNet 8/8); co-execution CPU FIR + coproc share one FMA at negligible cost; DC-NXT at
**260 MHz** gives arbiter = 8.4 % of one FMA, 3.9× less added area per accelerator than a dedicated FMA.
The pipelined single unit retires the dual as the headline design.

---

*This report reflects the completed final design. The contribution — a CPU-priority, owner-tagged,
no-drain APU arbiter that time-shares the CPU's one FMA, driven by a **pipelined single coprocessor**
that issues a batch's independent MACs back-to-back to hide the FMA latency — is implemented and
evaluated end-to-end on the real X-HEEP SoC. Across a full L × policy sweep it is **bit-exact** with
the CPU (LeNet 8/8 MNIST), runs a toy MLP at **5.45×** and LeNet-300-100 at **8.37×** with `cyc/MAC`
flat in L, co-executes with the CPU's own floating-point work on the one FMA at negligible cost, and —
synthesized in TSMC 40 nm at its **260 MHz** operating point — adds a **1.8 k µm² arbiter (8.4 % of one
FMA)** instead of a whole FMA, i.e. **3.9× less added area per accelerator** than a dedicated-FMA
design: the "no second FPU" claim, quantified. Earlier design points (interleaved-pair dot product,
serial GEMV, dual coprocessor) are retained above only as the evolution that led here. Optional
follow-ups (FMA-specific fmax vs L, full-SoC clock, energy/MAC) are listed in §9; none is blocking.*

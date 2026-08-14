# Update — Two coprocessors on one FMA + a parametric arbitration-policy framework

This update scales the shared-FMA design from **one** DMA coprocessor to **two**, both time-sharing the
CPU's single FMA, and turns the arbiter's grant decision into a **swappable policy** (three policies
provided). A compact benchmark measures the whole thing end to end. Still **no second FPU** — the two
coprocessors add zero floating-point datapath; they fill the FMA-issue slots the single one leaves idle.

---

## 1. Second coprocessor (acc1) on DMA channel 2

A second identical GEMV coprocessor (`dma_fp_dot_accel_is`) is instantiated on **DMA channel 2**, whose
HW-FIFO is a different bus master port than channel 1's, so the two weight streams fetch operands in
parallel. A dense layer is split by **output rows** (M-split): acc0 computes the first ⌈M/2⌉ rows, acc1
the rest, each an independent dot product of the same input vector — so per-row accumulation order, and
therefore **bit-exactness**, is untouched. The only shared resource is the FMA, serialised by the
arbiter to one issue per cycle.

## 2. Arbiter: 2 → 3 requestors, parametric policy

`dma_apu_arbiter.sv` now serves **CPU + acc0 + acc1**. The owner tag widens **1 → 2 bits**
(0 = CPU, 1 = acc0, 2 = acc1) and round-trips through FPnew with the result, so responses route to the
right owner even across out-of-order lanes. The grant decision is an **elaboration-time policy
parameter** (`ARB_POLICY`), so only the selected policy synthesises (honest per-policy area); the
datapath (operand mux, tag, tag-routed response) is shared. Three policies:

| ARB_POLICY | Policy | Purpose |
|---|---|---|
| 0 | CPU strict priority; the two accs round-robin the leftover slots | preserves the CPU-priority guarantee (default) |
| 1 | full round-robin over {CPU, acc0, acc1} | removes CPU priority → measures its cost |
| 2 | weighted (QoS) round-robin, programmable `W_CPU/W_ACC0/W_ACC1` | tunable grant share |

Adding a policy is a ~15-line `generate` block. Select at RTL build time via the `` `ARB_POLICY_SEL ``
define in `cv32e40px_top.sv`.

## 3. Benchmark and results

`sw/applications/perf_bench/main.c` runs a tiny 3-layer MLP (128→64→32→16, 10 752 MACs) — same workload
*types* as `ml_lenet`/`ml_coexec` but small enough to finish in seconds — and reports every metric.
**Policy 0, combinational FMA (`FPU_ADDMUL_LAT = 0`), all bit-exact vs the CPU:**

| Scenario | GEMV cyc | cyc/MAC | Result |
|---|---:|---:|---|
| [B] one coprocessor (acc0) | 22 042 | 2.05 | **3.22×** vs CPU (27 341 vs 88 174), bit-exact |
| [C] acc0 ∥ CPU-FIR | 24 657 | — | accelerator +11 % (bus); **CPU-FIR +0 %** (priority holds) |
| [D] **two coprocessors (acc0+acc1)** | **14 837** | **1.37** | **GEMV 1.48×**, bit-exact, imbalance 188 cyc (≈1 %) |
| [E] two coprocessors ∥ CPU-FIR | 23 913 | — | accelerator +61 %; **CPU-FIR +11 %** (bus, not FMA starvation) |

- **The second coprocessor pays off:** GEMV 2.05 → 1.37 cyc/MAC (1.48×), FMA-issue utilisation ~50 % →
  ~75 %, channels balanced to ~1 %. The gap from an ideal 2× is operand-bus/arbitration overhead
  between the two streams, not the FMA — and no FP datapath was added.
- **Bit-exact in every configuration**, including two coprocessors + the CPU all issuing concurrently —
  the 2-bit tag routing is correct under three-way, out-of-order traffic.
- **CPU priority holds where it must:** the CPU's FIR slows 0 % (one coprocessor) / 11 % (two, bus
  only, never FMA-starved); the coprocessors absorb the sharing cost (+11 % / +61 %). The CPU's
  real-time work is protected by construction; the coprocessors are best-effort.

**Next:** compare policies 0/1/2 on this same benchmark (CPU-priority vs peer-fairness vs QoS), and
sweep FMA latency.

---

## Files

| File | Change |
|---|---|
| `x-heep/hw/vendor/xheep/cv32e40px/rtl/dma_apu_arbiter.sv` | 3-requestor, 2-bit tag, parametric policy (generate-per-policy) |
| `x-heep/hw/vendor/xheep/cv32e40px/rtl/cv32e40px_fp_wrapper.sv` | owner tag widened 1 → 2 bit (`.TagType(logic [1:0])`) |
| `x-heep/hw/vendor/xheep/cv32e40px/rtl/cv32e40px_top.sv` | 2nd APU port group wired to arbiter `dma1_*`; tag nets 2-bit; `ARB_POLICY_SEL` select |
| `x-heep/hw/core-v-mini-mcu/{cpu_subsystem,core_v_mini_mcu}.sv.tpl`, `cv32e40px_xif_wrapper.sv`, `hw/system/x_heep_system.sv.tpl` | thread the 2nd APU port group (`apu_ext1_*`) through the SoC hierarchy |
| `x-heep/tb/testharness.sv.tpl` | instantiate acc1 on DMA channel 2 (HW-FIFO + done + APU) |
| `x-heep/sw/applications/perf_bench/main.c` | **new** — compact end-to-end benchmark (single/contention/dual/3-way) |
| `thesis_progress.md` | §11 new subsection "Scaling the sharing: two coprocessors on one FMA" |

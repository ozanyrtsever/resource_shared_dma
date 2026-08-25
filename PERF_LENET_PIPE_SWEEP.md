# perf_lenet_pipe — sweep sonuçları (gerçek LeNet-300-100 / MNIST, pipelined single coprocessor)

**Workload:** LeNet-300-100 `784-300-100-10`, FP32, **B=8** (266 200 MAC/img, 2 129 600 MAC/batch). Gerçek eğitilmiş ağırlıklar + gerçek MNIST görüntüleri.
**Donanım:** pipelined tek coprocessor (`dma_fp_dot_accel_pipe`), CPU'nun tek FMA'sını paylaşır (ikinci FPU yok), APU arbiter. Verilator, trace off.
**Sweep:** L = 0..5 × arbiter policy P = 0/1/2 (+ L=0'da QoS ağırlık varyantları). Klasör: `sweep_results/lenet_pipe/`. **21/21 config `ok`, hepsi `bit-exact=1` ve `correct=8/8, match_ref=8/8`.**

**5 metrik** (setup/load [DMA yoksa 0] / compute / total, cycle):
`[1]` CPU alone inference · `[2]` CPU alone FIR · `[3]` Coproc alone inference · `[4]` Shared CPU FIR · `[5]` Shared Coproc inference.
FIR = CPU'nun kendi 32-tap DSP işi (inference değil); "shared" = coproc inference ederken CPU aynı anda FIR koşar, tek FMA'yı paylaşır.

---

## 1. Ana tablo — L-sweep @ P0 (CPU-strict)

`[1]`,`[2]`,`[3]` policy'den bağımsız; sadece shared `[4]`/`[5]` policy'ye duyarlı (§2).

| L | [1] CPU inf | [2] CPU FIR | [3] Coproc inf: setup/load/compute/**total** | cyc/MAC | speedup | [5] shared coproc **total** (alone) | [4] shared CPU FIR **total** (alone) | acc |
|:-:|:-----------:|:-----------:|:--------------------------------------------:|:-------:|:-------:|:-----------------------------------:|:------------------------------------:|:---:|
| 0 | 21 219 250 | 33 936 | 1647 / 12 183 / 2 400 642 / **2 533 197** | 1.12 | 8.37× | 2 536 289 (2 533 197) | 35 779 (33 936) | 8/8 |
| 1 | 21 222 530 | 33 936 | 1647 / 12 183 / 2 400 997 / **2 536 832** | 1.12 | 8.36× | 2 540 148 (2 536 832) | 35 910 (33 936) | 8/8 |
| 2 | 21 225 810 | 34 064 | 1647 / 12 183 / 2 401 355 / **2 540 470** | 1.12 | 8.35× | 2 543 986 (2 540 470) | 36 062 (34 064) | 8/8 |
| 3 | 21 229 090 | 34 192 | 1647 / 12 183 / 2 401 720 / **2 544 115** | 1.12 | 8.34× | 2 547 540 (2 544 115) | 36 062 (34 192) | 8/8 |
| 4 | 21 232 450 | 34 320 | 1647 / 12 183 / 2 402 081 / **2 547 756** | 1.12 | 8.33× | 2 551 286 (2 547 756) | 36 292 (34 320) | 8/8 |
| 5 | 21 238 930 | 34 448 | 1647 / 12 183 / 2 402 427 / **2 551 382** | 1.12 | 8.32× | 2 554 750 (2 551 382) | 36 382 (34 448) | 8/8 |

**Okunuşu:**
- **`[3]` compute (coproc inference) L boyunca sabit:** 2 400 642 → 2 402 427 (**+0.07%**), **cyc/MAC = 1.12 düz** her L'de → pipelining FMA latency'sini gizliyor (serial olsa 1.12+L olurdu). setup=1647, load=12 183 sabit (load MLP'den büyük çünkü LeNet girişi 784×8 kelime).
- **speedup 8.37×→8.32×** (LeNet büyük olduğu için MLP'nin 5.4×'inden yüksek; hafif düşüş: CPU golden de FMA'yı L'de kullanıyor).
- **Shared:** coproc inference **+%0.12** (L0) — neredeyse hiç (LeNet GEMM 2.1M MAC, FIR 4096 MAC yanında ihmal); CPU FIR **+%5.4** (L0).
- **Accuracy 8/8 + match_ref 8/8** her config'te → gerçek ağ accelerator'da doğru çalışıyor.

---

## 1b. Gerçek zaman (cycle × 3.845 ns @ 260 MHz, P0)

DC sentezinde çekirdek (`cv32e40px_top`) **3.845 ns'de WNS=0** → gerçek fmax **260 MHz**. Verilator cycle × 3.845 ns = gerçek süre.

| L | CPU inf | CPU FIR | coproc inf | shared coproc | shared CPU FIR |
|:-:|--------:|--------:|-----------:|--------------:|---------------:|
| 0 | 81.588 ms | 130.48 µs | 9.740 ms | 9.752 ms | 137.57 µs |
| 1 | 81.601 ms | 130.48 µs | 9.754 ms | 9.767 ms | 138.07 µs |
| 2 | 81.613 ms | 130.98 µs | 9.768 ms | 9.782 ms | 138.66 µs |
| 3 | 81.626 ms | 131.47 µs | 9.782 ms | 9.795 ms | 138.66 µs |
| 4 | 81.639 ms | 131.96 µs | 9.796 ms | 9.810 ms | 139.54 µs |
| 5 | 81.664 ms | 132.45 µs | 9.810 ms | 9.823 ms | 139.89 µs |

**Resim başına (L0, B=8):** coproc inference **1.22 ms/img**, CPU inference **10.20 ms/img** → coproc **8.37× hızlı** (gerçek zamanda).
**Not:** periyot = **core** sentez fmax'i (260 MHz). Tam-SoC clock'u (DMA/bus/memory dahil) farklı olabilir → onun için full-SoC sentez gerekir. Herhangi bir cycle değerini `× 3.845 ns` ile zamana çevirebilirsin.

---

## 2. Policy karşılaştırması @ L=0 (shared senaryo)

`[1]`=21 219 250, `[2]`=33 936, `[3]` total=2 533 197 **her policy'de aynı** (policy sadece paylaşımı etkiler):

| Config (L0) | [5] shared coproc **total** (alone 2 533 197) | [4] shared CPU FIR **total** (alone 33 936) |
|---|:---:|:---:|
| **P0** CPU-strict | 2 536 289 (+0.12%) | 35 779 (**+5.4%**) |
| **P1** round-robin | 2 536 289 (+0.12%) | 35 779 (+5.4%) |
| **P2** QoS `w1-1-1` (eşit) | 2 536 138 (+0.12%) | 38 354 (+13.0%) |
| **P2** QoS `w4-1-1` (CPU-öncelik) | 2 536 717 (+0.14%) | 36 422 (+7.3%) |
| **P2** QoS `w1-4-1` (acc-öncelik) | **2 534 096 (+0.04%)** | 39 301 (**+15.8%**) |
| **P2** QoS `w1-4-4` (acc-öncelik) | 2 534 096 (+0.04%) | 39 301 (+15.8%) |

**Okunuşu:** policy CPU-vs-coproc önceliğini ayarlıyor (MLP ile aynı desen):
- **P0/P1:** CPU'yu korur → CPU FIR +5.4% (coproc +0.12%).
- **P2 acc-öncelik (`w1-4-*`):** coproc'u korur → coproc inference **+0.04%** (neredeyse sıfır), ama CPU FIR **+15.8%**.
- **P2 CPU-öncelik (`w4-1-1`):** ara denge (FIR +7.3%).

---

## 3. Özet gözlemler
- **Pipelining çalışıyor (gerçek model):** coproc inference cyc/MAC = **1.12 düz** (L=0..5), compute ~2.40M sabit.
- **CPU'ya karşı ~8.3× hızlı**, her L'de (MLP'nin 5.4×'inden yüksek — büyük model overhead'i daha iyi amortize ediyor).
- **Paylaşım neredeyse bedava:** coproc +%0.12 (LeNet GEMM devasa, FIR ihmal); CPU FIR +%5.4 (P0).
- **Policy = CPU↔coproc önceliği:** P0/P1 CPU'yu korur; P2 ağırlıklarıyla dengeyi (+5.4%…+15.8% CPU) ayarlarsın.
- **21/21 bit-exact + 8/8 accuracy** → gerçek LeNet accelerator'da doğru + CPU ile birebir.

---

## Appendix — tüm 21 config, 5 total cycle (+ acc)

| Config | [1] CPU inf | [2] CPU FIR | [3] coproc inf | [5] shared coproc | [4] shared CPU FIR | acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| L0_P0 | 21 219 250 | 33 936 | 2 533 197 | 2 536 289 | 35 779 | 8/8 |
| L0_P1 | 21 219 250 | 33 936 | 2 533 197 | 2 536 289 | 35 779 | 8/8 |
| L0_P2 w1-1-1 | 21 219 250 | 33 936 | 2 533 197 | 2 536 138 | 38 354 | 8/8 |
| L0_P2 w4-1-1 | 21 219 250 | 33 936 | 2 533 197 | 2 536 717 | 36 422 | 8/8 |
| L0_P2 w1-4-1 | 21 219 250 | 33 936 | 2 533 197 | 2 534 096 | 39 301 | 8/8 |
| L0_P2 w1-4-4 | 21 219 250 | 33 936 | 2 533 197 | 2 534 096 | 39 301 | 8/8 |
| L1_P0 | 21 222 530 | 33 936 | 2 536 832 | 2 540 148 | 35 910 | 8/8 |
| L1_P1 | 21 222 530 | 33 936 | 2 536 832 | 2 540 148 | 35 910 | 8/8 |
| L1_P2 w4-1-1 | 21 222 530 | 33 936 | 2 536 832 | 2 540 549 | 36 333 | 8/8 |
| L2_P0 | 21 225 810 | 34 064 | 2 540 470 | 2 543 986 | 36 062 | 8/8 |
| L2_P1 | 21 225 810 | 34 064 | 2 540 470 | 2 543 986 | 36 062 | 8/8 |
| L2_P2 w4-1-1 | 21 225 810 | 34 064 | 2 540 470 | 2 544 431 | 36 462 | 8/8 |
| L3_P0 | 21 229 090 | 34 192 | 2 544 115 | 2 547 540 | 36 062 | 8/8 |
| L3_P1 | 21 229 090 | 34 192 | 2 544 115 | 2 547 540 | 36 062 | 8/8 |
| L3_P2 w4-1-1 | 21 229 090 | 34 192 | 2 544 115 | 2 548 098 | 36 554 | 8/8 |
| L4_P0 | 21 232 450 | 34 320 | 2 547 756 | 2 551 286 | 36 292 | 8/8 |
| L4_P1 | 21 232 450 | 34 320 | 2 547 756 | 2 551 286 | 36 292 | 8/8 |
| L4_P2 w4-1-1 | 21 232 450 | 34 320 | 2 547 756 | 2 551 711 | 36 681 | 8/8 |
| L5_P0 | 21 238 930 | 34 448 | 2 551 382 | 2 554 750 | 36 382 | 8/8 |
| L5_P1 | 21 238 930 | 34 448 | 2 551 382 | 2 554 750 | 36 382 | 8/8 |
| L5_P2 w4-1-1 | 21 238 930 | 34 448 | 2 551 382 | 2 555 233 | 36 751 | 8/8 |

*Kaynak: `sweep_results/lenet_pipe/` (21 config, hepsi `ok` + `bit-exact=1` + `8/8`). Coproc inference: setup=1647, load=12 183 sabit; compute L ile ~%0.07 değişir.*

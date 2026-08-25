# perf_bench_pipe — sweep sonuçları (toy MLP, pipelined single coprocessor)

**Workload:** MLP `128-64-32-16`, FP32, **B=8** (10 752 MAC/img, 86 016 MAC/batch).
**Donanım:** pipelined tek coprocessor (`dma_fp_dot_accel_pipe`), CPU'nun tek FMA'sını paylaşır (ikinci FPU yok), APU arbiter. Verilator, trace off.
**Sweep:** L (fpu_addmul_lat) = 0..5 × arbiter policy P = 0/1/2 (+ L=0'da QoS ağırlık varyantları). Klasör: `sweep_results/bench_pipe/`. **21/21 config `ok`, hepsi `bit-exact=1`.**

**5 metrik** (her biri setup/load [DMA yoksa 0] / compute / total, cycle):
`[1]` CPU alone inference · `[2]` CPU alone FIR · `[3]` Coproc alone inference · `[4]` Shared CPU FIR · `[5]` Shared Coproc inference.
FIR = CPU'nun kendi 32-tap DSP işi (inference değil); "shared" = coproc inference ederken CPU aynı anda FIR koşar, tek FMA'yı paylaşır.

---

## 1. Ana tablo — L-sweep @ P0 (CPU-strict)

`[1]`, `[2]`, `[3]` policy'den bağımsız (aynı L'de P0/P1/P2 eşit). Sadece shared `[4]`/`[5]` policy'ye duyarlı (bkz. §2).

| L | [1] CPU inf | [2] CPU FIR | [3] Coproc inf: setup/load/compute/**total** | cyc/MAC | speedup | [5] shared coproc **total** (alone) | [4] shared CPU FIR **total** (alone) |
|:-:|:-----------:|:-----------:|:--------------------------------------------:|:-------:|:-------:|:-----------------------------------:|:------------------------------------:|
| 0 | 701 086 | 33 936 | 1359 / 2559 / 98 459 / **128 446** | 1.14 | 5.45× | 131 974 (128 446) | 34 591 (33 936) |
| 1 | 701 854 | 33 936 | 1359 / 2559 / 98 586 / **129 469** | 1.14 | 5.42× | 133 138 (129 469) | 34 658 (33 936) |
| 2 | 702 750 | 34 064 | 1359 / 2559 / 98 703 / **130 482** | 1.14 | 5.38× | 134 230 (130 482) | 34 857 (34 064) |
| 3 | 703 774 | 34 192 | 1359 / 2559 / 98 802 / **131 477** | 1.14 | 5.35× | 135 277 (131 477) | 34 927 (34 192) |
| 4 | 705 438 | 34 320 | 1359 / 2559 / 98 886 / **132 457** | 1.14 | 5.32× | 136 500 (132 457) | 35 288 (34 320) |
| 5 | 707 102 | 34 448 | 1359 / 2559 / 98 914 / **133 381** | 1.14 | 5.30× | 137 345 (133 381) | 35 158 (34 448) |

**Okunuşu:**
- **`[3]` compute (coproc inference) L boyunca sabit:** 98 459 → 98 914 (+0.5%), **cyc/MAC = 1.14 düz** her L'de → pipelining FMA latency'sini gizliyor (serial olsa 1.14+L olurdu). setup=1359, load=2559 sabit.
- **speedup 5.45×→5.30×** (hafif düşüş: `[1]` CPU golden de FMA'yı L latency'sinde kullanıp yavaşlıyor).
- **`[1]` ve `[2]`** L ile hafif büyüyor (CPU da FMA'yı L'de kullanıyor).
- **Shared:** coproc inference +%2.7 (L0), CPU FIR +%1.9 (L0) — ikisi tek FMA'yı paylaşırken düşük çakışma.

---

## 1b. Gerçek zaman (cycle × 3.845 ns @ 260 MHz, P0)

DC sentezinde çekirdek (`cv32e40px_top`) **3.845 ns'de WNS=0** → gerçek fmax **260 MHz**. Verilator cycle × 3.845 ns = gerçek süre.

| L | CPU inf | CPU FIR | coproc inf | shared coproc | shared CPU FIR |
|:-:|--------:|--------:|-----------:|--------------:|---------------:|
| 0 | 2695.68 µs | 130.48 µs | 493.87 µs | 507.44 µs | 133.00 µs |
| 1 | 2698.63 µs | 130.48 µs | 497.81 µs | 511.92 µs | 133.26 µs |
| 2 | 2702.07 µs | 130.98 µs | 501.70 µs | 516.11 µs | 134.03 µs |
| 3 | 2706.01 µs | 131.47 µs | 505.53 µs | 520.14 µs | 134.29 µs |
| 4 | 2712.41 µs | 131.96 µs | 509.30 µs | 524.84 µs | 135.68 µs |
| 5 | 2718.81 µs | 132.45 µs | 512.85 µs | 528.09 µs | 135.18 µs |

**Resim başına (L0, B=8):** coproc inference **61.7 µs/img**, CPU inference **336.96 µs/img** → coproc **5.46× hızlı** (gerçek zamanda).
**Not:** periyot = **core** sentez fmax'i (260 MHz). Tam-SoC clock'u (DMA/bus/memory dahil) farklı olabilir → onun için full-SoC sentez gerekir. Herhangi bir cycle değerini `× 3.845 ns` ile zamana çevirebilirsin.

---

## 2. Policy karşılaştırması @ L=0 (shared senaryo)

`[1]`=701 086, `[2]`=33 936, `[3]` total=128 446 **her policy'de aynı** (policy sadece paylaşımı etkiler). Fark shared `[4]`/`[5]`'te:

| Config (L0) | [5] shared coproc **total** (alone 128 446) | [4] shared CPU FIR **total** (alone 33 936) |
|---|:---:|:---:|
| **P0** CPU-strict | 131 974 (+2.7%) | 34 591 (**+1.9%**) |
| **P1** round-robin | 131 969 (+2.7%) | 34 586 (+1.9%) |
| **P2** QoS `w1-1-1` (eşit) | 131 709 (+2.5%) | 38 247 (+12.7%) |
| **P2** QoS `w4-1-1` (CPU-öncelik) | 131 785 (+2.6%) | 35 433 (+4.4%) |
| **P2** QoS `w1-4-1` (acc-öncelik) | **129 392 (+0.7%)** | 39 508 (**+16.4%**) |
| **P2** QoS `w1-4-4` (acc-öncelik) | 129 392 (+0.7%) | 39 508 (+16.4%) |

**Okunuşu:** policy CPU-vs-coproc önceliğini ayarlıyor:
- **P0/P1:** CPU'yu korur → CPU FIR sadece **+1.9%** yavaşlar (coproc +2.7%).
- **P2 acc-öncelik (`w1-4-*`):** coproc'u korur → coproc inference **+0.7%** (neredeyse hiç), ama CPU FIR **+16.4%**.
- **P2 CPU-öncelik (`w4-1-1`):** ara denge (FIR +4.4%).

---

## 3. Özet gözlemler
- **Pipelining çalışıyor:** coproc inference cyc/MAC = **1.14 düz** (L=0..5), compute ~98.5K sabit.
- **CPU'ya karşı ~5.3–5.45× hızlı**, her L'de.
- **Paylaşım ucuz:** P0'da coproc +2.7% / CPU-FIR +1.9% — iki iş tek FMA'yı düşük çakışmayla paylaşıyor.
- **Policy = CPU↔coproc önceliği kaldıracı:** P0/P1 CPU'yu korur; P2 ağırlıklarıyla dengeyi (+1.9%…+16.4% CPU) ayarlarsın.
- **21/21 bit-exact.**

---

## Appendix — tüm 21 config, 5 total cycle

| Config | [1] CPU inf | [2] CPU FIR | [3] coproc inf | [5] shared coproc | [4] shared CPU FIR |
|---|:---:|:---:|:---:|:---:|:---:|
| L0_P0 | 701 086 | 33 936 | 128 446 | 131 974 | 34 591 |
| L0_P1 | 701 086 | 33 936 | 128 446 | 131 969 | 34 586 |
| L0_P2 w1-1-1 | 701 086 | 33 936 | 128 446 | 131 709 | 38 247 |
| L0_P2 w4-1-1 | 701 086 | 33 936 | 128 446 | 131 785 | 35 433 |
| L0_P2 w1-4-1 | 701 086 | 33 936 | 128 446 | 129 392 | 39 508 |
| L0_P2 w1-4-4 | 701 086 | 33 936 | 128 446 | 129 392 | 39 508 |
| L1_P0 | 701 854 | 33 936 | 129 469 | 133 138 | 34 658 |
| L1_P1 | 701 854 | 33 936 | 129 469 | 133 127 | 34 714 |
| L1_P2 w4-1-1 | 701 854 | 33 936 | 129 469 | 132 926 | 35 551 |
| L2_P0 | 702 750 | 34 064 | 130 482 | 134 230 | 34 857 |
| L2_P1 | 702 750 | 34 064 | 130 482 | 134 282 | 34 902 |
| L2_P2 w4-1-1 | 702 750 | 34 064 | 130 482 | 133 915 | 35 650 |
| L3_P0 | 703 774 | 34 192 | 131 477 | 135 277 | 34 927 |
| L3_P1 | 703 774 | 34 192 | 131 477 | 135 275 | 34 927 |
| L3_P2 w4-1-1 | 703 774 | 34 192 | 131 477 | 134 896 | 35 660 |
| L4_P0 | 705 438 | 34 320 | 132 457 | 136 500 | 35 288 |
| L4_P1 | 705 438 | 34 320 | 132 457 | 136 529 | 35 328 |
| L4_P2 w4-1-1 | 705 438 | 34 320 | 132 457 | 135 955 | 35 777 |
| L5_P0 | 707 102 | 34 448 | 133 381 | 137 345 | 35 158 |
| L5_P1 | 707 102 | 34 448 | 133 381 | 137 345 | 35 158 |
| L5_P2 w4-1-1 | 707 102 | 34 448 | 133 381 | 136 966 | 35 927 |

*Kaynak: `sweep_results/bench_pipe/` (21 config, hepsi `ok` + `bit-exact=1`). Coproc inference: setup=1359, load=2559 tüm config'lerde sabit; compute L ile ~%0.5 değişir.*

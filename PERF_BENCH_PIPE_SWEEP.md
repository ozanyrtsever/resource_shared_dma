# perf_bench_pipe — sweep sonuçları (toy MLP, pipelined single coprocessor)

**Workload:** MLP `128-64-32-16`, FP32, **B=8** (10 752 MAC/img, 86 016 MAC/batch).
**Donanım:** pipelined tek coprocessor (`dma_fp_dot_accel_pipe`), CPU'nun tek FMA'sını paylaşır (ikinci FPU yok), APU arbiter. Verilator, trace off.
**Sweep:** L (fpu_addmul_lat) = 0..5 × arbiter policy P = 0/1/2 (+ L=0'da QoS ağırlık varyantları). Klasör: `sweep_results/bench_pipe/`. **21/21 config `ok`, hepsi `bit-exact=1`.**

**CPU baseline = optimize batched inference** (`fc_cpu`: ağırlık batch boyunca tekrar kullanılır + 8 bağımsız akümülatör; disassembly-verified). Naive değil → speedup şişkin değil, savunulabilir. (Naive L=0'da 8.15 cyc/MAC idi; optimize 3.72.)

**8 metrik** (her biri setup/load [DMA yoksa 0] / compute / total, cycle):
`[1]` CPU alone inference · `[2]` CPU alone FIR · `[3]` Coproc alone inference · `[4]` Shared CPU FIR · `[5]` Shared Coproc inference · `[6]` CPU alone **MAC** (saf-FMA, `cyc/fmadd` basar) · `[7]` Shared CPU MAC · `[8]` Shared Coproc inference (MAC).
FIR = CPU'nun kendi 32-tap DSP işi (memory-bound). **MAC = saf-FMA CPU kernel'i** (register-only, 8 bağımsız akümülatör) — her cycle FMA ister, arbiter'ı strese sokar. FIR ve MAC **ayrı run'larda** paylaşır (aynı anda değil).

---

## 1. Ana tablo — L-sweep @ P0 (CPU-strict)

`[1]`, `[2]`, `[3]` policy'den bağımsız (aynı L'de P0/P1/P2 eşit). Sadece shared `[4]`/`[5]` policy'ye duyarlı (bkz. §2).

| L | [1] CPU inf | [2] CPU FIR | [3] Coproc inf: setup/load/compute/**total** | cyc/MAC | speedup | [5] shared coproc **total** (alone) | [4] shared CPU FIR **total** (alone) |
|:-:|:-----------:|:-----------:|:--------------------------------------------:|:-------:|:-------:|:-----------------------------------:|:------------------------------------:|
| 0 | 319 732 | 33 936 | 1368 / 2559 / 98 421 / **129 169** | 1.14 | 2.47× | 132 913 (129 169) | 34 804 (33 936) |
| 1 | 319 732 | 33 936 | 1368 / 2559 / 98 453 / **130 097** | 1.14 | 2.45× | 134 099 (130 097) | 34 920 (33 936) |
| 2 | 482 676 | 34 064 | 1368 / 2559 / 98 593 / **131 133** | 1.14 | 3.68× | 135 271 (131 133) | 35 287 (34 064) |
| 3 | 548 068 | 34 192 | 1368 / 2559 / 98 752 / **132 188** | 1.14 | 4.14× | 136 265 (132 188) | 35 318 (34 192) |
| 4 | 624 212 | 34 320 | 1368 / 2559 / 98 858 / **133 190** | 1.14 | 4.68× | 137 206 (133 190) | 35 304 (34 320) |
| 5 | 700 372 | 34 448 | 1368 / 2559 / 98 844 / **134 072** | 1.14 | 5.22× | 137 994 (134 072) | 35 206 (34 448) |

**Okunuşu:**
- **`[3]` coproc compute L boyunca sabit:** ~98.5K, **cyc/MAC = 1.14 düz** her L'de → pipelining FMA latency'sini gizliyor (B bağımsız MAC arka arkaya → pipeline dolu).
- **speedup L ile ARTIYOR: 2.47× → 5.22×.** Sebep: **optimize CPU inference L ile yavaşlıyor** (CPU cyc/MAC: L0,1'de **3.72**, sonra 5.61 / 6.37 / 7.26 / **8.14**), çünkü **cv32e40p ~2 outstanding FP** tutabiliyor → yüksek L'de FMA pipeline'ını dolduramıyor, latency-bound oluyor. Coproc ise **düz 1.14** (pipeline'ı dolduruyor). Aradaki fark = coproc'un **pipeline-doldurma avantajı**, ve L (pipeline derinliği) arttıkça **büyüyor**.
- **L=0 coproc'un en zorlandığı nokta** (speedup en düşük, 2.47×): sığ pipeline'ı sınırlı CPU bile besleyebiliyor. Pipelined FMA'da (L≥2) coproc net öne geçiyor.
- **Shared (FIR):** coproc inference +%2.9 (L0), CPU FIR +%2.6 (L0) — memory-bound FIR ile düşük çakışma. **Saf-FMA çekişmesi için MAC → §2b.**

---

## 1b. Gerçek zaman (cycle × 3.845 ns @ 260 MHz, P0)

DC sentezinde çekirdek (`cv32e40px_top`) **3.845 ns'de WNS=0** → gerçek fmax **260 MHz**. Verilator cycle × 3.845 ns = gerçek süre.

| L | CPU inf | CPU FIR | coproc inf | shared coproc | shared CPU FIR |
|:-:|--------:|--------:|-----------:|--------------:|---------------:|
| 0 | 1229.37 µs | 130.48 µs | 496.65 µs | 511.05 µs | 133.82 µs |
| 1 | 1229.37 µs | 130.48 µs | 500.22 µs | 515.61 µs | 134.27 µs |
| 2 | 1855.89 µs | 130.98 µs | 504.21 µs | 520.12 µs | 135.68 µs |
| 3 | 2107.32 µs | 131.47 µs | 508.26 µs | 523.94 µs | 135.80 µs |
| 4 | 2400.10 µs | 131.96 µs | 512.12 µs | 527.56 µs | 135.74 µs |
| 5 | 2692.93 µs | 132.45 µs | 515.51 µs | 530.59 µs | 135.37 µs |

**Resim başına (L0, B=8):** coproc inference **62.1 µs/img**, CPU inference **153.7 µs/img** → coproc **2.48× hızlı** (gerçek zamanda). (L=5'te fark ~5.2×'e çıkar — CPU latency-bound.)
**Not:** periyot = **core** sentez fmax'i (260 MHz). Tam-SoC clock'u (DMA/bus/memory dahil) farklı olabilir → onun için full-SoC sentez gerekir. Herhangi bir cycle değerini `× 3.845 ns` ile zamana çevirebilirsin.

---

## 2. Policy karşılaştırması @ L=0 — FIR shared senaryosu

`[1]`=319 732, `[2]`=33 936, `[3]` total=129 169 **her policy'de aynı** (policy sadece paylaşımı etkiler). Fark shared `[4]`/`[5]`'te:

| Config (L0) | [5] shared coproc **total** (alone 129 169) | [4] shared CPU FIR **total** (alone 33 936) |
|---|:---:|:---:|
| **P0** CPU-strict | 132 913 (+2.9%) | 34 804 (**+2.6%**) |
| **P1** round-robin | 132 913 (+2.9%) | 34 804 (+2.6%) |
| **P2** QoS `w1-1-1` (eşit) | 132 494 (+2.6%) | 38 420 (+13.2%) |
| **P2** QoS `w4-1-1` (CPU-öncelik) | 132 520 (+2.6%) | 35 503 (+4.6%) |
| **P2** QoS `w1-4-1` (acc-öncelik) | **130 290 (+0.9%)** | 40 052 (**+18.0%**) |
| **P2** QoS `w1-4-4` (acc-öncelik) | 130 290 (+0.9%) | 40 052 (+18.0%) |

**Okunuşu:** FIR memory-bound olduğundan **P0 ile P1 aynı** (CPU FIR +2.6% vs +2.6%) — FMA'yı aynı cycle'da nadiren istiyor, çakışma az. Politika farkı ancak QoS ile görünüyor:
- **P0/P1:** CPU FIR **+2.6%** (coproc +2.9%).
- **P2 acc-öncelik (`w1-4-*`):** coproc'u korur → coproc **+0.9%**, CPU FIR **+18.0%**.
- **P2 CPU-öncelik (`w4-1-1`):** ara denge (FIR +4.6%).

> **P0 vs P1 burada ayrışmıyor** çünkü FIR FMA'yı doldurmuyor. Gerçek arbiter ayrışması için **saf-FMA çekişmesi** → **§2b**.

---

## 2b. Saf-FMA çekişmesi (MAC kernel) — arbiter policy'lerinin AYRIŞTIĞI yer

MAC = register-only, **8 bağımsız akümülatör**lü CPU kernel'i → FMA'yı ~her cycle ister. Coproc ile aynı FMA'yı gerçekten aynı cycle'da çekiştirir → policy burada belirleyici olur.

### (a) `[6]` cyc/fmadd — CPU'nun FP-issue tavanı (tek başına, coproc yok)

| L | 0 | 1 | 2 | 3 | 4 | 5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **cyc/fmadd** | **1.51** | **1.51** | 3.26 | 4.13 | 5.13 | 6.00 |

**Bulgu:** L≤1'de düz **~1.51** (issue-bound — 8 akümülatör latency'yi gizliyor); L≥2'de **≈L+1** (latency-bound). Yani **cv32e40p'nin APU'su ~2 outstanding FP op** tutabiliyor; ötesinde 8 akümülatöre rağmen CPU kendi latency'sine takılıyor. Bu, §1'deki "speedup L ile artıyor"un da sebebi — optimize inference de aynı tavana çarpıyor (CPU cyc/MAC 3.72→8.14).

### (b) `[7]`/`[8]` P0 vs P1 — CPU-MAC ve coproc yavaşlaması

macA = `[6]` alone; co = `[3]` alone. Parantez = shared/alone yüzdesi.

| L | cyc/fmadd | [6] macA | **P0** CPU-MAC | **P1** CPU-MAC | P0 coproc | P1 coproc |
|:-:|:-:|--:|:--:|:--:|:--:|:--:|
| 0 | 1.51 | 49 581 | 52 901 (**+6.7%**) | 81 619 (**+64.6%**) | 159 778 (+23.7%) | 157 840 (+22.2%) |
| 1 | 1.51 | 49 581 | 52 906 (**+6.7%**) | 81 618 (**+64.6%**) | 160 738 (+23.6%) | 158 780 (+22.0%) |
| 2 | 3.26 | 107 053 | 110 304 (+3.0%) | 110 313 (+3.0%) | 162 727 (+24.1%) | 162 735 (+24.1%) |
| 3 | 4.13 | 135 597 | 138 425 (+2.1%) | 138 456 (+2.1%) | 158 961 (+20.3%) | 158 986 (+20.3%) |
| 4 | 5.13 | 168 238 | 170 345 (+1.3%) | 170 345 (+1.3%) | 147 562 (+10.8%) | 147 562 (+10.8%) |
| 5 | 6.00 | 196 911 | 198 784 (+1.0%) | 198 792 (+1.0%) | 152 081 (+13.4%) | 152 074 (+13.4%) |

**Okunuşu — asıl arbiter sonucu:**
- **L=0,1 (CPU issue-bound → gerçek çekişme):** **P0** CPU-MAC'i sadece **+6.7%** yavaşlatır (CPU korunur, coproc çekişmeyi yer); **P1** CPU-MAC'i **+64.6%** yavaşlatır (round-robin FMA'yı böler). **~58 puanlık fark** — oysa FIR'de fark ~%0'dı (§2). **CPU-priority garantisinin "dişleri" ölçüldü.**
- **L≥2 (CPU latency-bound → çekişme yok):** P0 ≡ P1 (fark 0.0 pp). cv32e40p FP'yi seyrek issue ediyor, coproc'la aynı cycle'da nadiren çakışıyor → tüm policy'ler birleşiyor. **Politika, CPU FMA'yı doyurabildiğinde (cv32e40p'de L≤1) belirleyici.**

### (c) QoS @ L0 (MAC çekişmesi)

macA=49 581, co=129 169.

| Config | CPU-MAC | coproc |
|---|:--:|:--:|
| **P0** CPU-strict | 52 901 (+6.7%) | 159 778 (+23.7%) |
| **P1** round-robin | 81 619 (+64.6%) | 157 840 (+22.2%) |
| **P2** `w1-1-1` (eşit) | 81 247 (+63.9%) | 157 410 (+21.9%) |
| **P2** `w4-1-1` (CPU-öncelik) | 60 639 (+22.3%) | 160 036 (+23.9%) |
| **P2** `w1-4-1` (acc-öncelik) | 117 752 (+137.5%) | 146 367 (+13.3%) |

**Okunuşu:** QoS ağırlıkları CPU↔coproc dengesini sürekli ayarlıyor — `w4-1-1` CPU'yu korur (CPU +22.3%, P1'in +64.6%'sından iyi), `w1-4-1` coproc'u korur (CPU +137.5%, coproc sadece +13.3%). P1 ≈ P2-`w1-1-1`. **Saf-FMA çekişmesi altında politikalar tam bir spektrum oluşturuyor** — FIR'in gösteremediği.

---

## 3. Özet gözlemler
- **Pipelining çalışıyor:** coproc inference cyc/MAC = **1.14 düz** (L=0..5), compute ~98.5K sabit.
- **CPU'ya karşı 2.47× (L0) → 5.22× (L5), L ile ARTAN** — çünkü optimize CPU L ile yavaşlıyor (~2 outstanding, latency-bound: cyc/MAC 3.72→8.14), coproc düz. **Fark = pipeline-doldurma avantajı, derinlikle büyüyor.**
- **Baseline optimize + disassembly-verified** (ağırlık batch-reuse + 8-way ILP) → speedup şişkin değil, savunulabilir. En düşük speedup (L=0, 2.47×) coproc'un en zorlandığı nokta.
- **Paylaşım ucuz (FIR):** P0'da coproc +2.9% / CPU-FIR +2.6% — memory-bound FIR düşük çakışma.
- **Saf-FMA çekişmesi (MAC, §2b):** L≤1'de **P0 CPU'yu korur (+6.7%), P1 böler (+64.6%)** — ~58 puan fark, FIR'in gösteremediği. L≥2'de P0≡P1. QoS ağırlıkları CPU +22%…+137% ayarlıyor.
- **21/21 bit-exact** (8 metrik dahil).

---

## Appendix — tüm 21 config (8 total cycle)

`[6]` macA aynı L'de policy'den bağımsız sabit; `[7]`/`[8]` = MAC shared (CPU-MAC / coproc). §2b'nin ham verisi.

| Config | [1] CPU inf | [2] CPU FIR | [3] coproc inf | [5] sh coproc | [4] sh FIR | [6] macA | [7] sh MAC | [8] sh coproc(MAC) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| L0_P0 | 319 732 | 33 936 | 129 169 | 132 913 | 34 804 | 49 581 | 52 901 | 159 778 |
| L0_P1 | 319 732 | 33 936 | 129 169 | 132 913 | 34 804 | 49 581 | 81 619 | 157 840 |
| L0_P2 w1-1-1 | 319 732 | 33 936 | 129 169 | 132 494 | 38 420 | 49 581 | 81 247 | 157 410 |
| L0_P2 w4-1-1 | 319 732 | 33 936 | 129 169 | 132 520 | 35 503 | 49 581 | 60 639 | 160 036 |
| L0_P2 w1-4-1 | 319 732 | 33 936 | 129 169 | 130 290 | 40 052 | 49 581 | 117 752 | 146 367 |
| L0_P2 w1-4-4 | 319 732 | 33 936 | 129 169 | 130 290 | 40 052 | 49 581 | 117 752 | 146 367 |
| L1_P0 | 319 732 | 33 936 | 130 097 | 134 099 | 34 920 | 49 581 | 52 906 | 160 738 |
| L1_P1 | 319 732 | 33 936 | 130 097 | 134 069 | 34 927 | 49 581 | 81 618 | 158 780 |
| L1_P2 w4-1-1 | 319 732 | 33 936 | 130 097 | 133 712 | 35 665 | 49 581 | 60 615 | 161 009 |
| L2_P0 | 482 676 | 34 064 | 131 133 | 135 271 | 35 287 | 107 053 | 110 304 | 162 727 |
| L2_P1 | 482 676 | 34 064 | 131 133 | 135 273 | 35 290 | 107 053 | 110 313 | 162 735 |
| L2_P2 w4-1-1 | 482 676 | 34 064 | 131 133 | 134 632 | 35 744 | 107 053 | 117 935 | 162 365 |
| L3_P0 | 548 068 | 34 192 | 132 188 | 136 265 | 35 318 | 135 597 | 138 425 | 158 961 |
| L3_P1 | 548 068 | 34 192 | 132 188 | 136 265 | 35 318 | 135 597 | 138 456 | 158 986 |
| L3_P2 w4-1-1 | 548 068 | 34 192 | 132 188 | 135 654 | 35 727 | 135 597 | 144 735 | 159 706 |
| L4_P0 | 624 212 | 34 320 | 133 190 | 137 206 | 35 304 | 168 238 | 170 345 | 147 562 |
| L4_P1 | 624 212 | 34 320 | 133 190 | 137 253 | 35 354 | 168 238 | 170 345 | 147 562 |
| L4_P2 w4-1-1 | 624 212 | 34 320 | 133 190 | 136 632 | 35 866 | 168 238 | 174 590 | 151 805 |
| L5_P0 | 700 372 | 34 448 | 134 072 | 137 994 | 35 206 | 196 911 | 198 784 | 152 081 |
| L5_P1 | 700 372 | 34 448 | 134 072 | 137 994 | 35 206 | 196 911 | 198 792 | 152 074 |
| L5_P2 w4-1-1 | 700 372 | 34 448 | 134 072 | 137 665 | 36 004 | 196 911 | 202 051 | 148 946 |

*Kaynak: `sweep_results/bench_pipe/` (21 config, hepsi `ok` + `bit-exact=1`). CPU baseline = optimize `fc_cpu`. Coproc: setup=1368, load=2559 sabit; compute L ile ~%0.4 değişir. CPU inf L ile artıyor (3.72→8.14 cyc/MAC, ~2 outstanding). `[6]` cyc/fmadd: L=0,1→1.51; L=2→3.26; L=3→4.13; L=4→5.13; L=5→6.00.*

# perf_bench_pipe — sweep sonuçları (toy MLP, pipelined single coprocessor)

**Workload:** MLP `128-64-32-16`, FP32, **B=8** (10 752 MAC/img, 86 016 MAC/batch).
**Donanım:** pipelined tek coprocessor (`dma_fp_dot_accel_pipe`), CPU'nun tek FMA'sını paylaşır (ikinci FPU yok), APU arbiter. Verilator, trace off.
**Sweep:** L (fpu_addmul_lat) = 0..5 × arbiter policy P = 0/1/2 (+ L=0'da QoS ağırlık varyantları). Klasör: `sweep_results/bench_pipe/`. **21/21 config `ok`, hepsi `bit-exact=1`.**

**8 metrik** (her biri setup/load [DMA yoksa 0] / compute / total, cycle):
`[1]` CPU alone inference · `[2]` CPU alone FIR · `[3]` Coproc alone inference · `[4]` Shared CPU FIR · `[5]` Shared Coproc inference · `[6]` CPU alone **MAC** (saf-FMA, `cyc/fmadd` basar) · `[7]` Shared CPU MAC · `[8]` Shared Coproc inference (MAC).
FIR = CPU'nun kendi 32-tap DSP işi (memory+dependency-bound, ~8 cyc/fmadd). **MAC = saf-FMA CPU kernel'i** (register-only, 8 bağımsız akümülatör) — her cycle FMA ister, arbiter'ı gerçekten strese sokar. "shared" = coproc inference ederken CPU aynı anda o kernel'i koşar (tek FMA). FIR ve MAC **ayrı run'larda** paylaşır (aynı anda değil).

---

## 1. Ana tablo — L-sweep @ P0 (CPU-strict)

`[1]`, `[2]`, `[3]` policy'den bağımsız (aynı L'de P0/P1/P2 eşit). Sadece shared `[4]`/`[5]` policy'ye duyarlı (bkz. §2).

| L | [1] CPU inf | [2] CPU FIR | [3] Coproc inf: setup/load/compute/**total** | cyc/MAC | speedup | [5] shared coproc **total** (alone) | [4] shared CPU FIR **total** (alone) |
|:-:|:-----------:|:-----------:|:--------------------------------------------:|:-------:|:-------:|:-----------------------------------:|:------------------------------------:|
| 0 | 700 971 | 33 809 | 1362 / 2559 / 98 421 / **127 573** | 1.14 | 5.49× | 131 145 (127 573) | 34 590 (33 809) |
| 1 | 701 739 | 33 809 | 1362 / 2559 / 98 453 / **128 501** | 1.14 | 5.46× | 132 309 (128 501) | 34 657 (33 809) |
| 2 | 702 635 | 33 937 | 1362 / 2559 / 98 593 / **129 537** | 1.14 | 5.42× | 133 401 (129 537) | 34 856 (33 937) |
| 3 | 703 659 | 34 065 | 1362 / 2559 / 98 752 / **130 592** | 1.14 | 5.38× | 134 448 (130 592) | 34 926 (34 065) |
| 4 | 705 323 | 34 193 | 1362 / 2559 / 98 858 / **131 594** | 1.14 | 5.35× | 135 671 (131 594) | 35 287 (34 193) |
| 5 | 706 987 | 34 321 | 1362 / 2559 / 98 844 / **132 476** | 1.14 | 5.33× | 136 516 (132 476) | 35 157 (34 321) |

**Okunuşu:**
- **`[3]` compute (coproc inference) L boyunca sabit:** 98 421 → 98 844 (+0.4%), **cyc/MAC = 1.14 düz** her L'de → pipelining FMA latency'sini gizliyor (serial olsa 1.14+L olurdu). setup=1362, load=2559 sabit.
- **speedup 5.49×→5.33×** (hafif düşüş: `[1]` CPU golden de FMA'yı L latency'sinde kullanıp yavaşlıyor).
- **`[1]` ve `[2]`** L ile hafif büyüyor (CPU da FMA'yı L'de kullanıyor).
- **Shared (FIR):** coproc inference +%2.8 (L0), CPU FIR +%2.3 (L0) — memory-bound FIR ile düşük çakışma. **Saf-FMA çekişmesi için MAC kernel'i → §2b (asıl arbiter ayrışması orada).**

---

## 1b. Gerçek zaman (cycle × 3.845 ns @ 260 MHz, P0)

DC sentezinde çekirdek (`cv32e40px_top`) **3.845 ns'de WNS=0** → gerçek fmax **260 MHz**. Verilator cycle × 3.845 ns = gerçek süre.

| L | CPU inf | CPU FIR | coproc inf | shared coproc | shared CPU FIR |
|:-:|--------:|--------:|-----------:|--------------:|---------------:|
| 0 | 2695.23 µs | 130.00 µs | 490.52 µs | 504.25 µs | 133.00 µs |
| 1 | 2698.19 µs | 130.00 µs | 494.09 µs | 508.73 µs | 133.26 µs |
| 2 | 2701.63 µs | 130.49 µs | 498.07 µs | 512.93 µs | 134.02 µs |
| 3 | 2705.57 µs | 130.98 µs | 502.13 µs | 516.95 µs | 134.29 µs |
| 4 | 2711.97 µs | 131.47 µs | 505.98 µs | 521.65 µs | 135.68 µs |
| 5 | 2718.36 µs | 131.96 µs | 509.37 µs | 524.90 µs | 135.18 µs |

**Resim başına (L0, B=8):** coproc inference **61.3 µs/img**, CPU inference **336.9 µs/img** → coproc **5.49× hızlı** (gerçek zamanda).
**Not:** periyot = **core** sentez fmax'i (260 MHz). Tam-SoC clock'u (DMA/bus/memory dahil) farklı olabilir → onun için full-SoC sentez gerekir. Herhangi bir cycle değerini `× 3.845 ns` ile zamana çevirebilirsin.

---

## 2. Policy karşılaştırması @ L=0 — FIR shared senaryosu

`[1]`=700 971, `[2]`=33 809, `[3]` total=127 573 **her policy'de aynı** (policy sadece paylaşımı etkiler). Fark shared `[4]`/`[5]`'te:

| Config (L0) | [5] shared coproc **total** (alone 127 573) | [4] shared CPU FIR **total** (alone 33 809) |
|---|:---:|:---:|
| **P0** CPU-strict | 131 145 (+2.8%) | 34 590 (**+2.3%**) |
| **P1** round-robin | 131 140 (+2.8%) | 34 585 (+2.3%) |
| **P2** QoS `w1-1-1` (eşit) | 130 880 (+2.6%) | 38 246 (+13.1%) |
| **P2** QoS `w4-1-1` (CPU-öncelik) | 130 956 (+2.7%) | 35 432 (+4.8%) |
| **P2** QoS `w1-4-1` (acc-öncelik) | **128 563 (+0.8%)** | 39 507 (**+16.9%**) |
| **P2** QoS `w1-4-4` (acc-öncelik) | 128 563 (+0.8%) | 39 507 (+16.9%) |

**Okunuşu:** FIR memory-bound olduğundan **P0 ile P1 neredeyse aynı** (CPU FIR +2.3% vs +2.3%) — FMA'yı aynı cycle'da nadiren istiyor, çakışma az. Politika farkı ancak QoS ağırlıklarıyla görünüyor:
- **P0/P1:** CPU FIR **+2.3%** (coproc +2.8%).
- **P2 acc-öncelik (`w1-4-*`):** coproc'u korur → coproc **+0.8%**, CPU FIR **+16.9%**.
- **P2 CPU-öncelik (`w4-1-1`):** ara denge (FIR +4.8%).

> **P0 vs P1 burada ayrışmıyor** çünkü FIR FMA'yı doldurmuyor. Gerçek arbiter ayrışması için **saf-FMA çekişmesi** gerek → **§2b**.

---

## 2b. Saf-FMA çekişmesi (MAC kernel) — arbiter policy'lerinin AYRIŞTIĞI yer

MAC = register-only, **8 bağımsız akümülatör**lü CPU kernel'i → FMA'yı ~her cycle ister (FIR gibi memory-bound değil). Coproc ile aynı FMA'yı gerçekten aynı cycle'da çekiştirir → policy burada belirleyici olur.

### (a) `[6]` cyc/fmadd — CPU'nun FP-issue tavanı (tek başına, coproc yok)

| L | 0 | 1 | 2 | 3 | 4 | 5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **cyc/fmadd** | **1.51** | **1.51** | 3.27 | 4.14 | 5.13 | 6.01 |

**Bulgu:** L≤1'de düz **~1.51** (issue-bound — 8 akümülatör latency'yi gizliyor); L≥2'de **≈L+1** (latency-bound). Yani **cv32e40p'nin APU'su ~2 outstanding FP op** tutabiliyor; ötesinde 8 akümülatöre rağmen CPU kendi latency'sine takılıyor. Bu bir **cv32e40p donanım limiti** (kernel kusuru değil), ve arbiter'ın ne kadar çekişme görebileceğini belirliyor.

### (b) `[7]`/`[8]` P0 vs P1 — CPU-MAC ve coproc yavaşlaması

macA = `[6]` alone; co = `[3]` alone. Parantez = shared/alone yüzdesi.

| L | cyc/fmadd | [6] macA | **P0** CPU-MAC | **P1** CPU-MAC | P0 coproc | P1 coproc |
|:-:|:-:|--:|:--:|:--:|:--:|:--:|
| 0 | 1.51 | 49 709 | 52 909 (**+6.4%**) | 81 341 (**+63.6%**) | 158 421 (+24.2%) | 155 887 (+22.2%) |
| 1 | 1.51 | 49 709 | 52 905 (**+6.4%**) | 81 256 (**+63.5%**) | 159 407 (+24.1%) | 156 863 (+22.1%) |
| 2 | 3.27 | 107 181 | 110 197 (+2.8%) | 110 201 (+2.8%) | 160 607 (+24.0%) | 160 611 (+24.0%) |
| 3 | 4.14 | 135 725 | 138 465 (+2.0%) | 138 471 (+2.0%) | 157 603 (+20.7%) | 157 605 (+20.7%) |
| 4 | 5.13 | 168 366 | 170 440 (+1.2%) | 170 440 (+1.2%) | 146 173 (+11.1%) | 146 173 (+11.1%) |
| 5 | 6.01 | 197 039 | 198 800 (+0.9%) | 198 843 (+0.9%) | 150 545 (+13.6%) | 150 565 (+13.7%) |

**Okunuşu — asıl sonuç:**
- **L=0,1 (CPU issue-bound → gerçek çekişme):** **P0** CPU-MAC'i sadece **+6.4%** yavaşlatır (CPU korunur, coproc çekişmeyi yer); **P1** CPU-MAC'i **+63.6%** yavaşlatır (round-robin FMA'yı böler). **~57 puanlık fark** — oysa FIR'de fark ~%0'dı (P0 34 590 vs P1 34 585). **CPU-priority garantisinin "dişleri" ilk kez görünüyor.**
- **L≥2 (CPU latency-bound → çekişme yok):** P0 ≡ P1 (fark 0.0 pp). cv32e40p FP'yi seyrek issue ediyor (≈L+1), coproc'la aynı cycle'da nadiren çakışıyor → tüm policy'ler birleşiyor. **Politika, CPU FMA'yı doyurabildiğinde (cv32e40p'de L≤1) belirleyici.**

### (c) QoS @ L0 (MAC çekişmesi) — ağırlıklarla ayar

macA=49 709, co=127 573.

| Config | CPU-MAC | coproc |
|---|:--:|:--:|
| **P0** CPU-strict | 52 909 (+6.4%) | 158 421 (+24.2%) |
| **P1** round-robin | 81 341 (+63.6%) | 155 887 (+22.2%) |
| **P2** `w1-1-1` (eşit) | 81 338 (+63.6%) | 155 841 (+22.2%) |
| **P2** `w4-1-1` (CPU-öncelik) | 60 676 (+22.1%) | 158 464 (+24.2%) |
| **P2** `w1-4-1` (acc-öncelik) | 117 599 (+136.6%) | 144 393 (+13.2%) |

**Okunuşu:** QoS ağırlıkları CPU↔coproc dengesini sürekli ayarlıyor — `w4-1-1` CPU'yu korur (CPU +22%, P1'in +64%'ünden iyi), `w1-4-1` coproc'u korur (CPU +137%, coproc sadece +13%). P1 ≈ P2-`w1-1-1` (eşit ağırlık = round-robin). **Saf-FMA çekişmesi altında politikalar tam bir spektrum oluşturuyor** — FIR'in gösteremediği.

---

## 3. Özet gözlemler
- **Pipelining çalışıyor:** coproc inference cyc/MAC = **1.14 düz** (L=0..5), compute ~98.5K sabit.
- **CPU'ya karşı ~5.33–5.49× hızlı**, her L'de.
- **Paylaşım ucuz (FIR):** P0'da coproc +2.8% / CPU-FIR +2.3% — memory-bound FIR tek FMA'yı düşük çakışmayla paylaşıyor.
- **Saf-FMA çekişmesi (MAC, §2b) — asıl arbiter sonucu:** L≤1'de (CPU issue-bound) **P0 CPU'yu korur (+6.4%), P1 böler (+63.6%)** — ~57 puan fark, FIR'in gösteremediği. L≥2'de cv32e40p FP'yi seyrek issue ettiğinden çakışma kalmıyor → P0≡P1. QoS ağırlıkları dengeyi sürekli ayarlıyor (CPU +22%…+137%).
- **cv32e40p FP-issue tavanı:** cyc/fmadd L≤1'de **~1.51** (issue-bound), L≥2'de **≈L+1** → APU ~2 outstanding FP tutuyor.
- **21/21 bit-exact** (8 metrik dahil).

---

## Appendix — tüm 21 config (8 total cycle)

`[6]` macA (MAC alone, saf-FMA) aynı L'de policy'den bağımsız sabit; `[7]`/`[8]` = MAC shared (CPU-MAC / coproc). §2b'nin ham verisi.

| Config | [1] CPU inf | [2] CPU FIR | [3] coproc inf | [5] sh coproc | [4] sh FIR | [6] macA | [7] sh MAC | [8] sh coproc(MAC) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| L0_P0 | 700 971 | 33 809 | 127 573 | 131 145 | 34 590 | 49 709 | 52 909 | 158 421 |
| L0_P1 | 700 971 | 33 809 | 127 573 | 131 140 | 34 585 | 49 709 | 81 341 | 155 887 |
| L0_P2 w1-1-1 | 700 971 | 33 809 | 127 573 | 130 880 | 38 246 | 49 709 | 81 338 | 155 841 |
| L0_P2 w4-1-1 | 700 971 | 33 809 | 127 573 | 130 956 | 35 432 | 49 709 | 60 676 | 158 464 |
| L0_P2 w1-4-1 | 700 971 | 33 809 | 127 573 | 128 563 | 39 507 | 49 709 | 117 599 | 144 393 |
| L0_P2 w1-4-4 | 700 971 | 33 809 | 127 573 | 128 563 | 39 507 | 49 709 | 117 599 | 144 393 |
| L1_P0 | 701 739 | 33 809 | 128 501 | 132 309 | 34 657 | 49 709 | 52 905 | 159 407 |
| L1_P1 | 701 739 | 33 809 | 128 501 | 132 298 | 34 713 | 49 709 | 81 256 | 156 863 |
| L1_P2 w4-1-1 | 701 739 | 33 809 | 128 501 | 132 097 | 35 550 | 49 709 | 60 683 | 159 451 |
| L2_P0 | 702 635 | 33 937 | 129 537 | 133 401 | 34 856 | 107 181 | 110 197 | 160 607 |
| L2_P1 | 702 635 | 33 937 | 129 537 | 133 453 | 34 901 | 107 181 | 110 201 | 160 611 |
| L2_P2 w4-1-1 | 702 635 | 33 937 | 129 537 | 133 086 | 35 649 | 107 181 | 118 007 | 161 089 |
| L3_P0 | 703 659 | 34 065 | 130 592 | 134 448 | 34 926 | 135 725 | 138 465 | 157 603 |
| L3_P1 | 703 659 | 34 065 | 130 592 | 134 446 | 34 926 | 135 725 | 138 471 | 157 605 |
| L3_P2 w4-1-1 | 703 659 | 34 065 | 130 592 | 134 067 | 35 659 | 135 725 | 144 702 | 157 138 |
| L4_P0 | 705 323 | 34 193 | 131 594 | 135 671 | 35 287 | 168 366 | 170 440 | 146 173 |
| L4_P1 | 705 323 | 34 193 | 131 594 | 135 700 | 35 327 | 168 366 | 170 440 | 146 173 |
| L4_P2 w4-1-1 | 705 323 | 34 193 | 131 594 | 135 126 | 35 776 | 168 366 | 174 676 | 150 382 |
| L5_P0 | 706 987 | 34 321 | 132 476 | 136 516 | 35 157 | 197 039 | 198 800 | 150 545 |
| L5_P1 | 706 987 | 34 321 | 132 476 | 136 516 | 35 157 | 197 039 | 198 843 | 150 565 |
| L5_P2 w4-1-1 | 706 987 | 34 321 | 132 476 | 136 137 | 35 926 | 197 039 | 202 088 | 147 459 |

*Kaynak: `sweep_results/bench_pipe/` (21 config, hepsi `ok` + `bit-exact=1`). Coproc inference: setup=1362, load=2559 sabit; compute L ile ~%0.4 değişir. `[6]` cyc/fmadd: L=0,1→1.51; L=2→3.27; L=3→4.14; L=4→5.13; L=5→6.01.*

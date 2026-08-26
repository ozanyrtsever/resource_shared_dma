# perf_lenet_pipe — sweep sonuçları (gerçek LeNet-300-100 / MNIST, pipelined single coprocessor)

**Workload:** LeNet-300-100 `784-300-100-10`, FP32, **B=8** (266 200 MAC/img, 2 129 600 MAC/batch). Gerçek eğitilmiş ağırlıklar + gerçek MNIST görüntüleri.
**Donanım:** pipelined tek coprocessor (`dma_fp_dot_accel_pipe`), CPU'nun tek FMA'sını paylaşır (ikinci FPU yok), APU arbiter. Verilator, trace off.
**Sweep:** L = 0..5 × arbiter policy P = 0/1/2 (+ L=0'da QoS ağırlık varyantları). Klasör: `sweep_results/lenet_pipe/`. **21/21 config `ok`, hepsi `bit-exact=1` ve `correct=8/8, match_ref=8/8`.**

**8 metrik** (setup/load [DMA yoksa 0] / compute / total, cycle):
`[1]` CPU alone inference · `[2]` CPU alone FIR · `[3]` Coproc alone inference · `[4]` Shared CPU FIR · `[5]` Shared Coproc inference · `[6]` CPU alone **MAC** (saf-FMA, `cyc/fmadd` basar) · `[7]` Shared CPU MAC · `[8]` Shared Coproc inference (MAC).
FIR = CPU'nun kendi 32-tap DSP işi (memory-bound). **MAC = saf-FMA CPU kernel'i** (register-only, 8 bağımsız akümülatör) — her cycle FMA ister, arbiter'ı gerçekten strese sokar. FIR ve MAC **ayrı run'larda** coproc ile paylaşır (aynı anda değil).

---

## 1. Ana tablo — L-sweep @ P0 (CPU-strict)

`[1]`,`[2]`,`[3]` policy'den bağımsız; sadece shared `[4]`/`[5]` policy'ye duyarlı (§2).

| L | [1] CPU inf | [2] CPU FIR | [3] Coproc inf: setup/load/compute/**total** | cyc/MAC | speedup | [5] shared coproc **total** (alone) | [4] shared CPU FIR **total** (alone) | acc |
|:-:|:-----------:|:-----------:|:--------------------------------------------:|:-------:|:-------:|:-----------------------------------:|:------------------------------------:|:---:|
| 0 | 21 348 322 | 33 936 | 1644 / 12 183 / 2 401 094 / **2 526 962** | 1.12 | 8.44× | 2 529 649 (2 526 962) | 35 765 (33 936) | 8/8 |
| 1 | 21 351 602 | 33 936 | 1644 / 12 183 / 2 401 869 / **2 531 017** | 1.12 | 8.43× | 2 533 659 (2 531 017) | 35 897 (33 936) | 8/8 |
| 2 | 21 354 882 | 34 064 | 1644 / 12 183 / 2 402 279 / **2 534 707** | 1.12 | 8.42× | 2 537 282 (2 534 707) | 36 048 (34 064) | 8/8 |
| 3 | 21 358 162 | 34 192 | 1644 / 12 183 / 2 402 309 / **2 538 017** | 1.12 | 8.41× | 2 541 114 (2 538 017) | 36 038 (34 192) | 8/8 |
| 4 | 21 361 442 | 34 320 | 1644 / 12 183 / 2 402 330 / **2 541 318** | 1.12 | 8.40× | 2 544 501 (2 541 318) | 36 293 (34 320) | 8/8 |
| 5 | 21 368 002 | 34 448 | 1644 / 12 183 / 2 402 758 / **2 545 026** | 1.12 | 8.39× | 2 548 037 (2 545 026) | 36 368 (34 448) | 8/8 |

**Okunuşu:**
- **`[3]` compute (coproc inference) L boyunca sabit:** 2 401 094 → 2 402 758 (**+0.07%**), **cyc/MAC = 1.12 düz** her L'de → pipelining FMA latency'sini gizliyor (serial olsa 1.12+L olurdu). setup=1644, load=12 183 sabit (load MLP'den büyük çünkü LeNet girişi 784×8 kelime).
- **speedup 8.44×→8.39×** (LeNet büyük olduğu için MLP'nin 5.4×'inden yüksek; hafif düşüş: CPU golden de FMA'yı L'de kullanıyor).
- **Shared (FIR):** coproc inference **+%0.11** (L0) — neredeyse hiç (LeNet GEMM 2.1M MAC, FIR 4096 MAC yanında ihmal); CPU FIR **+%5.4** (L0). **Saf-FMA çekişmesi (MAC) → §2b (coproc +%31, gerçek arbiter ayrışması orada).**
- **Accuracy 8/8 + match_ref 8/8** her config'te → gerçek ağ accelerator'da doğru çalışıyor.

---

## 1b. Gerçek zaman (cycle × 3.845 ns @ 260 MHz, P0)

DC sentezinde çekirdek (`cv32e40px_top`) **3.845 ns'de WNS=0** → gerçek fmax **260 MHz**. Verilator cycle × 3.845 ns = gerçek süre.

| L | CPU inf | CPU FIR | coproc inf | shared coproc | shared CPU FIR |
|:-:|--------:|--------:|-----------:|--------------:|---------------:|
| 0 | 82.084 ms | 130.48 µs | 9.716 ms | 9.727 ms | 137.52 µs |
| 1 | 82.097 ms | 130.48 µs | 9.732 ms | 9.742 ms | 138.02 µs |
| 2 | 82.110 ms | 130.98 µs | 9.746 ms | 9.756 ms | 138.60 µs |
| 3 | 82.122 ms | 131.47 µs | 9.759 ms | 9.771 ms | 138.57 µs |
| 4 | 82.135 ms | 131.96 µs | 9.771 ms | 9.784 ms | 139.55 µs |
| 5 | 82.160 ms | 132.45 µs | 9.786 ms | 9.797 ms | 139.83 µs |

**Resim başına (L0, B=8):** coproc inference **1.21 ms/img**, CPU inference **10.26 ms/img** → coproc **8.44× hızlı** (gerçek zamanda).
**Not:** periyot = **core** sentez fmax'i (260 MHz). Tam-SoC clock'u (DMA/bus/memory dahil) farklı olabilir → onun için full-SoC sentez gerekir. Herhangi bir cycle değerini `× 3.845 ns` ile zamana çevirebilirsin.

---

## 2. Policy karşılaştırması @ L=0 — FIR shared senaryosu

`[1]`=21 348 322, `[2]`=33 936, `[3]` total=2 526 962 **her policy'de aynı** (policy sadece paylaşımı etkiler):

| Config (L0) | [5] shared coproc **total** (alone 2 526 962) | [4] shared CPU FIR **total** (alone 33 936) |
|---|:---:|:---:|
| **P0** CPU-strict | 2 529 649 (+0.11%) | 35 765 (**+5.4%**) |
| **P1** round-robin | 2 529 649 (+0.11%) | 35 765 (+5.4%) |
| **P2** QoS `w1-1-1` (eşit) | 2 529 437 (+0.10%) | 38 356 (+13.0%) |
| **P2** QoS `w4-1-1` (CPU-öncelik) | 2 530 103 (+0.12%) | 36 420 (+7.3%) |
| **P2** QoS `w1-4-1` (acc-öncelik) | **2 527 458 (+0.02%)** | 39 310 (**+15.8%**) |
| **P2** QoS `w1-4-4` (acc-öncelik) | 2 527 458 (+0.02%) | 39 310 (+15.8%) |

**Okunuşu:** FIR memory-bound + LeNet devasa olduğundan **P0 ile P1 aynı** (CPU FIR +5.4% vs +5.4%; coproc +0.11%). Politika farkı ancak QoS'la görünüyor:
- **P0/P1:** CPU FIR +5.4% (coproc +0.11%).
- **P2 acc-öncelik (`w1-4-*`):** coproc'u korur → coproc **+0.02%**, CPU FIR **+15.8%**.
- **P2 CPU-öncelik (`w4-1-1`):** ara denge (FIR +7.3%).

> **P0 vs P1 burada ayrışmıyor** çünkü FIR FMA'yı doldurmuyor. Gerçek arbiter ayrışması için **saf-FMA çekişmesi** → **§2b**.

---

## 2b. Saf-FMA çekişmesi (MAC kernel) — arbiter policy'lerinin AYRIŞTIĞI yer

MAC = register-only, **8 bağımsız akümülatör**lü CPU kernel'i → FMA'yı ~her cycle ister. Coproc'la aynı FMA'yı gerçekten aynı cycle'da çekiştirir (LeNet'te MAC ~786K fmadd, coproc run'ının ~yarısını kaplar) → policy burada belirleyici.

### (a) `[6]` cyc/fmadd — CPU'nun FP-issue tavanı (tek başına, coproc yok)

| L | 0 | 1 | 2 | 3 | 4 | 5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **cyc/fmadd** | **1.51** | **1.51** | 3.26 | 4.14 | 5.13 | 6.01 |

**Bulgu:** L≤1'de düz **~1.51** (issue-bound); L≥2'de **≈L+1** (latency-bound) → **cv32e40p APU ~2 outstanding FP** tutuyor (8 akümülatöre rağmen). MLP ile aynı — bu bir CPU/donanım özelliği.

### (b) `[7]`/`[8]` P0 vs P1 — CPU-MAC ve coproc yavaşlaması

macA = `[6]` alone; co = `[3]` alone. Parantez = shared/alone yüzdesi.

| L | cyc/fmadd | [6] macA | **P0** CPU-MAC | **P1** CPU-MAC | P0 coproc | P1 coproc |
|:-:|:-:|--:|:--:|:--:|:--:|:--:|
| 0 | 1.51 | 1 191 982 | 1 327 106 (**+11.3%**) | 1 956 991 (**+64.2%**) | 3 312 217 (+31.1%) | 3 213 163 (+27.2%) |
| 1 | 1.51 | 1 191 982 | 1 327 116 (**+11.3%**) | 1 957 617 (**+64.2%**) | 3 315 858 (+31.0%) | 3 217 604 (+27.1%) |
| 2 | 3.26 | 2 571 310 | 2 651 398 (+3.1%) | 2 651 398 (+3.1%) | 3 305 519 (+30.4%) | 3 305 519 (+30.4%) |
| 3 | 4.14 | 3 256 366 | 3 335 429 (+2.4%) | 3 335 425 (+2.4%) | 3 166 189 (+24.8%) | 3 166 185 (+24.8%) |
| 4 | 5.13 | 4 039 727 | 4 095 124 (+1.4%) | 4 095 124 (+1.4%) | 2 861 149 (+12.6%) | 2 861 149 (+12.6%) |
| 5 | 6.01 | 4 727 856 | 4 774 323 (+1.0%) | 4 774 340 (+1.0%) | 2 921 375 (+14.8%) | 2 921 394 (+14.8%) |

**Okunuşu — asıl sonuç:**
- **L=0,1 (CPU issue-bound → gerçek çekişme):** **P0** CPU-MAC'i **+11.3%** yavaşlatır (CPU korunur); **P1** CPU-MAC'i **+64.2%** yavaşlatır (round-robin böler). **~53 puanlık fark** — oysa FIR'de P0=P1 (+5.4%). **CPU-priority garantisi burada net görünüyor.** Ödünleşme: P0'da coproc daha çok yavaşlar (+31% vs P1 +27%) çünkü CPU'ya öncelik veriyor.
- **L≥2 (CPU latency-bound):** P0 ≡ P1 (fark 0.0 pp) — cv32e40p FP'yi seyrek issue ediyor, çakışma kalmıyor.

### (c) QoS @ L0 (MAC çekişmesi)

macA=1 191 982, co=2 526 962.

| Config | CPU-MAC | coproc |
|---|:--:|:--:|
| **P0** CPU-strict | 1 327 106 (+11.3%) | 3 312 217 (+31.1%) |
| **P1** round-robin | 1 956 991 (+64.2%) | 3 213 163 (+27.2%) |
| **P2** `w1-1-1` (eşit) | 1 956 928 (+64.2%) | 3 213 085 (+27.2%) |
| **P2** `w4-1-1` (CPU-öncelik) | 1 499 332 (+25.8%) | 3 263 293 (+29.1%) |
| **P2** `w1-4-1` (acc-öncelik) | 2 870 333 (+140.8%) | 2 928 202 (+15.9%) |

**Okunuşu:** `w4-1-1` CPU'yu korur (+25.8%, P1'in +64%'ünden iyi), `w1-4-1` coproc'u korur (CPU +141%, coproc +16%). P1 ≈ P2-`w1-1-1`. Saf-FMA çekişmesinde politikalar tam spektrum — FIR'in gösteremediği.

---

## 3. Özet gözlemler
- **Pipelining çalışıyor (gerçek model):** coproc inference cyc/MAC = **1.12 düz** (L=0..5), compute ~2.40M sabit.
- **CPU'ya karşı ~8.4× hızlı**, her L'de (MLP'nin 5.4×'inden yüksek — büyük model overhead'i daha iyi amortize ediyor).
- **Paylaşım neredeyse bedava (FIR):** coproc +%0.11 (LeNet GEMM devasa, FIR ihmal); CPU FIR +%5.4 (P0).
- **Saf-FMA çekişmesi (MAC, §2b) — asıl arbiter sonucu:** L≤1'de (CPU issue-bound) **P0 CPU'yu korur (+11.3%), P1 böler (+64.2%)** — ~53 puan fark, FIR'in gösteremediği (P0=P1=+5.4%). Ödünleşme: P0'da coproc +31% (P1 +27%). L≥2'de cv32e40p FP'yi seyrek issue ettiğinden P0≡P1. QoS: CPU +11%…+141%.
- **cv32e40p FP-issue tavanı:** cyc/fmadd L≤1'de **~1.51**, L≥2'de **≈L+1** → APU ~2 outstanding FP.
- **21/21 bit-exact + 8/8 accuracy** → gerçek LeNet accelerator'da doğru + CPU ile birebir.

---

## Appendix — tüm 21 config (8 total cycle, hepsi acc 8/8)

`[6]` macA (MAC alone) aynı L'de policy'den bağımsız sabit; `[7]`/`[8]` = MAC shared (CPU-MAC / coproc). §2b'nin ham verisi.

| Config | [1] CPU inf | [2] CPU FIR | [3] coproc inf | [5] sh coproc | [4] sh FIR | [6] macA | [7] sh MAC | [8] sh coproc(MAC) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| L0_P0 | 21 348 322 | 33 936 | 2 526 962 | 2 529 649 | 35 765 | 1 191 982 | 1 327 106 | 3 312 217 |
| L0_P1 | 21 348 322 | 33 936 | 2 526 962 | 2 529 649 | 35 765 | 1 191 982 | 1 956 991 | 3 213 163 |
| L0_P2 w1-1-1 | 21 348 322 | 33 936 | 2 526 962 | 2 529 437 | 38 356 | 1 191 982 | 1 956 928 | 3 213 085 |
| L0_P2 w4-1-1 | 21 348 322 | 33 936 | 2 526 962 | 2 530 103 | 36 420 | 1 191 982 | 1 499 332 | 3 263 293 |
| L0_P2 w1-4-1 | 21 348 322 | 33 936 | 2 526 962 | 2 527 458 | 39 310 | 1 191 982 | 2 870 333 | 2 928 202 |
| L0_P2 w1-4-4 | 21 348 322 | 33 936 | 2 526 962 | 2 527 458 | 39 310 | 1 191 982 | 2 870 333 | 2 928 202 |
| L1_P0 | 21 351 602 | 33 936 | 2 531 017 | 2 533 659 | 35 897 | 1 191 982 | 1 327 116 | 3 315 858 |
| L1_P1 | 21 351 602 | 33 936 | 2 531 017 | 2 533 659 | 35 897 | 1 191 982 | 1 957 617 | 3 217 604 |
| L1_P2 w4-1-1 | 21 351 602 | 33 936 | 2 531 017 | 2 534 006 | 36 425 | 1 191 982 | 1 499 321 | 3 267 135 |
| L2_P0 | 21 354 882 | 34 064 | 2 534 707 | 2 537 282 | 36 048 | 2 571 310 | 2 651 398 | 3 305 519 |
| L2_P1 | 21 354 882 | 34 064 | 2 534 707 | 2 537 282 | 36 048 | 2 571 310 | 2 651 398 | 3 305 519 |
| L2_P2 w4-1-1 | 21 354 882 | 34 064 | 2 534 707 | 2 537 674 | 36 215 | 2 571 310 | 2 843 664 | 3 265 428 |
| L3_P0 | 21 358 162 | 34 192 | 2 538 017 | 2 541 114 | 36 038 | 3 256 366 | 3 335 429 | 3 166 189 |
| L3_P1 | 21 358 162 | 34 192 | 2 538 017 | 2 541 114 | 36 038 | 3 256 366 | 3 335 425 | 3 166 185 |
| L3_P2 w4-1-1 | 21 358 162 | 34 192 | 2 538 017 | 2 541 616 | 36 313 | 3 256 366 | 3 480 851 | 3 168 945 |
| L4_P0 | 21 361 442 | 34 320 | 2 541 318 | 2 544 501 | 36 293 | 4 039 727 | 4 095 124 | 2 861 149 |
| L4_P1 | 21 361 442 | 34 320 | 2 541 318 | 2 544 501 | 36 293 | 4 039 727 | 4 095 124 | 2 861 149 |
| L4_P2 w4-1-1 | 21 361 442 | 34 320 | 2 541 318 | 2 544 924 | 36 666 | 4 039 727 | 4 196 478 | 2 958 908 |
| L5_P0 | 21 368 002 | 34 448 | 2 545 026 | 2 548 037 | 36 368 | 4 727 856 | 4 774 323 | 2 921 375 |
| L5_P1 | 21 368 002 | 34 448 | 2 545 026 | 2 548 037 | 36 368 | 4 727 856 | 4 774 340 | 2 921 394 |
| L5_P2 w4-1-1 | 21 368 002 | 34 448 | 2 545 026 | 2 548 519 | 36 766 | 4 727 856 | 4 854 952 | 2 882 066 |

*Kaynak: `sweep_results/lenet_pipe/` (21 config, hepsi `ok` + `bit-exact=1` + `8/8`). Coproc inference: setup=1644, load=12 183 sabit; compute L ile ~%0.07 değişir. `[6]` cyc/fmadd: L=0,1→1.51; L=2→3.26; L=3→4.14; L=4→5.13; L=5→6.01.*

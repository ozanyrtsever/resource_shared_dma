# perf_lenet_pipe — sweep sonuçları (gerçek LeNet-300-100 / MNIST, pipelined single coprocessor)

**Workload:** LeNet-300-100 `784-300-100-10`, FP32, **B=8** (266 200 MAC/img, 2 129 600 MAC/batch). Gerçek eğitilmiş ağırlıklar + gerçek MNIST görüntüleri.
**Donanım:** pipelined tek coprocessor (`dma_fp_dot_accel_pipe`), CPU'nun tek FMA'sını paylaşır (ikinci FPU yok), APU arbiter. Verilator, trace off.
**Sweep:** L = 0..5 × arbiter policy P = 0/1/2 (+ L=0'da QoS ağırlık varyantları). Klasör: `sweep_results/lenet_pipe/`. **21/21 config `ok`, hepsi `bit-exact=1` ve `correct=8/8, match_ref=8/8`.**

**CPU baseline = optimize batched inference** (`fc_cpu`: ağırlık batch boyunca tekrar kullanılır + 8 bağımsız akümülatör; disassembly-verified). Naive değil → speedup savunulabilir. (L=0'da 4.01 cyc/MAC.)

**8 metrik** (setup/load [DMA yoksa 0] / compute / total, cycle):
`[1]` CPU alone inference · `[2]` CPU alone FIR · `[3]` Coproc alone inference · `[4]` Shared CPU FIR · `[5]` Shared Coproc inference · `[6]` CPU alone **MAC** (saf-FMA, `cyc/fmadd` basar) · `[7]` Shared CPU MAC · `[8]` Shared Coproc inference (MAC).
FIR = CPU'nun kendi 32-tap DSP işi (memory-bound). **MAC = saf-FMA CPU kernel'i** (register-only, 8 bağımsız akümülatör) — arbiter'ı strese sokar. FIR ve MAC **ayrı run'larda** coproc ile paylaşır (aynı anda değil).

---

## 1. Ana tablo — L-sweep @ P0 (CPU-strict)

`[1]`,`[2]`,`[3]` policy'den bağımsız; sadece shared `[4]`/`[5]` policy'ye duyarlı (§2).

| L | [1] CPU inf | [2] CPU FIR | [3] Coproc inf: setup/load/compute/**total** | cyc/MAC | speedup | [5] shared coproc **total** (alone) | [4] shared CPU FIR **total** (alone) | acc |
|:-:|:-----------:|:-----------:|:--------------------------------------------:|:-------:|:-------:|:-----------------------------------:|:------------------------------------:|:---:|
| 0 | 8 530 871 | 33 809 | 1650 / 12 183 / 2 400 708 / **2 536 456** | 1.12 | 3.36× | 2 540 019 (2 536 456) | 36 657 (33 809) | 8/8 |
| 1 | 8 530 871 | 33 809 | 1650 / 12 183 / 2 401 485 / **2 540 513** | 1.12 | 3.35× | 2 543 880 (2 540 513) | 36 786 (33 809) | 8/8 |
| 2 | 12 530 011 | 33 937 | 1650 / 12 183 / 2 402 283 / **2 544 591** | 1.12 | 4.92× | 2 547 327 (2 544 591) | 36 943 (33 937) | 8/8 |
| 3 | 14 130 481 | 34 065 | 1650 / 12 183 / 2 402 699 / **2 548 287** | 1.12 | 5.54× | 2 551 169 (2 548 287) | 36 916 (34 065) | 8/8 |
| 4 | 15 997 151 | 34 193 | 1650 / 12 183 / 2 402 718 / **2 551 586** | 1.12 | 6.26× | 2 554 905 (2 551 586) | 36 979 (34 193) | 8/8 |
| 5 | 17 863 821 | 34 321 | 1650 / 12 183 / 2 402 760 / **2 554 908** | 1.12 | 6.99× | 2 558 696 (2 554 908) | 37 216 (34 321) | 8/8 |

**Okunuşu:**
- **`[3]` coproc compute L boyunca sabit:** ~2.40M, **cyc/MAC = 1.12 düz** her L'de → pipelining FMA latency'sini gizliyor. setup=1650, load=12 183 sabit (load MLP'den büyük çünkü LeNet girişi 784×8 kelime).
- **speedup L ile ARTIYOR: 3.36× → 6.99×.** Sebep: **optimize CPU inference L ile yavaşlıyor** (CPU cyc/MAC: L0,1'de **4.01**, sonra 5.88 / 6.64 / 7.51 / **8.39**), çünkü **cv32e40p ~2 outstanding FP** tutabiliyor → yüksek L'de FMA pipeline'ını dolduramıyor. Coproc ise **düz 1.12** (pipeline'ı dolduruyor). Fark = coproc'un **pipeline-doldurma avantajı**, L (derinlik) ile büyüyor.
- **L=0 coproc'un en zorlandığı nokta** (speedup en düşük, 3.36×); pipelined FMA'da (L≥2) net öne geçiyor. (LeNet MLP'den yüksek çünkü büyük model DMA overhead'ini daha iyi amortize ediyor.)
- **Shared (FIR):** coproc inference **+%0.1** (L0) — neredeyse hiç (LeNet GEMM 2.1M MAC, FIR 4096 MAC yanında ihmal); CPU FIR **+%8.4** (L0). **Saf-FMA çekişmesi (MAC) → §2b.**
- **Accuracy 8/8 + match_ref 8/8** her config'te → gerçek ağ accelerator'da doğru çalışıyor.

---

## 1b. Gerçek zaman (cycle × 3.845 ns @ 260 MHz, P0)

DC sentezinde çekirdek (`cv32e40px_top`) **3.845 ns'de WNS=0** → gerçek fmax **260 MHz**. Verilator cycle × 3.845 ns = gerçek süre.

| L | CPU inf | CPU FIR | coproc inf | shared coproc | shared CPU FIR |
|:-:|--------:|--------:|-----------:|--------------:|---------------:|
| 0 | 32.801 ms | 130.00 µs | 9.753 ms | 9.766 ms | 140.95 µs |
| 1 | 32.801 ms | 130.00 µs | 9.768 ms | 9.781 ms | 141.44 µs |
| 2 | 48.178 ms | 130.49 µs | 9.784 ms | 9.794 ms | 142.05 µs |
| 3 | 54.332 ms | 130.98 µs | 9.798 ms | 9.809 ms | 141.94 µs |
| 4 | 61.509 ms | 131.47 µs | 9.811 ms | 9.824 ms | 142.18 µs |
| 5 | 68.686 ms | 131.96 µs | 9.824 ms | 9.838 ms | 143.10 µs |

**Resim başına (L0, B=8):** coproc inference **1.22 ms/img**, CPU inference **4.10 ms/img** → coproc **3.36× hızlı** (gerçek zamanda). (L=5'te fark ~7×'e çıkar — CPU latency-bound.)
**Not:** periyot = **core** sentez fmax'i (260 MHz). Tam-SoC clock'u (DMA/bus/memory dahil) farklı olabilir → onun için full-SoC sentez gerekir. Herhangi bir cycle değerini `× 3.845 ns` ile zamana çevirebilirsin.

---

## 2. Policy karşılaştırması @ L=0 — FIR shared senaryosu

`[1]`=8 530 871, `[2]`=33 809, `[3]` total=2 536 456 **her policy'de aynı** (policy sadece paylaşımı etkiler):

| Config (L0) | [5] shared coproc **total** (alone 2 536 456) | [4] shared CPU FIR **total** (alone 33 809) |
|---|:---:|:---:|
| **P0** CPU-strict | 2 540 019 (+0.1%) | 36 657 (**+8.4%**) |
| **P1** round-robin | 2 540 019 (+0.1%) | 36 657 (+8.4%) |
| **P2** QoS `w1-1-1` (eşit) | 2 538 610 (+0.1%) | 40 476 (+19.7%) |
| **P2** QoS `w4-1-1` (CPU-öncelik) | 2 540 238 (+0.1%) | 37 783 (+11.8%) |
| **P2** QoS `w1-4-1` (acc-öncelik) | **2 538 842 (+0.1%)** | 47 164 (**+39.5%**) |
| **P2** QoS `w1-4-4` (acc-öncelik) | 2 538 842 (+0.1%) | 47 164 (+39.5%) |

**Okunuşu:** FIR memory-bound + LeNet devasa olduğundan **P0 ile P1 aynı** (CPU FIR +8.4% vs +8.4%; coproc +0.1%). Politika farkı ancak QoS ile görünüyor:
- **P0/P1:** CPU FIR +8.4% (coproc +0.1%).
- **P2 acc-öncelik (`w1-4-*`):** coproc'u korur → coproc **+0.1%**, CPU FIR **+39.5%**.
- **P2 CPU-öncelik (`w4-1-1`):** ara denge (FIR +11.8%).

> **P0 vs P1 burada ayrışmıyor** çünkü FIR FMA'yı doldurmuyor. Gerçek arbiter ayrışması için **saf-FMA çekişmesi** → **§2b**.

---

## 2b. Saf-FMA çekişmesi (MAC kernel) — arbiter policy'lerinin AYRIŞTIĞI yer

MAC = register-only, **8 bağımsız akümülatör**lü CPU kernel'i → FMA'yı ~her cycle ister. Coproc'la aynı FMA'yı gerçekten aynı cycle'da çekiştirir (LeNet'te MAC ~786K fmadd, coproc run'ının bir kısmını kaplar) → policy burada belirleyici.

### (a) `[6]` cyc/fmadd — CPU'nun FP-issue tavanı (tek başına, coproc yok)

| L | 0 | 1 | 2 | 3 | 4 | 5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **cyc/fmadd** | **1.51** | **1.51** | 3.26 | 4.13 | 5.13 | 6.00 |

**Bulgu:** L≤1'de düz **~1.51** (issue-bound); L≥2'de **≈L+1** (latency-bound) → **cv32e40p APU ~2 outstanding FP** tutuyor (8 akümülatöre rağmen). §1'deki "speedup L ile artıyor"un sebebi de bu — optimize inference aynı tavana çarpıyor (CPU cyc/MAC 4.01→8.39). MLP ile aynı — CPU/donanım özelliği.

### (b) `[7]`/`[8]` P0 vs P1 — CPU-MAC ve coproc yavaşlaması

macA = `[6]` alone; co = `[3]` alone. Parantez = shared/alone yüzdesi.

| L | cyc/fmadd | [6] macA | **P0** CPU-MAC | **P1** CPU-MAC | P0 coproc | P1 coproc |
|:-:|:-:|--:|:--:|:--:|:--:|:--:|
| 0 | 1.51 | 1 188 910 | 1 323 475 (**+11.3%**) | 1 979 258 (**+66.5%**) | 3 323 288 (+31.0%) | 3 240 942 (+27.8%) |
| 1 | 1.51 | 1 188 910 | 1 323 475 (**+11.3%**) | 1 979 388 (**+66.5%**) | 3 326 861 (+31.0%) | 3 244 664 (+27.7%) |
| 2 | 3.26 | 2 568 238 | 2 654 284 (+3.4%) | 2 654 295 (+3.4%) | 3 312 198 (+30.2%) | 3 312 170 (+30.2%) |
| 3 | 4.13 | 3 253 294 | 3 337 017 (+2.6%) | 3 337 008 (+2.6%) | 3 180 754 (+24.8%) | 3 180 751 (+24.8%) |
| 4 | 5.13 | 4 036 655 | 4 094 374 (+1.4%) | 4 094 374 (+1.4%) | 2 867 867 (+12.4%) | 2 867 867 (+12.4%) |
| 5 | 6.00 | 4 724 784 | 4 773 422 (+1.0%) | 4 773 507 (+1.0%) | 2 931 679 (+14.7%) | 2 931 749 (+14.7%) |

**Okunuşu — asıl arbiter sonucu:**
- **L=0,1 (CPU issue-bound → gerçek çekişme):** **P0** CPU-MAC'i **+11.3%** yavaşlatır (CPU korunur); **P1** CPU-MAC'i **+66.5%** yavaşlatır (round-robin böler). **~55 puanlık fark** — oysa FIR'de P0=P1 (+8.4%). **CPU-priority garantisi burada net.** Ödünleşme: P0'da coproc daha çok yavaşlar (+31% vs P1 +27.8%) çünkü CPU'ya öncelik veriyor.
- **L≥2 (CPU latency-bound):** P0 ≡ P1 (fark 0.0 pp) — cv32e40p FP'yi seyrek issue ediyor, çakışma kalmıyor.

### (c) QoS @ L0 (MAC çekişmesi)

macA=1 188 910, co=2 536 456.

| Config | CPU-MAC | coproc |
|---|:--:|:--:|
| **P0** CPU-strict | 1 323 475 (+11.3%) | 3 323 288 (+31.0%) |
| **P1** round-robin | 1 979 258 (+66.5%) | 3 240 942 (+27.8%) |
| **P2** `w1-1-1` (eşit) | 1 963 388 (+65.1%) | 3 222 955 (+27.1%) |
| **P2** `w4-1-1` (CPU-öncelik) | 1 519 402 (+27.8%) | 3 282 551 (+29.4%) |
| **P2** `w1-4-1` (acc-öncelik) | 2 874 650 (+141.8%) | 2 938 244 (+15.8%) |

**Okunuşu:** `w4-1-1` CPU'yu korur (+27.8%, P1'in +66.5%'inden iyi), `w1-4-1` coproc'u korur (CPU +141.8%, coproc +15.8%). P1 ≈ P2-`w1-1-1`. Saf-FMA çekişmesinde politikalar tam spektrum — FIR'in gösteremediği.

---

## 3. Özet gözlemler
- **Pipelining çalışıyor (gerçek model):** coproc inference cyc/MAC = **1.12 düz** (L=0..5), compute ~2.40M sabit.
- **CPU'ya karşı 3.36× (L0) → 6.99× (L5), L ile ARTAN** — optimize CPU L ile yavaşlıyor (~2 outstanding: cyc/MAC 4.01→8.39), coproc düz. **Fark = pipeline-doldurma avantajı, derinlikle büyüyor.** (MLP'nin 2.47→5.22×'inden yüksek — büyük model overhead'i daha iyi amortize ediyor.)
- **Baseline optimize + disassembly-verified** → speedup savunulabilir; en düşük (L=0, 3.36×) coproc'un en zorlandığı nokta.
- **Paylaşım neredeyse bedava (FIR):** coproc +%0.1 (LeNet GEMM devasa, FIR ihmal); CPU FIR +%8.4 (P0).
- **Saf-FMA çekişmesi (MAC, §2b):** L≤1'de **P0 CPU'yu korur (+11.3%), P1 böler (+66.5%)** — ~55 puan fark. L≥2'de P0≡P1. QoS: CPU +11%…+142%.
- **21/21 bit-exact + 8/8 accuracy** → gerçek LeNet accelerator'da doğru + CPU ile birebir.

---

## Appendix — tüm 21 config (8 total cycle, hepsi acc 8/8)

`[6]` macA aynı L'de policy'den bağımsız sabit; `[7]`/`[8]` = MAC shared (CPU-MAC / coproc). §2b'nin ham verisi.

| Config | [1] CPU inf | [2] CPU FIR | [3] coproc inf | [5] sh coproc | [4] sh FIR | [6] macA | [7] sh MAC | [8] sh coproc(MAC) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| L0_P0 | 8 530 871 | 33 809 | 2 536 456 | 2 540 019 | 36 657 | 1 188 910 | 1 323 475 | 3 323 288 |
| L0_P1 | 8 530 871 | 33 809 | 2 536 456 | 2 540 019 | 36 657 | 1 188 910 | 1 979 258 | 3 240 942 |
| L0_P2 w1-1-1 | 8 530 871 | 33 809 | 2 536 456 | 2 538 610 | 40 476 | 1 188 910 | 1 963 388 | 3 222 955 |
| L0_P2 w4-1-1 | 8 530 871 | 33 809 | 2 536 456 | 2 540 238 | 37 783 | 1 188 910 | 1 519 402 | 3 282 551 |
| L0_P2 w1-4-1 | 8 530 871 | 33 809 | 2 536 456 | 2 538 842 | 47 164 | 1 188 910 | 2 874 650 | 2 938 244 |
| L0_P2 w1-4-4 | 8 530 871 | 33 809 | 2 536 456 | 2 538 842 | 47 164 | 1 188 910 | 2 874 650 | 2 938 244 |
| L1_P0 | 8 530 871 | 33 809 | 2 540 513 | 2 543 880 | 36 786 | 1 188 910 | 1 323 475 | 3 326 861 |
| L1_P1 | 8 530 871 | 33 809 | 2 540 513 | 2 543 880 | 36 786 | 1 188 910 | 1 979 388 | 3 244 664 |
| L1_P2 w4-1-1 | 8 530 871 | 33 809 | 2 540 513 | 2 544 043 | 37 946 | 1 188 910 | 1 519 854 | 3 286 586 |
| L2_P0 | 12 530 011 | 33 937 | 2 544 591 | 2 547 327 | 36 943 | 2 568 238 | 2 654 284 | 3 312 198 |
| L2_P1 | 12 530 011 | 33 937 | 2 544 591 | 2 547 327 | 36 943 | 2 568 238 | 2 654 295 | 3 312 170 |
| L2_P2 w4-1-1 | 12 530 011 | 33 937 | 2 544 591 | 2 547 555 | 37 983 | 2 568 238 | 2 843 571 | 3 279 053 |
| L3_P0 | 14 130 481 | 34 065 | 2 548 287 | 2 551 169 | 36 916 | 3 253 294 | 3 337 017 | 3 180 754 |
| L3_P1 | 14 130 481 | 34 065 | 2 548 287 | 2 551 169 | 36 916 | 3 253 294 | 3 337 008 | 3 180 751 |
| L3_P2 w4-1-1 | 14 130 481 | 34 065 | 2 548 287 | 2 551 289 | 38 080 | 3 253 294 | 3 485 392 | 3 189 724 |
| L4_P0 | 15 997 151 | 34 193 | 2 551 586 | 2 554 905 | 36 979 | 4 036 655 | 4 094 374 | 2 867 867 |
| L4_P1 | 15 997 151 | 34 193 | 2 551 586 | 2 554 905 | 36 979 | 4 036 655 | 4 094 374 | 2 867 867 |
| L4_P2 w4-1-1 | 15 997 151 | 34 193 | 2 551 586 | 2 554 967 | 38 283 | 4 036 655 | 4 195 731 | 2 968 887 |
| L5_P0 | 17 863 821 | 34 321 | 2 554 908 | 2 558 696 | 37 216 | 4 724 784 | 4 773 422 | 2 931 679 |
| L5_P1 | 17 863 821 | 34 321 | 2 554 908 | 2 558 696 | 37 216 | 4 724 784 | 4 773 507 | 2 931 749 |
| L5_P2 w4-1-1 | 17 863 821 | 34 321 | 2 554 908 | 2 558 838 | 38 392 | 4 724 784 | 4 853 722 | 2 893 101 |

*Kaynak: `sweep_results/lenet_pipe/` (21 config, hepsi `ok` + `bit-exact=1` + `8/8`). CPU baseline = optimize `fc_cpu`. Coproc: setup=1650, load=12 183 sabit; compute L ile ~%0.09 değişir. CPU inf L ile artıyor (4.01→8.39 cyc/MAC, ~2 outstanding). `[6]` cyc/fmadd: L=0,1→1.51; L=2→3.26; L=3→4.13; L=4→5.13; L=5→6.00.*

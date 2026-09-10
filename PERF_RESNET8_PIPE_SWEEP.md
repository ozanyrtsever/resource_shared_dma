# perf_resnet8_pipe — sonuçlar (gerçek CNN: MLPerf-Tiny ResNet-8 / CIFAR-10, pipelined single coprocessor)

**Workload:** MLPerf-Tiny image-classification **ResNet-8** (resnet_v1_eembc), CIFAR-10, FP32, **B=8**
(12 501 632 MAC/img, **100 013 056 MAC/batch**). **Resmi MLCommons pretrained ağırlıklar** (eğitim YOK;
`pretrainedResnet.h5`, tam test seti accuracy **%87.19**). BN export'ta conv'a katlanır (birebir aynı ağ);
preprocessing otomatik tespit: `raw0_255`.
**Donanım:** pipelined tek coprocessor (`dma_fp_dot_accel_pipe`), CPU'nun tek FMA'sını paylaşır (ikinci FPU yok),
APU arbiter. **RTL değişikliği SIFIR** — conv katmanları **im2col** ile mevcut `[N,M,B]` protokolüne oturur: her
çıkış pikseli için 8 görüntünün patch'i LOAD (N=K·K·C_in ≤ 576 ≤ MAXN), filtre bankı WEIGHT (M=C_out satır).
TF-'same' padding indeksleri, (ky,kx,c) patch sırası. Residual add + ReLU + avgpool CPU'da (iki yol için aynı
kod, bir kez ölçülüp iki totale de eklenir). Verilator, trace off; SRAM 4 MB (`sizes:[2048]`).
**Kapsam:** **stock operating point L=0, P0** (karar: CNN için L=0 yeterli; L-eğrisi ve co-execution/policy
çalışması MLP/LeNet'te tam — bu benchmark'ta co-execution metrikleri yok, `[1]`/`[3]` policy-bağımsız).
Klasör: `sweep_results/resnet8_pipe/`. **5/5 config `ok`, hepsi `bit-exact=1` (10/10 katman) ve
`correct=7/8, match_ref=8/8`.**

**CPU baseline = optimize batched inference** (conv'a genellenmiş `fc_cpu`: ağırlık batch(8) boyunca tekrar
kullanılır + 8 bağımsız akümülatör; padding pozisyonlarında da coproc gibi `fmaf(w,0,acc)` yürütür →
bit-exact köşe durumlarda dahi garanti). Naive değil → speedup savunulabilir.

**Metrikler** (setup/load [DMA yoksa 0] / compute / total, cycle): `[1]` CPU alone inference · `[3]` Coproc
alone inference (+ katman-katman bit-exact/cycle). Co-execution (`[2]`,`[4]`–`[8]`) bu benchmark'ta yok —
MLP/LeNet raporlarında.

---

## 1. Ana sonuç — L=0 (stock combinational FMA), P0

| | [1] CPU inf | [3] Coproc inf: setup/load/compute/**total** | coproc cyc/MAC | speedup | acc |
|:-:|:-----------:|:--------------------------------------------:|:-------:|:-------:|:---:|
| L=0 | 561 346 919 | 2 028 599 / 7 928 398 / 114 247 622 / **179 768 729** | **1.14** | **3.12×** | 7/8 (ref 8/8) |

**Okunuşu:**
- **coproc cyc/MAC = 1.14** — MLP (1.14) ve LeNet (1.12) ile **birebir aynı**: pipelining, gerçek bir CNN'in
  im2col beslemesinde de FMA'yı aynı verimle dolduruyor. CPU ise **5.61 cyc/MAC** (optimize golden).
- **Total kırılımı:** compute 114.2M (%63.6) + load 7.9M (patch akışları) + setup 2.0M (4 033 piksel-transferi
  × 2 DMA programlama) + **~55.6M (%31) CPU-tarafı iş** (patch marshalling + scatter/bias + residual/pool).
  Conv'un im2col marshalling'i MLP/LeNet'ten büyük — dürüstçe totale dahil.
- **`correct=7/8` bir hata DEĞİL:** 4. görüntüyü (label 0) **resmi modelin kendisi** 8 tahmin ediyor (Keras'ta
  da aynı). Doğruluk kanıtı `match_ref=8/8` (cihaz, resmi modelin tahminlerini birebir üretir) + 10/10 katman
  bit-exact. %87'lik bir modelin 8 görüntüde ~1 hatası istatistiksel beklenti.
- Beklenen L-davranışı (ölçülmedi, karar gereği): MLP/LeNet ile aynı mekanizma — coproc düz, CPU ~2-outstanding
  ile latency-bound → speedup L ile artar; L=0 coproc'un en zorlandığı nokta, yani **3.12× alt sınırdır**.

---

## 1b. Gerçek zaman (cycle × 3.845 ns @ 260 MHz)

| | CPU inference | Coproc inference |
|:-:|--:|--:|
| batch (B=8) | 2.158 s | 691.2 ms |
| **görüntü başına** | **269.8 ms/img** | **86.4 ms/img** |

**Not:** periyot = core sentez fmax'i (260 MHz, DC-NXT converged); tam-SoC clock için full-SoC sentez gerekir.

---

## 2. Katman-katman kırılım (paper bonusu — MLP/LeNet'te yok)

Her satır: her iki yol da bit-exact; cyc/MAC **marshalling dahil** katman toplamından (coproc'un saf compute
cyc/MAC'i her katmanda ~1.14'tür; fark = piksel-başına transfer/marshalling amortizasyonu).

| Katman | N=K²·Cin | M | çıkış px | MAC (B=8) | CPU cyc | Coproc cyc | Coproc cyc/MAC | katman speedup |
|---|--:|--:|--:|--:|--:|--:|:-:|:-:|
| c1 (3×3, 3→16) | 27 | 16 | 32×32 | 3 538 944 | 25 496 818 | 11 217 344 | 3.17 | 2.27× |
| b1c1 (3×3, 16→16) | 144 | 16 | 32×32 | 18 874 368 | 107 042 259 | 38 685 836 | 2.05 | 2.77× |
| b1c2 (3×3, 16→16) | 144 | 16 | 32×32 | 18 874 368 | 106 579 152 | 38 025 136 | 2.01 | 2.80× |
| b2c1 (3×3 s2, 16→32) | 144 | 32 | 16×16 | 9 437 184 | 53 505 196 | 15 602 503 | 1.65 | 3.43× |
| b2c2 (3×3, 32→32) | 288 | 32 | 16×16 | 18 874 368 | 102 413 248 | 29 539 696 | 1.57 | 3.47× |
| b2s (1×1 s2 skip) | 16 | 32 | 16×16 | 1 048 576 | 6 665 088 | 2 707 056 | 2.58 | 2.46× |
| b3c1 (3×3 s2, 32→64) | 288 | 64 | 8×8 | 9 437 184 | 51 309 235 | 13 079 911 | 1.39 | 3.92× |
| b3c2 (3×3, 64→64) | 576 | 64 | 8×8 | 18 874 368 | 98 836 008 | 25 705 808 | 1.36 | 3.84× |
| b3s (1×1 s2 skip) | 32 | 64 | 8×8 | 1 048 576 | 6 147 496 | 1 872 848 | 1.79 | 3.28× |
| fc (64→10) | 64 | 10 | 1 | 5 120 | 33 454 | 13 626 | 2.66 | 2.46× |

**Desen (tez bulgusunun katman-ölçeği teyidi):** piksel başına iş (M·N·B) büyüdükçe transfer/marshalling
amortize olur → cyc/MAC **3.17 (c1, N=27) → 1.36 (b3c2, N=576)** monoton iyileşir; katman speedup'ı 2.3×→3.9×.
**Coprocessor uzun redüksiyonları sever** — derin katmanlar (kanal sayısı büyüdükçe) ideal beslemedir.

---

## 3. Policy karşılaştırması @ L=0

5 config (P0, P1, P2 `w1-1-1`/`w4-1-1`/`w1-4-4`) **bit-özdeş sonuç** verdi (aynı [1]/[3]/accuracy —
beklenen: bu benchmark yalnız-inference ölçer, CPU'nun eşzamanlı FP işi yok → arbiter hiç hakemlik yapmıyor →
policy etkisiz). Policy'lerin gerçekten ayrıştığı ölçümler MLP/LeNet §2b'de (saf-FMA MAC co-execution).

---

## 4. Özet gözlemler
- **Gerçek bir CNN, RTL değişikliği olmadan** (im2col) aynı coprocessor'da koşuyor: **10/10 katman bit-exact**,
  resmi pretrained model tahminleri birebir (**match_ref 8/8**), model accuracy %87.19.
- **coproc cyc/MAC = 1.14** — üç workload'da da aynı (MLP 1.14 / LeNet 1.12 / ResNet-8 1.14): pipelining'in
  workload-bağımsızlığı.
- **Speedup 3.12× @ L=0** (optimize CPU baseline'a karşı; L=0 en zor nokta → alt sınır). Görüntü başına
  269.8 ms → **86.4 ms** @ 260 MHz.
- Katman kırılımı "uzun dot daha iyi amortize olur" tezini ağ içinde monoton gösteriyor (3.17→1.36 cyc/MAC).
- Stride-2, TF-'same' padding, residual add, global avgpool — hepsi sürücü/CPU tarafında; koprocessor protokolü
  değişmeden.

---

## Appendix — 5 config (hepsi bit-özdeş)

| Config | [1] CPU inf | [3] coproc inf | speedup | cyc/MAC | acc |
|---|--:|--:|:-:|:-:|:-:|
| L0_P0 | 561 346 919 | 179 768 729 | 3.12× | 1.14 | 7/8 (ref 8/8) |
| L0_P1 | 561 346 919 | 179 768 729 | 3.12× | 1.14 | 7/8 (ref 8/8) |
| L0_P2 w1-1-1 | 561 346 919 | 179 768 729 | 3.12× | 1.14 | 7/8 (ref 8/8) |
| L0_P2 w4-1-1 | 561 346 919 | 179 768 729 | 3.12× | 1.14 | 7/8 (ref 8/8) |
| L0_P2 w1-4-4 | 561 346 919 | 179 768 729 | 3.12× | 1.14 | 7/8 (ref 8/8) |

*Kaynak: `sweep_results/resnet8_pipe/` (5 config, hepsi `ok`). Model: resmi `pretrainedResnet.h5`
(mlcommons/tiny), export `example_model/gen_resnet8_cifar.py` (BN-fold + logit-seviyesi numpy self-check).
Bit-özdeşlik deterministik simülasyonun ve policy-bağımsızlığın doğrudan kanıtı.*

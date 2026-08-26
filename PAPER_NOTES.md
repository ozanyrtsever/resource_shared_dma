# Teknik dosya — paper yazımı için (shared-FMA pipelined coprocessor, CV32E40P/X-HEEP)

*Bu bir paper taslağı DEĞİL. Paper'ı SEN yazarken çekeceğin ham teknik malzeme: ne yapıldı, ne bitti,
nasıl çalışıyor, sistem ne, tüm teknik detaylar + sayılar + nereden geldikleri. Tek design: pipelined
single coprocessor (dual ve GEMV terk edildi). Ham perf tabloları ayrı raporlarda (bkz. §9).*

---

## 1. Tek cümlelik sistem
CV32E40P çekirdeğinin **tek, var olan FPnew FMA'sını**, CPU ile bir **DMA-beslemeli reduction
coprocessor** arasında, APU arayüzü üzerinden, **CPU-öncelikli bir arbiter** ile zaman-paylaşımlı
kullandırıyoruz — **ikinci FPU eklemeden**. Coprocessor **pipelined**: bir batched matris çarpımının B
bağımsız MAC'ini arka arkaya issue edip FMA'yı dolu tutuyor → per-MAC maliyeti **FMA latency'sinden
bağımsız** → tek birim FMA'yı doyuruyor (ikinci coprocessor'a gerek yok).

---

## 2. Sistem bileşenleri (ne var)
- **Core:** CV32E40P (4-stage in-order RISC-V), FP'yi **APU** portundan konuşur.
- **FPU:** FPnew instance. ADDMUL lane'i = pipelined **FMA** (`op0*op1 + op2`). Pipe derinliği =
  `FPU_ADDMUL_LAT` parametresi (= bizim latency `L`, 0..5). out_ready=1 → **1 MAC/cycle** kabul eder (tam
  pipelined). `PipeConfig=AFTER`, FP32.
- **SoC:** X-HEEP (core-v-mini-mcu). DMA'da **hardware-FIFO modu** var: bir peripheral kendini FIFO gibi
  gösterir, DMA push/pop yapar, full/empty backpressure verir.
- **Coprocessor:** `dma_fp_dot_accel_pipe.sv` (tb/). DMA hw_fifo kanalına bağlı. **Kendi çarpıcısı YOK** —
  her MAC'i APU'dan paylaşılan FMA'ya `fmadd` olarak yollar.
- **Arbiter:** `dma_apu_arbiter.sv`. `cv32e40px_top` içinde, core'un APU'su ile fp_wrapper arasına giriyor.
- **Bağlantı:** `testharness.sv.tpl` — coprocessor DMA ch1 hw_fifo'ya + arbiter'ın `apu_ext` portuna;
  `COPROC_PIPE` knob'u pipelined vs serial modülü seçiyor. `COPROC_FPU_SHARE` define'ı arbiter'ı açıyor.

---

## 3. Nasıl çalışıyor — teknik detaylar

### 3.1 DMA hw_fifo protokolü (bir inference katmanı)
İki ardışık DMA transferi:
1. **LOAD** = header `[N, M, B]` + B giriş vektörü. Coprocessor bunları **bir kez** input buffer'a yazar
   ("input-stationary"). N=dot uzunluğu, M=çıkış satırı (nöron), B=batch.
2. **WEIGHT** = tüm `M×N` ağırlık matrisi **tek akışta** streamlenir. Her streamlenen ağırlık, B girişin
   hepsiyle çarpılır → **weight başına B MAC** (weight-reuse, weight-stationary olmadan).
- İki transfer arasındaki **FIFO flush**, coprocessor'ı "loaded" state'inden compute'a çevirir.
- Sonuçlar (M×B) aynı kanaldan pop edilir.
- **LeNet özel:** layer-1'de M*N=300*784=235200 > DMA'nın 16-bit SIZE limiti (65535) → ağırlıklar **tek 2D
  transfer** ile akıtılır (size_d1=N iç, size_d2=M dış, inc 1,1). MLP'de 1D yeter.

### 3.2 Arbiter (paylaşım mekanizması) — iç yapı
- **3 requestor:** CPU + 2 coprocessor portu (`apu_ext`, `apu_ext1`) — tek-coprocessor kullanımında acc0
  aktif, acc1 idle.
- **2-bit owner tag:** her op'la birlikte FMA'ya gider (0=CPU,1=acc0,2=acc1), sonuçla geri döner →
  `cpu_rvalid/dma_rvalid` tag ile ayrıştırılır. **Out-of-order safe.**
- **Drain YOK:** her cycle kombinasyonel `grant`; `inflight_q` sadece sayaç. Sahip değiştirmek için
  pipeline durdurmaz. FMA'ya op basmaya devam edebilir → **pipelined multi-issue destekli** (bu, pipelined
  coprocessor için kritik — arbiter'a dokunmadan çalıştı).
- **Policy (elaboration-time parametre, sadece seçilen sentezlenir):**
  - **P0** CPU-strict + acc-RR (default): CPU her zaman kazanır, coprocessor CPU'nun bıraktığı slot'ları
    alır. **Garanti: CPU'nun kendi FP işi aç kalmaz.**
  - **P1** full round-robin (CPU önceliğini kaldırır → maliyetini ölçmek için).
  - **P2** QoS-weighted (deficit-credit; programlanabilir CPU:acc0:acc1 ağırlıkları).
- **Ölçülebilir policy farkı ancak saf-FMA çekişmesinde:** FIR gibi memory-bound bir CPU işi FMA'yı aynı
  cycle'da nadiren istediğinden **P0≈P1**; P0'ı P1'den ayıran gerçek çekişme için saf-FMA MAC kernel'i (§6.2b)
  gerekli. cv32e40p yalnızca ~2 outstanding FP tuttuğundan bu ayrışma **L≤1'de** görülür (§6.2).
- **Değişmez (invariant):** arbitrasyon sadece **sahipler arası** sıra değiştirir; bir sahibin kendi op
  akışı bozulmaz → bit-exact her policy'de korunur.
- **Insertion:** `cv32e40px_top`'ta `` `ifdef COPROC_FPU_SHARE `` altında; base'de core.APU → fp_wrapper
  DOĞRUDAN, share'de core.APU → arbiter → fp_wrapper (`w_apu_*` telleri).

### 3.3 Pipelined coprocessor — iç yapı
Hesaplar: `Y[M][B] = W[M][N] · X[N][B]` (B=1 → matris-vektör). **Decoupled issue/collect:**
- **iss_b** (issue pointer, 0..B-1): granted her cycle bir MAC teklif eder — operandlar `{acc_q[iss_b],
  w_q, x[iss_b][i]}`, FMA'ya `fmadd`.
- **col_b** (collect pointer): dönen her sonucu lane'ine yazar (`acc_q[col_b] <= rdata`). Sonuçlar issue
  sırasında retire eder (FPnew ADDMUL in-order pipe) → **basit round-robin sayaç yeter**, blok içinde tag'e
  gerek yok.
- **inflight** sayacı: issue-retire; aynı anda **L+1'e kadar** MAC uçuşta → FMA dolu.
- **Hazard interlock** `inflight < B`: lane iss_b'yi issue etmeden önce önceki (i-1) yazımının bitmiş
  olmasını garanti eder. Sadece `B < L+1` (batch latency'yi örtemeyecek kadar küçük) olunca stall eder;
  `B ≥ L+1` iken hiç ateşlemez. B=8, L≤5 → hiç stall yok.
- **State machine:** LHDR_N/M/B (header) → LDAT (girişleri buffer'a) → RECV (bir ağırlık al) → ISSUE (o
  ağırlığın B lane'ini pipelined issue et) → DRAIN (inflight=0 olana kadar bekle) → OUT (M×B sonucu yay).
- **Buffer:** `xbuf_ram` submodülü (1R1W); silikonda SRAM makrosu (DC'de black-box → logic-only alan).

### 3.4 Latency-gizleme matematiği (ANA katkı)
- **Serial** coprocessor: bir MAC at, sonucu `1+L` cycle bekle, sonra sıradaki → `cyc/MAC = 1.14 + L`,
  FMA util `1/(1+L)`. Latency **tamamen açık**.
- **Pipelined:** bir ağırlığın B lane'i bağımsız (farklı acc_q[b]) → arka arkaya issue. Lane b, B issue
  sonra tekrar gelir; o ana kadar b'nin önceki sonucu retire olmuştur (`B ≥ L+1`). → **`cyc/MAC` L'den
  BAĞIMSIZ** (düz ~1.12-1.14). Latency **gizlenir**.
- **Sonuç:** tek pipelined birim FMA'yı ~%88 doyurur → **ikinci coprocessor gereksiz.** (Serial'de idle'ı
  ikinci birim dolduruyordu; pipelined idle'ı kendisi dolduruyor.)
- **Bit-exact:** interleave lane'ler *arası*; her dot-product'ın toplama sırası (k=0..N-1) DEĞİŞMEZ →
  serial ile ve CPU ile bit-bit aynı. Runtime `B=1` → interlock serial'e indirger (aynı RTL, aynı kanıt).

---

## 4. Design evolüsyonu / kararlar (ne denendi, ne kaldı/atıldı)
- **GEMV design (B=1) → ATILDI.** Memory-bound, 1 MAC/word. Ölçüldü ama arbiter policy'si anlamlı fark
  vermedi (FMA doymuyor). Terk edildi; sadece batched GEMM.
- **Serial batched GEMM + dual coprocessor → ATILDI.** Serial cyc/MAC=1.14+L; ikinci coprocessor idle'ı
  doldurmak için vardı (değeri L ile artıyordu, L=0'da anlamsız). Pipelined tek birim bunu geçince dual
  emekliye ayrıldı.
- **Pipelined batched GEMM single coprocessor → NİHAİ DESIGN.** cyc/MAC düz, tek birim yeter, ikinci FPU
  yok. Arbiter'a dokunulmadı (zaten tag ile out-of-order safe multi-issue destekliyordu).
- Serial modül (`dma_fp_dot_accel_is.sv`) baseline olarak duruyor (`COPROC_PIPE` knob'u ile seçilir) —
  serial-vs-pipelined kıyası için, ama tez/paper **pipelined**'i kullanıyor.

---

## 5. Ne yapıldı / doğrulandı
- **Fonksiyonel (Verilator):** her config CPU golden ile **bit-exact**; LeNet ayrıca MNIST etiketleri +
  offline referans ile (**8/8 doğru**). Cycle'lar `mcycle`'dan; **8 metrik**: (1) CPU-alone inference, (2)
  CPU-alone FIR, (3) coproc-alone inference, (4) shared CPU-FIR, (5) shared coproc inference, (6) CPU-alone
  **MAC** (saf-FMA), (7) shared CPU-MAC, (8) shared coproc (MAC) — her biri setup/load/compute/total. FIR =
  memory-bound DSP işi (~8 cyc/fmadd); **MAC = register-only, 8 bağımsız akümülatörlü saf-FMA kernel** (CPU
  her cycle FMA ister → arbiter'ı strese sokar). FIR ve MAC **ayrı** co-execution run'larında (aynı anda değil).
- **Sweep'ler:** FMA latency `L=0..5` × arbiter policy `P=0/1/2` (her benchmark 21 config), **hepsi
  bit-exact** → (i) `cyc/MAC` L'den bağımsız mı (pipelining kanıtı), (ii) policy'nin co-execution'a etkisi.
  Benchmark'lar: `perf_bench_pipe` (toy MLP), `perf_lenet_pipe` (gerçek LeNet).
- **Sentez (DC-NXT, TSMC 40 nm G):** `cv32e40px_top` (arbiter içinde) hiyerarşi korunarak
  (`-no_autoungroup -no_boundary_optimization`) → FMA/FPU/arbiter/CPU alanları **tek run'dan ayrıştırıldı**;
  coprocessor ayrı (buffer black-box → logic-only). **Operating clock convergence** (`converge.sh`,
  `CLK += |WNS|/2`): en yavaş config'de timing tam kapanana kadar (WNS→0) → **3.845 ns = 260 MHz**. Alan o
  noktada raporlandı (0.1 ns'de şişkin olurdu). 1 GE = 0.9576 µm².
- **Gerçek zaman:** 3.845 ns'de timing kapandığı için, Verilator cycle × 3.845 ns = wall-clock (core'un
  260 MHz'inde; tam-SoC clock'u için full-SoC sentez lazım).
- **Co-execution:** CPU kendi FP FIR'ini coprocessor inference ederken **aynı FMA'da** koşturur — maliyet
  ölçüldü.

---

## 6. Sonuçlar — teknik özet (ham tablolar §9'daki raporlarda)

### 6.1 Performans
- **`cyc/MAC` L'den bağımsız:** ≈**1.14** (MLP) / **1.12** (LeNet), L=0..5 düz (serial olsa 1.14+L olurdu).
- **CPU'ya hızlanma:** **5.49×** (MLP) / **8.44×** (LeNet), L boyunca ~düz.
- **FMA util** (paylaşılan birim) ≈ %87-88 her L'de.
- **Gerçek zaman @ 260 MHz:** MLP inference **61.3 µs/img** (CPU 337 µs); LeNet **1.21 ms/img** (CPU 10.26
  ms). **8/8 MNIST doğru**, bit-exact.

### 6.2 Co-execution (CPU FP işi ∥ coproc inference, tek FMA)

**(a) FIR (memory-bound CPU işi).** CPU-FIR'i coproc inference ile eşzamanlı: coproc'a **+%0.11 (LeNet) /
+%2.8 (MLP)**, CPU-FIR'e **+%5.4 (LeNet) / +%2.3 (MLP)** (P0). İki FP işi tek FMA'yı ~bedavaya paylaşıyor.
Ama FIR FMA'yı doldurmadığından **P0 ile P1 burada AYNI** → arbiter'ı ayırt ettirmiyor.

**(b) MAC (saf-FMA CPU işi) — arbiter'ın asıl testi.** Register-only, 8 bağımsız akümülatör → CPU her cycle
FMA ister, coproc'la gerçekten aynı cycle'da çekişir. İki ana bulgu:
- **cv32e40p FP-issue tavanı:** MAC-alone `cyc/fmadd` L=0,1'de düz **~1.51** (issue-bound), L≥2'de **≈L+1**
  (latency-bound) → **çekirdek ~2 outstanding FP op** tutuyor (8 akümülatöre rağmen; bir **cv32e40p donanım
  limiti**). Yani CPU FMA'yı ancak **L≤1'de** doyurabiliyor.
- **Policy'ler AYRIŞIYOR (L≤1, gerçek çekişme):** **P0** CPU-MAC'i **+6.4% (MLP) / +11.3% (LeNet)** yavaşlatır
  (CPU korunur); **P1** **+63.6% / +64.2%** yavaşlatır (round-robin FMA'yı böler) → **~53–57 puanlık fark**,
  FIR'de ~0 olan. **CPU-priority garantisinin "dişleri" ilk kez ölçülüyor.** Ödünleşme: P0'da coproc daha çok
  yavaşlar (LeNet +31% vs P1 +27%). **L≥2'de** CPU latency-bound olduğundan çekişme kalmıyor → **P0≡P1**.
- **QoS spektrumu (L0):** `w4-1-1` CPU'yu korur (CPU +22–26%), `w1-4-1` coproc'u korur (CPU +137–141%,
  coproc +13–16%). Ağırlıklarla sürekli ayar.
- **Çıkarım (paper için güçlü):** arbiter policy'si tam da **CPU FMA'yı doyurabildiğinde** (cv32e40p'de L≤1)
  belirleyici; doyuramadığında zaten çekişme yok. FIR'deki "P0≈P1" bir arbiter kusuru değil, workload özelliği.

### 6.3 Alan — "ikinci FPU yok" (260 MHz, L=0, P0)
- Bileşen: CPU core **52.9k µm²** + FPU **33.2k** (FMA **21.5k**) + arbiter **1.8k** = **88.0k µm² (91.9k
  GE)**. Coprocessor logic **4.9k µm² (5.1k GE)** (buffer ayrı SRAM makrosu).
- **Paylaşım = arbiter, FMA değil:** arbiter = bir FMA'nın **%8.4'ü**, core'un **%2.0'si**. Accel başına
  shared (accel+arbiter=6.7k) vs dedicated (accel+FMA=26.5k) → **3.9× daha az eklenen alan.**
- **Policy alanı:** P1 (1.5k) < P0 (1.8k) < P2/QoS (2.3k); en büyüğü bile ≈%10 of FMA.

### 6.4 Frekans
- Paylaşımlı core **260 MHz**'de timing kapatıyor (3.845 ns converged; 250 MHz'de marjla da).
- **Coprocessor darboğaz değil:** kendi kritik yolu core clock'undan rahat hızlı. **L=0'da core'un binding
  path'i CPU load-store pipeline'ında (FMA değil)** → paylaşım logic'i kritik yol olmuyor.
- (FMA-özel kritik yol `report_timing -through` ile `study.tcl`'e eklendi; 0.1 ns re-run bekliyor.)

---

## 7. Parametreler & konfigürasyon
| Param | Değer | Nerede |
|---|---|---|
| MAXN (max dot uzunluğu) | 1024 | accel modül param |
| MAXB (max batch) | 8 | accel modül param; benchmark B=8 |
| FPU_ADDMUL_LAT (L) | 0..5 sweep | configs/cv32e40px_fpu_dma.hjson (fpu_addmul_lat) |
| ARB_POLICY_SEL (P) | 0/1/2 | cv32e40px_top ( `ifndef` guard, sweep sed'ler) |
| ARB_W_CPU:ACC0:ACC1 | 4:1:1 (default) | cv32e40px_top; P2 QoS ağırlıkları |
| COPROC_PIPE / COPROC_SERIAL | pipe (default) | testharness knob |
| COPROC_FPU_SHARE | on | core-v-mini-mcu.core (arbiter'ı açar) |
| Operating clock | 3.845 ns / 260 MHz | converge.sh sonucu |
| Kütüphane | TSMC 40nm G, sc12mc_cln40g_base_rvt, SS/0.81V/125C | 1 GE=0.9576 µm² |

---

## 8. Dosya / akış haritası
**RTL:**
- `tb/dma_fp_dot_accel_pipe.sv` — pipelined coprocessor (nihai).
- `tb/dma_fp_dot_accel_is.sv` — serial baseline (kıyas için, COPROC_PIPE ile seçilir).
- `tb/xbuf_ram.sv` — input buffer submodülü (SRAM makrosu).
- `hw/vendor/xheep/cv32e40px/rtl/dma_apu_arbiter.sv` — arbiter.
- `hw/vendor/xheep/cv32e40px/rtl/cv32e40px_top.sv` — arbiter insertion + apu_ext portları.
- `tb/testharness.sv.tpl` — coprocessor + arbiter bağlantısı, COPROC_PIPE knob.

**Yazılım (benchmark):** `sw/applications/perf_bench_pipe/main.c` (toy MLP), `perf_lenet_pipe/main.c`
(gerçek LeNet + accuracy). 5 metrik + RAPOR bloğu.

**Akışlar:**
- `sweep.sh` — Verilator L×policy sweep (PROJECT/OUTDIR env, resumable, 21 config).
- `dc_scripts/converge.sh` — operating clock convergence (CLK+=|WNS|/2) + tam matris.
- `dc_scripts/study.sh` + `study.tcl` — tek-clock DC alan/fmax (hiyerarşi korunmuş, dc_reports/'a yazar).

### 3.5 ⚠️ Ölçüm scope'u (paper'da MUTLAKA belirt)
- **Alan/fmax = `cv32e40px_top` (çekirdek) sentezi**, tam SoC değil. Yani DMA/bus/memory dahil değil.
  Çekirdek 260 MHz'de kapatıyor; tam-SoC clock'u farklı olabilir → gerçek zaman "260 MHz core varsayımıyla".
- **Alan hiyerarşiden ayrıştırıldı** (per-module standalone değil, gerçek-bağlam). accel ayrı sentezlendi.
- **base (arbiter'sız core) yok** → fmax "paylaşımlı core'un mutlak hızı" (arbiter'ın clock delta-maliyeti
  değil). İstersen base L=0 eklenip delta çıkarılabilir.

---

## 9. İlgili güncel raporlar (ham tablolar)
- `PERF_BENCH_PIPE_SWEEP.md` — toy-MLP perf (5 metrik × L × policy + gerçek zaman).
- `PERF_LENET_PIPE_SWEEP.md` — gerçek LeNet perf (+ accuracy + gerçek zaman).
- `DC_STUDY_OPERATING.md` — DC alan & fmax @ 260 MHz (+250 MHz), convergence, bileşen breakdown.

## 10. Açık / opsiyonel işler
- **FMA-özel kritik yol:** `study.tcl`'e `report_timing -through [FMA]` eklendi; 0.1 ns re-run ile FMA
  fmax'i L=0..5 çıkarılacak.
- **Full-SoC sentez:** mutlak çip clock'u + tam-SoC gerçek zaman için (ağır; ayrı iş).
- **Enerji/MAC:** DC power ile (shared vs dedicated enerji karşılaştırması — en güçlü "no 2nd FPU"
  argümanı).
- **base-vs-share fmax:** arbiter'ın clock maliyetini delta olarak çıkarmak için (2 küçük run).
- **Related-work sitasyonları:** Tanase 2026 (Computers 15(4) 219), NTX, Snitch — doğrulanacak (bkz.
  repo `thesis_progress.md §10`, `COMPARISON_vs_tanase2026.md`).

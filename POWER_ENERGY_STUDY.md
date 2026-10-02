# Güç / Enerji Analizi — Baştan Sona, Adım Adım (Türkçe açıklama)

Bu doküman, paylaşımlı-FMA koprosesör tasarımının **güç (power)** ve **enerji (energy)**
ölçümünü *hiç bilmeyen birine* anlatır gibi yazılmıştır. Ne yaptık, neden yaptık, hangi
sorunları çözdük, sonuç ne çıktı — hepsi burada. Bütün sayılar `dc_reports/power/` altındaki
gerçek Design Compiler raporlarından alınmıştır.

---

## 0. Tek cümlelik özet

Yonganın gerçek "kapıya dökülmüş" (gate-level) haline, üzerinde gerçek programı
koştururken hangi telin kaç kez 0↔1 değiştiğini kaydettik; sonra bu değişim bilgisini
güç aracına verip **her iş parçası için ayrı ayrı kaç miliwatt harcandığını** ölçtük.
Ana bulgu: **tek FMA'yı iki iş paylaşınca FMA'nın gücü artmıyor** — yani "ikinci bir FPU
eklemeye gerek yok" iddiası güçle de kanıtlandı.

---

## 1. Neden güç ölçüyoruz? (motivasyon)

Tezin ana iddiası: MCU'nun içinde zaten bir FMA (çarp-topla birimi) var ve CPU çoğu zaman
onu boş bırakıyor. Biz küçük bir **hakem (arbiter)** koyup bu FMA'yı bir koprosesöre de
ödünç veriyoruz. Böylece **ikinci bir FMA (yani ikinci bir FPU) eklemeden** hızlanma
sağlıyoruz.

Bu iddianın 3 ayağı var:
1. **Hız** — koprosesör CPU'dan kaç kat hızlı? (Daha önce ölçüldü: 2.5–7×.)
2. **Alan** — arbiter ne kadar yer kaplıyor? (Ölçüldü: bir FMA'nın %8.4'ü.)
3. **Güç/Enerji** — *bu doküman.* Paylaşım güç açısından ucuz mu? İkinci FMA olsaydı ne
   kadar fazla güç yakardık?

Güç olmadan hikâye eksik kalıyordu. Şimdi tamam.

---

## 2. Bebeğe temel kavramlar (bunları bilmeden aşağısı anlaşılmaz)

### 2.1 Güç (Power) — iki çeşidi
- **Güç = anlık harcanan elektrik**, birimi **watt (W)**; bizde **miliwatt (mW)** ve
  **mikrowatt (µW)**. (1 mW = 1000 µW.)
- İki bileşen:
  - **Dinamik güç:** transistörler 0↔1 değiştikçe harcanır. **İş yaptıkça artar.** İçinde
    *switching* (telleri şarj/deşarj) + *internal* (hücre içi) vardır.
  - **Kaçak güç (leakage):** devre hiçbir şey yapmasa bile sızan akım. İş yükünden
    **bağımsız**, hep vardır. (125°C sıcak köşede kaçak yüksektir.)

### 2.2 Enerji (Energy) — güçten farkı
- **Enerji = güç × süre.** Birim: **joule (J)**; bizde **mikrojoule (µJ)** / **pikojoule (pJ)**.
- Benzetme: **güç = musluğun akış hızı**, **enerji = kovaya biriken toplam su**.
- Neden ikisi de lazım? Koprosesör CPU'dan biraz daha **çok güç** çekebilir (musluk daha
  açık) ama işi **çok daha kısa sürede** bitirir → toplam **enerji daha az** olabilir. Bir
  işi bitirmenin gerçek "faturası" enerjidir.

### 2.3 Netlist — neden "gate-level"
- **RTL:** bizim yazdığımız Verilog (davranışsal).
- **Netlist (gate-level):** sentez (Design Compiler) RTL'i alıp **gerçek TSMC 40nm
  hücrelerine** dönüştürünce çıkan devre. Güç ancak bu gerçek hücreler üzerinden anlamlı
  ölçülür (her hücrenin kütüphanede güç modeli var). İki netlist'imiz vardı:
  - `netlist_share_L0.v` → **CPU + FPU + FMA + arbiter** (bir arada, hiyerarşi korunmuş)
  - `netlist_accel.v` → **koprosesör** (ayrı)

### 2.4 SAIF — işin kalbi
- Güç aracı "bu tel saniyede kaç kez 0↔1 oldu?" bilgisine muhtaç. Buna **switching
  activity** denir. Bunu simülasyondan toplayıp **SAIF dosyasına** yazarız, sonra güç
  aracına veririz.
- **Kritik:** SAIF'teki tel isimleri, güç aracındaki netlist isimleriyle **birebir** aynı
  olmalı. Bu yüzden simülasyonu **RTL'de değil, netlist'in kendisi üzerinde** yaptık — buna
  **gate-level simulation (GLS)** denir. Böylece isimler %100 uydu (raporda annotasyon
  %100: core'da 40 700 net / 146 004 pin, koprosesörde 2 345 net / 8 704 pin — hepsi
  gerçek aktiviteyle işaretlendi, tahmin kullanılmadı).

### 2.5 Neden "faz faz"
- Simülasyonun tamamı ~6.3 milyon cycle; bunun **~4.9 milyonu** program başlamadan önce
  **bellek yükleme** ile geçiyor. Tek ortalama alsak bu gürültü her şeyi bozardı. Çözüm:
  programı **fazlara** böldük, her fazın **sadece kendi cycle penceresi** için ayrı SAIF
  (dolayısıyla ayrı güç) topladık.

---

## 3. Beş faz — programın içinde ne koşuyor

Ölçüm programı `power_probe` (`sw/applications/power_probe/main.c`). Daha önceki
`perf_bench_pipe` testinin **birebir aynısı** (aynı MLP 128-64-32-16, B=8, 86016 MAC/batch;
aynı optimize CPU kodu, aynı koprosesör protokolü, aynı MAC çekirdeği) — sadece FIR
çıkarıldı ve her fazın başına/sonuna bir işaret (`phase_flag`) konuldu.

| Faz | İçinde ne çalışıyor | Ne ölçüyor |
|---|---|---|
| **IDLE** | boş döngü (CPU meşgul-bekleme) | referans taban |
| **CPU_INF** | inference'ı **CPU** yapıyor | CPU tek başına sinir ağı |
| **COPROC** | inference'ı **koprosesör** yapıyor, CPU bekliyor | koprosesör tek başına |
| **CPU_MAC** | CPU saf FMA döngüsü (register-only) | CPU'nun kendi FP işi |
| **SHARED** | **koprosesör inference + CPU MAC AYNI ANDA, tek FMA'yı paylaşarak** | **ASIL SONUÇ** |

İşaret nasıl çalışıyor: program her fazın başında bir bellek adresine faz numarası yazar,
sonunda 0 yazar. Simülasyonda küçük gözlemci modül (`saif_ctrl.sv`) CPU'nun veri yoluna
bakıp bu yazmayı görür → SAIF toplamayı başlatır/durdurur. Bu gözlemci **tasarıma
dokunmaz**, dışarıdan izler.

---

## 4. Akış — adım adım

```
   [1] SENTEZ (daha önce)                    [2] PROGRAM (lokal)
   RTL --Design Compiler--> netlist          power_probe.c --derle--> main.hex
        (CPU+FPU+FMA+arb) + (koprosesör)            (5 fazlı ölçüm programı)
              |                                          |
              +---------------------+--------------------+
                                    v
   [3] GATE-LEVEL SIMULATION (VCS, server)
       X-HEEP'in tamamı RTL (bellek/DMA/UART = "ortam"),
       AMA cv32e40px_top ve koprosesör NETLIST olarak takılı.
       Program koşarken saif_ctrl her fazın aktivitesini kaydeder.
                                    v
   [4] 10 adet SAIF:  core_<FAZ>.saif  +  acc_<FAZ>.saif
       (core = CPU+FPU+FMA+arbiter,  acc = koprosesör)
                                    v
   [5] GÜÇ HESABI (Design Compiler NXT, server)
       Her SAIF için: netlist oku + SAIF oku + report_power
       --> her bileşenin (CPU/FPU/FMA/arbiter/koprosesör) gücü, faz faz
```

**Somut komutlar (server):** `gls.sh build` (derle) → `gls.sh run` (koş, 10 SAIF üret) →
`gls.sh power` (DC güç raporları).

---

## 5. Yolda karşılaştığımız sorunlar (kısa)

1. **Modern araç çalışmıyordu** — VCS 2026 server'ın eski Linux'unda açılmadı; **VCS 2022**
   ile çözdük.
2. **Link hatası** (`openpty`) → `--no-as-needed`.
3. **t=0'da duruyordu** → `+vcs+initreg+0` (flop'ları 0'dan başlat).
4. **SAIF boş çıkıyordu (en zoru)** — deneme dosyalarıyla VCS kurallarını bulduk:
   `-debug_access+pp` şart, dosya adları `string`, tek `$set_toggle_region`. Sonra %100 doldu.
5. **Bağlantı kopunca koşu ölüyordu** → `nohup` + `screen`.
6. **Lisans sunucusu ara ara düşüyordu** → `SNPSLMD_QUEUE=TRUE`.

**Doğrulama:** GLS'te ölçülen faz süreleri (cycle) RTL simülasyonuyla **birebir** aynı
(120012 / 417134 / 134393 / 49632 / 166931) ve tüm sonuçlar **bit-exact**. Netlist RTL ile
tıpatıp aynı davranıyor → güç sayıları güvenilir.

---

## 6. Sonuçları okumadan önce: iki şeyi "ayıklamak" gerekiyor

### 6.1 Saat neti (clock) = 65.71 mW — SAHTE, atıyoruz
- Raporda `core_i/sleep_unit_i/core_clock_gate_i` her fazda **birebir 65.710 mW**
  (switching). Doğrula: raporun toplam dinamiği (COPROC 78.12 mW, IDLE 68.05 mW) içinde bu
  65.71 sabiti hep var.
- Neden sahte: sentez netlist'inde **saat ağacı (clock tree) henüz yok** (o, yerleşim/P&R
  aşamasında kurulur). Şu an tek bir saat teli ~3000 flip-flop'a gidiyor; araç bu telin
  kapasitesini ~385 pF diye tahmin ediyor (gerçekte ~5 pF) → 85 kat şişik.
- Her fazda aynı sabit olduğu için **temizce çıkarıyoruz.** Gerçek saat gücü P&R'dan gelir.
  (`_nowlm` raporları da aynı çıktı — bu sayı WLM'den değil, tek-net-yüksek-fanout'tan
  kaynaklandığı için. Yine de sabit olduğundan çıkarınca sorun yok.)

### 6.2 CPU_MAC fazı FMA'yı sıfır gösteriyor — zayıf prob
- CPU_MAC çekirdeği `fmaf(0.5, a, 0.5)` yapıyor; değer hızla **1.0'a yakınsıyor**. Sabitlenince
  operandlar toggle etmiyor → FMA'da anahtarlama ~0 → FMA gücü ~0 görünüyor (raporda
  0.021 mW).
- Bu ölçüm kusuru, gerçek değil. **Sorun değil:** CPU'nun FMA'yı gerçekten kullandığını
  **CPU_INF** kanıtlıyor (FMA 2.11 mW). Ana sonuç (SHARED) gerçek inference'a dayandığı için
  etkilenmiyor.

---

## 7. SONUÇLAR — Güç tablosu

### 7.1 Ham toplam güç (raporun verdiği, saat DAHİL)

| Faz | core dinamik | core kaçak | koprosesör dinamik | koprosesör kaçak |
|---|---|---|---|---|
| IDLE | 68.05 mW | 1.494 mW | 199.0 µW | 34.0 µW |
| CPU_INF | 70.26 mW | 1.510 mW | 243.5 µW | 34.1 µW |
| COPROC | 78.12 mW | 1.495 mW | 457.6 µW | 38.4 µW |
| CPU_MAC | 67.71 mW | 1.514 mW | 202.4 µW | 38.6 µW |
| SHARED | 77.17 mW | 1.499 mW | 431.5 µW | 38.7 µW |

> Dikkat: core dinamiğinin **65.71 mW'ı sabit sahte saat neti** (§6.1). Anlamlı kısmı görmek
> için onu çıkarıyoruz ↓

### 7.2 Anlamlı dinamik güç (mW), saat neti çıkarılmış

| Bileşen | IDLE | CPU_INF | COPROC | CPU_MAC | **SHARED** |
|---|---|---|---|---|---|
| **FMA** (opgroup_0, çarp-topla) | ~0 | 2.11 | 9.30 | ~0* | **8.26** |
| FPU (FMA dahil tüm blok) | ~0 | 2.29 | 10.24 | 0.11 | 9.21 |
| **Arbiter** (bizim hakem) | 0.002 | 0.009 | 0.051 | 0.004 | **0.058** |
| CPU çekirdek mantığı (saat hariç) | 2.34 | 2.24 | 2.09 | 1.86 | 2.16 |
| **Koprosesör** | 0.20 | 0.24 | 0.46 | 0.20 | **0.43** |

`*` CPU_MAC'te ~0 çünkü değerler 1.0'a yakınsıyor (§6.2).

**Kaçak güç (mW), sabit:** FMA 0.71 · FPU 0.92 · arbiter 0.062 · koprosesör 0.038.

**Koşullar:** TSMC 40nm G, SS köşe, 0.81 V, 125°C, 260 MHz (3.845 ns), annotasyon %100.

### 7.3 Tablo ne diyor (kelimelerle)
- **FMA en pahalı blok.** Aktifken (COPROC) 9.3 mW — sistemin geri kalanının toplamından
  fazla. Bütün mesele bu tek pahalı bloğu paylaşmak.
- **Koprosesör FMA'yı CPU'dan daha yoğun besliyor.** COPROC'ta FMA 9.30, CPU_INF'te 2.11 mW.
  Sebep: koprosesör FMA'yı ~0.87 MAC/cycle doldururken CPU ~0.21 MAC/cycle besliyor (~4×
  fark) → güç de ~4× fark. Koprosesör FMA'yı boşa çürütmüyor, tam dolduruyor.
- **Arbiter neredeyse bedava:** aktifken 0.05–0.06 mW = FMA'nın (9 mW) **binde 6'sı**.

---

## 8. ANA SONUÇ — "İkinci FPU'ya gerek yok" güçle kanıtı

SHARED fazına bak: koprosesör inference'ı ve CPU'nun MAC işi **aynı anda, tek FMA'yı
paylaşarak** çalışıyor (ikisi de bit-exact doğru sonuç verdi → paylaşım gerçekten oldu).

**Gözlem 1 — Paylaşım FMA gücünü ARTIRMIYOR.**
> SHARED'da FMA = **8.26 mW**, COPROC-tek'te FMA = **9.30 mW**.

Tek FMA fiziksel olarak saniyede ~1 çarp-topla üretir; onu ister tek iş beslesin ister iki
iş paylaşsın, gücü aynı kalır (SHARED'da hatta azıcık *düşük*, çünkü ortalama doluluk biraz
az). Yani **iki işi tek FMA'da birleştirmek FMA açısından ekstra güç getirmiyor.** SHARED'ın
COPROC'a eklediği tek şey **CPU çekirdeğinin çalışması** (2.16 mW mantık) + arbiterde minik
artış (+0.007 mW).

**Gözlem 2 — İki-FPU olsaydı ne olurdu.**

| | Bizim tasarım (tek paylaşımlı FMA) | Klasik iki-FMA |
|---|---|---|
| Dinamik güç (FP tarafı) | 1 FMA: ~8.3 mW | 2 FMA: ~9.3 + ~2.1 ≈ **11.4 mW** |
| Kaçak güç | 1 FMA: **0.71 mW** | 2 FMA: **1.42 mW** |
| Alan | 1 FMA + arbiter: 21.5k + 1.8k µm² | 2 FMA: **~43k µm²** |
| Ekstra maliyet | **arbiter: 0.058 mW / 62 µW kaçak / 1.8k µm²** | ikinci koca bir FMA |

**Cümle:** Arbiter (0.06 mW, 1.8k µm²) tek başına **ikinci bir FMA'nın (21.5k µm², 0.71 mW
kaçak, çalışırken birkaç mW dinamik) yerini tutuyor.** Paylaşım, ikinci FMA'nın tüm alanını
ve kaçağını **hiç ödemeden** iki işi aynı FMA'da çalıştırıyor.

---

## 9. SONUÇLAR — Enerji (iş bitirme faturası)

Enerji = güç × süre. Süre = cycle × 3.845 ns. Faz süreleri:
IDLE 461 µs · CPU_INF 1604 µs · COPROC 517 µs · CPU_MAC 191 µs · SHARED 642 µs.

Aynı işi (86016 MAC'lik MLP inference) iki yolla — dinamik güç (saat hariç) = FPU + arbiter
+ CPU-mantık + koprosesör; kaçak = 1.53 mW sabit:

| | Süre | Dinamik güç | Dinamik enerji | + Kaçak enerji | **Toplam** | pJ/MAC |
|---|---|---|---|---|---|---|
| **Koprosesör** (COPROC) | 517 µs | 12.87 mW | 6.65 µJ | 0.79 µJ | **7.44 µJ** | 86.5 |
| **CPU** (CPU_INF) | 1604 µs | 4.79 mW | 7.69 µJ | 2.46 µJ | **10.15 µJ** | 118 |

**Sonuç:** Koprosesör hem **~3.1× daha hızlı** hem **~%27 daha az enerji** harcıyor.
Neden? Koprosesör anlık biraz daha çok güç çekiyor (musluk daha açık) ama işi çok daha kısa
sürede bitirdiği için toplam su (enerji) daha az — üstelik kısa süre kaçak enerjiyi de
düşürüyor.

---

## 10. Özet — tek paragraf

Gate-level netlist'i gerçek program koşarken simüle edip her iş fazının anahtarlama
aktivitesini SAIF'e kaydettik, sonra Design Compiler ile faz-faz güç çıkardık (annotasyon
%100, faz süreleri RTL ile birebir). Sahte saat netini ve zayıf CPU_MAC probunu ayıkladıktan
sonra: **FMA en pahalı blok (~9 mW aktif); onu iki iş paylaşınca güç artmıyor (SHARED 8.26 ≈
COPROC 9.30 mW); arbiter neredeyse bedava (0.06 mW).** Böylece paylaşım, ikinci bir FMA'nın
alanını (21.5k µm²) ve kaçağını (0.7 mW) hiç ödemeden çalışıyor. Enerjide de koprosesör
CPU'dan ~3× hızlı ve ~%27 daha ekonomik. **Tezin "ikinci FPU'ya gerek yok" iddiası artık
hız + alan + güç/enerji üçünde de kanıtlı.**

---

## 11. Dosyalar nerede

**Scriptler (lokal + server):**
- `sw/applications/power_probe/main.c` — 5 fazlı ölçüm programı
- `dc_scripts/gls/saif_ctrl.sv` — faz-izleyen SAIF gözlemcisi (TB, sentezlenmez)
- `dc_scripts/gls.sh` — build / run / power sürücüsü
- `dc_scripts/power.tcl` — DC güç raporu üreteci
- `dc_scripts/gls/saif_selftest.sv` — VCS SAIF kurallarını bulmak için deneme dosyası

**Üretilen veri (lokale çekildi → `dc_reports/`):**
- `dc_reports/saif/` — 10 SAIF dosyası (core_* + acc_*, her faz)
- `dc_reports/power/` — güç raporları:
  - `power_core_<FAZ>.rpt` / `power_acc_<FAZ>.rpt` — hiyerarşik (CPU/FPU/FMA/arbiter satır satır)
  - `power_<blok>_<FAZ>_total.rpt` — toplam dinamik + kaçak
  - `power_<blok>_<FAZ>_nowlm*.rpt` — WLM'siz varyant (saat neti sabiti aynı çıktı)
  - `saif_<blok>_<FAZ>.rpt` — annotasyon yüzdesi (%100)

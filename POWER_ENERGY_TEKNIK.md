# Güç / Enerji Analizi — TEKNİK derinlik (kodlar, scriptler, içeride ne oluyor)

Bu doküman [POWER_ENERGY_STUDY.md](POWER_ENERGY_STUDY.md)'nin teknik tamamlayıcısıdır.
Orada "ne/niye" var; burada **hangi dosyayı neden yazdık, içinde satır satır ne yapılıyor,
komutlar/araçlar içeride ne çeviriyor** var.

> **Önce en önemli teknik gerçek:** Güç çalışması için **tasarım RTL'ine (arbiter,
> koprosesör, cv32e40px) TEK SATIR dokunmadık.** Zaten sentezlenmiş iki netlist'i
> (`netlist_share_L0.v`, `netlist_accel.v`) olduğu gibi kullandık. Yeni yazdığımız Verilog
> **tek dosya**: `saif_ctrl.sv` — bu da **sentezlenmeyen** bir test-bench gözlemcisi (yonganın
> içinde yer almaz). Geri kalan yeni şeyler yazılım (`power_probe/main.c`) ve script
> (`gls.sh`, `power.tcl`). Yani ölçüm altyapısı, ölçtüğü şeyi değiştirmedi.

Yazdığımız 4 dosya:

| Dosya | Tür | Görevi |
|---|---|---|
| `sw/applications/power_probe/main.c` | C (yazılım) | 5 fazlı ölçüm programı; her fazın sınırını donanıma "işaret"ler |
| `dc_scripts/gls/saif_ctrl.sv` | Verilog (TB, sentezlenmez) | O işareti yakalar, faz başına SAIF yazar |
| `dc_scripts/gls.sh` | bash | netlist'i simülasyona sokar (hibrit), koşturur, DC güç raporu ürettirir |
| `dc_scripts/power.tcl` | DC Tcl | tek (blok,faz) için `report_power` üretir |

---

## 1. Büyük resim — mekanizma tek şemada

```
 YAZILIM (power_probe)         DONANIM SİMÜLASYONU (VCS)            GÜÇ (DC)
 ─────────────────────         ────────────────────────            ────────
 phase_flag = 3  ───store──►  cv32e40px_top_i.data_addr_o == paddr
 (COPROC başladı)              & data_we_o & data_req_o & data_gnt_i
                                        │  (saif_ctrl.sv görür)
                                        ▼
                               $toggle_start   ← aktivite sayacı AÇIK
      ...koprosesör çalışır, teller 0↔1 olur, VCS her toggle'ı sayar...
 phase_flag = 0  ───store──►            │
 (COPROC bitti)                         ▼
                               $toggle_stop
                               $toggle_report("core_COPROC.saif", ..., cv32e40px_top_i)
                               $toggle_report("acc_COPROC.saif",  ..., dma_fp_dot_accel_i)
                                        │
                                        ▼  (SAIF dosyaları)
                               power.tcl: read_verilog netlist
                                          read_saif core_COPROC.saif
                                          report_power  → mW
```

İki taraf **hangi sinyalde buluşuyor?** `phase_flag` adlı global değişkene yapılan **store**,
CPU çekirdeğinin veri-yolu portlarında görünür: `data_req_o`, `data_gnt_i`, `data_we_o`,
`data_addr_o`, `data_wdata_o`. Bu portlar **RTL'de de gate-level netlist'te de aynı isimde**
kaldığı için gözlemci iki dünyada da çalışır. Buluşma noktası budur.

---

## 2. `power_probe/main.c` — ölçüm programı

Bu, eski `perf_bench_pipe`'ın kopyası; **tek fark faz işaretleri**. Ölçtüğümüz iş yükünün,
daha önce performans ölçtüğümüz iş yüküyle **birebir aynı** olması için bilinçli böyle.

### 2.1 Faz işareti mekanizması (satır 60–67, 163–200)
```c
volatile uint32_t phase_flag;              // satır 60
#define PHASE_START(id) (phase_flag = (id)) // faza gir: adrese id yaz
#define PHASE_END()     (phase_flag = 0)    // fazdan çık: adrese 0 yaz
```
- `volatile` **şart**: derleyici bu yazmayı "gereksiz" sanıp silmesin (kimse okumuyor gibi
  görünüyor ama donanım gözlemliyor). `volatile` → her atama gerçek bir **store
  instruction**'a dönüşür → veri yolunda görünür.
- Faz gövdesi hep şu kalıpta (örn. COPROC, satır 177–181):
  ```c
  PRINTF("PHASE COPROC start\n");          // UART yazımı PENCERE DIŞINDA (gürültü katmasın)
  PHASE_START(PH_COPROC);                   // ← store: phase_flag=3, pencere AÇILIR
  CSR_READ(CSR_REG_MCYCLE,&t0);             // yazılım da süreyi ölçer (doğrulama için)
  infer(BATCH,0);                           // ASIL İŞ: koprosesör inference
  CSR_READ(CSR_REG_MCYCLE,&t1);
  PHASE_END();                              // ← store: phase_flag=0, pencere KAPANIR
  PRINTF("PHASE COPROC end ... dur=%u\n", t1-t0);
  ```
- Neden `PRINTF` pencere dışında? UART'a yazmak on binlerce cycle sürer ve DMA/UART'ı
  çalıştırır — o gürültü faz gücüne karışmasın diye işaret **işten hemen önce/sonra** atılır,
  yazdırma dışarıda kalır.

### 2.2 Beş fazın gövdesi ne çalıştırıyor
- **IDLE** (164): `for(k=0;k<20000;k++) idle_sink=k;` — sürekli store yapan boş döngü. "CPU
  meşgul ama FP yok" tabanı. (`idle_sink` de `volatile`.)
- **CPU_INF** (171): `golden()` → `fc_cpu()` 3 kez. `fc_cpu` (97–109) optimize CPU matris
  çarpımı: 8 bağımsız akümülatör (`a0..a7`) + ağırlık başına tek yükleme → CPU'nun FMA'yı en
  yoğun beslediği hali. Bunlar `fmaf(w, x, acc)` yani FMA komutu.
- **COPROC** (179): `infer(B,0)` → `fc()` 3 kez, `contend=0` (CPU beklerken sadece
  `while(!dma_is_ready)` polling). İşi koprosesör yapıyor.
- **CPU_MAC** (188): `mac_finish()` → saf register-only FMA döngüsü (72–80): 8 akümülatör ×
  32 tur = mac_one başına 256 `fmaf`. `fmaf(0.5,a,0.5)`. *(Bu fazın FMA gücü ~0 çünkü değer
  1.0'a yakınsayıp toggle kesiliyor — STUDY §6.2.)*
- **SHARED** (197–198): `infer(B,2)` + `mac_finish()`. `contend=2` demek: `fc()` içindeki
  bekleme döngüsü artık boş beklemiyor, **CPU MAC işini araya sıkıştırıyor** (aşağıda).

### 2.3 Paylaşımın kalbi: `fc()` içindeki `contend` (119–135)
```c
CSR_READ(...,&t0); dma_launch(&tr);                 // koprosesöre WEIGHT akışını başlat
if(contend==2){ while(!dma_is_ready(CH0)) if(!mac_done) mac_step(MAC_CHUNK); }  // ← SHARED
else          { while(!dma_is_ready(CH0)); }        // ← COPROC (sadece bekle)
```
- `dma_launch` koprosesörü çalıştırır (DMA ağırlıkları FMA'ya akıtır, her cycle bir MAC).
- COPROC'ta CPU boş döner. SHARED'da CPU **aynı bekleme döngüsünde** `mac_step` çağırıp kendi
  FMA işini yapar. Böylece **koprosesör ve CPU aynı anda FMA ister** → arbiter devreye girer,
  tek FMA ikisine sırayla hizmet eder. Güç ölçümünün "asıl senaryosu" bu satır.
- `mac_step(1)` = 1 `mac_one` = 256 fmadd; kaba taneli araya-girme (koprosesörün akışını
  boğmadan CPU'ya iş verir).

### 2.4 Doğruluk kontrolü (116, 182, 201)
`bitexact()` koprosesör çıktısını (`out3`) CPU golden çıktısıyla (`ga3`) **bit bit**
karşılaştırır. SHARED'da hem koprosesör hem CPU MAC doğru sonuç verdiyse → paylaşım gerçekten
oldu, biri diğerini bozmadı. (Sonuç: `ALL bit-exact=1`.)

---

## 3. `saif_ctrl.sv` — SAIF gözlemcisi (yeni yazılan tek "Verilog")

Sentezlenmez, yongaya girmez. VCS'e **ikinci bir top modül** olarak verilir
(`vcs -top tb_top -top saif_ctrl`), simülasyon boyunca yandan izler.

### 3.1 Store'u yakalama (satır 43–48)
```verilog
always @(posedge `CORE.clk_i) begin
  cyc++;
  if (`CORE.data_req_o && `CORE.data_gnt_i && `CORE.data_we_o && `CORE.data_addr_o == paddr) begin
    if (`CORE.data_wdata_o != 0) begin       // faza GİRİŞ (phase_flag = 1..5)
      cur = `CORE.data_wdata_o; ...
```
- `` `CORE `` bir makro: `tb_top.testharness_i.x_heep_system_i...cv32e40px_top_i` — hiyerarşik
  yol. Gözlemci, çekirdeğin veri-yolu portlarına **doğrudan hiyerarşiden** bakıyor (port
  bağlamaya gerek yok, simülatör her düğümü görür).
- Koşul: "geçerli bir yazma işlemi var (`req & gnt & we`) **ve** hedef adres bizim
  `phase_flag` adresimiz". O anda yazılan değer (`data_wdata_o`) = faz numarası.
- `paddr` nereden? Komut satırından: `+phase_addr=<hex>` (gls.sh bunu `main.map`'ten çıkarır,
  §5.4). Böylece adres kod içine gömülü değil, esnek.

### 3.2 SAIF pencereleri (49–63)
```verilog
if (do_saif) begin
  if (reported) $toggle_reset;   // önceki fazın sayımını sıfırla (ilk fazda atla → uyarı vermesin)
  $toggle_start;                 // aktivite saymayı BAŞLAT
end
...
$toggle_stop;                                        // faz bitti: saymayı DURDUR
fcore = $sformatf("%0s/core_%0s.saif", outdir, names[cur]);
$toggle_report(fcore, 1.0e-9, "...cv32e40px_top_i");  // core bölgesini dosyaya yaz
$toggle_report(facc,  1.0e-9, "...dma_fp_dot_accel_i");// coproc bölgesini dosyaya yaz
```
- `$toggle_start/stop` VCS'in yerleşik SAIF görevleri: iki çağrı arasındaki tüm net
  toggle'larını sayar.
- `$toggle_report(dosya, ölçek, scope)`: sayılan aktiviteyi **verilen scope alt-ağacı için**
  SAIF olarak yazar. İki ayrı çağrı → biri `cv32e40px_top` (CPU+FPU+FMA+arbiter), biri
  koprosesör → **iki ayrı dosya**, çünkü ikisi ayrı netlist ve DC'de ayrı raporlanacak.
- `initial` bloğunda (36–40) `$set_toggle_region(tb_top.testharness_i)`: hangi ağacın
  izleneceğini söyler. İkisini de kapsayan üst düğüm seçildi (VCS tek bölge kabul ediyor).

### 3.3 VCS'in bize çıkardığı 3 zorluk (kod yorumlarında da yazılı)
Bunları küçük deneme dosyasıyla (`saif_selftest.sv`) tek tek bulduk:
1. **`$set_gate_level_monitoring` çağrılmıyor** (37): "off" hiçbir şey kaydetmedi, "on"
   çağrısızla aynıydı → gereksiz. Netlist netleri (Verilog `wire`) varsayılan kaydediliyor.
   **Ama derlemede `-debug_access+pp` şart** (yoksa SAIF sessizce boş çıkıyor).
2. **Dosya adı SV `string` olmalı** (19–22): `{...}` birleştirme ifadesi veya NUL-dolgulu
   reg-vektör "Invalid first argument" veriyor. Çözüm: `$sformatf` ile `string`, derlemede
   `-tf_sv_string`.
3. **Tek `$set_toggle_region`**: ikinci çağrı yok sayılıyor → tek geniş bölge + raporda scope
   ile ayırma.

### 3.4 Yan doğrulama (48, 62)
Gözlemci her fazın **cycle sayısını** da basıyor (`dur=%0d cycles`). Bu, yazılımın `mcycle`
ile ölçtüğü `dur=` ile **birebir** tutmalı — tuttu (120012/417134/134393/49632/166931). Yani
donanım penceresi ile yazılım penceresi aynı yerde → SAIF doğru cycle'ları kapsıyor.

---

## 4. Hibrit netlist takası — `gls.sh build` (en teknik kısım)

Amaç: **X-HEEP'in tamamı RTL koşsun** (bellek, DMA, bus, UART = programı çalıştıracak ortam),
**ama sadece `cv32e40px_top` ve koprosesör NETLIST olsun** (gücünü ölçeceğimiz bloklar).
Buna "hibrit GLS" denir. Zorluğu: bir modülü RTL'den çıkarıp yerine netlist koymak.

### 4.1 İsim uyuşmazlığı sorunu ve wrapper (satır 42–47, 80–81)
- Sentez (Design Compiler), parametreli modüle **mangle** isim verir:
  `cv32e40px_top` → `cv32e40px_top_FPU1_FPU_ADDMUL_LAT0`. Ama X-HEEP'in üst RTL'i hâlâ
  `cv32e40px_top` diye örnekliyor. İsimler tutmazsa bağlanmaz.
- Çözüm: **ince bir wrapper** üretiyoruz. `gen_wrap` (44) RTL dosyasından modül başlığını
  (port listesi dahil `module ... );`) `awk` ile kesip alıyor, sonuna şunu ekliyor:
  ```verilog
  cv32e40px_top_FPU1_FPU_ADDMUL_LAT0 u_gl (.*);   // netlist top'unu .* ile bağla
  endmodule
  ```
  Yani dış dünya `cv32e40px_top` (RTL isim, RTL portlar) görüyor; içeride `.*` (isimden
  otomatik bağlama) ile mangle-isimli netlist top'una geçiyor. `u_gl` = "gate-level unit";
  DC power'da instance yolu bu yüzden `.../cv32e40px_top_i/u_gl` (gls.sh satır 36).

### 4.2 Dosya listesini yeniden yazma (84–104)
Başlangıç noktası: fusesoc'un ürettiği `.scr` (tüm-RTL dosya listesi, ~669 satır). Bunu 3
adımda GLS listesine çeviriyoruz:
- **(a) netlist'lenen RTL'i çıkar** (84–88, `awk`): `cv32e40px/rtl/` ve `fpnew/src/`
  altındaki RTL satırlarını at (bunlar artık netlist içinde). **Paketleri (`_pkg.sv`) tut**
  (tip tanımları hâlâ lazım). Koprosesör RTL'ini (`dma_fp_dot_accel_pipe.sv`) at ama
  `xbuf_ram.sv`'yi **tut** (sentezde black-box'tı, RTL'i simülasyonda gerçek çalışacak).
- **(b) çift-tanımı temizle** (90–104): DC bazı **parametresiz alt modülleri** (opene906
  bölme birimi `pa_fdsu_*`) netlist'e **aynı isimle** yazar. Aynı isim hem netlist'te hem
  kalan RTL'de olursa VCS "çift tanım" hatası verir. Script netlist'teki tüm modül isimlerini
  çıkarıp (90–91), kalan RTL dosyalarında bu isimlerden birini tanımlayan dosyaları da atıyor
  (`dropped_dup.txt`'e loglar; 10 dosya çıktı).
- **(c) ekle** (105–106): hücre modeli (`sc12mc..._udp.v` + `sc12mc...c40.v`) + iki netlist +
  iki wrapper + `saif_ctrl.sv`.

### 4.3 VCS derleme bayrakları (52–55, 111–112)
```
+define+ARM_UD_MODEL   → hücre kütüphanesi Verilog modeli bu makro ile sarılı; olmazsa hücreler boş
+notimingcheck +nospecify → sıfır-gecikmeli FONKSİYONEL GLS (specify blokları/timing check kapalı → hız)
+vcs+initreg+random    → (build'de) X-init; (run'da +initreg+0 ile ezilir, §5.3)
-tf_sv_string          → $toggle_report'a SV string geçebilmek
-debug_access+pp       → SAIF toggle kaydı için ŞART
+define+COPROC_FPU_SHARE→ arbiter+koprosesör yolunu aç (tasarım define'ı)
-CFLAGS ... -Wl,--no-as-needed -lutil → uartdpi.c'nin openpty'si linklensin (Ubuntu ld --as-needed sorunu)
```
- **Neden sıfır-gecikmeli fonksiyonel GLS?** Gerçek hücre gecikmeleriyle (SDF) koşmuyoruz;
  sadece **fonksiyonel doğruluk + toggle sayısı** lazım. Güç, toggle **sayısına** ve
  frekansa bağlı; gecikme detayı DC tarafında SDC'den (3.845 ns) geliyor. Bu yüzden GLS'i
  hızlı (gecikmesiz) koşup güç frekansını DC'de veriyoruz.

Sonuç: `gls_sim` adında, X-HEEP'in RTL'i + iki gerçek netlist + gözlemciden oluşan tek
simülasyon binary'si.

---

## 5. `gls.sh` diğer komutlar

### 5.1 `build-rtl` / `run-rtl` (61–68, 131–136)
Netlist yok, tümü RTL + gözlemci. Amaç: **doğrulama** — GLS'e geçmeden önce gözlemcinin
pencereleri ve programın sayıları RTL'de doğru mu? (Doğru çıktı, sonra GLS ile birebir
eşleşti.)

### 5.2 `run` (122–130) — SAIF üretimi
```
./gls_sim +firmware=main.hex +phase_addr=$ADDR +saif +saif_dir=saif +vcs+initreg+0 +vcs+lic+wait
```
- `+firmware` programın hex'i; `+saif` gözlemciye "SAIF yaz" der; `+saif_dir` çıktı klasörü.
- Eski SAIF klasörünü silmeyip kenara alır (124) — yarım koşuları kaybetmemek için.
- Sonunda gözlemci loglarını (`[SAIF] ... dur=`) ve UART'ı (`[1]/[3]/...`, bit-exact) ekrana
  döker → hemen doğrulama.

### 5.3 `+vcs+initreg+0` neden (126)
Netlist flop'ları başta X (bilinmiyor). Rastgele init'te (`+random`) pad-mux flip-flop'u t=0'da
rastgele değer alıp `pad_cell_input.sv`'deki bir kontrolü patlatıp `$stop` yaptırıyordu. Hepsini
**0'dan başlatınca** (`+0`) o kontrol geçiyor, boot düzgün ilerliyor.

### 5.4 `phase_addr`'ı bulma (118)
```bash
ADDR=$(awk '$2=="phase_flag"{print $1; exit}' "$MAP"); ADDR=${ADDR#0x}
```
`main.map` (linker haritası) içinde `phase_flag` sembolünün adresini bulur, `0x` ön ekini
atar, `+phase_addr=`'e verir. Böylece adres değişse bile script kendi bulur.

### 5.5 `power` (139–164)
İç içe döngü: 5 faz × 2 blok = 10 kez `dcnxt_shell -f power.tcl` çağırır, her çağrıya env ile
(NAME/NETLIST/SDC/TOP/SAIF/INST/RPT) o (blok,faz)'ı verir. Zaten üretilmiş raporu atlar
(SKIP). Sonunda tüm raporlardan özet tablo çeker (156–162).

---

## 6. `power.tcl` — tek raporun DC içi işlemleri (satır 24–43)

```tcl
read_verilog $env(NETLIST)     ;# gate-level netlist'i oku (cv32e40px_top... veya accel)
current_design $env(TOP)       ;# hangi top? (mangle isim)
link                           ;# hücreleri kütüphaneyle bağla (accel'de xbuf black-box → uyarı normal)
read_sdc $env(SDC)             ;# create_clock 3.845 ns → DİNAMİK GÜÇ BU FREKANSTA hesaplanır
read_saif -input $env(SAIF) -instance_name $env(INST)   ;# aktiviteyi netlist'e "giydir"
report_saif -hier -missing     ;# kaç net eşleşti? (%100 → hepsi giydirildi)
report_power -analysis_effort medium            > ..._total.rpt   ;# tek toplam sayı
report_power -analysis_effort medium -hierarchy > ....rpt         ;# CPU/FPU/FMA/arbiter satır satır
```
- **En kritik satır `read_saif -instance_name`**: SAIF'in içindeki hiyerarşi
  (`tb_top/.../u_gl/...`) ile DC'deki netlist top'unu **hizalar**. `INST` (gls.sh 36-37) bu
  yüzden `.../cv32e40px_top_i/u_gl` — wrapper'ın `u_gl` katmanına kadar iner. Yanlış olsa
  annotasyon %0 çıkardı; %100 çıkması hizanın doğru olduğunu kanıtlar.
- **Güç neyden geliyor?** DC her hücre için: dinamik = (SAIF'ten gelen toggle sayısı) ×
  (hücrenin kapasitesi) × V² × frekans; kaçak = kütüphane sabiti. Yani "aktivite × devre" =
  güç. Aktivite SAIF'ten, devre netlist+kütüphaneden, frekans SDC'den.
- **`_nowlm` (40–43):** WLM (wire-load model, tel kapasite tahmini) kaldırılıp tekrar rapor.
  Amaç: sahte saat neti sayısını (STUDY §6.1) çapraz kontrol. Sonuç aynı çıktı → o sayı
  WLM'den değil, tek-net-yüksek-fanout'tan; yine de sabit olduğu için çıkarınca sorun yok.

---

## 7. Uçtan uca özet (bir COPROC fazını takip et)

1. `power_probe` COPROC fazına girerken `phase_flag=3` **store**'u atar.
2. `cv32e40px_top_i.data_addr_o` o adrese, `data_wdata_o=3` olur; `saif_ctrl` bunu yakalar →
   `$toggle_start`.
3. Koprosesör 134393 cycle boyunca inference yapar; VCS her netin toggle'ını sayar.
4. `phase_flag=0` store'u → `saif_ctrl` `$toggle_stop` + `core_COPROC.saif` &
   `acc_COPROC.saif` yazar.
5. `power.tcl` netlist'i okur, `core_COPROC.saif`'i `-instance_name .../u_gl` ile giydirir,
   `report_power` → FMA satırı 9.30 mW, arbiter 0.051 mW, ...
6. Enerji = güç × (134393 × 3.845 ns) = STUDY §9'daki değerler.

---

## 8. "Yazdığımız RTL" konusuna net cevap

- **Tasarım RTL'i (katkımızın kendisi)** — `dma_apu_arbiter.sv`, `cv32e40px_top.sv`
  değişiklikleri, `dma_fp_dot_accel_pipe.sv`, `xbuf_ram.sv` — bunlar **daha önce** yazıldı ve
  **sentezlendi**; güç çalışması bunların **netlist'ini** kullandı, **değiştirmedi**.
- **Güç çalışması için yazılan tek Verilog** `saif_ctrl.sv` — sentezlenmez, yongaya girmez,
  sadece simülasyonda yandan izler. Yani ölçüm aleti ölçtüğü devreye karışmıyor — bu
  metodolojik olarak doğru olanı.
- Geri kalan yeni iş **yazılım** (`power_probe/main.c`) ve **script** (`gls.sh`, `power.tcl`).

Dolayısıyla güç sonuçları, **hız ve alan sonuçlarıyla tam olarak aynı devrenin** (aynı
netlist, aynı arbiter, aynı koprosesör) ölçümüdür — tutarlı.

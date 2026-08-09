# Dev Ortamı ve Build/Run Flow — Shared-FMA DMA projesi

Bu dosya: ortamın nasıl kurulu olduğu, build/run adımları, her adımın **ne işe yaradığı**,
ne zaman koşulacağı ve karşılaştığımız tuzaklar. Bir daha "ne yapıyordum" dememek için.

Çalışma dizini: `/home/ozan/thesis/resource_shared_dma/x-heep`

---

## 1. Ortam (bir kez kurulu — sadece hatırlatma)

| Bileşen | Ne | Neden / Not |
|---|---|---|
| **Python = venv** | `.venv/` (python3.12), x-heep kökünde | Çıplak `python` yok (sadece `/usr/bin/python3`); mcu-gen/fusesoc `python` arıyor → venv PATH'e koyar. **`source .venv/bin/activate`** |
| **Verilator 5.040** | `/usr/local/bin/verilator` (kaynaktan derlenmiş) | Çalışan sürüm. 5.020 çok eski (trace bug), conda'nın 5.048'i çok yeni (unwaived warning). Sistem geneli, hep çalışır. |
| **Conda (miniforge3)** | `~/miniforge3` var ama **KULLANMA** | Bu proje için terk edildi. `conda activate` edersen: (1) verilator 5.048'e kayar → build kırılır, (2) Makefile conda-branch'ine geçer → yanlış python. Dormant kalsın. |
| **RISC-V toolchain** | `/opt/corev`, prefix `riscv32-corev-` | Hard-float `rv32imfc`. App derlemesi için env gerekli (aşağıda). |
| **Verible** | mcu-gen sonundaki format adımı için | Yoksa mcu-gen sonda hata verir ama **dosyalar zaten üretilir** → kozmetik, boşver. |

### Her yeni terminalde (session başında)
```bash
cd /home/ozan/thesis/resource_shared_dma/x-heep
source .venv/bin/activate                 # (.venv) görmelisin — (base) görürsen: conda deactivate
export RISCV_XHEEP=/opt/corev
export COMPILER_PREFIX=riscv32-corev-
```

---

## 2. Build / Run adımları — ne işe yarar

| Komut | Ne yapar | Ne zaman koşulur | Süre |
|---|---|---|---|
| `make mcu-gen X_HEEP_CFG=configs/cv32e40px_fpu_dma.hjson` | Config + template'lerden (`.tpl`) RTL üretir (`.sv`). CPU/FPU/peripheral/`fpu_addmul_lat` bunlarla gelir. | **Config** ya da **`.tpl`** değiştiğinde | orta |
| `make verilator-build` | RTL'i derleyip **sim binary**'yi (`Vtestharness`) üretir | **mcu-gen sonrası** ya da herhangi **RTL/testharness** değişikliğinde | **yavaş** (~dk) |
| `make app PROJECT=<name> ARCH=rv32imfc_zicsr` | C uygulamasını `.hex`'e derler | **C** değiştiğinde | hızlı |
| `make verilator-run` | Mevcut sim binary'yi app'in `.hex`'iyle koşturur | Her koşuda | hızlı |
| `make verilator-waves` | Waveform (FST) üretir → gtkwave | Dalga şekli lazımsa | — |

### Karar kuralı — "neyi değiştirdim → ne koşarım"
- **Sadece C app** değişti → `make app ...` + `make verilator-run`  *(mcu-gen/build YOK)*
- **RTL** (accel, arbiter, testharness) değişti → `make verilator-build` + app + run
- **Config** (`.hjson`: cpu, fpu, `fpu_addmul_lat`, peripheral) ya da **`.tpl`** değişti → `make mcu-gen ...` + `make verilator-build` + app + run

---

## 3. Tuzaklar (yaşadıklarımız)

- **"python bulamıyor"** → `source .venv/bin/activate` (conda DEĞİL). Fallback: `make ... PY=python3`.
- **conda** → bu projede **aktive etme** (verilator 5.048 build'i kırar).
- **`testharness.sv` GENERATED** — `tb/testharness.sv.tpl`'den üretilir. Düzenlemen gerekeni **`.tpl`'ye** yaz; generated `.sv`'ye yazarsan mcu-gen ezer.
- **`num_pairs_i` (accel N)** — testharness'ta sabit (`16'd32`). App'teki N ile eşleşmeli. `.tpl`'de olduğu için mcu-gen'de korunur. Farklı N için `.tpl`'yi düzenle + rebuild.
- **`fpu_addmul_lat` (L = FMA pipeline derinliği)** — config'te. **Sadece mcu-gen + verilator-build sonrası** geçerli olur. Config değeri ≠ mevcut binary'nin L'si (binary son build'de neyle kurulduysa o). Emin olmak için rebuild.
- **verible yok** → mcu-gen sonunda format hatası ama `.sv`'ler üretilmiş → `make verilator-build`'i ayrı koştur, boşver.
- **Yeni C app** → sadece `make app` + `make verilator-run`; RTL değişmediyse verilator-build gereksiz.

---

## 4. Projeye özel anahtarlar

| Şey | Yer | Ne |
|---|---|---|
| **Config** | `configs/cv32e40px_fpu_dma.hjson` | cv32e40px + `cpu_features:{fpu:true, fpu_addmul_lat:L}` + DMA `hw_fifo_mode_en:yes` |
| **`+define+COPROC_FPU_SHARE`** | `core-v-mini-mcu.core` (~satır 389) | **Ana toggle**: accel + arbiter'ı açar. **Yorum yaparsan → stock X-HEEP baseline** (temiz geri dönüş). |
| **Arbiter** (katkı) | `hw/vendor/xheep/cv32e40px/rtl/dma_apu_arbiter.sv` | CPU-öncelikli FMA paylaşım arbiter'ı, `cv32e40px_top` içinde |
| **Accel** | `tb/dma_fp_dot_accel.sv` | DMA **kanal 1** hw_fifo'ya takılı dot-product coprocessor; FMA'yı APU üzerinden paylaşır |
| **`fpu_addmul_lat` loader fix** | `util/xheep_gen/load_config.py` | cv32e40px'e `fpu_addmul_lat`/`fpu_others_lat` forward eder (upstream eksikti) |
| **Verilator waiver** | `hw/vendor/waiver/lint/cv32e40px.vlt` | 5.040 için `WIDTHEXPAND`/`WIDTHTRUNC` decoder waiver'ı (L≥2'de gerekli) |
| **DC area flow** | `dc/` | `run_area.sh` tek komut; `synth_area.tcl` per-modül |

---

## 5. Sık kullanılan tam akışlar (kopyala-yapıştır)

**Yeni bir C app koştur (RTL sabit):**
```bash
source .venv/bin/activate
export RISCV_XHEEP=/opt/corev; export COMPILER_PREFIX=riscv32-corev-
make app PROJECT=ml_fc_bringup ARCH=rv32imfc_zicsr
make verilator-run
```

**L'yi (fpu_addmul_lat) değiştir (ör. 0'a al):**
```bash
# configs/cv32e40px_fpu_dma.hjson içinde fpu_addmul_lat: 0
source .venv/bin/activate
make mcu-gen X_HEEP_CFG=configs/cv32e40px_fpu_dma.hjson
make verilator-build
```

**Baseline'a dön (accel/arbiter kapat):**
```bash
# core-v-mini-mcu.core: '+define+COPROC_FPU_SHARE' satırını yorum yap
make verilator-build      # mcu-gen gerekmez; ifdef'ler generated .sv'de
make app PROJECT=example_matfloat ARCH=rv32imfc_zicsr && make verilator-run
```

**Waveform:**
```bash
make verilator-waves      # sonra gtkwave ile aç
```

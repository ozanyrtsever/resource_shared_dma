# DC sentez çalışması — operating-point alan & Fmax (converge ile bulundu)

**Araç:** Synopsys DC-NXT (Y-2026.03-SP1). **Kütüphane:** TSMC 40 nm G, `sc12mc_cln40g_base_rvt` (SS/0.81 V/125 °C). **1 GE = 0.9576 µm².**
**Yöntem:** `dc_scripts/converge.sh` — `CLK += |WNS|/2` (senin formülün) ile en yavaş config (share_L0, kombinasyonel FMA) üzerinden clock'u gerçek fmax'e yakınsatır (WNS→0); sonra `study.sh` tüm 9 config'i o clock'ta koşar, hiyerarşi korunur (`-no_autoungroup -no_boundary_optimization`) → FMA/FPU/arbiter/core `report_area -hierarchy`'den ayrıştırılır. Accel ayrı (core dışında). io_delay 0.5, rst false_path.
**BAZ operating point: 3.845 ns = 260 MHz** (converge'in bulduğu, WNS=0 → gerçek fmax). **4.0 ns = 250 MHz, §2'de EK olarak** verildi (karşılaştırma). Baz sayılar §1 ve §3'ten. Kaynak: `dc_reports/clk3p845/` (baz), `dc_reports/clk4p0/` (ek).

---

## 0. Convergence izi (converge.sh, `CLK += |WNS|/2`)

| iter | CLK (ns) | share_L0 WNS (ns) |
|:-:|:-------:|:----------------:|
| 1 | 0.100 | −3.94 |
| 2 | 2.070 | −1.93 |
| 3 | 3.035 | −0.84 |
| 4 | 3.455 | −0.40 |
| 5 | 3.655 | −0.19 |
| 6 | 3.750 | −0.13 |
| 7 | 3.815 | −0.02 |
| 8 | **3.845** | **0.00 → yakınsadı** |

→ **Operating clock = 3.845 ns ≈ 260 MHz** (WNS tam 0 = fmax kenarı, loose değil).

---

## 1. Operating point: 3.845 ns = **260 MHz** (converged)

### 1a. Config bazında toplam alan + fmax (OZET)
| config | total area (µm²) | GE | WNS (ns) | fmax (MHz) |
|---|---:|---:|:-:|:-:|
| share_L0 (P0) | 87 957 | 91 862 | 0.00 | 260.1 |
| share_L1 | 81 151 | 84 744 | 0.00 | 260.1 |
| share_L2 | 81 695 | 85 312 | 0.00 | 260.1 |
| share_L3 | 82 012 | 85 643 | 0.00 | 260.1 |
| share_L4 | 82 179 | 85 817 | 0.00 | 260.1 |
| share_L5 | 82 750 | 86 414 | 0.00 | 260.1 |
| share_P1_L0 | 88 223 | 92 129 | 0.00 | 260.1 |
| share_P2_L0 | 90 313 | 94 322 | −0.12 | 252.2 |
| **accel** (ayrı) | **4 922** | **5 140** | 0.01 | 260.8 |

*Not: L=0 (kombinasyonel FMA) en büyük core (88K); L≥1 pipelined FMA'da core daha küçük (81-83K). P2 (QoS arbiter) 3.845 ns'i tam tutturamadı → 252 MHz.*

### 1b. Bileşen ayrıştırma (share_L0, P0 — hiyerarşiden)
| bileşen | µm² | GE | core'un %'si |
|---|---:|---:|:-:|
| CPU core (`cv32e40px_core`) | 52 904 | 55 246 | 60.1% |
| FPU (`cv32e40px_fp_wrapper`) | 33 234 | 34 706 | 37.8% |
| — of which **FMA** (`fpnew_fma_multi`) | **21 546** | **22 500** | 24.5% |
| **arbiter** (`dma_apu_arbiter`, P0) | **1 803** | **1 883** | **2.0%** |
| **core total** (`cv32e40px_top`) | **87 957** | **91 862** | 100% |
| FMA @ L=5 (pipelined) | 16 815 | 17 560 | — |

### 1c. Arbiter policy alanı (L=0)
| policy | µm² | GE |
|---|---:|---:|
| P0 CPU-strict | 1 803 | 1 883 |
| P1 round-robin | 1 518 | 1 585 |
| P2 QoS | 2 263 | 2 363 |

---

## 2. Operating point: 4.0 ns = **250 MHz**

### 2a. Config bazında toplam alan + fmax (OZET)
| config | total area (µm²) | GE | WNS (ns) | fmax (MHz) |
|---|---:|---:|:-:|:-:|
| share_L0 (P0) | 85 261 | 89 035 | 0.00 | 250.0 |
| share_L1 | 79 276 | 82 786 | 0.00 | 250.0 |
| share_L2 | 79 932 | 83 471 | 0.00 | 250.0 |
| share_L3 | 80 047 | 83 592 | 0.00 | 250.0 |
| share_L4 | 79 711 | 83 240 | 0.00 | 250.0 |
| share_L5 | 81 164 | 84 757 | 0.00 | 250.0 |
| share_P1_L0 | 85 510 | 89 296 | 0.00 | 250.0 |
| share_P2_L0 | 89 435 | 93 395 | −0.02 | 248.8 |
| **accel** (ayrı) | **4 914** | **5 132** | 0.00 | 250.0 |

### 2b. Bileşen ayrıştırma (share_L0, P0)
| bileşen | µm² | GE |
|---|---:|---:|
| CPU core | 53 127 | 55 479 |
| FPU | 30 630 | 31 987 |
| — **FMA** | **19 295** | **20 149** |
| **arbiter** (P0) | **1 496** | **1 562** |
| **core total** | **85 261** | **89 035** |
| FMA @ L=5 | 16 602 | 17 337 |

### 2c. Arbiter policy alanı (L=0)
| policy | µm² | GE |
|---|---:|---:|
| P0 | 1 496 | 1 562 |
| P1 | 1 420 | 1 483 |
| P2 QoS | 2 516 | 2 627 |

---

## 3. "İkinci FPU yok" — türetilen (260 MHz, L=0)
| | µm² | GE |
|---|---:|---:|
| **Kaçınılan** — bir FMA (`fpnew_fma_multi`) | 21 546 | 22 500 |
| **Kaçınılan** — tam 2. FPU (`fp_wrapper`) | 33 234 | 34 706 |
| **Eklenen** — sharing arbiter (P0) | 1 803 | 1 883 |
| arbiter / bir FMA | **8.4%** | |
| arbiter / tüm core | **2.0%** | |
| **Tasarruf (FMA − arbiter) / accel** | **19 743** | 20 617 |

**Dedicated vs shared (accel başına eklenen alan):**
- Dedicated-FMA accel: accel + kendi FMA = 4 922 + 21 546 = **26 468 µm²**
- **Shared** accel: accel + arbiter = 4 922 + 1 803 = **6 725 µm²**
- → **3.9× daha az alan** (arbiter, koca bir FMA'nın yerini alıyor).

---

## 4. Özet
- **Fmax:** paylaşımlı çekirdek **260 MHz** (3.845 ns, converge). 250 MHz'de de rahat çalışır. Accel ~260 MHz (kritik yolda değil).
- **Bileşen (260 MHz, L0):** core 52.9K + FPU 33.2K (FMA 21.5K) + arbiter 1.8K = 88.0K µm² (91.9K GE).
- **İkinci FPU yok:** arbiter = bir FMA'nın **%8.4'ü** / core'un **%2.0'si**; accel başına **3.9× daha az** alan.
- **Policy:** P1 < P0 < P2 (QoS ~1.5× P0); P2 en yavaş (252 MHz).
- **L etkisi:** L=0 kombinasyonel FMA en büyük (21.5K); L≥1 pipelined FMA daha küçük (~16.8K @L5) → L≥1 build'de core ~82K.
- Bu sayılar **gerçek operating-point** (fmax kenarı, 0.1ns şişkin değil).

*Kaynak: `dc_reports/clk3p845/` (260 MHz) + `dc_reports/clk4p0/` (250 MHz) — OZET.txt + area_*.rpt hiyerarşi. Convergence: `dc_reports/clk*/timing_share_L0.rpt`.*

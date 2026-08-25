#!/usr/bin/env bash
# Bizim sistemin ALAN + FMAX'i (base YOK; "sistemimiz su hizda calisiyor"). x-heep kokunden: ./dc_scripts/study.sh
# 9 run: core share (cv32e40px_top, arbiter iceride) L=0..5 -> report_area -hier'den FMA/FPU/arbiter/core
#        + arbiter policy P1,P2 @L0 + accel (core disinda, kendi top'u; x_buf black-box).
# Konvansiyon: clk = $CLK ns + 0.5 IO delay -> alan ve fmax AYNI run'dan.  fmax = 1000/(CLK - WNS).
#
# CLOCK secimi:  CLK=<ns> ./dc_scripts/study.sh    (varsayilan 0.1)
#   HER CLK KENDI KLASORUNE yazar -> dc_reports/clk<CLK>/  ==> uzerine YAZMAZ, wipe GEREKMEZ.
#   Ornekler:  CLK=0.1 (max-hiz, fmax dogru)   CLK=4.0 (gercek fmax ~250MHz)   CLK=10 (100MHz)
#   Ayni CLK'yi tekrar kosarsan resume kaldigi yerden devam eder.
set -u
cd "$(dirname "$0")/.."

# dcnxt_shell'i PATH'e ekle (yoksa). Farkli yerdeyse:  DCSH_BIN=/dogru/yol ...
DCSH_BIN="${DCSH_BIN:-/2tb/ECE/synopsys/2025-26/bin}"
command -v dcnxt_shell >/dev/null 2>&1 || export PATH="$DCSH_BIN:$PATH"
command -v dcnxt_shell >/dev/null 2>&1 || { echo "HATA: dcnxt_shell bulunamadi (DCSH_BIN=$DCSH_BIN dogru mu?)"; exit 1; }

CLK="${CLK:-0.1}"                          # clock (ns);  CLK=4.0 ./dc_scripts/study.sh
CLKTAG=$(printf '%s' "$CLK" | tr '.' 'p')  # 0.1->0p1, 4.0->4p0, 10->10
RPT="${RPT:-dc_reports/clk$CLKTAG}"        # HER CLK KENDI KLASORU (uzerine yazmaz)
mkdir -p "$RPT" dc_work
echo ">>> clock = $CLK ns   ->   raporlar: $RPT/"

run() {  # run <isim> <top> <defines> <params>
  local RUN=$1 TOP=$2 DEFS=$3 PARAMS=$4
  local FLIST=dc_scripts/rtl_core.f
  [ "$TOP" = dma_fp_dot_accel_pipe ] && FLIST=dc_scripts/rtl_accel_pipe.f
  # resume: bu klasorde (bu CLK'de) basariyla biten run'i ATLA
  if [ -s "$RPT/area_$RUN.rpt" ] && grep -q "Total cell area" "$RPT/area_$RUN.rpt"; then
    echo "  $RUN : SKIP (zaten tamam)"
  else
    echo "############  $RUN  ($CLK ns)  ############"
    rm -rf "dc_work/$RUN" "$RPT/area_$RUN.rpt" "$RPT/timing_$RUN.rpt"
    RUN=$RUN TOP=$TOP CLK=$CLK DEFS=$DEFS PARAMS=$PARAMS FLIST=$FLIST RPT=$RPT \
        dcnxt_shell -f dc_scripts/study.tcl 2>&1 | tee "$RPT/log_$RUN.txt" \
        | grep --line-buffered -iE "beginning|elaborat|optimization complete|Total cell area|error:|\*\*error|fatal" || true   # faz akisi + hatalar
  fi
  local A W F
  A=$(grep -i "Total cell area" "$RPT/area_$RUN.rpt" 2>/dev/null | awk '{print $NF}')
  W=$(grep "slack (" "$RPT/timing_$RUN.rpt" 2>/dev/null | awk '{print $NF}' | sort -g | head -1)
  if [ -n "${W:-}" ]; then F=$(awk -v c="$CLK" -v w="$W" 'BEGIN{printf "%.1f", 1000/(c-w)}'); else F="?"; fi
  if [ -z "${A:-}" ]; then
    echo "==== $RUN : FAILED -> bkz $RPT/log_$RUN.txt ====" | tee -a "$RPT/OZET.txt"
  else
    echo "==== $RUN : total-area $A um2 , WNS ${W:-?} ns , fmax ~$F MHz ====" | tee -a "$RPT/OZET.txt"
  fi
}

: > "$RPT/OZET.txt"

# --- BIZIM SISTEM: core (share, arbiter iceride), L=0..5 ---
for L in 0 1 2 3 4 5; do
  run share_L$L cv32e40px_top "SYNTHESIS COPROC_FPU_SHARE ARB_POLICY_SEL=0" "FPU=1, FPU_ADDMUL_LAT=$L"
done
# --- arbiter policy ALAN kiyasi (L-bagimsiz -> @L0): P1,P2 (P0 zaten share_L0) ---
for P in 1 2; do
  run share_P${P}_L0 cv32e40px_top "SYNTHESIS COPROC_FPU_SHARE ARB_POLICY_SEL=$P" "FPU=1, FPU_ADDMUL_LAT=0"
done
# --- accel (core disinda, kendi top'u; x_buf black-box) ---
run accel dma_fp_dot_accel_pipe "SYNTHESIS" "MAXN=1024, MAXB=8"

# ================================ OZET ================================
echo; echo "===== OZET ($CLK ns -> $RPT) ====="; cat "$RPT/OZET.txt"
echo; echo "===== BILESENLER (share_L0 hierarchy) ====="
grep -iE "fpnew_fma|fp_wrapper|dma_apu_arbiter|cv32e40px_core|Total cell" "$RPT/area_share_L0.rpt" 2>/dev/null
echo; echo "----- FMA/FPU @ L=5 -----"
grep -iE "fpnew_fma|fp_wrapper" "$RPT/area_share_L5.rpt" 2>/dev/null
echo; echo "----- arbiter alani, policy'ye gore -----"
grep -E "arb_i " "$RPT/area_share_L0.rpt"    2>/dev/null | head -1 | sed 's/^/  P0: /'
grep -E "arb_i " "$RPT/area_share_P1_L0.rpt" 2>/dev/null | head -1 | sed 's/^/  P1: /'
grep -E "arb_i " "$RPT/area_share_P2_L0.rpt" 2>/dev/null | head -1 | sed 's/^/  P2: /'
echo; echo "===== BITTI  ($RPT) ====="

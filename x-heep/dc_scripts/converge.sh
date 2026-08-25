#!/usr/bin/env bash
# CLOCK yakinsama -- SENIN formulun:  yeni_CLK = CLK + |WNS|/2   (CLK ile tahmini-min-periyodun ORTASI).
# Probe = share_L0 (bottleneck, kombinasyonel FMA = en yavas -> clock'u O belirler).
# WNS 0'a ALTTAN yaklasir (fmax kenari, LOOSE DEGIL). |WNS|<=TOL olunca minik tam-adimla pozitife gecip
# durur (cunku /2 tek basina strict pozitife tam degmez). Sonra TUM matrisi bulunan clock'ta kosar.
#
# Kullanim (tcsh, tmux icinde):  ./dc_scripts/converge.sh    |    env CLK0=0.1 TOL=0.05 ./dc_scripts/converge.sh
set -u
cd "$(dirname "$0")/.."
DCSH_BIN="${DCSH_BIN:-/2tb/ECE/synopsys/2025-26/bin}"
command -v dcnxt_shell >/dev/null 2>&1 || export PATH="$DCSH_BIN:$PATH"
command -v dcnxt_shell >/dev/null 2>&1 || { echo "HATA: dcnxt_shell bulunamadi (DCSH_BIN=$DCSH_BIN)"; exit 1; }

CLK="${CLK0:-0.1}"          # baslangic clock (ns)
TOL="${TOL:-0.05}"          # kenara bu kadar (ns) yaklasinca pozitife gecip dur
MAXIT="${MAXIT:-20}"
PROBE="${PROBE:-share_L0}"  # clock'u belirleyen en yavas config

# $1=clk -> share_L0'i (yoksa) sentezle, worst WNS'i STDOUT'a yaz (ayni clock tekrar istenirse cache)
wns() {
  local clk=$1 tag rpt
  tag=$(printf '%s' "$clk" | tr '.' 'p'); rpt="dc_reports/clk$tag"; mkdir -p "$rpt"
  if [ ! -s "$rpt/timing_$PROBE.rpt" ]; then
    rm -rf "dc_work/$PROBE"
    echo "      ($clk ns) share_L0 sentezleniyor... (canli faz akisi; tam log: $rpt/log_$PROBE.txt)" >&2
    # NOT: tum ilerleme ciktisi STDERR'e (>&2) -> wns() stdout'u sadece WNS icin ayrildi (yoksa bozulur)
    RUN=$PROBE TOP=cv32e40px_top CLK="$clk" RPT="$rpt" FLIST=dc_scripts/rtl_core.f \
      DEFS="SYNTHESIS COPROC_FPU_SHARE ARB_POLICY_SEL=0" PARAMS="FPU=1, FPU_ADDMUL_LAT=0" \
      dcnxt_shell -f dc_scripts/study.tcl 2>&1 | tee "$rpt/log_$PROBE.txt" \
      | grep --line-buffered -iE "beginning|elaborat|optimization complete|Total cell area|error:|\*\*error|fatal" \
      | sed 's/^/        | /' >&2
  fi
  grep "slack (" "$rpt/timing_$PROBE.rpt" 2>/dev/null | awk '{print $NF}' | sort -g | head -1
}

echo "=== CLOCK YAKINSAMA  (formul: CLK += |WNS|/2 ; probe=$PROBE ; TOL=$TOL) ==="
for it in $(seq 1 "$MAXIT"); do
  W=$(wns "$CLK")
  [ -z "${W:-}" ] && { echo "HATA: WNS okunamadi (clk=$CLK) -> dc_reports/clk$(printf '%s' "$CLK"|tr '.' 'p')/log_$PROBE.txt"; exit 1; }
  echo "  iter $it : CLK=$CLK ns  ->  WNS($PROBE) = $W ns"
  if awk -v w="$W" 'BEGIN{exit (w>=0)?0:1}'; then    # pozitif -> siki + karsiliyor -> BITTI
    echo ">>> BITTI: operating clock = $CLK ns  (fmax ~$(awk -v c="$CLK" 'BEGIN{printf "%.1f",1000/c}') MHz)"; break
  fi
  # adim: |WNS|>TOL iken senin formulun (|WNS|/2); kenara cok yakinsa (|WNS|<=TOL) minik tam-adim -> pozitife gec
  CLK=$(awk -v c="$CLK" -v w="$W" -v t="$TOL" 'BEGIN{ a=(w<0?-w:w); s=(a<=t? a+0.01 : a/2); printf "%.3f", c+s }')
  [ "$it" = "$MAXIT" ] && echo ">>> MAXIT ($MAXIT) doldu, yakinsamadi -> son CLK=$CLK (TOL'u buyut ya da elle bak)"
done

echo; echo "=== TUM MATRIS @ $CLK ns (bulunan operating clock) ==="
CLK="$CLK" ./dc_scripts/study.sh

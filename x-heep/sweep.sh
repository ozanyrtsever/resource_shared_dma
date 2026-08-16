#!/bin/bash
# sweep.sh — RESUMABLE sweep over L (fpu_addmul_lat) x arbiter policy x QoS weights.
# Skips any config already done (output has "ALL bit-exact=1"), so re-running only fills the gaps.
# IMPORTANT: disable tracing in core-v-mini-mcu.core (comment the --trace* lines) for reasonable speed.
#
# ONE script, two benchmarks -- pick via env vars (defaults keep the perf_bench behavior):
#   perf_bench (toy MLP):  nohup ./sweep.sh > sweep_nohup.log 2>&1 &
#   LeNet-300-100       :  PROJECT=perf_lenet OUTDIR=sweep_results/lenet RUN_TIMEOUT=14400 \
#                          nohup ./sweep.sh > sweep_lenet_nohup.log 2>&1 &
# (LeNet is ~25x the MACs -> each config ~1.5-3 h; raise RUN_TIMEOUT and trim CONFIGS if needed.)

cd /home/ozan/thesis/resource_shared_dma/x-heep || { echo "wrong dir"; exit 1; }
export MAKEFLAGS="-j$(nproc)"                       # parallel Verilator C++ compile -> ~5x faster build
PROJECT="${PROJECT:-perf_bench}"                    # app to build/run (perf_bench | perf_lenet)
OUTDIR="${OUTDIR:-sweep_results/final}"            # override for a separate folder -> no collision
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"                 # per-config sim timeout (s); LeNet needs more
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/00_summary.csv"
LOG="$OUTDIR/00_sweep.log"
[ -f "$SUMMARY" ] || echo "name,L,policy,w_cpu,w_acc0,w_acc1,status,bit_exact" > "$SUMMARY"

CFG=configs/cv32e40px_fpu_dma.hjson
TOP=hw/vendor/xheep/cv32e40px/rtl/cv32e40px_top.sv
UART=build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/sim-verilator/uart0.log

# "L POLICY W_CPU W_ACC0 W_ACC1"   (weights used only by policy 2)
CONFIGS=(
  "0 0 1 1 1" "0 1 1 1 1" "0 2 1 1 1" "0 2 4 1 1" "0 2 1 4 4" "0 2 1 4 1"   # L=0 (+QoS weights)
  "1 0 1 1 1" "1 1 1 1 1" "1 2 4 2 1"
  "2 0 1 1 1" "2 1 1 1 1" "2 2 4 2 1"
  "3 0 1 1 1" "3 1 1 1 1" "3 2 4 2 1"
  "4 0 1 1 1" "4 1 1 1 1" "4 2 4 2 1"
  "5 0 1 1 1" "5 1 1 1 1" "5 2 4 2 1"
)

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "SWEEP START  app=$PROJECT  -> $OUTDIR  (${#CONFIGS[@]} configs, timeout=${RUN_TIMEOUT}s, resumable)"

i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i+1))
  read -r L P WC WA0 WA1 <<< "$cfg"
  NAME="L${L}_P${P}_w${WC}-${WA0}-${WA1}"
  OUTFILE="$OUTDIR/${NAME}.txt"
  if [ -f "$OUTFILE" ] && grep -q "ALL bit-exact=1" "$OUTFILE"; then
    log "[$i/${#CONFIGS[@]}] $NAME  -- SKIP (already done)"; continue
  fi
  log "[$i/${#CONFIGS[@]}] $NAME  (L=$L policy=$P w=$WC:$WA0:$WA1)"

  sed -i "s/fpu_addmul_lat: *[0-9]*/fpu_addmul_lat: $L/"                 "$CFG"
  sed -i "s/\`define ARB_POLICY_SEL *[0-9]*/\`define ARB_POLICY_SEL $P/" "$TOP"
  sed -i "s/\`define ARB_W_CPU *[0-9]*/\`define ARB_W_CPU $WC/"          "$TOP"
  sed -i "s/\`define ARB_W_ACC0 *[0-9]*/\`define ARB_W_ACC0 $WA0/"       "$TOP"
  sed -i "s/\`define ARB_W_ACC1 *[0-9]*/\`define ARB_W_ACC1 $WA1/"       "$TOP"

  BUILDLOG="$OUTDIR/${NAME}.build.log"; STATUS=ok
  make mcu-gen X_HEEP_CFG="$CFG"                        > "$BUILDLOG" 2>&1 || STATUS=mcu-gen-fail
  [ "$STATUS" = ok ] && { make verilator-build         >>"$BUILDLOG" 2>&1 || STATUS=build-fail; }
  [ "$STATUS" = ok ] && { make app PROJECT="$PROJECT" ARCH=rv32imfc_zicsr >>"$BUILDLOG" 2>&1 || STATUS=app-fail; }
  [ "$STATUS" = ok ] && { timeout "$RUN_TIMEOUT" make verilator-run >>"$BUILDLOG" 2>&1 || STATUS=run-fail; }

  { echo "############################################################"
    echo "# $NAME   L(lat)=$L  policy=$P  weights(CPU:ACC0:ACC1)=$WC:$WA0:$WA1"
    echo "# status=$STATUS   $(date)"
    echo "############################################################"
    [ -f "$UART" ] && cat "$UART" || echo "(no uart0.log — see ${NAME}.build.log)"; } > "$OUTFILE"

  BE=$(grep -oE 'ALL bit-exact=[0-9]' "$OUTFILE" | grep -oE '[0-9]$' | tail -1); [ -z "$BE" ] && BE=NA
  echo "$NAME,$L,$P,$WC,$WA0,$WA1,$STATUS,$BE" >> "$SUMMARY"
  log "    -> status=$STATUS  bit-exact=$BE"
done
log "SWEEP DONE.  Summary: $SUMMARY"

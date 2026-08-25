#!/usr/bin/env bash
#=============================================================================
# One command for the whole "no second FPU" area study (Design Compiler).
# Runs every point in a FRESH dcnxt_shell (zero state carryover) with its own WORK
# dir, then prints the summary tables. Run from anywhere; cd's to x-heep root.
#
#   bash dc_scripts/run_area.sh
#
# Matrix (10 runs):
#   (a) full FPU  cv32e40px_fp_wrapper  @ L = 0..5        -> 6 runs
#       (each run's *_hier.rpt also gives the FMA-alone area: fpnew_fma_multi row)
#   (c) arbiter   dma_apu_arbiter       @ POLICY = 0,1,2  -> 3 runs
#   (d) accel     dma_fp_dot_accel_is   (LOGIC only)      -> 1 run
#
# Prereqs on this server:
#   - Edit LIB_DB at the top of dc_scripts/synth_area.tcl to the standard-cell .db here.
#   - The RTL edit that factors x_buf into tb/xbuf_ram.sv must be applied (so the
#     accel instantiates u_xbuf); dc_scripts/rtl_accel_is.f omits xbuf_ram to black-box it.
#=============================================================================
set -u
cd "$(dirname "$0")/.."                     # -> x-heep root
mkdir -p dc_reports
CLK_NS="${CLK_NS:-10.0}"
DCSH="${DCSH:-dcnxt_shell}"                  # DC shell binary (dcnxt_shell = DC NXT).
                                            # override w/ full path if not in PATH, e.g.:
                                            #   DCSH=/2tb/ECE/synopsys/2025-26/bin/dcnxt_shell bash dc_scripts/run_area.sh

# --- run one point in a fresh dcnxt_shell.  $1 = RUN label (must equal what the tcl
#     builds from TOP+suffix); remaining args = env assignments (incl TOP) --------
runone() {
    local RUN="$1"; shift
    echo "########################  SYNTH: $RUN  ########################"
    rm -rf "dc_work/$RUN" "dc_reports/area_${RUN}.rpt" "dc_reports/area_${RUN}_hier.rpt"
    env "$@" CLK_NS="$CLK_NS" "$DCSH" -f dc_scripts/synth_area.tcl 2>&1 | tee "dc_reports/log_${RUN}.txt"
}

# ============================== RUN THE MATRIX ==============================
# (a) full FPU + FMA-in-context, ADDMUL-latency sweep
for L in 0 1 2 3 4 5; do
    runone "cv32e40px_fp_wrapper_L$L"  TOP=cv32e40px_fp_wrapper  L="$L"
done
# (c) arbiter, three policies
for P in 0 1 2; do
    runone "dma_apu_arbiter_P$P"       TOP=dma_apu_arbiter       POLICY="$P"
done
# (d) accelerators, LOGIC only (x_buf black-boxed as a 32 KB SRAM macro): serial + pipelined
runone "dma_fp_dot_accel_is"           TOP=dma_fp_dot_accel_is
runone "dma_fp_dot_accel_pipe"         TOP=dma_fp_dot_accel_pipe

# ================================ SUMMARY ==================================
# Total cell area from a flat area report.
area_of() { grep -i "Total cell area" "dc_reports/area_$1.rpt" 2>/dev/null | tail -1 | awk '{print $NF}'; }
ge()      { awk -v a="${1:-}" 'BEGIN{ if(a=="") print "?"; else printf "%.0f", a/0.9576 }'; }  # 1 GE = 0.9576 um^2 (NAND2_X1M_A12TR40)
# FMA-alone (fpnew_fma_multi) absolute area from the fp_wrapper hierarchy report at latency L.
fma_of()  { awk '/fpnew_fma_multi/ && $1 ~ /^[0-9]/ {print $1; exit}' \
              "dc_reports/area_cv32e40px_fp_wrapper_L$1_hier.rpt" 2>/dev/null; }

echo
echo "##################  AREA SUMMARY  (TSMC40, clk=${CLK_NS} ns)  ##################"
echo
echo "--- (a) Full FPU (cv32e40px_fp_wrapper) and FMA (fpnew_fma_multi, from hier), by ADDMUL latency L ---"
printf "%3s %18s %18s %18s\n" "L" "full FPU (um^2)" "FMA (um^2)" "rest = FPU-FMA"
for L in 0 1 2 3 4 5; do
    f=$(area_of "cv32e40px_fp_wrapper_L$L"); m=$(fma_of "$L")
    if [ -n "${f:-}" ] && [ -n "${m:-}" ]; then
        r=$(awk -v a="$f" -v b="$m" 'BEGIN{printf "%.2f", a-b}')
    else
        r="?"
    fi
    [ -z "${m:-}" ] && m="MISSING: FMA row ungrouped -> run 'TOP=fpnew_fma_multi L=$L dcnxt_shell -f dc_scripts/synth_area.tcl'"
    printf "%3s %18s %18s %18s\n" "$L" "${f:-?}" "$m" "$r"
done

echo
echo "--- (c) Arbiter (dma_apu_arbiter), per policy ---"
printf "%-26s %18s\n" "policy" "area (um^2)"
printf "%-26s %18s\n" "P0  CPU-strict + accRR" "$(area_of dma_apu_arbiter_P0)"
printf "%-26s %18s\n" "P1  all round-robin"    "$(area_of dma_apu_arbiter_P1)"
printf "%-26s %18s\n" "P2  QoS weighted 4:2:1" "$(area_of dma_apu_arbiter_P2)"

echo
echo "--- (d) Accelerators, LOGIC only (buffer excluded) ---"
printf "%-32s %16s %12s\n" "design" "area (um^2)" "area (GE)"
IS=$(area_of dma_fp_dot_accel_is);  PIPE=$(area_of dma_fp_dot_accel_pipe)
printf "%-32s %16s %12s\n" "serial (dma_fp_dot_accel_is)"   "${IS:-?}"   "$(ge "${IS:-}")"
printf "%-32s %16s %12s\n" "pipe   (dma_fp_dot_accel_pipe)" "${PIPE:-?}" "$(ge "${PIPE:-}")"
if [ -n "${IS:-}" ] && [ -n "${PIPE:-}" ]; then
    awk -v a="$IS" -v b="$PIPE" 'BEGIN{printf "   pipelining cost = %+.2f um^2 (%+.0f GE, %+.1f%%) vs serial\n", b-a, (b-a)/0.9576, 100*(b-a)/a}'
fi
echo "   input buffer x_buf = MAXB*MAXN*32b = 8*1024*32 = 262144 b = 32 KB single-port SRAM /accel"
echo "   (40nm 6T rough estimate ~0.11 mm^2 for 32 KB). Never counted as flip-flops.  1 GE = 0.9576 um^2."

echo
echo "--- 'no second FPU' comparison ---"
FMA0=$(fma_of 0); FPU0=$(area_of cv32e40px_fp_wrapper_L0)
ARB=$(area_of dma_apu_arbiter_P0); ACC=$(area_of dma_fp_dot_accel_is)
awk -v fma="$FMA0" -v fpu="$FPU0" -v arb="$ARB" -v acc="$ACC" 'BEGIN{
    if (fma=="" || arb=="") { print "  (need FMA and arbiter areas)"; exit }
    printf "  AVOIDED  a dedicated FMA (fpnew_fma_multi, L=0) : %12.2f um^2\n", fma
    if (fpu!="") printf "  AVOIDED  a full 2nd FPU (fp_wrapper,   L=0) : %12.2f um^2\n", fpu
    printf "  ADDED    the sharing arbiter (P0)              : %12.2f um^2  (%.1f%% of one FMA)\n", arb, 100*arb/fma
    printf "  SAVING   FMA - arbiter                         : %12.2f um^2\n", fma-arb
    if (acc!="") printf "  (accel logic = %.2f um^2 is common to BOTH designs -- shared or dedicated FMA)\n", acc
    print  "  -> sharing the CPU FMA costs a small arbiter instead of a whole FMA/FPU."
}'
echo "##################################  DONE  ##################################"

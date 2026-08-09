#!/usr/bin/env bash
#=============================================================================
# One command to synthesize EVERY module for the "no second FPU" area study.
# Each module runs in a FRESH dc_shell process (zero state carryover) with its
# own WORK dir. Prints a summary table + the area comparison at the end.
#
#     bash dc/run_area.sh                 # the 3 key modules
#     MODULES="cv32e40px_top" bash dc/run_area.sh   # override the list
#=============================================================================
set -u
cd "$(dirname "$0")/.."                     # -> x-heep root, wherever invoked from
mkdir -p dc/reports

MODULES="${MODULES:-cv32e40px_fp_wrapper dma_apu_arbiter dma_fp_dot_accel}"
CLK_NS="${CLK_NS:-10.0}"

for TOP in $MODULES; do
    echo "########################  SYNTH: $TOP  ########################"
    rm -rf "dc/work/$TOP" "dc/reports/area_${TOP}.rpt"      # clean stale outputs
    TOP="$TOP" CLK_NS="$CLK_NS" dc_shell -f dc/synth_area.tcl 2>&1 | tee "dc/reports/log_${TOP}.txt"
done

# ------------------------------- summary ------------------------------------
area_of() { grep -i "Total cell area" "dc/reports/area_$1.rpt" 2>/dev/null | tail -1 | awk '{print $NF}'; }

echo
echo "########################  AREA SUMMARY (clk=${CLK_NS} ns)  ########################"
printf "%-26s %14s\n" "module" "area(um^2)"
for TOP in $MODULES; do printf "%-26s %14s\n" "$TOP" "$(area_of "$TOP")"; done

fma=$(area_of cv32e40px_fp_wrapper)
arb=$(area_of dma_apu_arbiter)
acc=$(area_of dma_fp_dot_accel)
if [ -n "${fma:-}" ] && [ -n "${arb:-}" ] && [ -n "${acc:-}" ]; then
  awk -v f="$fma" -v a="$arb" -v c="$acc" 'BEGIN{
    if (f+0>0) {
      printf "\n# AVOIDED  one FMA (fp_wrapper) : %12.2f\n", f
      printf "# ADDED    arbiter              : %12.2f  (%.1f%% of one FMA)\n", a, 100*a/f
      printf "# ADDED    accel FSM            : %12.2f  (%.1f%% of one FMA)\n", c, 100*c/f
      printf "# SAVING = FMA - arbiter        : %12.2f  (accel FSM is common to both designs)\n", f-a
    }
  }'
fi
echo "########################  DONE  ########################"

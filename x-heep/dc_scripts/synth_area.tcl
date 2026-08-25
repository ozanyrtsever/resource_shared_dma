#=============================================================================
# Area synthesis of ONE module for the shared-FMA thesis (Design Compiler).
#
# Driven per-run by dc_scripts/run_area.sh (a FRESH dcnxt_shell each -> zero state mixing).
# To run one module by hand:  TOP=dma_apu_arbiter dcnxt_shell -f dc_scripts/synth_area.tcl
#
# Environment inputs:
#   TOP      module to synthesize (default cv32e40px_fp_wrapper)
#   CLK_NS   clock period in ns (keep IDENTICAL across all runs for a fair compare)
#   L        ADD/MUL(FMA) pipeline latency:
#              cv32e40px_fp_wrapper -> parameter FPU_ADDMUL_LAT
#              fpnew_fma / fpnew_fma_multi -> parameter NumPipeRegs
#   POLICY   arbiter policy for dma_apu_arbiter -> parameter ARB_POLICY (0/1/2)
#   MAXN,MAXB  accelerator buffer dims for dma_fp_dot_accel_is
#
# Output reports go to dc_reports/area_<RUN>.rpt (+ _hier/refs/timing), where
#   <RUN> = <TOP> + suffix (_L#, _P#, _N#, _B#) so sweeps never overwrite each other.
#
# Edit LIB_DB below ONCE on the machine that runs Design Compiler.
#=============================================================================

# ----------------------------- USER SETTING ---------------------------------
# .db yolu: env LIB_DB > dc_scripts/ > dc/ (asagidaki GUARD bulunamazsa durdurur)
if {[info exists ::env(LIB_DB)]} { set LIB_DB $::env(LIB_DB) } else {
  set LIB_DB "./dc_scripts/sc12mc_cln40g_base_rvt_c40_ss_typical_max_0p81v_125c.db"
  if {![file exists $LIB_DB]} { set LIB_DB "./dc/sc12mc_cln40g_base_rvt_c40_ss_typical_max_0p81v_125c.db" }
}
# ----------------------------------------------------------------------------

set XHEEP [pwd]
proc envd {name def} {
    if {[info exists ::env($name)]} { return $::env($name) } else { return $def }
}
set TOP    [envd TOP    "cv32e40px_fp_wrapper"]
set CLK_NS [envd CLK_NS 10.0]
set L      [envd L      ""]
set POLICY [envd POLICY ""]
set MAXN   [envd MAXN   ""]
set MAXB   [envd MAXB   ""]
set GEMV   [envd GEMV   ""]   ;# 1 -> synthesize the accelerator in lean GEMV mode (GEMV_ONLY=1)

# GUARD: never silently fall back to gtech (which reports area = 0).
if {![file exists $LIB_DB]} {
    puts "FATAL: standard-cell library not found: $LIB_DB"
    puts "       Edit LIB_DB at the top of dc_scripts/synth_area.tcl to point at the .db on THIS server."
    exit 1
}

# ---- per-TOP: filelist, analyze defines, elaborate params, run-id suffix ----
set FLIST   "dc_scripts/rtl_core.f"
set DEFINES {SYNTHESIS}
set PARAMS  ""
set SUF     ""
switch $TOP {
    cv32e40px_fp_wrapper {
        # WHOLE FPU (ADDMUL FMA + DIVSQRT + NONCOMP + CONV). FPU_ADDMUL_LAT is a real
        # module parameter -> sweep L with no file edits. The FMA-alone number for the
        # same L is read from this run's *_hier.rpt (fpnew_fma_multi row).
        if {$L ne ""} { set PARAMS "FPU_ADDMUL_LAT=$L"; set SUF "_L$L" }
    }
    fpnew_fma_multi {
        # OPTIONAL standalone FMA (the module the wrapper actually instantiates: ADDMUL
        # MERGED, FP32-only mask=16, PipeConfig=AFTER=1). Use only as a cross-check of the
        # hier number. If DC rejects the typed FpFmtConfig override, use the RTL wrapper
        # trick in dc/README.md instead.
        set p "FpFmtConfig=16, PipeConfig=1"
        if {$L ne ""} { set p "$p, NumPipeRegs=$L"; set SUF "_L$L" }
        set PARAMS $p
    }
    fpnew_fma {
        # Lean single-format FP32 FMA (NOT the one the design uses; PARALLEL only).
        set p "FpFormat=0, PipeConfig=1"
        if {$L ne ""} { set p "$p, NumPipeRegs=$L"; set SUF "_L$L" }
        set PARAMS $p
    }
    dma_apu_arbiter {
        # ARB_POLICY selects the policy via a conditional generate, so only the chosen
        # policy elaborates -> the per-policy area is honest. Weights matter only for P2
        # and do NOT change area (fixed 10-bit datapath), so one P2 run suffices.
        if {$POLICY ne ""} {
            if {$POLICY == 2} {
                set PARAMS "ARB_POLICY=2, W_CPU=4, W_ACC0=2, W_ACC1=1"
            } else {
                set PARAMS "ARB_POLICY=$POLICY"
            }
            set SUF "_P$POLICY"
        }
    }
    dma_fp_dot_accel_is {
        # CURRENT accelerator (batched GEMM). dc_scripts/rtl_accel_is.f OMITS tb/xbuf_ram.sv on
        # purpose, so the u_xbuf input buffer links as a BLACK BOX (area 0): the reported
        # area is the accelerator's LOGIC only. The buffer itself is a 32 KB SRAM macro,
        # reported separately (memory compiler / estimate) -- never as flip-flops.
        # ("unresolved reference xbuf_ram" warnings during link are EXPECTED, not errors.)
        set FLIST "dc_scripts/rtl_accel_is.f"
        set pl {}
        if {$MAXN ne ""} { lappend pl "MAXN=$MAXN"; append SUF "_N$MAXN" }
        if {$MAXB ne ""} { lappend pl "MAXB=$MAXB"; append SUF "_B$MAXB" }
        if {$GEMV ne ""} { lappend pl "GEMV_ONLY=$GEMV"; append SUF "_G$GEMV" }
        set PARAMS [join $pl ", "]
    }
    dma_fp_dot_accel_pipe {
        # PIPELINED accelerator (batched GEMM, pipelined issue). Same black-box buffer convention
        # as dma_fp_dot_accel_is: dc_scripts/rtl_accel_pipe.f OMITS tb/xbuf_ram.sv -> u_xbuf links as a
        # black box (area 0) -> LOGIC-only area. Buffer reported separately as a 32 KB SRAM macro.
        set FLIST "dc_scripts/rtl_accel_pipe.f"
        set pl {}
        if {$MAXN ne ""} { lappend pl "MAXN=$MAXN"; append SUF "_N$MAXN" }
        if {$MAXB ne ""} { lappend pl "MAXB=$MAXB"; append SUF "_B$MAXB" }
        set PARAMS [join $pl ", "]
    }
    dma_fp_dot_accel {
        # OLD buffer-less accelerator (kept for reference).
        set FLIST "dc_scripts/rtl_accel.f"
    }
    cv32e40px_top {
        set DEFINES {SYNTHESIS COPROC_FPU_SHARE}
        set PARAMS  "FPU=1"
    }
    default { }
}
set RUN [envd RUN_LABEL "${TOP}${SUF}"]   ;# caller can force the report label (e.g. add a clock tag)

# ------------------------------- library ------------------------------------
set target_library    $LIB_DB
set synthetic_library "dw_foundation.sldb"
set link_library      [concat "*" $LIB_DB $synthetic_library]
define_design_lib WORK -path ./dc_work/$RUN          ;# per-RUN work dir -> no clash across sweep
set search_path [concat $search_path \
    $XHEEP/hw/vendor/pulp_platform/common_cells/include \
    $XHEEP/hw/vendor/pulp_platform/register_interface/include ]
file mkdir dc_reports

# ------------------------------ read RTL ------------------------------------
set fh [open $XHEEP/$FLIST r]
set files {}
foreach line [split [read $fh] "\n"] {
    set line [string trim $line]
    if {$line ne "" && ![string match "#*" $line]} { lappend files $XHEEP/$line }
}
close $fh

analyze -format sverilog -define $DEFINES $files
if {$PARAMS ne ""} { elaborate $TOP -parameters $PARAMS } else { elaborate $TOP }
# NOTE: after a PARAMETERIZED elaborate, DC renames the design and makes the new name
# current. Re-selecting by the base name would fail, so only do it when NOT parameterized.
if {$PARAMS eq ""} { current_design $TOP }
link
check_design
uniquify

# ----------------------------- constraints ----------------------------------
create_clock -name clk -period $CLK_NS [get_ports clk_i]
set_max_area 0

# ----------------------------- synthesize -----------------------------------
compile_ultra
compile_ultra -incremental

# ------------------------------- reports ------------------------------------
report_area -hierarchy > dc_reports/area_${RUN}_hier.rpt
report_area            > dc_reports/area_${RUN}.rpt
report_reference       > dc_reports/refs_${RUN}.rpt
report_timing          > dc_reports/timing_${RUN}.rpt

# Robust area read: parse the report just written. (The design 'area' attribute name/query
# varies across DC versions -- get_attribute [current_design] area warns on some -- but
# "Total cell area:" in the report is stable.)
set A ""
if {[catch {open dc_reports/area_${RUN}.rpt r} fa] == 0} {
    foreach ln [split [read $fa] "\n"] {
        if {[regexp {Total cell area:\s*([0-9.]+)} $ln -> m]} { set A $m }
    }
    close $fa
}
set F [sizeof_collection [all_registers]]
# GUARD: empty/0 area means unmapped (library not applied) -> shout, don't fail silently.
if {$A eq "" || $A == 0} {
    puts "!!!!!! $RUN : area empty/0 -> design UNMAPPED (check library). See log. !!!!!!"
} else {
    puts "==== $RUN : Total cell area = $A , FFs = $F ===="
}
exit

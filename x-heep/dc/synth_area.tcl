#=============================================================================
# Area synthesis of ONE module for the shared-FMA thesis (Design Compiler).
#
# Driven per-module by dc/run_area.sh (fresh dc_shell each -> zero state mixing).
# You normally do NOT run this by hand; use:   bash dc/run_area.sh
# To run a single module by hand:              TOP=dma_apu_arbiter dc_shell -f dc/synth_area.tcl
#
# Edit only LIB_DB below (once). TOP/CLK_NS come from the environment.
#=============================================================================

# ----------------------------- USER SETTING ---------------------------------
set LIB_DB "./dc/sc12mc_cln40g_base_rvt_c40_ss_typical_max_0p81v_125c.db"
# ----------------------------------------------------------------------------

set XHEEP  [pwd]
set TOP    [expr {[info exists env(TOP)]    ? $env(TOP)    : "cv32e40px_fp_wrapper"}]
set CLK_NS [expr {[info exists env(CLK_NS)] ? $env(CLK_NS) : 10.0}]

# GUARD: never silently fall back to gtech (which reports area = 0).
if {![file exists $LIB_DB]} {
    puts "FATAL: standard-cell library not found: $LIB_DB (run from x-heep root, fix LIB_DB)"
    exit 1
}

# Per-TOP: filelist, analyze defines, elaborate params
set FLIST   "dc/rtl_core.f"
set DEFINES {SYNTHESIS}
set PARAMS  ""
switch $TOP {
    dma_fp_dot_accel     { set FLIST "dc/rtl_accel.f" }
    cv32e40px_fp_wrapper { }
    dma_apu_arbiter      { }
    cv32e40px_top        { set DEFINES {SYNTHESIS COPROC_FPU_SHARE}; set PARAMS "FPU=1" }
    default              { }
}

# ------------------------------- library ------------------------------------
set target_library    $LIB_DB
set synthetic_library "dw_foundation.sldb"
set link_library      [concat "*" $LIB_DB $synthetic_library]
define_design_lib WORK -path ./dc/work/$TOP          ;# per-TOP work dir -> no clash
set search_path [concat $search_path \
    $XHEEP/hw/vendor/pulp_platform/common_cells/include \
    $XHEEP/hw/vendor/pulp_platform/register_interface/include ]
file mkdir dc/reports

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
current_design $TOP
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
report_area -hierarchy > dc/reports/area_${TOP}_hier.rpt
report_area            > dc/reports/area_${TOP}.rpt
report_reference       > dc/reports/refs_${TOP}.rpt
report_timing          > dc/reports/timing_${TOP}.rpt

set A [get_attribute [current_design] area]
set F [sizeof_collection [all_registers]]
# GUARD: 0 area means unmapped (library not applied) -> shout, don't fail silently.
if {$A == 0} {
    puts "!!!!!! $TOP : area = 0  -> design UNMAPPED (check library). See log. !!!!!!"
} else {
    puts "==== $TOP : Total cell area = $A , FFs = $F ===="
}
exit

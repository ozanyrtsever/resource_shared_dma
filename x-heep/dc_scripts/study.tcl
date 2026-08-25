# Tek run'lik sentez. Env: RUN TOP CLK DEFS PARAMS FLIST
# AMAC: bizim sistemin ALAN + FMAX'i. cv32e40px_top'u (share, arbiter iceride) HIYERARSIYI KORUYARAK
#   sentezle -> report_area -hierarchy'den FMA / FPU / arbiter / core AYRISTIR. accel ayri (kendi top'u).
# Konvansiyon: clk = $CLK ns (0.1 ile basla, WNS'e gore ayarla) + 0.5 ns IO delay.
#   Alan ve fmax AYNI run'dan.  fmax = 1000/(CLK - WNS).
# .db yolu: env LIB_DB > dc_scripts/ > dc/ (server'da nerede ise). Gerekirse: env LIB_DB=/yol/....db ...
set LIB ""
if {[info exists ::env(LIB_DB)]} { set LIB $::env(LIB_DB) } else {
  foreach c {./dc_scripts/sc12mc_cln40g_base_rvt_c40_ss_typical_max_0p81v_125c.db \
             ./dc/sc12mc_cln40g_base_rvt_c40_ss_typical_max_0p81v_125c.db} {
    if {[file exists $c]} { set LIB $c; break }
  }
}
if {$LIB eq "" || ![file exists $LIB]} { puts "FATAL: .db bulunamadi -> env LIB_DB=/yol/....db ver"; exit 1 }

set target_library $LIB
set link_library   "* $LIB dw_foundation.sldb"
set search_path [concat $search_path \
    hw/vendor/pulp_platform/common_cells/include \
    hw/vendor/pulp_platform/register_interface/include]
define_design_lib WORK -path ./dc_work/$env(RUN)

# filelist'i oku (bos satir ve # yorumlarini atla)
set fh [open $env(FLIST) r]
set files {}
foreach l [split [read $fh] "\n"] {
    set l [string trim $l]
    if {$l ne "" && ![string match "#*" $l]} { lappend files $l }
}
close $fh

# NOT: DEFS bosluklu olabilir ("SYNTHESIS COPROC_FPU_SHARE ARB_POLICY_SEL=0") -> LISTE olarak ver (split),
#      yoksa DC tek makro sanip COPROC_FPU_SHARE'i uygulamaz.
analyze -format sverilog -define [split $env(DEFS)] $files
if {$env(PARAMS) ne ""} {
    elaborate $env(TOP) -parameters $env(PARAMS)
} else {
    elaborate $env(TOP)
}
link

# HIYERARSIYI KORU: asil is compile'daki -no_autoungroup'ta. Ek guvence: bu modulleri ungroup'tan
# muaf tut. Parametreli olduklari icin isimleri mangle olur (fpnew_fma_multi_<suffix>) -> WILDCARD +
# -quiet (bulamazsa hata basmaz) + bos-koleksiyon guard (CMD-036 olmasin).
set _keep [get_designs -quiet {*fpnew_fma* *fp_wrapper* *dma_apu_arbiter* *cv32e40px_core*}]
if {[sizeof_collection $_keep] > 0} { set_ungroup $_keep false }

create_clock -name clk -period $env(CLK) [get_ports clk_i]
set_false_path -from [get_ports rst_ni]            ;# async reset -> timing'den cikar (gercek kritik yol gorunsun)
set_input_delay  0.5 -clock clk \
    [remove_from_collection [all_inputs] [get_ports {clk_i rst_ni}]]
set_output_delay 0.5 -clock clk [all_outputs]
set_max_area 0

# -no_autoungroup -no_boundary_optimization: hiyerarsiyi ERITME -> report_area -hier ayristirilabilir kalir
if {[catch {compile_ultra -no_autoungroup -no_boundary_optimization}]} { compile }

# cikti klasoru: study.sh RPT env'i ile clock-basina ayri klasor verir (uzerine yazmasin)
set RPT "dc_reports"
if {[info exists ::env(RPT)]} { set RPT $::env(RPT) }
report_area -hierarchy > $RPT/area_$env(RUN).rpt
report_timing -nosplit > $RPT/timing_$env(RUN).rpt          ;# global worst (core'un kritik yolu)

# FMA'YA OZEL kritik yol: FMA'dan gecen en kotu yollar (global worst FMA'da olmasa bile).
# -through = FMA hucresinden gecen yollar -> FMA'nin bu tasarimdaki kritik yolu/gecikmesi.
set fma [get_cells -quiet -hier -filter "ref_name =~ fpnew_fma_multi*"]
if {[sizeof_collection $fma] > 0} {
  report_timing -nosplit -through $fma -max_paths 5 > $RPT/timing_fma_$env(RUN).rpt
} else {
  echo "UYARI: fpnew_fma_multi bulunamadi -> FMA-ozel timing atlandi (bu run'da FMA yok?)"
}
exit

# power.tcl -- tek (blok, faz) icin DC power raporu. gls.sh power cagirir.
# Env: NAME NETLIST SDC TOP SAIF INST RPT  (LIB_DB opsiyonel; study.tcl ile ayni .db arama)
#   NETLIST: study.tcl'nin yazdigi gate-level netlist  (hiyerarsi korunmus)
#   SDC    : ayni run'in write_sdc'si -> clk 3.845 ns (260 MHz) -> dinamik guc bu frekansta
#   SAIF   : saif_ctrl.sv'nin o faz icin yazdigi toggle dosyasi
#   INST   : SAIF hiyerarsisinde TOP'a karsilik gelen instance yolu (tb_top/.../cv32e40px_top_i/u_gl)
# Cikti: $RPT/power_$NAME.rpt (report_power -hierarchy: CPU / FPU / FMA / arbiter ayri satirlar),
#        $RPT/power_${NAME}_total.rpt, $RPT/saif_$NAME.rpt (annotasyon yuzdesi -> %90+ olmali)
set LIB ""
if {[info exists ::env(LIB_DB)]} { set LIB $::env(LIB_DB) } else {
  foreach c {./dc_scripts/sc12mc_cln40g_base_rvt_c40_ss_typical_max_0p81v_125c.db \
             ./dc/sc12mc_cln40g_base_rvt_c40_ss_typical_max_0p81v_125c.db} {
    if {[file exists $c]} { set LIB $c; break }
  }
}
if {$LIB eq "" || ![file exists $LIB]} { puts "FATAL: .db bulunamadi -> env LIB_DB=/yol/....db ver"; exit 1 }
set target_library $LIB
set link_library   "* $LIB"

set NAME $env(NAME)
set RPT  $env(RPT)
define_design_lib WORK -path ./dc_work/power_$NAME

read_verilog $env(NETLIST)
current_design $env(TOP)
link                                   ;# accel'de xbuf_ram cozulmez (black-box, sentezdeki gibi) -> uyari normal
read_sdc $env(SDC)                     ;# create_clock 3.845 ns -> guc bu saatte

# SAIF: netlist tel isimleri simule edilen netlist'le birebir -> yuksek annotasyon beklenir
read_saif -input $env(SAIF) -instance_name $env(INST) -verbose
report_saif -hier -missing > $RPT/saif_$NAME.rpt

report_power -analysis_effort medium -nosplit > $RPT/power_${NAME}_total.rpt
report_power -analysis_effort medium -hierarchy -levels 4 -nosplit > $RPT/power_$NAME.rpt

# 2) WLM'SIZ rapor (sadece pin kapasitansi). Sebep: sentez netlist'inde saat agaci yok (CTS oncesi); ICG cikisi
#    ~3K flop CK pinine TEK net -> wire-load modeli fanout'u ekstrapole edip ~400 pF biciyor (gercek CK pin toplami
#    ~5 pF) -> sleep_unit_i/core_clock_gate_i satiri ~65 mW gibi anlamsiz cikiyor. WLM'siz rapor saat neti icin adil
#    ALT SINIR (yalniz CK pinleri), diger netler icin "kablosuz" alt sinir verir. Iki rapor birlikte araligi gosterir.
remove_wire_load_model
set_app_var auto_wire_load_selection false
report_power -analysis_effort medium -nosplit > $RPT/power_${NAME}_nowlm_total.rpt
report_power -analysis_effort medium -hierarchy -levels 4 -nosplit > $RPT/power_${NAME}_nowlm.rpt
puts "POWER yazildi: $RPT/power_$NAME.rpt  (+ _nowlm)"
exit

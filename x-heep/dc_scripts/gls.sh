#!/bin/bash
# gls.sh -- HIBRIT gate-level sim (VCS) + faz-basina SAIF + DC power.  TB/RTL DEGISMEZ.
#   SoC'nin tamami RTL (bellek, DMA, bus, UART = uyarici ortam); sadece iki blok NETLIST:
#     cv32e40px_top  -> dc_reports/clk3p845/netlist_share_L0.v  (CPU + FPU/FMA + arbiter, hiyerarsi korunmus)
#     accel (pipe)   -> dc_reports/clk3p845/netlist_accel.v     (coproc; xbuf_ram black-box -> RTL'i kalir)
#   DC parametreli top'lari mangle isimle yazar (cv32e40px_top_FPU1_FPU_ADDMUL_LAT0 ...) -> RTL isimli ince
#   wrapper (RTL basligi + `.*`) ile baglanir; wrapper'lari bu script RTL basligindan uretir.
#
# LOKALDE (fusesoc + RISC-V toolchain burada):  make vcs-build (vcs yok -> sadece .scr uretir, hata normal)
#                                              make app PROJECT=power_probe ARCH=rv32imfc_zicsr
#   sonra rsync: x-heep/ (her zamanki) + build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/sim-vcs/ (scr + generated/)
# SERVER'DA (vcs + dcnxt_shell PATH'te), x-heep kokunden:
#   bash dc_scripts/gls.sh build-rtl # tum-RTL VCS (+monitor) derle  -> build/.../sim-vcs/rtl_sim
#   bash dc_scripts/gls.sh run-rtl   # power_probe kos (RTL)          -> sayilar Verilator ile ayni mi? (SAIF yok)
#   bash dc_scripts/gls.sh build     # hibrit GLS derle               -> build/.../sim-vcs/gls_sim
#   bash dc_scripts/gls.sh run       # power_probe kos (GLS)          -> build/.../sim-vcs/saif/{core,acc}_<FAZ>.saif
#   bash dc_scripts/gls.sh power     # DC report_power -hierarchy, faz x {core,acc} -> dc_reports/power/
# Onkosul (server): hucre modelleri dc_scripts/pdk/ icinde: sc12mc_cln40g_base_rvt_c40.v + *_udp.v (env PDK=... ile degistir)
# .scr'deki lokal mutlak yollar (/home/ozan/...) bu kokle degistirilir (env SRC_ROOT ile ayarlanabilir).
set -e
SRC_ROOT=${SRC_ROOT:-/home/ozan/thesis/resource_shared_dma/x-heep}
CMD=${1:-help}
ROOT=$(pwd)
SIM=$ROOT/build/openhwgroup.org_systems_core-v-mini-mcu_1.0.5/sim-vcs
SCR=$SIM/openhwgroup.org_systems_core-v-mini-mcu_1.0.5.scr
RTL_SIM=$SIM/openhwgroup.org_systems_core-v-mini-mcu_1.0.5
NET=${NET:-$ROOT/dc_reports/clk3p845}
PDK=${PDK:-$ROOT/dc_scripts/pdk}
GLS=$SIM/gls
CORE_TOP=cv32e40px_top_FPU1_FPU_ADDMUL_LAT0
ACC_TOP=dma_fp_dot_accel_pipe_MAXN1024_MAXB8
CORE_RTL=$ROOT/hw/vendor/xheep/cv32e40px/rtl/cv32e40px_top.sv
ACC_RTL=$ROOT/tb/dma_fp_dot_accel_pipe.sv
HEX=$ROOT/sw/build/main.hex
MAP=$ROOT/sw/build/main.map
CORE_INST=tb_top/testharness_i/x_heep_system_i/core_v_mini_mcu_i/cpu_subsystem_i/cv32e40px_xif_wrapper_i/cv32e40px_top_i/u_gl
ACC_INST=tb_top/testharness_i/dma_fp_dot_accel_i/u_gl
PHASES="IDLE CPU_INF COPROC CPU_MAC SHARED"

need() { [ -e "$1" ] || { echo "YOK: $1   ($2)"; exit 1; }; }

# RTL basligi (module ... parametre + port listesi ... ");") + netlist top'unu .* ile baglayan govde
gen_wrap() {  # <rtl.sv> <rtl_modul> <netlist_top> <cikti.sv>
  awk -v m="$2" '$0 ~ "^module "m"([ #(]|$)" {f=1} f {print} f && /^\);/ {exit}' "$1" > "$4"
  grep -q "^module $2" "$4" || { echo "wrapper uretilemedi: $2 basligi $1 icinde bulunamadi"; exit 1; }
  printf '  // GLS wrapper (gls.sh uretti): RTL arayuzu -> DC netlist top (isimler DC tarafindan mangle edildi)\n  %s u_gl (.*);\nendmodule\n' "$3" >> "$4"
}

# fusesoc'un vcs secenekleri (core-v-mini-mcu.core vcs_options) + bizim define'lar. -kdb/-race yok (hiz).
# -debug_access+pp: SAIF toggle bolgesi icin SART (yoksa SAIF bos yazilir, uyari bile yok - selftest ile bulundu).
# -tf_sv_string: PLI gorevlerine SV string gecirebilmek icin ($toggle_report dosya adi).
VCS_OPTS="-full64 -sverilog -override_timescale=1ns/1ps -assert disable_cover -assert svaext -fgp -notice -ntb_opts error
          -xlrm uniq_prior_final +lint=TFIPC-L +define+COPROC_FPU_SHARE +vcs+initreg+random -tf_sv_string -debug_access+pp"
# -lutil (uartdpi.c: openpty) nesnelerden ONCE geliyor; Ubuntu linker'i --as-needed ile onu atar -> --no-as-needed sart
VCS_C='-CFLAGS -pthread -LDFLAGS -pthread -LDFLAGS -Wl,--no-as-needed -LDFLAGS -lutil'

# .scr'yi bu makineye uyarla: lokal mutlak yollar -> bu kok
fix_scr() { sed "s#$SRC_ROOT#$ROOT#g" "$SCR"; }

case $CMD in
build-rtl)
  need "$SCR" "lokalde make vcs-build + sim-vcs/ rsync"
  mkdir -p "$GLS"
  { fix_scr; echo "$ROOT/dc_scripts/gls/saif_ctrl.sv"; } > "$GLS/rtl.scr"
  cd "$SIM"
  vcs $VCS_OPTS $VCS_C -top tb_top -top saif_ctrl -f gls/rtl.scr -o rtl_sim -l rtl_build.log
  echo "OK: $SIM/rtl_sim"
  ;;

build)
  need "$SCR" "lokalde make vcs-build + sim-vcs/ rsync"
  need "$NET/netlist_share_L0.v" "study.sh WRITE_NETLIST=1"; need "$NET/netlist_accel.v" "study.sh WRITE_NETLIST=1"
  UDP=$(ls "$PDK"/*udp*.v 2>/dev/null | head -1); CELLS=$(ls "$PDK"/*.v 2>/dev/null | grep -v udp | head -1)
  [ -n "$UDP" ] && [ -n "$CELLS" ] || { echo "hucre modelleri yok: $PDK/ icine sc12mc_cln40g_base_rvt_c40.v ve *_udp.v kopyala"; exit 1; }
  grep -q "^module $CORE_TOP" "$NET/netlist_share_L0.v" || { echo "core netlist top $CORE_TOP degil:"; grep -m3 "^module" "$NET/netlist_share_L0.v"; exit 1; }
  grep -q "^module $ACC_TOP"  "$NET/netlist_accel.v"    || { echo "accel netlist top $ACC_TOP degil:";  grep -m3 "^module" "$NET/netlist_accel.v";    exit 1; }
  mkdir -p "$GLS"

  # 1) wrapper'lar
  gen_wrap "$CORE_RTL" cv32e40px_top         "$CORE_TOP" "$GLS/wrap_core.sv"
  gen_wrap "$ACC_RTL"  dma_fp_dot_accel_pipe "$ACC_TOP"  "$GLS/wrap_accel.sv"

  # 2) dosya listesi: fusesoc .scr'den netlist'lenen RTL'i cikar (paketler kalir), netlist + hucre + wrapper + monitor ekle
  awk '
    /\/xheep\/cv32e40px\/rtl\// && !/_pkg\.sv$/ {next}   # core RTL   -> netlist_share_L0 (pkg dosyalari kalir)
    /\/fpnew\/src\//            && !/_pkg\.sv$/ {next}   # FPU RTL    -> netlist icinde
    /tb\/dma_fp_dot_accel_pipe\.sv$/            {next}   # accel RTL  -> netlist_accel   (tb/xbuf_ram.sv KALIR: black-box)
    {print}' <(fix_scr) > "$GLS/gls.scr"
  # DC parametresiz alt modulleri AYNI isimle yazar -> kalan RTL'de ayni isimli modul varsa (cift tanim) o dosyayi da cikar
  grep -h "^module" "$NET/netlist_share_L0.v" "$NET/netlist_accel.v" | awk '{print $2}' | sed 's/[(;].*//' | sort -u > "$GLS/net_mods.txt"
  PAT=$(paste -sd'|' "$GLS/net_mods.txt")
  : > "$GLS/gls.scr.tmp"; : > "$GLS/dropped_dup.txt"
  while IFS= read -r line; do
    case "$line" in
      ""|+*|-*) echo "$line" >> "$GLS/gls.scr.tmp"; continue;;
    esac
    f="$SIM/$line"; [ -f "$f" ] || f="$line"
    if [ -f "$f" ] && grep -qE "^[[:space:]]*module[[:space:]]+($PAT)([[:space:]#(;]|$)" "$f"; then
      echo "$line" >> "$GLS/dropped_dup.txt"
    else
      echo "$line" >> "$GLS/gls.scr.tmp"
    fi
  done < "$GLS/gls.scr"
  mv "$GLS/gls.scr.tmp" "$GLS/gls.scr"
  { echo "$UDP"; echo "$CELLS"; echo "$NET/netlist_share_L0.v"; echo "$NET/netlist_accel.v"
    echo "$GLS/wrap_core.sv"; echo "$GLS/wrap_accel.sv"; echo "$ROOT/dc_scripts/gls/saif_ctrl.sv"; } >> "$GLS/gls.scr"
  echo "gls.scr: $(grep -c . "$GLS/gls.scr") satir; cift-tanim nedeniyle cikarilan RTL: $(wc -l < "$GLS/dropped_dup.txt") (bkz $GLS/dropped_dup.txt)"

  # 3) VCS: ayni secenekler + hucre modeli (ARM_UD_MODEL) + sifir-gecikmeli fonksiyonel GLS (specify/timing check yok)
  cd "$SIM"
  vcs $VCS_OPTS $VCS_C -top tb_top -top saif_ctrl -f gls/gls.scr -o gls_sim \
      +define+ARM_UD_MODEL +notimingcheck +nospecify -l gls_build.log
  echo "OK: $SIM/gls_sim"
  ;;

run|run-rtl)
  need "$HEX" "make app PROJECT=power_probe ARCH=rv32imfc_zicsr"; need "$MAP" "main.map yok"
  ADDR=$(awk '$2=="phase_flag"{print $1; exit}' "$MAP"); ADDR=${ADDR#0x}
  [ -n "$ADDR" ] || { echo "phase_flag main.map'te yok (power_probe derlendi mi?)"; exit 1; }
  grep -q power_probe "$ROOT/sw/build/CMakeCache.txt" 2>/dev/null || echo "UYARI: sw/build power_probe gibi gorunmuyor"
  cd "$SIM"
  if [ "$CMD" = run ]; then
    need gls_sim "once: gls.sh build"
    [ -d saif ] && mv saif "saif_prev_$(date +%m%d_%H%M%S)"   # eski (kismi) SAIF'leri silme, kenara al
    mkdir -p saif
    # +vcs+initreg+0: tum flop'lar 0'dan baslar (random: pad-mux reg'i t=0'da rastgele -> pad_cell_input $stop)
    ./gls_sim +firmware="$HEX" +phase_addr=$ADDR +saif +saif_dir=saif +vcs+initreg+0 +vcs+lic+wait -l gls_run.log
    echo "---- fazlar (monitor) ----"; grep "^\[SAIF\]" gls_run.log || true
    echo "---- program (UART) ----";   grep -E "PHASE|^\[[0-9]\]|bit-exact" uart0.log || true
    echo "---- SAIF ----"; ls -la saif
  else
    need rtl_sim "once: gls.sh build-rtl"
    ./rtl_sim +firmware="$HEX" +phase_addr=$ADDR +vcs+initreg+0 +vcs+lic+wait -l rtl_run.log
    echo "---- fazlar (monitor, RTL) ----"; grep "^\[SAIF\]" rtl_run.log || true
    echo "---- program (UART, tum-RTL VCS) ----"; grep -E "PHASE|^\[[0-9]\]|bit-exact" uart0.log || true
  fi
  ;;

power)
  RPT=$ROOT/dc_reports/power; mkdir -p "$RPT"
  SAIFDIR=${SAIFDIR:-$SIM/saif}          # env SAIFDIR=... ile baska klasor (orn. kismi kosunun saif_prev_*)
  for PH in $PHASES; do for BLK in core acc; do
    case $BLK in
      core) NL=netlist_share_L0; TOP=$CORE_TOP; INST=$CORE_INST;;
      acc)  NL=netlist_accel;    TOP=$ACC_TOP;  INST=$ACC_INST;;
    esac
    SAIF=$SAIFDIR/${BLK}_${PH}.saif
    [ -s "$SAIF" ] || { echo "atla: $SAIF yok"; continue; }
    if [ -s "$RPT/power_${BLK}_${PH}.rpt" ] && grep -q "Total Dynamic Power" "$RPT/power_${BLK}_${PH}.rpt"; then
      echo "  ${BLK}_${PH}: SKIP (zaten var)"; continue; fi
    echo "############  power ${BLK} ${PH}  ############"
    NAME=${BLK}_${PH} NETLIST=$NET/$NL.v SDC=$NET/$NL.sdc TOP=$TOP SAIF=$SAIF INST=$INST RPT=$RPT \
      dcnxt_shell -f dc_scripts/power.tcl 2>&1 | tee "$RPT/log_${BLK}_${PH}.txt" \
      | grep --line-buffered -iE "error|annotated|Total Dynamic|Cell Leakage|fatal" || true
  done; done
  echo; echo "===== OZET (Switch Int Leak(uW) Total(mW) %) — WLM ve WLM'siz ====="
  KEY='^ *(cv32e40px_top|fpu_gen_fp_wrapper_i|gen_operation_groups_0__i_opgroup_block|fpu_gen_arb_i|core_i|sleep_unit_i|register_file_i|dma_fp_dot_accel_pipe)'
  for V in "" "_nowlm"; do for BLK in core acc; do for PH in $PHASES; do
    f=$RPT/power_${BLK}_${PH}${V}.rpt; [ -s "$f" ] || continue
    echo "--- $BLK $PH ${V:-_wlm}"
    grep -E "$KEY" "$f" | sed -E 's/ \([^)]*\)//; s/^/    /'      # parantezli uzun modul adlarini kirp
  done; done; done
  echo "raporlar: $RPT/power_<blok>_<faz>[_nowlm].rpt (hiyerarsik), *_total.rpt, saif_<blok>_<faz>.rpt (annotasyon)"
  ;;

*) sed -n 2,17p "$0";;
esac

// saif_ctrl.sv -- TESTBENCH monitoru (sentezlenmez, tasarima dokunmaz). gls.sh bunu ikinci top olarak
// derler: vcs -top tb_top -top saif_ctrl. Normal Verilator/VCS build'lerinde YOK.
//
// Is: power_probe'un faz isaretlerini yakala, faz basina SAIF yaz.
//   Program her fazin basinda `phase_flag = id`, sonunda `phase_flag = 0` store'u atar. Bu store core'un
//   data-bus PORTLARINDA gorunur (data_req_o & data_gnt_i & data_we_o & data_addr_o == +phase_addr).
//   Portlar RTL'de de gate-level netlist'te de ayni isimde -> ayni monitor iki build'de de calisir.
//   +saif verilirse: faz basi $toggle_start, faz sonu $toggle_stop + iki scope icin $toggle_report:
//       <saif_dir>/core_<FAZ>.saif   (cv32e40px_top: CPU + FPU/FMA + arbiter)
//       <saif_dir>/acc_<FAZ>.saif    (coproc)
//   Her durumda faz suresini cycle olarak basar -> UART'taki "dur=" ile birebir tutmali (dogrulama).
// VCS NOTLARI (selftest ile dogrulandi): (1) PLI gorevleri SV `string` tipini KABUL ETMEZ -> dosya adlari
//   reg-vektor metin ($sformat), scope'lar literal. (2) tek $set_toggle_region (ikinci cagri yok sayilir)
//   -> bolge = testharness, raporlar scope ile ayrilir. (3) ilk fazdan once $toggle_reset uyari verir -> atla.
`timescale 1ns/1ps
module saif_ctrl;
  `define CORE tb_top.testharness_i.x_heep_system_i.core_v_mini_mcu_i.cpu_subsystem_i.cv32e40px_xif_wrapper_i.cv32e40px_top_i

  // Dosya adlari SV `string` (+ derlemede -tf_sv_string): selftest ile dogrulanan TEK calisan form.
  // ({..} birlestirme ifadesi ve NUL-dolgulu reg-vektor "Invalid first argument" verir.)
  string  names[6] = '{"none", "IDLE", "CPU_INF", "COPROC", "CPU_MAC", "SHARED"};
  string  outdir, fcore, facc;
  logic [31:0] paddr;
  int     do_saif = 0, cur = 0, reported = 0;
  longint cyc = 0, c0 = 0;
  time    t0;

  initial begin
    if (!$value$plusargs("phase_addr=%h", paddr)) begin
      $display("[SAIF] HATA: +phase_addr=<hex> verilmedi (gls.sh main.map'ten alir)");
      $finish;
    end
    if (!$value$plusargs("saif_dir=%s", outdir)) outdir = "saif";
    do_saif = $test$plusargs("saif");
    $display("[SAIF] phase_addr=0x%08h  saif=%0d  dir=%0s", paddr, do_saif, outdir);
    if (do_saif) begin
      // $set_gate_level_monitoring CAGRILMIYOR: selftest'te "off" HICBIR SEY kaydetmedi, "on" ile cagrisiz ayni;
      // netlist netleri (Verilog wire) varsayilanla kaydediliyor. Derlemede -debug_access+pp SART.
      $set_toggle_region(tb_top.testharness_i);  // tek bolge: core ve coproc'u kapsar (ikinci cagri yok sayilir)
    end
  end

  always @(posedge `CORE.clk_i) begin
    cyc++;
    if (`CORE.data_req_o && `CORE.data_gnt_i && `CORE.data_we_o && `CORE.data_addr_o == paddr) begin
      if (`CORE.data_wdata_o != 0) begin                       // faz basladi
        cur = `CORE.data_wdata_o; c0 = cyc; t0 = $time;
        $display("[SAIF] %-8s start  cyc=%0d  t=%0t", names[cur], cyc, $time);
        if (do_saif) begin
          if (reported) $toggle_reset;
          $toggle_start;
        end
      end else if (cur != 0) begin                              // faz bitti
        if (do_saif) begin
          $toggle_stop;
          fcore = $sformatf("%0s/core_%0s.saif", outdir, names[cur]);
          facc  = $sformatf("%0s/acc_%0s.saif",  outdir, names[cur]);
          $toggle_report(fcore, 1.0e-9, "tb_top.testharness_i.x_heep_system_i.core_v_mini_mcu_i.cpu_subsystem_i.cv32e40px_xif_wrapper_i.cv32e40px_top_i");
          $toggle_report(facc,  1.0e-9, "tb_top.testharness_i.dma_fp_dot_accel_i");
          reported = 1;
        end
        $display("[SAIF] %-8s end    cyc=%0d  dur=%0d cycles  (%0t)", names[cur], cyc, cyc - c0, $time - t0);
        cur = 0;
      end
    end
  end
endmodule

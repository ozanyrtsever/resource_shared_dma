// saif_selftest.sv -- saif_ctrl.sv'nin CAGRI DIZISINI birebir prova eder (30 sn).
// Dogrulanan gercekler (VCS T-2022.06, bu kurulum):
//   * -debug_access+pp SART (yoksa SAIF bos yazilir, uyari yok).  * -tf_sv_string ile SV string PLI'ya gecer.
//   * Dosya adi: `string` degisken ($sformatf) veya literal. {..} birlestirme / NUL-dolgulu reg-vektor OLMAZ.
//   * $set_gate_level_monitoring cagrilmaz ("off" hicbir sey kaydetmez; "on" == cagrisiz).
//   * Tek $set_toggle_region; alt-instance scope raporu calisir; INSTANCE agaci top'tan baslar.
//   vcs -full64 -sverilog -tf_sv_string -debug_access+pp dc_scripts/gls/saif_selftest.sv -o st_plain
//   ./st_plain +saif_dir=. ; grep -c "(NET" st_*_P1.saif st_*_P2.saif
`timescale 1ns/1ps
module st_blk(clk, q);                 // Verilog reg/wire "netlist" benzeri blok
  input clk; output q;
  reg [7:0] cnt; initial cnt = 0;
  always @(posedge clk) cnt <= cnt + 1;
  assign q = cnt[7];
endmodule

module st_top;
  reg clk; initial clk = 0;
  always #5 clk = ~clk;
  wire q1, q2;
  st_blk core_i(.clk(clk), .q(q1));    // "core" scope
  st_blk acc_i (.clk(clk), .q(q2));    // "acc" scope
endmodule

module st_ctrl;                         // saif_ctrl.sv ile ayni yapi
  string  names[3] = '{"none", "P1", "P2"};
  string  outdir, fcore, facc;
  int     reported = 0;
  initial begin
    if (!$value$plusargs("saif_dir=%s", outdir)) outdir = ".";
    $set_toggle_region(st_top);
    for (int cur = 1; cur <= 2; cur++) begin
      #100;
      if (reported) $toggle_reset;
      $toggle_start;
      #1000;
      $toggle_stop;
      fcore = $sformatf("%0s/st_core_%0s.saif", outdir, names[cur]);
      facc  = $sformatf("%0s/st_acc_%0s.saif",  outdir, names[cur]);
      $toggle_report(fcore, 1.0e-9, "st_top.core_i");
      $toggle_report(facc,  1.0e-9, "st_top.acc_i");
      reported = 1;
      $display("SELFTEST %0s -> %0s , %0s", names[cur], fcore, facc);
    end
    $finish;
  end
endmodule

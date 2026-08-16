// xbuf_ram.sv — input buffer for dma_fp_dot_accel_is, factored into its own module so a
// synthesis / area flow can treat it as an SRAM macro (a black box) instead of flip-flops.
// Functionally identical to the previous inline `x_buf`: one clocked write port (no reset;
// every location is written during LOAD before it is ever read) and one COMBINATIONAL read
// port. In silicon this maps to a 1R1W SRAM, not registers.
// verilator lint_off UNUSEDSIGNAL
module xbuf_ram #(
    parameter int unsigned MAXB = 8,      // batch dimension (buffer rows)
    parameter int unsigned MAXN = 1024,   // dot length      (buffer columns)
    parameter int unsigned BW   = 3,      // batch-index width  = (MAXB>1)?$clog2(MAXB):1
    parameter int unsigned IW   = 10      // element-index width = $clog2(MAXN)
)(
    input  logic          clk_i,
    input  logic          we_i,           // write enable (LOAD)
    input  logic [BW-1:0] wb_i,           // write batch index
    input  logic [IW-1:0] wa_i,           // write element index
    input  logic [31:0]   wd_i,           // write data
    input  logic [BW-1:0] rb_i,           // read batch index  (EXECUTE)
    input  logic [IW-1:0] ra_i,           // read element index
    output logic [31:0]   rd_o            // read data (combinational)
);
  logic [31:0] mem [MAXB][MAXN];
  always_ff @(posedge clk_i) if (we_i) mem[wb_i][wa_i] <= wd_i;  // no reset: written before read
  assign rd_o = mem[rb_i][ra_i];                                 // combinational read
endmodule

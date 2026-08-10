// dma_apu_arbiter.sv — CPU-priority FULLY-PIPELINED shared-FPU APU arbiter (thesis contribution v2).
//
// The CPU's and the DMA coprocessor's FMA ops may be IN FLIGHT SIMULTANEOUSLY in the shared FPnew
// pipeline (NO draining). The CPU has strict priority for the single 1-op/cycle issue slot; the
// coprocessor uses any slot the CPU does not.
//
// Each issued op is tagged with its owner (fpu_tag_o: 1=DMA coprocessor, 0=CPU). FPnew carries that
// tag through whichever lane the op uses and returns it WITH the result (fpu_tag_i). The response is
// routed by that returned tag, so it is correct EVEN IF FPnew's lanes retire out of order — e.g. a
// CPU fdiv/fsqrt (slow DIVSQRT lane) concurrent with the coprocessor's fmadd (fast ADDMUL lane).
//
// Requires: cv32e40px_fp_wrapper to expose fpnew's tag (apu_tag_i/apu_tag_o), and the top-level to
// wire arbiter.fpu_tag_o -> wrapper.apu_tag_i and wrapper.apu_tag_o -> arbiter.fpu_tag_i.
module dma_apu_arbiter
  import cv32e40px_apu_core_pkg::*;
(
    input  logic clk_i,
    input  logic rst_ni,
    output logic fma_active_o,

    // ---- CPU side (from the core) ----
    input  logic                           cpu_req_i,
    output logic                           cpu_gnt_o,
    input  logic [APU_NARGS_CPU-1:0][31:0] cpu_operands_i,
    input  logic [APU_WOP_CPU-1:0]         cpu_op_i,
    input  logic [APU_NDSFLAGS_CPU-1:0]    cpu_flags_i,
    output logic                           cpu_rvalid_o,
    output logic [31:0]                    cpu_rdata_o,
    output logic [APU_NUSFLAGS_CPU-1:0]    cpu_rflags_o,

    // ---- DMA coprocessor side ----
    input  logic                           dma_req_i,
    output logic                           dma_gnt_o,
    input  logic [APU_NARGS_CPU-1:0][31:0] dma_operands_i,
    input  logic [APU_WOP_CPU-1:0]         dma_op_i,
    input  logic [APU_NDSFLAGS_CPU-1:0]    dma_flags_i,
    output logic                           dma_rvalid_o,
    output logic [31:0]                    dma_rdata_o,
    output logic [APU_NUSFLAGS_CPU-1:0]    dma_rflags_o,

    // ---- FPU (FMA wrapper) side ----
    output logic                           fpu_req_o,
    input  logic                           fpu_gnt_i,
    output logic [APU_NARGS_CPU-1:0][31:0] fpu_operands_o,
    output logic [APU_WOP_CPU-1:0]         fpu_op_o,
    output logic [APU_NDSFLAGS_CPU-1:0]    fpu_flags_o,
    output logic                           fpu_tag_o,      // owner tag of the op issued this cycle
    input  logic                           fpu_rvalid_i,
    input  logic [31:0]                    fpu_rdata_i,
    input  logic [APU_NUSFLAGS_CPU-1:0]    fpu_rflags_i,
    input  logic                           fpu_tag_i       // owner tag returned WITH the result
);

  logic [2:0] inflight_q;                     // ops in the pipeline (for the FMA clock-gate only)

  // ---- Issue: CPU strict priority for the single 1-op/cycle slot; BOTH may be in flight ----
  wire issue_cpu = cpu_req_i;
  wire issue_dma = dma_req_i & ~cpu_req_i;

  assign fpu_req_o      = issue_cpu | issue_dma;
  assign fpu_operands_o = issue_dma ? dma_operands_i : cpu_operands_i;
  assign fpu_op_o       = issue_dma ? dma_op_i       : cpu_op_i;
  assign fpu_flags_o    = issue_dma ? dma_flags_i    : cpu_flags_i;
  assign fpu_tag_o      = issue_dma;          // tag the issued op with its owner (1 = DMA, 0 = CPU)
  assign fma_active_o   = dma_req_i | (inflight_q != '0);

  assign cpu_gnt_o = issue_cpu & fpu_gnt_i;
  assign dma_gnt_o = issue_dma & fpu_gnt_i;

  // ---- Response: route by the tag returned with the result (out-of-order safe) ----
  wire resp_owner = fpu_tag_i;
  assign cpu_rvalid_o = fpu_rvalid_i & ~resp_owner;
  assign dma_rvalid_o = fpu_rvalid_i &  resp_owner;
  assign cpu_rdata_o  = fpu_rdata_i;          // consumer latches only on its own rvalid
  assign dma_rdata_o  = fpu_rdata_i;
  assign cpu_rflags_o = fpu_rflags_i;
  assign dma_rflags_o = fpu_rflags_i;

  // ---- In-flight counter (drives fma_active_o for the FMA clock-gate) ----
  wire issue = fpu_req_o & fpu_gnt_i;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) inflight_q <= '0;
    else         inflight_q <= inflight_q + 3'(issue) - 3'(fpu_rvalid_i);
  end

endmodule

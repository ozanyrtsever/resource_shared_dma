// dma_apu_arbiter.sv — CPU-priority shared-FPU APU arbiter (thesis contribution).
// Muxes the CPU's and the DMA coprocessor's APU/FMA requests onto ONE shared FPnew
// FMA. CPU has strict priority; the coprocessor uses the FMA only when the CPU is
// idle, and is DRAINED (its in-flight op finishes — FPnew has no flush) before the
// CPU is served. In-flight ops never mix owners, so responses route by owner.
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
    input  logic                           fpu_rvalid_i,
    input  logic [31:0]                    fpu_rdata_i,
    input  logic [APU_NUSFLAGS_CPU-1:0]    fpu_rflags_i
);

  logic [2:0] inflight_q;                 // ops issued into the FMA but not yet returned
  logic       owner_q;                    // 0 = CPU, 1 = DMA (owner of in-flight ops)
  wire        pipe_empty = (inflight_q == '0);

  // Who may ISSUE a new op this cycle. Never mix owners: while the pipe is non-empty
  // only the current owner may issue more. When empty, CPU has priority.
  wire can_cpu = cpu_req_i & (pipe_empty | ~owner_q);
  wire can_dma = dma_req_i & ~cpu_req_i & (pipe_empty | owner_q);
  wire issue_cpu = can_cpu;               // CPU priority
  wire issue_dma = can_dma & ~can_cpu;

  // Request routing to the FMA
  assign fpu_req_o      = issue_cpu | issue_dma;
  assign fpu_operands_o = issue_dma ? dma_operands_i : cpu_operands_i;
  assign fpu_op_o       = issue_dma ? dma_op_i       : cpu_op_i;
  assign fpu_flags_o    = issue_dma ? dma_flags_i    : cpu_flags_i;
  assign fma_active_o   = dma_req_i | (inflight_q != '0);

  // Grant back to the selected requester only
  assign cpu_gnt_o = issue_cpu & fpu_gnt_i;
  assign dma_gnt_o = issue_dma & fpu_gnt_i;

  // Response routing: all in-flight ops belong to owner_q, so route by owner.
    // Aynı-çevrim (latency-0) issue+complete: owner_q henüz güncellenmedi -> issue_dma kullan
  wire resp_owner = (pipe_empty & fpu_req_o & fpu_gnt_i) ? issue_dma : owner_q;
  assign cpu_rvalid_o = fpu_rvalid_i & ~resp_owner;
  assign dma_rvalid_o = fpu_rvalid_i &  resp_owner;
  assign cpu_rdata_o  = fpu_rdata_i;      // consumer only latches on its own rvalid
  assign dma_rdata_o  = fpu_rdata_i;
  assign cpu_rflags_o = fpu_rflags_i;
  assign dma_rflags_o = fpu_rflags_i;

  wire issue = fpu_req_o & fpu_gnt_i;     // op accepted into the FMA
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      inflight_q <= '0;
      owner_q    <= 1'b0;
    end else begin
      inflight_q <= inflight_q + 3'(issue) - 3'(fpu_rvalid_i);
      if (pipe_empty && issue) owner_q <= issue_dma;   // latch owner on entering the pipe
    end
  end

endmodule

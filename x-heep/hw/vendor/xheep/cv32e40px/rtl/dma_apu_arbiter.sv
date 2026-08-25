// dma_apu_arbiter.sv — shared-FPU APU arbiter, THREE requestors (CPU + acc0 + acc1), PARAMETRIC POLICY.
// Only the ARB_POLICY-selected policy elaborates (generate) -> honest per-policy area, no dead logic.
// FPnew round-trips the 2-bit owner tag (0=CPU,1=acc0,2=acc1) with each result -> out-of-order safe.
// Invariant (all policies): bit-exactness — arbitration reorders ops BETWEEN owners only.
module dma_apu_arbiter
  import cv32e40px_apu_core_pkg::*;
#(
    parameter int unsigned ARB_POLICY = 0,   // 0=CPU-strict+accRR  1=all-RR  2=QoS-weighted
    parameter int unsigned W_CPU  = 4,       // QoS weights (ARB_POLICY==2 only)
    parameter int unsigned W_ACC0 = 1,
    parameter int unsigned W_ACC1 = 1
)(
    input  logic clk_i,
    input  logic rst_ni,
    output logic fma_active_o,

    // ---- CPU side ----
    input  logic                           cpu_req_i,
    output logic                           cpu_gnt_o,
    input  logic [APU_NARGS_CPU-1:0][31:0] cpu_operands_i,
    input  logic [APU_WOP_CPU-1:0]         cpu_op_i,
    input  logic [APU_NDSFLAGS_CPU-1:0]    cpu_flags_i,
    output logic                           cpu_rvalid_o,
    output logic [31:0]                    cpu_rdata_o,
    output logic [APU_NUSFLAGS_CPU-1:0]    cpu_rflags_o,

    // ---- acc0 (first DMA coprocessor, channel 1) ----
    input  logic                           dma_req_i,
    output logic                           dma_gnt_o,
    input  logic [APU_NARGS_CPU-1:0][31:0] dma_operands_i,
    input  logic [APU_WOP_CPU-1:0]         dma_op_i,
    input  logic [APU_NDSFLAGS_CPU-1:0]    dma_flags_i,
    output logic                           dma_rvalid_o,
    output logic [31:0]                    dma_rdata_o,
    output logic [APU_NUSFLAGS_CPU-1:0]    dma_rflags_o,

    // ---- acc1 (second DMA coprocessor, channel 2) ----
    input  logic                           dma1_req_i,
    output logic                           dma1_gnt_o,
    input  logic [APU_NARGS_CPU-1:0][31:0] dma1_operands_i,
    input  logic [APU_WOP_CPU-1:0]         dma1_op_i,
    input  logic [APU_NDSFLAGS_CPU-1:0]    dma1_flags_i,
    output logic                           dma1_rvalid_o,
    output logic [31:0]                    dma1_rdata_o,
    output logic [APU_NUSFLAGS_CPU-1:0]    dma1_rflags_o,

    // ---- FPU (FMA wrapper) side ----
    output logic                           fpu_req_o,
    input  logic                           fpu_gnt_i,
    output logic [APU_NARGS_CPU-1:0][31:0] fpu_operands_o,
    output logic [APU_WOP_CPU-1:0]         fpu_op_o,
    output logic [APU_NDSFLAGS_CPU-1:0]    fpu_flags_o,
    output logic [1:0]                     fpu_tag_o,
    input  logic                           fpu_rvalid_i,
    input  logic [31:0]                    fpu_rdata_i,
    input  logic [APU_NUSFLAGS_CPU-1:0]    fpu_rflags_i,
    input  logic [1:0]                     fpu_tag_i
);

  logic [3:0] inflight_q;
  wire  [2:0] req = {dma1_req_i, dma_req_i, cpu_req_i};   // [0]=CPU [1]=acc0 [2]=acc1
  logic [2:0] grant;                                       // one-hot; driven by the selected policy

  // ===================== policy (ONLY the selected one elaborates) =====================
  generate
    // -------- P0: CPU strict priority; the two accs round-robin the leftover slots --------
    if (ARB_POLICY == 0) begin : g_policy
      logic rr_acc_q;
      always_comb begin
        grant = 3'b000;
        if (req[0])                     grant[0] = 1'b1;         // CPU always wins
        else if (req[1] && req[2])      grant[rr_acc_q ? 2 : 1] = 1'b1;
        else if (req[1])                grant[1] = 1'b1;
        else if (req[2])                grant[2] = 1'b1;
      end
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni)                    rr_acc_q <= 1'b0;
        else if (grant[1] & fpu_gnt_i)  rr_acc_q <= 1'b1;        // acc0 served -> next prefer acc1
        else if (grant[2] & fpu_gnt_i)  rr_acc_q <= 1'b0;
      end
    end
    // -------- P1: full round-robin over the three peers (rotating priority) --------
    else if (ARB_POLICY == 1) begin : g_policy
      logic [1:0] rr_ptr_q;
      always_comb begin
        grant = 3'b000;
        unique case (rr_ptr_q)
          2'd0:    begin if (req[0]) grant[0]=1'b1; else if (req[1]) grant[1]=1'b1; else if (req[2]) grant[2]=1'b1; end
          2'd1:    begin if (req[1]) grant[1]=1'b1; else if (req[2]) grant[2]=1'b1; else if (req[0]) grant[0]=1'b1; end
          default: begin if (req[2]) grant[2]=1'b1; else if (req[0]) grant[0]=1'b1; else if (req[1]) grant[1]=1'b1; end
        endcase
      end
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni)                    rr_ptr_q <= 2'd0;
        else if (fpu_req_o & fpu_gnt_i) begin
          if      (grant[0])            rr_ptr_q <= 2'd1;
          else if (grant[1])            rr_ptr_q <= 2'd2;
          else if (grant[2])            rr_ptr_q <= 2'd0;
        end
      end
    end
    // -------- P2: QoS weighted round-robin (deficit credits; grant share ~ weights) --------
    else begin : g_policy
      localparam logic signed [9:0] WV [3] = '{10'(W_CPU), 10'(W_ACC0), 10'(W_ACC1)};
      logic signed [9:0] cred_q [3];
      logic signed [9:0] wsum;   // sum of the REQUESTING requestors' weights -> drift-free quantum
      always_comb wsum = (req[0] ? WV[0] : 10'sd0) + (req[1] ? WV[1] : 10'sd0) + (req[2] ? WV[2] : 10'sd0);

      always_comb begin
        logic signed [9:0] best; logic [1:0] bi; logic found;
        best = '0; bi = 2'd0; found = 1'b0;
        for (int i = 0; i < 3; i++)
          if (req[i] && (!found || cred_q[i] > best)) begin best = cred_q[i]; bi = 2'(i); found = 1'b1; end
        grant = 3'b000;
        if (found) unique case (bi)
          2'd0:    grant = 3'b001;
          2'd1:    grant = 3'b010;
          default: grant = 3'b100;
        endcase
      end
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) for (int i = 0; i < 3; i++) cred_q[i] <= '0;
        else if (fpu_req_o & fpu_gnt_i) begin
          for (int i = 0; i < 3; i++) begin
            if      (grant[i]) cred_q[i] <= cred_q[i] + WV[i] - wsum;
            else if (req[i])   cred_q[i] <= cred_q[i] + WV[i];
          end
        end
      end
    end
  endgenerate

  // ===================== shared datapath (policy-independent) =====================
  wire issue_cpu  = grant[0];
  wire issue_acc0 = grant[1];
  wire issue_acc1 = grant[2];

  assign fpu_req_o      = |grant;
  assign fpu_operands_o = issue_acc1 ? dma1_operands_i : issue_acc0 ? dma_operands_i : cpu_operands_i;
  assign fpu_op_o       = issue_acc1 ? dma1_op_i       : issue_acc0 ? dma_op_i       : cpu_op_i;
  assign fpu_flags_o    = issue_acc1 ? dma1_flags_i    : issue_acc0 ? dma_flags_i    : cpu_flags_i;
  assign fpu_tag_o      = issue_acc1 ? 2'd2 : issue_acc0 ? 2'd1 : 2'd0;
  assign fma_active_o   = dma_req_i | dma1_req_i | (inflight_q != '0);

  assign cpu_gnt_o  = issue_cpu  & fpu_gnt_i;
  assign dma_gnt_o  = issue_acc0 & fpu_gnt_i;
  assign dma1_gnt_o = issue_acc1 & fpu_gnt_i;

  assign cpu_rvalid_o  = fpu_rvalid_i & (fpu_tag_i == 2'd0);
  assign dma_rvalid_o  = fpu_rvalid_i & (fpu_tag_i == 2'd1);
  assign dma1_rvalid_o = fpu_rvalid_i & (fpu_tag_i == 2'd2);
  assign cpu_rdata_o  = fpu_rdata_i;  assign cpu_rflags_o  = fpu_rflags_i;
  assign dma_rdata_o  = fpu_rdata_i;  assign dma_rflags_o  = fpu_rflags_i;
  assign dma1_rdata_o = fpu_rdata_i;  assign dma1_rflags_o = fpu_rflags_i;

  wire fire = fpu_req_o & fpu_gnt_i;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) inflight_q <= '0;
    else         inflight_q <= inflight_q + 4'(fire) - 4'(fpu_rvalid_i);
  end

endmodule

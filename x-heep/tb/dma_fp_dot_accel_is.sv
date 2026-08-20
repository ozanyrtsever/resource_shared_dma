// dma_fp_dot_accel_is.sv — INPUT-STATIONARY, RUNTIME N/M/B, BATCHED GEMM (B>1) / GEMV (B=1).
//
// WHAT IT COMPUTES:  Y[M][B] = W[M][N] * X[N][B]     (a matrix-matrix product; B=1 -> matrix-vector)
//   M = number of output rows (neurons)   N = dot length (input features)   B = batch size (# inputs)
//
// PROTOCOL (two DMA transfers, back to back):
//   LOAD  transfer = [N, M, B, x0_0..x0_{N-1}, x1_0.., ..., x_{B-1}_{N-1}]  -> the B input vectors are
//                    buffered ONCE into u_xbuf ("input-stationary").
//   WEIGHT transfer = [w0 .. w_{M*N-1}]  -> the whole weight matrix is streamed ONCE; each weight is
//                    reused across all B inputs -> B multiply-accumulates (MACs) per streamed weight.
//   (A DMA FIFO flush between the two transfers flips the FSM from "loaded" into the compute phase.)
//
// WHY BATCHING: reusing one weight for B MACs is weight-reuse WITHOUT weight-stationary. For B>=2 the
//   shared FMA becomes the bottleneck (compute-bound) instead of the operand bus (memory-bound).
//   MAXB=1 collapses to the original single-input GEMV (the input buffer degenerates to one vector).
//
// KEY IDEA: this block has NO multiplier of its own. Every MAC is issued to the CPU's FMA over an
//   APU-like handshake (fpu_req/gnt/operands/rvalid/rdata). That is the whole point: share, don't add.
//
// To the DMA it looks like a FIFO: the DMA pushes data in (header/inputs/weights) and pops results out
//   on the same hw_fifo channel; full/empty give backpressure; done_o ends a transfer.
// verilator lint_off UNUSEDSIGNAL
module dma_fp_dot_accel_is
  import fifo_pkg::*;                         // fifo_req_t / fifo_resp_t (DMA hardware-FIFO handshake)
#(
    parameter int unsigned MAXN = 1024,     // max dot length (>=640)
    parameter int unsigned MAXB = 8        // max batch size B (GEMM mode)
)(
    input  logic        clk_i, rst_ni,       // clock + asynchronous active-low reset
    // ---- DMA hardware-FIFO side (data in / results out) ----
    input  fifo_req_t   hw_fifo_req_i,        // .push/.pop/.flush/.data from the DMA
    output fifo_resp_t  hw_fifo_resp_o,       // .full/.alm_full/.empty/.data back to the DMA
    output logic        done_o,               // this transfer has finished (DMA end-of-transfer)
    // ---- shared-FMA side (APU-like): one MAC = one FMA request to the CPU's FPnew ----
    output logic              fpu_req_o,       // request the shared FMA
    input  logic              fpu_gnt_i,       // arbiter grants the FMA this cycle
    output logic [2:0][31:0]  fpu_operands_o,  // {op2,op1,op0} for fmadd = op0*op1 + op2
    output logic [5:0]        fpu_op_o,        // FP operation code (0 = fmadd)
    output logic [14:0]       fpu_flags_o,     // FP flags (rounding mode etc.; 0 = default)
    input  logic              fpu_rvalid_i,    // FMA result is valid this cycle
    input  logic [31:0]       fpu_rdata_i,     // FMA result data (the MAC output)
    input  logic [4:0]        fpu_rflags_i     // FMA status flags (unused here)
);
  localparam int unsigned IW = $clog2(MAXN);
  // Effective batch dimension: GEMV_ONLY collapses it to 1 at ELABORATION -> the accumulator array,
  // the batch counters and the loop bounds all fold away (lean GEMV). GEMM keeps the full MAXB.
  localparam int unsigned NB = MAXB;
  localparam int unsigned BW = (NB>1) ? $clog2(NB) : 1;


  // FSM states: load the header, load the inputs, then per weight do B MACs, then emit B results.
  typedef enum logic [3:0] { LHDR_N, LHDR_M, LHDR_B, LDAT, RECV, REQ, WAIT, OUT, DONE } state_e;
  state_e state_q;

  // ---- internal state (registers) ----
  logic [31:0] x_rd;                    // input value read COMBINATIONALLY from the buffer (u_xbuf output)
  logic [31:0] acc_q [NB];
  logic [31:0] w_q;                     // the weight currently being streamed / reused across the B lanes
  logic [15:0] n_q, m_q, bn_q;          // runtime dot-length N / row-count M / batch-count B (from header)
  logic [15:0] i_q, r_q;                // i_q = element index (0..N-1), r_q = output-row index (0..M-1)
  logic [15:0] ld_b, ex_b, out_b;       // batch index for the LOAD / EXECUTE / OUTPUT phases respectively
  logic        loaded_q;                // 1 once the inputs are buffered (used by the inter-transfer flush)

  // ---- combinational outputs ----
  assign fpu_op_o       = 6'd0;         // fmadd
  assign fpu_flags_o    = 15'd0;        // default rounding, no special flags
  // THE MAC: operands = {op2=acc, op1=weight, op0=x}. fpnew fmadd computes op0*op1+op2 =
  //   x_rd * w_q + acc_q[ex_b]  -> accumulate into batch lane ex_b. w_q is fixed while ex_b sweeps 0..B-1
  //   (same weight reused across all B inputs), x_rd = the ex_b-th input's i_q-th element.
  assign fpu_operands_o = {acc_q[ex_b[BW-1:0]], w_q, x_rd};
  assign fpu_req_o      = (state_q == REQ);   // only ask for the FMA while in REQ

  // ---- DMA FIFO backpressure/handshake ----
  // The accel can accept a push only while reading the header / loading inputs / receiving a weight.
  wire can_push = (state_q==LHDR_N)||(state_q==LHDR_M)||(state_q==LHDR_B)||(state_q==LDAT)||(state_q==RECV);
  assign hw_fifo_resp_o = '{ full:!can_push, alm_full:!can_push,   // block pushes when not accepting data
                            empty:(state_q!=OUT),                  // results are readable ONLY in OUT
                            data:acc_q[out_b[BW-1:0]] };           // the result being emitted (lane out_b)
  assign done_o = (state_q == DONE);    // transfer finished
  // batch count for the loop bounds: compile-time 1 in GEMV mode (folds the counters), else runtime B
  wire [15:0] BEFF = bn_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      // ---- reset: back to header, clear all counters + the B accumulators ----
      state_q<=LHDR_N; w_q<='0; n_q<='0; m_q<='0; bn_q<='0; i_q<='0; r_q<='0;
      ld_b<='0; ex_b<='0; out_b<='0; loaded_q<=1'b0;
      for (int k=0;k<NB;k++) acc_q[k]<='0;
      // x_buf (u_xbuf) is NOT reset: every location is written during LDAT before it is ever read
    end else if (hw_fifo_req_i.flush) begin
      // ---- flush between the two DMA transfers: keep the buffered inputs + N/M/B, restart compute ----
      for (int k=0;k<NB;k++) acc_q[k]<='0;
      i_q<='0; r_q<='0; ex_b<='0; out_b<='0; ld_b<='0;   // x_buf + n_q + m_q + bn_q KEPT
      state_q <= loaded_q ? RECV : LHDR_N;   // if inputs already loaded -> jump straight to compute (RECV)
    end else begin
      case (state_q)
        // ---- read the 3-word header: N, then M, then B ----
        LHDR_N: if (hw_fifo_req_i.push) begin n_q<=hw_fifo_req_i.data[15:0]; state_q<=LHDR_M; end
        LHDR_M: if (hw_fifo_req_i.push) begin m_q<=hw_fifo_req_i.data[15:0]; state_q<=LHDR_B; end
        LHDR_B: if (hw_fifo_req_i.push) begin bn_q<=hw_fifo_req_i.data[15:0]; i_q<='0; ld_b<='0; state_q<=LDAT; end

        // ---- LOAD phase: stream B*N input words into the buffer (u_xbuf writes; see xbuf_we below) ----
        LDAT: if (hw_fifo_req_i.push) begin
          if (i_q+1==n_q) begin                 // finished one input vector (N elements)
            i_q<='0;
            if (ld_b+1==BEFF) begin loaded_q<=1'b1; state_q<=DONE; end  // all B inputs loaded -> end LOAD xfer
            else ld_b<=ld_b+1'b1;               // move to the next input vector
          end else i_q<=i_q+1'b1;               // next element of the current input vector
        end

        // ---- receive ONE weight, then fire B MACs with it ----
        RECV: if (hw_fifo_req_i.push) begin w_q<=hw_fifo_req_i.data; ex_b<='0; state_q<=REQ; end

        // ---- REQ: issue the FMA for batch lane ex_b (weight w_q, input x_rd = x[ex_b][i_q]) ----
        REQ:  if (fpu_gnt_i) begin               // arbiter granted the FMA
          if (fpu_rvalid_i) begin                // result came back same cycle (L=0 / already pipelined)
            acc_q[ex_b[BW-1:0]]<=fpu_rdata_i;    // write the MAC result into lane ex_b
            if (ex_b+1==BEFF) begin              // done all B lanes for THIS weight
              if (i_q+1==n_q) begin i_q<='0; out_b<='0; state_q<=OUT; end  // row complete -> emit results
              else begin i_q<=i_q+1'b1; state_q<=RECV; end                 // else fetch the next weight
            end else begin ex_b<=ex_b+1'b1; state_q<=REQ; end   // next batch lane, SAME weight (reuse!)
          end else state_q<=WAIT;                // result not ready yet -> wait for it (pipeline latency L>0)
        end

        // ---- WAIT: FMA was granted but result is still in the pipeline; wait for rvalid ----
        WAIT: if (fpu_rvalid_i) begin            // (this serial wait is why cyc/MAC ~= 2+L for one accel)
          acc_q[ex_b[BW-1:0]]<=fpu_rdata_i;
          if (ex_b+1==BEFF) begin
            if (i_q+1==n_q) begin i_q<='0; out_b<='0; state_q<=OUT; end
            else begin i_q<=i_q+1'b1; state_q<=RECV; end
          end else begin ex_b<=ex_b+1'b1; state_q<=REQ; end
        end

        // ---- OUT: emit the B results of the finished row (DMA pops acc_q[out_b]) ----
        OUT: if (hw_fifo_req_i.pop) begin
          if (out_b+1==BEFF) begin                 // all B results of this row emitted
            if (r_q+1==m_q) begin loaded_q<=1'b0; state_q<=DONE; end   // last row -> whole GEMM done
            else begin
              for (int k=0;k<NB;k++) acc_q[k]<='0;   // clear the accumulators for the next row
              r_q<=r_q+1'b1; out_b<='0; state_q<=RECV; // next row: go receive its first weight
            end
          end else out_b<=out_b+1'b1;              // emit the next batch lane's result
        end

        DONE: ;                                    // hold (done_o asserted); flush/reset move us on
        default: state_q<=LHDR_N;
      endcase
    end
  end

  // ---- input buffer, factored into its own module so synthesis can treat it as a 1R1W SRAM macro ----
  //      (dc/rtl_accel_is.f omits xbuf_ram.sv -> u_xbuf links as a black box -> logic-only area).
  // write-enable is EXACTLY when the old inline x_buf wrote: not reset, not flush, in LDAT, on a push.
  wire xbuf_we = rst_ni && !hw_fifo_req_i.flush && (state_q==LDAT) && hw_fifo_req_i.push;
  xbuf_ram #(.MAXB(NB), .MAXN(MAXN), .BW(BW), .IW(IW)) u_xbuf (
    .clk_i (clk_i),
    .we_i  (xbuf_we),                    // write pulse (LOAD phase)
    .wb_i  (ld_b[BW-1:0]),               // write batch index  (which input vector)
    .wa_i  (i_q[IW-1:0]),                // write element index (position within the vector)
    .wd_i  (hw_fifo_req_i.data),         // write data (the streamed input word)
    .rb_i  (ex_b[BW-1:0]),               // read batch index   (which lane's input for the current MAC)
    .ra_i  (i_q[IW-1:0]),                // read element index  (current dot position)
    .rd_o  (x_rd)                        // combinational read data -> feeds fpu_operands_o op0
  );

endmodule

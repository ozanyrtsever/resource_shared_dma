// dma_fp_dot_accel_pipe.sv — PIPELINED input-stationary batched GEMM (shares the CPU's FMA).
//
// Drop-in twin of dma_fp_dot_accel_is.sv (identical ports/protocol), but the compute engine is
// PIPELINED. The serial baseline does "issue one MAC -> WAIT 1+L for its result -> issue next"
// (cyc/MAC = 1.14 + L). This engine issues the B independent batch-MACs of a weight back-to-back
// (one per granted cycle) and collects results as they retire, keeping the FMA pipeline full ->
// cyc/MAC ~= 1.14 for ALL latencies L. That is why ONE pipelined unit suffices (no 2nd coprocessor).
//
// COMPUTES:  Y[M][B] = W[M][N] * X[N][B]      (runtime B; B=1 auto-degenerates to serial GEMV, see below)
// PROTOCOL:  two DMA transfers -- LOAD [N,M,B, X...] (buffered once), then WEIGHT stream [w0..w_{M*N-1}].
//
// KEY IDEA (unchanged): NO local multiplier. Every MAC is one fmadd issued to the CPU's shared FMA over
// the APU handshake. The arbiter grants every cycle and routes results back by owner-tag, so this block
// may hold up to (L+1) MACs in flight at once.
//
// CORRECTNESS:
//   * For a fixed weight w[i] the B lanes acc_q[0..B-1] += w[i]*x[b][i] are INDEPENDENT -> issue freely.
//   * Lane b is re-issued every B issues; its previous result has retired by then iff B >= L+1, else the
//     interlock 'inflight < B' stalls (the true pipeline limit -- still correct, just slower).
//   * fpnew ADDMUL is an in-order pipe -> our results retire in issue order -> a round-robin 'col_b'
//     counter knows which lane each result belongs to (no per-op tag needed inside the accelerator).
//   * Per-lane accumulation order stays k=0..N-1 -> BIT-IDENTICAL to the serial unit and to the CPU.
//   * B=1: interlock forces inflight<1 -> fully serial issue -> exactly the old GEMV behaviour.
// verilator lint_off UNUSEDSIGNAL
module dma_fp_dot_accel_pipe
  import fifo_pkg::*;
#(
    parameter int unsigned MAXN = 1024,   // max dot length
    parameter int unsigned MAXB = 8       // max batch size B
)(
    input  logic        clk_i, rst_ni,
    // ---- DMA hardware-FIFO side ----
    input  fifo_req_t   hw_fifo_req_i,
    output fifo_resp_t  hw_fifo_resp_o,
    output logic        done_o,
    // ---- shared-FMA side (APU-like) ----
    output logic              fpu_req_o,
    input  logic              fpu_gnt_i,
    output logic [2:0][31:0]  fpu_operands_o,
    output logic [5:0]        fpu_op_o,
    output logic [14:0]       fpu_flags_o,
    input  logic              fpu_rvalid_i,
    input  logic [31:0]       fpu_rdata_i,
    input  logic [4:0]        fpu_rflags_i
);
  localparam int unsigned IW = $clog2(MAXN);
  localparam int unsigned NB = MAXB;                 // batched GEMM: full B lanes (no GEMV fold)
  localparam int unsigned BW = (NB>1) ? $clog2(NB) : 1;

  // Load header (N,M,B); load inputs; then per row: RECV a weight, ISSUE its B MACs (pipelined),
  // DRAIN the pipe once all N weights are issued, then OUT the B row-results.
  typedef enum logic [3:0] { LHDR_N, LHDR_M, LHDR_B, LDAT, RECV, ISSUE, DRAIN, OUT, DONE } state_e;
  state_e state_q;

  logic [31:0] x_rd;                 // input value read combinationally from the buffer = x[iss_b][i_q]
  logic [31:0] acc_q [NB];           // B running dot-product accumulators (one per batch lane)
  logic [31:0] w_q;                  // current weight w[i_q], reused across the B lanes
  logic [15:0] n_q, m_q, bn_q;       // runtime N / M / B (from the header)
  logic [15:0] i_q, r_q;             // i_q = element index (0..N-1), r_q = output row (0..M-1)
  logic [15:0] ld_b;                 // batch index during the LOAD phase
  logic [15:0] iss_b, col_b, out_b;  // issue lane / collect lane / output lane (each 0..B-1)
  logic [15:0] inflight;             // MACs issued but not yet retired (<= L+1); 16-bit to match BEFF/counters
  logic        loaded_q;

  wire [15:0] BEFF = bn_q;           // effective batch = runtime B

  assign fpu_op_o    = 6'd0;         // fmadd
  assign fpu_flags_o = 15'd0;        // default rounding
  // MAC operands {op2=acc, op1=w, op0=x} -> fmadd = x*w + acc, accumulated into ISSUE lane iss_b.
  assign fpu_operands_o = {acc_q[iss_b[BW-1:0]], w_q, x_rd};
  // Request the FMA while issuing, UNLESS the hazard interlock fires: do not issue lane iss_b until its
  // previous (i-1) result has been collected -> fewer than B MACs in flight.
  assign fpu_req_o = (state_q==ISSUE) && (inflight < BEFF);

  wire in_compute   = (state_q==RECV) || (state_q==ISSUE) || (state_q==DRAIN);
  wire issue_fire   = fpu_req_o & fpu_gnt_i;              // a MAC accepted by the FMA this cycle
  wire collect_fire = fpu_rvalid_i & in_compute;         // a result of ours returns this cycle

  wire can_push = (state_q==LHDR_N)||(state_q==LHDR_M)||(state_q==LHDR_B)||(state_q==LDAT)||(state_q==RECV);
  assign hw_fifo_resp_o = '{ full:!can_push, alm_full:!can_push,
                            empty:(state_q!=OUT),
                            data:acc_q[out_b[BW-1:0]] };
  assign done_o = (state_q == DONE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q<=LHDR_N; w_q<='0; n_q<='0; m_q<='0; bn_q<='0; i_q<='0; r_q<='0;
      ld_b<='0; iss_b<='0; col_b<='0; out_b<='0; inflight<='0; loaded_q<=1'b0;
      for (int k=0;k<NB;k++) acc_q[k]<='0;
    end else if (hw_fifo_req_i.flush) begin
      for (int k=0;k<NB;k++) acc_q[k]<='0;
      i_q<='0; r_q<='0; iss_b<='0; col_b<='0; out_b<='0; ld_b<='0; inflight<='0;
      state_q <= loaded_q ? RECV : LHDR_N;
    end else begin
      // ---- COLLECT: whenever one of our results returns, write it into lane col_b (round-robin) ----
      if (collect_fire) begin
        acc_q[col_b[BW-1:0]] <= fpu_rdata_i;
        col_b <= (col_b+1==BEFF) ? 16'd0 : col_b+1'b1;
      end
      // ---- in-flight bookkeeping: +1 on issue, -1 on collect ----
      inflight <= inflight + 16'(issue_fire) - 16'(collect_fire);

      case (state_q)
        LHDR_N: if (hw_fifo_req_i.push) begin n_q<=hw_fifo_req_i.data[15:0]; state_q<=LHDR_M; end
        LHDR_M: if (hw_fifo_req_i.push) begin m_q<=hw_fifo_req_i.data[15:0]; state_q<=LHDR_B; end
        LHDR_B: if (hw_fifo_req_i.push) begin bn_q<=hw_fifo_req_i.data[15:0]; i_q<='0; ld_b<='0; state_q<=LDAT; end

        LDAT: if (hw_fifo_req_i.push) begin
          if (i_q+1==n_q) begin
            i_q<='0;
            if (ld_b+1==BEFF) begin loaded_q<=1'b1; state_q<=DONE; end
            else ld_b<=ld_b+1'b1;
          end else i_q<=i_q+1'b1;
        end

        // receive one weight for element i_q, then issue its B MACs
        RECV: if (hw_fifo_req_i.push) begin w_q<=hw_fifo_req_i.data; iss_b<='0; state_q<=ISSUE; end

        // ISSUE: fire one MAC per granted cycle across the B lanes (collect runs in parallel above)
        ISSUE: if (issue_fire) begin
          if (iss_b+1==BEFF) begin                          // issued the last lane of this weight
            if (i_q+1==n_q) state_q<=DRAIN;                  // last element of the row -> drain
            else begin i_q<=i_q+1'b1; state_q<=RECV; end     // else fetch the next weight
          end else iss_b<=iss_b+1'b1;                        // next lane, SAME weight (reuse)
        end

        // DRAIN: everything issued; wait for the pipe to empty, then emit the row
        DRAIN: if (inflight==0) begin out_b<='0; state_q<=OUT; end

        // OUT: emit the B results of the finished row (DMA pops acc_q[out_b])
        OUT: if (hw_fifo_req_i.pop) begin
          if (out_b+1==BEFF) begin
            if (r_q+1==m_q) begin loaded_q<=1'b0; state_q<=DONE; end          // last row -> done
            else begin
              for (int k=0;k<NB;k++) acc_q[k]<='0;                            // clear accs for next row
              r_q<=r_q+1'b1; i_q<='0; iss_b<='0; col_b<='0; out_b<='0; state_q<=RECV;
            end
          end else out_b<=out_b+1'b1;
        end

        DONE: ;
        default: state_q<=LHDR_N;
      endcase
    end
  end

  // ---- input buffer: 1R1W SRAM macro (dc/rtl_accel_*.f omits xbuf_ram.sv -> black box for logic area) ----
  wire xbuf_we = rst_ni && !hw_fifo_req_i.flush && (state_q==LDAT) && hw_fifo_req_i.push;
  xbuf_ram #(.MAXB(NB), .MAXN(MAXN), .BW(BW), .IW(IW)) u_xbuf (
    .clk_i (clk_i),
    .we_i  (xbuf_we),
    .wb_i  (ld_b[BW-1:0]),
    .wa_i  (i_q[IW-1:0]),
    .wd_i  (hw_fifo_req_i.data),
    .rb_i  (iss_b[BW-1:0]),        // read lane = ISSUE lane
    .ra_i  (i_q[IW-1:0]),
    .rd_o  (x_rd)
  );
endmodule

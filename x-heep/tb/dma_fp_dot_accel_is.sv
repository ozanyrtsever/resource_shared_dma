// dma_fp_dot_accel_is.sv — INPUT-STATIONARY GEMV, RUNTIME-CONFIGURABLE N/M
// LOAD transfer = [N, M, x0..x_{N-1}] (accel N,M latch'ler + x_buf doldurur)
// GEMV transfer = [w0..w_{M*N-1}]     (başlıksız; accel M dot sonucu üretir)
// -> tek binary her katman şeklini işler; boyut porttan DEĞİL stream'den gelir.
// verilator lint_off UNUSEDSIGNAL
module dma_fp_dot_accel_is
  import fifo_pkg::*;
#(
    parameter int unsigned MAXN = 1024      // en büyük dot uzunluğu (>=640)
)(
    input  logic        clk_i, rst_ni,
    input  fifo_req_t   hw_fifo_req_i,
    output fifo_resp_t  hw_fifo_resp_o,
    output logic        done_o,
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
  typedef enum logic [2:0] { LHDR_N, LHDR_M, LDAT, RECV, REQ, WAIT, OUT, DONE } state_e;
  state_e state_q;

  logic [31:0] x_buf [MAXN];
  logic [31:0] acc_q, w_q;
  logic [15:0] n_q, m_q;        // runtime dot-uzunluğu / satır-sayısı
  logic [15:0] i_q, r_q;
  logic        loaded_q;

  assign fpu_op_o       = 6'd0;
  assign fpu_flags_o    = 15'd0;
  assign fpu_operands_o = {acc_q, w_q, x_buf[i_q[IW-1:0]]};
  assign fpu_req_o      = (state_q == REQ);

  wire can_push = (state_q==LHDR_N)||(state_q==LHDR_M)||(state_q==LDAT)||(state_q==RECV);
  assign hw_fifo_resp_o = '{ full:!can_push, alm_full:!can_push,
                            empty:(state_q!=OUT), data:acc_q };
  assign done_o = (state_q == DONE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q<=LHDR_N; acc_q<='0; w_q<='0; n_q<='0; m_q<='0; i_q<='0; r_q<='0; loaded_q<=1'b0;
      for (int k=0;k<MAXN;k++) x_buf[k]<='0;
    end else if (hw_fifo_req_i.flush) begin
      acc_q<='0; i_q<='0; r_q<='0;               // x_buf + n_q + m_q KORUNUR
      state_q <= loaded_q ? RECV : LHDR_N;
    end else begin
      case (state_q)
        LHDR_N: if (hw_fifo_req_i.push) begin n_q<=hw_fifo_req_i.data[15:0]; state_q<=LHDR_M; end
        LHDR_M: if (hw_fifo_req_i.push) begin m_q<=hw_fifo_req_i.data[15:0]; i_q<='0; state_q<=LDAT; end
        LDAT: if (hw_fifo_req_i.push) begin
          x_buf[i_q[IW-1:0]] <= hw_fifo_req_i.data;
          if (i_q+1==n_q) begin loaded_q<=1'b1; i_q<='0; state_q<=DONE; end
          else i_q<=i_q+1'b1;
        end
        RECV: if (hw_fifo_req_i.push) begin w_q<=hw_fifo_req_i.data; state_q<=REQ; end
        REQ:  if (fpu_gnt_i) begin
          if (fpu_rvalid_i) begin acc_q<=fpu_rdata_i;
            if (i_q+1==n_q) begin i_q<='0; state_q<=OUT; end
            else begin i_q<=i_q+1'b1; state_q<=RECV; end
          end else state_q<=WAIT;
        end
        WAIT: if (fpu_rvalid_i) begin acc_q<=fpu_rdata_i;
          if (i_q+1==n_q) begin i_q<='0; state_q<=OUT; end
          else begin i_q<=i_q+1'b1; state_q<=RECV; end
        end
        OUT: if (hw_fifo_req_i.pop) begin
          if (r_q+1==m_q) begin loaded_q<=1'b0; state_q<=DONE; end
          else begin acc_q<='0; r_q<=r_q+1'b1; state_q<=RECV; end
        end
        DONE: ;
        default: state_q<=LHDR_N;
      endcase
    end
  end
endmodule

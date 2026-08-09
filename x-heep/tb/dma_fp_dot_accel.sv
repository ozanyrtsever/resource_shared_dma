// dma_fp_dot_accel.sv
// HW-FIFO accelerator: araya giren [a,b] operandlarını eşler, dış FMA ile acc += a*b
// biriktirir, N çift sonra skaler dot product'ı sunar, done basar.
// verilator lint_off UNUSEDSIGNAL
// fpu_rflags_i: FMA exception flag'leri bu accel'de kasıtlı kullanılmıyor
module dma_fp_dot_accel
  import fifo_pkg::*;
 (
    input  logic        clk_i,
    input  logic        rst_ni,

    input  logic [15:0] num_pairs_i,      // N = çift sayısı = SIZE_D1 / 2

    // HW-FIFO tarafı (DMA bunu write FIFO'su sanır)
    input  fifo_req_t   hw_fifo_req_i,     // .push/.pop/.flush/.data
    output fifo_resp_t  hw_fifo_resp_o,    // .full/.empty/.alm_full/.data
    output logic        done_o,            // -> DMA hw_fifo_done_i

    // FMA tarafı (APU-stili -> cv32e40p_fp_wrapper)
    output logic              fpu_req_o,
    input  logic              fpu_gnt_i,
    output logic [2:0][31:0]  fpu_operands_o,  // [0]=a, [1]=b, [2]=acc
    output logic [5:0]        fpu_op_o,        // 6'd0  = FMADD
    output logic [14:0]       fpu_flags_o,     // 15'd0 = FP32/RNE
    input  logic              fpu_rvalid_i,
    input  logic [31:0]       fpu_rdata_i,
    input  logic [4:0]        fpu_rflags_i
);

  typedef enum logic [2:0] {
    RECV_A,    // ilk operand (a) bekle
    RECV_B,    // ikinci operand (b) bekle
    FMA_REQ,   // FMA'e istek (a,b,acc)
    FMA_WAIT,  // FMA sonucunu bekle
    RESULT,    // sonucu çıkışa sun, pop bekle
    DONE       // bitti
  } state_e;

  state_e      state_q;
  logic [31:0] acc_q;
  logic [31:0] a_q, b_q;
  logic [15:0] cnt_q;        // tamamlanan çift sayısı
  logic busy;

  // --- FMA isteği (sabit: FP32 FMADD, RNE) ---
  assign fpu_op_o       = 6'd0;
  assign fpu_flags_o    = 15'd0;
  assign fpu_operands_o = {acc_q, b_q, a_q};   // [2]=acc, [1]=b, [0]=a
  assign fpu_req_o      = (state_q == FMA_REQ);

  assign busy = (state_q != RECV_A) && (state_q != RECV_B);  // "şu an push alamam"
  // --- HW-FIFO yanıtı ---
  assign hw_fifo_resp_o = '{
      full:     busy,  // FMA/çıkış meşgulken push alma
      alm_full: busy,
      empty:    (state_q != RESULT),                          // sonuç yalnız RESULT'ta hazır
      data:     acc_q
  };
  assign done_o = (state_q == DONE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= RECV_A; acc_q <= '0; a_q <= '0; b_q <= '0; cnt_q <= '0;
    end else if (hw_fifo_req_i.flush) begin     // transfer başı: temizle
      state_q <= RECV_A; acc_q <= '0; a_q <= '0; b_q <= '0; cnt_q <= '0;
    end else begin
      case (state_q)
        RECV_A: if (hw_fifo_req_i.push) begin a_q <= hw_fifo_req_i.data; state_q <= RECV_B; end
        RECV_B: if (hw_fifo_req_i.push) begin b_q <= hw_fifo_req_i.data; state_q <= FMA_REQ; end

        FMA_REQ: if (fpu_gnt_i) begin //  slave FPU gives gnt=0 after receiving req, so that it needs another state to wait for the result 
          if (fpu_rvalid_i) begin               // LAT=0: sonuç aynı çevrim
            acc_q <= fpu_rdata_i; cnt_q <= cnt_q + 1'b1;
            state_q <= (cnt_q + 1'b1 == num_pairs_i) ? RESULT : RECV_A;
          end else state_q <= FMA_WAIT;
        end

        FMA_WAIT: if (fpu_rvalid_i) begin        // LAT>=1: sonuç sonraki çevrim(ler)de
          acc_q <= fpu_rdata_i; cnt_q <= cnt_q + 1'b1;
          state_q <= (cnt_q + 1'b1 == num_pairs_i) ? RESULT : RECV_A;
        end

        RESULT: if (hw_fifo_req_i.pop) state_q <= DONE;   // write unit sonucu çekti
        DONE: ;                                            // flush'a kadar done_o=1
        default: state_q <= RECV_A;
      endcase
    end
  end
endmodule

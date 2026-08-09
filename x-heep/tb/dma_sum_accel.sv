// dma_sum_accel.sv — Y1: FP-free HW-FIFO tap proof (integer sum-reduction)
// N int32 kelimeyi DMA HW-FIFO'dan tüketir, 32-bit toplamını (1 kelime) sunar, done verir.
// FPU/APU YOK. (Y2: accumulate-add -> paylaşılan CPU FPU üzerinden FMA.)
module dma_sum_accel
  import fifo_pkg::*;                 // <-- X-HEEP'in fifo_pkg'ı (gotcha 1)
(
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic [15:0] num_words_i,  // kaç kelime indirgenecek (= DMA src boyutu)
    input  fifo_req_t   hw_fifo_req_i, // DMA'dan: .push/.pop/.flush/.data
    output fifo_resp_t  hw_fifo_resp_o,// DMA'ya:  .empty/.full/.alm_full/.data
    output logic        done_o
);

  typedef enum logic [1:0] {ACCUM, RESULT, DONE} state_e;
  state_e      state_q;
  logic [31:0] acc_q;
  logic [15:0] cnt_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= ACCUM; acc_q <= '0; cnt_q <= '0;
    end else if (hw_fifo_req_i.flush) begin           // DMA reset/yeni transfer
      state_q <= ACCUM; acc_q <= '0; cnt_q <= '0;
    end else begin
      unique case (state_q)
        ACCUM: if (hw_fifo_req_i.push) begin           // her push'ta birik
          acc_q <= acc_q + hw_fifo_req_i.data;
          cnt_q <= cnt_q + 16'd1;
          if (cnt_q + 16'd1 == num_words_i) state_q <= RESULT;  // N'inci kelime -> sonuç hazır
        end
        RESULT: if (hw_fifo_req_i.pop) state_q <= DONE; // DMA sonucu çekti
        DONE: ;                                         // flush'a kadar bekle
        default: state_q <= ACCUM;
      endcase
    end
  end

  // DMA'nın write-FIFO'su biziz: durum sinyalleri
  always_comb begin
    hw_fifo_resp_o          = '0;
    hw_fifo_resp_o.data     = acc_q;
    hw_fifo_resp_o.empty    = (state_q != RESULT); // sonuç yalnız RESULT'ta hazır -> sadece o an pop edilebilir
    hw_fifo_resp_o.full     = (state_q != ACCUM);  // yalnız ACCUM'da push kabul et
    hw_fifo_resp_o.alm_full = (state_q != ACCUM);
  end

  assign done_o = (state_q == DONE);
endmodule

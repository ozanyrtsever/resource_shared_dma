// Y1: DMA HW-FIFO tap kanıtı (FP'siz toplama-indirgemesi).
// DMA, src'den N int32 kelimeyi kanal 1'deki dma_sum_accel'e akıtır; accel toplamı
// üretir, DMA onu dst[0]'a yazar. dst[0] == sum(src) ise tap on-SoC çalışıyor.
#include <stdio.h>
#include <stdint.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"

#define PRINTF_IN_SIM 1
#if TARGET_SIM && PRINTF_IN_SIM
  #define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
  #define PRINTF(...)
#endif

#define N      16     // testharness'taki num_words_i (16'd16) ile EŞLEŞMELİ
#define DMA_CH 1      // dma_sum_accel'in bağlı olduğu kanal

int32_t src_buf[N] __attribute__((aligned(4)));
int32_t dst_buf[N] __attribute__((aligned(4)));

int main(void) {
    int32_t golden = 0;
    for (int i = 0; i < N; i++) { src_buf[i] = i + 1; golden += src_buf[i]; } // 1..16 -> 136
    dst_buf[0] = 0xDEADBEEF;

    dma_target_t tgt_src = {
        .ptr = (uint8_t *) src_buf, .inc_d1_du = 1,
        .trig = DMA_TRIG_MEMORY,    .type = DMA_DATA_TYPE_WORD,
    };
    dma_target_t tgt_dst = {
        .ptr = (uint8_t *) dst_buf, .inc_d1_du = 1,
        .trig = DMA_TRIG_MEMORY,    .type = DMA_DATA_TYPE_WORD,
    };
    dma_trans_t trans = {
        .src = &tgt_src, .dst = &tgt_dst,
        .mode = DMA_TRANS_MODE_SINGLE,
        .hw_fifo_en = 1,                 // <-- HW-FIFO modu
        .dim = DMA_DIM_CONF_1D,
        .size_d1_du = N,
        .end = DMA_TRANS_END_POLLING,    // interrupt yerine poll (basit)
        .channel = DMA_CH,               // <-- kanal 1
    };

    dma_init(NULL);

    dma_config_flags_t r;
    r = dma_validate_transaction(&trans, DMA_ENABLE_REALIGN, DMA_PERFORM_CHECKS_INTEGRITY);
    if (r != DMA_CONFIG_OK) { PRINTF("validate FAIL: 0x%x\n", r); return -1; }
    r = dma_load_transaction(&trans);
    if (r != DMA_CONFIG_OK) { PRINTF("load FAIL: 0x%x\n", r); return -2; }
    r = dma_launch(&trans);
    if (r != DMA_CONFIG_OK) { PRINTF("launch FAIL: 0x%x\n", r); return -3; }

    while (!dma_is_ready(DMA_CH));   // accel done -> DMA idle

    PRINTF("dst[0] = %d  (golden = %d)\n", (int) dst_buf[0], (int) golden);
    if (dst_buf[0] != golden) { PRINTF("Y1 tap: FAIL\n"); return -4; }
    PRINTF("Y1 HW-FIFO tap: PASS\n");
    return 0;
}

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"

#define PRINTF_IN_SIM 1
#if TARGET_SIM && PRINTF_IN_SIM
  #define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
  #define PRINTF(...)
#endif

#define NPAIRS 32            // testharness'taki num_pairs_i (16'd32) ile EŞLEŞMELİ
#define NWORDS (2*NPAIRS)    // interleaved [a0,b0,a1,b1,...]
#define DMA_CH 1

float src_buf[NWORDS] __attribute__((aligned(4)));  // interleaved a,b
float dst_buf[NWORDS] __attribute__((aligned(4)));  // dst[0] = dot product

int main(void) {
    CSR_SET_BITS(CSR_REG_MSTATUS, (1 << 13));       // FP'yi etkinleştir (FS)
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);     // mcycle sayacı

    float golden = 0.0f;
    for (int i = 0; i < NPAIRS; i++) {
        float a = (float)(i + 1);
        float b = (float)(i + 1);
        src_buf[2*i] = a; src_buf[2*i+1] = b;       // interleaved
        golden += a * b;                            // Σ (i+1)^2 = 11440
    }
    dst_buf[0] = 0.0f;

    dma_target_t tgt_src = { .ptr=(uint8_t*)src_buf, .inc_d1_du=1, .trig=DMA_TRIG_MEMORY, .type=DMA_DATA_TYPE_WORD };
    dma_target_t tgt_dst = { .ptr=(uint8_t*)dst_buf, .inc_d1_du=1, .trig=DMA_TRIG_MEMORY, .type=DMA_DATA_TYPE_WORD };
    dma_trans_t trans = {
        .src=&tgt_src, .dst=&tgt_dst, .mode=DMA_TRANS_MODE_SINGLE,
        .hw_fifo_en=1, .dim=DMA_DIM_CONF_1D, .size_d1_du=NWORDS,
        .end=DMA_TRANS_END_POLLING, .channel=DMA_CH,
    };

    dma_init(NULL);
    if (dma_validate_transaction(&trans, DMA_ENABLE_REALIGN, DMA_PERFORM_CHECKS_INTEGRITY) != DMA_CONFIG_OK) { PRINTF("validate FAIL\n"); return -1; }
    if (dma_load_transaction(&trans) != DMA_CONFIG_OK) { PRINTF("load FAIL\n"); return -2; }

    unsigned int cyc;
    CSR_WRITE(CSR_REG_MCYCLE, 0);
    if (dma_launch(&trans) != DMA_CONFIG_OK) { PRINTF("launch FAIL\n"); return -3; }
    while (!dma_is_ready(DMA_CH));                   // accel done -> DMA idle
    CSR_READ(CSR_REG_MCYCLE, &cyc);

    float dotp = dst_buf[0];
    PRINTF("accel dotp = %d  (golden = %d)  accel cycles = %u\n",
           (int)dotp, (int)golden, cyc);
    if (fabs(dotp - golden) > 0.5f) { PRINTF("Y2 shared-FMA dotp: FAIL\n"); return -4; }
    PRINTF("Y2 shared-FMA dotp: PASS\n");
    return 0;
}

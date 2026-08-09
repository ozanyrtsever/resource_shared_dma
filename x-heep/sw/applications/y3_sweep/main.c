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

#define N       32              // <-- SÜPÜR: bunu VE testharness num_pairs_i'yi BİRLİKTE değiştir
#define NMAX    128
#define NWORDS  (2*N)
#define DMA_CH  1

float a_arr[NMAX], b_arr[NMAX];
float src_buf[2*NMAX] __attribute__((aligned(4)));
float dst_buf[2*NMAX] __attribute__((aligned(4)));

int main(void) {
    CSR_SET_BITS(CSR_REG_MSTATUS, (1 << 13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);

    float golden = 0.0f;
    for (int i = 0; i < N; i++) {
        float a=(float)(i+1), b=(float)(i+1);
        a_arr[i]=a; b_arr[i]=b; src_buf[2*i]=a; src_buf[2*i+1]=b; golden += a*b;
    }

    // --- CPU baseline (kendi FMA'sı) ---
    volatile float cpu_res = 0.0f;
    unsigned int cpu_cyc;
    CSR_WRITE(CSR_REG_MCYCLE, 0);
    for (int i = 0; i < N; i++) cpu_res += a_arr[i]*b_arr[i];
    CSR_READ(CSR_REG_MCYCLE, &cpu_cyc);

    // --- Accel (paylaşılan FMA) ---
    dst_buf[0]=0.0f;
    dma_target_t ts={.ptr=(uint8_t*)src_buf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    dma_target_t td={.ptr=(uint8_t*)dst_buf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    dma_trans_t tr={.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                    .dim=DMA_DIM_CONF_1D,.size_d1_du=NWORDS,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    if (dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY)!=DMA_CONFIG_OK){PRINTF("val FAIL\n");return -1;}
    if (dma_load_transaction(&tr)!=DMA_CONFIG_OK){PRINTF("load FAIL\n");return -2;}
    unsigned int acc_cyc;
    CSR_WRITE(CSR_REG_MCYCLE, 0);
    if (dma_launch(&tr)!=DMA_CONFIG_OK){PRINTF("launch FAIL\n");return -3;}
    while (!dma_is_ready(DMA_CH));
    CSR_READ(CSR_REG_MCYCLE, &acc_cyc);

    float acc_res = dst_buf[0];
    PRINTF("N=%d  cpu=%u  accel=%u  (accel dotp %d vs golden %d)\n",
           N, cpu_cyc, acc_cyc, (int)acc_res, (int)golden);
    if (fabs(acc_res-golden)>0.5f){PRINTF("MISMATCH\n");return -4;}
    return 0;
}

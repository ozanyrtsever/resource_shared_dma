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

#define NP 32              // accel çift sayısı -> testharness num_pairs_i = 32 OLMALI
#define NW (2*NP)
#define M  32              // CPU'nun eşzamanlı FP MAC sayısı
#define DMA_CH 1

float src_buf[NW] __attribute__((aligned(4)));
float dst_buf[NW] __attribute__((aligned(4)));
float xg[M], yg[M];

int main(void) {
    CSR_SET_BITS(CSR_REG_MSTATUS, (1 << 13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);

    float gA=0.0f;  // accel golden
    for (int i=0;i<NP;i++){ float a=i+1,b=i+1; src_buf[2*i]=a; src_buf[2*i+1]=b; gA+=a*b; }
    float gC=0.0f;  // CPU golden
    for (int i=0;i<M;i++){ xg[i]=0.5f*(i+1); yg[i]=2.0f; gC += xg[i]*yg[i]; }

    dma_target_t ts={.ptr=(uint8_t*)src_buf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    dma_target_t td={.ptr=(uint8_t*)dst_buf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    dma_trans_t tr={.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                    .dim=DMA_DIM_CONF_1D,.size_d1_du=NW,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    if (dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY)!=DMA_CONFIG_OK){PRINTF("val FAIL\n");return -1;}
    if (dma_load_transaction(&tr)!=DMA_CONFIG_OK){PRINTF("load FAIL\n");return -2;}

    volatile float c1=0.0f, c2=0.0f;
    unsigned int cpu_alone, cpu_cont;

    // 1) CPU FP tek başına (accel boşta)
    CSR_WRITE(CSR_REG_MCYCLE,0);
    for (int i=0;i<M;i++) c1 += xg[i]*yg[i];
    CSR_READ(CSR_REG_MCYCLE,&cpu_alone);

    // 2) ÇEKİŞME: accel'i başlat, HEMEN CPU FP yap
    dst_buf[0]=0.0f;
    dma_launch(&tr);                          // accel arka planda başladı
    CSR_WRITE(CSR_REG_MCYCLE,0);
    for (int i=0;i<M;i++) c2 += xg[i]*yg[i];  // CPU FP, accel çalışırken
    CSR_READ(CSR_REG_MCYCLE,&cpu_cont);
    while(!dma_is_ready(DMA_CH));             // accel bitene kadar bekle

    int accOK  = (fabs(dst_buf[0]-gA) < 0.5f);
    int cpuOK  = (fabs(c1-gC)<0.5f) && (fabs(c2-gC)<0.5f);

    PRINTF("CPU FP alone=%u  under-contention=%u  (delta=%d cyc)\n",
           cpu_alone, cpu_cont, (int)cpu_cont-(int)cpu_alone);
    PRINTF("accel dotp=%d (golden %d) OK=%d ; CPU dotp=%d (golden %d) OK=%d\n",
           (int)dst_buf[0],(int)gA,accOK, (int)c2,(int)gC,cpuOK);
    if (!(accOK && cpuOK)) { PRINTF("CONTENTION: FAIL (corruption!)\n"); return -3; }
    PRINTF("CONTENTION: PASS - both correct under sharing, CPU-priority delta above\n");
    return 0;
}

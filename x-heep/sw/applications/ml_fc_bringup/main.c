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
#define N 32
#define M 32
#define DMA_CH 1
float x[N] __attribute__((aligned(4)));
float W[M][N] __attribute__((aligned(4)));
float bvec[M], y[M], yg[M];
float dummy[2] __attribute__((aligned(4)));
dma_target_t ts, td; dma_trans_t tr;

static inline void xfer(const void* src, void* dst, uint16_t nwords){
    ts.ptr=(uint8_t*)src; td.ptr=(uint8_t*)dst; tr.size_d1_du=nwords;
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
}
int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    for(int i=0;i<N;i++) x[i]=0.1f*(i+1);
    for(int j=0;j<M;j++){ bvec[j]=0.01f*j; for(int i=0;i<N;i++) W[j][i]=0.001f*((i+j)%7); }

    ts=(dma_target_t){.ptr=(uint8_t*)x,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=N,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    if(dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY)!=DMA_CONFIG_OK){PRINTF("val FAIL\n");return -1;}

    unsigned int cyc_accel, cyc_cpu, c1;
    CSR_WRITE(CSR_REG_MCYCLE,0);
    xfer(x, dummy, N);                       // 1) LOAD x (ilk transfer)
    xfer(W, y, (uint16_t)(M*N));             // 2) GEMV: tüm W tek transferde → y[0..M-1] dot'ları
    for(int j=0;j<M;j++){ float s=y[j]+bvec[j]; y[j]=s>0.0f?s:0.0f; }   // bias+relu (CPU)
    CSR_READ(CSR_REG_MCYCLE,&c1); cyc_accel=c1;

    CSR_WRITE(CSR_REG_MCYCLE,0);
    for(int j=0;j<M;j++){ float s=bvec[j]; for(int i=0;i<N;i++) s+=x[i]*W[j][i]; yg[j]=s>0.0f?s:0.0f; }
    CSR_READ(CSR_REG_MCYCLE,&cyc_cpu);

    int ok=1; float maxe=0;
    for(int j=0;j<M;j++){ float e=fabsf(y[j]-yg[j]); if(e>maxe)maxe=e; if(e>1e-3f)ok=0; }
    PRINTF("GEMV FC (N=%d M=%d): accel=%u cpu=%u maxerr=%d/1e6 %s\n",
           N,M,cyc_accel,cyc_cpu,(int)(maxe*1e6f),ok?"MATCH":"MISMATCH");
    return ok?0:-1;
}

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define DMA_CH 1
#define N 32
#define M 8
static inline uint32_t f2u(float f){ union{float f;uint32_t u;} c; c.f=f; return c.u; }
float x[N], W[M*N], out[M], golden[M];
uint32_t loadbuf[N+2];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;
static inline void xfer(const void* s, void* d, uint16_t n){
    ts.ptr=(uint8_t*)s; td.ptr=(uint8_t*)d; tr.size_d1_du=n;
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
}
int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    // deterministik veriler
    for(int i=0;i<N;i++) x[i]=0.1f*(float)(i+1);
    for(int m=0;m<M;m++) for(int i=0;i<N;i++) W[m*N+i]=0.01f*(float)((m+1)*(i+1))-0.05f;
    // CPU golden (füzyonlu)
    for(int m=0;m<M;m++){ float s=0.0f;
        for(int i=0;i<N;i++) s=fmaf(x[i],W[m*N+i],s); golden[m]=s; }
    // LOAD başlığı + x
    loadbuf[0]=N; loadbuf[1]=M;
    for(int i=0;i<N;i++) loadbuf[2+i]=f2u(x[i]);
    // DMA kurulumu (bir kez)
    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=N+2,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);
    // koş: LOAD (başlıklı) sonra GEMV
    xfer(loadbuf, dummy, N+2);
    xfer(W, out, (uint16_t)(M*N));
    // doğrula
    int bad=0; float maxe=0;
    for(int m=0;m<M;m++){ float e=fabsf(out[m]-golden[m]); if(e>maxe)maxe=e; if(e>1e-6f)bad++; }
    PRINTF("RT-GEMV N=%d M=%d: maxerr=%d/1e6 bad=%d/%d %s\n",
           N,M,(int)(maxe*1e6f),bad,M, bad?"FAIL":"PASS");
    return bad?-1:0;
}

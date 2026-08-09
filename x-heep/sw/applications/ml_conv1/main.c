#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "conv_test_data.h"     // StarDist conv1 verisi
#define PRINTF_IN_SIM 1
#if TARGET_SIM && PRINTF_IN_SIM
  #define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
  #define PRINTF(...)
#endif
#define K 9          // kernel 3x3x1 = N
#define OC 32        // çıktı kanalı = M
#define IH 8         // padded input 8x8
#define DMA_CH 1
static inline float w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
float patch[K] __attribute__((aligned(4)));
float wf[OC*K] __attribute__((aligned(4)));   // conv_weights -> float, GEMV-ready
float dst[OC]  __attribute__((aligned(4)));
float dummy[2];
dma_target_t ts, td; dma_trans_t tr;
static inline void xfer(const void* s, void* d, uint16_t n){
    ts.ptr=(uint8_t*)s; td.ptr=(uint8_t*)d; tr.size_d1_du=n;
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
}
int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    for(int i=0;i<OC*K;i++) wf[i]=w2f(conv_weights[i]);       // ağırlıklar
    for(int ki=0;ki<3;ki++) for(int kj=0;kj<3;kj++)          // im2col: piksel (0,0)
        patch[ki*3+kj] = w2f(conv_input_padded[(0+ki)*IH + (0+kj)]);

    ts=(dma_target_t){.ptr=(uint8_t*)patch,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=K,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    xfer(patch, dummy, K);              // LOAD patch (N=9)
    xfer(wf, dst, (uint16_t)(OC*K));    // GEMV -> dst[oc]=dot(patch,W[oc])

    int ok=1; float maxe=0;
    for(int oc=0; oc<OC; oc++){
        float y = dst[oc] + w2f(conv_biases[oc]);
        float g = w2f(conv_expected[oc]);          // piksel (0,0), kanal oc
        float e=fabsf(y-g); if(e>maxe)maxe=e; if(e>1e-3f) ok=0;
    }
    PRINTF("StarDist conv1 px(0,0): N=%d M=%d maxerr=%d/1e6 %s\n",
           K,OC,(int)(maxe*1e6f),ok?"MATCH":"MISMATCH");
    return ok?0:-1;
}

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "conv_test_data.h"
#define PRINTF_IN_SIM 1
#if TARGET_SIM && PRINTF_IN_SIM
  #define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
  #define PRINTF(...)
#endif
#define OH 6
#define OW 6
#define K 9
#define OC 32
#define IH 8
#define DMA_CH 1
static inline float w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
float fpad[IH*IH] __attribute__((aligned(4)));   // input -> float (bir kez)
float wf[OC*K]    __attribute__((aligned(4)));   // weights -> float (bir kez)
float patch[K]    __attribute__((aligned(4)));
float out_px[OC]  __attribute__((aligned(4)));   // GEMV çıktısı (M=32)
float dummy[2]    __attribute__((aligned(4)));   // LOAD hedefi (accel çıktı üretmez)
float acc_out[OH*OW*OC], cpu_out[OH*OW*OC];
dma_target_t ts, td; dma_trans_t tr;
static inline void xfer(const void* s, void* d, uint16_t n){
    ts.ptr=(uint8_t*)s; td.ptr=(uint8_t*)d; tr.size_d1_du=n;
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
}
int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    for(int i=0;i<IH*IH;i++) fpad[i]=w2f(conv_input_padded[i]);   // ADİL: bir kez
    for(int i=0;i<OC*K;i++)  wf[i]  =w2f(conv_weights[i]);

    // --- DMA transaction kurulumu (BİR KEZ) — hw_fifo + inc + channel + init + validate ---
    ts=(dma_target_t){.ptr=(uint8_t*)patch,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=K,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    unsigned int c1, cyc_accel, cyc_cpu;
    // --- ACCEL ---
    CSR_WRITE(CSR_REG_MCYCLE,0);
    for(int oh=0;oh<OH;oh++) for(int ow=0;ow<OW;ow++){
        for(int ki=0;ki<3;ki++) for(int kj=0;kj<3;kj++)
            patch[ki*3+kj]=fpad[(oh+ki)*IH+(ow+kj)];
        xfer(patch, dummy, K);              // LOAD: x_buf <- patch (N=9)
        xfer(wf, out_px, (uint16_t)(OC*K)); // GEMV: out_px[oc]=dot(patch,W[oc]), M=32
        for(int oc=0;oc<OC;oc++) acc_out[(oh*OW+ow)*OC+oc]=out_px[oc]+w2f(conv_biases[oc]);
    }
    CSR_READ(CSR_REG_MCYCLE,&c1); cyc_accel=c1;

    // --- CPU (adil: float dizilerden, füzyonlu) ---
    CSR_WRITE(CSR_REG_MCYCLE,0);
    for(int oh=0;oh<OH;oh++) for(int ow=0;ow<OW;ow++) for(int oc=0;oc<OC;oc++){
        float s=0.0f;
        for(int ki=0;ki<3;ki++) for(int kj=0;kj<3;kj++)
            s = fmaf(fpad[(oh+ki)*IH+(ow+kj)], wf[oc*K+ki*3+kj], s);  // FÜZYONLU (accel gibi)
        cpu_out[(oh*OW+ow)*OC+oc] = s + w2f(conv_biases[oc]);
    }
    CSR_READ(CSR_REG_MCYCLE,&cyc_cpu);

    // --- verify ---
    int nbad=0, worst=-1; float e_cpu=0, e_gold=0;
    for(int i=0;i<OH*OW*OC;i++){
        float a=acc_out[i];
        float ec=fabsf(a-cpu_out[i]);            if(ec>e_cpu)  e_cpu=ec;
        float eg=fabsf(a-w2f(conv_expected[i])); if(eg>e_gold) e_gold=eg;
        if(ec>1e-4f){ nbad++; if(worst<0)worst=i; }
    }
    PRINTF("conv1 FULL: accel=%u cpu=%u  SPEEDUP=%u.%02ux\n",
           cyc_accel, cyc_cpu, cyc_cpu/cyc_accel, (cyc_cpu*100/cyc_accel)%100);
    PRINTF("err_vs_CPU=%d/1e6  err_vs_golden=%d/1e6  nbad=%d/%d\n",
           (int)(e_cpu*1e6f), (int)(e_gold*1e6f), nbad, OH*OW*OC);
    if(worst>=0){ int px=worst/OC, oc=worst%OC;
        PRINTF("bad px(%d,%d) oc=%d: accel=%d cpu=%d golden=%d (x1e5)\n",
            px/OW, px%OW, oc,
            (int)(acc_out[worst]*1e5f),
            (int)(cpu_out[worst]*1e5f),
            (int)(w2f(conv_expected[worst])*1e5f));
    }
    return (e_cpu<1e-4f)?0:-1;
}

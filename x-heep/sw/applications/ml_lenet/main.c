// LeNet-300-100 on MNIST, forward pass with the shared-FMA GEMV coprocessor.
// Per layer: LOAD [N,M,x] into the accel, stream the weight matrix (row-chunked to
// the DMA's 16-bit size limit), get M dots; add bias + ReLU (hidden layers).
// The CPU (fused) forward is run ONCE (image 0) for the bit-exact check + speedup;
// the rest verify against the NumPy reference prediction (cheap). Bit-exactness of
// the shared FMA means one on-device image already proves per-input correctness.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "lenet_mnist_data.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define DMA_CH 1
#define NIMG 5                 // sınıflandırılacak görüntü (<= N_TEST). CPU forward sadece img 0.

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

float a0[L1_N], a1[L1_M], a2[L2_M], logits[L3_M];
float c1[L1_M], c2[L2_M], clog[L3_M];
uint32_t loadbuf[L1_N + 2];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;

static inline void xfer(const void* s, void* d, uint32_t n){   // n <= 65535 (HW SIZE_D1 = 16-bit)
    ts.ptr=(uint8_t*)s; td.ptr=(uint8_t*)d; tr.size_d1_du=n;
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
}
// accel GEMV: out[m] = dot(x[0..N-1], W[m][0..N-1]). Rows chunked so every transfer
// <= 65535 words; each row is an independent dot of the same buffered x -> exact.
static void gemv_accel(const unsigned int* W, const float* x, int N, int M, float* out){
    int max_rows = 65535 / N;  if(max_rows < 1) max_rows = 1;
    loadbuf[0]=(uint32_t)N;
    for(int i=0;i<N;i++) loadbuf[2+i]=f2u(x[i]);
    for(int done=0; done<M; ){
        int g = M - done; if(g > max_rows) g = max_rows;
        loadbuf[1]=(uint32_t)g;
        xfer(loadbuf, dummy, (uint32_t)(N+2));                 // LOAD: x_buf + [N,g]
        xfer(&W[done*N], &out[done], (uint32_t)(g*N));         // GEMV: this row group
        done += g;
    }
}
static void gemv_cpu(const unsigned int* W, const float* x, int N, int M, float* out){
    for(int m=0;m<M;m++){ float s=0.0f;
        for(int n=0;n<N;n++) s=fmaf(x[n], w2f(W[m*N+n]), s);
        out[m]=s; }
}
static void relu(float* v, const unsigned int* b, int M){
    for(int m=0;m<M;m++){ v[m]+=w2f(b[m]); if(v[m]<0) v[m]=0.0f; }
}
static int argmax(const float* v, int n){ int b=0; for(int i=1;i<n;i++) if(v[i]>v[b]) b=i; return b; }

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=L1_N+2,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    unsigned int cyc_accel=0, cyc_cpu=0, t0, t1;
    int correct=0, match_ref=0, bitexact=1;

    for(int k=0;k<NIMG;k++){
        for(int i=0;i<L1_N;i++) a0[i]=w2f(test_images[k*L1_N + i]);

        if(k==0) CSR_READ(CSR_REG_MCYCLE,&t0);
        gemv_accel(fc1_w, a0, L1_N, L1_M, a1);  relu(a1, fc1_b, L1_M);
        gemv_accel(fc2_w, a1, L2_N, L2_M, a2);  relu(a2, fc2_b, L2_M);
        gemv_accel(fc3_w, a2, L3_N, L3_M, logits);
        for(int m=0;m<L3_M;m++) logits[m]+=w2f(fc3_b[m]);
        if(k==0){ CSR_READ(CSR_REG_MCYCLE,&t1); cyc_accel=t1-t0; }
        int pred = argmax(logits, L3_M);

        if(k==0){                                    // CPU (fused) forward: bit-exact + speedup, once
            CSR_READ(CSR_REG_MCYCLE,&t0);
            gemv_cpu(fc1_w, a0, L1_N, L1_M, c1);  relu(c1, fc1_b, L1_M);
            gemv_cpu(fc2_w, c1, L2_N, L2_M, c2);  relu(c2, fc2_b, L2_M);
            gemv_cpu(fc3_w, c2, L3_N, L3_M, clog);
            for(int m=0;m<L3_M;m++) clog[m]+=w2f(fc3_b[m]);
            CSR_READ(CSR_REG_MCYCLE,&t1); cyc_cpu=t1-t0;
            for(int m=0;m<L3_M;m++) if(logits[m]!=clog[m]) bitexact=0;
        }

        if(pred==test_labels[k]) correct++;
        if(pred==ref_pred[k])    match_ref++;
        PRINTF("img %d: label=%d pred=%d ref=%d%s\n",
               k, test_labels[k], pred, ref_pred[k], (k==0)?(bitexact?"  [bit-exact vs CPU]":"  [BITMISS]"):"");
    }

    PRINTF("LeNet-300-100 MNIST: correct=%d/%d  match_ref=%d/%d  (img0 bit-exact=%d)\n",
           correct,NIMG, match_ref,NIMG, bitexact);
    PRINTF("cycles (1 img): accel=%u cpu=%u  SPEEDUP=%u.%02ux  [no 2nd FPU]\n",
           cyc_accel, cyc_cpu, cyc_cpu/cyc_accel, (cyc_cpu*100/cyc_accel)%100);
    return (correct==NIMG && bitexact) ? 0 : -1;
}

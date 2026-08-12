// LeNet-300-100 on MNIST via the shared-FMA GEMV coprocessor — 2D SINGLE-TRANSFER version.
// Each layer: LOAD [N,M,x] once (1D), then stream the WHOLE M×N weight matrix in ONE 2D DMA
// transfer (size_d1=N inner, size_d2=M outer) — no row-chunking, x-buffer filled once.
//
// The accel forward (image 0) is instrumented into a cycle decomposition (nothing excluded):
//   setup : DMA descriptor programming (validate + load_transaction), per transfer  (CPU)
//   load  : LOAD transfer wall time (x streamed into the accel's x-buffer)
//   gemv  : GEMV transfer wall time (weights streamed + FMA + writeback, overlapped)
//   act   : ReLU + bias between layers (CPU)
//   -> setup + load + gemv + act == accel forward total.
// (fetch/execute/writeback overlap INSIDE 'gemv' and are not separable from C — that base is
//  the streaming-compute cost; contention & FMA-latency are isolated in ml_coexec / L-sweeps.)
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "lenet_mnist_data.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define DMA_CH 1
#define NIMG 5                 // classified images (<= N_TEST). CPU forward only for img 0.

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

float a0[L1_N], a1[L1_M], a2[L2_M], logits[L3_M];
float c1[L1_M], c2[L2_M], clog[L3_M];
uint32_t loadbuf[L1_N + 2];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;

// --- cycle-decomposition accumulators (accumulated over a forward pass) ---
unsigned int cyc_setup, cyc_load, cyc_gemv;
int          g_measure;        // only accumulate for the instrumented image

// accel GEMV: out[m] = dot(x[0..N-1], W[m][0..N-1]) for m in 0..M-1.
// x LOADed once (1D), then the WHOLE M×N weight matrix in ONE 2D transfer.
static void gemv_accel_2d(const unsigned int* W, const float* x, int N, int M, float* out){
    unsigned int t0,t1;
    loadbuf[0]=(uint32_t)N; loadbuf[1]=(uint32_t)M;
    for(int i=0;i<N;i++) loadbuf[2+i]=f2u(x[i]);

    // --- program + LOAD (1D): x-buffer filled once ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts.ptr=(uint8_t*)loadbuf; ts.inc_d1_du=1; ts.inc_d2_du=0;
    td.ptr=(uint8_t*)dummy;   td.inc_d1_du=1; td.inc_d2_du=0;
    tr.dim=DMA_DIM_CONF_1D; tr.size_d1_du=(uint32_t)(N+2); tr.size_d2_du=0;
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);
    dma_load_transaction(&tr);
    CSR_READ(CSR_REG_MCYCLE,&t1); if(g_measure) cyc_setup += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    dma_launch(&tr); while(!dma_is_ready(DMA_CH));
    CSR_READ(CSR_REG_MCYCLE,&t1); if(g_measure) cyc_load += (t1-t0);

    // --- program + GEMV (2D): whole M×N weights in ONE transfer ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts.ptr=(uint8_t*)W;   ts.inc_d1_du=1; ts.inc_d2_du=1;
    td.ptr=(uint8_t*)out; td.inc_d1_du=1; td.inc_d2_du=1;      // linear dst -> M results to out[0..M-1]
    tr.dim=DMA_DIM_CONF_2D; tr.size_d1_du=(uint32_t)N; tr.size_d2_du=(uint32_t)M;
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);
    dma_load_transaction(&tr);
    CSR_READ(CSR_REG_MCYCLE,&t1); if(g_measure) cyc_setup += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    dma_launch(&tr); while(!dma_is_ready(DMA_CH));
    CSR_READ(CSR_REG_MCYCLE,&t1); if(g_measure) cyc_gemv += (t1-t0);
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
    setvbuf(stdout, NULL, _IONBF, 0);   // canlı ilerleme (buffer beklemez)
    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=L1_N+2,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);

    unsigned int cyc_accel=0, cyc_cpu=0, t0, t1;
    unsigned int rep_setup=0, rep_load=0, rep_gemv=0;
    int correct=0, match_ref=0, bitexact=1;

    for(int k=0;k<NIMG;k++){
        for(int i=0;i<L1_N;i++) a0[i]=w2f(test_images[k*L1_N + i]);

        g_measure = (k==0);
        if(k==0){ cyc_setup=0; cyc_load=0; cyc_gemv=0; CSR_READ(CSR_REG_MCYCLE,&t0); }
        gemv_accel_2d(fc1_w, a0, L1_N, L1_M, a1);  relu(a1, fc1_b, L1_M);
        gemv_accel_2d(fc2_w, a1, L2_N, L2_M, a2);  relu(a2, fc2_b, L2_M);
        gemv_accel_2d(fc3_w, a2, L3_N, L3_M, logits);
        for(int m=0;m<L3_M;m++) logits[m]+=w2f(fc3_b[m]);
        if(k==0){ CSR_READ(CSR_REG_MCYCLE,&t1); cyc_accel=t1-t0;
                  rep_setup=cyc_setup; rep_load=cyc_load; rep_gemv=cyc_gemv; }
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

    // --- cycle decomposition of the accel forward (image 0), nothing excluded ---
    unsigned int rep_act = cyc_accel - (rep_setup + rep_load + rep_gemv);   // relu+bias+argmax(CPU)
    PRINTF("=== accel forward decomposition (img0), total=%u cyc ===\n", cyc_accel);
    PRINTF("  setup(DMA program) = %8u  (%u%%)\n", rep_setup, rep_setup*100/cyc_accel);
    PRINTF("  load (x stream)    = %8u  (%u%%)\n", rep_load,  rep_load *100/cyc_accel);
    PRINTF("  gemv (compute)     = %8u  (%u%%)\n", rep_gemv,  rep_gemv *100/cyc_accel);
    PRINTF("  act  (relu+bias)   = %8u  (%u%%)\n", rep_act,   rep_act  *100/cyc_accel);
    return (correct==NIMG && bitexact) ? 0 : -1;
}

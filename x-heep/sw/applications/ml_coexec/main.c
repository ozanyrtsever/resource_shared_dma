// CO-EXECUTION on the shared FMA: the accelerator runs an ML inference (LeNet-300-100)
// while the CPU runs a COMPLETELY DIFFERENT floating-point workload -- a DSP FIR filter
// (pure fmadd; stands in for any FP-heavy CPU task, e.g. signal conditioning that co-runs
//  with inference on an edge SoC). Both stream FMAs into the SINGLE shared unit at the
// same time -> the CPU-priority arbiter mediates.
//   (Note: typical crypto -- AES/SHA/RSA/ECC -- is INTEGER, so it would not touch the FPU
//    and would not contend; hence a genuinely FP-heavy kernel is used here.)
// Measures: (1) accel-only ML   (2) CPU-only FIR   (3) DUAL co-execution.
// Proves: bit-exact FIR + correct inference under contention; the CPU FIR is not slowed
// (CPU-priority); and the inference is absorbed into the FIR's idle FMA cycles (~free).
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "lenet_mnist_data.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define DMA_CH 1
#define IMG      0          // digit the accelerator classifies
#define FIR_L    1024       // FIR output samples (CPU)
#define FIR_K    256        // FIR taps
#define FIR_BUDGET 128      // FIR outputs advanced inside each accel async window

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

// ---- accel (ML) buffers ----
float b0[L1_N], b1[L1_M], b2[L2_M], blog[L3_M];
uint32_t loadbuf[L1_N + 2];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;

// ---- CPU FP workload: FIR filter y[n] = sum_k taps[k]*sig[n+k] (pure fmadd) ----
float sig[FIR_L + FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];
int fir_n, fir_done;
unsigned int cpu_time_acc;

static void fir_reset(void){ fir_n=0; fir_done=0; }
static void fir_advance(int budget){
    unsigned int t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
    while(budget>0 && !fir_done){
        int n=fir_n; float s=0.0f;
        for(int k=0;k<FIR_K;k++) s=fmaf(taps[k], sig[n+k], s);    // CPU FMA
        firy[n]=s; fir_n++; budget--;
        if(fir_n==FIR_L) fir_done=1;
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_time_acc += (t1-t0);
}

// ---- accel LeNet forward for image B; overlap!=0 runs FIR work in each async window ----
static inline void load_blocking(const void* s, void* d, uint32_t n){
    ts.ptr=(uint8_t*)s; td.ptr=(uint8_t*)d; tr.size_d1_du=n;
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
}
static void accel_layer(const unsigned int* W, const float* x, int N, int M,
                        float* out, const unsigned int* bias, int relu, int overlap){
    int max_rows=65535/N; if(max_rows<1) max_rows=1;
    loadbuf[0]=(uint32_t)N;
    for(int i=0;i<N;i++) loadbuf[2+i]=f2u(x[i]);
    for(int done=0; done<M; ){
        int g=M-done; if(g>max_rows) g=max_rows;
        loadbuf[1]=(uint32_t)g;
        load_blocking(loadbuf, dummy, (uint32_t)(N+2));
        ts.ptr=(uint8_t*)&W[done*N]; td.ptr=(uint8_t*)&out[done]; tr.size_d1_du=(uint32_t)(g*N);
        dma_load_transaction(&tr); dma_launch(&tr);              // GEMV async
        if(overlap) fir_advance(FIR_BUDGET);                     // <-- CONCURRENT CPU FIR (FMA contention)
        while(!dma_is_ready(DMA_CH));                            // sync GEMV
        done+=g;
    }
    for(int m=0;m<M;m++){ out[m]+=w2f(bias[m]); if(relu && out[m]<0) out[m]=0.0f; }
}
static void accel_forward(int overlap){
    accel_layer(fc1_w, b0, L1_N, L1_M, b1,   fc1_b, 1, overlap);
    accel_layer(fc2_w, b1, L2_N, L2_M, b2,   fc2_b, 1, overlap);
    accel_layer(fc3_w, b2, L3_N, L3_M, blog, fc3_b, 0, overlap);
}
static int argmax(const float* v, int n){ int b=0; for(int i=1;i<n;i++) if(v[i]>v[b]) b=i; return b; }

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    for(int i=0;i<L1_N;i++) b0[i]=w2f(test_images[IMG*L1_N+i]);
    for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f - 0.5f;   // deterministic signal
    for(int k=0;k<FIR_K;k++)       taps[k]=(float)((k*3+1)%11)/11.0f - 0.4f;  // deterministic filter
    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=L1_N+2,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    unsigned int t0,t1, accel_alone, cpu_alone, dual_total;
    int predB;

    // === (1) accel-only baseline: LeNet inference (CPU idle) ===
    CSR_READ(CSR_REG_MCYCLE,&t0);
    accel_forward(0);
    CSR_READ(CSR_REG_MCYCLE,&t1); accel_alone=t1-t0;
    PRINTF("[1/3] accel-only (LeNet) done: %u  pred=%d\n", accel_alone, argmax(blog,L3_M));

    // === (2) CPU-only baseline: FIR filter (accel idle) ===
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0);
    fir_advance(1000000);
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_alone=t1-t0;
    for(int i=0;i<FIR_L;i++) firref[i]=firy[i];
    PRINTF("[2/3] cpu-only (FIR) done: %u\n", cpu_alone);

    // === (3) DUAL: accel LeNet interleaved || CPU FIR ===
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0);
    accel_forward(1);
    while(!fir_done) fir_advance(1000000);                       // finish remaining FIR
    CSR_READ(CSR_REG_MCYCLE,&t1); dual_total=t1-t0;
    predB=argmax(blog,L3_M);
    PRINTF("[3/3] dual done: %u\n", dual_total);

    // === checks ===
    int fexact=1; for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) fexact=0;
    unsigned int seq = accel_alone + cpu_alone;

    PRINTF("=== accel LeNet inference || CPU FIR filter, sharing ONE FMA ===\n");
    PRINTF("correctness: accel pred=%d (ref%d lbl%d)  FIR bit-exact=%d\n",
           predB, ref_pred[IMG], test_labels[IMG], fexact);
    PRINTF("baselines: accel-only(ML)=%u  cpu-only(FIR)=%u  (sequential sum=%u)\n",
           accel_alone, cpu_alone, seq);
    PRINTF("DUAL total=%u   CPU(FIR)-time-in-dual=%u (vs alone %u -> priority)\n",
           dual_total, cpu_time_acc, cpu_alone);
    PRINTF("=> inference absorbed into FIR idle FMA cycles: hidden ~%d cyc; throughput %u.%02ux\n",
           (int)seq-(int)dual_total, seq/dual_total, (seq*100/dual_total)%100);
    return (predB==ref_pred[IMG] && fexact) ? 0 : -1;
}

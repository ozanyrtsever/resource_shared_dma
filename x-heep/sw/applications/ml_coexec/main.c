// CO-EXECUTION on the shared FMA (2D single-transfer version): the accelerator runs an ML
// inference (LeNet-300-100) while the CPU runs a DIFFERENT floating-point workload -- a DSP FIR
// filter (pure fmadd, reads arrays -> also contends the bus). Both stream FMAs into the SINGLE
// shared unit -> the CPU-priority arbiter mediates.
//
// Measurement (cycle decomposition, nothing excluded):
//   (1) accel-only  -> per-phase setup / load / gemv (accel forward, no contention)
//   (2) cpu-only    -> FIR baseline
//   (3) dual        -> accel GEMV under CONTENTION: the CPU FIR runs (bus-heavy) until each GEMV
//                      is ready, so gemv_dual is the accel's contended wall time.
//   MEMORY+DRAIN contention = gemv_dual - gemv_alone   (the accel's slowdown from co-running)
//   CPU-priority           = FIR time-in-dual vs alone (CPU FMA not starved; only bus cost)
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
#define FIR_CHUNK 16        // FIR outputs advanced per poll iteration while contending

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

// ---- accel (ML) buffers ----
float b0[L1_N], b1[L1_M], b2[L2_M], blog[L3_M];
uint32_t loadbuf[L1_N + 2];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;

// ---- CPU FP workload: FIR filter y[n] = sum_k taps[k]*sig[n+k] (pure fmadd, array reads) ----
float sig[FIR_L + FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];
int fir_n, fir_done;
unsigned int cpu_time_acc;

// ---- per-forward cycle accumulators ----
unsigned int g_setup, g_load, g_gemv;

static void fir_reset(void){ fir_n=0; fir_done=0; }
static void fir_advance(int budget){
    unsigned int t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
    while(budget>0 && !fir_done){
        int n=fir_n; float s=0.0f;
        for(int k=0;k<FIR_K;k++) s=fmaf(taps[k], sig[n+k], s);    // CPU FMA + bus reads
        firy[n]=s; fir_n++; budget--;
        if(fir_n==FIR_L) fir_done=1;
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_time_acc += (t1-t0);
}

// accel layer via 2D single transfer. contend!=0: CPU FIR runs (bus-heavy) until the GEMV is
// ready -> the gemv bracket then measures the accel's CONTENDED wall time.
static void accel_layer_2d(const unsigned int* W, const float* x, int N, int M,
                           float* out, const unsigned int* bias, int relu, int contend){
    unsigned int t0,t1;
    loadbuf[0]=(uint32_t)N; loadbuf[1]=(uint32_t)M;
    for(int i=0;i<N;i++) loadbuf[2+i]=f2u(x[i]);

    // program + LOAD (1D)
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts.ptr=(uint8_t*)loadbuf; ts.inc_d1_du=1; ts.inc_d2_du=0;
    td.ptr=(uint8_t*)dummy;   td.inc_d1_du=1; td.inc_d2_du=0;
    tr.dim=DMA_DIM_CONF_1D; tr.size_d1_du=(uint32_t)(N+2); tr.size_d2_du=0;
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);
    dma_load_transaction(&tr);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    dma_launch(&tr); while(!dma_is_ready(DMA_CH));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load += (t1-t0);

    // program + GEMV (2D whole matrix)
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts.ptr=(uint8_t*)W;   ts.inc_d1_du=1; ts.inc_d2_du=1;
    td.ptr=(uint8_t*)out; td.inc_d1_du=1; td.inc_d2_du=1;
    tr.dim=DMA_DIM_CONF_2D; tr.size_d1_du=(uint32_t)N; tr.size_d2_du=(uint32_t)M;
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);
    dma_load_transaction(&tr);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    dma_launch(&tr);
    if(contend){ while(!dma_is_ready(DMA_CH)){ if(!fir_done) fir_advance(FIR_CHUNK); } }
    else       { while(!dma_is_ready(DMA_CH)); }
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv += (t1-t0);

    for(int m=0;m<M;m++){ out[m]+=w2f(bias[m]); if(relu && out[m]<0) out[m]=0.0f; }
}
static void accel_forward_2d(int contend){
    g_setup=0; g_load=0; g_gemv=0;
    accel_layer_2d(fc1_w, b0, L1_N, L1_M, b1,   fc1_b, 1, contend);
    accel_layer_2d(fc2_w, b1, L2_N, L2_M, b2,   fc2_b, 1, contend);
    accel_layer_2d(fc3_w, b2, L3_N, L3_M, blog, fc3_b, 0, contend);
}
static int argmax(const float* v, int n){ int b=0; for(int i=1;i<n;i++) if(v[i]>v[b]) b=i; return b; }

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    setvbuf(stdout, NULL, _IONBF, 0);   // canlı ilerleme (buffer beklemez)
    for(int i=0;i<L1_N;i++) b0[i]=w2f(test_images[IMG*L1_N+i]);
    for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f - 0.5f;
    for(int k=0;k<FIR_K;k++)       taps[k]=(float)((k*3+1)%11)/11.0f - 0.4f;
    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=L1_N+2,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);

    unsigned int t0,t1, accel_alone, cpu_alone, dual_total;
    unsigned int setup_a, load_a, gemv_alone, gemv_dual;
    int predB;

    // === (1) accel-only baseline: per-phase decomposition, no contention ===
    CSR_READ(CSR_REG_MCYCLE,&t0);
    accel_forward_2d(0);
    CSR_READ(CSR_REG_MCYCLE,&t1); accel_alone=t1-t0;
    setup_a=g_setup; load_a=g_load; gemv_alone=g_gemv;
    PRINTF("[1/3] accel-only (LeNet 2D)=%u pred=%d | setup=%u load=%u gemv=%u\n",
           accel_alone, argmax(blog,L3_M), setup_a, load_a, gemv_alone);

    // === (2) CPU-only baseline: FIR filter (accel idle) ===
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0);
    fir_advance(1000000);
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_alone=t1-t0;
    for(int i=0;i<FIR_L;i++) firref[i]=firy[i];
    PRINTF("[2/3] cpu-only (FIR)=%u\n", cpu_alone);

    // === (3) DUAL: accel LeNet || CPU FIR (contend until each GEMV ready) ===
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0);
    accel_forward_2d(1);
    while(!fir_done) fir_advance(1000000);                       // finish remaining FIR
    CSR_READ(CSR_REG_MCYCLE,&t1); dual_total=t1-t0;
    gemv_dual=g_gemv;
    predB=argmax(blog,L3_M);
    PRINTF("[3/3] dual=%u | gemv-in-dual=%u\n", dual_total, gemv_dual);

    // === checks + decomposition ===
    int fexact=1; for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) fexact=0;
    unsigned int seq = accel_alone + cpu_alone;
    int contention = (int)gemv_dual - (int)gemv_alone;          // accel slowdown from co-running

    PRINTF("=== accel LeNet inference || CPU FIR filter, sharing ONE FMA ===\n");
    PRINTF("correctness: accel pred=%d (ref%d lbl%d)  FIR bit-exact=%d\n",
           predB, ref_pred[IMG], test_labels[IMG], fexact);
    PRINTF("--- accel-forward decomposition (alone): total=%u ---\n", accel_alone);
    PRINTF("  setup=%u  load=%u  gemv=%u\n", setup_a, load_a, gemv_alone);
    PRINTF("--- accel GEMV contention (dual vs alone) ---\n");
    PRINTF("  gemv_alone=%u  gemv_dual=%u  -> MEMORY+DRAIN=%d (%d%%)\n",
           gemv_alone, gemv_dual, contention, gemv_alone? contention*100/(int)gemv_alone : 0);
    PRINTF("--- CPU-priority (FIR) ---\n");
    PRINTF("  FIR time-in-dual=%u vs alone=%u  (delta=%d -> bus, not FMA-starvation)\n",
           cpu_time_acc, cpu_alone, (int)cpu_time_acc-(int)cpu_alone);
    PRINTF("throughput: seq=%u dual=%u -> hidden~%d, %u.%02ux\n",
           seq, dual_total, (int)seq-(int)dual_total, seq/dual_total, (seq*100/dual_total)%100);
    return (predB==ref_pred[IMG] && fexact) ? 0 : -1;
}

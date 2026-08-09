// LeNet-300-100 DUAL-STREAM on the shared FMA.
// The CPU classifies image A while the accelerator classifies image B, CONCURRENTLY,
// both issuing FMAs into the SINGLE shared unit -> the CPU-priority arbiter mediates.
// The CPU alone uses the FMA only ~10% of the time (~10 cyc/FMA of load/index/loop
// overhead); the accelerator slots image B's dot products into those idle FMA cycles.
// Measures three things for a clean comparison:
//   (1) accel-only forward (image B)        -> baseline
//   (2) CPU-only forward   (image A)        -> baseline
//   (3) DUAL: accel B interleaved || CPU A  -> concurrency
// Proves: bit-exact results under contention; the CPU is not slowed (priority);
// and the second inference is delivered almost for free (hidden in idle FMA cycles).
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "lenet_mnist_data.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define DMA_CH 1
#define IMG_A 0            // CPU classifies this MNIST digit
#define IMG_B 1            // accel classifies this one
#define CPU_BUDGET 64      // CPU rows advanced inside each accel async window

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

float a0[L1_N], a1[L1_M], a2[L2_M], alog[L3_M];      // image A activations (CPU)
float b0[L1_N], b1[L1_M], b2[L2_M], blog[L3_M];      // image B activations (accel)
float aref[L3_M], bref[L3_M];                        // saved baseline logits (bit-exact check)
uint32_t loadbuf[L1_N + 2];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;

// ---- resumable CPU forward for image A (advance by a row budget) ----
const unsigned int* AW[3]; const unsigned int* AB[3];
int AN[3], AM[3], A_relu[3];
float* AIN[3]; float* AOUT[3];
int a_layer, a_row, a_done;
unsigned int cpu_time_acc;

static void cpu_reset(void){ a_layer=0; a_row=0; a_done=0; }
static void cpu_advance(int budget){
    unsigned int t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
    while(budget>0 && !a_done){
        const unsigned int* W=AW[a_layer]; const float* x=AIN[a_layer]; int N=AN[a_layer];
        float s=0.0f;
        for(int i=0;i<N;i++) s=fmaf(x[i], w2f(W[a_row*N+i]), s);   // CPU FMA
        AOUT[a_layer][a_row]=s; a_row++; budget--;
        if(a_row==AM[a_layer]){                                   // layer done -> bias + ReLU
            for(int m=0;m<AM[a_layer];m++){
                AOUT[a_layer][m]+=w2f(AB[a_layer][m]);
                if(A_relu[a_layer] && AOUT[a_layer][m]<0) AOUT[a_layer][m]=0.0f;
            }
            a_layer++; a_row=0; if(a_layer==3) a_done=1;
        }
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_time_acc += (t1-t0);
}

// ---- accel forward for image B; when overlap!=0, run CPU-A work in each async window ----
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
        load_blocking(loadbuf, dummy, (uint32_t)(N+2));               // LOAD x_buf (short, blocking)
        ts.ptr=(uint8_t*)&W[done*N]; td.ptr=(uint8_t*)&out[done]; tr.size_d1_du=(uint32_t)(g*N);
        dma_load_transaction(&tr); dma_launch(&tr);                   // GEMV: launch async
        if(overlap) cpu_advance(CPU_BUDGET);                          // <-- CONCURRENT CPU-A (FMA contention)
        while(!dma_is_ready(DMA_CH));                                 // sync GEMV
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
    for(int i=0;i<L1_N;i++){ a0[i]=w2f(test_images[IMG_A*L1_N+i]); b0[i]=w2f(test_images[IMG_B*L1_N+i]); }
    // CPU-A resumable descriptors
    AW[0]=fc1_w; AW[1]=fc2_w; AW[2]=fc3_w;  AB[0]=fc1_b; AB[1]=fc2_b; AB[2]=fc3_b;
    AN[0]=L1_N; AN[1]=L2_N; AN[2]=L3_N;      AM[0]=L1_M; AM[1]=L2_M; AM[2]=L3_M;
    A_relu[0]=1; A_relu[1]=1; A_relu[2]=0;   AIN[0]=a0; AIN[1]=a1; AIN[2]=a2;
    AOUT[0]=a1; AOUT[1]=a2; AOUT[2]=alog;
    // DMA setup
    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=L1_N+2,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    unsigned int t0,t1, accel_alone, cpu_alone, dual_total;
    int predB_alone, predA_alone, predA, predB;

    // === (1) accel-only baseline (image B) ===
    CSR_READ(CSR_REG_MCYCLE,&t0);
    accel_forward(0);
    CSR_READ(CSR_REG_MCYCLE,&t1); accel_alone=t1-t0;
    predB_alone=argmax(blog,L3_M); for(int m=0;m<L3_M;m++) bref[m]=blog[m];
    PRINTF("[1/3] accel-only done: %u\n", accel_alone);

    // === (2) CPU-only baseline (image A) ===
    cpu_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0);
    cpu_advance(1000000);                       // advance to completion
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_alone=t1-t0;
    predA_alone=argmax(alog,L3_M); for(int m=0;m<L3_M;m++) aref[m]=alog[m];
    PRINTF("[2/3] cpu-only done: %u\n", cpu_alone);

    // === (3) DUAL: accel B interleaved || CPU A ===
    cpu_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0);
    accel_forward(1);                           // drives accel B + cpu_advance windows
    while(!a_done) cpu_advance(1000000);        // finish remaining CPU-A after accel done
    CSR_READ(CSR_REG_MCYCLE,&t1); dual_total=t1-t0;
    predA=argmax(alog,L3_M); predB=argmax(blog,L3_M);
    PRINTF("[3/3] dual done: %u\n", dual_total);

    // === checks ===
    int aexact=1,bexact=1;
    for(int m=0;m<L3_M;m++){ if(alog[m]!=aref[m]) aexact=0; if(blog[m]!=bref[m]) bexact=0; }
    unsigned int seq = accel_alone + cpu_alone;

    PRINTF("=== LeNet dual-stream (CPU=img%d, accel=img%d), sharing ONE FMA ===\n", IMG_A, IMG_B);
    PRINTF("correctness: A pred=%d(ref%d lbl%d) B pred=%d(ref%d lbl%d)  bitexact A=%d B=%d\n",
           predA, ref_pred[IMG_A], test_labels[IMG_A], predB, ref_pred[IMG_B], test_labels[IMG_B],
           aexact, bexact);
    PRINTF("baselines: accel-only=%u  cpu-only=%u  (sequential sum=%u)\n", accel_alone, cpu_alone, seq);
    PRINTF("DUAL total=%u   CPU-time-in-dual=%u (vs alone %u -> priority)\n",
           dual_total, cpu_time_acc, cpu_alone);
    PRINTF("=> 2 inferences in %u cyc vs %u sequential; accel B hidden ~%d cyc; throughput %u.%02ux\n",
           dual_total, seq, (int)seq-(int)dual_total, seq/dual_total, (seq*100/dual_total)%100);
    return (predA==ref_pred[IMG_A] && predB==ref_pred[IMG_B] && aexact && bexact) ? 0 : -1;
}

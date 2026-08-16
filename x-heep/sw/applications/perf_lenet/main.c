// perf_lenet — the SAME benchmark as perf_bench, but on the REAL LeNet-300-100 / MNIST network
// (784->300->100->10) with real trained weights and real MNIST test images.
//
// Six scenarios, one metric block each (identical structure to perf_bench):
//   [A] CPU alone, 8 images (the golden reference)
//   [B] one coprocessor, B=1  (GEMV, memory-bound)
//   [C] one coprocessor, B=8  (GEMM, compute-bound)     -> batching saturates the shared FMA
//   [D] one coprocessor + the CPU's FIR                 -> arbiter contention (2 requestors)
//   [E] two coprocessors (acc0+acc1, rows split)        -> dual throughput + channel balance
//   [F] two coprocessors + the CPU's FIR                -> arbiter contention (3 requestors)
// All bit-exact vs the CPU golden; also checks the batched predictions against the MNIST labels.
//
// THE ONE STRUCTURAL DIFFERENCE from perf_bench: LeNet layer-1 has M*N = 300*784 = 235200 weights, which
// exceeds the DMA's 16-bit per-dimension SIZE limit (65535). So the weight matrix is streamed as a single
// 2D transfer (size_d1 = N inner, size_d2 = M outer, inc = 1,1 -> contiguous), instead of the 1D
// size_d1 = M*N used by perf_bench. Same total order of weights, just delivered 2D. (Proven in ml_lenet.)
//
// REQUIRES the batched accelerator (LOAD header [N,M,B,x...], x_buf[MAXB][MAXN], MAXN>=784) on ch1/ch2.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "lenet_mnist_data.h"          // L1_N..L3_M, fc{1,2,3}_{w,b}, test_images/labels/ref_pred, N_TEST
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0 1                          // DMA channel of accelerator 0 (acc0)
#define CH1 2                          // DMA channel of accelerator 1 (acc1)
#define BATCH 8                        // batch size B (must be <= N_TEST = number of stored MNIST images)

// LeNet-300-100 layer dimensions, taken from the data header (N = input length, M = output length)
#define N1 L1_N   // 784   (28x28 MNIST image, flattened)
#define M1 L1_M   // 300
#define N2 L2_N   // 300
#define M2 L2_M   // 100
#define N3 L3_N   // 100
#define M3 L3_M   // 10    (one logit per digit class)
#define TOT_MAC (M1*N1 + M2*N2 + M3*N3)     // multiply-accumulates per image = 266 200 (~25x the toy MLP)

// the CPU's own DSP job used to create arbiter contention: a 32-tap FIR over 128 outputs
#define FIR_L 128
#define FIR_K 32
#define FIR_CHUNK 8

// bit-reinterpret helpers: the accelerator streams raw 32-bit words, and the header's weights are stored
// as uint32 bit patterns, so we convert between float and its bits as needed.
static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }  // bits -> float
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }      // float -> bits

// The trained weights/biases (fc1_w..fc3_b) and images (test_images) live in the header as `const unsigned
// int` (rodata) and are streamed straight from there; only the writable working buffers are declared here.
float x0[BATCH*N1];                                     // input batch, layout [B][N1]
float out1[M1*BATCH], out2[M2*BATCH], out3[M3*BATCH];   // accelerator outputs, layout [M][B]
float ga1[BATCH*M1], ga2[BATCH*M2], ga3[BATCH*M3];      // CPU golden results, layout [B][M]
uint32_t loadbuf0[3 + BATCH*N1], loadbuf1[3 + BATCH*N1];// LOAD packets: [N,M,B, x...] for acc0 / acc1
float dummy0[4], dummy1[4];                             // dummy DMA destinations for the LOAD phase
dma_target_t ts0, td0, ts1, td1;                        // DMA src/dst descriptors (channel 0 / 1)
dma_trans_t  tr0, tr1;                                  // DMA transactions (channel 0 / 1)
float sig[FIR_L+FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];  // FIR input / taps / output / reference
int fir_n, fir_done;                                   // FIR progress index / done flag
unsigned int cpu_time_acc;                             // accumulated CPU cycles spent inside the FIR
unsigned int g_setup, g_load, g_gemv, g_imbal;         // per-phase cycle accumulators (reset each forward)

static void fir_reset(void){ fir_n=0; fir_done=0; }    // restart the FIR from output 0

// Advance the CPU FIR by up to `budget` outputs (each = 32 fmaf); time is added to cpu_time_acc. Called
// interleaved with the accelerator's completion polling so the CPU's FMA use contends with the accelerator.
static void fir_advance(int budget){
    unsigned int t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
    while(budget>0 && !fir_done){
        int n=fir_n; float s=0.0f;
        for(int k=0;k<FIR_K;k++) s=fmaf(taps[k], sig[n+k], s); // one FIR output = 32 fused multiply-adds
        firy[n]=s; fir_n++; budget--;
        if(fir_n==FIR_L) fir_done=1;
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_time_acc += (t1-t0);
}

// Apply bias then optional ReLU to an accelerator output block Y[M][B]. The bias comes from the header as
// uint32 bit patterns, so it is converted with w2f (this is a LeNet-vs-perf_bench difference).
static void relu_bias(float* Y, const unsigned int* bias, int M, int B, int relu){
    for(int m=0;m<M;m++) for(int b=0;b<B;b++){
        Y[m*B+b] += w2f(bias[m]);
        if(relu && Y[m*B+b]<0.0f) Y[m*B+b]=0.0f;
    }
}

// CPU golden: B independent LeNet forwards on the CPU's own FMA, using the real weights (w2f-decoded).
static void golden_forward(int B){
    for(int b=0;b<B;b++){
        for(int m=0;m<M1;m++){ float s=0.0f; for(int i=0;i<N1;i++) s=fmaf(x0[b*N1+i], w2f(fc1_w[m*N1+i]), s);
            ga1[b*M1+m]=s+w2f(fc1_b[m]); if(ga1[b*M1+m]<0.0f) ga1[b*M1+m]=0.0f; }   // layer1 + relu
        for(int m=0;m<M2;m++){ float s=0.0f; for(int i=0;i<N2;i++) s=fmaf(ga1[b*M1+i], w2f(fc2_w[m*N2+i]), s);
            ga2[b*M2+m]=s+w2f(fc2_b[m]); if(ga2[b*M2+m]<0.0f) ga2[b*M2+m]=0.0f; }   // layer2 + relu
        for(int m=0;m<M3;m++){ float s=0.0f; for(int i=0;i<N3;i++) s=fmaf(ga2[b*M2+i], w2f(fc3_w[m*N3+i]), s);
            ga3[b*M3+m]=s+w2f(fc3_b[m]); }                                          // layer3 (no relu) -> logits
    }
}
// bit-exactness: every accelerator logit out3[m][b] must equal the CPU logit ga3[b][m] exactly.
static int bitexact(int B){ for(int b=0;b<B;b++) for(int m=0;m<M3;m++) if(out3[m*B+b]!=ga3[b*M3+m]) return 0; return 1; }
// predicted class for image b = argmax over the M logits in column b of Y[M][B].
static int argmax_col(const float* Y, int b, int B, int M){ int best=0; for(int m=1;m<M;m++) if(Y[m*B+b]>Y[best*B+b]) best=m; return best; }

// ---- ONE-COPROCESSOR batched layer: LOAD [N,M,B,x] (1D) into acc0, then stream ALL M*N weights (2D). ----
// x layout: transpose=0 -> x[b*N+i] (raw [B][N] input); transpose=1 -> x[i*B+b] (previous [M_prev][B] output).
static void fc_single(const unsigned int* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
    unsigned int t0,t1;
    loadbuf0[0]=N; loadbuf0[1]=M; loadbuf0[2]=B;                // header the accelerator reads first
    for(int b=0;b<B;b++) for(int i=0;i<N;i++)                   // pack the B input vectors after the header
        loadbuf0[3+b*N+i] = transpose ? f2u(x[i*B+b]) : f2u(x[b*N+i]);
    // --- phase 1: LOAD (1D) — stream the B input vectors into the accelerator's x-buffer ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)loadbuf0; ts0.inc_d1_du=1; ts0.inc_d2_du=0;   // 1D contiguous source
    td0.ptr=(uint8_t*)dummy0;   td0.inc_d1_du=1; td0.inc_d2_du=0;
    tr0.dim=DMA_DIM_CONF_1D; tr0.size_d1_du=(uint32_t)(N*B+3); tr0.size_d2_du=0;
    dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);            // DMA-descriptor programming -> setup
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0); while(!dma_is_ready(CH0));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=(t1-t0);             // streaming the inputs in -> load
    // --- phase 2: WEIGHT stream (2D) — the WHOLE M*N matrix in ONE transfer (bypasses the 16-bit SIZE limit)
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)W;  ts0.inc_d1_du=1; ts0.inc_d2_du=1;    // 2D: N inner x M outer, inc 1,1 = contiguous
    td0.ptr=(uint8_t*)Y;  td0.inc_d1_du=1; td0.inc_d2_du=1;    // contiguous dst -> M*B results land at Y[0..M*B-1]
    tr0.dim=DMA_DIM_CONF_2D; tr0.size_d1_du=(uint32_t)N; tr0.size_d2_du=(uint32_t)M;
    dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0);
    if(contend){ while(!dma_is_ready(CH0)){ if(!fir_done) fir_advance(FIR_CHUNK); } }  // interleave the CPU FIR
    else       { while(!dma_is_ready(CH0)); }                  // or just wait for the accelerator
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv+=(t1-t0);            // the compute phase -> gemv
}

// ---- TWO-COPROCESSOR batched layer (M-split): acc0 rows 0..M0-1 (ch0), acc1 rows M0..M-1 (ch1). ----
// Both accelerators buffer the SAME B inputs; they differ only in which output rows they compute.
static void fc_dual(const unsigned int* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
    unsigned int t0,t1, tc0=0, tc1=0;
    int M0=(M+1)/2, Mx=M-M0;                                    // acc0 gets M0 rows, acc1 gets the rest
    loadbuf0[0]=N; loadbuf0[1]=M0; loadbuf0[2]=B;               // acc0 header
    loadbuf1[0]=N; loadbuf1[1]=Mx; loadbuf1[2]=B;               // acc1 header
    for(int b=0;b<B;b++) for(int i=0;i<N;i++){                  // pack the SAME inputs into both LOAD packets
        uint32_t v = transpose ? f2u(x[i*B+b]) : f2u(x[b*N+i]);
        loadbuf0[3+b*N+i]=v; loadbuf1[3+b*N+i]=v;
    }
    // --- LOAD (1D) both channels ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)loadbuf0; ts0.inc_d1_du=1; ts0.inc_d2_du=0;
    td0.ptr=(uint8_t*)dummy0;   td0.inc_d1_du=1; td0.inc_d2_du=0;
    tr0.dim=DMA_DIM_CONF_1D; tr0.size_d1_du=(uint32_t)(N*B+3); tr0.size_d2_du=0;
    dma_load_transaction(&tr0);
    ts1.ptr=(uint8_t*)loadbuf1; ts1.inc_d1_du=1; ts1.inc_d2_du=0;
    td1.ptr=(uint8_t*)dummy1;   td1.inc_d1_du=1; td1.inc_d2_du=0;
    tr1.dim=DMA_DIM_CONF_1D; tr1.size_d1_du=(uint32_t)(N*B+3); tr1.size_d2_du=0;
    dma_load_transaction(&tr1);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0); dma_launch(&tr1);
    while(!dma_is_ready(CH0)); while(!dma_is_ready(CH1));       // wait for both LOADs
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=(t1-t0);
    // --- WEIGHT stream (2D) both channels: acc0 rows [0..M0), acc1 rows [M0..M) ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)&W[0];      ts0.inc_d1_du=1; ts0.inc_d2_du=1;
    td0.ptr=(uint8_t*)&Y[0];      td0.inc_d1_du=1; td0.inc_d2_du=1;
    tr0.dim=DMA_DIM_CONF_2D; tr0.size_d1_du=(uint32_t)N; tr0.size_d2_du=(uint32_t)M0;   // acc0: first M0 rows
    dma_load_transaction(&tr0);
    ts1.ptr=(uint8_t*)&W[M0*N];   ts1.inc_d1_du=1; ts1.inc_d2_du=1;
    td1.ptr=(uint8_t*)&Y[M0*B];   td1.inc_d1_du=1; td1.inc_d2_du=1;
    tr1.dim=DMA_DIM_CONF_2D; tr1.size_d1_du=(uint32_t)N; tr1.size_d2_du=(uint32_t)Mx;   // acc1: remaining rows
    dma_load_transaction(&tr1);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0); dma_launch(&tr1);   // both now share the ONE FMA
    { int d0=0,d1=0;
      while(!(d0&&d1)){                                          // wait for both, recording each finish time
        if(!d0 && dma_is_ready(CH0)){ d0=1; CSR_READ(CSR_REG_MCYCLE,&tc0); }  // acc0 done at tc0
        if(!d1 && dma_is_ready(CH1)){ d1=1; CSR_READ(CSR_REG_MCYCLE,&tc1); }  // acc1 done at tc1
        if(contend && !fir_done) fir_advance(FIR_CHUNK);        // optional 3rd requestor: the CPU FIR
      }
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv+=(t1-t0);
    g_imbal += (tc0>tc1)?(tc0-tc1):(tc1-tc0);                    // channel imbalance = |acc0 end - acc1 end|
}

// One full LeNet forward. dual=0 -> single accelerator (fc_single); dual=1 -> two accelerators (fc_dual).
// Between layers the previous output [M][B] feeds the next input with transpose=1. Layer 3 has no ReLU.
static void fwd(int B, int dual, int contend){
    g_setup=0; g_load=0; g_gemv=0; g_imbal=0;                    // reset per-phase accumulators
    if(dual){
        fc_dual(fc1_w, x0,   N1, M1, B, out1, 0, contend); relu_bias(out1, fc1_b, M1, B, 1);
        fc_dual(fc2_w, out1, N2, M2, B, out2, 1, contend); relu_bias(out2, fc2_b, M2, B, 1);
        fc_dual(fc3_w, out2, N3, M3, B, out3, 1, contend); relu_bias(out3, fc3_b, M3, B, 0);
    } else {
        fc_single(fc1_w, x0,   N1, M1, B, out1, 0, contend); relu_bias(out1, fc1_b, M1, B, 1);
        fc_single(fc2_w, out1, N2, M2, B, out2, 1, contend); relu_bias(out2, fc2_b, M2, B, 1);
        fc_single(fc3_w, out2, N3, M3, B, out3, 1, contend); relu_bias(out3, fc3_b, M3, B, 0);
    }
}

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));                       // enable the FPU (mstatus.FS = Initial)
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);                   // start the cycle counter (mcycle)
    setvbuf(stdout, NULL, _IONBF, 0);                            // unbuffered UART -> live progress
    // load B real MNIST images (stored as uint32 bit patterns) into the input batch [B][N1]
    for(int b=0;b<BATCH;b++) for(int i=0;i<N1;i++) x0[b*N1+i]=w2f(test_images[b*N1+i]);
    // FIR stimulus + taps (the CPU-side DSP job used in the contention scenarios)
    for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f-0.5f;
    for(int k=0;k<FIR_K;k++) taps[k]=(float)((k*3+1)%11)/11.0f-0.4f;

    // DMA descriptors: hw_fifo_en=1 routes the stream through the accelerator; both channels start as 1D.
    ts0=(dma_target_t){.ptr=(uint8_t*)loadbuf0,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td0=(dma_target_t){.ptr=(uint8_t*)dummy0,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr0=(dma_trans_t){.src=&ts0,.dst=&td0,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+3,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH0};
    ts1=(dma_target_t){.ptr=(uint8_t*)loadbuf1,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td1=(dma_target_t){.ptr=(uint8_t*)dummy1,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr1=(dma_trans_t){.src=&ts1,.dst=&td1,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+3,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH1};
    dma_init(NULL);
    dma_validate_transaction(&tr0,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);
    dma_validate_transaction(&tr1,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    unsigned int t0,t1;
    PRINTF("=== PERF LENET  LeNet-300-100 %d-%d-%d-%d  batch B=%d  (%d MAC/img, %d MAC/batch) ===\n",
           N1,M1,M2,M3, BATCH, TOT_MAC, TOT_MAC*BATCH);

    // [A] CPU golden: the CPU alone runs all BATCH images (reference for correctness AND for speedup)
    CSR_READ(CSR_REG_MCYCLE,&t0); golden_forward(BATCH); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int cpu_b = t1-t0;
    PRINTF("[A] CPU golden (%d imgs)          : %u cyc\n", BATCH, cpu_b);

    // [B] one coprocessor, B=1 (GEMV, memory-bound): one MAC per streamed weight -> ~2 cyc/MAC baseline
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(1,0,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int b1_tot=t1-t0, b1_gemv=g_gemv; int be1=bitexact(1);
    PRINTF("[B] accel B=1 (GEMV,mem-bound)    : %u cyc  bit-exact=%d\n", b1_tot, be1);
    PRINTF("    setup=%u load=%u gemv=%u  gemv/MAC=%u.%02u  FMA-util~%u%%\n",     // cyc/MAC and 1/(cyc/MAC)
           g_setup,g_load,b1_gemv, b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, TOT_MAC*100/(b1_gemv?b1_gemv:1));

    // [C] one coprocessor, B=8 (GEMM, compute-bound): each weight reused 8x -> cyc/MAC drops, FMA saturates
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(BATCH,0,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int c_tot=t1-t0, c_setup=g_setup, c_load=g_load, c_gemv=g_gemv; int bec=bitexact(BATCH);
    unsigned int cmac = TOT_MAC*BATCH;                                          // total MACs for 8 images
    PRINTF("[C] accel B=%d (GEMM,compute-bnd) : %u cyc  speedup=%u.%02ux  bit-exact=%d\n",
           BATCH, c_tot, cpu_b/c_tot,(cpu_b*100/c_tot)%100, bec);               // speedup vs CPU [A]
    PRINTF("    setup=%u load=%u gemv=%u  gemv/MAC=%u.%02u  FMA-util~%u%%\n",
           c_setup,c_load,c_gemv, c_gemv/cmac,(c_gemv*100/cmac)%100, cmac*100/(c_gemv?c_gemv:1));
    PRINTF("    -> memory->compute: gemv/MAC %u.%02u (B=1) => %u.%02u (B=%d)\n", // the batching effect
           b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, c_gemv/cmac,(c_gemv*100/cmac)%100, BATCH);

    // [D] one coprocessor (B=8) WHILE the CPU runs its FIR -> 2-way contention on the shared FMA
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0); fir_advance(1000000); CSR_READ(CSR_REG_MCYCLE,&t1);   // FIR alone -> baseline
    unsigned int fir_alone=t1-t0; for(int i=0;i<FIR_L;i++) firref[i]=firy[i];            // keep FIR reference
    fir_reset(); cpu_time_acc=0;
    fwd(BATCH,0,1); while(!fir_done) fir_advance(1000000);       // accel + FIR together, then finish the FIR
    unsigned int d_gemv=g_gemv, fir_c=cpu_time_acc; int bed=bitexact(BATCH), firok=1;
    for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) firok=0;     // FIR result must be unchanged by sharing
    int cont = (int)d_gemv-(int)c_gemv;                          // accelerator slow-down from contention
    PRINTF("[D] accel B=%d || CPU-FIR         : gemv_alone=%u gemv_cont=%u  CONTENTION=%d (%d%%)\n",
           BATCH, c_gemv, d_gemv, cont, c_gemv?cont*100/(int)c_gemv:0);
    PRINTF("    acc bit-exact=%d  FIR bit-exact=%d  FIR slow=%d%%   <-- policy-sensitive\n",   // CPU's own cost
           bed, firok, fir_alone?((int)fir_c-(int)fir_alone)*100/(int)fir_alone:0);

    // [E] two coprocessors (acc0+acc1, rows split), B=8, CPU idle -> dual throughput + channel balance
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(BATCH,1,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int e_tot=t1-t0, e_gemv=g_gemv, e_imb=g_imbal; int bee=bitexact(BATCH);
    PRINTF("[E] DUAL acc0+acc1 B=%d           : %u cyc  gemv=%u  bit-exact=%d  imbalance=%u\n",
           BATCH, e_tot, e_gemv, bee, e_imb);
    PRINTF("    gemv %u.%02ux vs single  |  gemv/MAC %u.%02u (single %u.%02u)\n",   // dual vs single speedup
           c_gemv/(e_gemv?e_gemv:1),(c_gemv*100/(e_gemv?e_gemv:1))%100,
           e_gemv/cmac,(e_gemv*100/cmac)%100, c_gemv/cmac,(c_gemv*100/cmac)%100);

    // [F] two coprocessors (B=8) WHILE the CPU runs its FIR -> 3-way contention (acc0+acc1+CPU)
    fir_reset(); cpu_time_acc=0;
    fwd(BATCH,1,1); while(!fir_done) fir_advance(1000000);
    unsigned int f_gemv=g_gemv, fir_t=cpu_time_acc; int bef=bitexact(BATCH), firok2=1;
    for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) firok2=0;
    int cont2=(int)f_gemv-(int)e_gemv;                          // dual slow-down from the CPU's contention
    PRINTF("[F] DUAL B=%d || CPU-FIR (3-way)  : gemv_alone=%u gemv_cont=%u  CONTENTION=%d (%d%%)\n",
           BATCH, e_gemv, f_gemv, cont2, e_gemv?cont2*100/(int)e_gemv:0);
    PRINTF("    acc bit-exact=%d  FIR bit-exact=%d  FIR slow=%d%%\n",
           bef, firok2, fir_alone?((int)fir_t-(int)fir_alone)*100/(int)fir_alone:0);

    // accuracy sanity: turn the batched accelerator logits into predictions and check them against the
    // MNIST labels and the offline reference predictions (proves the real network runs correctly on the accel)
    int correct=0, match_ref=0;
    for(int b=0;b<BATCH;b++){ int p=argmax_col(out3,b,BATCH,M3);
        if(p==test_labels[b]) correct++; if(p==ref_pred[b]) match_ref++; }
    PRINTF("[acc] batched preds: correct=%d/%d  match_ref=%d/%d\n", correct,BATCH, match_ref,BATCH);

    // final one-line summary (grepped by the sweep script for bit-exactness + the headline cyc/MAC numbers)
    int all = be1&&bec&&bed&&bee&&bef&&firok&&firok2;
    PRINTF("=== ALL bit-exact=%d | B=1 %u.%02u cyc/MAC -> B=%d %u.%02u cyc/MAC | dual gemv %u.%02ux ===\n",
           all, b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, BATCH, c_gemv/cmac,(c_gemv*100/cmac)%100,
           c_gemv/(e_gemv?e_gemv:1),(c_gemv*100/(e_gemv?e_gemv:1))%100);
    return all?0:-1;                                            // exit 0 only if every run was bit-exact
}

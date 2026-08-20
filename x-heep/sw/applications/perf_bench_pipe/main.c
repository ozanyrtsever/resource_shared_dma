// perf_bench_pipe — performance benchmark for the PIPELINED single shared-FMA coprocessor.
//
// Twin of perf_bench, ADAPTED to the new hardware (dma_fp_dot_accel_pipe): ONE coprocessor only, batched
// GEMM (BATCH=8). All the dual-coprocessor scenarios ([E]/[F]) are gone -- DMA channel 2 is never touched,
// so acc1 stays idle. Everything else (workload, metrics, bit-exact checks, output format) matches
// perf_bench so the numbers are directly comparable to the serial baseline.
//
// The point of the pipelined engine: it issues the B independent batch-MACs of a weight back-to-back and
// collects results as they retire, keeping the shared FMA full -> gemv/MAC ~= 1.14 for ALL latencies L
// (the serial baseline is 1.14 + L). Sweep L (fpu_addmul_lat) x policy with sweep.sh: [C] should stay flat.
//
// Scenarios (one metric block each):
//   [A] CPU alone (golden reference)                 -> baseline cycles + bit-exact reference
//   [B] the coprocessor at B=1 (serial reference)    -> runtime B=1 makes the same unit run serially
//   [C] the coprocessor at B=8 (PIPELINED GEMM)      -> the headline single-coprocessor result
//   [D] the coprocessor at B=8 + the CPU's own FIR   -> arbiter contention on the shared FMA (2 requestors)
// All bit-exact vs the CPU golden. REQUIRES the accelerator on DMA channel 1 (behind COPROC_FPU_SHARE),
// built with COPROC_PIPE (pipelined) -- or COPROC_SERIAL to measure the serial baseline with this same app.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0 1                        // DMA channel of the (single) coprocessor acc0
#define BATCH 8                      // batch size B for the compute-bound GEMM run

// ---- the 3-layer MLP dimensions (N = input length, M = output length per layer) ----
#define N1 128
#define M1 64
#define N2 64
#define M2 32
#define N3 32
#define M3 16
#define TOT_MAC (M1*N1 + M2*N2 + M3*N3)     // multiply-accumulates per image (= 10 752)

// ---- the CPU's own DSP job used to create arbiter contention: a 32-tap FIR over 128 outputs ----
#define FIR_L 128                    // number of FIR outputs to produce
#define FIR_K 32                     // FIR tap count (32 fmaf per output)
#define FIR_CHUNK 8                  // how many FIR outputs to advance per interleaved step

// bit-reinterpret helpers: the accelerator streams raw 32-bit words, so floats travel as their bit pattern
static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }  // bits -> float
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }      // float -> bits

// ---- data buffers ----
float W1[M1*N1], W2[M2*N2], W3[M3*N3];                  // weights per layer (row-major [M][N])
float Bb1[M1], Bb2[M2], Bb3[M3];                        // biases per layer
float x0[BATCH*N1];                                     // input batch, layout [B][N1]
float out1[M1*BATCH], out2[M2*BATCH], out3[M3*BATCH];   // accelerator outputs, layout [M][B]
float ga1[BATCH*M1], ga2[BATCH*M2], ga3[BATCH*M3];      // CPU golden results, layout [B][M]
uint32_t loadbuf0[3 + BATCH*N1];                        // LOAD packet: [N,M,B, x...] for acc0
float dummy0[4];                                        // dummy DMA destination for the LOAD phase
dma_target_t ts0, td0;                                  // DMA src/dst descriptors (channel 0)
dma_trans_t  tr0;                                       // DMA transaction (channel 0)
float sig[FIR_L+FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];  // FIR input / taps / output / reference
int fir_n, fir_done;                                   // FIR progress index / done flag
unsigned int cpu_time_acc;                             // accumulated CPU cycles spent inside the FIR
unsigned int g_setup, g_load, g_gemv;                  // per-phase cycle accumulators (reset each forward)

static void fir_reset(void){ fir_n=0; fir_done=0; }    // restart the FIR from output 0

// Advance the CPU FIR by up to `budget` outputs; time spent is added to cpu_time_acc. Called interleaved
// with the accelerator's completion polling so the CPU's FMA use contends with the accelerator's.
static void fir_advance(int budget){
    unsigned int t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);           // start timing this FIR chunk
    while(budget>0 && !fir_done){
        int n=fir_n; float s=0.0f;
        for(int k=0;k<FIR_K;k++) s=fmaf(taps[k], sig[n+k], s); // one FIR output = 32 fused multiply-adds
        firy[n]=s; fir_n++; budget--;
        if(fir_n==FIR_L) fir_done=1;                           // produced all FIR_L outputs
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_time_acc += (t1-t0);      // accumulate CPU-in-FIR cycles
}

// Apply bias then optional ReLU to an accelerator output block Y[M][B] (in place).
static void relu_bias(float* Y, const float* bias, int M, int B, int relu){
    for(int m=0;m<M;m++) for(int b=0;b<B;b++){
        Y[m*B+b] += bias[m];
        if(relu && Y[m*B+b]<0.0f) Y[m*B+b]=0.0f;
    }
}

// CPU golden: B independent forward passes on the CPU's own FMA. ga3[b*M3+m] = logit of image b, class m.
static void golden_forward(int B){
    for(int b=0;b<B;b++){
        for(int m=0;m<M1;m++){ float s=0.0f; for(int i=0;i<N1;i++) s=fmaf(x0[b*N1+i], W1[m*N1+i], s);
            ga1[b*M1+m]=s+Bb1[m]; if(ga1[b*M1+m]<0.0f) ga1[b*M1+m]=0.0f; }
        for(int m=0;m<M2;m++){ float s=0.0f; for(int i=0;i<N2;i++) s=fmaf(ga1[b*M1+i], W2[m*N2+i], s);
            ga2[b*M2+m]=s+Bb2[m]; if(ga2[b*M2+m]<0.0f) ga2[b*M2+m]=0.0f; }
        for(int m=0;m<M3;m++){ float s=0.0f; for(int i=0;i<N3;i++) s=fmaf(ga2[b*M2+i], W3[m*N3+i], s);
            ga3[b*M3+m]=s+Bb3[m]; }
    }
}
// bit-exactness: every accelerator logit out3[m][b] must equal the CPU logit ga3[b][m] exactly.
static int bitexact(int B){ for(int b=0;b<B;b++) for(int m=0;m<M3;m++) if(out3[m*B+b]!=ga3[b*M3+m]) return 0; return 1; }

// ---- ONE-COPROCESSOR batched layer: LOAD [N,M,B,x] into acc0, then stream the weights -> Y[M*B]. ----
// x layout: transpose=0 -> x[b*N+i] (a raw [B][N] input); transpose=1 -> x[i*B+b] (the previous layer's
// [M_prev][B] output, read so that element i of image b is x[i*B+b]).
static void layer1(const float* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
    unsigned int t0,t1;
    loadbuf0[0]=N; loadbuf0[1]=M; loadbuf0[2]=B;                // header the accelerator reads first
    for(int b=0;b<B;b++) for(int i=0;i<N;i++)                   // pack the B input vectors after the header
        loadbuf0[3+b*N+i] = transpose ? f2u(x[i*B+b]) : f2u(x[b*N+i]);
    // --- phase 1: program + run the LOAD transfer (streams the header+inputs into the accel's x-buffer) ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)loadbuf0; td0.ptr=(uint8_t*)dummy0; tr0.size_d1_du=(uint32_t)(N*B+3); dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);             // DMA-descriptor programming cost -> setup
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0); while(!dma_is_ready(CH0));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=(t1-t0);              // time to stream the inputs in -> load
    // --- phase 2: program + run the WEIGHT stream (M*N weights -> accel computes -> M*B results to Y) ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)W; td0.ptr=(uint8_t*)Y; tr0.size_d1_du=(uint32_t)(M*N); dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0);
    if(contend){ while(!dma_is_ready(CH0)){ if(!fir_done) fir_advance(FIR_CHUNK); } } // interleave CPU FIR
    else       { while(!dma_is_ready(CH0)); }                   // or just wait for the accelerator
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv+=(t1-t0);              // the compute phase -> gemv
}

// One full network forward on the SINGLE coprocessor. Between layers the previous output [M][B] is fed as
// the next input with transpose=1 (element i of image b lives at out[i*B+b]). Layer 3 has no ReLU.
static void fwd(int B, int contend){
    g_setup=0; g_load=0; g_gemv=0;                              // reset per-phase accumulators
    layer1(W1, x0,   N1, M1, B, out1, 0, contend); relu_bias(out1, Bb1, M1, B, 1);
    layer1(W2, out1, N2, M2, B, out2, 1, contend); relu_bias(out2, Bb2, M2, B, 1);
    layer1(W3, out2, N3, M3, B, out3, 1, contend); relu_bias(out3, Bb3, M3, B, 0);
}

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));                       // enable the FPU (mstatus.FS = Initial)
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);                   // start the cycle counter (mcycle)
    setvbuf(stdout, NULL, _IONBF, 0);                            // unbuffered UART -> live progress, no hang
    // synthetic but deterministic inputs/weights/biases (values don't matter for timing, only dimensions do)
    for(int b=0;b<BATCH;b++) for(int i=0;i<N1;i++) x0[b*N1+i]=0.05f*(float)(((b*3+i)%13)-6);
    for(int m=0;m<M1;m++){ for(int i=0;i<N1;i++) W1[m*N1+i]=0.01f*(float)(((m*7+i*3)%17)-8); Bb1[m]=0.01f*(float)((m%5)-2); }
    for(int m=0;m<M2;m++){ for(int i=0;i<N2;i++) W2[m*N2+i]=0.01f*(float)(((m*5+i*2)%15)-7); Bb2[m]=0.01f*(float)((m%5)-2); }
    for(int m=0;m<M3;m++){ for(int i=0;i<N3;i++) W3[m*N3+i]=0.01f*(float)(((m*3+i*5)%13)-6); Bb3[m]=0.01f*(float)((m%5)-2); }
    for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f-0.5f;   // FIR stimulus
    for(int k=0;k<FIR_K;k++) taps[k]=(float)((k*3+1)%11)/11.0f-0.4f;        // FIR taps

    // DMA descriptor: hw_fifo_en=1 routes the stream through the accelerator; channel starts as 1D.
    ts0=(dma_target_t){.ptr=(uint8_t*)loadbuf0,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td0=(dma_target_t){.ptr=(uint8_t*)dummy0,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr0=(dma_trans_t){.src=&ts0,.dst=&td0,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+3,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH0};
    dma_init(NULL);
    dma_validate_transaction(&tr0,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    unsigned int t0,t1;
    PRINTF("=== PERF BENCH PIPE  MLP %d-%d-%d-%d  batch B=%d (pipelined GEMM, single coproc)  (%d MAC/img, %d MAC/batch) ===\n",
           N1,M1,M2,M3, BATCH, TOT_MAC, TOT_MAC*BATCH);

    // [A] CPU golden: the CPU alone runs all BATCH images (the reference for correctness AND for speedup)
    CSR_READ(CSR_REG_MCYCLE,&t0); golden_forward(BATCH); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int cpu_b = t1-t0;
    PRINTF("[A] CPU golden (%d imgs)          : %u cyc\n", BATCH, cpu_b);

    // [B] the coprocessor at B=1 (serial reference): runtime B=1 forces the interlock to serialize issue,
    //     so this reproduces the serial memory-bound cost (~2.05+L cyc/MAC) as an in-run baseline for [C].
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(1,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int b1_tot=t1-t0, b1_gemv=g_gemv; int be1=bitexact(1);
    PRINTF("[B] accel B=1 (serial ref)        : %u cyc  bit-exact=%d\n", b1_tot, be1);
    PRINTF("    setup=%u load=%u gemv=%u  gemv/MAC=%u.%02u  FMA-util~%u%%\n",
           g_setup,g_load,b1_gemv, b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, TOT_MAC*100/(b1_gemv?b1_gemv:1));

    // [C] the SINGLE-coprocessor PIPELINED result at B=8 (compute-bound). This is the headline number:
    //     when swept over L it should stay ~1.14 cyc/MAC (the serial baseline would grow to 1.14+L).
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(BATCH,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int c_tot=t1-t0, c_setup=g_setup, c_load=g_load, c_gemv=g_gemv; int bec=bitexact(BATCH);
    unsigned int cmac = TOT_MAC*BATCH;                                          // total MACs for BATCH images
    PRINTF("[C] accel B=%d (PIPELINED GEMM)   : %u cyc  speedup=%u.%02ux  bit-exact=%d\n",
           BATCH, c_tot, cpu_b/c_tot,(cpu_b*100/c_tot)%100, bec);               // speedup vs CPU [A]
    PRINTF("    setup=%u load=%u gemv=%u  gemv/MAC=%u.%02u  FMA-util~%u%%\n",
           c_setup,c_load,c_gemv, c_gemv/cmac,(c_gemv*100/cmac)%100, cmac*100/(c_gemv?c_gemv:1));
    PRINTF("    -> batch+pipeline: gemv/MAC %u.%02u (B=1 serial) => %u.%02u (B=%d pipelined)\n",
           b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, c_gemv/cmac,(c_gemv*100/cmac)%100, BATCH);

    // [D] the coprocessor (B=8) WHILE the CPU runs its FIR -> 2-way contention on the shared FMA.
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0); fir_advance(1000000); CSR_READ(CSR_REG_MCYCLE,&t1);   // FIR alone -> baseline
    unsigned int fir_alone=t1-t0; for(int i=0;i<FIR_L;i++) firref[i]=firy[i];            // keep FIR reference
    fir_reset(); cpu_time_acc=0;
    fwd(BATCH,1); while(!fir_done) fir_advance(1000000);        // accel + FIR together (contend=1), finish FIR
    unsigned int d_gemv=g_gemv, fir_c=cpu_time_acc; int bed=bitexact(BATCH), firok=1;
    for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) firok=0;     // FIR result must be unchanged by sharing
    int cont = (int)d_gemv-(int)c_gemv;                          // accelerator slow-down from contention
    PRINTF("[D] accel B=%d || CPU-FIR         : gemv_alone=%u gemv_cont=%u  CONTENTION=%d (%d%%)\n",
           BATCH, c_gemv, d_gemv, cont, c_gemv?cont*100/(int)c_gemv:0);
    PRINTF("    acc bit-exact=%d  FIR bit-exact=%d  FIR slow=%d%%   <-- policy-sensitive\n",   // CPU's own cost
           bed, firok, fir_alone?((int)fir_c-(int)fir_alone)*100/(int)fir_alone:0);

    // final one-line summary (grepped by sweep.sh for "ALL bit-exact=1" + the headline cyc/MAC numbers)
    int all = be1&&bec&&bed&&firok;
    PRINTF("=== ALL bit-exact=%d | B=1 %u.%02u cyc/MAC -> B=%d %u.%02u cyc/MAC | FMA-util %u%% ===\n",
           all, b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, BATCH, c_gemv/cmac,(c_gemv*100/cmac)%100,
           cmac*100/(c_gemv?c_gemv:1));
    return all?0:-1;                                            // exit 0 only if every run was bit-exact
}

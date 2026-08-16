// perf_bench — full performance benchmark for the BATCHED shared-FMA coprocessor(s).
//
// WHAT IT DOES: runs a small 3-layer MLP (128->64->32->16) six ways and prints a metric block for each:
//   [A] CPU alone (the golden reference)                          -> baseline cycles + bit-exact reference
//   [B] one coprocessor, batch B=1  (GEMV, memory-bound)          -> cyc/MAC when 1 MAC per fetched weight
//   [C] one coprocessor, batch B=8  (GEMM, compute-bound)         -> batching makes the shared FMA the limiter
//   [D] one coprocessor + the CPU's own FIR filter, together      -> arbiter contention (2 requestors)
//   [E] two coprocessors (acc0+acc1, rows split)                  -> dual throughput + channel balance
//   [F] two coprocessors + the CPU's FIR, together                -> arbiter contention (3 requestors)
// Everything is bit-exact vs the CPU golden. Sweep L (fpu_addmul_lat) x policy (ARB_POLICY_SEL) with sweep.sh.
//
// KEY IDEA (why batching): at B=1 each streamed weight does ONE multiply-accumulate, so the operand bus is
// the limiter (memory-bound). At B=8 each streamed weight is reused for 8 MACs (one per batched image), so
// the shared FMA is the limiter (compute-bound) -> that is when the arbiter policy actually matters.
//
// REQUIRES the batched accelerator (LOAD header [N,M,B,x0..x_{B-1}], x_buf[MAXB][MAXN]) + a testharness with
// two accelerators instantiated (.MAXB(8)) on DMA channels 1 and 2.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0 1                        // DMA channel of accelerator 0 (acc0)
#define CH1 2                        // DMA channel of accelerator 1 (acc1)
#define BATCH 8                      // batch size B for the compute-bound runs

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
uint32_t loadbuf0[3 + BATCH*N1], loadbuf1[3 + BATCH*N1];// LOAD packets: [N,M,B, x...] for acc0 / acc1
float dummy0[4], dummy1[4];                             // dummy DMA destinations for the LOAD phase
dma_target_t ts0, td0, ts1, td1;                        // DMA src/dst descriptors (channel 0 / 1)
dma_trans_t  tr0, tr1;                                  // DMA transactions (channel 0 / 1)
float sig[FIR_L+FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];  // FIR input / taps / output / reference
int fir_n, fir_done;                                   // FIR progress index / done flag
unsigned int cpu_time_acc;                             // accumulated CPU cycles spent inside the FIR
unsigned int g_setup, g_load, g_gemv, g_imbal;         // per-phase cycle accumulators (reset each forward)

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
        // layer 1: ga1[b][m] = relu( bias1[m] + sum_i x0[b][i]*W1[m][i] )
        for(int m=0;m<M1;m++){ float s=0.0f; for(int i=0;i<N1;i++) s=fmaf(x0[b*N1+i], W1[m*N1+i], s);
            ga1[b*M1+m]=s+Bb1[m]; if(ga1[b*M1+m]<0.0f) ga1[b*M1+m]=0.0f; }
        // layer 2: ga2[b][m] = relu( bias2[m] + sum_i ga1[b][i]*W2[m][i] )
        for(int m=0;m<M2;m++){ float s=0.0f; for(int i=0;i<N2;i++) s=fmaf(ga1[b*M1+i], W2[m*N2+i], s);
            ga2[b*M2+m]=s+Bb2[m]; if(ga2[b*M2+m]<0.0f) ga2[b*M2+m]=0.0f; }
        // layer 3 (no ReLU): ga3[b][m] = bias3[m] + sum_i ga2[b][i]*W3[m][i]
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

// ---- TWO-COPROCESSOR batched layer (M-split): acc0 does rows 0..M0-1 (ch0), acc1 does M0..M-1 (ch1). ----
// Both accelerators buffer the SAME B inputs; they only differ in which output rows they compute.
static void layer2(const float* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
    unsigned int t0,t1, tc0=0, tc1=0;
    int M0=(M+1)/2, Mx=M-M0;                                    // acc0 gets M0 rows, acc1 gets the rest
    loadbuf0[0]=N; loadbuf0[1]=M0; loadbuf0[2]=B;               // acc0 header (M0 rows)
    loadbuf1[0]=N; loadbuf1[1]=Mx; loadbuf1[2]=B;               // acc1 header (Mx rows)
    for(int b=0;b<B;b++) for(int i=0;i<N;i++){                  // pack the SAME inputs into both LOAD packets
        uint32_t v = transpose ? f2u(x[i*B+b]) : f2u(x[b*N+i]);
        loadbuf0[3+b*N+i]=v; loadbuf1[3+b*N+i]=v;
    }
    // --- LOAD both channels (inputs into both accelerators) ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)loadbuf0; td0.ptr=(uint8_t*)dummy0; tr0.size_d1_du=(uint32_t)(N*B+3); dma_load_transaction(&tr0);
    ts1.ptr=(uint8_t*)loadbuf1; td1.ptr=(uint8_t*)dummy1; tr1.size_d1_du=(uint32_t)(N*B+3); dma_load_transaction(&tr1);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0); dma_launch(&tr1);
    while(!dma_is_ready(CH0)); while(!dma_is_ready(CH1));       // wait for both LOADs
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=(t1-t0);
    // --- WEIGHT stream both channels: acc0 gets rows [0..M0), acc1 gets rows [M0..M) ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)&W[0];      td0.ptr=(uint8_t*)&Y[0];      tr0.size_d1_du=(uint32_t)(M0*N); dma_load_transaction(&tr0);
    ts1.ptr=(uint8_t*)&W[M0*N];   td1.ptr=(uint8_t*)&Y[M0*B];   tr1.size_d1_du=(uint32_t)(Mx*N); dma_load_transaction(&tr1);
    dma_launch(&tr0); dma_launch(&tr1);                          // both accelerators now share the ONE FMA
    { int d0=0,d1=0;
      while(!(d0&&d1)){                                          // wait for both, recording each finish time
        if(!d0 && dma_is_ready(CH0)){ d0=1; CSR_READ(CSR_REG_MCYCLE,&tc0); }  // acc0 finished at tc0
        if(!d1 && dma_is_ready(CH1)){ d1=1; CSR_READ(CSR_REG_MCYCLE,&tc1); }  // acc1 finished at tc1
        if(contend && !fir_done) fir_advance(FIR_CHUNK);        // optional 3rd requestor: the CPU FIR
      }
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv+=(t1-t0);
    g_imbal += (tc0>tc1)?(tc0-tc1):(tc1-tc0);                    // channel imbalance = |acc0 end - acc1 end|
}

// One full network forward. dual=0 -> single accelerator (layer1); dual=1 -> two accelerators (layer2).
// Between layers the previous output [M][B] is fed as the next input with transpose=1 (element i of image
// b lives at out[i*B+b]). Layer 3 has no ReLU (relu=0).
static void fwd(int B, int dual, int contend){
    g_setup=0; g_load=0; g_gemv=0; g_imbal=0;                    // reset per-phase accumulators
    if(dual){
        layer2(W1, x0,   N1, M1, B, out1, 0, contend); relu_bias(out1, Bb1, M1, B, 1);
        layer2(W2, out1, N2, M2, B, out2, 1, contend); relu_bias(out2, Bb2, M2, B, 1);
        layer2(W3, out2, N3, M3, B, out3, 1, contend); relu_bias(out3, Bb3, M3, B, 0);
    } else {
        layer1(W1, x0,   N1, M1, B, out1, 0, contend); relu_bias(out1, Bb1, M1, B, 1);
        layer1(W2, out1, N2, M2, B, out2, 1, contend); relu_bias(out2, Bb2, M2, B, 1);
        layer1(W3, out2, N3, M3, B, out3, 1, contend); relu_bias(out3, Bb3, M3, B, 0);
    }
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
    PRINTF("=== PERF BENCH  MLP %d-%d-%d-%d  batch B=%d  (%d MAC/img, %d MAC/batch) ===\n",
           N1,M1,M2,M3, BATCH, TOT_MAC, TOT_MAC*BATCH);

    // [A] CPU golden: the CPU alone runs all BATCH images (the reference for correctness AND for speedup)
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

    // [D] one coprocessor (B=8) WHILE the CPU runs its FIR -> 2-way contention on the shared FMA.
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0); fir_advance(1000000); CSR_READ(CSR_REG_MCYCLE,&t1);   // FIR alone -> baseline
    unsigned int fir_alone=t1-t0; for(int i=0;i<FIR_L;i++) firref[i]=firy[i];            // keep FIR reference
    fir_reset(); cpu_time_acc=0;
    fwd(BATCH,0,1); while(!fir_done) fir_advance(1000000);       // accel + FIR together (contend=1), finish FIR
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

    // final one-line summary (grepped by sweep.sh to record bit-exactness + the headline cyc/MAC numbers)
    int all = be1&&bec&&bed&&bee&&bef&&firok&&firok2;
    PRINTF("=== ALL bit-exact=%d | B=1 %u.%02u cyc/MAC -> B=%d %u.%02u cyc/MAC | dual gemv %u.%02ux ===\n",
           all, b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, BATCH, c_gemv/cmac,(c_gemv*100/cmac)%100,
           c_gemv/(e_gemv?e_gemv:1),(c_gemv*100/(e_gemv?e_gemv:1))%100);
    return all?0:-1;                                            // exit 0 only if every run was bit-exact
}

// perf_lenet — the SAME benchmark as perf_bench, but on the REAL LeNet-300-100 / MNIST
// network (784->300->100->10) with real trained weights and real MNIST test images.
// Runs it at batch B=1 (GEMV, memory-bound) and B=BATCH (GEMM, compute-bound) so the shared
// FMA becomes the bottleneck and the arbiter policy actually matters. Reports every metric
// separately: per-phase decomposition (setup/load/gemv/act), cyc/MAC, FMA-issue utilisation,
// single vs CPU speedup, ARBITER CONTENTION (accel || CPU-FIR, and dual 3-way), and the
// dual-accelerator (acc0+acc1 M-split) aggregate + channel imbalance. All bit-exact vs a CPU golden.
//
// ONLY structural difference from perf_bench: LeNet layer-1 has M*N = 300*784 = 235200 > 65535
// (the DMA 16-bit SIZE limit), so the weight matrix is streamed as ONE 2D transfer
// (size_d1 = N inner, size_d2 = M outer, inc = 1,1 = contiguous) instead of a 1D size_d1 = M*N.
// This is the same 2D single-transfer proven in ml_lenet. Everything else mirrors perf_bench.
//
// REQUIRES the batched accelerator (LOAD header [N,M,B,x0..x_{B-1}], x_buf[MAXB][MAXN], MAXN>=784)
// instantiated with .MAXB(8) on ch1 (acc0) and ch2 (acc1). Sweep L (fpu_addmul_lat) x policy
// (ARB_POLICY_SEL) with sweep_lenet.sh.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "lenet_mnist_data.h"          // L1_N..L3_M, fc{1,2,3}_{w,b}, test_images/labels/ref_pred, N_TEST
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0 1
#define CH1 2
#define BATCH 8                        // compute-bound batch size (<= N_TEST)

// LeNet-300-100 dimensions (from the data header)
#define N1 L1_N   // 784
#define M1 L1_M   // 300
#define N2 L2_N   // 300
#define M2 L2_M   // 100
#define N3 L3_N   // 100
#define M3 L3_M   // 10
#define TOT_MAC (M1*N1 + M2*N2 + M3*N3)     // per image = 266200

#define FIR_L 128
#define FIR_K 32
#define FIR_CHUNK 8

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

// weights/biases are const (rodata) and streamed straight from there (as ml_lenet does).
float x0[BATCH*N1];                                     // input batch  [B][N1]
float out1[M1*BATCH], out2[M2*BATCH], out3[M3*BATCH];   // accel outputs [M][B]
float ga1[BATCH*M1], ga2[BATCH*M2], ga3[BATCH*M3];      // CPU golden    [B][M]
uint32_t loadbuf0[3 + BATCH*N1], loadbuf1[3 + BATCH*N1];
float dummy0[4], dummy1[4];
dma_target_t ts0, td0, ts1, td1;
dma_trans_t  tr0, tr1;
float sig[FIR_L+FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];
int fir_n, fir_done;
unsigned int cpu_time_acc;
unsigned int g_setup, g_load, g_gemv, g_imbal;

static void fir_reset(void){ fir_n=0; fir_done=0; }
static void fir_advance(int budget){
    unsigned int t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
    while(budget>0 && !fir_done){
        int n=fir_n; float s=0.0f;
        for(int k=0;k<FIR_K;k++) s=fmaf(taps[k], sig[n+k], s);
        firy[n]=s; fir_n++; budget--;
        if(fir_n==FIR_L) fir_done=1;
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); cpu_time_acc += (t1-t0);
}

static void relu_bias(float* Y, const unsigned int* bias, int M, int B, int relu){
    for(int m=0;m<M;m++) for(int b=0;b<B;b++){
        Y[m*B+b] += w2f(bias[m]);
        if(relu && Y[m*B+b]<0.0f) Y[m*B+b]=0.0f;
    }
}

// CPU golden: B independent LeNet forwards  (ga3[b*M3+m] = logit for image b, class m)
static void golden_forward(int B){
    for(int b=0;b<B;b++){
        for(int m=0;m<M1;m++){ float s=0.0f; for(int i=0;i<N1;i++) s=fmaf(x0[b*N1+i], w2f(fc1_w[m*N1+i]), s);
            ga1[b*M1+m]=s+w2f(fc1_b[m]); if(ga1[b*M1+m]<0.0f) ga1[b*M1+m]=0.0f; }
        for(int m=0;m<M2;m++){ float s=0.0f; for(int i=0;i<N2;i++) s=fmaf(ga1[b*M1+i], w2f(fc2_w[m*N2+i]), s);
            ga2[b*M2+m]=s+w2f(fc2_b[m]); if(ga2[b*M2+m]<0.0f) ga2[b*M2+m]=0.0f; }
        for(int m=0;m<M3;m++){ float s=0.0f; for(int i=0;i<N3;i++) s=fmaf(ga2[b*M2+i], w2f(fc3_w[m*N3+i]), s);
            ga3[b*M3+m]=s+w2f(fc3_b[m]); }
    }
}
static int bitexact(int B){ for(int b=0;b<B;b++) for(int m=0;m<M3;m++) if(out3[m*B+b]!=ga3[b*M3+m]) return 0; return 1; }
static int argmax_col(const float* Y, int b, int B, int M){ int best=0; for(int m=1;m<M;m++) if(Y[m*B+b]>Y[best*B+b]) best=m; return best; }

// ---- single-channel batched layer: LOAD [N,M,B,x] (1D) then stream WHOLE M*N weights (2D). ----
// x layout: transpose=0 -> x[b*N+i] (layer1 input [B][N]); transpose=1 -> x[i*B+b] (prev out [N][B]).
static void fc_single(const unsigned int* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
    unsigned int t0,t1;
    loadbuf0[0]=N; loadbuf0[1]=M; loadbuf0[2]=B;
    for(int b=0;b<B;b++) for(int i=0;i<N;i++)
        loadbuf0[3+b*N+i] = transpose ? f2u(x[i*B+b]) : f2u(x[b*N+i]);
    // --- program + LOAD (1D): the B input vectors -> accel x-buffer ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)loadbuf0; ts0.inc_d1_du=1; ts0.inc_d2_du=0;
    td0.ptr=(uint8_t*)dummy0;   td0.inc_d1_du=1; td0.inc_d2_du=0;
    tr0.dim=DMA_DIM_CONF_1D; tr0.size_d1_du=(uint32_t)(N*B+3); tr0.size_d2_du=0;
    dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0); while(!dma_is_ready(CH0));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=(t1-t0);
    // --- program + GEMV (2D): whole M*N weights in ONE transfer (N inner x M outer, contiguous) ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)W;  ts0.inc_d1_du=1; ts0.inc_d2_du=1;
    td0.ptr=(uint8_t*)Y;  td0.inc_d1_du=1; td0.inc_d2_du=1;   // contiguous dst -> M*B results to Y[0..M*B-1]
    tr0.dim=DMA_DIM_CONF_2D; tr0.size_d1_du=(uint32_t)N; tr0.size_d2_du=(uint32_t)M;
    dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0);
    if(contend){ while(!dma_is_ready(CH0)){ if(!fir_done) fir_advance(FIR_CHUNK); } }
    else       { while(!dma_is_ready(CH0)); }
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv+=(t1-t0);
}

// ---- dual: acc0 rows 0..M0-1 (ch1) || acc1 rows M0..M-1 (ch2), both buffer the B inputs ----
static void fc_dual(const unsigned int* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
    unsigned int t0,t1, tc0=0, tc1=0;
    int M0=(M+1)/2, Mx=M-M0;
    loadbuf0[0]=N; loadbuf0[1]=M0; loadbuf0[2]=B;
    loadbuf1[0]=N; loadbuf1[1]=Mx; loadbuf1[2]=B;
    for(int b=0;b<B;b++) for(int i=0;i<N;i++){
        uint32_t v = transpose ? f2u(x[i*B+b]) : f2u(x[b*N+i]);
        loadbuf0[3+b*N+i]=v; loadbuf1[3+b*N+i]=v;
    }
    // --- program + LOAD (1D) both channels ---
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
    while(!dma_is_ready(CH0)); while(!dma_is_ready(CH1));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=(t1-t0);
    // --- program + GEMV (2D) both channels: acc0 rows [0..M0), acc1 rows [M0..M) ---
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)&W[0];      ts0.inc_d1_du=1; ts0.inc_d2_du=1;
    td0.ptr=(uint8_t*)&Y[0];      td0.inc_d1_du=1; td0.inc_d2_du=1;
    tr0.dim=DMA_DIM_CONF_2D; tr0.size_d1_du=(uint32_t)N; tr0.size_d2_du=(uint32_t)M0;
    dma_load_transaction(&tr0);
    ts1.ptr=(uint8_t*)&W[M0*N];   ts1.inc_d1_du=1; ts1.inc_d2_du=1;
    td1.ptr=(uint8_t*)&Y[M0*B];   td1.inc_d1_du=1; td1.inc_d2_du=1;
    tr1.dim=DMA_DIM_CONF_2D; tr1.size_d1_du=(uint32_t)N; tr1.size_d2_du=(uint32_t)Mx;
    dma_load_transaction(&tr1);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=(t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr0); dma_launch(&tr1);
    { int d0=0,d1=0;
      while(!(d0&&d1)){
        if(!d0 && dma_is_ready(CH0)){ d0=1; CSR_READ(CSR_REG_MCYCLE,&tc0); }
        if(!d1 && dma_is_ready(CH1)){ d1=1; CSR_READ(CSR_REG_MCYCLE,&tc1); }
        if(contend && !fir_done) fir_advance(FIR_CHUNK);
      }
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv+=(t1-t0);
    g_imbal += (tc0>tc1)?(tc0-tc1):(tc1-tc0);
}

static void fwd(int B, int dual, int contend){
    g_setup=0; g_load=0; g_gemv=0; g_imbal=0;
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
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    setvbuf(stdout, NULL, _IONBF, 0);
    // real MNIST images -> input batch [B][N1]
    for(int b=0;b<BATCH;b++) for(int i=0;i<N1;i++) x0[b*N1+i]=w2f(test_images[b*N1+i]);
    // FIR stimulus (CPU-side DSP job for the contention tests)
    for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f-0.5f;
    for(int k=0;k<FIR_K;k++) taps[k]=(float)((k*3+1)%11)/11.0f-0.4f;

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

    // [A] CPU golden (B forward passes)
    CSR_READ(CSR_REG_MCYCLE,&t0); golden_forward(BATCH); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int cpu_b = t1-t0;
    PRINTF("[A] CPU golden (%d imgs)          : %u cyc\n", BATCH, cpu_b);

    // [B] accel B=1  (GEMV, memory-bound baseline)
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(1,0,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int b1_tot=t1-t0, b1_gemv=g_gemv; int be1=bitexact(1);
    PRINTF("[B] accel B=1 (GEMV,mem-bound)    : %u cyc  bit-exact=%d\n", b1_tot, be1);
    PRINTF("    setup=%u load=%u gemv=%u  gemv/MAC=%u.%02u  FMA-util~%u%%\n",
           g_setup,g_load,b1_gemv, b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, TOT_MAC*100/(b1_gemv?b1_gemv:1));

    // [C] accel B=BATCH  (GEMM, compute-bound)
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(BATCH,0,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int c_tot=t1-t0, c_setup=g_setup, c_load=g_load, c_gemv=g_gemv; int bec=bitexact(BATCH);
    unsigned int cmac = TOT_MAC*BATCH;
    PRINTF("[C] accel B=%d (GEMM,compute-bnd) : %u cyc  speedup=%u.%02ux  bit-exact=%d\n",
           BATCH, c_tot, cpu_b/c_tot,(cpu_b*100/c_tot)%100, bec);
    PRINTF("    setup=%u load=%u gemv=%u  gemv/MAC=%u.%02u  FMA-util~%u%%\n",
           c_setup,c_load,c_gemv, c_gemv/cmac,(c_gemv*100/cmac)%100, cmac*100/(c_gemv?c_gemv:1));
    PRINTF("    -> memory->compute: gemv/MAC %u.%02u (B=1) => %u.%02u (B=%d)\n",
           b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, c_gemv/cmac,(c_gemv*100/cmac)%100, BATCH);

    // [D] accel B=BATCH || CPU-FIR  (ARBITER CONTENTION, compute-bound -> policy-sensitive)
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0); fir_advance(1000000); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int fir_alone=t1-t0; for(int i=0;i<FIR_L;i++) firref[i]=firy[i];
    fir_reset(); cpu_time_acc=0;
    fwd(BATCH,0,1); while(!fir_done) fir_advance(1000000);
    unsigned int d_gemv=g_gemv, fir_c=cpu_time_acc; int bed=bitexact(BATCH), firok=1;
    for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) firok=0;
    int cont = (int)d_gemv-(int)c_gemv;
    PRINTF("[D] accel B=%d || CPU-FIR         : gemv_alone=%u gemv_cont=%u  CONTENTION=%d (%d%%)\n",
           BATCH, c_gemv, d_gemv, cont, c_gemv?cont*100/(int)c_gemv:0);
    PRINTF("    acc bit-exact=%d  FIR bit-exact=%d  FIR slow=%d%%   <-- policy-sensitive\n",
           bed, firok, fir_alone?((int)fir_c-(int)fir_alone)*100/(int)fir_alone:0);

    // [E] dual (acc0+acc1) B=BATCH
    CSR_READ(CSR_REG_MCYCLE,&t0); fwd(BATCH,1,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int e_tot=t1-t0, e_gemv=g_gemv, e_imb=g_imbal; int bee=bitexact(BATCH);
    PRINTF("[E] DUAL acc0+acc1 B=%d           : %u cyc  gemv=%u  bit-exact=%d  imbalance=%u\n",
           BATCH, e_tot, e_gemv, bee, e_imb);
    PRINTF("    gemv %u.%02ux vs single  |  gemv/MAC %u.%02u (single %u.%02u)\n",
           c_gemv/(e_gemv?e_gemv:1),(c_gemv*100/(e_gemv?e_gemv:1))%100,
           e_gemv/cmac,(e_gemv*100/cmac)%100, c_gemv/cmac,(c_gemv*100/cmac)%100);

    // [F] dual B=BATCH || CPU-FIR  (3-way FMA contention)
    fir_reset(); cpu_time_acc=0;
    fwd(BATCH,1,1); while(!fir_done) fir_advance(1000000);
    unsigned int f_gemv=g_gemv, fir_t=cpu_time_acc; int bef=bitexact(BATCH), firok2=1;
    for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) firok2=0;
    int cont2=(int)f_gemv-(int)e_gemv;
    PRINTF("[F] DUAL B=%d || CPU-FIR (3-way)  : gemv_alone=%u gemv_cont=%u  CONTENTION=%d (%d%%)\n",
           BATCH, e_gemv, f_gemv, cont2, e_gemv?cont2*100/(int)e_gemv:0);
    PRINTF("    acc bit-exact=%d  FIR bit-exact=%d  FIR slow=%d%%\n",
           bef, firok2, fir_alone?((int)fir_t-(int)fir_alone)*100/(int)fir_alone:0);

    // accuracy sanity: predictions of the batched accel run vs the reference labels/preds
    int correct=0, match_ref=0;
    for(int b=0;b<BATCH;b++){ int p=argmax_col(out3,b,BATCH,M3);
        if(p==test_labels[b]) correct++; if(p==ref_pred[b]) match_ref++; }
    PRINTF("[acc] batched preds: correct=%d/%d  match_ref=%d/%d\n", correct,BATCH, match_ref,BATCH);

    int all = be1&&bec&&bed&&bee&&bef&&firok&&firok2;
    PRINTF("=== ALL bit-exact=%d | B=1 %u.%02u cyc/MAC -> B=%d %u.%02u cyc/MAC | dual gemv %u.%02ux ===\n",
           all, b1_gemv/TOT_MAC,(b1_gemv*100/TOT_MAC)%100, BATCH, c_gemv/cmac,(c_gemv*100/cmac)%100,
           c_gemv/(e_gemv?e_gemv:1),(c_gemv*100/(e_gemv?e_gemv:1))%100);
    return all?0:-1;
}

// perf_bench — compact end-to-end performance benchmark for the shared-FMA coprocessor(s).
// Same workload TYPES as ml_lenet / ml_coexec (multi-layer GEMV + a CPU FIR job) but on a TINY
// 3-layer MLP (128->64->32->16) so it finishes in ~1 min, and reports EVERY metric:
//   [A] CPU fused baseline (golden reference)
//   [B] single accelerator (acc0): per-phase decomposition setup/load/gemv/act + speedup   (= ml_lenet)
//   [C] acc0 || CPU-FIR : GEMV memory+drain contention + CPU-priority (FIR slow-down)       (= ml_coexec)
//   [D] DUAL accelerators (acc0+acc1, M-split): aggregate GEMV speedup + channel imbalance  (NEW)
//   [E] DUAL || CPU-FIR (3-way): contention with two accelerators + CPU-priority            (NEW)
// Everything is checked BIT-EXACT vs the CPU golden. Weights are synthetic/deterministic (perf is
// weight-value-independent). The arbitration policy is fixed at RTL build time (ARB_POLICY_SEL):
// rebuild with +define+ARB_POLICY_SEL=0/1/2 and re-run to compare CPU-strict / all-RR / QoS.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0 1                 // acc0 channel
#define CH1 2                 // acc1 channel

// tiny 3-layer MLP: 128 -> 64 -> 32 -> 16
#define N1 128
#define M1 64
#define N2 64
#define M2 32
#define N3 32
#define M3 16
#define MAXN 128
#define TOT_MAC (M1*N1 + M2*N2 + M3*N3)

// CPU FIR contending job
#define FIR_L 128
#define FIR_K 32
#define FIR_CHUNK 8

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

float W1[M1*N1], W2[M2*N2], W3[M3*N3];
float B1[M1], B2[M2], B3[M3];
float x0[N1];
float a1[M1], a2[M2], a3[M3];      // accel (single)
float d1[M1], d2[M2], d3[M3];      // accel (dual)
float g1[M1], g2[M2], g3[M3];      // cpu golden
uint32_t loadbuf0[MAXN+2], loadbuf1[MAXN+2];
float dummy0[4], dummy1[4];
dma_target_t ts0, td0, ts1, td1;
dma_trans_t  tr0, tr1;
float sig[FIR_L+FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];
int fir_n, fir_done;
unsigned int cpu_time_acc;
unsigned int g_setup, g_load, g_gemv, g_imbal;   // per-forward accumulators

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
static void gemv_cpu(const float* W, const float* x, int N, int M, float* out){
    for(int m=0;m<M;m++){ float s=0.0f; for(int n=0;n<N;n++) s=fmaf(x[n], W[m*N+n], s); out[m]=s; }
}
static void relu(float* v, const float* b, int M){ for(int m=0;m<M;m++){ v[m]+=b[m]; if(v[m]<0) v[m]=0.0f; } }
static int  bex(const float* p, const float* q, int M){ for(int m=0;m<M;m++) if(p[m]!=q[m]) return 0; return 1; }

// single-channel accel layer (acc0) with setup/load/gemv brackets; contend -> FIR runs during GEMV poll
static void accel_one(const float* W, const float* x, int N, int M, float* out, int contend){
    unsigned int t0,t1;
    loadbuf0[0]=N; loadbuf0[1]=M; for(int i=0;i<N;i++) loadbuf0[2+i]=f2u(x[i]);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)loadbuf0; td0.ptr=(uint8_t*)dummy0; tr0.size_d1_du=N+2; dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    dma_launch(&tr0); while(!dma_is_ready(CH0));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)W; td0.ptr=(uint8_t*)out; tr0.size_d1_du=M*N; dma_load_transaction(&tr0);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    dma_launch(&tr0);
    if(contend){ while(!dma_is_ready(CH0)){ if(!fir_done) fir_advance(FIR_CHUNK); } }
    else       { while(!dma_is_ready(CH0)); }
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv += (t1-t0);
}

// dual-channel accel layer: acc0 rows 0..M0-1 (ch1) || acc1 rows M0..M-1 (ch2), concurrent
static void accel_dual(const float* W, const float* x, int N, int M, float* out, int contend){
    unsigned int t0,t1, tc0=0, tc1=0;
    int M0=(M+1)/2, Mx=M-M0;
    loadbuf0[0]=N; loadbuf0[1]=M0; for(int i=0;i<N;i++) loadbuf0[2+i]=f2u(x[i]);
    loadbuf1[0]=N; loadbuf1[1]=Mx; for(int i=0;i<N;i++) loadbuf1[2+i]=f2u(x[i]);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)loadbuf0; td0.ptr=(uint8_t*)dummy0; tr0.size_d1_du=N+2; dma_load_transaction(&tr0);
    ts1.ptr=(uint8_t*)loadbuf1; td1.ptr=(uint8_t*)dummy1; tr1.size_d1_du=N+2; dma_load_transaction(&tr1);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup += (t1-t0);
    CSR_READ(CSR_REG_MCYCLE,&t0);
    dma_launch(&tr0); dma_launch(&tr1);
    while(!dma_is_ready(CH0)); while(!dma_is_ready(CH1));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load += (t1-t0);
    // GEMV both halves concurrently
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts0.ptr=(uint8_t*)&W[0];     td0.ptr=(uint8_t*)&out[0];  tr0.size_d1_du=M0*N; dma_load_transaction(&tr0);
    ts1.ptr=(uint8_t*)&W[M0*N];  td1.ptr=(uint8_t*)&out[M0]; tr1.size_d1_du=Mx*N; dma_load_transaction(&tr1);
    dma_launch(&tr0); dma_launch(&tr1);
    { int done0=0, done1=0;
      while(!(done0 && done1)){
        if(!done0 && dma_is_ready(CH0)){ done0=1; CSR_READ(CSR_REG_MCYCLE,&tc0); }
        if(!done1 && dma_is_ready(CH1)){ done1=1; CSR_READ(CSR_REG_MCYCLE,&tc1); }
        if(contend && !fir_done) fir_advance(FIR_CHUNK);
      }
    }
    CSR_READ(CSR_REG_MCYCLE,&t1); g_gemv += (t1-t0);
    g_imbal += (tc0>tc1) ? (tc0-tc1) : (tc1-tc0);
}

static void accel_forward(int dual, int contend){
    g_setup=0; g_load=0; g_gemv=0; g_imbal=0;
    if(dual){
        accel_dual(W1, x0, N1, M1, d1, contend); relu(d1, B1, M1);
        accel_dual(W2, d1, N2, M2, d2, contend); relu(d2, B2, M2);
        accel_dual(W3, d2, N3, M3, d3, contend);
    } else {
        accel_one(W1, x0, N1, M1, a1, contend); relu(a1, B1, M1);
        accel_one(W2, a1, N2, M2, a2, contend); relu(a2, B2, M2);
        accel_one(W3, a2, N3, M3, a3, contend);
    }
}

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
    setvbuf(stdout, NULL, _IONBF, 0);
    for(int i=0;i<N1;i++) x0[i]=0.05f*(float)((i%13)-6);
    for(int m=0;m<M1;m++){ for(int i=0;i<N1;i++) W1[m*N1+i]=0.01f*(float)(((m*7+i*3)%17)-8); B1[m]=0.01f*(float)((m%5)-2); }
    for(int m=0;m<M2;m++){ for(int i=0;i<N2;i++) W2[m*N2+i]=0.01f*(float)(((m*5+i*2)%15)-7); B2[m]=0.01f*(float)((m%5)-2); }
    for(int m=0;m<M3;m++){ for(int i=0;i<N3;i++) W3[m*N3+i]=0.01f*(float)(((m*3+i*5)%13)-6); B3[m]=0.01f*(float)((m%5)-2); }
    for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f-0.5f;
    for(int k=0;k<FIR_K;k++) taps[k]=(float)((k*3+1)%11)/11.0f-0.4f;

    ts0=(dma_target_t){.ptr=(uint8_t*)loadbuf0,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td0=(dma_target_t){.ptr=(uint8_t*)dummy0,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr0=(dma_trans_t){.src=&ts0,.dst=&td0,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+2,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH0};
    ts1=(dma_target_t){.ptr=(uint8_t*)loadbuf1,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td1=(dma_target_t){.ptr=(uint8_t*)dummy1,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr1=(dma_trans_t){.src=&ts1,.dst=&td1,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+2,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH1};
    dma_init(NULL);
    dma_validate_transaction(&tr0,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);
    dma_validate_transaction(&tr1,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    unsigned int t0,t1;
    PRINTF("=== PERF BENCH  MLP %d-%d-%d-%d  (%d MACs, policy fixed at build) ===\n", N1,M1,M2,M3, TOT_MAC);

    // [A] CPU fused baseline -> golden
    CSR_READ(CSR_REG_MCYCLE,&t0);
    gemv_cpu(W1,x0,N1,M1,g1); relu(g1,B1,M1);
    gemv_cpu(W2,g1,N2,M2,g2); relu(g2,B2,M2);
    gemv_cpu(W3,g2,N3,M3,g3);
    CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int cpu_cyc=t1-t0;
    PRINTF("[A] CPU fused baseline           : %u cyc\n", cpu_cyc);

    // [B] single accel (acc0) + decomposition
    CSR_READ(CSR_REG_MCYCLE,&t0); accel_forward(0,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int s_tot=t1-t0, s_setup=g_setup, s_load=g_load, s_gemv=g_gemv;
    int be_b=bex(a3,g3,M3);
    PRINTF("[B] single acc0 forward          : %u cyc  speedup=%u.%02ux  bit-exact=%d\n",
           s_tot, cpu_cyc/s_tot,(cpu_cyc*100/s_tot)%100, be_b);
    PRINTF("    decomp setup=%u load=%u gemv=%u act=%d   gemv/MAC=%u.%02u\n",
           s_setup,s_load,s_gemv,(int)s_tot-(int)(s_setup+s_load+s_gemv), s_gemv/TOT_MAC,(s_gemv*100/TOT_MAC)%100);

    // [C] acc0 || CPU-FIR contention
    fir_reset(); cpu_time_acc=0;
    CSR_READ(CSR_REG_MCYCLE,&t0); fir_advance(1000000); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int fir_alone=t1-t0; for(int i=0;i<FIR_L;i++) firref[i]=firy[i];
    fir_reset(); cpu_time_acc=0;
    accel_forward(0,1); while(!fir_done) fir_advance(1000000);
    unsigned int s_gemv_c=g_gemv, fir_c=cpu_time_acc;
    int be_c=bex(a3,g3,M3), fir_ok=bex(firy,firref,FIR_L);
    int cc=(int)s_gemv_c-(int)s_gemv;
    PRINTF("[C] acc0 || CPU-FIR              : gemv_alone=%u gemv_cont=%u  MEM+DRAIN=%d (%d%%)\n",
           s_gemv,s_gemv_c, cc, s_gemv?cc*100/(int)s_gemv:0);
    PRINTF("    acc bit-exact=%d  FIR bit-exact=%d  FIR slow=%d%% (bus, CPU-priority)\n",
           be_c, fir_ok, fir_alone?((int)fir_c-(int)fir_alone)*100/(int)fir_alone:0);

    // [D] dual accel (acc0+acc1, M-split)
    CSR_READ(CSR_REG_MCYCLE,&t0); accel_forward(1,0); CSR_READ(CSR_REG_MCYCLE,&t1);
    unsigned int du_tot=t1-t0, du_gemv=g_gemv, du_imb=g_imbal; int be_d=bex(d3,g3,M3);
    PRINTF("[D] DUAL acc0+acc1 (M-split)     : %u cyc  gemv=%u  bit-exact=%d  imbalance=%u cyc\n",
           du_tot, du_gemv, be_d, du_imb);
    PRINTF("    gemv %u.%02ux vs single  |  gemv/MAC %u.%02u (single %u.%02u)\n",
           s_gemv/(du_gemv?du_gemv:1),(s_gemv*100/(du_gemv?du_gemv:1))%100,
           du_gemv/TOT_MAC,(du_gemv*100/TOT_MAC)%100, s_gemv/TOT_MAC,(s_gemv*100/TOT_MAC)%100);

    // [E] dual || CPU-FIR (3-way)
    fir_reset(); cpu_time_acc=0;
    accel_forward(1,1); while(!fir_done) fir_advance(1000000);
    unsigned int du_gemv_c=g_gemv, fir_t=cpu_time_acc; int be_e=bex(d3,g3,M3), fir_ok2=bex(firy,firref,FIR_L);
    int cd=(int)du_gemv_c-(int)du_gemv;
    PRINTF("[E] DUAL || CPU-FIR (3-way)      : gemv_alone=%u gemv_cont=%u  contention=%d (%d%%)\n",
           du_gemv,du_gemv_c, cd, du_gemv?cd*100/(int)du_gemv:0);
    PRINTF("    acc bit-exact=%d  FIR bit-exact=%d  FIR slow=%d%%\n",
           be_e, fir_ok2, fir_alone?((int)fir_t-(int)fir_alone)*100/(int)fir_alone:0);

    int all_ok = be_b && be_c && be_d && be_e && fir_ok && fir_ok2;
    PRINTF("=== ALL bit-exact=%d | single %u.%02ux vs CPU | dual gemv %u.%02ux vs single ===\n",
           all_ok, cpu_cyc/s_tot,(cpu_cyc*100/s_tot)%100, s_gemv/(du_gemv?du_gemv:1),(s_gemv*100/(du_gemv?du_gemv:1))%100);
    return all_ok?0:-1;
}

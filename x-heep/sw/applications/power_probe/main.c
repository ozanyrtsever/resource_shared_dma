// power_probe -- perf_bench_pipe'in GUC ANALIZI icin surumu (gate-level sim / SAIF).
//
// perf_bench_pipe ile AYNI: toy MLP 128-64-32-16 B=8, ayni fc_cpu golden (optimize), ayni coproc
// protokolu (LOAD [N,M,B,x] -> WEIGHT), ayni MAC kernel'i (register-only, 8 bagimsiz akumulator),
// ayni contend interleave. Sadece FIR cikarildi ve her faz `PHASE <ad> start/end mcycle=` ile
// isaretlendi -> her fazin SAIF penceresi (VCS $toggle_start/stop) bu cycle araliklarina eslenir.
//
// FAZLAR (her biri ayri guc raporu olur):
//   IDLE     : bos dongu (referans: clock + leakage)
//   CPU_INF  : [1] CPU alone inference (golden)                 -> CPU + FMA
//   COPROC   : [3] Coproc alone inference (CPU polling)         -> coproc + FMA + arbiter
//   CPU_MAC  : [6] CPU alone MAC (saf-FMA kernel)               -> CPU'nun kendi FP isi
//   SHARED   : [7]/[8] coproc inference || CPU MAC (TEK FMA)    -> ASIL: iki is tek FMA'da
// Enerji = faz gucu (SAIF->DC report_power -hierarchy) x faz suresi (mcycle x 3.845 ns).
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0   1
#define BATCH 8

#define N1 128
#define M1 64
#define N2 64
#define M2 32
#define N3 32
#define M3 16
#define TOT_MAC (M1*N1 + M2*N2 + M3*N3)   // 10752 MAC/img
#define CMAC    (TOT_MAC*BATCH)           // 86016 MAC/batch

#define IDLE_SPIN 20000               // IDLE fazi uzunlugu (iterasyon)

// ---- CPU MAC-stress: perf_bench_pipe ile birebir ayni ----
#define MAC_ACC    8
#define MAC_ROUNDS 32
#define MAC_L      128
#define MAC_CHUNK  1
#define MAC_FMA    (MAC_L*MAC_ROUNDS*MAC_ACC)
#define MAC_CA 0.5f
#define MAC_CB 0.5f

static inline float    w2f(unsigned u){ union{unsigned u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)   { union{float f; uint32_t u;} c; c.f=f; return c.u; }

float W1[M1*N1], W2[M2*N2], W3[M3*N3], Bb1[M1], Bb2[M2], Bb3[M3];
float x0[BATCH*N1];
float out1[M1*BATCH], out2[M2*BATCH], out3[M3*BATCH];
float ga1[BATCH*M1], ga2[BATCH*M2], ga3[BATCH*M3];
uint32_t loadbuf[3+BATCH*N1];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;
unsigned g_setup, g_load, g_comp;
volatile int idle_sink;

// FAZ ISARETI: TB (dc_scripts/gls/saif_ctrl.sv) core'un data-bus portlarinda bu adrese yapilan store'u
// gorur: deger!=0 -> faz basladi ($toggle_start), 0 -> bitti ($toggle_stop + SAIF yaz). Adres: nm main.elf | grep phase_flag
volatile uint32_t phase_flag;
#define PH_IDLE 1
#define PH_CPU_INF 2
#define PH_COPROC 3
#define PH_CPU_MAC 4
#define PH_SHARED 5
#define PHASE_START(id) (phase_flag = (id))
#define PHASE_END()     (phase_flag = 0)

float mseed[MAC_ACC]; volatile float msink; int mac_i, mac_done; unsigned g_mac;
static void mac_seed(void){ for(int i=0;i<MAC_ACC;i++) mseed[i]=0.3f+0.1f*(float)i; }
static void mac_reset(void){ mac_i=0; mac_done=0; }
static inline void mac_one(void){
  float a0=mseed[0],a1=mseed[1],a2=mseed[2],a3=mseed[3],a4=mseed[4],a5=mseed[5],a6=mseed[6],a7=mseed[7];
  for(int r=0;r<MAC_ROUNDS;r++){
    a0=fmaf(MAC_CA,a0,MAC_CB); a1=fmaf(MAC_CA,a1,MAC_CB); a2=fmaf(MAC_CA,a2,MAC_CB); a3=fmaf(MAC_CA,a3,MAC_CB);
    a4=fmaf(MAC_CA,a4,MAC_CB); a5=fmaf(MAC_CA,a5,MAC_CB); a6=fmaf(MAC_CA,a6,MAC_CB); a7=fmaf(MAC_CA,a7,MAC_CB);
  }
  mseed[0]=a0;mseed[1]=a1;mseed[2]=a2;mseed[3]=a3;mseed[4]=a4;mseed[5]=a5;mseed[6]=a6;mseed[7]=a7;
  if(++mac_i==MAC_L) mac_done=1;
}
static void mac_step(int budget){
  unsigned t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
  while(budget-->0 && !mac_done) mac_one();
  CSR_READ(CSR_REG_MCYCLE,&t1); g_mac+=t1-t0;
}
static void mac_finish(void){
  unsigned t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
  while(!mac_done) mac_one();
  CSR_READ(CSR_REG_MCYCLE,&t1); g_mac+=t1-t0;
  msink=mseed[0]+mseed[1]+mseed[2]+mseed[3]+mseed[4]+mseed[5]+mseed[6]+mseed[7];
}

// ---- CPU golden: perf_bench_pipe ile birebir ayni (optimize, bit-exact) ----
static void relu_bias(float* Y, const float* b, int M, int B, int relu){
  for(int m=0;m<M;m++) for(int j=0;j<B;j++){ Y[m*B+j]+=b[m]; if(relu&&Y[m*B+j]<0) Y[m*B+j]=0; }
}
static void fc_cpu(const float* W, const float* x, const float* b, int N, int M, float* Y, int relu){
  for(int m=0;m<M;m++){
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    const float* wm=&W[m*N];
    for(int i=0;i<N;i++){ float w=wm[i];
      a0=fmaf(w,x[0*N+i],a0); a1=fmaf(w,x[1*N+i],a1); a2=fmaf(w,x[2*N+i],a2); a3=fmaf(w,x[3*N+i],a3);
      a4=fmaf(w,x[4*N+i],a4); a5=fmaf(w,x[5*N+i],a5); a6=fmaf(w,x[6*N+i],a6); a7=fmaf(w,x[7*N+i],a7); }
    float bb=b[m];
    #define ST(J,A){ float v=A+bb; Y[(J)*M+m]=(relu&&v<0)?0.0f:v; }
    ST(0,a0)ST(1,a1)ST(2,a2)ST(3,a3)ST(4,a4)ST(5,a5)ST(6,a6)ST(7,a7)
    #undef ST
  }
}
static void golden(int B){
  (void)B;
  fc_cpu(W1, x0,  Bb1, N1,M1, ga1, 1);
  fc_cpu(W2, ga1, Bb2, N2,M2, ga2, 1);
  fc_cpu(W3, ga2, Bb3, N3,M3, ga3, 0);
}
static int bitexact(int B){ for(int j=0;j<B;j++) for(int m=0;m<M3;m++) if(out3[m*B+j]!=ga3[j*M3+m]) return 0; return 1; }

// ---- coproc katmani: perf_bench_pipe ile ayni (contend: 0=yok, 2=CPU MAC interleave) ----
static void fc(const float* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
  unsigned t0,t1;
  loadbuf[0]=N; loadbuf[1]=M; loadbuf[2]=B;
  for(int j=0;j<B;j++) for(int i=0;i<N;i++) loadbuf[3+j*N+i] = transpose ? f2u(x[i*B+j]) : f2u(x[j*N+i]);
  CSR_READ(CSR_REG_MCYCLE,&t0);
  ts.ptr=(uint8_t*)loadbuf; td.ptr=(uint8_t*)dummy; tr.size_d1_du=(uint32_t)(N*B+3); dma_load_transaction(&tr);
  CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr); while(!dma_is_ready(CH0));
  CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0);
  ts.ptr=(uint8_t*)W; td.ptr=(uint8_t*)Y; tr.size_d1_du=(uint32_t)(M*N); dma_load_transaction(&tr);
  CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr);
  if(contend==2){ while(!dma_is_ready(CH0)) if(!mac_done) mac_step(MAC_CHUNK); }
  else          { while(!dma_is_ready(CH0)); }
  CSR_READ(CSR_REG_MCYCLE,&t1); g_comp+=t1-t0;
}
static void infer(int B, int contend){
  g_setup=g_load=g_comp=0;
  fc(W1,x0,   N1,M1,B,out1,0,contend); relu_bias(out1,Bb1,M1,B,1);
  fc(W2,out1, N2,M2,B,out2,1,contend); relu_bias(out2,Bb2,M2,B,1);
  fc(W3,out2, N3,M3,B,out3,1,contend); relu_bias(out3,Bb3,M3,B,0);
}

int main(void){
  CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
  CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
  setvbuf(stdout,NULL,_IONBF,0);
  for(int j=0;j<BATCH;j++) for(int i=0;i<N1;i++) x0[j*N1+i]=0.05f*(float)(((j*3+i)%13)-6);
  for(int m=0;m<M1;m++){ for(int i=0;i<N1;i++) W1[m*N1+i]=0.01f*(float)(((m*7+i*3)%17)-8); Bb1[m]=0.01f*(float)((m%5)-2); }
  for(int m=0;m<M2;m++){ for(int i=0;i<N2;i++) W2[m*N2+i]=0.01f*(float)(((m*5+i*2)%15)-7); Bb2[m]=0.01f*(float)((m%5)-2); }
  for(int m=0;m<M3;m++){ for(int i=0;i<N3;i++) W3[m*N3+i]=0.01f*(float)(((m*3+i*5)%13)-6); Bb3[m]=0.01f*(float)((m%5)-2); }
  ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  td=(dma_target_t){.ptr=(uint8_t*)dummy,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+3,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH0};
  dma_init(NULL);
  dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

  unsigned t0,t1;
  PRINTF("=== POWER_PROBE  MLP %d-%d-%d-%d  B=%d  (%d MAC/img, %d MAC/batch) ===\n", N1,M1,M2,M3,BATCH,TOT_MAC,CMAC);

  // Her faz: print -> PHASE_START -> t0 ... t1 -> PHASE_END -> print  (UART yazimi pencere DISINDA)
  // ---- IDLE ----
  PRINTF("PHASE IDLE    start\n");
  PHASE_START(PH_IDLE); CSR_READ(CSR_REG_MCYCLE,&t0);
  for(int k=0;k<IDLE_SPIN;k++) idle_sink=k;
  CSR_READ(CSR_REG_MCYCLE,&t1); PHASE_END();
  PRINTF("PHASE IDLE    end   mcycle=%u..%u  dur=%u\n", t0, t1, t1-t0);

  // ---- CPU_INF : [1] ----
  PRINTF("PHASE CPU_INF start\n");
  PHASE_START(PH_CPU_INF); CSR_READ(CSR_REG_MCYCLE,&t0);
  golden(BATCH);
  CSR_READ(CSR_REG_MCYCLE,&t1); PHASE_END();
  PRINTF("PHASE CPU_INF end   mcycle=%u..%u  dur=%u\n", t0, t1, t1-t0);
  unsigned cpu_inf=t1-t0;

  // ---- COPROC : [3] ----
  PRINTF("PHASE COPROC  start\n");
  PHASE_START(PH_COPROC); CSR_READ(CSR_REG_MCYCLE,&t0);
  infer(BATCH,0);
  CSR_READ(CSR_REG_MCYCLE,&t1); PHASE_END();
  PRINTF("PHASE COPROC  end   mcycle=%u..%u  dur=%u\n", t0, t1, t1-t0);
  unsigned co_tot=t1-t0, co_su=g_setup, co_ld=g_load, co_cp=g_comp; int be_co=bitexact(BATCH);

  // ---- CPU_MAC : [6] ----
  mac_seed(); mac_reset(); g_mac=0;
  PRINTF("PHASE CPU_MAC start\n");
  PHASE_START(PH_CPU_MAC); CSR_READ(CSR_REG_MCYCLE,&t0);
  mac_finish();
  CSR_READ(CSR_REG_MCYCLE,&t1); PHASE_END();
  PRINTF("PHASE CPU_MAC end   mcycle=%u..%u  dur=%u\n", t0, t1, t1-t0);
  unsigned mac_alone=g_mac;

  // ---- SHARED : [7]+[8] (coproc inference || CPU MAC, tek FMA) ----
  mac_seed(); mac_reset(); g_mac=0;
  PRINTF("PHASE SHARED  start\n");
  PHASE_START(PH_SHARED); CSR_READ(CSR_REG_MCYCLE,&t0);
  infer(BATCH,2);
  mac_finish();
  CSR_READ(CSR_REG_MCYCLE,&t1); PHASE_END();
  PRINTF("PHASE SHARED  end   mcycle=%u..%u  dur=%u\n", t0, t1, t1-t0);
  unsigned shm_tot=t1-t0, sh_mac=g_mac; int be_shm=bitexact(BATCH);

  // ---- perf_bench_pipe ile ayni formatta metrikler ----
  PRINTF("[1] CPU    alone inference : setup=0 load=0 compute=%u total=%u\n", cpu_inf, cpu_inf);
  PRINTF("[3] Coproc alone inference : setup=%u load=%u compute=%u total=%u  speedup=%u.%02ux cyc/MAC=%u.%02u bit-exact=%d\n",
         co_su, co_ld, co_cp, co_tot, cpu_inf/co_tot,(cpu_inf*100/co_tot)%100, co_cp/CMAC,(co_cp*100/CMAC)%100, be_co);
  PRINTF("[6] CPU    alone MAC       : setup=0 load=0 compute=%u total=%u  cyc/fmadd=%u.%02u\n",
         mac_alone, mac_alone, mac_alone/MAC_FMA,(mac_alone*100/MAC_FMA)%100);
  PRINTF("[7] Shared CPU MAC         : setup=0 load=0 compute=%u total=%u  (alone %u)\n", sh_mac, sh_mac, mac_alone);
  PRINTF("[8] Shared Coproc inf (MAC): total=%u  (alone %u) bit-exact=%d\n", shm_tot, co_tot, be_shm);
  int all = be_co && be_shm;
  PRINTF("=== ALL bit-exact=%d | coproc cyc/MAC=%u.%02u | speedup=%u.%02ux ===\n",
         all, co_cp/CMAC,(co_cp*100/CMAC)%100, cpu_inf/co_tot,(cpu_inf*100/co_tot)%100);
  return all?0:-1;
}

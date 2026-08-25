// perf_bench_pipe -- pipelined single-coprocessor benchmark (toy MLP 128-64-32-16), B=8.
//
// Coprocessor = dma_fp_dot_accel_pipe: CPU'nun TEK FMA'sini paylasir (ikinci FPU YOK). Inference = MLP.
// FIR = CPU'nun KENDI 32-tap FP filtresi (inference DEGIL); sadece "paylasim" senaryosunda CPU'nun
// coproc ile ayni anda FMA kullanmasini saglamak icin var.
//
// 5 METRIK (her biri: setup / load [DMA yoksa 0] / compute / total):
//   [1] CPU    alone inference  : CPU tum agi kendisi kosar          (DMA yok)
//   [2] CPU    alone FIR        : CPU FIR'i tek basina kosar         (DMA yok)
//   [3] Coproc alone inference  : coproc kosar, CPU bosta            (setup=DMA program, load=giris akisi, compute=GEMM)
//   [4] Shared CPU FIR          : [5] ile AYNI kosu; CPU'nun FIR cycle'i (coproc inference ederken)
//   [5] Shared Coproc inference : coproc inference'i, CPU ayni anda FIR koserken (tek FMA paylasilir)
// Tum inference sonuclari CPU golden ile bit-exact. Sonda 5 total'lik ozet RAPOR basilir.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0   1                       // coprocessor'un DMA kanali
#define BATCH 8                       // batch (GEMM)

#define N1 128
#define M1 64
#define N2 64
#define M2 32
#define N3 32
#define M3 16
#define TOT_MAC (M1*N1 + M2*N2 + M3*N3)   // MAC / img  = 10752
#define CMAC    (TOT_MAC*BATCH)           // MAC / batch = 86016

#define FIR_L 128                     // CPU FIR: cikti sayisi
#define FIR_K 32                      // tap sayisi (cikti basina 32 fmaf)
#define FIR_CHUNK 8                   // paylasimda coproc'u yoklarken bir adimda ilerletilen FIR ciktisi

static inline float    w2f(unsigned u){ union{unsigned u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)   { union{float f; uint32_t u;} c; c.f=f; return c.u; }

// ---- calisma tamponlari ----
float W1[M1*N1], W2[M2*N2], W3[M3*N3], Bb1[M1], Bb2[M2], Bb3[M3];
float x0[BATCH*N1];                                     // giris [B][N]
float out1[M1*BATCH], out2[M2*BATCH], out3[M3*BATCH];   // coproc ciktilari [M][B]
float ga1[BATCH*M1], ga2[BATCH*M2], ga3[BATCH*M3];      // CPU golden [B][M]
uint32_t loadbuf[3+BATCH*N1];                           // LOAD paketi: [N,M,B, x...]
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;
float sig[FIR_L+FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];
int   fir_n, fir_done;
unsigned g_setup, g_load, g_comp;     // coproc faz sayaclari (infer() her cagride sifirlar)
unsigned g_fir;                       // FIR'da harcanan CPU cycle (interleave dahil; fir_* fonksiyonlari toplar)

// ---- CPU FIR (kendi DSP isi) ----
static void fir_reset(void){ fir_n=0; fir_done=0; }
static inline void fir_one(void){ float s=0; for(int k=0;k<FIR_K;k++) s=fmaf(taps[k],sig[fir_n+k],s);
                                  firy[fir_n]=s; if(++fir_n==FIR_L) fir_done=1; }
static void fir_step(int budget){     // en fazla `budget` cikti ilerlet (coproc pollingi ile interleave)
  unsigned t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
  while(budget-->0 && !fir_done) fir_one();
  CSR_READ(CSR_REG_MCYCLE,&t1); g_fir+=t1-t0;
}
static void fir_finish(void){         // kalan FIR'i sonuna kadar bitir
  unsigned t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
  while(!fir_done) fir_one();
  CSR_READ(CSR_REG_MCYCLE,&t1); g_fir+=t1-t0;
}

// ---- CPU golden inference (bit-exact referans) ----
static void relu_bias(float* Y, const float* b, int M, int B, int relu){
  for(int m=0;m<M;m++) for(int j=0;j<B;j++){ Y[m*B+j]+=b[m]; if(relu&&Y[m*B+j]<0) Y[m*B+j]=0; }
}
static void golden(int B){
  for(int j=0;j<B;j++){
    for(int m=0;m<M1;m++){ float s=0; for(int i=0;i<N1;i++) s=fmaf(x0[j*N1+i],W1[m*N1+i],s);  ga1[j*M1+m]=s+Bb1[m]; if(ga1[j*M1+m]<0) ga1[j*M1+m]=0; }
    for(int m=0;m<M2;m++){ float s=0; for(int i=0;i<N2;i++) s=fmaf(ga1[j*M1+i],W2[m*N2+i],s); ga2[j*M2+m]=s+Bb2[m]; if(ga2[j*M2+m]<0) ga2[j*M2+m]=0; }
    for(int m=0;m<M3;m++){ float s=0; for(int i=0;i<N3;i++) s=fmaf(ga2[j*M2+i],W3[m*N3+i],s); ga3[j*M3+m]=s+Bb3[m]; }
  }
}
static int bitexact(int B){ for(int j=0;j<B;j++) for(int m=0;m<M3;m++) if(out3[m*B+j]!=ga3[j*M3+m]) return 0; return 1; }

// ---- coproc: bir katman. LOAD [N,M,B,x] -> agirliklari akit -> Y[M*B]. setup/load/compute'u g_*'a ekler. ----
// contend=1 ise compute (agirlik akisi) sirasinda CPU ayni anda FIR koser (tek FMA paylasilir).
static void fc(const float* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
  unsigned t0,t1;
  loadbuf[0]=N; loadbuf[1]=M; loadbuf[2]=B;
  for(int j=0;j<B;j++) for(int i=0;i<N;i++) loadbuf[3+j*N+i] = transpose ? f2u(x[i*B+j]) : f2u(x[j*N+i]);
  CSR_READ(CSR_REG_MCYCLE,&t0);                                    // setup: LOAD transfer'i programla
  ts.ptr=(uint8_t*)loadbuf; td.ptr=(uint8_t*)dummy; tr.size_d1_du=(uint32_t)(N*B+3); dma_load_transaction(&tr);
  CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr); while(!dma_is_ready(CH0));   // load: girisleri akit
  CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0);                                    // setup: WEIGHT transfer'i programla
  ts.ptr=(uint8_t*)W; td.ptr=(uint8_t*)Y; tr.size_d1_du=(uint32_t)(M*N); dma_load_transaction(&tr);
  CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr);                   // compute: coproc GEMM (+ contend ise CPU FIR)
  if(contend){ while(!dma_is_ready(CH0)) if(!fir_done) fir_step(FIR_CHUNK); }
  else       { while(!dma_is_ready(CH0)); }
  CSR_READ(CSR_REG_MCYCLE,&t1); g_comp+=t1-t0;
}
static void infer(int B, int contend){                            // coproc ile tam ag (3 katman)
  g_setup=g_load=g_comp=0;
  fc(W1,x0,   N1,M1,B,out1,0,contend); relu_bias(out1,Bb1,M1,B,1);
  fc(W2,out1, N2,M2,B,out2,1,contend); relu_bias(out2,Bb2,M2,B,1);
  fc(W3,out2, N3,M3,B,out3,1,contend); relu_bias(out3,Bb3,M3,B,0);
}

int main(void){
  CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));                           // FPU'yu ac
  CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);                       // cycle sayacini baslat
  setvbuf(stdout,NULL,_IONBF,0);
  // deterministik veriler (degerler timing'i etkilemez, sadece boyutlar onemli)
  for(int j=0;j<BATCH;j++) for(int i=0;i<N1;i++) x0[j*N1+i]=0.05f*(float)(((j*3+i)%13)-6);
  for(int m=0;m<M1;m++){ for(int i=0;i<N1;i++) W1[m*N1+i]=0.01f*(float)(((m*7+i*3)%17)-8); Bb1[m]=0.01f*(float)((m%5)-2); }
  for(int m=0;m<M2;m++){ for(int i=0;i<N2;i++) W2[m*N2+i]=0.01f*(float)(((m*5+i*2)%15)-7); Bb2[m]=0.01f*(float)((m%5)-2); }
  for(int m=0;m<M3;m++){ for(int i=0;i<N3;i++) W3[m*N3+i]=0.01f*(float)(((m*3+i*5)%13)-6); Bb3[m]=0.01f*(float)((m%5)-2); }
  for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f-0.5f;
  for(int k=0;k<FIR_K;k++) taps[k]=(float)((k*3+1)%11)/11.0f-0.4f;
  // DMA: hw_fifo_en=1 -> akis coprocessor'dan gecer
  ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  td=(dma_target_t){.ptr=(uint8_t*)dummy,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+3,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH0};
  dma_init(NULL);
  dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

  unsigned t0,t1;
  PRINTF("=== PERF BENCH PIPE  MLP %d-%d-%d-%d  B=%d  (%d MAC/img, %d MAC/batch) ===\n", N1,M1,M2,M3,BATCH,TOT_MAC,CMAC);

  // [1] CPU alone inference (DMA yok)
  CSR_READ(CSR_REG_MCYCLE,&t0); golden(BATCH); CSR_READ(CSR_REG_MCYCLE,&t1);
  unsigned cpu_inf = t1-t0;

  // [2] CPU alone FIR (DMA yok)
  fir_reset(); g_fir=0; fir_finish();
  unsigned fir_alone = g_fir; for(int i=0;i<FIR_L;i++) firref[i]=firy[i];

  // [3] Coproc alone inference (CPU bosta)
  CSR_READ(CSR_REG_MCYCLE,&t0); infer(BATCH,0); CSR_READ(CSR_REG_MCYCLE,&t1);
  unsigned co_tot=t1-t0, co_su=g_setup, co_ld=g_load, co_cp=g_comp; int be_co=bitexact(BATCH);

  // [4]+[5] Shared: coproc inference || CPU FIR (tek FMA)  -> ikisi de AYNI kosudan
  fir_reset(); g_fir=0;
  CSR_READ(CSR_REG_MCYCLE,&t0); infer(BATCH,1); CSR_READ(CSR_REG_MCYCLE,&t1);
  unsigned sh_tot=t1-t0, sh_su=g_setup, sh_ld=g_load, sh_cp=g_comp; int be_sh=bitexact(BATCH);
  fir_finish();                                    // coproc bitince kalan FIR'i tamamla
  unsigned sh_fir = g_fir;                          // shared'da CPU FIR toplam cycle (interleave + kalan)
  int firok=1; for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) firok=0;

  // ---------------- DETAY METRIKLER (setup / load / compute / total) ----------------
  PRINTF("[1] CPU    alone inference : setup=0 load=0 compute=%u total=%u\n", cpu_inf, cpu_inf);
  PRINTF("[2] CPU    alone FIR       : setup=0 load=0 compute=%u total=%u\n", fir_alone, fir_alone);
  PRINTF("[3] Coproc alone inference : setup=%u load=%u compute=%u total=%u  speedup=%u.%02ux cyc/MAC=%u.%02u bit-exact=%d\n",
         co_su, co_ld, co_cp, co_tot, cpu_inf/co_tot,(cpu_inf*100/co_tot)%100, co_cp/CMAC,(co_cp*100/CMAC)%100, be_co);
  PRINTF("[4] Shared CPU FIR         : setup=0 load=0 compute=%u total=%u  (alone %u)\n", sh_fir, sh_fir, fir_alone);
  PRINTF("[5] Shared Coproc inference: setup=%u load=%u compute=%u total=%u  (alone %u) bit-exact=%d\n",
         sh_su, sh_ld, sh_cp, sh_tot, co_tot, be_sh);

  // ---------------- OZET RAPOR (5 total cycle) ----------------
  PRINTF("\n--- RAPOR (total cycle) ---\n");
  PRINTF("  CPU    only   inference : %u\n", cpu_inf);
  PRINTF("  CPU    only   FIR       : %u\n", fir_alone);
  PRINTF("  Coproc only   inference : %u\n", co_tot);
  PRINTF("  Coproc shared inference : %u\n", sh_tot);
  PRINTF("  CPU    shared FIR       : %u\n", sh_fir);

  int all = be_co && be_sh && firok;
  PRINTF("=== ALL bit-exact=%d | coproc cyc/MAC=%u.%02u | speedup=%u.%02ux ===\n",
         all, co_cp/CMAC,(co_cp*100/CMAC)%100, cpu_inf/co_tot,(cpu_inf*100/co_tot)%100);
  return all?0:-1;
}

// perf_lenet_pipe -- pipelined single-coprocessor benchmark, REAL LeNet-300-100 / MNIST, B=8.
//
// Ayni yapi/olcumler: perf_bench_pipe. Farklar: gercek egitilmis agirliklar (header'da uint32 -> w2f),
// gercek MNIST goruntuleri, ve layer-1'de M*N=300*784=235200 > 16-bit DMA SIZE limiti oldugu icin
// agirliklar TEK 2D transfer ile akitilir (size_d1=N ic, size_d2=M dis, inc 1,1). Ayrica accuracy kontrolu.
//
// 5 METRIK (her biri: setup / load [DMA yoksa 0] / compute / total):
//   [1] CPU    alone inference   [2] CPU alone FIR   [3] Coproc alone inference
//   [4] Shared CPU FIR   [5] Shared Coproc inference   (+ sonda 5-total ozet RAPOR + accuracy)
// FIR = CPU'nun KENDI FP isi (inference DEGIL); paylasim maliyetini olcmek icin. Coproc CPU'nun TEK FMA'sini paylasir.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "../perf_lenet/lenet_mnist_data.h"   // L1_N..L3_M, fc{1,2,3}_{w,b}, test_images/labels, ref_pred, N_TEST
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0   1
#define BATCH 8                          // <= N_TEST

#define N1 L1_N   // 784
#define M1 L1_M   // 300
#define N2 L2_N   // 300
#define M2 L2_M   // 100
#define N3 L3_N   // 100
#define M3 L3_M   // 10
#define TOT_MAC (M1*N1 + M2*N2 + M3*N3)   // MAC / img  = 266200
#define CMAC    (TOT_MAC*BATCH)           // MAC / batch = 2129600

#define FIR_L 128
#define FIR_K 32
#define FIR_CHUNK 8

static inline float    w2f(unsigned u){ union{unsigned u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)   { union{float f; uint32_t u;} c; c.f=f; return c.u; }

// ---- calisma tamponlari (agirliklar/goruntuler header'da rodata; sadece yazilabilirler burada) ----
float x0[BATCH*N1];
float out1[M1*BATCH], out2[M2*BATCH], out3[M3*BATCH];   // coproc ciktilari [M][B]
float ga1[BATCH*M1], ga2[BATCH*M2], ga3[BATCH*M3];      // CPU golden [B][M]
uint32_t loadbuf[3+BATCH*N1];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;
float sig[FIR_L+FIR_K], taps[FIR_K], firy[FIR_L], firref[FIR_L];
int   fir_n, fir_done;
unsigned g_setup, g_load, g_comp;     // coproc faz sayaclari (infer() sifirlar)
unsigned g_fir;                       // FIR CPU cycle (interleave dahil)

// ---- CPU FIR (kendi DSP isi) ----
static void fir_reset(void){ fir_n=0; fir_done=0; }
static inline void fir_one(void){ float s=0; for(int k=0;k<FIR_K;k++) s=fmaf(taps[k],sig[fir_n+k],s);
                                  firy[fir_n]=s; if(++fir_n==FIR_L) fir_done=1; }
static void fir_step(int budget){ unsigned t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
  while(budget-->0 && !fir_done) fir_one(); CSR_READ(CSR_REG_MCYCLE,&t1); g_fir+=t1-t0; }
static void fir_finish(void){ unsigned t0,t1; CSR_READ(CSR_REG_MCYCLE,&t0);
  while(!fir_done) fir_one(); CSR_READ(CSR_REG_MCYCLE,&t1); g_fir+=t1-t0; }

// ---- CPU golden inference (gercek agirliklar w2f ile cozulur; bit-exact referans) ----
static void relu_bias(float* Y, const unsigned int* b, int M, int B, int relu){
  for(int m=0;m<M;m++) for(int j=0;j<B;j++){ Y[m*B+j]+=w2f(b[m]); if(relu&&Y[m*B+j]<0) Y[m*B+j]=0; }
}
static void golden(int B){
  for(int j=0;j<B;j++){
    for(int m=0;m<M1;m++){ float s=0; for(int i=0;i<N1;i++) s=fmaf(x0[j*N1+i],w2f(fc1_w[m*N1+i]),s);  ga1[j*M1+m]=s+w2f(fc1_b[m]); if(ga1[j*M1+m]<0) ga1[j*M1+m]=0; }
    for(int m=0;m<M2;m++){ float s=0; for(int i=0;i<N2;i++) s=fmaf(ga1[j*M1+i],w2f(fc2_w[m*N2+i]),s); ga2[j*M2+m]=s+w2f(fc2_b[m]); if(ga2[j*M2+m]<0) ga2[j*M2+m]=0; }
    for(int m=0;m<M3;m++){ float s=0; for(int i=0;i<N3;i++) s=fmaf(ga2[j*M2+i],w2f(fc3_w[m*N3+i]),s); ga3[j*M3+m]=s+w2f(fc3_b[m]); }
  }
}
static int bitexact(int B){ for(int j=0;j<B;j++) for(int m=0;m<M3;m++) if(out3[m*B+j]!=ga3[j*M3+m]) return 0; return 1; }
static int argmax_col(const float* Y, int j, int B, int M){ int best=0; for(int m=1;m<M;m++) if(Y[m*B+j]>Y[best*B+j]) best=m; return best; }

// ---- coproc: bir katman. LOAD [N,M,B,x] (1D) -> tum agirliklar (2D, tek transfer) -> Y[M*B]. ----
static void fc(const unsigned int* W, const float* x, int N, int M, int B, float* Y, int transpose, int contend){
  unsigned t0,t1;
  loadbuf[0]=N; loadbuf[1]=M; loadbuf[2]=B;
  for(int j=0;j<B;j++) for(int i=0;i<N;i++) loadbuf[3+j*N+i] = transpose ? f2u(x[i*B+j]) : f2u(x[j*N+i]);
  CSR_READ(CSR_REG_MCYCLE,&t0);                                    // setup: LOAD (1D) programla
  ts.ptr=(uint8_t*)loadbuf; ts.inc_d1_du=1; ts.inc_d2_du=0;
  td.ptr=(uint8_t*)dummy;   td.inc_d1_du=1; td.inc_d2_du=0;
  tr.dim=DMA_DIM_CONF_1D; tr.size_d1_du=(uint32_t)(N*B+3); tr.size_d2_du=0; dma_load_transaction(&tr);
  CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr); while(!dma_is_ready(CH0));   // load: girisleri akit
  CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0);                                    // setup: WEIGHT (2D) programla -- M*N tek transferde
  ts.ptr=(uint8_t*)W; ts.inc_d1_du=1; ts.inc_d2_du=1;
  td.ptr=(uint8_t*)Y; td.inc_d1_du=1; td.inc_d2_du=1;
  tr.dim=DMA_DIM_CONF_2D; tr.size_d1_du=(uint32_t)N; tr.size_d2_du=(uint32_t)M; dma_load_transaction(&tr);
  CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
  CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr);                   // compute: coproc GEMM (+ contend ise CPU FIR)
  if(contend){ while(!dma_is_ready(CH0)) if(!fir_done) fir_step(FIR_CHUNK); }
  else       { while(!dma_is_ready(CH0)); }
  CSR_READ(CSR_REG_MCYCLE,&t1); g_comp+=t1-t0;
}
static void infer(int B, int contend){
  g_setup=g_load=g_comp=0;
  fc(fc1_w,x0,   N1,M1,B,out1,0,contend); relu_bias(out1,fc1_b,M1,B,1);
  fc(fc2_w,out1, N2,M2,B,out2,1,contend); relu_bias(out2,fc2_b,M2,B,1);
  fc(fc3_w,out2, N3,M3,B,out3,1,contend); relu_bias(out3,fc3_b,M3,B,0);
}

int main(void){
  CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
  CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
  setvbuf(stdout,NULL,_IONBF,0);
  for(int j=0;j<BATCH;j++) for(int i=0;i<N1;i++) x0[j*N1+i]=w2f(test_images[j*N1+i]);   // gercek MNIST goruntuleri
  for(int i=0;i<FIR_L+FIR_K;i++) sig[i]=(float)((i*7+3)%17)/17.0f-0.5f;
  for(int k=0;k<FIR_K;k++) taps[k]=(float)((k*3+1)%11)/11.0f-0.4f;
  ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  td=(dma_target_t){.ptr=(uint8_t*)dummy,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=N1+3,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH0};
  dma_init(NULL);
  dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

  unsigned t0,t1;
  PRINTF("=== PERF LENET PIPE  LeNet-300-100 %d-%d-%d-%d  B=%d  (%d MAC/img, %d MAC/batch) ===\n", N1,M1,M2,M3,BATCH,TOT_MAC,CMAC);

  // [1] CPU alone inference
  CSR_READ(CSR_REG_MCYCLE,&t0); golden(BATCH); CSR_READ(CSR_REG_MCYCLE,&t1);
  unsigned cpu_inf = t1-t0;

  // [2] CPU alone FIR
  fir_reset(); g_fir=0; fir_finish();
  unsigned fir_alone = g_fir; for(int i=0;i<FIR_L;i++) firref[i]=firy[i];

  // [3] Coproc alone inference
  CSR_READ(CSR_REG_MCYCLE,&t0); infer(BATCH,0); CSR_READ(CSR_REG_MCYCLE,&t1);
  unsigned co_tot=t1-t0, co_su=g_setup, co_ld=g_load, co_cp=g_comp; int be_co=bitexact(BATCH);

  // [4]+[5] Shared: coproc inference || CPU FIR
  fir_reset(); g_fir=0;
  CSR_READ(CSR_REG_MCYCLE,&t0); infer(BATCH,1); CSR_READ(CSR_REG_MCYCLE,&t1);
  unsigned sh_tot=t1-t0, sh_su=g_setup, sh_ld=g_load, sh_cp=g_comp; int be_sh=bitexact(BATCH);
  fir_finish();
  unsigned sh_fir = g_fir;
  int firok=1; for(int i=0;i<FIR_L;i++) if(firy[i]!=firref[i]) firok=0;

  // accuracy: coproc logit -> tahmin; MNIST etiket + offline referans ile karsilastir
  int correct=0, match=0;
  for(int j=0;j<BATCH;j++){ int p=argmax_col(out3,j,BATCH,M3); if(p==test_labels[j]) correct++; if(p==ref_pred[j]) match++; }

  // ---------------- DETAY METRIKLER (setup / load / compute / total) ----------------
  PRINTF("[1] CPU    alone inference : setup=0 load=0 compute=%u total=%u\n", cpu_inf, cpu_inf);
  PRINTF("[2] CPU    alone FIR       : setup=0 load=0 compute=%u total=%u\n", fir_alone, fir_alone);
  PRINTF("[3] Coproc alone inference : setup=%u load=%u compute=%u total=%u  speedup=%u.%02ux cyc/MAC=%u.%02u bit-exact=%d\n",
         co_su, co_ld, co_cp, co_tot, cpu_inf/co_tot,(cpu_inf*100/co_tot)%100, co_cp/CMAC,(co_cp*100/CMAC)%100, be_co);
  PRINTF("[4] Shared CPU FIR         : setup=0 load=0 compute=%u total=%u  (alone %u)\n", sh_fir, sh_fir, fir_alone);
  PRINTF("[5] Shared Coproc inference: setup=%u load=%u compute=%u total=%u  (alone %u) bit-exact=%d\n",
         sh_su, sh_ld, sh_cp, sh_tot, co_tot, be_sh);
  PRINTF("[acc] correct=%d/%d  match_ref=%d/%d\n", correct,BATCH, match,BATCH);

  // ---------------- OZET RAPOR (5 total cycle) ----------------
  PRINTF("\n--- RAPOR (total cycle) ---\n");
  PRINTF("  CPU    only   inference : %u\n", cpu_inf);
  PRINTF("  CPU    only   FIR       : %u\n", fir_alone);
  PRINTF("  Coproc only   inference : %u\n", co_tot);
  PRINTF("  Coproc shared inference : %u\n", sh_tot);
  PRINTF("  CPU    shared FIR       : %u\n", sh_fir);

  int all = be_co && be_sh && firok;
  PRINTF("=== ALL bit-exact=%d | coproc cyc/MAC=%u.%02u | speedup=%u.%02ux | acc %d/%d ===\n",
         all, co_cp/CMAC,(co_cp*100/CMAC)%100, cpu_inf/co_tot,(cpu_inf*100/co_tot)%100, correct,BATCH);
  return all?0:-1;
}

// perf_resnet8_pipe -- pipelined single-coprocessor benchmark, REAL ResNet-8 (MLPerf-Tiny) / CIFAR-10, B=8.
//
// Conv katmanlari im2col ile mevcut [N,M,B] protokolune oturur (RTL degisikligi YOK):
//   her cikis pikseli icin 8 goruntunun patch'i LOAD edilir (N=K*K*Ci), filtre banki WEIGHT
//   olarak akitilir (M=Co satir), Co*8 sonuc doner. Padding = TF-'same' (referansin kendi
//   konvansiyonu); patch sirasi (ky,kx,ci) — export ile birebir. BN egitim sonrasi conv'a
//   katlanmis (matematiksel olarak ayni ag). Residual add + ReLU + avgpool CPU'da (iki yol
//   icin de ayni kod, BIR kez olculur ve iki totale de eklenir).
//
// METRIKLER (kirpik — co-execution MLP/LeNet'te zaten var):
//   [1] CPU alone inference (optimize golden: 8 batch-akumulator + agirlik reuse, patch-sirali
//       akumulasyon -> coproc ile BIT-EXACT)   [3] Coproc alone inference (setup/load/compute/total)
//   + katman-katman bit-exact (bug ilk katmanda gorunur, 27 saat sonra degil) + accuracy (8 img).
// Golden, padding pozisyonlarinda da coproc gibi fmaf(w, 0.0f, acc) YURUTUR (atlamaz) -> bit-exact
// butun kose durumlarinda garantili; MAC sayimi da standart sayimla (padding dahil) tutarli.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#include "../perf_resnet8/resnet8_cifar_data.h"   // c1_w/b .. b3s_w/b, fc_w/b, test_images/labels, ref_pred
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define CH0   1
#define BATCH 8

#define TOT_MAC 12501632                 // MAC/img (standart sayim, padding dahil)
#define CMAC    (TOT_MAC*BATCH)          // MAC/batch = 100 013 056

static inline float    w2f(unsigned u){ union{unsigned u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)   { union{float f; uint32_t u;} c; c.f=f; return c.u; }

// ---- tamponlar (HWC, [j][H][W][C]; max tensor 32*32*16 per img) ----
#define ABUF (BATCH*32*32*16)            // 131072 float = 512 KB
float S[ABUF], T1[ABUF], T2[ABUF], TC[ABUF];   // TC = coproc cikisi (golden ile karsilastirilir)
float ytmp[64*BATCH];                    // bir pikselin M*B sonucu (DMA dst)
uint32_t loadbuf[3 + BATCH*576];         // LOAD paketi: [N,M,B, patch*8]  (max N=576)
float pooled[BATCH*64], glog[BATCH*10];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;
unsigned g_setup, g_load, g_comp;        // coproc faz sayaclari ([3] breakdown)
unsigned cpu_tot, acc_tot, elem_tot;     // [1]/[3] toplamlari + paylasilan elementwise
int all_be = 1;

// ---- CPU golden conv (optimize: 8 bagimsiz batch-akumulator + agirlik reuse; (ky,kx,c) sirasi) ----
static void conv_cpu(const unsigned int* W, const unsigned int* bias, const float* in, float* out,
                     int Hi,int Wi,int Ci,int Ho,int Wo,int Co,int K,int St,int PB,int relu){
  int sz = Hi*Wi*Ci, osz = Ho*Wo*Co, N = K*K*Ci;
  for(int y=0;y<Ho;y++) for(int x=0;x<Wo;x++){
    for(int m=0;m<Co;m++){
      const unsigned int* wm = &W[m*N];
      float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
      int i=0;
      for(int ky=0;ky<K;ky++){ int iy=y*St+ky-PB;
        for(int kx=0;kx<K;kx++){ int ix=x*St+kx-PB;
          int inb = (iy>=0 && iy<Hi && ix>=0 && ix<Wi);
          const float* b0 = in + (iy*Wi+ix)*Ci;      // inb=0 iken kullanilmaz
          for(int c=0;c<Ci;c++,i++){
            float w = w2f(wm[i]);
            float x0=inb?b0[0*sz+c]:0.0f, x1=inb?b0[1*sz+c]:0.0f, x2=inb?b0[2*sz+c]:0.0f, x3=inb?b0[3*sz+c]:0.0f,
                  x4=inb?b0[4*sz+c]:0.0f, x5=inb?b0[5*sz+c]:0.0f, x6=inb?b0[6*sz+c]:0.0f, x7=inb?b0[7*sz+c]:0.0f;
            a0=fmaf(w,x0,a0); a1=fmaf(w,x1,a1); a2=fmaf(w,x2,a2); a3=fmaf(w,x3,a3);
            a4=fmaf(w,x4,a4); a5=fmaf(w,x5,a5); a6=fmaf(w,x6,a6); a7=fmaf(w,x7,a7);
          } } }
      float bb = w2f(bias[m]);
      #define STO(J,A){ float v=(A)+bb; if(relu&&v<0)v=0; out[(J)*osz+(y*Wo+x)*Co+m]=v; }
      STO(0,a0)STO(1,a1)STO(2,a2)STO(3,a3)STO(4,a4)STO(5,a5)STO(6,a6)STO(7,a7)
      #undef STO
    } }
}

// ---- coproc conv: her cikis pikseli icin LOAD(8 patch) + WEIGHT(filtre banki) ----
static void conv_acc(const unsigned int* W, const unsigned int* bias, const float* in, float* out,
                     int Hi,int Wi,int Ci,int Ho,int Wo,int Co,int K,int St,int PB,int relu){
  int sz = Hi*Wi*Ci, osz = Ho*Wo*Co, N = K*K*Ci;
  unsigned t0,t1;
  for(int y=0;y<Ho;y++) for(int x=0;x<Wo;x++){
    // patch marshalling (CPU isi; [3] total'ine dahil, alt-bracket'larin disinda)
    loadbuf[0]=(uint32_t)N; loadbuf[1]=(uint32_t)Co; loadbuf[2]=BATCH;
    int idx=3;
    for(int j=0;j<BATCH;j++){
      const float* aj = in + j*sz;
      for(int ky=0;ky<K;ky++){ int iy=y*St+ky-PB;
        for(int kx=0;kx<K;kx++){ int ix=x*St+kx-PB;
          if(iy>=0 && iy<Hi && ix>=0 && ix<Wi){
            const float* src = aj + (iy*Wi+ix)*Ci;
            for(int c=0;c<Ci;c++) loadbuf[idx++] = f2u(src[c]);
          } else
            for(int c=0;c<Ci;c++) loadbuf[idx++] = 0;   // 0.0f (padding)
        } } }
    // LOAD
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts.ptr=(uint8_t*)loadbuf; ts.inc_d1_du=1; ts.inc_d2_du=0;
    td.ptr=(uint8_t*)dummy;   td.inc_d1_du=1; td.inc_d2_du=0;
    tr.dim=DMA_DIM_CONF_1D; tr.size_d1_du=(uint32_t)(N*BATCH+3); tr.size_d2_du=0; dma_load_transaction(&tr);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr); while(!dma_is_ready(CH0));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_load+=t1-t0;
    // WEIGHT (compute)
    CSR_READ(CSR_REG_MCYCLE,&t0);
    ts.ptr=(uint8_t*)W; td.ptr=(uint8_t*)ytmp;
    tr.size_d1_du=(uint32_t)(Co*N); dma_load_transaction(&tr);
    CSR_READ(CSR_REG_MCYCLE,&t1); g_setup+=t1-t0;
    CSR_READ(CSR_REG_MCYCLE,&t0); dma_launch(&tr); while(!dma_is_ready(CH0));
    CSR_READ(CSR_REG_MCYCLE,&t1); g_comp+=t1-t0;
    // scatter + bias (+relu)  (Y[m*B+j] duzeni)
    for(int m=0;m<Co;m++){
      float bb = w2f(bias[m]);
      for(int j=0;j<BATCH;j++){
        float v = ytmp[m*BATCH+j] + bb; if(relu && v<0) v=0;
        out[j*osz + (y*Wo+x)*Co + m] = v;
      } }
  }
}

// ---- bir katmani iki yolla kosustur + bit-exact karsilastir + zamanlari topla ----
static void run_conv(const char* nm, const unsigned int* W, const unsigned int* b,
                     const float* in, float* gout, float* aout,
                     int Hi,int Wi,int Ci,int Ho,int Wo,int Co,int K,int St,int PB,int relu){
  unsigned t0,t1,ccpu,cacc;
  CSR_READ(CSR_REG_MCYCLE,&t0); conv_cpu(W,b,in,gout,Hi,Wi,Ci,Ho,Wo,Co,K,St,PB,relu);
  CSR_READ(CSR_REG_MCYCLE,&t1); ccpu=t1-t0; cpu_tot+=ccpu;
  CSR_READ(CSR_REG_MCYCLE,&t0); conv_acc(W,b,in,aout,Hi,Wi,Ci,Ho,Wo,Co,K,St,PB,relu);
  CSR_READ(CSR_REG_MCYCLE,&t1); cacc=t1-t0; acc_tot+=cacc;
  int be=1, n=BATCH*Ho*Wo*Co;
  for(int i=0;i<n;i++) if(f2u(gout[i])!=f2u(aout[i])){ be=0; break; }
  all_be &= be;
  PRINTF("[L %-4s] bit-exact=%d  cpu=%u  acc=%u\n", nm, be, ccpu, cacc);
}

// ---- paylasilan elementwise (iki yol icin ayni is; BIR kez olculur, iki totale eklenir) ----
static void add_relu(float* dst, const float* a, const float* b, int n){
  for(int i=0;i<n;i++){ float v=a[i]+b[i]; dst[i]=(v<0)?0.0f:v; }
}
static void avgpool8(const float* in, float* out){        // [j][8][8][64] -> [j][64]
  for(int j=0;j<BATCH;j++) for(int c=0;c<64;c++){
    float s2=0; for(int p=0;p<64;p++) s2 += in[j*8*8*64 + p*64 + c];
    out[j*64+c] = s2*(1.0f/64.0f);
  }
}

int main(void){
  CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
  CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);
  setvbuf(stdout,NULL,_IONBF,0);
  // goruntuler -> S ([8][32][32][3], header'da normalize HWC bit-pattern)
  for(int i=0;i<BATCH*32*32*3;i++) S[i]=w2f(test_images[i]);
  // DMA (lenet ile ayni: hw_fifo ch1, 1D, polling)
  ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  td=(dma_target_t){.ptr=(uint8_t*)dummy,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
  tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,.dim=DMA_DIM_CONF_1D,.size_d1_du=3+64*BATCH,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=CH0};
  dma_init(NULL);
  dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

  unsigned t0,t1;
  PRINTF("=== PERF RESNET8 PIPE  MLPerf-Tiny ResNet-8 / CIFAR-10  B=%d  (%d MAC/img, %d MAC/batch) ===\n",
         BATCH, TOT_MAC, CMAC);
  g_setup=g_load=g_comp=0; cpu_tot=acc_tot=elem_tot=0;

  // conv1 + block1
  run_conv("c1",  c1_w,c1_b,     S,  T1, TC, 32,32,3,  32,32,16, 3,1,1, 1);
  run_conv("b1c1",b1c1_w,b1c1_b, T1, T2, TC, 32,32,16, 32,32,16, 3,1,1, 1);
  run_conv("b1c2",b1c2_w,b1c2_b, T2, S,  TC, 32,32,16, 32,32,16, 3,1,1, 0);
  CSR_READ(CSR_REG_MCYCLE,&t0); add_relu(T2, T1, S, BATCH*32*32*16);
  CSR_READ(CSR_REG_MCYCLE,&t1); elem_tot+=t1-t0;
  // block2 (stride-2; TF-'same' -> PB=0, sadece alt/sag pad)
  run_conv("b2c1",b2c1_w,b2c1_b, T2, T1, TC, 32,32,16, 16,16,32, 3,2,0, 1);
  run_conv("b2c2",b2c2_w,b2c2_b, T1, S,  TC, 16,16,32, 16,16,32, 3,1,1, 0);
  run_conv("b2s", b2s_w, b2s_b,  T2, T1, TC, 32,32,16, 16,16,32, 1,2,0, 0);
  CSR_READ(CSR_REG_MCYCLE,&t0); add_relu(T2, T1, S, BATCH*16*16*32);
  CSR_READ(CSR_REG_MCYCLE,&t1); elem_tot+=t1-t0;
  // block3
  run_conv("b3c1",b3c1_w,b3c1_b, T2, T1, TC, 16,16,32, 8,8,64, 3,2,0, 1);
  run_conv("b3c2",b3c2_w,b3c2_b, T1, S,  TC, 8,8,64,   8,8,64, 3,1,1, 0);
  run_conv("b3s", b3s_w, b3s_b,  T2, T1, TC, 16,16,32, 8,8,64, 1,2,0, 0);
  CSR_READ(CSR_REG_MCYCLE,&t0); add_relu(T2, T1, S, BATCH*8*8*64);
  CSR_READ(CSR_REG_MCYCLE,&t1); elem_tot+=t1-t0;
  // global avgpool + FC (FC = 1x1 "conv": Hi=Wi=Ho=Wo=1, Ci=64, Co=10)
  CSR_READ(CSR_REG_MCYCLE,&t0); avgpool8(T2, pooled);
  CSR_READ(CSR_REG_MCYCLE,&t1); elem_tot+=t1-t0;
  run_conv("fc",  fc_w, fc_b,    pooled, glog, TC, 1,1,64, 1,1,10, 1,1,0, 0);

  // toplamlar: paylasilan elementwise iki yola da eklenir
  unsigned cpu_inf = cpu_tot + elem_tot;
  unsigned acc_inf = acc_tot + elem_tot;

  // accuracy: coproc logitlerinden (TC'nin ilk 80'i, [j][10])
  int correct=0, match=0;
  for(int j=0;j<BATCH;j++){
    int best=0; for(int m=1;m<10;m++) if(TC[j*10+m]>TC[j*10+best]) best=m;
    if(best==test_labels[j]) correct++;
    if(best==ref_pred[j])    match++;
  }

  PRINTF("[1] CPU    alone inference : setup=0 load=0 compute=%u total=%u\n", cpu_inf, cpu_inf);
  PRINTF("[3] Coproc alone inference : setup=%u load=%u compute=%u total=%u  speedup=%u.%02ux cyc/MAC=%u.%02u bit-exact=%d\n",
         g_setup, g_load, g_comp, acc_inf,
         cpu_inf/acc_inf,(unsigned)(((unsigned long long)cpu_inf*100/acc_inf)%100),
         g_comp/CMAC,(unsigned)(((unsigned long long)g_comp*100/CMAC)%100), all_be);
  PRINTF("[acc] correct=%d/%d  match_ref=%d/%d\n", correct,BATCH, match,BATCH);
  PRINTF("=== ALL bit-exact=%d | coproc cyc/MAC=%u.%02u | speedup=%u.%02ux | acc %d/%d ===\n",
         all_be, g_comp/CMAC,(unsigned)(((unsigned long long)g_comp*100/CMAC)%100),
         cpu_inf/acc_inf,(unsigned)(((unsigned long long)cpu_inf*100/acc_inf)%100), correct,BATCH);
  return all_be?0:-1;
}

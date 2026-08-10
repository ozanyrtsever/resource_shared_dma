#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF_IN_SIM 1
#if TARGET_SIM && PRINTF_IN_SIM
  #define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
  #define PRINTF(...)
#endif
#define ACC_N 32
#define ACC_M 32        // accel CPU loop'u boyunca meşgul kalır, daha az DMA bus trafiği
#define CPU_K 512       // register-only CPU fmadd sayısı
#define DMA_CH 1

static inline float    w2f(unsigned int u){ union{unsigned int u; float f;} c; c.u=u; return c.f; }
static inline uint32_t f2u(float f)        { union{float f; uint32_t u;} c; c.f=f; return c.u; }

float x_acc[ACC_N], w_acc[ACC_M*ACC_N], out_acc[ACC_M];
uint32_t loadbuf[ACC_N + 2];
float dummy[4];
volatile float g_seed = 1.000001f;   // opak seed: derleyici loop'u katlayamasın/CSE edemesin
volatile float g_sink;               // dead-code elimination'ı engeller
dma_target_t ts, td; dma_trans_t tr;

static inline void xfer(const void* s, void* d, uint32_t n){
    ts.ptr=(uint8_t*)s; td.ptr=(uint8_t*)d; tr.size_d1_du=n;
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
}

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);

    float gA=0.0f;
    for(int i=0;i<ACC_N;i++) x_acc[i]=0.1f*(float)(i+1);
    for(int m=0;m<ACC_M;m++) for(int i=0;i<ACC_N;i++) w_acc[m*ACC_N+i]=0.01f*(float)((m+1)*(i+1));
    for(int i=0;i<ACC_N;i++) gA += x_acc[i]*w_acc[i];

    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=ACC_N+2,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY);

    loadbuf[0]=ACC_N; loadbuf[1]=ACC_M;
    for(int i=0;i<ACC_N;i++) loadbuf[2+i]=f2u(x_acc[i]);
    xfer(loadbuf, dummy, ACC_N+2);

    unsigned int cpu_alone, cpu_cont;
    float k = 0.9999f, c = 0.0001f;      // register sabitler (loop dışına hoist edilir)
    float acc;

    // 1) CPU FP tek başına — register-only fmadd zinciri (bellek operandı YOK -> veri-bus yarışması YOK)
    acc = g_seed;
    CSR_WRITE(CSR_REG_MCYCLE,0);
    for(int i=0;i<CPU_K;i++) acc = fmaf(acc, k, c);
    CSR_READ(CSR_REG_MCYCLE,&cpu_alone);
    g_sink = acc;

    // 2) ÇEKİŞME — accel'i non-blocking başlat, AYNI register-only zincir
    acc = g_seed;
    ts.ptr=(uint8_t*)w_acc; td.ptr=(uint8_t*)out_acc; tr.size_d1_du=(uint32_t)(ACC_M*ACC_N);
    dma_load_transaction(&tr); dma_launch(&tr);
    CSR_WRITE(CSR_REG_MCYCLE,0);
    for(int i=0;i<CPU_K;i++) acc = fmaf(acc, k, c);
    CSR_READ(CSR_REG_MCYCLE,&cpu_cont);
    g_sink = acc;
    while(!dma_is_ready(DMA_CH));

    int accOK = (fabsf(out_acc[0]-gA) < 0.05f);
    PRINTF("CPU FP alone=%u  under-contention=%u  (delta=%d cyc)\n",
           cpu_alone, cpu_cont, (int)cpu_cont-(int)cpu_alone);
    PRINTF("accel dot0=%d golden=%d OK=%d\n", (int)(out_acc[0]*1000),(int)(gA*1000),accOK);
    if(!accOK){ PRINTF("CONTENTION: FAIL\n"); return -3; }
    PRINTF("CONTENTION: PASS - delta = CPU drain penalty (bus-free)\n");
    return 0;
}

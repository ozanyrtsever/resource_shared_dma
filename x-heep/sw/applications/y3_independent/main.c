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

#define NP 32              // testharness num_pairs_i ile AYNI olmalı (varsayılan 32)
#define NW (2*NP)
#define DMA_CH 1

#define CITER 15           // compute-bound yineleme (accel ~208 cyc içinde kalsın)
#define AN    12           // memory-bound integer dizi boyu

volatile uint32_t g_seed = 1u;


float src_buf[NW] __attribute__((aligned(4)));
float dst_buf[NW] __attribute__((aligned(4)));
volatile uint32_t work_buf[AN] __attribute__((aligned(4)));   // CPU'nun KENDİ integer verisi

// (1) compute-bound: saf register aritmetiği — bellek YOK, FP YOK
static uint32_t cpu_compute(uint32_t seed){
    uint32_t x=seed;
    for(int i=0;i<CITER;i++){ x^=x<<13; x^=x>>17; x^=x<<5; x+=0x9E3779B9u; }
    return x;
}
// (2) memory-bound: integer diziyi tara+yaz — FP YOK, ama bus kullanır
static uint32_t cpu_memwork(void){
    uint32_t s=0;
    for(int i=0;i<AN;i++) s += work_buf[i] ^ (uint32_t)i;
    for(int i=0;i<AN;i++) work_buf[i] = s + (uint32_t)i;
    return s;
}

// accel'i tazeden başlat + verilen kernel'i eşzamanlı ölç
static unsigned int run_concurrent(dma_trans_t* tr, float gA, int* okOut, int isMem, volatile uint32_t* sink){
    unsigned int c;
    dst_buf[0]=0.0f;
    dma_load_transaction(tr);      // her seferinde yeniden yükle (flush accel'i resetler)
    dma_launch(tr);                // accel arka planda başladı (~208 cyc)
    CSR_WRITE(CSR_REG_MCYCLE,0);
    if(isMem) *sink ^= cpu_memwork(); else *sink ^= cpu_compute(g_seed);
    CSR_READ(CSR_REG_MCYCLE,&c);
    while(!dma_is_ready(DMA_CH));   // accel bitsin
    *okOut = (fabs(dst_buf[0]-gA) < 0.5f);
    return c;
}

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT,0x1);

    float gA=0.0f;
    for(int i=0;i<NP;i++){ float a=i+1,b=i+1; src_buf[2*i]=a; src_buf[2*i+1]=b; gA+=a*b; }
    for(int i=0;i<AN;i++) work_buf[i]=(uint32_t)i*2654435761u;

    dma_target_t ts={.ptr=(uint8_t*)src_buf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    dma_target_t td={.ptr=(uint8_t*)dst_buf,.inc_d1_du=1,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    dma_trans_t tr={.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                    .dim=DMA_DIM_CONF_1D,.size_d1_du=NW,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);
    if(dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY)!=DMA_CONFIG_OK){PRINTF("val FAIL\n");return -1;}
    if(dma_load_transaction(&tr)!=DMA_CONFIG_OK){PRINTF("load FAIL\n");return -2;}

    volatile uint32_t sink=0;
    unsigned int c_alone,m_alone,c_cont,m_cont;
    int ok1,ok2;

    // --- ALONE (accel boşta) ---
    CSR_WRITE(CSR_REG_MCYCLE,0); sink^=cpu_compute(g_seed); CSR_READ(CSR_REG_MCYCLE,&c_alone);
    CSR_WRITE(CSR_REG_MCYCLE,0); sink^=cpu_memwork();   CSR_READ(CSR_REG_MCYCLE,&m_alone);

    // --- CONCURRENT (accel FMA + bus kullanıyor) ---
    c_cont = run_concurrent(&tr,gA,&ok1,0,&sink);
    m_cont = run_concurrent(&tr,gA,&ok2,1,&sink);

    PRINTF("== CPU BAGIMSIZ (NON-FP) is / accel FMA'yi kullanirken ==\n");
    PRINTF("compute-bound: alone=%u concurrent=%u  delta=%d cyc\n", c_alone,c_cont,(int)c_cont-(int)c_alone);
    PRINTF("memory-bound : alone=%u concurrent=%u  delta=%d cyc\n", m_alone,m_cont,(int)m_cont-(int)m_alone);
    PRINTF("accel OK: compute-run=%d mem-run=%d  sink=%08x\n", ok1,ok2,(unsigned)sink);
    if(!(ok1&&ok2)){PRINTF("INDEP: FAIL\n");return -3;}
    PRINTF("INDEP: PASS - accel dogru bitti; yukaridaki delta = CPU'ya binen latency\n");
    return 0;
}

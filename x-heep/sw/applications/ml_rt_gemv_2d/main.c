// ml_rt_gemv_2d — does the coprocessor GEMV work when the WHOLE M×N (> 65535) weight matrix is
// streamed in ONE 2D DMA transfer (size_d1 = N inner, size_d2 = M outer)?  Fast variant: golden is
// SPOT-CHECKED on a few rows only (full golden = 65792 FMAs = ~2.6 min in sim), so the real question
// (does the >65535 2D transfer COMPLETE, or hang?) shows quickly.
// NOTE: W is a ~263 KB .bss array -> crt0 zeroes it at startup, which costs ~40-60 s of sim BEFORE
// the first print. That startup delay is why "no UART output" appeared — it is not a hang.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dma.h"
#include "csr.h"
#include "x-heep.h"
#define PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define DMA_CH 1
#define N 256
#define M 257            // M*N = 65792 > 65535  (1D SIZE_D1 cannot; 2D can)

static inline uint32_t f2u(float f){ union{float f; uint32_t u;} c; c.f=f; return c.u; }

float x[N], W[M*N], out[M];
uint32_t loadbuf[N + 2];
float dummy[4];
dma_target_t ts, td; dma_trans_t tr;

#define NCHK 6
int   chk_row[NCHK]  = {0, 1, 128, 254, 255, 256};   // includes last row 256 (M-1), which crosses dst rows since M>N
float chk_gold[NCHK];

int main(void){
    CSR_SET_BITS(CSR_REG_MSTATUS,(1<<13));
    setvbuf(stdout, NULL, _IONBF, 0);
    PRINTF("[1] main reached (startup .bss zeroing done): N=%d M=%d total=%d >65535? %s\n",
           N, M, M*N, (M*N>65535)?"YES":"no");

    for(int i=0;i<N;i++) x[i]=0.1f*(float)(i+1);
    for(int m=0;m<M;m++) for(int i=0;i<N;i++) W[m*N+i]=0.01f*(float)((m+1)*(i+1))-0.05f;
    PRINTF("[2] data filled; spot-golden on %d rows...\n", NCHK);
    for(int c=0;c<NCHK;c++){ int m=chk_row[c]; float s=0.0f; for(int i=0;i<N;i++) s=fmaf(x[i],W[m*N+i],s); chk_gold[c]=s; }
    PRINTF("[3] spot-golden done\n");

    ts=(dma_target_t){.ptr=(uint8_t*)loadbuf,.inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    td=(dma_target_t){.ptr=(uint8_t*)dummy,  .inc_d1_du=1,.inc_d2_du=0,.trig=DMA_TRIG_MEMORY,.type=DMA_DATA_TYPE_WORD};
    tr=(dma_trans_t){.src=&ts,.dst=&td,.mode=DMA_TRANS_MODE_SINGLE,.hw_fifo_en=1,
                     .dim=DMA_DIM_CONF_1D,.size_d1_du=N+2,.size_d2_du=0,.end=DMA_TRANS_END_POLLING,.channel=DMA_CH};
    dma_init(NULL);

    loadbuf[0]=N; loadbuf[1]=M; for(int i=0;i<N;i++) loadbuf[2+i]=f2u(x[i]);
    PRINTF("[4] LOAD x (1D, %d words)...\n", N+2);
    if(dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY)!=DMA_CONFIG_OK){ PRINTF("    LOAD validate FAIL\n"); return -1; }
    dma_load_transaction(&tr); dma_launch(&tr); while(!dma_is_ready(DMA_CH));
    PRINTF("[5] LOAD done\n");

    ts.ptr=(uint8_t*)W;   ts.inc_d1_du=1; ts.inc_d2_du=1;
    td.ptr=(uint8_t*)out; td.inc_d1_du=1; td.inc_d2_du=1;
    tr.dim=DMA_DIM_CONF_2D; tr.size_d1_du=N; tr.size_d2_du=M;
    PRINTF("[6] GEMV 2D validate: size_d1=%d size_d2=%d total=%d...\n", N, M, N*M);
    if(dma_validate_transaction(&tr,DMA_ENABLE_REALIGN,DMA_PERFORM_CHECKS_INTEGRITY)!=DMA_CONFIG_OK){ PRINTF("    2D validate FAIL\n"); return -2; }
    PRINTF("[7] validated, launching 2D transfer...\n");
    dma_load_transaction(&tr); dma_launch(&tr);
    PRINTF("[8] launched, polling (this streams 65792 weights -> ~1-2 min accel work; if NO [9] -> hang)\n");
    while(!dma_is_ready(DMA_CH));
    PRINTF("[9] 2D GEMV COMPLETED (no hang!)\n");

    int bad=0;
    for(int c=0;c<NCHK;c++){ int m=chk_row[c]; float e=fabsf(out[m]-chk_gold[c]);
        PRINTF("    row %3d: got=%d gold=%d (x1000) %s\n", m,(int)(out[m]*1000),(int)(chk_gold[c]*1000), (e>1e-6f)?"BAD":"ok");
        if(e>1e-6f) bad++; }
    PRINTF("RT-GEMV-2D N=%d M=%d total=%d: spot bad=%d/%d %s\n", N,M,N*M,bad,NCHK,bad?"FAIL":"PASS");
    return bad?-1:0;
}

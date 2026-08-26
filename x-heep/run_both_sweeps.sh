#!/bin/bash
# run_both_sweeps.sh — MLP (perf_bench_pipe) SONRA LeNet (perf_lenet_pipe), ardisik, lokal/gece boyu.
# LeNet ancak MLP bitince baslar (ikisi ayni config dosyalarini sed'liyor -> paralel OLAMAZ, ardisik olmali).
# Apps'e MAC metrikleri eklendigi icin eski (MAC'siz) sonuclar yana alinir -> tam yeni kosu
# (yoksa sweep.sh "ALL bit-exact=1" goren config'leri ATLAR ve yeni metrikler cikmaz).
set -u
cd /home/ozan/thesis/resource_shared_dma/x-heep || { echo "wrong dir"; exit 1; }
TS=$(date +%Y%m%d_%H%M%S)

# eski sonuc klasorlerini yedekle (silme degil, tasi)
for d in bench_pipe lenet_pipe; do
  [ -d "sweep_results/$d" ] && mv "sweep_results/$d" "sweep_results/${d}.bak_${TS}" \
    && echo "[$(date +%H:%M:%S)] eski sweep_results/$d -> .bak_${TS} olarak yedeklendi"
done

echo "[$(date)] ========== 1/2  MLP  (perf_bench_pipe) =========="
PROJECT=perf_bench_pipe OUTDIR=sweep_results/bench_pipe RUN_TIMEOUT=3600  ./sweep.sh

echo "[$(date)] ========== 2/2  LeNet (perf_lenet_pipe) =========="
PROJECT=perf_lenet_pipe OUTDIR=sweep_results/lenet_pipe RUN_TIMEOUT=14400 ./sweep.sh

echo "[$(date)] ========== IKISI DE BITTI =========="
echo "Ozet:  sweep_results/bench_pipe/00_summary.csv  +  sweep_results/lenet_pipe/00_summary.csv"

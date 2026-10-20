#!/usr/bin/env bash
# run_sweep_int.sh -- the all-method K-sweep on the interacting multi-agent problem, ONE (method, K) per process,
# strictly sequential (no two runs share the machine; each process starts with an empty JIT cache).
# Results: results_sweep_int/<method>_K<K>.pkl (per-cycle details), logs: results_sweep_int/logs/<method>_K<K>.log.
# Existing result files are skipped, so the script can be restarted after an interruption.
#
#   PY=/path/to/python bash run_sweep_int.sh            (default: the HopfLaxMPC environment)
set -u
cd "$(dirname "$0")"
PY="${PY:-D:/software/anaconda/envs/HopfLaxMPC/python.exe}"
export PYTHONNOUSERSITE=1
mkdir -p results_sweep_int/logs
METHODS="Hopf_Lax_MPC PMP_Shooting DDP Vanilla_Hopf_Lax MPPI_12288 MPPI_20480"
echo "sweep started $(date '+%F %T')" >> results_sweep_int/logs/sweep.log
for K in 1 2 3 4 5 6 7 8 9 10; do
  for m in $METHODS; do
    if [ -f "results_sweep_int/${m}_K${K}.pkl" ]; then continue; fi
    echo "$(date '+%F %T') start ${m} K=${K}" >> results_sweep_int/logs/sweep.log
    "$PY" -u mpc_sweep_int.py --method "$m" --K "$K" > "results_sweep_int/logs/${m}_K${K}.log" 2>&1
    echo "$(date '+%F %T') end   ${m} K=${K} (exit $?): $(grep -h 'END' "results_sweep_int/logs/${m}_K${K}.log" | cut -c1-200)" >> results_sweep_int/logs/sweep.log
  done
done
echo "sweep finished $(date '+%F %T')" >> results_sweep_int/logs/sweep.log

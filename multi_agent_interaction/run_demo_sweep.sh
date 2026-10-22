#!/usr/bin/env bash
# run_demo_sweep.sh -- K-sweep of the two demo families (A_column, C_axis_side) with all seven methods,
# ONE (method, layout) per process, strictly sequential (timing runs must not share the machine).
# Results: results_demo_int/<family>_K<K>/<method>.pkl; logs: results_demo_int/logs/; progress: sweep.log.
# Finished runs are skipped, so the script can be restarted.  Usage: bash run_demo_sweep.sh [K list]
set -u
cd "$(dirname "$0")"
# Python: $PY if set, else the Windows HopfLaxMPC env if present, else `python` on PATH (activate HopfLaxMPC)
WINPY="D:/software/anaconda/envs/HopfLaxMPC/python.exe"
if [ -z "${PY:-}" ]; then if [ -x "$WINPY" ]; then PY="$WINPY"; else PY=python; fi; fi
export PYTHONNOUSERSITE=1
KS="${*:-2 3 4 5 6 7 8 9 10}"
METHODS="Hopf_Lax_MPC PMP_Shooting DDP_Box Vanilla_Hopf_Lax MPPI_12288 MPPI_20480 Collocation_IPOPT"
# GPU MPPI needs CuPy + CUDA (never on macOS): drop it where cupy does not import
if ! "$PY" -c "import cupy" > /dev/null 2>&1; then
  METHODS="${METHODS/ MPPI_12288 MPPI_20480/}"
  echo "cupy not importable: MPPI skipped"
fi
mkdir -p results_demo_int/logs
echo "demo sweep started $(date '+%F %T') K = $KS" >> results_demo_int/logs/sweep.log
# A_column for every K; C_axis_side only at K = 2 (it loses the saddle trap for K >= 3)
for K in $KS; do
  FAMS="A_column"; [ "$K" = "2" ] && FAMS="A_column C_axis_side"
  for f in $FAMS; do
    L="${f}_K${K}"
    for m in $METHODS; do
      if [ -f "results_demo_int/${L}/${m}.pkl" ]; then continue; fi
      echo "$(date '+%F %T') start ${L} ${m}" >> results_demo_int/logs/sweep.log
      "$PY" -u mpc_demo_int.py --layout "$L" --method "$m" > "results_demo_int/logs/${L}_${m}.log" 2>&1
      echo "$(date '+%F %T') end   ${L} ${m}: $(grep -h 'END\|ABORT' "results_demo_int/logs/${L}_${m}.log" | cut -c1-220)" >> results_demo_int/logs/sweep.log
    done
  done
done
echo "demo sweep finished $(date '+%F %T')" >> results_demo_int/logs/sweep.log

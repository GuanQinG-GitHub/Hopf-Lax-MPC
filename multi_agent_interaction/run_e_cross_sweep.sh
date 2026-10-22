#!/usr/bin/env bash
# run_e_cross_sweep.sh -- E_cross (crossing-saddle) sweep, ONE (method, K) per process, strictly sequential
# (timing runs must not share the machine).  Phases:
#   1  Hopf_Lax_MPC + PMP_Shooting at K = 6, 7, 8   (does the crossing saddle still hold?)
#   2  all seven methods at K = 2 .. 5              (repeat of the Mac screening, now with MPPI, timing-grade)
#   3  the other five methods at K = 6, 7, 8
# Results: results_demo_int/E_cross_K<K>/<method>.pkl; logs: results_demo_int/logs/; progress: e_cross.log
# (lines "PHASE n done" mark the phases).  Finished runs are skipped, so the script can be restarted.
set -u
cd "$(dirname "$0")"
WINPY="D:/software/anaconda/envs/HopfLaxMPC/python.exe"
if [ -z "${PY:-}" ]; then if [ -x "$WINPY" ]; then PY="$WINPY"; else PY=python; fi; fi
export PYTHONNOUSERSITE=1
ALL="Hopf_Lax_MPC PMP_Shooting DDP_Box Vanilla_Hopf_Lax MPPI_12288 MPPI_20480 Collocation_IPOPT"
REST="DDP_Box Vanilla_Hopf_Lax MPPI_12288 MPPI_20480 Collocation_IPOPT"
LOG=results_demo_int/logs/e_cross.log
mkdir -p results_demo_int/logs

run() {   # $1 = K, $2 = method
  L="E_cross_K$1"
  if [ -f "results_demo_int/${L}/$2.pkl" ]; then return; fi
  echo "$(date '+%F %T') start ${L} $2" >> "$LOG"
  "$PY" -u mpc_demo_int.py --layout "$L" --method "$2" > "results_demo_int/logs/${L}_$2.log" 2>&1
  echo "$(date '+%F %T') end   ${L} $2: $(grep -h 'END\|ABORT' "results_demo_int/logs/${L}_$2.log" | cut -c1-230)" >> "$LOG"
}

echo "$(date '+%F %T') E_cross sweep started" >> "$LOG"
for K in 6 7 8; do for m in Hopf_Lax_MPC PMP_Shooting; do run "$K" "$m"; done; done
echo "$(date '+%F %T') PHASE 1 done" >> "$LOG"
for K in 2 3 4 5; do for m in $ALL; do run "$K" "$m"; done; done
echo "$(date '+%F %T') PHASE 2 done" >> "$LOG"
for K in 6 7 8; do for m in $REST; do run "$K" "$m"; done; done
echo "$(date '+%F %T') PHASE 3 done" >> "$LOG"

#!/usr/bin/env bash
# Full N grid (5..200 step 5), ONE method per process, one at a time.  Before each method: wait until the
# machine is idle (total CPU < 10 % over a 30 s window) so no foreign load overlaps a timing run.
cd "$(dirname "$0")"
mkdir -p sweep_logs
PY=/d/software/anaconda/envs/HopfLaxMPC/python.exe
NS=$(seq -s, 5 5 200)
for m in ${METHODS:-Hopf_Lax_MPC PMP_Shooting DDP_Box MPPI MPPI_20480 Vanilla_Hopf_Lax Collocation_IPOPT}; do
  while true; do
    busy=$(powershell -NoProfile -Command "(Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 3 -MaxSamples 10).CounterSamples | Measure-Object CookedValue -Average | ForEach-Object { [int](\$_.Average) }")
    [ "${busy//[^0-9]/}" -lt 10 ] && break
    echo "$(date '+%F %T') waiting: machine busy (${busy}% CPU)"; sleep 120
  done
  echo "$(date '+%F %T') START $m"
  PYTHONNOUSERSITE=1 $PY mpc_testbed_N_sweep.py --scenario wall_saddle_bumps_rand --shared-stop 1e-6 \
      --Ns "$NS" --methods "$m" --redo --print-every 1000 > "sweep_logs/$m.log" 2>&1
  echo "$(date '+%F %T') END $m exit $?"
done
echo "$(date '+%F %T') ALL DONE"

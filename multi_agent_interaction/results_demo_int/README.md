# E_cross sweep results (K = 2-8, all seven methods)

Produced by `run_e_cross_sweep.sh` -> `mpc_demo_int.py` on the Windows CUDA workstation (`XL_Workstation`),
2026-10-21, **one (method, K) per process, strictly sequential** (timing-grade). Layout `E_cross_K<K>`
(mirror pairs crossing the axis at x = 5.2, odd K adds a leading axis agent), course
`wall_saddle_bumps_rand_sym` (moving obstacles active; = `wall_saddle_bumps_rand` with obs-1 on the axis,
which is now the only obs-1 position, so the course is today simply `wall_saddle_bumps_rand`), horizon N = 200, no per-cycle budget, shared stopping
rule eps = 1e-6, arrival radius 0.8 m, deadlock stop 3 s / 5 cm, 1500-cycle (30 s) cap, 30 min wall limit,
iteration caps 1000, MPPI = vanilla IT-MPC (lambda 0.01), pair penalty Rc 0.8 / Wc 500 / wc 0.1.

```
E_cross_K<K>/<method>.pkl     one closed-loop run (method in Hopf_Lax_MPC, PMP_Shooting, DDP_Box,
                              Vanilla_Hopf_Lax, MPPI_12288, MPPI_20480, Collocation_IPOPT)
e_cross_summary.txt           the summary table (ratios to Hopf_Lax_MPC at the same K)
logs/e_cross.log              sweep progress (start / end line per run); logs/E_cross_K<K>_<method>.log console output
```

Each pickle is a plain dict (numpy arrays, lists, dicts; load with `pickle.load(open(f, "rb"))`, numpy only):

| key | shape / type | meaning |
|---|---|---|
| `method`, `layout`, `K`, `N`, `label` | str / int | run identity |
| `starts`, `goals` | (K, 3) | start / goal position per agent |
| `X` | (K, 7, ncyc+1) | closed-loop state per agent at every cycle (x, y, z, heading, v, omega, vz) |
| `U` | (K, 3, ncyc) | applied control per cycle |
| `Pc` | (K, 7, ncyc) | applied costate (Hopf-Lax / PMP / Vanilla: exact; DDP_Box: adjoint costate; zeros for MPPI / IPOPT) -- re-roll predictions with `mpc_multi_int.build_ss_int(...).roll_pred` |
| `tlog` | (ncyc,) | simulated time of each cycle [s] (dt_apply = 0.02) |
| `tsolve` | (ncyc,) | **solver time per cycle [s]** (shared-stop evaluation excluded); use cycles 1: (cycle 0 = cold start) |
| `tcyc` | (ncyc,) | whole-cycle time [s] (reference window + solve + control) |
| `iters`, `status` | (ncyc,) / list | iterations and termination status per cycle (`shared-stop`, `converged`, `mu-max`, `ok`, ...) |
| `extra` | dict of (ncyc,) | method counters: HL `ndesc nesc nkick neig`; DDP `nbad` (Quu not PD) `nrej` (rejected steps) `n_eff mu a_min J`; Vanilla `nacc nrej t_hess lminH`; PMP `nres ntrial mu`; MPPI `ess hit whit`; IPOPT `ipopt_status f` |
| `dmin` | (ncyc,) | minimum pairwise distance over the fine steps of each cycle [m] |
| `Jreal`, `Jpen` | (K,) | true tracking+control / obstacle cost per agent over the whole run |
| `Jpair` | (K, K) | true pair-collision cost (upper triangle) |
| `J_full` | float | **trajectory cost** = sum Jreal + sum Jpen + sum Jpair (whole closed-loop run) |
| `t_reach` | (K,) | arrival time per agent [s] (NaN = not arrived) |
| `end` | str | `all-arrived`, `deadlock` (J_full = lower bound), `wall-limit` (partial mission: J_full not comparable), `maxcyc`, `aborted` |
| `ncyc`, `wall_s`, `t_compile` | int / float | cycles run, wall time of the run, compile time [s] |
| `meta` | dict | protocol (dt, dt_apply, scenario, reachtol, shared_stop_eps, lam_mppi, caps, deadlock window, ...) |
| `paths` | list of K dicts | reference path per agent (`pts` 3 x M, arclength `s`, heading `th`, `len`) |
| `obs`, `mobs` | dict | static / moving obstacle geometry (centres, half-axes) |
| `mobs_t` | (3, n_mobs, ncyc+1) | moving-obstacle centres at every cycle |
| `pillars` | (P, 2) | controller-only pillars (none in E_cross) |

Mean time per cycle = `tsolve[1:].mean()`; arrivals = `(~isnan(t_reach)).sum()`.

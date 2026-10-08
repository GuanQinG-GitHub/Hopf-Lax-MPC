# Hopf-Lax-MPC

Closed-loop and open-loop comparison of a certified Hopf-Lax quasi-Newton MPC solver ("M1_v2",
displayed as Hopf-Lax-MPC) against PMP shooting, DDP, direct collocation (CasADi/IPOPT) and GPU MPPI,
on a corridor scenario with a symmetric obstacle saddle.

This repository is the self-contained, minimal set of scripts that reproduces the two headline
results. It was distilled on 2026-09-22 from a longer design-iteration folder; every file here was
verified to run from this folder alone.

| result | data | produced by |
|---|---|---|
| `figs/anim_all_methods.mp4` -- closed-loop 3-D animation, 6 runners | `results_testbed_v10.pkl` (not in repo, 44 MB) | `mpc_testbed.py` -> `make_animations.py` |
| `figs/complexity_vs_N_maxiter.png` -- open-loop complexity vs horizon | `results_complexity.pkl` (in repo, 2.3 MB) | `complexity_vs_horizon.py` -> `plot_complexity_maxiter.py` |

![complexity vs horizon](figs/complexity_vs_N_maxiter.png)

## Repository structure

| file | role |
|---|---|
| `mpc_core.py` | plant, cost, dynamics, reference path (JAX); verified against the MATLAB reference |
| `mpc_solvers.py` | all solvers: the M1_v2 method (`chlqn_solve`, paper Algorithm 1 -- certified Hopf-Lax quasi-Newton) and the baselines it is compared against -- PMP (Levenberg-Marquardt shooting), iLQR, DDP, collocation (CasADi/IPOPT) |
| `mpc_mppi.py` | GPU MPPI (CuPy) |
| `mpc_tuned_params.py` | **all scenario tuning** in one module: `apply_wall_course(P)` (the v6 corridor course: moving wall that stops, lane blocker, closing door, shoulder blocker) and `apply(P)` (the v10 unified-route course: wall course plus a downstream fork with an arriving hot companion, re-calibrated horizons and MPPI pins). Flattened from the former five-module chain `mpc_tuned_params -> _v5 -> _v6 -> _v7 -> _v10`; verified field-for-field identical to what that chain produced |
| `mpc_testbed.py` | closed-loop comparison on the unified-route course under a hard 20 ms/cycle budget (formerly `mpc_testbed_v10.py`). Holds the per-method closed-loop runners and the piecewise obstacle motion used by the complexity study; the solvers themselves live in `mpc_solvers.py` |
| `make_animations.py` | 3-D animation renderer; reads a testbed pickle |
| `complexity_vs_horizon.py` | open-loop solve-trace sweep over horizon N at the obs-2 saddle from a symmetric initial guess; merges into `results_complexity.pkl` |
| `plot_complexity_maxiter.py` | renders `figs/complexity_vs_N_maxiter.png` from `results_complexity.pkl` |
| `COMPLEXITY_FIGURE.md` | full protocol, metric definitions, and takeaways of the complexity figure |
| `results_complexity.pkl` | recorded solve traces and structural per-iteration benchmarks behind the figure (Jul 28 2026) |
| `figs/` | the two published results |

Not in the repository: `results_testbed_v10.pkl`, the 44 MB recorded closed-loop run behind the
animation. Regenerate it with `mpc_testbed.py` (see below) or ask the authors for the file.

## Environment

All dependencies are pinned in `environment.yml` (conda) and `requirements.txt` (pip): Python 3.13,
numpy, scipy, JAX (CPU, shooting methods and DDP), CasADi with bundled IPOPT (collocation),
matplotlib, imageio-ffmpeg (mp4 rendering) and CuPy on CUDA 12.x (GPU MPPI). These are the exact
versions that produced the published results.

Both files work unchanged on **Windows, Linux and macOS**. Only the GPU MPPI baseline is
platform-dependent, and it is handled automatically:

| platform | GPU MPPI (CuPy) | how to run |
|---|---|---|
| Windows / Linux **with** an NVIDIA GPU (CUDA 12.x driver) | installed | all commands as written |
| Windows / Linux **without** an NVIDIA GPU | delete the `cupy-cuda12x` line before creating the environment | add `--no-mppi` |
| macOS (Intel or Apple Silicon) | **skipped automatically** -- no CUDA on Apple hardware, and CuPy publishes no macOS wheel | add `--no-mppi` |

The `cupy-cuda12x` entry carries the environment marker `; platform_system != "Darwin"`, so pip
installs it on Windows/Linux and silently skips it on macOS. macOS users therefore do **not** need to
edit any file -- just create the environment as below. Everything except MPPI is pure CPU and runs
identically on all three platforms.

Create and activate the environment in one go (same on every platform):

```bash
git clone https://github.com/GuanQinG-GitHub/Hopf-Lax-MPC.git
cd Hopf-Lax-MPC
conda env create -f environment.yml
conda activate HopfLaxMPC
```

Check the installation.  With a CUDA GPU (Windows/Linux):

```bash
python -c "import jax, casadi, cupy, imageio_ffmpeg; print('ok', cupy.cuda.runtime.getDeviceCount(), 'GPU(s)')"
```

On macOS, or on any machine without a GPU, leave `cupy` out of the check -- it is not installed
there and importing it will fail:

```bash
python -c "import jax, casadi, imageio_ffmpeg; print('ok', jax.devices())"
```

Pip users can instead run `pip install -r requirements.txt` inside any Python 3.13 environment; the
same marker applies, so that one command is also correct on all three platforms.

Troubleshooting: if the check above fails with `No module named 'cupy'` (or another listed package)
on a machine that should have it, pip found an existing copy in your per-user site-packages and
skipped installing it into the new environment. Set `PYTHONNOUSERSITE=1` before creating the
environment (PowerShell: `$env:PYTHONNOUSERSITE = 1`; bash/zsh: `export PYTHONNOUSERSITE=1`), or
install the missing package into the activated environment with
`pip install --ignore-installed "cupy-cuda12x[ctk]==14.1.1"`.

A second failure mode, seen on macOS: `conda env create` reports that the whole `pip` step failed and
the environment ends up with *no* packages at all. That is the old unconditional `cupy-cuda12x` pin --
conda hands the entire `pip:` list to pip as one install, so a single unresolvable wheel aborts all of
them. The marker above fixes it; if you hit it with an older checkout, just pull and re-create.

Run every command **from the repository root** (all paths are working-directory relative). The
PowerShell snippets below use `$py` for the interpreter; with the environment activated it is simply:

```powershell
$py = "python"
```

On macOS/Linux, drop the `& $py` prefix and call `python` directly -- e.g.
`python mpc_testbed.py --no-mppi`.

## 1. Closed-loop animation

Run the closed loop (writes `results_testbed_v10.pkl`), then render:

```powershell
& $py mpc_testbed.py            # PMP, DDP, collocation, M1_v2, MPPI x2; 20 ms/cycle budget
& $py mpc_testbed.py --smoke    # 60-cycle pipeline check
& $py mpc_testbed.py --no-mppi  # without a CUDA GPU (always on macOS)

& $py make_animations.py results_testbed_v10.pkl        # combined video + one clip per runner -> figs/
& $py make_animations.py results_testbed_v10.pkl c      # combined video only
```

The closed loop runs under a wall-clock budget, so a re-run reproduces the qualitative outcome
(PMP trapped at the wall, DDP trapped at the fork, Hopf-Lax-MPC reaches the goal, MPPI detours) but
not the recorded run bit-for-bit. The committed video was rendered from the recorded run with an
earlier colour style of `make_animations.py`; trajectories are identical.

## 2. Complexity figure

From the committed data (seconds; pixel-identical to the committed figure):

```powershell
& $py plot_complexity_maxiter.py                        # -> figs/complexity_vs_N_maxiter.png
```

Full regeneration of the data (hours; never run two timing jobs at once; the pickle is merged
incrementally, so single methods can be redone):

```powershell
$Ns = (6..30 | ForEach-Object { $_ * 10 }) -join ","   # 60,70,...,300
foreach ($m in "m1v2","pmp","ddp","coll","mppi8192","mppi12288","mppi16384","mppi20480") {
  & $py complexity_vs_horizon.py --methods $m --Ns $Ns
}
& $py complexity_vs_horizon.py --rescore      # REQUIRED after any m1v2/pmp sweep (common metric)

& $py complexity_vs_horizon.py --iterbench    # structural per-iteration benchmarks (m1v2/pmp/ddp)
& $py plot_complexity_maxiter.py
```

On macOS, run only the CPU methods -- `m1v2`, `pmp`, `ddp`, `coll`; the four `mppi*` sweeps need the
CUDA GPU. The committed `results_complexity.pkl` already contains the MPPI traces, so
`plot_complexity_maxiter.py` still reproduces the full figure.

Absolute ms values are machine-specific; orderings and slopes are the portable content. The initial
state `x0` is cached inside `results_complexity.pkl`; only if the pickle is deleted does the script
try to re-derive it by replaying a v6-course closed loop (it then looks for `results_testbed_v6.pkl`,
which is not part of this repository).

See `COMPLEXITY_FIGURE.md` for the protocol, the common optimality metric, and the interpretation
of the figure.

## 3. Closed loop without a real-time budget vs horizon

![no-budget horizon sweep](figs/N_sweep_nobudget.png)

The closed-loop testbed of section 1 runs every method under a hard 20 ms/cycle budget at one tuned
horizon each. This experiment removes the budget and sweeps the horizon instead: same scenario,
obstacles, warm starts and stop-at-goal protocol, but every solver runs to convergence or to an
iteration cap of 1000 (`lm_maxit`, `ilqr_Kmax`, IPOPT `max_iter`), and N is swept densely from
10 to 300 in steps of 5 (59 horizons) for all six runners (the Newton baseline from 50 to 300). MPPI keeps its
published protocol of one sampling update per cycle (there is no iteration loop to unbudget), at
K = 12288 and K = 20480.

| file | role |
|---|---|
| `mpc_testbed_N_sweep.py` | sweep driver with instrumented copies of the five testbed runners (the originals discard iteration counts); records per cycle the solve time, iteration count, termination status, final residual and method-specific counters; merges into `results_N_sweep.pkl`, saved after every (method, N) run so it can be interrupted and resumed |
| `plot_N_sweep.py` | the 2 x 2 figure above and a per-(method, N) table |
| `mpc_testbed_newton.py` | the Newton baseline in the **budgeted** testbed (hard 20 ms cut, viz logging) at N = 55, appended to the recorded six-runner pickle for the seven-runner animation `figs/with_newton/anim_all_methods.mp4` |
| `mpc_newton.py` | the **Newton (exact Hessian)** baseline: damped Newton on the Hopf-Lax objective with `jax.hessian`, Levenberg damping (H + mu I), batched Armijo line search, gradient-only stopping test; no curvature certificate, no saddle escape. Run with `--methods newton` (not in the default list); `--newton-tol` (1e-4 used), `--hess-mode` |
| `results_N_sweep_table.txt` | the per-(method, N) table printed by `plot_N_sweep.py` for the recorded sweep (405 runs, Oct 7-8 2026: 36 sparse runs in 1 h 10 min, 210 dense runs in 4 h 40 min, 31 Newton runs N = 50..200 in 32 min, 128 runs for N = 205..300 in 5 h 30 min) |
| `results_N_sweep.pkl` | the recorded per-cycle data behind the figures (55 MB, **not in the repository**; regenerate with the commands below, or ask the authors) |

```powershell
& $py mpc_testbed_N_sweep.py --selftest       # counted solver copies == originals, bitwise
& $py mpc_testbed_N_sweep.py --smoke           # 60-cycle pipeline check -> results_N_sweep_smoke.pkl
& $py mpc_testbed_N_sweep.py                   # sparse sweep N=50..300 (resumable; --methods, --Ns, --redo, --cap)
$Ns = ((2..60 | ForEach-Object { $_ * 5 }) -join ",")                       # 10,15,...,300
& $py mpc_testbed_N_sweep.py --methods pmp,m1v2,mppi12288,mppi20480,coll,ddp --Ns $Ns   # dense sweep, ~10 h
& $py mpc_testbed_N_sweep.py --methods newton --Ns $Ns --newton-tol 1e-4               # Newton baseline, ~2 h
& $py plot_N_sweep.py                          # -> figs/N_sweep_*.png + table (results_N_sweep_table.txt)
```

Statuses per cycle: `converged`, `acceptable` (IPOPT), `cap` (hit 1000 iterations while still
accepting steps), `stalled` (no accepted step for 50+ iterations, i.e. a damping crawl into the cap,
or a failed line search), `esc-stalled` (Hopf-Lax-MPC escape without an improving step), `diverged`,
`timeout` (only with `--cycle-timeout`). Timing statistics exclude cycle 1 and recompile-tagged cycles.

What the recorded sweep shows (panel by panel):

- **Cost.** Three regimes. Up to N ≈ 150 (0.75 s, about 1 m of lookahead) every method is trapped at
  the wall and the cost falls monotonically with N for all of them, identically for the four
  deterministic solvers. At N = 10 and 15 the horizon is so short that the controllers drive straight
  through the obstacles along the reference and "reach" the goal with an obstacle cost near 1000.
  From N = 155 the methods start to get around the wall, and between 155 and 200 the outcome flips
  between reached and trapped from one horizon to the next: Hopf-Lax-MPC reaches at 155, 170 to 180,
  190, 200 and above; collocation at 155 to 180, then is trapped at 185 to 200 and reaches again at
  250 and 300; DDP at 105 and 115 to 180, then 200 and 250; MPPI at 155 to 250 but with an order of
  magnitude more obstacle cost. PMP is trapped at every horizon. The flips happen for collocation
  too, which converges in every cycle, so in this band they are a property of the receding-horizon
  problem (frozen-obstacle prediction, the fork and the arriving companion, a non-convex route
  choice), not of a solver. Best closed-loop cost: Hopf-Lax-MPC 2.0 at N = 275, collocation 3.9
  at N = 295, DDP 5.1 at N = 265, PMP 8.6, MPPI 18 to 33.
- **N = 205 to 300.** Hopf-Lax-MPC reaches the goal at 19 of the 20 horizons (trapped only at 260)
  and holds the lowest cost of all methods, 2.0 to 4.1 at most horizons, but at five horizons (235,
  255, 270, 285, 295) it reaches with a cost of 9 to 28: those runs contain the stalled-cycle
  episodes at the fork diagnosed below (up to 8 percent of cycles), which push it onto the wide
  route. Collocation reaches from N = 250 on with a flat 3.9 to 4.1. DDP reaches at 14 of 20
  horizons with 5 to 10 and is trapped at 300. PMP stays trapped except at N = 295. MPPI reaches at
  18 of 20 but with costs of 40 to 100, and collapses at N = 295 to 300 (K = 20480 at 295: 5476).
  The Newton baseline reaches at 255, 260, 270 and 300 (3.5 to 5.6) but blows up at N = 275 to 295
  (cost 1200 to 1700, cycle time up to 0.5 s): the exact Hessian becomes strongly indefinite
  (smallest eigenvalue -2.4) and the damping is raised more than 6000 times per run, the
  sensitivity of inverting the exact Hessian at long horizons in its purest form.
- **Time.** Without a budget, Hopf-Lax-MPC and PMP stay at or below the 20 ms line on average up to
  N = 300 (1 to 25 ms, growing with N); collocation costs 5 to 260 ms per cycle; DDP costs 100 to
  900 ms per cycle on average with worst cycles above 2 s, except at N = 10.
- **Iterations.** Hopf-Lax-MPC and PMP need 1 to 5 iterations per cycle on average over the whole
  grid, collocation 6 to 18 (up to 40 in the band where it threads the fork). DDP averages 130 to
  700 and hits the cap at every horizon above 10.
- **Non-convergence.** DDP fails to converge in 10 to 67 percent of cycles (damping crawls at the
  stiff barriers; worst below N = 90). PMP has one isolated spike (63 percent at N = 125, where it
  first starts to feel the wall's saddle). Hopf-Lax-MPC has at most 2 percent of stalled cycles up to
  N = 250 and 5 percent at N = 300; collocation and the Newton baseline converge in every cycle.

**Newton with the exact Hessian** (`mpc_newton.py`, swept for N = 50 to 200). This baseline minimises
the same Hopf-Lax objective as Hopf-Lax-MPC but with the exact second derivative from `jax.hessian`
instead of the surrogate, damped Newton steps and a line search, and stops on the gradient alone.
Its correctness was checked before the sweep: the exact Hessian matches central finite differences of
the gradient to 1e-11 and agrees with the surrogate to 3e-9 at certified points; from Hopf-Lax-MPC's
warm start it converges in at most one iteration to the same costate; and in closed loop at N = 100
its controls match Hopf-Lax-MPC and PMP to 1e-3 until the wall. A looser tolerance (1e-3) was tried
first and rejected: at long horizons the small curvature turns a 1e-3 gradient into a 0.05 control
error from t = 0.5 s on and flips the N = 155 outcome, so the sweep uses the same 1e-4 certificate as
Hopf-Lax-MPC. What the sweep shows:

- **It behaves like a local minimiser of the same problem, i.e. like collocation.** Trapped at the
  wall for N <= 150 with costs identical to Hopf-Lax-MPC to three digits; reaches the goal at
  N = 155 to 175 (collocation: 155 to 180); trapped at the fork at N = 180 to 200 with the same cost
  as collocation (5.36 to 5.39). Hopf-Lax-MPC differs exactly where its escape fires: it reaches at
  N = 190 and 200 (2.9 to 3.1) where Newton and collocation are stuck at the fork.
- **The exact Hessian costs 3 to 4 times more per iteration**: 6 to 20 ms per iteration versus
  1.7 to 4.3 ms for Hopf-Lax-MPC at the same N, with the Hessian itself 70 percent of the solve time,
  and 2 to 6 iterations per cycle instead of 2 to 3. Per cycle that is 11 to 120 ms, above the 20 ms
  budget from N = 70 on.
- **Sensitivity shows up as iteration spikes, not failures**: 100 percent converged at every horizon,
  but single cycles need up to 503 iterations (N = 155) and the damping has to be raised 1400 to 4500
  times per run at N >= 155 because the exact Hessian is indefinite near the wall.
- **Newton in the 20 ms budgeted testbed (`mpc_testbed_newton.py`).** The horizon was chosen with
  the same rule as Hopf-Lax-MPC's N = 220: the largest horizon whose p99 cycle time stays inside the
  budget. Budgeted Newton runs at N = 40 to 200 (`--cycle-timeout 0.02`): the cut can only act between
  iterations, and one exact-Hessian iteration already costs 5 ms at N = 50 and 16 ms at N = 170, so the
  mean cycle time crosses 20 ms at N = 60 to 70 and reaches 27 to 35 ms (1.4 to 1.75 times the budget)
  at the horizons where Newton would reach the goal (N >= 170, with one iteration per cycle). Within
  the compliant range the cost falls monotonically with N, so the best compliant horizon is the largest
  one: **N = 55** (mean 16.2 ms, p99 20.9 ms, 0.2 percent of cycles cut, 99.8 percent converged; the
  neighbours 60 and 65 already have 65 percent cut cycles and p99 of 25 to 29 ms). At N = 55 the
  baseline is trapped at the wall like every method with that lookahead (J = 59.4 versus
  Hopf-Lax-MPC's 2.6 at its own N = 220). The seven-runner animation is
  `figs/with_newton/anim_all_methods.mp4`:

  | N (budgeted) | 50 | 55 | 60 | 65 | 70 | 100 | 150 | 170 | 180 | 200 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | mean cycle ms | 11.3 | 16.2 | 19.8 | 21.9 | 23.9 | 21.0 | 30.3 | 33.0 | 35.1 | 27.1 |
  | p99 cycle ms | 23.0 | 20.9 | 24.6 | 28.7 | 28.3 | 30.2 | 33.9 | 36.2 | 39.0 | 40.5 |
  | cycles cut | 6% | 0.2% | 65% | 67% | 68% | 94% | 97% | 99% | 99% | 100% |
  | J_total | 107 | 59.4 | 49.8 | 39.9 | 34.3 | 23.8 | 15.1 | 5.10 | 4.65 | 4.78 |
  | outcome | trapped | trapped | trapped | trapped | trapped | trapped | trapped | reached | reached | reached |

- **One failure mode of a gradient-only stop**: at N = 50 the trapped Newton iterate sits on a
  plateau created by control saturation (the clipped control makes the objective flat along the
  saturated costate direction), where the gradient is below tolerance while the PMP residual is 30,
  the objective is higher than at the PMP root (4.72 vs 4.56) and the control has the wrong sign, so
  it pushes 6 cm deeper into the wall and pays 40 percent more cost (107 vs 76). The residual-based
  descent of Hopf-Lax-MPC / PMP is not fooled by that plateau.

The cycle time above is the whole solve of a cycle, i.e. all of its iterations. The per-iteration
cost is the second figure, `figs/N_sweep_periter.png` (also written by `plot_N_sweep.py`): for every
cycle the solver time divided by its iteration count, then the mean (solid) and max (dashed) over the
cycles of a run. Cycles with zero iterations (warm start already converged) are excluded; for MPPI it
equals the cycle time since it does one update per cycle. The dotted lines are the open-loop
structural worst-case per-iteration benchmark of the complexity study (`results_complexity.pkl`,
`iterbench`), overlaid as a cross-check.

![per-iteration solve time](figs/N_sweep_periter.png)

- **Per-iteration cost.** DDP has the cheapest iteration (0.7 to 2.7 ms), then PMP (1.3 to 4.6 ms)
  and Hopf-Lax-MPC (1.7 to 5.8 ms mean; its worst cycle-average iteration, the kick-firing escape
  iteration, reaches 19 ms at N = 300). Collocation's IPOPT iteration is 4 to 34 ms, and one MPPI
  update costs 3.5 to 17 ms (K = 12288) or 6 to 32 ms (K = 20480). All grow close to linearly in N.
  The closed-loop means agree with the open-loop benchmark: PMP and DDP sit on their benchmark
  lines, and Hopf-Lax-MPC's mean lies between PMP's line and its own worst-case kick iteration.

Mean-only and max-only versions of the same quantity, on linear axes, are
`figs/N_sweep_periter_mean.png` and `figs/N_sweep_periter_max.png`.

Why this figure differs from the left panel of `figs/complexity_vs_N_maxiter.png` (section 2):

1. **Different statistic.** The complexity panel is the cost of the *maximal-content* iteration
   (structural worst case: for Hopf-Lax-MPC the kick-firing escape iteration with two shoots, two
   eigen-decompositions and two batched line searches; for collocation the most expensive IPOPT
   iteration of a solve). This figure is the *mean over all iterations* of a closed-loop run, which
   for Hopf-Lax-MPC are mostly plain descent iterations (one shoot, one line search). Hence 9.7 ms
   worst case vs 5.8 ms mean at N = 300. PMP and DDP have fixed iteration content, so the two
   figures agree for them (4.6 / 4.6 ms and 2.8 / 2.7 ms).
2. **Different scene.** The complexity study solves at the obs-2 saddle with the wall as the only
   moving body; the closed loop runs the full course with 7 moving bodies (wall, blocker, door,
   shoulder, two fork posts, arriving companion). Every MPPI sample step and every IPOPT function,
   gradient and Hessian evaluation prices all bodies, so those iterations get dearer. Measured at
   N = 300 on this machine, same code, only the scene swapped: one MPPI update 10.0 -> 17.1 ms
   (K = 12288) and 18.3 -> 32.3 ms (K = 20480); one IPOPT iteration 14.7 -> 33.1 ms. The JAX
   shooting rollouts are dominated by the dynamics sensitivities, so PMP / DDP / Hopf-Lax-MPC
   barely move.
3. **Different accounting for collocation.** Here `ms/it` is the whole CasADi/IPOPT call divided by
   its iteration count, and warm-started closed-loop solves take only 5 to 11 iterations, so the
   per-call fixed overhead (marshalling, initialisation, solution extraction) is spread over few
   iterations. The complexity panel timestamps consecutive IPOPT iterations through the callback
   and discards the first interval, so it excludes that overhead.
4. **Max vs worst case.** The dashed max in `N_sweep_periter.png` is the largest *cycle average*,
   which can exceed the structural worst case (Hopf-Lax-MPC 19 ms vs 9.7 ms at N = 300): a cycle
   average includes sequential line-search rollouts (up to 8 per descent iteration), kick re-shoots
   and OS timing noise, whereas the benchmark counts one accepted trial of prewarmed content.

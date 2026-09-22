# Complexity-vs-Horizon Study — figure documentation

**Figure (this folder):** `figs/complexity_vs_N_maxiter.png` (two panels: per-iteration cost, converged optimality; collocation/MPPI per-iteration cost uses the max observed iteration — see §2.4).  The median-iteration variant `complexity_vs_N.png` and its script `plot_complexity.py` live in the parent iteration folder, not here.
**Drawn series (2026-07-31 restyle):** the MPPI curves shown are only the two closed-loop
baselines K = 12288 and K = 20480 (the K = 8192 / 16384 sweeps remain in the pkl and still enter
J\*(N)); display names follow the closed-loop figures (Hopf–Lax-MPC, MPPI-(K=…)); the in-panel
gray notes were removed (their content lives in §2.4 / §2.1 below).
**Data:** `results_complexity.pkl`

---

## 1. Problem setup

**One open-loop optimal-control solve, repeated across horizon lengths N, at a single physically
meaningful state.** This isolates *solver* complexity from closed-loop effects (no warm starts, no
receding horizon, no disturbance rejection).

- **Plant / cost:** the v6 corridor scenario (`mpc_tuned_params.py`, `apply_wall_course`, model built by
  `mpc_core.py` / `mpc_solvers.py`), reduced to **obs-2 only, held static** at its stop position
  `(x, y, z) = (10.4, 0, 0.5)` — a wall centered on the reference lane. All other obstacles are
  removed so the solve has exactly one saddle: pass the wall on +y or −y.
- **Initial state `x0`:** replayed from the v6 closed loop — the state at cycle 270 (t = 5.40 s),
  the cycle where M1_v2 fires its kick. Numerically
  `x0 ≈ [8.0868, 6.6e-4, 0.4999, 2.1e-4, 1.456, 1.8e-3, 5.4e-4]`,
  i.e. **1.513 m before the wall shell, moving at 1.46 m/s, laterally symmetric to ~1e-3**.
  This is the discriminating configuration: a symmetric approach to a saddle, where methods
  without an escape/branch mechanism commit to whatever the numerics happen to break toward.
  Cached in the pkl (`D["x0"]`) so every method and every re-run uses the identical state.
- **Horizon grid:** N = 60, 70, …, 300 (fine grid, step 10). N > 300 is excluded from the figure
  (`NMAX = 300`): beyond it the shooting problem enters its sensitivity regime, reported
  separately from this figure.
- **Initialization (symmetric, per family):** shooting methods (M1_v2, PMP) start from the zero
  costate; DDP from the zero control sequence; collocation from the straight-line/reference
  initial guess; MPPI from the zero nominal sequence. No method receives side information about
  the saddle.

## 2. How the figure is generated

### 2.1 Common optimality metric (protocol fix "one metric for all")

Every family natively minimizes a slightly different functional (Hopf-Lax value, DDP rollout cost,
collocation NLP objective, MPPI sampled cost). For scoring, **every iterate of every method is
evaluated under the same functional**: `rollout_cost` — the ZOH rollout of the iterate's control
sequence on the true discrete dynamics. For the costate-space shooting methods the control sequence
is extracted per iterate via `make_uextract` (u_k = ustar(x_k, p_k) along the canonical rollout);
this rescoring is a post-pass (`--rescore`) that rewrites the trace J values in the pkl.

- **J\*(N)** = best (lowest) final J across all *converged* runs at that N (no separate
  global-optimum computation; a stalled/diverged final never defines the optimum).
- **Optimality band:** ε = 2 %. Measured justification: the shooting methods' certificate-stop +
  ZOH-extraction offset under the common metric reaches +1.35 % at N = 250–290 *with*
  |res| < 1e-10 (fully converged on their own functional), so a 1 % band sits below the
  measurement floor and would mislabel converged solves as DNF.

### 2.2 Solve traces

Each method is run to termination at every N with a per-iteration trace `(t, J)`; `REPS = 3`
repetitions are recorded for wall-clock stability (the iterate sequences themselves are
deterministic — for MPPI see §2.3). Termination: M1_v2 by its certificate; PMP by its LM
convergence test; DDP by convergence or a 25-iteration cap (its μ-rejection crawl otherwise never
terminates); collocation by IPOPT; MPPI runs a fixed budget (below).

- **M1_v2** (`chlqn`): gated eigen read η_s = 0.29, ridge threshold 0.1, kick α = 0.3, factored
  Newton descent, **batched rollouts** (the kick ±α comparison and the escape line search are each
  one vmapped objective dispatch — same implementation as the closed-loop testbed
  `mpc_testbed.py`).
- **PMP**: single-shooting Levenberg–Marquardt on the boundary residual.
- **DDP**: full backward pass + batched-α forward line search.
- **Collocation**: CasADi/IPOPT transcription; per-iteration objective recorded via an IPOPT
  `iteration_callback` (a strong Python reference to the callback must be kept alive — GC pitfall).
- **MPPI**: **iterated** protocol — fresh controller (seed 0), no shift, 150 successive sampling
  updates from the zero nominal; each update's J recorded; the method is credited with its
  **best-of-150** value. K ∈ {8192, 12288, 16384, 20480}; λ is bisected per update to an ESS
  target.

### 2.3 MPPI caveat (single-realization values)

The recorded MPPI numbers are **single seed-0 realizations** (the 3 reps re-run the same seed —
they exist for wall-clock statistics). A dedicated seed study (8 seeds × 4 K × N ∈ {200, 240, 300},
script in §4) established:

- seed-0 values reproduce exactly (implementation is correct; zero non-finite costs);
- beyond the collapse (N ≳ 180) the per-K values are draws from heavy-tailed, largely overlapping
  distributions — **cross-K ordering there must not be read** (e.g. K=8192's standout J=13.4 at
  N=300 is a 1-in-8 lucky draw; its other seeds land at 193–210 like every other K);
- at N = 240 there is one *real* K effect, and it is adverse: K=20480 reached its ESS target on
  8/8 seeds and confidently averaged into the wrong basin (J ≈ 35), while smaller K degenerated to
  ESS = 1 (random shooting) and lucked into the better basin on ~half the seeds. More samples →
  more faithful mean-field update → worse outcome at this multimodal geometry.

### 2.4 Left panel — per-iteration cost (worst-case content)

Reported quantity: the cost of the **maximal-content iteration**, as a **structural benchmark**
(median of 30 prewarmed executions of that content; never a max over observed wall-clock, which
would measure OS noise). This is the big-O-consistent convention:

| method | benchmarked content |
|---|---|
| M1_v2 | the kick-firing escape iteration: `S.all` → 7×7 eigh → batch-2 kick comparison → `S.all` → eigh → floored step → batch-7 escape line search |
| PMP | LM iteration: `S.Jr` + `S.res` + normal-equation solve + 1 accepted trial |
| DDP | backward pass + batched-α forward pass (fixed content — the line search is vectorized into the forward call) |
| Collocation | trace-based (unfixed IPOPT iteration content): **median** inter-iteration time in the main figure; the `_maxiter` variant uses the **max per solve (median across reps)**, which is near-monotone in N and is the honest "really worst iteration" reading |
| MPPI | one full update (kernel + λ-bisection + nominal update; fixed content), trace-based |

The first record's dt is always excluded (per-solve setup), and all JIT compilation happens outside
timed regions (prewarming, incl. both batch shapes for M1_v2). Absolute ms values are
machine-specific (MPPI kernel on GPU via CuPy; shooting methods on CPU via JAX); orderings and
slopes are the portable content.

### 2.5 Right panel — converged optimality

`J*(N) / J_final ∈ (0, 1]`, 1 = optimal. This is the bounded equivalent of the signed relative gap
1 − (J−J\*)/J\* (which goes negative past 100 % gap and would put DDP's stalls at −2000).
J_final = the terminal iterate for the deterministic methods, best-of-budget for MPPI. Dashed line:
the ε-band boundary 1/(1+ε). M1_v2 is drawn first with a wide line / large markers so it remains
visible as a rim beneath collocation where both sit at 1.0.

## 3. Takeaways

1. **Per-iteration cost ranks M1_v2 above PMP and DDP** (9.7 vs 4.6 vs 2.8 ms at N=300; all three
   monotone, near-linear in N) — **but the cheap iterations do not buy optimality**:
   - **PMP** converges in *fewer* iterations than M1_v2 (5–14 vs 9–39 at N ≥ 160) yet lands on the
     wrong branch of the saddle from N = 160 onward: optimality slides from 1.0 to 0.08. Fast
     convergence to the wrong answer.
   - **DDP** is optimal (and genuinely cheapest) up to N = 180, then falls off a cliff: from
     N = 190 it never converges (μ-rejection crawl, 25-iteration cap, optimality ≈ 0.1 → 0.005).
   - Both are *fine below the branch crossover N\* ≈ 150*, where the problem is effectively
     unimodal. The degradation story is strictly post-crossover.
2. **M1_v2 and collocation are the only methods that stay at ≈ 1.0 over the whole grid.**
   Collocation reaches gap 0.00 % at every N (it usually defines J\*); M1_v2 sits within
   +0.0–1.35 % (inside the 2 % band; the offset is its certificate-stop + ZOH-extraction floor).
   The correct claim is **comparable optimality at a fraction of the compute** (collocation is
   4–8× slower to the band; in the `_maxiter` variant its worst iteration is the most expensive
   of all methods at large N, ~23 ms) — not "better optimality".
3. **MPPI is budget-insensitive in its failure**: four budgets spanning 2.5× (8192 → 20480) show
   the same profile — a 0.5–23 % noise floor even in the good regime (optimality 0.85–0.99 at
   small N), then collapse beyond N ≈ 180 to optimality ≲ 0.2. Adding samples does not fix it (and
   at N = 240 measurably *hurts*, §2.3), while its per-update cost is the top line of the left
   panel (K=20480: 3.8 → 19 ms). Its ESS collapse (median 1 of 20480 samples) marks the
   dimensionality wall of sampling in 3N-dimensional noise space.
4. **Worst-case per-iteration cost is the fair convention**: trace-averaged estimates drift with N
   for unfixed-content solvers (DDP's estimator-composition "dip", collocation's non-monotone
   median). The structural worst-case benchmarks are monotone by construction; collocation's max
   variant shows the remaining wiggles (N = 140, 180) are genuine content variation (extra inertia
   corrections / refactorizations at some horizons), not noise.

## 4. Reproduction

All commands from `July23/`, using the JAX/CuPy environment
(`D:\software\anaconda\envs\mpcpy\python.exe`; plotting scripts also run under any Python with
numpy + matplotlib). **Never run two timing jobs concurrently.**

```powershell
$py = "D:\software\anaconda\envs\mpcpy\python.exe"
$Ns = (6..30 | ForEach-Object { $_ * 10 }) -join ","   # 60,70,...,300

# 1) solve sweeps (order free; each merges into results_complexity.pkl)
& $py complexity_vs_horizon.py --methods m1v2      --Ns $Ns
& $py complexity_vs_horizon.py --methods pmp       --Ns $Ns
& $py complexity_vs_horizon.py --methods ddp       --Ns $Ns
& $py complexity_vs_horizon.py --methods coll      --Ns $Ns
& $py complexity_vs_horizon.py --methods mppi8192  --Ns $Ns
& $py complexity_vs_horizon.py --methods mppi12288 --Ns $Ns
& $py complexity_vs_horizon.py --methods mppi16384 --Ns $Ns
& $py complexity_vs_horizon.py --methods mppi20480 --Ns $Ns

# 2) rescore shooting traces onto the common rollout_cost metric (REQUIRED after any
#    m1v2/pmp re-sweep -- forgetting it leaves raw Hopf-Lax J in the traces and fakes DNFs)
& $py complexity_vs_horizon.py --rescore

# 3) structural per-iteration benchmarks for m1v2/pmp/ddp
& $py complexity_vs_horizon.py --iterbench

# 4) figures
# (plot_complexity.py -- median-iteration variant -> figs/complexity_vs_N.png -- is in the parent folder)
& $py plot_complexity_maxiter.py    # -> figs/complexity_vs_N_maxiter.png (max-iteration variant)
```

Supporting / diagnostic scripts (NOT copied into this reproduction folder; they remain in the parent `July23/` iteration folder):

- `plot_complexity_3d_interactive.py` — interactive 3-D convergence view (x=N, y=iteration,
  z=cost, log z); azimuth slider only (mouse rotation disabled), prints `elev/azim` per move.
- `plot_complexity_3d_gap_interactive.py` — same, z = relative optimality gap (J−J\*)/J\*, symlog
  axis (< 10⁰ compressed to the floor = "optimal").
- `plot_complexity_setup.py` — 2-D problem-setup figure (x0, reference, obstacle, per-method
  converged solutions) at a chosen N.
- `mppi_seed_study.py` — MPPI seed-variance study behind §2.3 (8 seeds × 4 K × N ∈ {200,240,300};
  identical protocol to `mppi_solve_traced` but with `seed` varying; writes
  `mppi_seed_study.pkl`). Edit the OUT path at the top before running.

Notes:
- The sweep caches `x0` in the pkl; delete the pkl only if you intend to re-derive it (it will be
  re-extracted by replaying the v6 closed loop — slower).
- `results_complexity.pkl` is merged incrementally: re-running one method overwrites only that
  method's entries. `--iterbench` writes `D["iterbench"]`, read by the plots for m1v2/pmp/ddp.
- The figure range is capped at N ≤ 300 in the plot scripts (`NMAX`), independent of what the pkl
  contains.

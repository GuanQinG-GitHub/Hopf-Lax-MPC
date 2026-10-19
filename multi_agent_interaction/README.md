# Interacting agents: Hopf–Lax-MPC with pairwise collision avoidance

Extension of [`../multi_agent`](../multi_agent/README.md). The K agents there are decoupled. In the
recorded decoupled runs, pairs come within 0.65 m (K = 2), 0.31 m (K = 3), 0.12 m (K = 4) and 0.01 m
(K = 10).

Here the running cost gets a soft **pairwise collision penalty**. This couples the agents through the
costate equation, so the K rollouts become one joint rollout and the Jacobians become dense. The
algorithm (`core/mpc_solvers.hopf_lax_mpc_solve`) is still run **unchanged**. The solver object works
for any number of agents; K = 2, 3, 4 are tested.

![K=4](figs/K4_Rc0.8_Wc500/compare.png)

Contents

1. [Files](#1-files)
2. [Coupled problem: cost, Hamiltonian, canonical system](#2-coupled-problem-cost-hamiltonian-canonical-system)
3. [Implementation](#3-implementation)
4. [Correctness checks](#4-correctness-checks)
5. [Results](#5-results)
6. [Remarks and limitations](#6-remarks-and-limitations)
7. [Reproducing](#7-reproducing)

---

## 1. Files

| file | role |
|---|---|
| [`mpc_multi_int.py`](mpc_multi_int.py) | the pair penalty (`pair_penalty`, NumPy twin `pair_penalty_np`, `pair_distances`), the **coupled shooting object** `build_ss_int` (same API as `build_ss_multi`), the closed-loop step with the true pair cost `apply_step_int` |
| [`mpc_testbed_multi_int.py`](mpc_testbed_multi_int.py) | driver: closed loop for one K (`run_int`), results table, self-test (`--selftest`), CLI |
| [`plot_multi_int.py`](plot_multi_int.py) | `figs/K<K>_Rc<Rc>_Wc<Wc>/compare.png` (coupled vs decoupled) and the printed table |
| [`make_animation_multi_int.py`](make_animation_multi_int.py) | top-down animation with safety discs and minimum-distance HUD → `figs/K<K>_…/anim.mp4` (not versioned) |
| [`results_multi_int_table.txt`](results_multi_int_table.txt) | the recorded results table |
| `results_multi_int.pkl` | per-cycle data (not versioned; regenerate with section 7) |

Reused unchanged:
- **From `../multi_agent`:** the scene, agent layout, references and per-agent `u*` from `mpc_multi.py`,
  `make_env` from `mpc_testbed_multi.py`, and the drawing helpers from `plot_multi.py`.
- **From `../core`:** the model, the single-agent canonical RHS `fz_aug` and `fz_track`, and the solver.

---

## 2. Coupled problem: cost, Hamiltonian, canonical system

The notation is that of `../multi_agent/README.md` sections 2–5: agent k has state
$x_k\in\mathbb R^7$, position $q_k = [p_x,p_y,p_z]_k$, costate $p_k$, single-agent running cost
$L_k$ (tracking, control and obstacle barriers), terminal cost $\phi_k$ and dynamics $f$ (`dyn7`).

**Pair penalty.** For a safety radius $R_c$, weight $W_c$ and shell width $w_c$, in the same tanh family
as the obstacle barriers:

$$
r_{ij} = \sqrt{\frac{\|q_i-q_j\|^2}{R_c^2}+\varepsilon},\qquad
z_{ij} = \frac{r_{ij}-1}{w_c},\qquad
C(q_i,q_j) = \frac{W_c}{2}\big(1-\tanh z_{ij}\big).
$$

Its gradient is

$$
g_{ij} := \nabla_{q_i} C(q_i,q_j) = -\frac{W_c}{2w_c}\big(1-\tanh^2 z_{ij}\big)\frac{q_i-q_j}{R_c^2\,r_{ij}},
\qquad \nabla_{q_j} C(q_i,q_j) = -g_{ij}.
$$

$C$ is symmetric, and its gradient is antisymmetric: $g_{ji} = -g_{ij}$. The defaults are
$R_c = 0.8$ m, $W_c = 500$, $w_c = 0.1$ and $\varepsilon = 10^{-10}$.

**Coupled running cost and Hamiltonian.**

$$
L(\mathbf x,\mathbf u,t) = \sum_{k=1}^K L_k(x_k,u_k,t) + \sum_{i<j} C(q_i,q_j),
\qquad
H(\mathbf x,\mathbf p,\mathbf u,t) = \sum_{k=1}^K \Big[L_k(x_k,u_k,t) + p_k^\top f(x_k,u_k)\Big] + \sum_{i<j} C(q_i,q_j).
$$

**Optimal control is unchanged.** $C$ does not depend on $\mathbf u$, so
$\partial H/\partial u_k = 2Ru_k + [p_{k,5},p_{k,6},p_{k,7}]^\top$. Each agent therefore keeps the
clipped single-agent minimiser $u_k^\star(x_k,p_k)$. The dynamics stay decoupled.

**Costate equation: the only new term.**

$$
\dot p_k = -\frac{\partial H}{\partial x_k}
= -\underbrace{\Big(2Q(x_k-x_{r,k}) + \big[\nabla B_s + \nabla B_m;\,0_4\big] + h(x_k,p_k)\Big)}_{\texttt{dHdx7}(x_k,\,p_k)}
\;-\;\Big[\textstyle\sum_{j\ne k} g_{kj}\,;\ 0_4\Big].
$$

The coupling enters only through the three position entries of each costate. Every agent's costate
is pushed by the others' positions, so the canonical system of all K agents must be integrated
**jointly**.

**Objective, residual, derivatives.** The objective is
$\boldsymbol\Phi(\mathbf v) = \int_0^T L\,dt + \sum_k\phi_k(x_k(T))$, which includes the integrated pair
penalty. There is no terminal coupling, so the transversality residual keeps its per-agent form:

$$
\rho_k = p_k(T) - 2Q_T\big(x_k(T)-x_{r,k}(T)\big).
$$

The gradient and the curvature surrogate keep their form,
$\nabla\boldsymbol\Phi = -\mathbf S_x^\top\boldsymbol\rho$ and
$\mathbf M = \mathrm{sym}(-\mathbf S_x^\top\mathbf J_r)$. Now, however,
$\mathbf J_r = \partial\boldsymbol\rho/\partial\mathbf v$ and $\mathbf S_x$ have **nonzero off-diagonal
blocks** $\partial\rho_i/\partial v_j$ whenever agents i and j interact within the horizon. They are not
block-diagonal any more. These are the dense $7K\times7K$ matrices the unchanged algorithm already
forms in the decoupled case.

**Per-iteration cost.**
- **RHS evaluation:** each evaluation costs $\mathcal O(K)$ for the agent terms and $\mathcal O(K^2)$
  for the pair terms.
- **Sensitivities:** with the 7K tangent directions, the sensitivities cost
  $\mathcal O(NK^2 + NK^3)$. The cubic part has a small constant (a 3-vector per pair).
- **Dense algebra:** $\mathcal O(K^3)$, as before.

---

## 3. Implementation

`mpc_multi_int.build_ss_int(M, K, N, dt, JX, col, napply)`:

- **Joint rollout.** One `lax.scan` carries $Z_A\in\mathbb R^{K\times(2\cdot7+1)}$: per agent
  $[x_k, p_k, c_k]$, with $c_k = \int L_k$ exactly as in the decoupled object, plus a scalar
  $c_p = \int\sum_{i<j}C$.
  - **Agent terms:** the RHS is `jax.vmap(JX.fz_aug)` over agents, i.e. the core's per-agent dynamics,
    $-\partial H/\partial x$ and $L_k$, unchanged.
  - **Coupling:** the position entries of $\dot p_k$ get $-G_k$, with $G_k = \sum_{j\ne k}g_{kj}$, and
    $\dot c_p = \sum_{i<j}C$.
  - **Integration:** RK4 with the same sub-stage reference sampling as the core, so
    $\Phi = \sum_k(c_k+\phi_k) + c_p$.
- **Pair penalty** (`pair_penalty`). It is vectorised over the $K\times K$ difference tensor with the
  diagonal masked out, which works for any K and gives exactly 0 for K = 1. ε keeps the masked
  diagonal square root differentiable.
- **API.** The interface is identical to `build_ss_multi`: `res, J, all, Jr, Sx, g, resx, roll_pred,
  roll_costate, warm`, with flat agent-major 7K vectors and $X_r\in\mathbb R^{K\times(7+3n_m)\times(2N+1)}$.
  So `hopf_lax_mpc_solve` and `ensure_jbatch` run on it unchanged. `roll_pred` and `roll_costate`
  (the warm start) are joint rollouts too.
- **Self-test helpers.** `hamiltonian` (the joint H) and `costate_rhs` (the joint $\dot{\mathbf p}$).

**Closed loop** (`apply_step_int`).
- The plants are integrated per agent, as before.
- Each agent accrues tracking and obstacle cost until its own arrival.
- Each pair accrues the true collision penalty while at least one of its two agents is still active.
- The minimum pairwise distance is recorded over **every** fine RK4 step.
- Arrived agents stay in the problem: they keep flying and remain obstacles for the others.

---

## 4. Correctness checks

`python mpc_testbed_multi_int.py --selftest` (all passed):

| # | check | result |
|---|---|---|
| 1 | **Exact reductions** to `../multi_agent` `build_ss_multi`: $W_c=0$ with K = 2, 3, and K = 1 with the penalty on. Compared: ρ, Φ, $J_r$, $S_x$, `roll_costate`, `roll_pred`, and the solved costate with its iteration count | all **bitwise identical** |
| 2 | Analytic pair gradient $G$ vs `jax.grad` of $\sum_{i<j}C$, K = 2, 3, 5 (pairs inside and outside $R_c$); JAX vs NumPy penalty | rel. diff ≤ 4.4e-16; ≤ 5.7e-14 |
| 3 | **Hand-derived costate equation** (section 2) vs $-\nabla_x$ of the joint Hamiltonian with u held at u*, K = 2 and 4, agents clustered (minimum distance 0.04–0.42 m) | rel. diff ≤ 1.6e-16 |
| 4 | Gradient identity $-\mathbf S_x^\top\boldsymbol\rho$ vs `jax.grad(Φ)` with active coupling, K = 2, 3 | rel. diff 1.8e-9 … 5.7e-7 (RK4 discretisation, as in the single-agent case) |
| 5 | Permutation equivariance, K = 3: permuting agents permutes ρ and $x(T)$ and leaves Φ unchanged | ≤ 5.7e-14 / identical |
| 6 | Coupling is real, K = 2: off-diagonal $J_r$ blocks with agents 4 m apart vs 0.5 m apart | 0 vs 1.3e2 |

**Closed-loop reduction.** `--nocol` runs the coupled code with $W_c = 0$, and it reproduces the
recorded decoupled runs of `../multi_agent`:

| K | sum of costs | minimum distance | last arrival |
|---|---|---|---|
| 2 | 1.071 | 0.654 m | 14.34 s |
| 3 | 2.456 | 0.307 m | 14.38 s |
| 4 | 3.216 | 0.124 m | 14.92 s |

---

## 5. Results

Existing obstacle course, N = 220, no budget, iteration cap 1000. The coupled runs use $R_c = 0.8$ m,
$W_c = 500$, $w_c = 0.1$. Full table: [`results_multi_int_table.txt`](results_multi_int_table.txt).

| run | min distance | sum of agent costs | pair cost | last arrival | it/cycle (max) | converged | ms/iter |
|---|---|---|---|---|---|---|---|
| K=2 decoupled | 0.654 m | 1.071 | – | 14.34 s | 2.37 (3) | 100 % | 13.4 |
| **K=2 coupled** | **1.210 m** | 1.187 | 0.024 | 14.50 s | 2.56 (6) | 100 % | 11.2 |
| K=3 decoupled | 0.307 m | 2.456 | – | 14.38 s | 2.49 (5) | 100 % | 16.9 |
| **K=3 coupled** | **1.180 m** | 2.846 | 0.082 | 14.78 s | 3.02 (7) | 100 % | 16.9 |
| K=4 decoupled | 0.124 m | 3.216 | – | 14.92 s | 2.57 (5) | 100 % | 22.5 |
| **K=4 coupled** | **1.109 m** | 5.879 | 0.696 | 15.44 s | 3.46 (10) | 100 % | 20.8 |

Findings:

- **The coupling works.** With the penalty, every pair stays above $R_c$ for the whole run, at
  1.11–1.21 m minimum distance. The decoupled runs reach 0.65 / 0.31 / 0.12 m. All agents still reach
  their goals, 0.2–0.5 s later.
- **Price of separation.** The tracking and obstacle cost rises by 11 %, 16 % and 83 % for K = 2, 3, 4.
  The pair cost itself stays small, because the agents keep the penalty mostly inactive: the closed
  loop anticipates encounters over the 1.1 s horizon rather than pushing through the barrier. For
  K = 4 the paths are reshaped along the whole course, not only at the encounters (figure above).
- **Solver.** Every cycle of every run converged, with the certified stop of section 4 of the
  multi-agent README. The iteration count grows modestly (2.6 → 3.5 per cycle at K = 4). The saddle
  escape fired twice, at K = 4 (2 kicks), which is the expected pass-left-or-right symmetry.
- **Per-iteration cost is unchanged** within timing noise. The joint rollout costs no more than the
  vmapped independent rollouts, because the sensitivities were already formed densely over all 7K
  directions in the decoupled object. The $\mathcal O(K^2)$ pair term is negligible at these K.
  The machine was not idle during these runs: the decoupled K = 2 baseline took 13.4 ms/iter here,
  against 9.7 ms recorded in `../multi_agent`. Treat the timings as indicative only.

Figures: `figs/K<K>_Rc0.8_Wc500/compare.png`, for K = 2, 3, 4. Each shows the map with coupled
(solid) vs decoupled (dashed) paths and safety discs at the closest approach, the minimum pairwise
distance vs time, $y(t)$, the inter-agent Jacobian coupling $\|J_r^{\text{offdiag}}\|$, and the
iterations per cycle.

---

## 6. Remarks and limitations

- **Goal spacing vs $R_c$.** The layout puts the goals on the line x = 20 with spacing 4/(K−1) m. For
  K ≥ 6 this is below $R_c = 0.8$ m, so the penalty would compete with reaching the goals. A K-sweep
  must scale $R_c$ below the goal spacing (for example $R_c \approx 0.35$ m for K = 10). `--Rc` sets it.
- **Soft, not hard.** The penalty is a cost, not a constraint. Separation is not guaranteed; it is
  traded against tracking through $W_c$ and $w_c$.
- **Timing hygiene.** The scaling numbers of `../multi_agent` were taken on an idle machine. The runs
  here were not (see section 5), so compare ms/iter only within this table.

---

## 7. Reproducing

From this folder, with the `HopfLaxMPC` environment (on Windows set `PYTHONNOUSERSITE=1`, see the
repository README).

```powershell
python mpc_testbed_multi_int.py --selftest                 # section 4
python mpc_testbed_multi_int.py --K 2,3,4 --nocol           # decoupled baselines (Wc = 0)
python mpc_testbed_multi_int.py --K 2,3,4                   # coupled (Rc 0.8, Wc 500, wc 0.1)
python plot_multi_int.py > results_multi_int_table.txt      # figs/K*_Rc0.8_Wc500/compare.png + table
python make_animation_multi_int.py --K 4                    # figs/K4_Rc0.8_Wc500/anim.mp4
```

Options of `mpc_testbed_multi_int.py`:
- `--K` (comma list), `--N` (default 220)
- `--Rc`, `--Wc`, `--wc` (penalty), `--nocol`
- `--maxcyc`, `--cap`, `--out`, `--redo`, `--print-every`

Results merge into the pickle under the key (K, N, Rc, Wc, wc).

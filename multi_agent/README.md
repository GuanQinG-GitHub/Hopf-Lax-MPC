# Multi-agent scaling of Hopf–Lax-MPC

How does the per-iteration cost of Hopf–Lax-MPC grow with the number of vehicles, i.e. with the
dimension of the state space? K identical 7-D aerial vehicles are stacked into **one** optimal-control
problem of dimension n = 7K and solved by the unchanged Hopf–Lax-MPC algorithm, for K = 1, …, 10.

![scaling](figs/multi/scaling.png)

Contents

1. [Files in this folder](#1-files-in-this-folder)
2. [Single-agent model: dynamics, cost, Hamiltonian](#2-single-agent-model-dynamics-cost-hamiltonian)
3. [The Hopf–Lax objective and its derivatives](#3-the-hopflax-objective-and-its-derivatives)
4. [The algorithm per MPC cycle](#4-the-algorithm-per-mpc-cycle)
5. [Stacking K agents](#5-stacking-k-agents)
6. [Test scenario](#6-test-scenario)
7. [Protocol and measured quantities](#7-protocol-and-measured-quantities)
8. [Correctness checks](#8-correctness-checks)
9. [Results](#9-results)
10. [Complexity: measured scaling vs the analysis](#10-complexity-measured-scaling-vs-the-analysis)
11. [Reproducing everything](#11-reproducing-everything)

---

## 1. Files in this folder

| file | role |
|---|---|
| [`mpc_multi.py`](mpc_multi.py) | the multi-agent layer: frozen scene (`freeze_mobs`), planner obstacles (`planner_obstacles`), agent layout (`agent_layout`, `make_multi_scene`), stacked reference window (`ref_window_multi`), **stacked shooting object** (`build_ss_multi`), per-agent closed-loop step and true cost (`apply_step_multi`), stacked optimal control (`ustar_multi`) |
| [`mpc_testbed_multi.py`](mpc_testbed_multi.py) | the experiment driver: environment (`make_env`), the closed loop for one K (`run_multi`), the results table (`print_table`), the correctness tests (`selftest`), CLI |
| [`plot_multi.py`](plot_multi.py) | all figures from the results pickle: per-K `paths.png` / `review.png`, the scaling figure, and the printed table |
| [`make_animation_multi.py`](make_animation_multi.py) | top-down (default) and 3-D (`--three-d`) animations of one K, and publication stills (`--snapshot`) |
| [`results_multi_table.txt`](results_multi_table.txt) | the recorded results table (printed by `plot_multi.py`) |
| `results_multi.pkl` | the recorded per-cycle data (not versioned; regenerate with section 11) |
| [`figs/multi/`](figs/multi/) | `scaling.png`, `multi_agent_scaling.pdf`, and one folder `K1` … `K10` per test |

Shared modules used from sibling folders (added to `sys.path` at the top of each script):

| module | used for |
|---|---|
| [`../core/mpc_core.py`](../core/mpc_core.py) | `make_P`, `model_scn` (model, weights), `build_jax` (dynamics, barriers, Hamiltonian derivatives, RK4), `build_ss` (single-agent shooting object; reference for the self-test), `ref_window`, `plan_path` (RRT), `_value_and_jacfwd` (fused forward-mode Jacobian) |
| [`../core/mpc_solvers.py`](../core/mpc_solvers.py) | `hopf_lax_mpc_solve` (Algorithm 1), `ensure_jbatch` (batched objective), constants `HOPF_LAX_MPC_ETA_S`, `HOPF_LAX_MPC_RIDGE`, `HOPF_LAX_MPC_DELTA`, `HOPF_LAX_MPC_ESC_ALPHAS` |
| [`../core/mpc_tuned_params.py`](../core/mpc_tuned_params.py) | `apply`: the obstacle course of the closed-loop testbed (positions and barrier parameters reused here, frozen) |
| `../closed_loop/mpc_testbed.py` | the JAX compile-flag handler `CF` (recompile detection) |
| `../N_sweep/mpc_testbed_N_sweep.py` | `hopf_lax_mpc_status` (termination status from the solver trace) |

---

## 2. Single-agent model: dynamics, cost, Hamiltonian

All equations are implemented in `core/mpc_core.py`, `build_jax` (function names in brackets).

**State and control.** n₀ = 7, m₀ = 3:

$$
x = [p_x,\ p_y,\ p_z,\ \theta,\ v,\ \omega,\ v_z]^\top,\qquad
u = [u_1,\ u_2,\ u_3]^\top,\qquad |u_i| \le u_{\max} = 3 .
$$

**Dynamics** (`dyn7`): a planar unicycle with a decoupled vertical channel,

$$
\dot x = f(x,u) = [\,v\cos\theta,\ v\sin\theta,\ v_z,\ \omega,\ u_1,\ u_2,\ u_3\,]^\top .
$$

**Running and terminal cost.** With the tracking reference $x_r(t)$ (section 6), position
$q = [p_x,p_y,p_z]^\top$ and obstacle barriers $B$,

$$
L(x,u,t) = (x-x_r)^\top Q (x-x_r) + u^\top R u + B_s(q) + B_m(q),
\qquad
\phi(x_T) = (x_T - x_{r,T})^\top Q_T (x_T - x_{r,T}),
$$

$$
Q = \mathrm{diag}(2,\,0.4,\,2,\,0.5,\,0.3,\,0.3,\,0.3),\quad
Q_T = \mathrm{diag}(10,\,2,\,10,\,2,\,1,\,1,\,1),\quad
R = 0.05\,I_3 .
$$

**Obstacle barriers** (`obstacle`, `mobs_pen`). Each obstacle j is an ellipsoid with centre $c_j$,
half-axes $(a_j, b_j, d_j)$, weight $W_j$ and shell width $w_j$:

$$
r_j(q) = \sqrt{\left(\tfrac{q_x-c_{j,x}}{a_j}\right)^2+\left(\tfrac{q_y-c_{j,y}}{b_j}\right)^2+\left(\tfrac{q_z-c_{j,z}}{d_j}\right)^2+\varepsilon},
\qquad
B(q) = \sum_j \tfrac{W_j}{2}\left(1-\tanh\tfrac{r_j(q)-1}{w_j}\right),
$$

with $\varepsilon = 10^{-10}$. $B_s$ sums over the static obstacle, $B_m$ over the (here frozen) moving
set; the centres of $B_m$ are carried per stage inside the reference window.

**Hamiltonian and optimal control** (`ustar`). With costate $p\in\mathbb R^7$,

$$
H(x,p,u,t) = L(x,u,t) + p^\top f(x,u),
\qquad
u^\star(x,p) = \operatorname{clip}\!\left(-\tfrac12 R^{-1} [p_5,p_6,p_7]^\top,\ -u_{\max},\ u_{\max}\right),
$$

since $\partial H/\partial u = 2Ru + [p_5,p_6,p_7]^\top = 0$ and $R$ is diagonal (the clip is the
pointwise minimiser under the box constraint).

**Bi-characteristic (canonical) system** (`fz_track`, `dHdx7`):

$$
\dot x = f\big(x, u^\star(x,p)\big), \qquad
\dot p = -\frac{\partial H}{\partial x}
= -\Big( 2Q(x-x_r) + \big[\nabla_q B_s + \nabla_q B_m;\ 0_4\big] + h(x,p) \Big),
$$

$$
h(x,p) = \big[\,0,\ 0,\ 0,\ v(-p_1\sin\theta + p_2\cos\theta),\ p_1\cos\theta + p_2\sin\theta,\ p_4,\ p_3\,\big]^\top ,
$$

where $h = \partial(p^\top f)/\partial x$: the θ-entry from differentiating $v\cos\theta, v\sin\theta$,
the v-entry from the same two terms, the ω-entry from $\dot\theta = \omega$, the $v_z$-entry from
$\dot p_z = v_z$.

**Discretisation** (`rk4_canon_aug`). RK4 with step $\Delta t = 0.005$ s over N steps
($T = N\Delta t$). The reference and the obstacle centres are sampled at the RK4 sub-stages (two
columns per step). The running cost is integrated in the same pass by augmenting the canonical state
with $\dot c = L$, so one rollout returns $x(T)$, $p(T)$ and $\int_0^T L\,dt$.

---

## 3. The Hopf–Lax objective and its derivatives

Implemented in `build_ss` (single agent) and `build_ss_multi` (stacked), functions `_outs`, `S_all`.

The decision variable is the **initial costate** $v = p(0)$. One canonical rollout from
$(x(0), p(0)) = (x_c, v)$ defines

$$
\Phi(v) = \int_0^T L\big(x, u^\star(x,p), t\big)\,dt + \phi\big(x(T)\big) \qquad \text{(objective, } \texttt{S.J}\text{)},
$$

$$
\rho(v) = p(T) - 2Q_T\big(x(T) - x_{r,T}\big) \qquad \text{(transversality residual, } \texttt{S.res}\text{)} .
$$

$\rho(v) = 0$ is the PMP boundary condition; Hopf–Lax-MPC minimises $\Phi$ rather than only solving
$\rho = 0$. With the sensitivities

$$
J_r = \frac{\partial \rho}{\partial v},\qquad S_x = \frac{\partial x(T)}{\partial v},
$$

computed by forward-mode differentiation of the rollout (`S.all`: 7 tangent directions in one
vmapped JVP pass, returning $\rho, \Phi, J_r, S_x$ together), the gradient and the curvature
surrogate are

$$
\nabla\Phi(v) = -S_x^\top \rho, \qquad
M(v) = \operatorname{sym}\!\left(-S_x^\top J_r\right) \approx \nabla^2\Phi(v),
$$

the surrogate differing from the exact Hessian by terms proportional to $\|\rho\|$. Both come for
free from the one forward pass; no second-order differentiation is done.

---

## 4. The algorithm per MPC cycle

`core/mpc_solvers.py`, `hopf_lax_mpc_solve(S, v, x_c, X_r, PP, ε_c, η_s, α, δ, t_c, budget)`; the
stacked problem calls it **unchanged**. One iteration k:

1. **Rollout and sensitivities:** $\rho, \Phi, J_r, S_x \leftarrow$ `S.all(v)`;
   $g = -S_x^\top\rho$; $M = \operatorname{sym}(-S_x^\top J_r)$.
2. **Gated curvature read:** if escape mode is on, or $\|g\| \le \eta_s(1+|\Phi|)$ with
   $\eta_s = 0.29$, compute $M = V\Lambda V^\top$; escape $\Leftarrow \lambda_{\min} < -\varepsilon_c$
   ($\varepsilon_c = 10^{-3}$).
3. **Certificate:** stop if $\|g\| \le \texttt{tol}\,(1+|\Phi|)$ with tol = $10^{-4}$ and not in escape
   mode (a certified local minimiser).
4. **Newton descent** (not in escape): solve $J_r d = -\rho$; take the first
   $a\in\{1, \tfrac12, \dots, \tfrac1{128}\}$ with $\|\rho(v+ad)\| < \|\rho(v)\|$. On failure, read the
   curvature; a convex read ends the solve, a negative one enters escape mode.
5. **Saddle escape:** if the gradient is blind along the negative-curvature direction
   ($|g^\top v_{\min}|/\|g\| < 0.1$), **kick**: compare $\Phi(v \pm \alpha v_{\min})$, $\alpha = 0.3$
   (one batched evaluation), move to the lower side, re-shoot. Then a modified Newton step in
   $|M|$: $d = -V\,\mathrm{diag}\!\big(1/\max(|\lambda_i|,\delta)\big) V^\top g$, $\delta = 10^{-8}$,
   with a batched line search on $\Phi$ over $a \in 2^{-\{0,\dots,6\}}$.

Between cycles the costate is **warm-started** by rolling the canonical system over the applied
interval and taking $v \leftarrow p(\Delta t_{\text{apply}})$ (`S.roll_costate`). The applied control
is $u^\star(x_c, v)$, held for $\Delta t_{\text{apply}} = 0.02$ s (10 RK4 steps, 50 Hz).

---

## 5. Stacking K agents

`mpc_multi.py`, `build_ss_multi(M, K, N, dt, JX, napply)`.

Agents are fully **decoupled**: no inter-agent cost and no coupling in the dynamics. The problem is
stacked, not restructured. With agent states $x^{(k)}$, references $x_r^{(k)}$ and initial costates
$v^{(k)}$, k = 1, …, K:

$$
\mathbf x = \begin{bmatrix} x^{(1)} \\ \vdots \\ x^{(K)} \end{bmatrix}\in\mathbb R^{7K},\qquad
\mathbf v = \begin{bmatrix} v^{(1)} \\ \vdots \\ v^{(K)} \end{bmatrix}\in\mathbb R^{7K},\qquad
\boldsymbol\Phi(\mathbf v) = \sum_{k=1}^K \Phi_k\big(v^{(k)}\big),\qquad
\boldsymbol\rho(\mathbf v) = \begin{bmatrix} \rho_1(v^{(1)}) \\ \vdots \\ \rho_K(v^{(K)}) \end{bmatrix}.
$$

Each $\Phi_k, \rho_k$ is the single-agent objective and residual of section 3 for agent k with its own
reference window; the K rollouts are evaluated by `jax.vmap` over agents. The reference window becomes
a 3-D array $X_r \in \mathbb R^{K\times(7+3n_m)\times(2N+1)}$ (`ref_window_multi`, $n_m = 7$ obstacle
centres per column).

**What the algorithm sees.** `S.all` differentiates the stacked map
$\mathbf v \mapsto (\boldsymbol\rho, \mathbf x(T), \boldsymbol\Phi)$ with **7K** tangent directions,
exactly as `build_ss` does for 7. Because the agents are decoupled the Jacobians are block-diagonal,

$$
\mathbf J_r = \operatorname{blkdiag}(J_{r,1},\dots,J_{r,K}),\qquad
\mathbf S_x = \operatorname{blkdiag}(S_{x,1},\dots,S_{x,K}),\qquad
\mathbf M = \operatorname{blkdiag}(M_1,\dots,M_K),
$$

**but this structure is not exploited**: the full dense $7K\times7K$ matrices are formed and the
eigen-decomposition and linear solves of section 4 act on them. The object has the same API as
`build_ss` (`res, J, all, Jr, Sx, g, roll_pred, roll_costate, warm`), so `hopf_lax_mpc_solve` and
`ensure_jbatch` run on it unchanged. This is the honest "stack the states and run the algorithm"
cost. Line searches, the gate and the certificate act on the stacked quantities, so all agents share
one step size and one stopping test.

**Applied controls and true cost** (`ustar_multi`, `apply_step_multi`). Each agent applies
$u^{(k)} = u^\star(x^{(k)}, v^{(k)})$; the plant is integrated per agent, and each agent's true cost
(tracking $\Delta t_{\text{apply}}$-sampled, barrier integrated at every RK4 step) is accumulated
**until its own arrival**, so it equals what a solo run of that agent records.

---

## 6. Test scenario

`mpc_multi.py`, `make_multi_scene(P, K)`; parameters from `core/mpc_core.make_P`,
`core/mpc_tuned_params.apply`.

**Obstacles, all frozen at t = 0** (`freeze_mobs`: velocities set to zero). Centres $(x, y, z)$,
half-axes $(a, b, d)$, weight W, shell w:

| obstacle | centre | (a, b, d) | W | w |
|---|---|---|---|---|
| obs-1 (static bump) | (3.0, 0.25, 0.0) | (0.75, 1.50, 0.75) | 200 | 0.10 |
| wall | (12.0, 0.00, 0.5) | (0.80, 0.20, 5.0) | 500 | 0.25 |
| lane blocker | (11.8, 0.55, 0.5) | (0.50, 0.20, 5.0) | 900 | 0.05 |
| door post | (13.8, 3.33, 0.5) | (0.40, 0.80, 5.0) | 900 | 0.05 |
| shoulder blocker | (10.4, −0.55, 0.5) | (0.70, 0.30, 5.0) | 20000 | 0.08 |
| fork post | (16.2, 0.00, 0.5) | (0.30, 0.30, 5.0) | 900 | 0.07 |
| stage-2 blocker | (17.4, 0.60, 0.5) | (0.35, 0.30, 5.0) | 900 | 0.08 |
| fork companion | (16.2, −10.6, 0.5) | (0.80, 0.40, 5.0) | 50000 | 0.08 |

The large d = 5 makes the moving-set barriers effectively vertical columns. Frozen at t = 0, the door
sits at the top edge of the map and the companion is far off the map, so both are inert here.

**Agent layout** (`agent_layout`). Agent k (k = 0, …, K−1) spawns at
$(0,\ y_k,\ 0.5)$ with $x^{(k)}(0) = [0,\ y_k,\ 0.5,\ 0,\ 1.5,\ 0,\ 0]$ and must reach the
**opposite corner** $(20,\ -y_k,\ 0.5)$, with

$$
y_k = -2 + \frac{4k}{K-1}\quad (K>1), \qquad y_0 = 0\quad (K = 1),
$$

so routes cross the course diagonally and cross each other. The vehicle flies at constant altitude
0.5 m.

**References** (`planner_obstacles`, `core/mpc_core.plan_path`). One RRT path per agent, planned
against the static obstacle plus every frozen body inflated to a sphere of radius
$\max(a, b) + 0.25$, so each reference is a feasible route (unlike the single-agent testbed, whose
planner model is deliberately wrong). The tracking reference $x_r(t)$ is the path sampled at the cruise
speed $v_{cr} = 1.5$ m/s from the closest path point (`ref_window`): positions, heading, $v_{cr}$, turn
rate and climb rate along the path.

**Goal test.** Agent k has arrived when $\|q^{(k)} - q_{\text{goal}}^{(k)}\| < 0.30$ m.

---

## 7. Protocol and measured quantities

`mpc_testbed_multi.py`, `make_env`, `run_multi`.

| setting | value |
|---|---|
| agents K | 1, 2, …, 10 (stacked dimension n = 7K = 7 … 70) |
| horizon | N = 220 ($T = 1.1$ s), the tuned horizon of Hopf–Lax-MPC in the closed-loop testbed |
| real-time budget | **none** (`P.budget = ∞`) |
| iteration cap | 1000 (`P.lm_maxit`) |
| control rate | 50 Hz, 10 RK4 steps per applied control |
| run length | until every agent has arrived, at most 800 cycles (16 s) |
| warm-up | `S.warm` and the batched objective (B = 2, 7) compiled before the timed loop |

Per cycle the driver records: wall time of the solver call (`tsolve`) and of the whole cycle
(`tcyc`), the iteration count, the termination status (`converged`, `cap`, `stalled`,
`esc-stalled`, `timeout`), $\|g\|$ at exit, the numbers of descent / escape iterations, kicks and
eigen-reads, the surrogate $\lambda_{\min}$ and $\|\rho\|$ at the applied iterate, and the applied
costate (for the animation's predicted horizons). Per agent: trajectory, controls, tracking and
obstacle cost, arrival time.

**Per-iteration average cost** (the scaling metric):

$$
\bar t_{\text{iter}}(K) = \operatorname*{mean}_{c\,\ge 2,\ \text{iters}_c>0}\ \frac{t_{\text{solve},c}}{\text{iters}_c},
$$

over all cycles except the first (compile/cold start) and except cycles that needed no iteration.

**Timing hygiene.** A first sweep ran while another job occupied the CPU; it inflated K = 1 by
37 percent and K = 6 to 8 by 20 to 30 percent and was discarded. All reported numbers come from one
uninterrupted sweep on an idle machine.

---

## 8. Correctness checks

`python mpc_testbed_multi.py --selftest` (`selftest`):

1. **K = 1 vs the single-agent object.** `build_ss_multi` with one agent against `core.build_ss` at
   the same state: reference window identical, $\rho$, $\Phi$ and `roll_costate` identical,
   $J_r, S_x$ equal to round-off ($10^{-17}$), and `hopf_lax_mpc_solve` returns a bitwise-identical
   costate in the same number of iterations.
2. **K = 2 vs two solo solves.** Stacked $\rho$ equals the concatenation of the solo residuals
   ($3.6\cdot10^{-15}$), $\Phi$ equals their sum, the diagonal blocks of $J_r, S_x$ equal the solo
   Jacobians and the off-diagonal blocks are exactly zero; the converged stacked costate equals the
   two solo solutions to $2.3\cdot10^{-9}$.

**Closed loop** (`--K 2`, `--K 2 --solo 0`, `--K 2 --solo 1`): each agent of the stacked run matches
its solo run over the whole trajectory,

| agent | max \|ΔU\| | max \|ΔX\| | J stacked / solo | arrival |
|---|---|---|---|---|
| 0 | 5.0e-7 | 2.2e-8 | 0.5167 / 0.5167 | 14.22 s / 14.22 s |
| 1 | 2.6e-5 | 5.2e-7 | 0.5546 / 0.5546 | 14.34 s / 14.34 s |

---

## 9. Results

Full table: [`results_multi_table.txt`](results_multi_table.txt). Every run converged in every cycle
and every agent reached its goal.

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| state dim 7K | 7 | 14 | 21 | 28 | 35 | 42 | 49 | 56 | 63 | 70 |
| ms per iteration (mean) | 5.4 | 9.7 | 14.6 | 21.4 | 28.5 | 58.2 | 72.4 | 91.2 | 111.7 | 175.6 |
| iterations per cycle (mean) | 2.22 | 2.37 | 2.49 | 2.57 | 2.60 | 2.63 | 2.63 | 2.80 | 2.86 | 2.83 |
| solve ms per cycle (mean) | 11.8 | 22.7 | 36.0 | 54.5 | 73.5 | 151 | 188 | 253 | 315 | 490 |
| solve ms per cycle (max) | 21.7 | 29.3 | 61.3 | 89.0 | 119 | 252 | 314 | 403 | 670 | 1975 |
| sum of agent costs | 1.39 | 1.07 | 2.46 | 3.22 | 4.18 | 4.69 | 5.84 | 7.68 | 9.59 | 10.8 |

Findings:

- **Per-iteration cost grows between linearly and quadratically in K**: 5.3x from K = 1 to 5 (close
  to linear), 32x from K = 1 to 10 (linear would be 10x, quadratic 100x).
- **The iteration count is almost flat** (2.2 to 2.9 per cycle), so the cost per cycle follows the
  per-iteration cost.
- **Real-time margin at 10 Hz** (100 ms per update): a fully converged solve fits for K ≤ 5 (mean
  73 ms); a single iteration fits for K ≤ 8 (91 ms).
- **The step between K = 5 and K = 6** (2.0x instead of about 1.3x) is reproducible on an idle machine
  and is not algorithmic; see section 10.

Figures and media, per K in [`figs/multi/K<K>/`](figs/multi/):

| file | content |
|---|---|
| `paths.png` | x–y map: frozen obstacles (core and shell), each agent's RRT reference (dashed) and closed-loop path, start and goal |
| `review.png` | the map with 1 s ticks, plus y(t), speed, heading, acceleration command, distance to goal, and per-cycle iterations / solve time |
| `K10/anim.mp4` | top-down real-time animation (trails, predicted horizons, HUD with time, arrivals, solver stats) |
| `K10/anim3d.mp4` | the same in 3-D, in the style of `closed_loop/make_animations.py` |
| `K10/anim3d_last.png/.pdf` | final 3-D frame at 450 dpi, MATLAB-style box frame (style of `closed_loop/snapshot_frame.py`) |
| `figs/multi/scaling.png`, `multi_agent_scaling.pdf` | the scaling figure: per-iteration time (mean, max) vs K with linear / quadratic / cubic references, solve time per cycle, iterations per cycle, non-converged fraction; x ticks show K and 7K |
| `figs/multi_paths_K2.png` | the two-agent path figure from the correctness run |

---

## 10. Complexity: measured scaling vs the analysis

The paper's per-iteration bound is

$$
C = C_2 + \ell_{\text{New}}C_1 + \underbrace{2C_1 + C_2}_{\text{kick}} + \ell_{\text{esc}}C_1 + \mathcal O(n^3)
= \mathcal O(Nn^3 + Nn^2 m),
$$

with $C_1 = \mathcal O(Nn^2 + Nnm)$ for one rollout and $C_2 = (1+\kappa_1 n)C_1$ for the
sensitivities. That bound assumes **dense** dynamics and cost, where one rollout step costs
$\mathcal O(n^2)$. For K decoupled agents ($n = 7K$, $n_0 = 7$, $m = 3K$):

| term | general bound | decoupled stack, as implemented | measured share |
|---|---|---|---|
| (i) rollout $C_1$ | $\mathcal O(Nn^2)$ | K independent 7-D rollouts: $\mathcal O(N K)$ | small |
| (ii) sensitivities $C_2$ | $\mathcal O(Nn^3)$ | $7K$ tangents × a rollout of cost $\mathcal O(NK)$: $\mathcal O(N K^2)$ | dominant |
| (iii) eigen-decomposition, solves | $\mathcal O(n^3)$ | dense $7K\times7K$: $\mathcal O(K^3)$, tiny constant | < 0.1 ms at K = 8 |

So within K ≤ 10 the cost behaves like $aK + bK^2 + cK^3$ with $c$ negligible, which is the measured
"between linear and quadratic" curve; the cubic term would appear only once the $n\times n$ algebra
catches up with the rollouts, far beyond n = 70. The $\mathcal O(K^2)$ term exists only because a
dense Jacobian of a block-diagonal problem is formed; computing each agent's Jacobian with its own 7
tangents would make (ii) $\mathcal O(NK)$.

**The K = 5 → 6 step.** It sits entirely in the fused Jacobian pass `S.all`, which evaluates
$7K$ tangents × K agents $= 7K^2$ rollouts in one XLA CPU kernel; the single rollout, the batched
line search and the dense algebra grow smoothly across K = 5 → 6. Varying the number of tangent
directions independently of K reproduces the step whenever the batch exceeds about 200 rollouts
(K = 4: between 42 and 49 tangents; K = 5: between 35 and 42; K = 6: between 28 and 35; K = 7:
between 21 and 28), and $7K^2$ crosses 200 between K = 5 (175) and K = 6 (252). It is a batch-size
regime change of the compiled kernel, marked by the dashed line in the scaling figure.

---

## 11. Reproducing everything

From this folder, with the `HopfLaxMPC` environment (see the repository README). Keep other compute
jobs off the machine during timing runs.

```powershell
python mpc_testbed_multi.py --selftest                       # section 8, checks 1 and 2
python mpc_testbed_multi.py --K 2                             # two-agent closed loop
python mpc_testbed_multi.py --K 2 --solo 0                    # agent 0 alone (same layout)
python mpc_testbed_multi.py --K 2 --solo 1                    # agent 1 alone
python mpc_testbed_multi.py --K 1,2,3,4,5,6,7,8,9,10 --redo   # the scaling sweep (about 30 min)
python plot_multi.py > results_multi_table.txt                # figs/multi/K*/, scaling.png, multi_agent_scaling.pdf, table
python make_animation_multi.py --K 10                         # figs/multi/K10/anim.mp4
python make_animation_multi.py --K 10 --three-d               # figs/multi/K10/anim3d.mp4
python make_animation_multi.py --K 10 --three-d --snapshot last --snapshot-dpi 450   # anim3d_last.png/.pdf
```

`mpc_testbed_multi.py` options: `--K` (comma list), `--N` (default 220), `--solo k`, `--maxcyc`,
`--cap` (default 1000), `--out` (default `results_multi.pkl`), `--redo`, `--print-every`.
Results merge per K into the pickle and are saved after every run. `plot_multi.py` writes the scaling
figure as `scaling.png` and its vector copy `multi_agent_scaling.pdf`.

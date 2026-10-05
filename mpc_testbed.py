"""
mpc_testbed.py -- closed-loop MPC comparison on the v10 UNIFIED-ROUTE scenario: original
+y orientation (door descends from +y as in v6-v8), EVERY method dodges the wall +y, and the DDP
trap is the downstream fork with its companion at W=50000 (deterministic branch tie-break).

  * PMP   N = 260   trapped at the on-axis wall root
  * DDP   N = 300   +y nudge at the wall (like its v6-v8 baseline config); dodges +y WITH
                    everyone, threads blocker+door on ours' exact lane, then grinds at the fork
                    at (16.2,+0.39): TRAPPED 3/3 near-identically (J ~13.5)
  * M1_v2 N = 220   kicks at the wall AND at the fork -- visits both trap states, escapes both;
                    +y slalom deterministic (companion W=5e4 tie-break), J 2.5-3.7
  * MPPI  budget-saturating corners on THIS 8-body course: 12288/340 (one wave, 19.45 ms) and
          20480/175 (two waves, 19.79 ms); wide-lambda-bracket fix active for both

Design iteration, rationale and all measured numbers: mpc_tuned_params.py (apply).
Results go to results_testbed_v10.pkl.  Runners carry their own N (r.N) for summary/animations.

BUDGET-ENFORCEMENT NOTE (collocation).  Collocation runs with the HOST-STYLE cut
(build_coll_hostcut + _BudgetCB, first validated in mpc_testbed_coll_budget.py): a CasADi
iteration_callback compares wall-clock against the tick start at every IPOPT iteration and aborts
with the current iterate -- the same semantics as the other solvers' `if k > 1 and elapsed >= budget:
break` (>=1 iteration guaranteed, cut at iteration boundaries).  The verified mpc_solvers.build_coll
(ipopt.max_wall_time only, which misses the CasADi marshalling and is checked on IPOPT's schedule)
remains available via run_coll_mpc but is NOT used here.  Residual overshoot is one IPOPT iteration
(~4-5 ms at N=80) -- the same granularity rule as every other method, with dearer iterations.

USAGE
    python mpc_testbed.py             # PMP, DDP, collocation, M1_v2, MPPI x2; 20 ms budget
    python mpc_testbed.py --smoke     # 60 cycles
    python mpc_testbed.py --nobudget
    python mpc_testbed.py --no-mppi    # skip both MPPI runs (no CUDA GPU)
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from types import SimpleNamespace

import casadi as ca
import jax
import numpy as np

import mpc_core as C
import mpc_solvers as S


# ======================================================================================================
#  PIECEWISE MOVING-OBSTACLE MOTION (v6).  Bodies may carry a per-body stop time P.mobs.tstop[k]:
#      center_k(t) = c0_k + vel_k * min(t, tstop[k])       (tstop = inf -> old linear behaviour)
#  Installed by REPLACING mpc_core.mobs_center: every closed-loop consumer (ref_window, the true-cost
#  integration in apply_step, viz logging) resolves mobs_center through the mpc_core module at call
#  time, so this one assignment covers them all without touching mpc_core.py.  The standalone
#  animation scripts compute c0 + vel*t directly and need adapting for t > tstop (known caveat).
# ======================================================================================================
_mobs_center_linear = C.mobs_center


def _mobs_center_piecewise(mobs, t):
    ts = getattr(mobs, "tstop", None)
    if ts is None:
        return _mobs_center_linear(mobs, t)
    return mobs.c0 + mobs.vel * np.minimum(t, ts)


C.mobs_center = _mobs_center_piecewise


# ======================================================================================================
#  RECOMPILE DETECTOR.  Cycles whose measured time includes a JAX (re)compilation are TAGGED and
#  EXCLUDED from the summary's mean/max/p99 (together with the cycle-1 cold start): all compilation
#  is assumed doable in advance, so compile time is not solver work.  Detection is direct, not a
#  time threshold: jax_log_compiles makes JAX emit a log record on every compilation, and the handler
#  below raises a flag; each runner resets the flag at the tick start and reads it right after the
#  timed section.  Non-JAX ticks (collocation/IPOPT, MPPI's CuPy kernel) never fire it.
# ======================================================================================================
class _CompileFlag(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.fired = False

    def emit(self, record):
        self.fired = True


CF = _CompileFlag()
jax.config.update("jax_log_compiles", True)
_jlog = logging.getLogger("jax")
_jlog.addHandler(CF)
_jlog.propagate = False                                                # keep compile logs off stderr


# ======================================================================================================
#  PER-CYCLE ANIMATION LOG  (port of viz_init/viz_log/viz_pack).  Untimed: never inside a tic/toc block.
# ======================================================================================================
def viz_init():
    return SimpleNamespace(t=[], x=[], Xpred=[], Xrefw=[], mc=[], extra=[])


def viz_log(VZ, t, x, Xpred, Xrefw, M, extra=None):
    VZ.t.append(t)
    VZ.x.append(np.asarray(x).copy())
    VZ.Xpred.append(viz_thin(np.asarray(Xpred)[M.posidx, :]))
    VZ.Xrefw.append(viz_thin(np.asarray(Xrefw)[M.posidx, :]))
    VZ.mc.append(C.mobs_center(M.mobs, t))
    VZ.extra.append(extra)


def viz_thin(Q, npts=60):
    """~60 points per horizon keeps the results file small (350 cycles x 5 methods)."""
    ns = Q.shape[1]
    if ns <= npts:
        return Q
    idx = np.unique(np.concatenate([np.arange(0, ns, int(np.ceil(ns / npts))), [ns - 1]]))
    return Q[:, idx]


def viz_pack(res, VZ):
    res.tlog = np.array(VZ.t)
    res.xlog = np.array(VZ.x).T
    res.Xpred, res.Xrefw, res.mclog = VZ.Xpred, VZ.Xrefw, VZ.mc
    res.extra = VZ.extra
    return res


# ======================================================================================================
#  SHARED CLOSED-LOOP MACHINERY  (ports of apply_step / mk_res / min_clear)
# ======================================================================================================
def apply_step(x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX):
    """Apply u for napply fine steps and integrate the TRUE cost (tracking + true obstacle penalties at
    the obstacles' ACTUAL positions -- not the static-at-current prediction the solver saw)."""
    Uapp.append(np.asarray(u).copy())
    if p0 is not None:
        Pclog.append(np.asarray(p0).copy())
    dxk = x - Xref[:, 0]
    Jreal = Jreal + (dxk @ M.Q @ dxk + u @ M.R @ u) * P.dt_apply
    for s in range(1, P.napply + 1):
        x = np.asarray(JX.rk4_state(x, u, P.dt))
        X.append(x.copy())
        mc = C.mobs_center(M.mobs, t + s * P.dt)                       # TRUE position, not the prediction
        pen1, _ = JX.obstacle(x[M.posidx])
        pen2, _ = JX.mobs_pen(x[M.posidx], mc)
        Jpen = Jpen + (float(pen1) + float(pen2)) * P.dt
    t = t + P.napply * P.dt
    stop = np.linalg.norm(x[M.posidx] - P.pgoal[:len(M.posidx)]) < P.reachtol
    return x, X, Jreal, Jpen, Uapp, Pclog, 0, stop, t


def mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M):
    r = SimpleNamespace()
    r.X = np.array(X).T
    r.tcyc = np.array(tcyc)
    r.Jreal, r.Jpen, r.Jtrue = Jreal, Jpen, Jreal + Jpen
    r.reached = reached
    r.U = np.array(Uapp).T if Uapp else np.zeros((M.m, 0))
    r.Pc = np.array(Pclog).T if Pclog else np.zeros((M.n, 0))
    r.ncyc = len(tcyc)
    r.clear = min_clear(r.X, M)
    return r


def min_clear(X, M):
    c = np.inf
    for i in range(M.obs.center.shape[1]):
        c = min(c, np.linalg.norm(X[M.posidx, :] - M.obs.center[:, i:i + 1], axis=0).min())
    return float(c)


def runner(name, color, style, res, N=0):
    """N rides along so the summary and the animation HUD need no name->horizon lookup table
    (required now that MPPI appears twice with different horizons)."""
    return SimpleNamespace(name=name, color=color, style=style, res=res, N=N)


# ======================================================================================================
#  CLOSED-LOOP RUNNERS
# ======================================================================================================
def run_pmp_mpc(M, N, P, path, Sh, JX):
    x = M.x0.copy()
    Sh.warm(x, C.ref_window(x, N, path, M, P, 2, 0.0))                 # compile OUTSIDE the timed loop
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    seed = np.zeros(M.n)
    t = 0.0
    VZ = viz_init()
    LM, NR, RC = [], [], []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xr2 = C.ref_window(x, N, path, M, P, 2, t)
        Xref = Xr2[:M.n, ::2]
        p0 = S.resnewton_solve(Sh, seed, x, Xr2, P, tc)
        u = np.asarray(JX.ustar(x, p0))
        Pc = np.asarray(Sh.roll_costate(p0, x, Xr2))
        seed = Pc[:, P.napply]
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        # ---- lmin diagnostic at the APPLIED iterate (logging only; outside the timed section) ----
        rd, _, Jrd, Sxd = Sh.all(p0, x, Xr2)
        Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
        LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
        NR.append(float(np.linalg.norm(np.asarray(rd))))
        if c % 10 == 1:
            print(f"PMP  c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f} |res|{NR[-1]:9.2e} lmin{LM[-1]:10.2e}")
        viz_log(VZ, t, x, Sh.roll_pred(p0, x, Xr2), Xref, M)
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step(
            x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    res = mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    res.lmin, res.nres = np.array(LM), np.array(NR)
    res.recompile = np.array(RC)
    return viz_pack(res, VZ)


# ======================================================================================================
#  M1_v2  --  chlqn_solve  (paper Algorithm 1: Certified Hopf-Lax Quasi-Newton)
#
#  The ALGORITHM itself lives in mpc_solvers.py beside the baselines it is compared against (moved
#  there 2026-10-05, where it replaced the legacy guarded_rn_fast); only the closed-loop runner below
#  is testbed business.  The names are re-bound at module level because complexity_vs_horizon.py
#  reaches for them as T.chlqn_solve / T.M1V2_* / T.ensure_jbatch.
# ======================================================================================================
from mpc_solvers import (chlqn_solve, ensure_jbatch,                 # noqa: E402  (re-export)
                             M1V2_ETA_S, M1V2_RIDGE, M1V2_DELTA, M1V2_ESC_ALPHAS)


def run_m1v2_mpc(M, N, P, path, Sh, JX):
    """Closed-loop runner for M1_v2 -- clone of run_m1_mpc calling chlqn_solve.  The warm start
    (roll_costate over napply sub-steps, seed = Pc[:, napply]) IS the paper's Algorithm 2: integrate
    the canonical system over dt_apply and take v <- p(t + dt_apply)."""
    x = M.x0.copy()
    Xr0 = C.ref_window(x, N, path, M, P, 2, 0.0)
    Sh.warm(x, Xr0)                                                    # compile OUTSIDE the timed loop
    Jb = ensure_jbatch(Sh)                                             # + both batch shapes used by
    Jb(np.zeros((2, M.n)), x, Xr0)                                     #   the kick comparison (B=2)
    Jb(np.zeros((7, M.n)), x, Xr0)                                     #   and the escape alphas (B=7)
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    seed = np.zeros(M.n)
    t = 0.0
    VZ = viz_init()
    PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
    ITS, NDESC, NESC, NKICK, NEIG, LM, NR, RC = [], [], [], [], [], [], [], []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xr2 = C.ref_window(x, N, path, M, P, 2, t)
        Xref = Xr2[:M.n, ::2]
        p0, itk, infok = chlqn_solve(Sh, seed, x, Xr2, PP, P.M1_epsc, M1V2_ETA_S, P.M1_alpha,
                                     M1V2_DELTA, tc, P.budget)
        u = np.asarray(JX.ustar(x, p0))
        ITS.append(itk); NDESC.append(infok.ndesc); NESC.append(infok.nesc)
        NKICK.append(infok.nkick); NEIG.append(infok.neig)
        Pc = np.asarray(Sh.roll_costate(p0, x, Xr2))
        seed = Pc[:, P.napply]
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        rd, _, Jrd, Sxd = Sh.all(p0, x, Xr2)
        Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
        LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
        NR.append(float(np.linalg.norm(np.asarray(rd))))
        if c % 10 == 1:
            print(f"M1v2 c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f} |res|{NR[-1]:9.2e} lmin{LM[-1]:10.2e}")
        viz_log(VZ, t, x, Sh.roll_pred(p0, x, Xr2), Xref, M)
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step(
            x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    res = mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    res.lmin, res.nres = np.array(LM), np.array(NR)
    res.recompile = np.array(RC)
    res.m1_its, res.m1_ndesc = np.array(ITS), np.array(NDESC)
    res.m1_nesc, res.m1_nkick, res.m1_neig = np.array(NESC), np.array(NKICK), np.array(NEIG)
    print(f"M1v2 iter stats: mean {np.mean(ITS):.2f} iters/cycle (max {max(ITS)}) | descent {sum(NDESC)} "
          f"| escape {sum(NESC)} | kicks {sum(NKICK)} | eigen reads {sum(NEIG)} over {len(ITS)} cycles")
    return viz_pack(res, VZ)


def run_ilqr_mpc(M, N, P, path, JX):
    IL = S.build_ilqr(M, N, P.dt, JX)
    x = M.x0.copy()
    IL.warm(x, C.ref_window(x, N, path, M, P, 1, 0.0)[:M.n, :],
            C.mobs_center(M.mobs, 0.0))                                # compile OUTSIDE the timed loop
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    U = None
    t = 0.0
    VZ = viz_init()
    RC = []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xref = C.ref_window(x, N, path, M, P, 1, t)[:M.n, :]
        mc = C.mobs_center(M.mobs, t)                                  # iLQR sees static + moving-at-current
        Uinit = None if U is None else S.shift_U(U, P.napply)
        p0, U, _ = S.ilqr_run(IL, x, Xref, mc, Uinit, P, tc)
        u = np.asarray(JX.ustar(x, p0))
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        Xp, _ = IL.rollout_cost(x, U, Xref, mc)
        viz_log(VZ, t, x, np.asarray(Xp), Xref, M)
        if c % 10 == 1:
            print(f"iLQR c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f}")
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step(
            x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    res = mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    res.recompile = np.array(RC)
    return viz_pack(res, VZ)


# DDP +y INITIALIZATION NUDGE -- RESTORED (user-directed, 2026-07-29).  DDP's wall-commitment
# side is BISTABLE under budget-cut timing noise: 5 measured runs committed +y, 1 run flipped -y
# and slipped through the gauntlet side.  The baseline therefore keeps the SAME side-selection
# bias it has carried since v6 (+2 cm solver-visible state shift in the commitment window) --
# the bias that routed it to the viable branch on the v6-v8 courses.  On the mirrored course the
# identical hand-fed bias commits it to the priced branch: externally-supplied side selection
# cannot adapt to the course; the certified branch comparison can.  Set to 0 for plain DDP.
DDP_YSHIFT = 0.02      # [m] +y shift of the STATE HANDED TO THE SOLVER inside the window only
DDP_UBIAS = 0.0


def run_ddp_mpc(M, N, P, path, JX):
    """Full second-order DDP (mirrors run_ilqr_mpc; build_ddp/ddp_run add the dynamics+barrier Hessian)."""
    DP = S.build_ddp(M, N, P.dt, JX)
    x = M.x0.copy()
    DP.warm(x, C.ref_window(x, N, path, M, P, 1, 0.0)[:M.n, :], C.mobs_center(M.mobs, 0.0))
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    U = None
    t = 0.0
    VZ = viz_init()
    RC = []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xref = C.ref_window(x, N, path, M, P, 1, t)[:M.n, :]
        mc = C.mobs_center(M.mobs, t)
        # dodge-commitment window (v6-measured): wall shell within the horizon's reach (+0.6 shell
        # margin), state still on-axis -- the solver sees a +y-shifted state inside it.
        gap = (mc[0, 0] - M.mobs.ax[0]) - x[0]
        # |y| < 0.05: the v6-measured commitment window.  (A widened 0.2 bound that kept nudging
        # through the grind was tested and REJECTED: the persistent bias cannot overpower a
        # decisive late re-solve and destabilises -y excursions.  With 0.05 the INITIAL dodge is
        # +y in every run and the TRAPPED-by-16s verdict is robust: observed late -y crawls start
        # at t >= 10.5 and would need ~7 s more to reach.)
        in_window = (0.0 < gap < N * P.dt * max(x[4], 0.1) + 0.6) and (abs(x[1]) < 0.05)
        xs = x.copy()
        if in_window and DDP_YSHIFT != 0.0:
            xs[1] += DDP_YSHIFT
        Uinit = None if U is None else S.shift_U(U, P.napply)
        p0, U, _ = S.ddp_run(DP, xs, Xref, mc, Uinit, P, tc)
        u = np.asarray(JX.ustar(x, p0))
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        Xp, _ = DP.rollout_cost(x, U, Xref, mc)
        viz_log(VZ, t, x, np.asarray(Xp), Xref, M)
        if c % 10 == 1:
            print(f"DDP  c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f}")
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step(
            x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    res = mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    res.recompile = np.array(RC)
    return viz_pack(res, VZ)


def run_coll_mpc(M, N, P, path, JX):
    collf = S.build_coll(M, N, P.dt, P.budget)
    x = M.x0.copy()
    collf(x, C.ref_window(x, N, path, M, P, 1, 0.0), None)             # warm IPOPT (result discarded)
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    w = None
    t = 0.0
    VZ = viz_init()
    for c in range(1, P.maxcyc + 1):
        tc = time.perf_counter()
        Xr1 = C.ref_window(x, N, path, M, P, 1, t)
        Xref = Xr1[:M.n, :]
        _, u, wsol = collf(x, Xr1, w)
        w = S.shift_coll(wsol, M.n, M.m, N, P.napply)
        tcyc.append(time.perf_counter() - tc)
        Xp = wsol[:M.n * (N + 1)].reshape((M.n, N + 1), order="F")     # solver's OWN predicted states
        viz_log(VZ, t, x, Xp, Xref, M)
        if c % 10 == 1:
            print(f"Coll c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f}")
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step(
            x, u, None, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    return viz_pack(mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M), VZ)


# ======================================================================================================
#  COLLOCATION with HOST-STYLE BUDGET CUT (iteration_callback) -- see BUDGET-ENFORCEMENT NOTE above
# ======================================================================================================
class _BudgetCB(ca.Callback):
    """IPOPT iteration callback: abort (return nonzero) once wall-clock since the tick start exceeds
    the budget.  `t0`/`deadline`/`calls` are (re)set by the caller before every solve.  `calls > 1`
    mirrors the other solvers' `k > 1` guard: at least one iteration is always allowed."""

    def __init__(self, name, nx, ng, npar):
        ca.Callback.__init__(self)
        self.nx, self.ng, self.npar = nx, ng, npar
        self.t0, self.deadline, self.calls = 0.0, np.inf, 0
        self.construct(name, {})

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        nm = ca.nlpsol_out(i)
        if nm == "f":
            return ca.Sparsity.scalar()
        if nm in ("x", "lam_x"):
            return ca.Sparsity.dense(self.nx, 1)
        if nm in ("g", "lam_g"):
            return ca.Sparsity.dense(self.ng, 1)
        if nm == "lam_p":
            return ca.Sparsity.dense(self.npar, 1)
        return ca.Sparsity(0, 0)

    def eval(self, arg):
        self.calls += 1
        if self.calls > 1 and (time.perf_counter() - self.t0) >= self.deadline:
            return [1.0]                                              # nonzero -> User_Requested_Stop
        return [0.0]


def build_coll_hostcut(M, N, dt, budget):
    """Identical NLP to mpc_solvers.build_coll (same cost, constraints, options), with the budget
    enforced by _BudgetCB against the tick start `tc` instead of ipopt.max_wall_time alone.
    `call(xc, Xrv, W0, tc)` -- the extra tc is the only signature change."""
    n, m = M.n, M.m
    nm = M.mobs.c0.shape[1]
    Q, QT, R = ca.DM(M.Q), ca.DM(M.QT), ca.DM(M.R)

    def dyn(x, u):
        th = x[3]
        return ca.vertcat(x[4] * ca.cos(th), x[4] * ca.sin(th), x[6], x[5], u[0], u[1], u[2])

    def rk4(x, u):
        k1 = dyn(x, u); k2 = dyn(x + dt / 2 * k1, u)
        k3 = dyn(x + dt / 2 * k2, u); k4 = dyn(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def barrier(pos, Cc, ax, ay, az, W, w, eps):
        pen = 0
        for i in range(Cc.shape[1]):
            dd = pos - Cc[:, i]
            rho = ca.sqrt((dd[0] / ax[i]) ** 2 + (dd[1] / ay[i]) ** 2 + (dd[2] / az[i]) ** 2 + eps)
            pen = pen + W[i] * 0.5 * (1 - ca.tanh((rho - 1) / w[i]))
        return pen

    xk, uk = ca.SX.sym("x", n), ca.SX.sym("u", m)
    FP = ca.Function("FP", [xk, uk], [rk4(xk, uk)])
    par = ca.MX.sym("par", n)
    Xr = ca.MX.sym("Xr", n + 3 * nm, N + 1)
    X = ca.MX.sym("X", n, N + 1)
    U = ca.MX.sym("U", m, N)
    Xn = FP.map(N)(X[:, :N], U)
    J = 0
    o = M.obs
    for k in range(N):
        dxk = X[:, k] - Xr[:n, k]
        mck = ca.reshape(Xr[n:, k], 3, nm)
        pen = (barrier(X[:3, k], ca.DM(o.center), o.ax, o.ay, o.az, o.W, o.w, o.eps)
               + barrier(X[:3, k], mck, M.mobs.ax, M.mobs.ay, M.mobs.az, M.mobs.W, M.mobs.w, M.mobs.eps))
        J = J + (dxk.T @ Q @ dxk + U[:, k].T @ R @ U[:, k] + pen) * dt
    dxT = X[:, N] - Xr[:n, N]
    J = J + dxT.T @ QT @ dxT
    D = X[:, 1:] - Xn
    G = ca.vertcat(X[:, 0] - par, ca.reshape(D, -1, 1))

    nx = n * (N + 1) + m * N
    nG = n * (N + 1)
    npar = n + (n + 3 * nm) * (N + 1)
    cb = _BudgetCB("budget_cb", nx, nG, npar)
    opts = {"ipopt.max_iter": 100, "ipopt.tol": 1e-8, "ipopt.print_level": 0,
            "print_time": 0, "error_on_fail": False,
            "iteration_callback": cb}
    if np.isfinite(budget):                                           # backstop, as in build_coll
        opts["ipopt.max_wall_time"] = float(budget)
    s = ca.nlpsol("s", "ipopt", {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
                                 "f": J, "g": G, "p": ca.vertcat(par, ca.reshape(Xr, -1, 1))}, opts)
    lbx = np.concatenate([-np.inf * np.ones(n * (N + 1)), np.tile(-M.umax, N)])
    ubx = np.concatenate([np.inf * np.ones(n * (N + 1)), np.tile(M.umax, N)])

    def call(xc, Xrv, W0, tc):
        if W0 is None:
            X0 = np.tile(xc.reshape(-1, 1), (1, N + 1))
            W0 = np.concatenate([X0.flatten(order="F"), np.zeros(m * N)])
        cb.t0, cb.deadline, cb.calls = tc, budget, 0                  # host-style cut: elapsed since tc
        r = s(x0=W0, p=np.concatenate([xc, Xrv.flatten(order="F")]),
              lbg=np.zeros(nG), ubg=np.zeros(nG), lbx=lbx, ubx=ubx)
        w = np.asarray(r["x"]).ravel()
        Uo = w[n * (N + 1):].reshape((m, N), order="F")
        return float(r["f"]), Uo[:, 0], w

    call.cb = cb                                                      # keep the callback alive
    call.solver = s                                                   # expose stats() for diagnostics
    return call


def run_coll_hostcut_mpc(M, N, P, path, JX):
    """Clone of run_coll_mpc using build_coll_hostcut: the tick start tc is passed into the solve so
    the budget meters the ENTIRE tick, exactly like the other methods."""
    collf = build_coll_hostcut(M, N, P.dt, P.budget)
    x = M.x0.copy()
    collf(x, C.ref_window(x, N, path, M, P, 1, 0.0), None, time.perf_counter())  # warm IPOPT (discarded)
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    w = None
    t = 0.0
    VZ = viz_init()
    RC = []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xr1 = C.ref_window(x, N, path, M, P, 1, t)
        Xref = Xr1[:M.n, :]
        _, u, wsol = collf(x, Xr1, w, tc)
        w = S.shift_coll(wsol, M.n, M.m, N, P.napply)
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        Xp = wsol[:M.n * (N + 1)].reshape((M.n, N + 1), order="F")     # solver's OWN predicted states
        viz_log(VZ, t, x, Xp, Xref, M)
        if c % 10 == 1:
            print(f"Coll c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f}")
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step(
            x, u, None, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    res = mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    res.recompile = np.array(RC)
    return viz_pack(res, VZ)


def run_mppi_mpc(M, N, P, path, JX, K=None, label="MPPI"):
    """MPPI on the GPU. Same cost, same warm-start-by-shifting discipline as the other methods.

    v7: (N, K) are explicit arguments because MPPI runs twice -- the user-validated 280/16384 pin
    and the 300/20480 whole-budget-in-one-update configuration (K=None falls back to P.K_mppi).
    """
    import mpc_mppi as MPPI
    import jax
    from jax import lax
    import jax.numpy as jnp

    if K is None:
        K = P.K_mppi

    class _MPPIWideLam(MPPI.MPPI):
        """v9 fix (MEASURED, results_mppi440_tune.pkl): the lambda bisection's upper bracket
        (log10 lam <= 6) is too narrow for long-horizon cost spreads -- at N=440 dynamically
        diverged samples spread the costs so far that even lam = 1e6 cannot reach the 5% ESS
        target, the weights collapse to ONE sample (random shooting, ESS min 1.0) and the run
        diverges (J 137).  Widening to 9 restores the target (ESS 4.9%, completes at 14.78,
        J 161.7, bit-reproducible).  For configurations whose target is reachable below 1e6 the
        wider bracket is a NO-OP (the bisection converges to the same lambda), verified on
        20480/230.  Single-knob numerical-range fix: sigma/alpha/ess_target/sg stay published."""

        def _tune_lambda(self, S, rho):
            import cupy as cp
            nsub = min(self.K, 2048)
            d_h = cp.asnumpy(S[:nsub] - rho)
            lo, hi = -6.0, 9.0
            target = self.ess_target * nsub
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                w = np.exp(-d_h / (10.0 ** mid))
                w = w / w.sum()
                ess = 1.0 / np.sum(w ** 2)
                if ess < target:
                    lo = mid
                else:
                    hi = mid
            self.lam = 10.0 ** (0.5 * (lo + hi))
            self.gamma = self.lam * (1.0 - self.alpha)
            self.d_gsig = cp.asarray(np.ascontiguousarray(self.gamma / self.sigma ** 2,
                                                          dtype=np.float64))
            return self.lam

    ctrl = _MPPIWideLam(M, P, N=N, K=K, lam=P.lam_mppi, alpha=P.alpha_mppi,
                        iters=P.iters_mppi, seed=0, ess_target=P.ess_target_mppi)

    @jax.jit
    def roll_many(x0, Us):
        """Roll out a batch of control sequences (animation cloud only; untimed)."""
        def one(U):
            def body(x, u):
                x2 = JX.rk4_state(x, u, P.dt)
                return x2, x2
            _, Xs = lax.scan(body, x0, U.T)
            return jnp.vstack([x0[None, :], Xs]).T
        return jax.vmap(one)(Us)

    x = M.x0.copy()
    Xref0 = C.ref_window(x, N, path, M, P, 1, 0.0)[:M.n, :]
    ctrl.warm(x, Xref0, C.mobs_center(M.mobs, 0.0))                    # NVRTC compile OUTSIDE the loop
    roll_many(x, np.zeros((2, M.m, N)))                                # and the JAX cloud roller too

    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    t = 0.0
    VZ = viz_init()
    ESS, RC = [], []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xref = C.ref_window(x, N, path, M, P, 1, t)[:M.n, :]
        mc = C.mobs_center(M.mobs, t)                                  # static-at-current, like everyone
        u, U, ess = ctrl.solve(x, Xref, mc)
        ctrl.shift(P.napply)                                           # warm start for the next cycle
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        ESS.append(ess)
        # ---- untimed logging ----
        Xp = np.asarray(roll_many(x, U[None, :, :])[0])                # nominal predicted trajectory
        cloud = np.asarray(roll_many(x, ctrl.top_sample_controls(24)))
        cloud = [viz_thin(cl[M.posidx, :], 40) for cl in cloud]
        if c % 10 == 1:
            print(f"{label:<4s} c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f} ESS{ess:8.1f}/{K}"
                  f" ({100 * ess / K:4.1f}%)")
        viz_log(VZ, t, x, Xp, Xref, M, extra=cloud)
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step(
            x, u, None, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    res = mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    res.ess = np.array(ESS)
    res.recompile = np.array(RC)
    ess_a = np.array(ESS)
    print(f"{label} ESS: mean {ess_a.mean():.1f}/{K} ({100 * ess_a.mean() / K:.1f}%), "
          f"min {ess_a.min():.1f} -- a collapse to single digits means sampling noise, not optimisation")
    return viz_pack(res, VZ)


# ======================================================================================================
#  MAIN
# ======================================================================================================
def main(argv):
    smoke = "--smoke" in argv
    nobudget = "--nobudget" in argv
    use_mpath = "--matlab-path" in argv

    P = C.make_P(smoke=smoke)
    import mpc_tuned_params
    P = mpc_tuned_params.apply(P)      # unified +y route + hot-companion fork; DDP N=300 -- ALL tuning lives in that module
    if nobudget:
        P.budget = np.inf
        print("*** NOBUDGET (budget=inf; deterministic, for parity) ***")
    if smoke:
        print(f"*** SMOKE (maxcyc={P.maxcyc}) ***")

    M = C.model_scn(P)
    JX = C.build_jax(M)

    tp = time.perf_counter()
    if use_mpath:
        path = C.path_from_matlab("results_matlab_nobudget.mat")
        print("using MATLAB's RRT path (parity mode)")
    else:
        path = C.plan_path(P.p0, P.pgoal, P.obs_plan, P)
    tp = time.perf_counter() - tp

    Sh_pmp = C.build_ss(M, P.N_pmp, P.dt, JX, napply=P.napply)     # napply sizes the warm-start roll
    Sh_m1 = C.build_ss(M, P.N_m1, P.dt, JX, napply=P.napply)       # M1_v2 uses M1's tuned horizon

    print(f"=== mpc_testbed_v10 (python): unified +y route + hot-companion fork; hard {1e3 * P.budget:.0f} ms cut ==="
          f"  dt={P.dt}, N[pmp,m1v2,ddp,coll]=[{P.N_pmp} {P.N_m1} {P.N_ddp} {P.N_coll}],"
          f" MPPI ({P.N_mppi},{P.K_mppi}) + ({P.N_mppi2},{P.K_mppi2}),"
          f" eta_s={M1V2_ETA_S}, ridge={M1V2_RIDGE}, delta={M1V2_DELTA}")
    print(f"RRT path: {path.pts.shape[1]} pts, len {path.len:.2f}, max|y|={np.abs(path.pts[1]).max():.3f}, "
          f"planned {tp:.2f}s")

    R = []
    tt = time.perf_counter()
    R.append(runner("PMP", (.85, .15, .15), "-", run_pmp_mpc(M, P.N_pmp, P, path, Sh_pmp, JX), P.N_pmp))
    print(f"PMP done         ({time.perf_counter() - tt:.0f} s)")
    tt = time.perf_counter()
    R.append(runner("DDP", (.90, .45, .0), "-", run_ddp_mpc(M, P.N_ddp, P, path, JX), P.N_ddp))
    print(f"DDP done         ({time.perf_counter() - tt:.0f} s)")
    tt = time.perf_counter()
    R.append(runner("Collocation", (0, .45, .85), "-", run_coll_hostcut_mpc(M, P.N_coll, P, path, JX),
                    P.N_coll))
    print(f"Collocation done ({time.perf_counter() - tt:.0f} s)  [host-style budget cut]")
    tt = time.perf_counter()
    R.append(runner("M1_v2", (.10, .55, .20), "-", run_m1v2_mpc(M, P.N_m1, P, path, Sh_m1, JX), P.N_m1))
    print(f"M1_v2 done       ({time.perf_counter() - tt:.0f} s)")
    if "--nompppi" not in argv and "--no-mppi" not in argv:
        tt = time.perf_counter()
        R.append(runner("MPPI", (.60, .20, .70), "-",
                        run_mppi_mpc(M, P.N_mppi, P, path, JX, K=P.K_mppi), P.N_mppi))
        print(f"MPPI done        ({time.perf_counter() - tt:.0f} s)")
        tt = time.perf_counter()
        R.append(runner("MPPI-20480", (.40, .10, .50), "-",
                        run_mppi_mpc(M, P.N_mppi2, P, path, JX, K=P.K_mppi2, label="MP2"), P.N_mppi2))
        print(f"MPPI-20480 done  ({time.perf_counter() - tt:.0f} s)")

    summary(R, P)
    # ---- explicit obstacle-motion record for plotting scripts (self-describing: consumers need not
    #      know the piecewise-motion law).  centers_t[:, :, k] = true moving-obstacle centres at
    #      t = k*dt_apply (the animation frame grid), evaluated through the tstop-aware mobs_center. ----
    tgrid = np.arange(P.maxcyc + 1) * P.dt_apply
    obs_motion = dict(
        t=tgrid,
        centers_t=np.stack([C.mobs_center(M.mobs, tk) for tk in tgrid], axis=-1),
        moving=dict(c0=M.mobs.c0.copy(), vel=M.mobs.vel.copy(),
                    tstop=np.array(getattr(M.mobs, "tstop", np.full(M.mobs.c0.shape[1], np.inf))),
                    ax=M.mobs.ax.copy(), ay=M.mobs.ay.copy(), az=M.mobs.az.copy(),
                    W=M.mobs.W.copy(), w=M.mobs.w.copy(), eps=M.mobs.eps),
        static=dict(center=M.obs.center.copy(), rphys=M.obs.rphys.copy(),
                    ax=M.obs.ax.copy(), ay=M.obs.ay.copy(), az=M.obs.az.copy(),
                    W=M.obs.W.copy(), w=M.obs.w.copy(), eps=M.obs.eps),
    )
    out = "results_testbed_v10_nobudget.pkl" if nobudget else "results_testbed.pkl"
    with open(out, "wb") as f:
        pickle.dump({"R": R, "P": P, "M": M, "path": path, "obs_motion": obs_motion}, f)
    print(f"\nsaved {out}")
    return 0


def summary(R, P):
    """Headline table.

    Jtrue is split into its two parts because they answer different questions. J_track is the quadratic
    tracking cost (how well the RRT reference was followed). J_obs is the integrated TRUE obstacle
    penalty, evaluated at the obstacles' ACTUAL positions -- not the static-at-current prediction the
    solver optimised against. The split is what distinguishes "detoured wide but safely" from "sat in a
    barrier": a trapped method grinds up J_obs while going nowhere.
    """
    print(f"\n{'method':<13}{'N':>5}{'Tp[s]':>7}{'mean ms':>9}{'max ms':>8}{'p99 ms':>8}"
          f"{'J_track':>10}{'J_obs':>11}{'J_total':>10}{'reached':>9}{'t_reach':>9}{'cyc':>6}{'rcmp':>6}")
    print("-" * 111)
    for r in R:
        # Timing excludes the cycle-1 cold start (as in MATLAB) AND every cycle tagged by the
        # recompile detector (all compilation is assumed doable in advance).  `rcmp` counts the
        # recompile cycles excluded beyond cycle 1.
        tt = np.asarray(r.res.tcyc)
        keep = np.ones(tt.size, dtype=bool)
        keep[0] = False
        nrc = 0
        rc = getattr(r.res, "recompile", None)
        if rc is not None and rc.size == tt.size:
            nrc = int(np.sum(rc[1:]))
            keep &= ~rc
        if keep.any():
            tt = tt[keep]
        Nm = getattr(r, "N", 0)                                        # v7: runners carry their own N
        tr = f"{r.res.tlog[-1] + P.dt_apply:.2f}" if r.res.reached else "--"
        print(f"{r.name:<13}{Nm:>5d}{Nm * P.dt:>7.2f}{1e3 * tt.mean():>9.2f}{1e3 * tt.max():>8.2f}"
              f"{1e3 * np.percentile(tt, 99):>8.2f}{r.res.Jreal:>10.3f}{r.res.Jpen:>11.3f}"
              f"{r.res.Jtrue:>10.3f}{('yes' if r.res.reached else 'TRAPPED'):>9}{tr:>9}{r.res.ncyc:>6d}"
              f"{nrc:>6d}")
    print("timing: cycle 1 (cold start) and 'rcmp' recompile-tagged cycles excluded from mean/max/p99")


if __name__ == "__main__":
    sys.exit(main(sys.argv))

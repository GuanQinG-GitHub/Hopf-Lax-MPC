"""
mpc_testbed_N_sweep.py -- closed-loop horizon sweep WITHOUT a real-time budget.

Same scenario, obstacles, warm starts, DDP nudge, Hopf_Lax_MPC thresholds, MPPI lambda bisection and
800-cycle / stop-at-goal protocol as mpc_testbed.py main() -- with three changes only:
  * the per-cycle wall-clock budget is removed (P.budget = inf, or --cycle-timeout as a safety net),
  * the iteration caps are raised (--cap, default 1000: P.lm_maxit, P.ilqr_Kmax, IPOPT max_iter),
  * the prediction horizon N is swept per method (--Ns), instead of one tuned N per method.
For every (method, N) the closed-loop outcome (J_track / J_obs / J_total, reached, t_reach), the
per-cycle computation time and the per-cycle iteration details (count, termination status, final
residual, method-specific counters) are recorded.  Results merge into results_N_sweep.pkl
(one entry per (method, N); saved after every finished run; reruns skip finished keys unless --redo).

The runners below are instrumented COPIES of the mpc_testbed.py runners: the originals discard the
iteration information at the solver call, hard-code IPOPT max_iter=100, and keep the MPPI lambda
class local.  mpc_testbed.py / mpc_solvers.py are imported, never modified.

USAGE
    python mpc_testbed_N_sweep.py                                 # all six methods, N=50..300
    python mpc_testbed_N_sweep.py --methods PMP_Shooting,Hopf_Lax_MPC --Ns 50,100
    python mpc_testbed_N_sweep.py --smoke                         # 60 cycles -> results_N_sweep_smoke.pkl
    python mpc_testbed_N_sweep.py --selftest                      # counted copies == originals (bitwise)
    python mpc_testbed_N_sweep.py --cycle-timeout 30              # safety net; cuts show up as 'timeout'
Methods: PMP_Shooting, DDP, Collocation_IPOPT, Hopf_Lax_MPC, MPPI (K = P.K_mppi = 12288), MPPI_<K> (e.g. MPPI_20480),
         Vanilla_Hopf_Lax = damped Newton on the Hopf-Lax objective with the EXACT Hessian (mpc_solvers;
                  gradient tolerance --vanilla-hopf-lax-tol, default 1e-3; not in the default method list)
"""
from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time
from types import SimpleNamespace

import casadi as ca
import numpy as np

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d) for _d in ("core", "closed_loop")]
import mpc_core as C
import mpc_solvers as S
import mpc_testbed as T                   # import-time: piecewise mobs_center + JAX compile-flag handler;
                                          # runners, _BudgetCB, DDP nudge constants (cHLQN itself lives in mpc_solvers)
import mpc_tuned_params

PKL = "results_N_sweep.pkl"
PKL_SMOKE = "results_N_sweep_smoke.pkl"
NS_DEFAULT = [50, 100, 150, 200, 250, 300]
METHODS_DEFAULT = ["PMP_Shooting", "DDP", "Collocation_IPOPT", "Hopf_Lax_MPC", "MPPI", "MPPI_20480"]
CAP_DEFAULT = 1000
CF = T.CF


# ======================================================================================================
#  COUNTED SOLVER COPIES
# ======================================================================================================
def pmp_shooting_solve_counted(Sh, p, xc, Xr, P, tc, stop=None):
    """Verbatim copy of mpc_solvers.pmp_shooting_solve (LM on res(p)=0) that also returns the iteration
    count, the final residual norm, the termination status and the number of damping trials.
    stop: optional shared stopping rule (core/shared_stop.SharedStop), as in the original."""
    n = p.size
    mu = 1e-6
    rr = np.asarray(Sh.res(p, xc, Xr))
    nr = np.linalg.norm(rr)
    it, ntrial, status = 0, 0, "cap"
    for k in range(1, P.lm_maxit + 1):
        if stop is None:
            if nr < P.lm_tol:
                status = "converged"
                break
        elif stop.check_costate(Sh, p, xc, Xr):                       # SHARED rule (untimed)
            status = "shared-stop"
            break
        if k > 1 and (time.perf_counter() - tc - (stop.t_spent if stop is not None else 0.0)) >= P.budget:
            status = "timeout"                                        # HARD wall-clock cut (>=1 step done)
            break
        Jr = np.asarray(Sh.Jr(p, xc, Xr))
        A = Jr.T @ Jr
        b = -Jr.T @ rr
        acc = False
        it += 1
        for _ in range(20):                                           # LM damping search
            ntrial += 1
            step = np.linalg.solve(A + mu * np.eye(n), b)
            pt = p + step
            rt = np.asarray(Sh.res(pt, xc, Xr))
            if np.linalg.norm(rt) < nr:
                p, rr, nr = pt, rt, np.linalg.norm(rt)
                mu = max(mu / 3, 1e-12)
                acc = True
                break
            mu = mu * 5
        if not acc:
            status = "stalled"
            break
    else:                                                             # loop exhausted: re-test tolerance
        if stop is None and nr < P.lm_tol:
            status = "converged"
    if not np.isfinite(nr):
        status = "diverged"
    return p, it, float(nr), status, ntrial, mu


def ddp_run_counted(IL, x, Xref, mc, Uinit, P, tc, mu0=1e-3, stop=None):
    """Verbatim copy of mpc_solvers.ddp_run that also returns termination status and counters.
    stop: optional shared stopping rule (core/shared_stop.SharedStop), as in the original."""
    N, m = IL.N, IL.m
    U = np.zeros((m, N)) if Uinit is None else Uinit
    X, J = IL.rollout_cost(x, U, Xref, mc)
    X, J = np.asarray(X), float(J)
    mu, dX = mu0, np.inf
    alphas = S.DDP_ALPHAS                                            # shared with warm() so forward() is compiled
    it = 0
    nbad = nrej = since_acc = 0
    dV1 = np.nan
    a_last, a_min = np.nan, np.nan
    status, conv_by = "cap", ""
    for it in range(1, P.ilqr_Kmax + 1):
        if it > 1 and (time.perf_counter() - tc - (stop.t_spent if stop is not None else 0.0)) >= P.budget:
            status = "timeout"
            break
        if stop is not None:
            if stop(U):                                               # SHARED rule (untimed; only on a new U)
                status, conv_by = "shared-stop", "dJ"
                break
            if since_acc >= 20:                                       # stall exit, as in mpc_solvers.ddp_run
                status = "stalled"
                break
        kff, Kk, dV1, Dv2, bad = IL.backward(X, U, Xref, mc, mu)
        if bool(bad):                                                 # Quu not PD -> raise damping, retry
            nbad += 1
            since_acc += 1
            mu = mu * 4
            continue
        dV1, Dv2 = float(dV1), float(Dv2)
        if stop is None and abs(dV1) < P.ilqr_tol:                    # expected reduction ~ 0 -> converged
            status, conv_by = "converged", "dV"
            break
        Xn, Un, Jn = IL.forward(x, X, U, kff, Kk, Xref, mc, alphas)
        Jn = np.asarray(Jn)
        improved = False
        a_acc = np.nan
        for i in range(len(alphas)):                                  # FIRST improving alpha (as iLQR)
            if Jn[i] < J:
                a = float(alphas[i])
                Xa, Ua = np.asarray(Xn[i]), np.asarray(Un[i])
                dX = np.linalg.norm(Xa - X) / (1 + np.linalg.norm(X))
                X, U, J = Xa, Ua, float(Jn[i])
                improved, a_acc = True, a
                break
        if improved:
            since_acc = 0
            a_last = a_acc
            a_min = a_acc if not np.isfinite(a_min) else min(a_min, a_acc)
            mu = mu * 2.0 if a_acc < 0.5 else max(mu * 0.7, 1e-9)
            if stop is None and dX < P.ilqr_tol:
                status, conv_by = "converged", "dX"
                break
        else:
            nrej += 1
            since_acc += 1
            mu = mu * 4
    if not np.isfinite(J):
        status = "diverged"
    elif status == "cap" and since_acc >= 50:
        status = "stalled"                                            # pure mu-crawl into the cap
    p0 = np.asarray(IL.costate0(X, U, Xref, mc))
    info = SimpleNamespace(status=status, conv_by=conv_by, nbad=nbad, nrej=nrej, n_eff=it - nbad,
                           mu=float(mu), dV1=float(dV1) if np.isfinite(dV1) else np.nan,
                           dX=float(dX) if np.isfinite(dX) else np.nan, a_last=a_last, a_min=a_min, J=J)
    return p0, U, it, info


def build_collocation_ipopt_nobudget(M, N, dt, budget, max_iter, stop=None):
    """Verbatim copy of mpc_testbed.build_collocation_ipopt_hostcut with ipopt.max_iter as a parameter
    (and the same optional shared stopping rule `stop`)."""
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
    cb = T._BudgetCB("budget_cb", nx, nG, npar)
    opts = {"ipopt.max_iter": int(max_iter), "ipopt.tol": 1e-8, "ipopt.print_level": 0,
            "print_time": 0, "error_on_fail": False,
            "iteration_callback": cb}
    if stop is not None:                                              # shared rule, as in the testbed copy
        cb.stop, cb.ulay = stop, (n, N, m)
        opts["ipopt.tol"] = 1e-14
        opts["ipopt.acceptable_iter"] = 0
    if np.isfinite(budget):                                           # backstop, as in build_collocation_ipopt
        opts["ipopt.max_wall_time"] = float(budget) + (0.05 if stop is not None else 0.0)
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


COLLOCATION_IPOPT_STATUS = {"Solve_Succeeded": "converged", "Solved_To_Acceptable_Level": "acceptable",
               "Maximum_Iterations_Exceeded": "cap", "User_Requested_Stop": "timeout",
               "Maximum_WallTime_Exceeded": "timeout", "Maximum_CpuTime_Exceeded": "timeout",
               "Restoration_Failed": "stalled", "Error_In_Step_Computation": "stalled",
               "Search_Direction_Becomes_Too_Small": "stalled",
               "Infeasible_Problem_Detected": "infeasible",
               "Invalid_Number_Detected": "diverged", "Diverging_Iterates": "diverged"}


def hopf_lax_mpc_status(tr, it, maxit):
    """Exit reason of mpc_testbed.hopf_lax_mpc_solve from its diagnostic trace (one record per started
    iteration, appended before any branch) and the completed-iteration count `it`."""
    if len(tr) == it:                                                 # loop exhausted, no early break
        return "cap" if it >= maxit else "timeout"
    last = tr[-1]
    if last["branch"] is None:
        return "converged"                                            # certified minimizer break
    if last["branch"] == "desc" and not last["ac"]:
        return "stalled"                                              # convex line-search stall
    if last["branch"] == "esc" and not last["ac"]:
        return "esc-stalled"                                          # no improving escape alpha
    return "unknown"


# ======================================================================================================
#  CLOSED-LOOP RUNNERS (instrumented copies; same timed bracket as mpc_testbed.py)
# ======================================================================================================
def _timing_stats(tcyc, rc):
    tt = np.asarray(tcyc)
    keep = np.ones(tt.size, dtype=bool)
    keep[0] = False
    rc = np.asarray(rc, dtype=bool)
    nrc = 0
    if rc.size == tt.size:
        nrc = int(np.sum(rc[1:]))
        keep &= ~rc
    if keep.any():
        tt = tt[keep]
    return tt, nrc


def _progress(label, N, c, t, x, ITS, TS, ST, every):
    if c % every == 1 or c == 1:
        its = np.asarray(ITS)
        ms = 1e3 * np.asarray(TS)
        nonconv = sum(s not in ("converged", "acceptable", "ok", "shared-stop") for s in ST)
        print(f"{label:<10s} N={N:<4d} c{c:4d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f} | "
              f"iters mean {its.mean():6.1f} max {its.max():4d} | solve ms mean {ms.mean():8.2f} "
              f"max {ms.max():8.1f} | non-conv {nonconv}/{len(ST)}", flush=True)


def _finish(res, extra, X_full, P, M, method, N, K, label, TL, TC, TS, RC, ITS, ST, RS, wall, aborted, err):
    e = dict(method=method, N=int(N), K=K, label=label,
             Jreal=float(res.Jreal), Jpen=float(res.Jpen), Jtrue=float(res.Jtrue),
             reached=int(res.reached),
             t_reach=float(TL[-1] + P.dt_apply) if res.reached else np.nan,
             ncyc=int(res.ncyc), clear=float(res.clear),
             X=res.X[:, ::P.napply].copy(), U=res.U.copy(), tlog=np.asarray(TL),
             tcyc=np.asarray(TC), tsolve=np.asarray(TS), recompile=np.asarray(RC, dtype=bool),
             iters=np.asarray(ITS, dtype=int), status=list(ST), resid=np.asarray(RS, dtype=float),
             extra={k: (np.asarray(v) if not isinstance(v, list) or (v and not isinstance(v[0], str))
                        else list(v)) for k, v in extra.items()},
             aborted=bool(aborted), error=err, wall_s=float(wall), stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
             meta=dict(dt=P.dt, dt_apply=P.dt_apply, napply=P.napply, maxcyc=P.maxcyc, budget=P.budget,
                       lm_tol=P.lm_tol, lm_maxit=P.lm_maxit, ilqr_tol=P.ilqr_tol, ilqr_Kmax=P.ilqr_Kmax,
                       ipopt_tol=1e-8, ipopt_max_iter=P.ipopt_max_iter, reachtol=P.reachtol,
                       hopf_lax_mpc_epsc=P.hopf_lax_mpc_epsc, hopf_lax_mpc_alpha=P.hopf_lax_mpc_alpha, eta_s=S.HOPF_LAX_MPC_ETA_S, ridge=S.HOPF_LAX_MPC_RIDGE,
                       delta=S.HOPF_LAX_MPC_DELTA, iters_mppi=P.iters_mppi, ess_target=P.ess_target_mppi,
                       DDP_YSHIFT=T.DDP_YSHIFT, vanilla_hopf_lax_tol=getattr(P, "vanilla_hopf_lax_tol", np.nan),
                       hess_mode=getattr(P, "hess_mode", "")))
    return e


def run_pmp_shooting(M, N, P, path, JX, every):
    Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
    x = M.x0.copy()
    Sh.warm(x, C.ref_window(x, N, path, M, P, 2, 0.0))                 # compile OUTSIDE the timed loop
    sst = T.make_shared_stop(M, N, P, JX, path)                       # None unless --shared-stop
    if sst is not None:
        Sh.roll_controls(np.zeros(M.n), x, C.ref_window(x, N, path, M, P, 2, 0.0))
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    seed = np.zeros(M.n)
    t = 0.0
    TL, TS, RC, ITS, ST, RS = [], [], [], [], [], []
    NTR, MU, LM, NR = [], [], [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            T.shared_reset(sst, x, N, path, M, P, t)
            tc = time.perf_counter()
            Xr2 = C.ref_window(x, N, path, M, P, 2, t)
            Xref = Xr2[:M.n, ::2]
            ts = time.perf_counter()
            p0, itk, nr, st, ntr, mu = pmp_shooting_solve_counted(Sh, seed, x, Xr2, P, tc, stop=sst)
            TS.append(time.perf_counter() - ts - T.shared_spent(sst))
            u = np.asarray(JX.ustar(x, p0))
            Pc = np.asarray(Sh.roll_costate(p0, x, Xr2))
            seed = Pc[:, P.napply]
            tcyc.append(time.perf_counter() - tc - T.shared_spent(sst))
            RC.append(CF.fired)
            ITS.append(itk); ST.append(st); RS.append(nr); NTR.append(ntr); MU.append(mu); TL.append(t)
            # ---- lmin diagnostic at the APPLIED iterate (logging only; outside the timed section) ----
            rd, _, Jrd, Sxd = Sh.all(p0, x, Xr2)
            Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
            LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
            NR.append(float(np.linalg.norm(np.asarray(rd))))
            _progress("PMP_Shooting", N, c, t, x, ITS, TS, ST, every)
            x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = T.apply_step(
                x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
            if stop:
                reached = 1
                break
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                       # noqa: BLE001
            aborted, err = True, repr(ex)
            print(f"PMP N={N}: ABORTED at cycle {c}: {err}")
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    return _finish(res, dict(ntrial=NTR, mu=MU, lmin=LM, nres=NR), X, P, M, "PMP_Shooting", N, None, "PMP_Shooting",
                   TL, tcyc, TS, RC, ITS, ST, RS, time.perf_counter() - t_start, aborted, err)


def run_hopf_lax_mpc(M, N, P, path, JX, every):
    Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
    x = M.x0.copy()
    Xr0 = C.ref_window(x, N, path, M, P, 2, 0.0)
    Sh.warm(x, Xr0)                                                    # compile OUTSIDE the timed loop
    Jb = S.ensure_jbatch(Sh)                                           # + both batch shapes used by
    Jb(np.zeros((2, M.n)), x, Xr0)                                     #   the kick comparison (B=2)
    Jb(np.zeros((7, M.n)), x, Xr0)                                     #   and the escape alphas (B=7)
    sst = T.make_shared_stop(M, N, P, JX, path)                       # None unless --shared-stop
    if sst is not None:
        Sh.roll_controls(np.zeros(M.n), x, Xr0)
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    seed = np.zeros(M.n)
    t = 0.0
    PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
    TL, TS, RC, ITS, ST, RS = [], [], [], [], [], []
    NDESC, NESC, NKICK, NEIG, ESC, LM, NR = [], [], [], [], [], [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            T.shared_reset(sst, x, N, path, M, P, t)
            tc = time.perf_counter()
            Xr2 = C.ref_window(x, N, path, M, P, 2, t)
            Xref = Xr2[:M.n, ::2]
            tr = []
            ts = time.perf_counter()
            try:
                p0, itk, infok = S.hopf_lax_mpc_solve(Sh, seed, x, Xr2, PP, P.hopf_lax_mpc_epsc, S.HOPF_LAX_MPC_ETA_S, P.hopf_lax_mpc_alpha,
                                               S.HOPF_LAX_MPC_DELTA, tc, P.budget, trace=tr, stop=sst)
                st = hopf_lax_mpc_status(tr, itk, PP.maxit)
                if sst is not None and st == "converged":             # stopped by the shared rule
                    st = "shared-stop"
            except np.linalg.LinAlgError as ex:                       # eigh on a NaN M: divergence
                raise RuntimeError(f"diverged (LinAlgError: {ex})") from ex
            TS.append(time.perf_counter() - ts - T.shared_spent(sst))
            u = np.asarray(JX.ustar(x, p0))
            Pc = np.asarray(Sh.roll_costate(p0, x, Xr2))
            seed = Pc[:, P.napply]
            tcyc.append(time.perf_counter() - tc - T.shared_spent(sst))
            RC.append(CF.fired)
            ITS.append(itk); ST.append(st); TL.append(t)
            NDESC.append(infok.ndesc); NESC.append(infok.nesc); NKICK.append(infok.nkick)
            NEIG.append(infok.neig); ESC.append(bool(infok.escape))
            rd, _, Jrd, Sxd = Sh.all(p0, x, Xr2)
            Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
            LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
            NR.append(float(np.linalg.norm(np.asarray(rd))))
            RS.append(float(tr[-1].get("norm_grad", tr[-1].get("ng", np.nan))) if tr else np.nan)  # final |grad|
            _progress("Hopf_Lax_MPC", N, c, t, x, ITS, TS, ST, every)
            x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = T.apply_step(
                x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
            if stop:
                reached = 1
                break
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                       # noqa: BLE001
            aborted, err = True, repr(ex)
            print(f"Hopf_Lax_MPC N={N}: ABORTED at cycle {c}: {err}")
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    if ITS:
        print(f"Hopf_Lax_MPC N={N} iter stats: mean {np.mean(ITS):.2f} iters/cycle (max {max(ITS)}) | descent "
              f"{sum(NDESC)} | escape {sum(NESC)} | kicks {sum(NKICK)} | eigen reads {sum(NEIG)} over "
              f"{len(ITS)} cycles")
    return _finish(res, dict(ndesc=NDESC, nesc=NESC, nkick=NKICK, neig=NEIG, escape=ESC, lmin=LM, nres=NR),
                   X, P, M, "Hopf_Lax_MPC", N, None, "Hopf_Lax_MPC", TL, tcyc, TS, RC, ITS, ST, RS,
                   time.perf_counter() - t_start, aborted, err)


def run_vanilla_hopf_lax(M, N, P, path, JX, every):
    """Baseline: damped Newton on the Hopf-Lax objective with the EXACT Hessian (mpc_solvers.vanilla_hopf_lax_solve).
    Same warm start, same timed bracket and the same gradient-type stopping test as run_hopf_lax_mpc, but with
    tolerance P.vanilla_hopf_lax_tol and WITHOUT the curvature certificate / escape of hopf_lax_mpc_solve."""
    Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
    x = M.x0.copy()
    Xr0 = C.ref_window(x, N, path, M, P, 2, 0.0)
    Sh.warm(x, Xr0)                                                    # compile OUTSIDE the timed loop
    Jb = S.ensure_jbatch(Sh)
    Jb(np.zeros((7, M.n)), x, Xr0)                                     # batched line search (B=7)
    t0 = time.perf_counter()
    Hf = S.ensure_hessian(Sh, getattr(P, "hess_mode", "hessian"))
    np.asarray(Hf(np.zeros(M.n), x, Xr0))                              # exact Hessian: compile now
    print(f"Vanilla_Hopf_Lax N={N}: jax.hessian ({Sh.H_mode}) compiled in {time.perf_counter() - t0:.1f} s")
    sst = T.make_shared_stop(M, N, P, JX, path)                       # None unless --shared-stop
    if sst is not None:
        Sh.roll_controls(np.zeros(M.n), x, Xr0)
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    seed = np.zeros(M.n)
    t = 0.0
    PP = SimpleNamespace(tol=P.vanilla_hopf_lax_tol, maxit=P.lm_maxit)
    TL, TS, RC, ITS, ST, RS = [], [], [], [], [], []
    NACC, NREJ, NCHOL, MU, LMH_IN, LMM_IN, THESS, JJ, LMH, LM, NR = [], [], [], [], [], [], [], [], [], [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            T.shared_reset(sst, x, N, path, M, P, t)
            tc = time.perf_counter()
            Xr2 = C.ref_window(x, N, path, M, P, 2, t)
            Xref = Xr2[:M.n, ::2]
            ts = time.perf_counter()
            p0, itk, infok = S.vanilla_hopf_lax_solve(Sh, seed, x, Xr2, PP, tc, P.budget, stop=sst)
            TS.append(time.perf_counter() - ts - T.shared_spent(sst))
            u = np.asarray(JX.ustar(x, p0))
            Pc = np.asarray(Sh.roll_costate(p0, x, Xr2))
            seed = Pc[:, P.napply]
            tcyc.append(time.perf_counter() - tc - T.shared_spent(sst))
            RC.append(CF.fired)
            st_v = "shared-stop" if (sst is not None and sst.fired) else infok.status
            ITS.append(itk); ST.append(st_v); RS.append(infok.norm_grad); TL.append(t)
            NACC.append(infok.nacc); NREJ.append(infok.nrej); NCHOL.append(infok.nchol); MU.append(infok.mu)
            LMH_IN.append(infok.lminH); LMM_IN.append(infok.lminM); THESS.append(infok.t_hess); JJ.append(infok.J)
            # ---- untimed diagnostics at the APPLIED iterate: surrogate lmin, |res|, exact-Hessian lmin ----
            rd, _, Jrd, Sxd = Sh.all(p0, x, Xr2)
            Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
            LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
            NR.append(float(np.linalg.norm(np.asarray(rd))))
            Hx = np.asarray(Hf(p0, x, Xr2))
            LMH.append(float(np.linalg.eigvalsh(0.5 * (Hx + Hx.T)).min()) if np.all(np.isfinite(Hx)) else np.nan)
            _progress("Vanilla_Hopf_Lax", N, c, t, x, ITS, TS, ST, every)
            x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = T.apply_step(
                x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
            if stop:
                reached = 1
                break
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                       # noqa: BLE001
            aborted, err = True, repr(ex)
            print(f"Vanilla_Hopf_Lax N={N}: ABORTED at cycle {c}: {err}")
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    if ITS:
        share = float(np.sum(THESS) / max(np.sum(TS), 1e-12))
        print(f"Vanilla_Hopf_Lax N={N} iter stats: mean {np.mean(ITS):.2f} iters/cycle (max {max(ITS)}) | accepted "
              f"{sum(NACC)} | rejected {sum(NREJ)} | extra Cholesky {sum(NCHOL)} | Hessian share of solve "
              f"time {100 * share:.0f}% over {len(ITS)} cycles")
    return _finish(res, dict(nacc=NACC, nrej=NREJ, nchol=NCHOL, mu=MU, lminH_in=LMH_IN, lminM_in=LMM_IN,
                             t_hess=THESS, J=JJ, lminH=LMH, lmin=LM, nres=NR),
                   X, P, M, "Vanilla_Hopf_Lax", N, None, "Vanilla_Hopf_Lax", TL, tcyc, TS, RC, ITS, ST, RS,
                   time.perf_counter() - t_start, aborted, err)


def run_ddp(M, N, P, path, JX, every):
    DP = S.build_ddp(M, N, P.dt, JX)
    x = M.x0.copy()
    DP.warm(x, C.ref_window(x, N, path, M, P, 1, 0.0)[:M.n, :], C.mobs_center(M.mobs, 0.0))
    sst = T.make_shared_stop(M, N, P, JX, path)                       # None unless --shared-stop
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    U = None
    t = 0.0
    TL, TS, RC, ITS, ST, RS = [], [], [], [], [], []
    NBAD, NREJ, NEFF, MU, DX, DV1, ALAST, AMIN, JJ, CONV = [], [], [], [], [], [], [], [], [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            T.shared_reset(sst, x, N, path, M, P, t)
            tc = time.perf_counter()
            Xref = C.ref_window(x, N, path, M, P, 1, t)[:M.n, :]
            mc = C.mobs_center(M.mobs, t)
            # dodge-commitment window (v6-measured): wall shell within the horizon's reach (+0.6 shell
            # margin), state still on-axis -- the solver sees a +y-shifted state inside it.
            gap = (mc[0, 0] - M.mobs.ax[0]) - x[0]
            in_window = (0.0 < gap < N * P.dt * max(x[4], 0.1) + 0.6) and (abs(x[1]) < 0.05)
            xs = x.copy()
            if in_window and T.DDP_YSHIFT != 0.0:
                xs[1] += T.DDP_YSHIFT
            Uinit = None if U is None else S.shift_U(U, P.napply)
            ts = time.perf_counter()
            p0, U, itk, info = ddp_run_counted(DP, xs, Xref, mc, Uinit, P, tc, stop=sst)
            TS.append(time.perf_counter() - ts - T.shared_spent(sst))
            u = np.asarray(JX.ustar(x, p0))
            tcyc.append(time.perf_counter() - tc - T.shared_spent(sst))
            RC.append(CF.fired)
            ITS.append(itk); ST.append(info.status); RS.append(abs(info.dV1)); TL.append(t)
            NBAD.append(info.nbad); NREJ.append(info.nrej); NEFF.append(info.n_eff); MU.append(info.mu)
            DX.append(info.dX); DV1.append(info.dV1); ALAST.append(info.a_last); AMIN.append(info.a_min)
            JJ.append(info.J); CONV.append(info.conv_by)
            _progress("DDP", N, c, t, x, ITS, TS, ST, every)
            x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = T.apply_step(
                x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
            if stop:
                reached = 1
                break
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                       # noqa: BLE001
            aborted, err = True, repr(ex)
            print(f"DDP N={N}: ABORTED at cycle {c}: {err}")
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    return _finish(res, dict(nbad=NBAD, nrej=NREJ, n_eff=NEFF, mu=MU, dX=DX, dV1=DV1, a_last=ALAST,
                             a_min=AMIN, J=JJ, conv_by=CONV),
                   X, P, M, "DDP", N, None, "DDP", TL, tcyc, TS, RC, ITS, ST, RS,
                   time.perf_counter() - t_start, aborted, err)


def _ipopt_last(st, key):
    try:
        v = st["iterations"][key]
        return float(v[-1]) if len(v) else np.nan
    except Exception:                                                 # noqa: BLE001
        return np.nan


def run_collocation_ipopt(M, N, P, path, JX, every):
    sst = T.make_shared_stop(M, N, P, JX, path)                       # None unless --shared-stop
    collf = build_collocation_ipopt_nobudget(M, N, P.dt, P.budget, P.ipopt_max_iter, stop=sst)
    x = M.x0.copy()
    collf(x, C.ref_window(x, N, path, M, P, 1, 0.0), None, time.perf_counter())  # warm IPOPT (discarded)
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    w = None
    t = 0.0
    TL, TS, RC, ITS, ST, RS = [], [], [], [], [], []
    RAW, CBC, FF, IPR, IDU = [], [], [], [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            T.shared_reset(sst, x, N, path, M, P, t)
            tc = time.perf_counter()
            Xr1 = C.ref_window(x, N, path, M, P, 1, t)
            Xref = Xr1[:M.n, :]
            ts = time.perf_counter()
            f, u, wsol = collf(x, Xr1, w, tc)
            TS.append(time.perf_counter() - ts - T.shared_spent(sst))
            w = S.shift_collocation_ipopt(wsol, M.n, M.m, N, P.napply)
            tcyc.append(time.perf_counter() - tc - T.shared_spent(sst))
            RC.append(CF.fired)
            st = collf.solver.stats()
            raw = str(st.get("return_status", "?"))
            ST.append("shared-stop" if (sst is not None and sst.fired) else COLLOCATION_IPOPT_STATUS.get(raw, raw))
            RAW.append(raw)
            ITS.append(int(st.get("iter_count", -1))); CBC.append(int(collf.cb.calls))
            FF.append(f)
            IPR.append(_ipopt_last(st, "inf_pr")); IDU.append(_ipopt_last(st, "inf_du"))
            RS.append(IDU[-1]); TL.append(t)
            _progress("Coll", N, c, t, x, ITS, TS, ST, every)
            x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = T.apply_step(
                x, u, None, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
            if stop:
                reached = 1
                break
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                       # noqa: BLE001
            aborted, err = True, repr(ex)
            print(f"Coll N={N}: ABORTED at cycle {c}: {err}")
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    return _finish(res, dict(ipopt_status=RAW, iter_count=ITS, cb_calls=CBC, f=FF,
                             inf_pr=IPR, inf_du=IDU),
                   X, P, M, "Collocation_IPOPT", N, None, "Collocation_IPOPT", TL, tcyc, TS, RC, ITS, ST, RS,
                   time.perf_counter() - t_start, aborted, err)


def run_mppi(M, N, P, path, JX, K, every):
    """Copy of mpc_testbed.run_mppi_mpc (incl. its wide-lambda subclass) minus the untimed animation
    cloud; additionally records the tuned lambda per cycle.  1 sampling update per cycle, as published."""
    class _MPPIWideLam(S.MPPI):
        def _tune_lambda(self, Sc, rho):
            import cupy as cp
            nsub = min(self.K, 2048)
            d_h = cp.asnumpy(Sc[:nsub] - rho)
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
    label = "MPPI" if K == P.K_mppi else f"MPPI_{K}"                   # method name = runner name
    x = M.x0.copy()
    Xref0 = C.ref_window(x, N, path, M, P, 1, 0.0)[:M.n, :]
    ctrl.warm(x, Xref0, C.mobs_center(M.mobs, 0.0))                    # NVRTC compile OUTSIDE the loop
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    t = 0.0
    TL, TS, RC, ITS, ST, RS = [], [], [], [], [], []
    ESS, LAM = [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            tc = time.perf_counter()
            Xref = C.ref_window(x, N, path, M, P, 1, t)[:M.n, :]
            mc = C.mobs_center(M.mobs, t)                              # static-at-current, like everyone
            ts = time.perf_counter()
            u, U, ess = ctrl.solve(x, Xref, mc)
            TS.append(time.perf_counter() - ts)
            ctrl.shift(P.napply)                                       # warm start for the next cycle
            tcyc.append(time.perf_counter() - tc)
            RC.append(CF.fired)
            ESS.append(float(ess)); LAM.append(float(ctrl.lam))
            ITS.append(int(P.iters_mppi)); ST.append("ok"); RS.append(np.nan); TL.append(t)
            _progress(label, N, c, t, x, ITS, TS, ST, every)
            x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = T.apply_step(
                x, u, None, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
            if stop:
                reached = 1
                break
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                       # noqa: BLE001
            aborted, err = True, repr(ex)
            print(f"{label} N={N}: ABORTED at cycle {c}: {err}")
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    if ESS:
        ess_a = np.array(ESS)
        print(f"{label} N={N} ESS: mean {ess_a.mean():.1f}/{K} ({100 * ess_a.mean() / K:.1f}%), "
              f"min {ess_a.min():.1f}")
    return _finish(res, dict(ess=ESS, lam=LAM), X, P, M, label, N, K, label,
                   TL, tcyc, TS, RC, ITS, ST, RS, time.perf_counter() - t_start, aborted, err)


RUNNERS = {"PMP_Shooting": run_pmp_shooting, "Hopf_Lax_MPC": run_hopf_lax_mpc, "DDP": run_ddp, "Collocation_IPOPT": run_collocation_ipopt, "Vanilla_Hopf_Lax": run_vanilla_hopf_lax}


def run_one(method, N, M, P, path, JX, every):
    if method == "MPPI" or method.startswith("MPPI_"):
        return run_mppi(M, N, P, path, JX, P.K_mppi if method == "MPPI" else int(method[5:]), every)
    if method not in RUNNERS:
        raise SystemExit(f"unknown method {method!r}; use PMP_Shooting, DDP, Collocation_IPOPT, Hopf_Lax_MPC, Vanilla_Hopf_Lax, MPPI, MPPI_<K>")
    return RUNNERS[method](M, N, P, path, JX, every)


# ======================================================================================================
#  TABLE
# ======================================================================================================
def print_table(entries, P):
    print(f"\n{'method':<12}{'N':>5}{'Tp[s]':>6}{'mean ms':>9}{'max ms':>9}{'p99 ms':>9}{'it mean':>8}"
          f"{'it max':>7}{'%conv':>7}{'%cap':>6}{'%stall':>7}{'J_track':>9}{'J_obs':>9}{'J_total':>9}"
          f"{'reached':>9}{'t_reach':>8}{'cyc':>5}{'rcmp':>5}")
    print("-" * 150)
    for e in entries:
        tt, nrc = _timing_stats(e["tcyc"], e["recompile"])
        st = e["status"]
        n = max(len(st), 1)
        pconv = 100 * sum(s in ("converged", "acceptable", "ok", "shared-stop") for s in st) / n
        pcap = 100 * sum(s == "cap" for s in st) / n
        pstall = 100 * sum(s in ("stalled", "esc-stalled") for s in st) / n
        its = e["iters"]
        tr = f"{e['t_reach']:.2f}" if e["reached"] else "--"
        flag = " ABORTED" if e["aborted"] else ""
        print(f"{e['label']:<12}{e['N']:>5d}{e['N'] * P.dt:>6.2f}{1e3 * tt.mean():>9.2f}{1e3 * tt.max():>9.1f}"
              f"{1e3 * np.percentile(tt, 99):>9.1f}{its.mean():>8.1f}{its.max():>7d}{pconv:>7.1f}{pcap:>6.1f}"
              f"{pstall:>7.1f}{e['Jreal']:>9.3f}{e['Jpen']:>9.3f}{e['Jtrue']:>9.3f}"
              f"{('yes' if e['reached'] else 'TRAPPED'):>9}{tr:>8}{e['ncyc']:>5d}{nrc:>5d}{flag}")
    print("timing: whole-cycle ms, cycle 1 (cold start) and 'rcmp' recompile-tagged cycles excluded")


# ======================================================================================================
#  ENVIRONMENT, SELFTEST, MAIN
# ======================================================================================================
def make_env(smoke, cycle_timeout, cap, vanilla_hopf_lax_tol=1e-3, hess_mode="hessian"):
    P = C.make_P(smoke=smoke)
    P = mpc_tuned_params.apply(P)                                     # scenario exactly as mpc_testbed
    P.budget = float(cycle_timeout)
    P.lm_maxit = int(cap)
    P.ilqr_Kmax = int(cap)
    P.ipopt_max_iter = int(cap)
    P.vanilla_hopf_lax_tol = float(vanilla_hopf_lax_tol)                                 # Vanilla_Hopf_Lax gradient certificate
    P.hess_mode = str(hess_mode)
    M = C.model_scn(P)
    JX = C.build_jax(M)
    path = C.plan_path(P.p0, P.pgoal, P.obs_plan, P)
    return P, M, JX, path


def obs_motion_record(M, P):
    tgrid = np.arange(P.maxcyc + 1) * P.dt_apply
    return dict(
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


def selftest(P, M, JX, path, N=50, ncyc=5):
    """Counted copies must reproduce the originals bitwise on identical inputs (budget=inf)."""
    ok = True

    def check(name, a, b):
        nonlocal ok
        a, b = np.asarray(a), np.asarray(b)
        same = a.shape == b.shape and np.array_equal(a, b)
        diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else np.inf
        print(f"  {name:<34s} {'IDENTICAL' if same else 'DIFF max ' + format(diff, '.2e')}")
        ok &= same

    print("== selftest: PMP pmp_shooting_solve_counted vs mpc_solvers.pmp_shooting_solve")
    Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
    x = M.x0.copy(); t = 0.0; seed = np.zeros(M.n)
    Sh.warm(x, C.ref_window(x, N, path, M, P, 2, 0.0))
    X, Uapp, Pclog = [x.copy()], [], []
    Jr_ = Jp_ = 0.0
    for c in range(ncyc):
        Xr2 = C.ref_window(x, N, path, M, P, 2, t)
        tc = time.perf_counter()
        p_ref = S.pmp_shooting_solve(Sh, seed.copy(), x, Xr2, P, tc)
        p_new, itk, nr, st, ntr, mu = pmp_shooting_solve_counted(Sh, seed.copy(), x, Xr2, P, tc)
        check(f"cycle {c + 1} p  (it={itk}, {st}, |res|={nr:.1e})", p_ref, p_new)
        u = np.asarray(JX.ustar(x, p_new))
        seed = np.asarray(Sh.roll_costate(p_new, x, Xr2))[:, P.napply]
        x, X, Jr_, Jp_, Uapp, Pclog, _, _, t = T.apply_step(x, u, p_new, X, Xr2[:M.n, ::2], Jr_, Jp_,
                                                             Uapp, Pclog, M, P, t, JX)

    print("== selftest: DDP ddp_run_counted vs mpc_solvers.ddp_run")
    DP = S.build_ddp(M, N, P.dt, JX)
    x = M.x0.copy(); t = 0.0; U = None
    DP.warm(x, C.ref_window(x, N, path, M, P, 1, 0.0)[:M.n, :], C.mobs_center(M.mobs, 0.0))
    X, Uapp, Pclog = [x.copy()], [], []
    Jr_ = Jp_ = 0.0
    for c in range(ncyc):
        Xref = C.ref_window(x, N, path, M, P, 1, t)[:M.n, :]
        mc = C.mobs_center(M.mobs, t)
        Uinit = None if U is None else S.shift_U(U, P.napply)
        tc = time.perf_counter()
        p_ref, U_ref, it_ref = S.ddp_run(DP, x, Xref, mc, None if Uinit is None else Uinit.copy(), P, tc)
        p_new, U, it_new, info = ddp_run_counted(DP, x, Xref, mc, None if Uinit is None else Uinit.copy(),
                                                 P, tc)
        check(f"cycle {c + 1} U  (it={it_new}/{it_ref}, {info.status}:{info.conv_by})", U_ref, U)
        check(f"cycle {c + 1} p0", p_ref, p_new)
        ok &= (it_ref == it_new)
        u = np.asarray(JX.ustar(x, p_new))
        x, X, Jr_, Jp_, Uapp, Pclog, _, _, t = T.apply_step(x, u, p_new, X, Xref, Jr_, Jp_,
                                                             Uapp, Pclog, M, P, t, JX)

    print("== selftest: build_collocation_ipopt_nobudget(max_iter=100) vs mpc_testbed.build_collocation_ipopt_hostcut(budget=inf)")
    cf_ref = T.build_collocation_ipopt_hostcut(M, N, P.dt, np.inf)
    cf_new = build_collocation_ipopt_nobudget(M, N, P.dt, np.inf, 100)
    x = M.x0.copy(); t = 0.0; w_ref = w_new = None
    X, Uapp, Pclog = [x.copy()], [], []
    Jr_ = Jp_ = 0.0
    for c in range(3):
        Xr1 = C.ref_window(x, N, path, M, P, 1, t)
        tc = time.perf_counter()
        _, u_ref, ws_ref = cf_ref(x, Xr1, w_ref, tc)
        _, u_new, ws_new = cf_new(x, Xr1, w_new, tc)
        st = cf_new.solver.stats()
        check(f"cycle {c + 1} w  (iter_count={st.get('iter_count')}, {st.get('return_status')})", ws_ref, ws_new)
        w_ref = S.shift_collocation_ipopt(ws_ref, M.n, M.m, N, P.napply)
        w_new = S.shift_collocation_ipopt(ws_new, M.n, M.m, N, P.napply)
        x, X, Jr_, Jp_, Uapp, Pclog, _, _, t = T.apply_step(x, u_new, None, X, Xr1[:M.n, :], Jr_, Jp_,
                                                             Uapp, Pclog, M, P, t, JX)
    print(f"  IPOPT stats keys: {sorted(st.keys())}")
    if "iterations" in st:
        print(f"  stats['iterations'] keys: {sorted(st['iterations'].keys())}")

    print("== selftest: Hopf_Lax_MPC status derivation (hopf_lax_mpc_solve trace)")
    Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
    x = M.x0.copy(); t = 0.0; seed = np.zeros(M.n)
    Xr0 = C.ref_window(x, N, path, M, P, 2, 0.0)
    Sh.warm(x, Xr0)
    Jb = S.ensure_jbatch(Sh); Jb(np.zeros((2, M.n)), x, Xr0); Jb(np.zeros((7, M.n)), x, Xr0)
    PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
    X, Uapp, Pclog = [x.copy()], [], []
    Jr_ = Jp_ = 0.0
    REF = []                                                           # (x, Xr2, seed_in, p0) for the Newton checks
    for c in range(ncyc):
        Xr2 = C.ref_window(x, N, path, M, P, 2, t)
        tr = []
        seed_in = seed.copy()
        p0, itk, info = S.hopf_lax_mpc_solve(Sh, seed, x, Xr2, PP, P.hopf_lax_mpc_epsc, S.HOPF_LAX_MPC_ETA_S, P.hopf_lax_mpc_alpha,
                                      S.HOPF_LAX_MPC_DELTA, time.perf_counter(), P.budget, trace=tr)
        st = hopf_lax_mpc_status(tr, itk, PP.maxit)
        good = st != "unknown" and (len(tr) - itk) in (0, 1)
        print(f"  cycle {c + 1}: it={itk} trace={len(tr)} status={st} ndesc={info.ndesc} "
              f"nkick={info.nkick} -> {'OK' if good else 'BAD'}")
        ok &= good
        REF.append((x.copy(), Xr2.copy(), seed_in, np.array(p0)))
        u = np.asarray(JX.ustar(x, p0))
        seed = np.asarray(Sh.roll_costate(p0, x, Xr2))[:, P.napply]
        x, X, Jr_, Jp_, Uapp, Pclog, _, _, t = T.apply_step(x, u, p0, X, Xr2[:M.n, ::2], Jr_, Jp_,
                                                             Uapp, Pclog, M, P, t, JX)

    print("== selftest: Vanilla_Hopf_Lax (exact Hessian) -- H vs finite differences, H vs surrogate, convergence")
    t0 = time.perf_counter()
    Hf = S.ensure_hessian(Sh, getattr(P, "hess_mode", "hessian"))
    np.asarray(Hf(np.zeros(M.n), REF[0][0], REF[0][1]))
    print(f"  jax.hessian compiled in {time.perf_counter() - t0:.1f} s (mode {Sh.H_mode})")
    PPn = SimpleNamespace(tol=P.vanilla_hopf_lax_tol, maxit=P.lm_maxit)
    h = 1e-5
    for ci in (0, 2, 4):
        xs, Xr2, seed_in, p_ref = REF[ci]
        Hraw = np.asarray(Hf(p_ref, xs, Xr2))
        asym = float(np.linalg.norm(Hraw - Hraw.T) / max(np.linalg.norm(Hraw), 1e-300))
        H = 0.5 * (Hraw + Hraw.T)
        Hfd = np.zeros_like(H)
        for j in range(M.n):
            e = np.zeros(M.n); e[j] = h
            Hfd[:, j] = (np.asarray(Sh.g(p_ref + e, xs, Xr2)) - np.asarray(Sh.g(p_ref - e, xs, Xr2))) / (2 * h)
        rel = float(np.linalg.norm(H - Hfd) / max(np.linalg.norm(H), 1e-300))
        umax = float(np.abs(np.asarray(JX.ustar(xs, p_ref))).max())
        saturated = umax >= 0.999 * float(np.min(M.umax))
        # (b) exact vs surrogate eigenvalues at the certified iterate
        rd, Jd, Jrd, Sxd = Sh.all(p_ref, xs, Xr2)
        Mc = -(np.asarray(Sxd).T @ np.asarray(Jrd)); Mc = 0.5 * (Mc + Mc.T)
        lamH, lamM = np.linalg.eigvalsh(H), np.linalg.eigvalsh(Mc)
        nres = float(np.linalg.norm(np.asarray(rd)))
        same_sign = (lamH[0] < -P.hopf_lax_mpc_epsc) == (lamM[0] < -P.hopf_lax_mpc_epsc)
        # (d) gradient identity -Sx'res vs jax.grad
        gid = -np.asarray(Sxd).T @ np.asarray(rd); gj = np.asarray(Sh.g(p_ref, xs, Xr2))
        gdiff = float(np.linalg.norm(gid - gj) / (1 + np.linalg.norm(gj)))
        # (c) Newton from the Hopf_Lax_MPC warm start converges to the certified point
        pn, itn, infon = S.vanilla_hopf_lax_solve(Sh, seed_in, xs, Xr2, PPn, time.perf_counter(), np.inf)
        pn2, itn2, _ = S.vanilla_hopf_lax_solve(Sh, seed_in, xs, Xr2, PPn, time.perf_counter(), np.inf)
        dv = float(np.linalg.norm(pn - p_ref) / (1 + np.linalg.norm(p_ref)))
        det = bool(np.array_equal(pn, pn2) and itn == itn2)
        fd_ok = (rel < 1e-4 and asym < 1e-8) or saturated
        conv_ok = infon.status == "converged" and itn <= 3 and dv <= 1e-3
        sat_note = " (control saturated: FD not asserted)" if saturated else ""
        print(f"  cycle {ci + 1}: |H-H_fd|/|H| {rel:.1e} asym {asym:.1e}{sat_note}"
              f" | lminH {lamH[0]:+.2e} lminM {lamM[0]:+.2e} max|dlam| {np.abs(lamH - lamM).max():.1e} |res| {nres:.1e}"
              f" sign {'OK' if same_sign else 'BAD'} | grad id {gdiff:.1e} | vanilla {infon.status} it={itn}"
              f" nrej={infon.nrej} |dv| {dv:.1e} deterministic {det} -> {'OK' if (fd_ok and conv_ok and same_sign and det) else 'BAD'}")
        ok &= fd_ok and conv_ok and same_sign and det

    ok &= selftest_shared_stop(P, M, JX, path, N)
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def selftest_shared_stop(P, M, JX, path, N=50, eps=1e-3):
    """Shared stopping rule (core/shared_stop.py): evaluator and U extraction are consistent, the counted
    copies still match the originals with the rule ON, and one cycle per method stops on the rule."""
    import shared_stop as SSTOP
    ok = True
    print(f"== selftest: shared stopping rule |dJ_ocp| <= {eps:g} (1 + |J_ocp|), N={N}")
    cost = SSTOP.build_ocp_cost(M, N, P.dt, JX)
    x = M.x0.copy(); t = 5.0                                          # mid-course, off the line, wall moving
    x[:2], x[3] = (7.5, 0.1), 0.05                                    # (at x0 the optimum is trivially u = 0)
    Xr2 = C.ref_window(x, N, path, M, P, 2, t)
    Xref = C.ref_window(x, N, path, M, P, 1, t)[:M.n, :]
    mc = C.mobs_center(M.mobs, t)
    DP = S.build_ddp(M, N, P.dt, JX)
    Ur = np.random.default_rng(0).uniform(-1, 1, (M.m, N))
    jd = float(DP.rollout_cost(x, Ur, Xref, mc)[1]); je = cost(x, Ur, Xref, mc)
    print(f"  J_ocp(U) == DDP rollout_cost J              {'IDENTICAL' if jd == je else 'DIFF'}  ({je:.6f})")
    ok &= jd == je
    Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
    p = S.pmp_shooting_solve(Sh, np.zeros(M.n), x, Xr2, P, time.perf_counter())
    Uc = np.asarray(Sh.roll_controls(p, x, Xr2))
    du = float(np.max(np.abs(Uc[:, 0] - np.asarray(JX.ustar(x, p)))))
    print(f"  roll_controls(p)[:, 0] == ustar(x, p)       max|d| {du:.1e} -> {'OK' if du < 1e-12 else 'BAD'}")
    ok &= du < 1e-12

    def mk():
        s_ = SSTOP.SharedStop(cost, eps); s_.reset(x.copy(), Xref, mc); return s_

    # counted copies == originals with the rule ON (fresh, identically reset rules)
    s1, s2 = mk(), mk()
    pa = S.pmp_shooting_solve(Sh, np.zeros(M.n), x, Xr2, P, time.perf_counter(), stop=s1)
    pb, itp, _, stp, _, _ = pmp_shooting_solve_counted(Sh, np.zeros(M.n), x, Xr2, P, time.perf_counter(), stop=s2)
    same = np.array_equal(pa, pb)
    print(f"  PMP counted == original (rule on)           {'IDENTICAL' if same else 'DIFF'}  it={itp} {stp} "
          f"J_hist={np.round(s2.J_hist, 4).tolist()}")
    ok &= same and stp == "shared-stop"
    s1, s2 = mk(), mk()
    pa, Ua, ita = S.ddp_run(DP, x, Xref, mc, None, P, time.perf_counter(), stop=s1)
    pb, Ub, itb, info = ddp_run_counted(DP, x, Xref, mc, None, P, time.perf_counter(), stop=s2)
    same = np.array_equal(Ua, Ub) and ita == itb
    print(f"  DDP counted == original (rule on)           {'IDENTICAL' if same else 'DIFF'}  it={itb} {info.status} "
          f"J_hist={np.round(s2.J_hist, 4).tolist()}")
    ok &= same and info.status == "shared-stop"
    # collocation: the U the callback tested last == the U of the returned iterate
    s3 = mk()
    cf = build_collocation_ipopt_nobudget(M, N, P.dt, np.inf, 1000, stop=s3)
    s3.reset(x.copy(), Xref, mc)
    _, _, w = cf(x, C.ref_window(x, N, path, M, P, 1, t), None, time.perf_counter())
    Uw = w[M.n * (N + 1):].reshape((M.m, N), order="F")
    du = float(np.max(np.abs(Uw - s3._last)))
    st = str(cf.solver.stats().get("return_status"))
    print(f"  IPOPT: callback U == returned U             max|d| {du:.1e}  status {st}, {len(s3.J_hist)} iterates, "
          f"fired={s3.fired} -> {'OK' if du == 0.0 and s3.fired else 'BAD'}")
    ok &= du == 0.0 and s3.fired
    # Hopf_Lax_MPC and Vanilla_Hopf_Lax with the rule on
    s4 = mk()
    Jb = S.ensure_jbatch(Sh)
    _, ith, infh = S.hopf_lax_mpc_solve(Sh, np.zeros(M.n), x, Xr2, SimpleNamespace(tol=P.lm_tol, maxit=1000),
                                        P.hopf_lax_mpc_epsc, S.HOPF_LAX_MPC_ETA_S, P.hopf_lax_mpc_alpha,
                                        S.HOPF_LAX_MPC_DELTA, time.perf_counter(), np.inf, stop=s4)
    print(f"  Hopf_Lax_MPC (rule on): it={ith} fired={s4.fired} J_hist={np.round(s4.J_hist, 4).tolist()}")
    ok &= s4.fired
    s5 = mk()
    _, itv, infv = S.vanilla_hopf_lax_solve(Sh, np.zeros(M.n), x, Xr2, SimpleNamespace(tol=1e-4, maxit=1000),
                                            time.perf_counter(), np.inf, stop=s5)
    print(f"  Vanilla_Hopf_Lax (rule on): it={itv} {infv.status} fired={s5.fired} "
          f"J_hist={np.round(s5.J_hist, 4).tolist()}")
    ok &= s5.fired
    return ok


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default=",".join(METHODS_DEFAULT))
    ap.add_argument("--Ns", default=",".join(str(n) for n in NS_DEFAULT))
    ap.add_argument("--smoke", action="store_true", help="60-cycle pipeline check -> " + PKL_SMOKE)
    ap.add_argument("--out", default=None, help=f"results pickle (default {PKL}, smoke: {PKL_SMOKE})")
    ap.add_argument("--redo", action="store_true", help="recompute (method, N) entries already in the pickle")
    ap.add_argument("--cycle-timeout", type=float, default=np.inf,
                    help="optional per-cycle wall-clock safety net in seconds (cuts recorded as 'timeout')")
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT,
                    help="iteration cap for PMP_Shooting/Hopf_Lax_MPC/Vanilla_Hopf_Lax (lm_maxit), DDP (ilqr_Kmax) and IPOPT (max_iter)")
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--selftest", action="store_true", help="verify the counted solver copies, then exit")
    ap.add_argument("--vanilla-hopf-lax-tol", type=float, default=1e-3,
                    help="Vanilla_Hopf_Lax gradient certificate |grad| <= tol (1 + |Phi|)  (Hopf-Lax-MPC uses lm_tol = 1e-4)")
    ap.add_argument("--hess-mode", default="hessian", choices=["hessian", "fwdfwd", "fwdgrad"],
                    help="how the exact Hessian is formed (jax.hessian = forward-over-reverse; fallbacks)")
    ap.add_argument("--shared-stop", type=float, default=None, metavar="EPS",
                    help="ONE stopping rule for every iterative solver: |dJ_ocp| <= EPS (1 + |J_ocp|) "
                         "(core/shared_stop.py); default output results_N_sweep_shared[_smoke].pkl")
    args = ap.parse_args(argv[1:])

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    Ns = [int(s) for s in args.Ns.split(",")]
    out = args.out or (PKL_SMOKE if args.smoke else PKL)
    if args.shared_stop is not None and args.out is None:             # never mix with the main-branch data
        out = out.replace(".pkl", "_shared.pkl")
    P, M, JX, path = make_env(args.smoke, args.cycle_timeout, args.cap, args.vanilla_hopf_lax_tol, args.hess_mode)
    P.shared_stop_eps = args.shared_stop
    print(f"=== N sweep, NO budget (budget={P.budget}, caps={args.cap}), maxcyc={P.maxcyc}, dt={P.dt}; "
          f"methods={methods}, Ns={Ns} -> {out}")

    if args.selftest:
        return selftest(P, M, JX, path)

    D = {}
    if os.path.exists(out):
        with open(out, "rb") as f:
            D = pickle.load(f)
    D.setdefault("schema", 1)
    D.setdefault("runs", {})
    D["config"] = dict(Ns=Ns, methods=methods, smoke=args.smoke, cycle_timeout=args.cycle_timeout,
                       cap=args.cap, ipopt_max_iter=args.cap, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                       host=platform.node(), vanilla_hopf_lax_tol=args.vanilla_hopf_lax_tol, hess_mode=args.hess_mode,
                       shared_stop_eps=args.shared_stop)
    D["P"], D["path"], D["obs_motion"] = P, path, obs_motion_record(M, P)

    def save():
        tmp = out + ".tmp"                                            # atomic: a kill mid-write cannot
        with open(tmp, "wb") as f:                                    # corrupt the finished results
            pickle.dump(D, f)
        os.replace(tmp, out)

    done = []
    try:
        for method in methods:
            for N in Ns:
                key = (method, N)
                if key in D["runs"] and not args.redo:
                    print(f"-- {key} already in {out}; skipping (use --redo to recompute)")
                    done.append(D["runs"][key])
                    continue
                print(f"\n#### {method} N={N}  ({time.strftime('%H:%M:%S')})", flush=True)
                e = run_one(method, N, M, P, path, JX, args.print_every)
                D["runs"][key] = e
                save()
                done.append(e)
                print_table([e], P)
                print(f"#### {method} N={N} done in {e['wall_s']:.0f} s -> saved {out}", flush=True)
    except KeyboardInterrupt:
        print("\n*** interrupted: finished runs are saved; the interrupted run is discarded ***")
        save()
        return 130
    print_table(done, P)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

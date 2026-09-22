"""
complexity_vs_horizon.py -- single-solve complexity study at the obs-2 saddle.

At a FIXED on-axis state in front of the (static) obs-2 wall, with a SYMMETRIC initial guess, run
each method to convergence at a sweep of horizons N and record a per-iteration (elapsed, J) trace.
No wall-clock budget: the question is how much computation each method needs to reach the optimal
trajectory cost, and how that scales with N.  Per the study protocol, no separate global-optimum
computation: the best converged J recorded across methods/reps at a given N serves as J*(N)
(the plot script applies the epsilon-band against it).

SCENE   v6 wall-course parameters (mpc_tuned_params.apply_wall_course) with the moving set reduced to obs-2 ONLY, static at its
        frozen position (10.4, 0, 0.5) -- blocker/door removed, obs-1 kept (inert, ~6 m behind).
        Same RRT reference (straight y=0).  Time-invariant problem.
X0      M1_v2's saddle-detection (kick) cycle state from the verified v6 closed loop -- on-axis,
        cruise speed, ~1 m before the wall shell.  Extracted from results_testbed_v6.pkl (v6 course run) when it
        holds an M1_v2 run, else re-derived by replaying the closed loop once; cached in the output
        pickle so every later chunk uses the identical state.
SEEDS   symmetric: PMP/M1_v2 costate p0 = 0.  (DDP U=0, collocation cold tile, MPPI zero-nominal --
        those methods are added in later chunks of this study.)
SWEEP   default N = 60..560 (dt = 0.005, Tp up to 2.8 s): deliberately past the shooting methods'
        conditioning cliff (cond ~ e^{2 lambda Tp}) -- divergence/DNF at large N is a reported data
        point.  The runner aborts extending the sweep after two consecutive diverged horizons.

USAGE
    python complexity_vs_horizon.py --methods m1v2              # sweep our method (default)
    python complexity_vs_horizon.py --methods pmp
    python complexity_vs_horizon.py --methods m1v2 --Ns 60,120,240
Results merge into results_complexity.pkl (reruns overwrite only the (method, N) entries they run).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np

import mpc_core as C
import mpc_testbed as T                          # installs the piecewise mobs_center; M1_v2 solver (identical to the v6 testbed)
import mpc_tuned_params as PRM

PKL = "results_complexity.pkl"
NS_DEFAULT = [60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400, 480, 560]
MAXIT = 400
REPS = 3


# ======================================================================================================
#  SCENE
# ======================================================================================================
def build_env():
    """v6 parameters, moving set reduced to the static obs-2 wall at its frozen position."""
    P = C.make_P()
    P = PRM.apply_wall_course(P)
    m = P.mobs
    P.mobs = SimpleNamespace(
        c0=np.array([[10.4], [0.0], [0.5]]),
        vel=np.zeros((3, 1)),
        ax=np.array([m.ax[0]]), ay=np.array([m.ay[0]]), az=np.array([m.az[0]]),
        W=np.array([m.W[0]]), w=np.array([m.w[0]]), eps=m.eps,
    )                                                                # static -> no tstop needed
    M = C.model_scn(P)
    JX = C.build_jax(M)
    path = C.plan_path(P.p0, P.pgoal, P.obs_plan, P)
    return P, M, JX, path


def get_x0():
    """M1_v2's kick-cycle state: cached -> from the v6 results pickle -> replayed closed loop."""
    if os.path.exists(PKL):
        with open(PKL, "rb") as f:
            D = pickle.load(f)
        if "x0" in D:
            return np.asarray(D["x0"]), D.get("x0_info", "")
    try:
        with open("results_testbed_v6.pkl", "rb") as f:
            d = pickle.load(f)
        for r in d["R"]:
            if r.name == "M1_v2" and np.asarray(r.res.m1_nkick).sum() > 0:
                nk = np.asarray(r.res.m1_nkick)
                idx = int(np.argmax(nk > 0))
                info = f"from results_testbed_v6.pkl, kick cycle {idx}, t={r.res.tlog[idx]:.2f}s"
                return np.asarray(r.res.xlog[:, idx]).copy(), info
    except FileNotFoundError:
        pass
    print("x0: replaying the v6 closed loop for M1_v2 (results pickle lacks it) ...")
    P = C.make_P()
    P = PRM.apply_wall_course(P)
    M = C.model_scn(P)
    JX = C.build_jax(M)
    path = C.plan_path(P.p0, P.pgoal, P.obs_plan, P)
    Sh = C.build_ss(M, P.N_m1, P.dt, JX, napply=P.napply)
    res = T.run_m1v2_mpc(M, P.N_m1, P, path, Sh, JX)
    nk = np.asarray(res.m1_nkick)
    idx = int(np.argmax(nk > 0))
    info = f"replayed v6 closed loop, kick cycle {idx}, t={res.tlog[idx]:.2f}s"
    return np.asarray(res.xlog[:, idx]).copy(), info


# ======================================================================================================
#  INSTRUMENTED SOLVERS (numerics identical to the closed-loop versions; adds per-iter timestamps)
# ======================================================================================================
def m1_solve_traced(Sh, v0, xc, Xr, P):
    """mpc_testbed.chlqn_solve with per-iteration (elapsed, J, |res|, |g|, iterate) records.
    Statement-for-statement copy of the branch logic (LM-free descent, gated eigen read, hysteresis,
    kick, floored escape step); budget off, maxit = MAXIT."""
    epsc, eta_s, ridge_thr = P.M1_epsc, T.M1V2_ETA_S, T.M1V2_RIDGE
    alpha, delta, tol = P.M1_alpha, T.M1V2_DELTA, P.lm_tol
    v = v0.copy()
    n = v.size
    escape = False
    Dv = Vc = vmin = None
    trace = []
    status = "maxit"
    t0 = time.perf_counter()
    for k in range(1, MAXIT + 1):
        res, J, Jr, Sx = Sh.all(v, xc, Xr)
        res, J = np.asarray(res), float(J)
        Jr, Sx = np.asarray(Jr), np.asarray(Sx)
        nr = np.linalg.norm(res)
        g = -Sx.T @ res
        ng = np.linalg.norm(g)
        trace.append(dict(k=k, t=time.perf_counter() - t0, J=J, nr=nr, ng=ng, v=v.copy()))
        if not np.isfinite(nr) or not np.isfinite(J) or nr > 1e12:
            status = "diverged"
            break
        Hc = -(Sx.T @ Jr)
        Hc = 0.5 * (Hc + Hc.T)

        if escape or ng <= eta_s * (1 + abs(J)):                      # gated detection / escape-exit read
            Dv, Vc = np.linalg.eigh(Hc)
            im = int(np.argmin(Dv))
            vmin = Vc[:, im]
            escape = bool(Dv[im] < -epsc)

        if ng <= tol * (1 + abs(J)) and not escape:                   # certified minimizer
            status = "converged"
            break

        if not escape:                                                # descent: solve J_r d = -rho
            try:
                d = np.linalg.solve(Jr, -res)
            except np.linalg.LinAlgError:
                d = None
            if d is None or not np.all(np.isfinite(d)):
                d = np.linalg.lstsq(Jr, -res, rcond=None)[0]
            ac, a = False, 1.0
            for _ in range(8):
                vt = v + a * d
                if np.linalg.norm(np.asarray(Sh.res(vt, xc, Xr))) < nr:
                    v, ac = vt, True
                    break
                a *= 0.5
            if not ac:                                                # stall: steering read
                Dv, Vc = np.linalg.eigh(Hc)
                im = int(np.argmin(Dv))
                vmin = Vc[:, im]
                escape = bool(Dv[im] < -epsc)
                if not escape:
                    status = "stalled"
                    break
        else:                                                         # escape mode (BATCHED rollouts,
            ridge = abs(g @ vmin) / max(ng, np.finfo(float).eps)      # matching mpc_testbed)
            Jb = T.ensure_jbatch(Sh)
            if ridge < ridge_thr:
                Jpm = np.asarray(Jb(np.stack([v - alpha * vmin, v + alpha * vmin]), xc, Xr))
                s = -1.0 if Jpm[0] < Jpm[1] else 1.0
                v = v + alpha * s * vmin
                res, J, Jr, Sx = Sh.all(v, xc, Xr)
                res, J = np.asarray(res), float(J)
                Jr, Sx = np.asarray(Jr), np.asarray(Sx)
                g = -Sx.T @ res
                Hc = -(Sx.T @ Jr)
                Hc = 0.5 * (Hc + Hc.T)
                Dv, Vc = np.linalg.eigh(Hc)
            dd = np.maximum(np.abs(Dv), delta)
            step = -Vc @ ((Vc.T @ g) / dd)
            Vt = v[None, :] + T.M1V2_ESC_ALPHAS[:, None] * step[None, :]
            Jts = np.asarray(Jb(Vt, xc, Xr))
            ok = np.isfinite(Jts) & (Jts < J - 1e-9 * abs(J))
            ac = bool(ok.any())
            if ac:
                v = Vt[int(np.argmax(ok))]                            # FIRST improving alpha
            if not ac:
                status = "esc-stalled"
                break
    t_end = time.perf_counter() - t0
    Jf = float(Sh.J(v, xc, Xr))
    nrf = float(np.linalg.norm(np.asarray(Sh.res(v, xc, Xr))))
    trace.append(dict(k="final", t=t_end, J=Jf, nr=nrf, ng=None, v=v.copy()))
    return v, trace, status


def pmp_solve_traced(Sh, v0, xc, Xr, P):
    """resnewton_solve (LM on the residual) with per-iteration records; budget off, maxit = MAXIT."""
    v = v0.copy()
    n = v.size
    mu = 1e-6
    trace = []
    status = "maxit"
    t0 = time.perf_counter()
    rr = np.asarray(Sh.res(v, xc, Xr))
    nr = np.linalg.norm(rr)
    for k in range(1, MAXIT + 1):
        J = float(Sh.J(v, xc, Xr))
        trace.append(dict(k=k, t=time.perf_counter() - t0, J=J, nr=nr, ng=None, v=v.copy()))
        if not np.isfinite(nr) or nr > 1e12:
            status = "diverged"
            break
        if nr < P.lm_tol:
            status = "converged"
            break
        Jr = np.asarray(Sh.Jr(v, xc, Xr))
        A = Jr.T @ Jr
        b = -Jr.T @ rr
        acc = False
        for _ in range(20):
            step = np.linalg.solve(A + mu * np.eye(n), b)
            vt = v + step
            rt = np.asarray(Sh.res(vt, xc, Xr))
            if np.linalg.norm(rt) < nr:
                v, rr, nr = vt, rt, np.linalg.norm(rt)
                mu = max(mu / 3, 1e-12)
                acc = True
                break
            mu = mu * 5
        if not acc:
            status = "stalled"
            break
    t_end = time.perf_counter() - t0
    Jf = float(Sh.J(v, xc, Xr))
    trace.append(dict(k="final", t=t_end, J=Jf, nr=float(nr), ng=None, v=v.copy()))
    return v, trace, status


def ddp_solve_traced(DP, x0, Xref, mc, ilqr_tol):
    """mpc_solvers.ddp_run without the budget cut; per-iteration (elapsed, J) records.  J here IS the
    common metric (DP.rollout_cost).  mu schedule / line search copied statement-for-statement."""
    import mpc_solvers as S
    N, m = DP.N, DP.m
    U = np.zeros((m, N))
    X, J = DP.rollout_cost(x0, U, Xref, mc)
    X, J = np.asarray(X), float(J)
    mu = 1e-3
    alphas = S.DDP_ALPHAS
    trace = []
    status = "maxit"
    t0 = time.perf_counter()
    for it in range(1, MAXIT + 1):
        kff, Kk, dV1, Dv2, bad = DP.backward(X, U, Xref, mc, mu)
        if bool(bad):
            mu = mu * 4
            trace.append(dict(k=it, t=time.perf_counter() - t0, J=J))
            if mu > 1e12:
                status = "stalled"
                break
            continue
        dV1, Dv2 = float(dV1), float(Dv2)
        if abs(dV1) < ilqr_tol:
            status = "converged"
            break
        Xn, Un, Jn = DP.forward(x0, X, U, kff, Kk, Xref, mc, alphas)
        Jn = np.asarray(Jn)
        improved, a_acc = False, np.nan
        for i in range(len(alphas)):
            if Jn[i] < J:
                a_acc = float(alphas[i])
                Xa, Ua = np.asarray(Xn[i]), np.asarray(Un[i])
                dX = np.linalg.norm(Xa - X) / (1 + np.linalg.norm(X))
                X, U, J = Xa, Ua, float(Jn[i])
                improved = True
                break
        trace.append(dict(k=it, t=time.perf_counter() - t0, J=J))
        if not np.isfinite(J):
            status = "diverged"
            break
        if improved:
            mu = mu * 2.0 if a_acc < 0.5 else max(mu * 0.7, 1e-9)
            if dX < ilqr_tol:
                status = "converged"
                break
        else:
            mu = mu * 4
            if mu > 1e12:
                status = "stalled"
                break
    trace.append(dict(k="final", t=time.perf_counter() - t0, J=float(J), nr=None))
    return U, trace, status


def coll_solve_traced(M, N, dt, x0, Xr1, evalJ, Xref, mc):
    """Collocation NLP (identical to build_coll / build_coll_hostcut: cost, constraints, ipopt tol
    1e-8) run to CONVERGENCE (no wall-time limit, max_iter = MAXIT), with an iteration callback
    capturing (elapsed, U-block of the iterate).  Common-metric J evaluated OFFLINE afterwards."""
    import casadi as ca
    n, m = M.n, M.m
    nm = M.mobs.c0.shape[1]
    rec = {"t": [], "U": []}
    t0_ref = [0.0]

    class _RecCB(ca.Callback):
        def __init__(self, name, nx, ng, npar):
            ca.Callback.__init__(self)
            self.nx, self.ng, self.npar = nx, ng, npar
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
            nmn = ca.nlpsol_out(i)
            if nmn == "f":
                return ca.Sparsity.scalar()
            if nmn in ("x", "lam_x"):
                return ca.Sparsity.dense(self.nx, 1)
            if nmn in ("g", "lam_g"):
                return ca.Sparsity.dense(self.ng, 1)
            if nmn == "lam_p":
                return ca.Sparsity.dense(self.npar, 1)
            return ca.Sparsity(0, 0)

        def eval(self, arg):
            # NOTE an exception escaping a CasADi callback is treated as a user stop request and
            # ABORTS the solve -- keep this body exception-safe and record any failure for diagnosis.
            try:
                w = np.array(ca.DM(arg[0]).full()).ravel()
                rec["t"].append(time.perf_counter() - t0_ref[0])
                rec["U"].append(w[n * (N + 1):].reshape((m, N), order="F").copy())
            except Exception as ex:                                   # pragma: no cover
                rec.setdefault("err", []).append(repr(ex))
            return [0.0]

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
    XrS = ca.MX.sym("Xr", n + 3 * nm, N + 1)
    X = ca.MX.sym("X", n, N + 1)
    U = ca.MX.sym("U", m, N)
    Xn_ = FP.map(N)(X[:, :N], U)
    Jc = 0
    o = M.obs
    for k in range(N):
        dxk = X[:, k] - XrS[:n, k]
        mck = ca.reshape(XrS[n:, k], 3, nm)
        pen = (barrier(X[:3, k], ca.DM(o.center), o.ax, o.ay, o.az, o.W, o.w, o.eps)
               + barrier(X[:3, k], mck, M.mobs.ax, M.mobs.ay, M.mobs.az, M.mobs.W, M.mobs.w, M.mobs.eps))
        Jc = Jc + (dxk.T @ Q @ dxk + U[:, k].T @ R @ U[:, k] + pen) * dt
    dxT = X[:, N] - XrS[:n, N]
    Jc = Jc + dxT.T @ QT @ dxT
    D_ = X[:, 1:] - Xn_
    G = ca.vertcat(X[:, 0] - par, ca.reshape(D_, -1, 1))

    nx = n * (N + 1) + m * N
    nG = n * (N + 1)
    npar = n + (n + 3 * nm) * (N + 1)
    cb = _RecCB("rec_cb", nx, nG, npar)
    opts = {"ipopt.max_iter": MAXIT, "ipopt.tol": 1e-8, "ipopt.print_level": 0,
            "print_time": 0, "error_on_fail": False, "iteration_callback": cb}
    s = ca.nlpsol("s", "ipopt", {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
                                 "f": Jc, "g": G, "p": ca.vertcat(par, ca.reshape(XrS, -1, 1))}, opts)
    lbx = np.concatenate([-np.inf * np.ones(n * (N + 1)), np.tile(-M.umax, N)])
    ubx = np.concatenate([np.inf * np.ones(n * (N + 1)), np.tile(M.umax, N)])
    X0w = np.tile(np.asarray(x0).reshape(-1, 1), (1, N + 1))
    W0 = np.concatenate([X0w.flatten(order="F"), np.zeros(m * N)])

    def one_solve():
        rec["t"].clear(); rec["U"].clear()
        t0_ref[0] = time.perf_counter()
        r = s(x0=W0, p=np.concatenate([np.asarray(x0), Xr1.flatten(order="F")]),
              lbg=np.zeros(nG), ubg=np.zeros(nG), lbx=lbx, ubx=ubx)
        t_end = time.perf_counter() - t0_ref[0]
        st = s.stats()["return_status"]
        w = np.asarray(r["x"]).ravel()
        Uf = w[n * (N + 1):].reshape((m, N), order="F")
        trace = [dict(k=i + 1, t=tt, J=float(evalJ(x0, Ui, Xref, mc)[1]))
                 for i, (tt, Ui) in enumerate(zip(rec["t"], rec["U"]))]
        trace.append(dict(k="final", t=t_end, J=float(evalJ(x0, Uf, Xref, mc)[1]), nr=None))
        status = "converged" if st in ("Solve_Succeeded", "Solved_To_Acceptable_Level") else st
        return Uf, trace, status

    one_solve.cb = cb                                                # keep the Python callback alive:
    return one_solve                                                 # GC'd callback -> instant abort


def mppi_solve_traced(M, P, N, K, x0, Xref, mc, evalJ, n_updates=150):
    """Iterated MPPI on the fixed state: fresh controller (seed 0), n_updates repeated solves with no
    shift; (elapsed, U) recorded per update, common-metric J evaluated offline."""
    import mpc_mppi as MPPI
    ctrl = MPPI.MPPI(M, P, N=N, K=K, lam=P.lam_mppi, alpha=P.alpha_mppi,
                     iters=1, seed=0, ess_target=P.ess_target_mppi)
    ctrl.warm(x0, Xref, mc)
    ctrl.U = np.zeros((M.m, N))                                       # reset nominal after warm
    ts, Us, ess_log = [], [], []
    t0 = time.perf_counter()
    for _ in range(n_updates):
        _, U, ess = ctrl.solve(x0, Xref, mc)
        ts.append(time.perf_counter() - t0)
        Us.append(np.asarray(U).copy())
        ess_log.append(float(ess))
    trace = [dict(k=i + 1, t=tt, J=float(evalJ(x0, Ui, Xref, mc)[1]), ess=e)
             for i, (tt, Ui, e) in enumerate(zip(ts, Us, ess_log))]
    Jf = min(r["J"] for r in trace)
    ib = int(np.argmin([r["J"] for r in trace]))                      # best update = what one would keep
    trace.append(dict(k="final", t=ts[-1], J=trace[-1]["J"], nr=None))
    return Us[ib], trace, "floor", Jf


# ======================================================================================================
#  SWEEP DRIVER
# ======================================================================================================
def run_shooting(method, Ns, P, M, JX, path, x0):
    out = {}
    consec_div = 0
    for N in Ns:
        Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
        Xr = C.ref_window(x0, N, path, M, P, 2, 0.0)
        Sh.warm(x0, Xr)                                              # compile OUTSIDE all timing
        if method == "m1v2":                                         # + the two batch shapes
            Jb = T.ensure_jbatch(Sh)
            Jb(np.zeros((2, M.n)), x0, Xr)
            Jb(np.zeros((7, M.n)), x0, Xr)
        solver = m1_solve_traced if method == "m1v2" else pmp_solve_traced
        reps = []
        for _ in range(REPS):
            _, trace, status = solver(Sh, np.zeros(M.n), x0, Xr, P)
        # (identical deterministic iterate sequence; reps differ only in wall clock)
            reps.append(dict(trace=trace, status=status))
        fin = reps[0]["trace"][-1]
        nit = len(reps[0]["trace"]) - 1
        tmed = float(np.median([r["trace"][-1]["t"] for r in reps]))
        print(f"  {method} N={N:4d}: iters {nit:3d}  status {reps[0]['status']:<11s} "
              f"t_end(med) {1e3 * tmed:8.2f} ms  J_final {fin['J']:.6f}  |res|_final {fin['nr']:.2e}")
        out[N] = dict(reps=reps, N=N)
        if reps[0]["status"] == "diverged":
            consec_div += 1
            if consec_div >= 2:
                print(f"  {method}: two consecutive diverged horizons -- stopping the sweep here.")
                break
        else:
            consec_div = 0
    return out


def run_ddp(Ns, P, M, JX, path, x0):
    import mpc_solvers as S
    out = {}
    for N in Ns:
        DP = S.build_ddp(M, N, P.dt, JX)
        Xref = C.ref_window(x0, N, path, M, P, 1, 0.0)[:M.n, :]
        mc = C.mobs_center(M.mobs, 0.0)
        DP.warm(x0, Xref, mc)
        reps = []
        for _ in range(REPS):
            _, trace, status = ddp_solve_traced(DP, x0, Xref, mc, P.ilqr_tol)
            reps.append(dict(trace=trace, status=status))
        fin = reps[0]["trace"][-1]
        nit = len(reps[0]["trace"]) - 1
        tmed = float(np.median([r["trace"][-1]["t"] for r in reps]))
        print(f"  ddp N={N:4d}: iters {nit:3d}  status {reps[0]['status']:<11s} "
              f"t_end(med) {1e3 * tmed:8.2f} ms  J_final {fin['J']:.6f}")
        out[N] = dict(reps=reps, N=N)
    return out


def run_coll(Ns, P, M, JX, path, x0):
    import mpc_solvers as S
    out = {}
    for N in Ns:
        IL = S.build_ilqr(M, N, P.dt, JX)                            # rollout_cost = common metric
        Xr1 = C.ref_window(x0, N, path, M, P, 1, 0.0)
        Xref = Xr1[:M.n, :]
        mc = C.mobs_center(M.mobs, 0.0)
        IL.rollout_cost(x0, np.zeros((M.m, N)), Xref, mc)            # compile the evaluator
        solve = coll_solve_traced(M, N, P.dt, x0, Xr1, IL.rollout_cost, Xref, mc)
        solve()                                                      # throwaway warm solve
        reps = []
        for _ in range(REPS):
            _, trace, status = solve()
            reps.append(dict(trace=trace, status=status))
        fin = reps[0]["trace"][-1]
        nit = len(reps[0]["trace"]) - 1
        tmed = float(np.median([r["trace"][-1]["t"] for r in reps]))
        print(f"  coll N={N:4d}: iters {nit:3d}  status {reps[0]['status']:<22s} "
              f"t_end(med) {1e3 * tmed:8.2f} ms  J_final {fin['J']:.6f}")
        out[N] = dict(reps=reps, N=N)
    return out


def run_mppi(K, Ns, P, M, JX, path, x0):
    import mpc_solvers as S
    out = {}
    for N in Ns:
        IL = S.build_ilqr(M, N, P.dt, JX)
        Xref = C.ref_window(x0, N, path, M, P, 1, 0.0)[:M.n, :]
        mc = C.mobs_center(M.mobs, 0.0)
        IL.rollout_cost(x0, np.zeros((M.m, N)), Xref, mc)
        reps = []
        for _ in range(REPS):
            _, trace, status, Jbest = mppi_solve_traced(M, P, N, K, x0, Xref, mc, IL.rollout_cost)
            reps.append(dict(trace=trace, status=status))
        tr = reps[0]["trace"]
        Jmin = min(r["J"] for r in tr[:-1])
        ess = np.array([r["ess"] for r in tr[:-1]])
        print(f"  mppi{K} N={N:4d}: updates {len(tr) - 1:3d}  floor J {Jmin:.6f}  "
              f"ESS med {np.median(ess):.0f}/{K}  t_end(med) "
              f"{1e3 * float(np.median([r['trace'][-1]['t'] for r in reps])):8.2f} ms")
        out[N] = dict(reps=reps, N=N)
    return out


def make_uextract(JX, N, dt):
    """ZOH control sequence induced by a costate iterate: u_k = ustar(x_k, p_k) along the canonical
    rollout.  Lets shooting iterates be scored under the SAME rollout_cost functional as the U-space
    methods (protocol fix (a): one common metric for the epsilon-band)."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    @jax.jit
    def uex(p0, xc, Xr):
        z0 = jnp.concatenate([xc, p0])
        RA = Xr[:, 0:2 * N:2].T
        RM = Xr[:, 1:2 * N:2].T
        RB = Xr[:, 2:2 * N + 1:2].T

        def body(z, tr):
            u = JX.ustar(z[:7], z[7:])
            z2 = JX.rk4_canon(z, tr[0], tr[1], tr[2], dt)
            return z2, u

        _, Us = lax.scan(body, z0, (RA, RM, RB))
        return Us.T                                                  # (m, N)

    return uex


def rescore_shooting(P, M, JX, path, x0):
    """Offline re-scoring of stored PMP/M1_v2 traces under the common rollout_cost metric.
    Timings untouched; original Hopf-Lax J kept as 'Jhl'."""
    import mpc_solvers as S
    with open(PKL, "rb") as f:
        D = pickle.load(f)
    Ns = sorted({N for (m, N) in D["runs"] if m in ("m1v2", "pmp")})
    for N in Ns:
        IL = S.build_ilqr(M, N, P.dt, JX)
        Xr2 = C.ref_window(x0, N, path, M, P, 2, 0.0)
        Xref = C.ref_window(x0, N, path, M, P, 1, 0.0)[:M.n, :]
        mc = C.mobs_center(M.mobs, 0.0)
        uex = make_uextract(JX, N, P.dt)
        IL.rollout_cost(x0, np.zeros((M.m, N)), Xref, mc)
        nrec = 0
        for m in ("m1v2", "pmp"):
            e = D["runs"].get((m, N))
            if e is None:
                continue
            for rep in e["reps"]:
                for rec in rep["trace"]:
                    if "v" in rec and np.all(np.isfinite(rec["v"])):
                        if "Jhl" not in rec:
                            rec["Jhl"] = rec["J"]
                        U = np.asarray(uex(rec["v"], x0, Xr2))
                        rec["J"] = float(IL.rollout_cost(x0, U, Xref, mc)[1])
                        nrec += 1
        print(f"  rescore N={N:4d}: {nrec} iterates re-scored under rollout_cost")
    with open(PKL, "wb") as f:
        pickle.dump(D, f)
    print("rescore: saved")
    # residual-gap report at overlapping on-axis horizons
    for N in Ns:
        pe, de = D["runs"].get(("pmp", N)), D["runs"].get(("ddp", N))
        me = D["runs"].get(("m1v2", N))
        vals = []
        for tag, e in (("pmp", pe), ("m1v2", me), ("ddp", de)):
            if e and e["reps"][0]["status"] == "converged":
                vals.append((tag, e["reps"][0]["trace"][-1]["J"]))
        if len(vals) >= 2:
            js = [v for _, v in vals]
            gap = (max(js) - min(js)) / max(min(js), 1e-12)
            print(f"  N={N:4d}: " + "  ".join(f"{t}={v:.6f}" for t, v in vals) +
                  f"   spread {100 * gap:.2f}%")


def iterbench(P, M, JX, path, x0, Ns, reps=30):
    """WORST-CASE per-iteration cost, benchmarked STRUCTURALLY: median of `reps` prewarmed
    executions of each method's maximal-content iteration branch (never a max over observed
    wall-clock times).  Convention: the bounded line-search/damping loops are executed at their
    observed accepted-first-trial content (1 trial); loop bounds (7 / 20) are stated in text.
      m1v2: the KICK-firing escape iteration, BATCHED implementation = S.all + eigh + batched
            kick side comparison (one B=2 vmapped S.J) + re-shoot S.all + eigh + floored step
            + batched alpha trials (one B=7 vmapped S.J) -- fixed content, no trial convention needed
      pmp : S.Jr + S.res + normal-equation solve + 1 accepted LM trial (1x S.res)
      ddp : backward pass + batched-alpha forward pass (the line search is vectorized INTO forward)
    Collocation (uniform IPOPT iterations) and MPPI (uniform updates) keep trace-based estimates."""
    import mpc_solvers as S
    with open(PKL, "rb") as f:
        D = pickle.load(f)
    D.setdefault("iterbench", {})
    n = M.n
    print(f"{'N':>5}{'m1v2 ms':>9}{'pmp ms':>8}{'ddp ms':>8}")
    for N in Ns:
        Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
        Xr2 = C.ref_window(x0, N, path, M, P, 2, 0.0)
        Sh.warm(x0, Xr2)
        Jbat = T.ensure_jbatch(Sh)
        Jbat(np.zeros((2, n)), x0, Xr2)
        Jbat(np.zeros((7, n)), x0, Xr2)
        v = np.zeros(n)

        def m1_iter():
            res, J, Jr, Sx = Sh.all(v, x0, Xr2)
            res, Jr, Sx = np.asarray(res), np.asarray(Jr), np.asarray(Sx)
            Hc = -(Sx.T @ Jr)
            Hc = 0.5 * (Hc + Hc.T)
            Dv, Vc = np.linalg.eigh(Hc)
            vm = Vc[:, 0]
            np.asarray(Jbat(np.stack([v - 0.3 * vm, v + 0.3 * vm]), x0, Xr2))   # batched +/- comparison
            r2, J2, Jr2, Sx2 = Sh.all(v, x0, Xr2)
            r2, Jr2, Sx2 = np.asarray(r2), np.asarray(Jr2), np.asarray(Sx2)
            g = -Sx2.T @ r2
            H2 = -(Sx2.T @ Jr2)
            H2 = 0.5 * (H2 + H2.T)
            D2, V2 = np.linalg.eigh(H2)
            step = -V2 @ ((V2.T @ g) / np.maximum(np.abs(D2), 1e-8))
            np.asarray(Jbat(v[None, :] + T.M1V2_ESC_ALPHAS[:, None] * step[None, :], x0, Xr2))

        def pmp_iter():
            rr = np.asarray(Sh.res(v, x0, Xr2))
            Jr = np.asarray(Sh.Jr(v, x0, Xr2))
            A = Jr.T @ Jr
            step = np.linalg.solve(A + 1e-6 * np.eye(n), -Jr.T @ rr)
            np.asarray(Sh.res(v + step, x0, Xr2))

        DP = S.build_ddp(M, N, P.dt, JX)
        Xref = C.ref_window(x0, N, path, M, P, 1, 0.0)[:M.n, :]
        mc = C.mobs_center(M.mobs, 0.0)
        DP.warm(x0, Xref, mc)
        U0 = np.zeros((M.m, N))
        X0r = np.asarray(DP.rollout_cost(x0, U0, Xref, mc)[0])

        def ddp_iter():
            kff, Kk, dV1, Dv2_, bad = DP.backward(X0r, U0, Xref, mc, 1e-3)
            Xn, Un, Jn = DP.forward(x0, X0r, U0, kff, Kk, Xref, mc, S.DDP_ALPHAS)
            np.asarray(Jn)

        row = []
        for name, fn in (("m1v2", m1_iter), ("pmp", pmp_iter), ("ddp", ddp_iter)):
            fn(); fn()                                               # warm this exact content
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            D["iterbench"][(name, N)] = 1e3 * float(np.median(ts))
            row.append(D["iterbench"][(name, N)])
        with open(PKL, "wb") as f:
            pickle.dump(D, f)
        print(f"{N:>5}{row[0]:>9.2f}{row[1]:>8.2f}{row[2]:>8.2f}")
    print("iterbench: saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="m1v2")
    ap.add_argument("--Ns", default=",".join(str(n) for n in NS_DEFAULT))
    ap.add_argument("--rescore", action="store_true",
                    help="re-score stored shooting traces under rollout_cost, then exit")
    ap.add_argument("--iterbench", action="store_true",
                    help="structural worst-case per-iteration benchmark, then exit")
    args = ap.parse_args()
    Ns = [int(s) for s in args.Ns.split(",")]
    methods = args.methods.split(",")

    P, M, JX, path = build_env()
    x0, x0_info = get_x0()
    print(f"x0 ({x0_info}):\n  {np.array2string(x0, precision=6)}")
    print(f"  gap to wall shell: {10.4 - 0.8 - x0[0]:.3f} m;  scene: static wall at (10.4, 0, 0.5)")

    if args.rescore:
        rescore_shooting(P, M, JX, path, x0)
        return
    if args.iterbench:
        iterbench(P, M, JX, path, x0, Ns)
        return

    D = {}
    if os.path.exists(PKL):
        with open(PKL, "rb") as f:
            D = pickle.load(f)
    D["x0"], D["x0_info"] = x0, x0_info
    D["scene"] = dict(wall_c=(10.4, 0.0, 0.5), dt=P.dt, note="v6 params, obs-2 only, static")
    D.setdefault("runs", {})

    for method in methods:
        if method in ("m1v2", "pmp"):
            res = run_shooting(method, Ns, P, M, JX, path, x0)
        elif method == "ddp":
            res = run_ddp(Ns, P, M, JX, path, x0)
        elif method == "coll":
            res = run_coll(Ns, P, M, JX, path, x0)
        elif method.startswith("mppi"):
            res = run_mppi(int(method[4:]), Ns, P, M, JX, path, x0)
        else:
            raise SystemExit(f"unknown method '{method}'")
        for N, entry in res.items():
            D["runs"][(method, N)] = entry
        with open(PKL, "wb") as f:
            pickle.dump(D, f)
        print(f"{method}: saved {len(res)} horizons into {PKL}")


if __name__ == "__main__":
    main()

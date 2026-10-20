"""
mpc_testbed_baselines_int.py -- the baseline MPC methods on K INTERACTING agents (pair collision penalty).

Same protocol as mpc_testbed_multi_int.py (Hopf-Lax-MPC): frozen scene and column layout
(mpc_multi_int.make_multi_scene_int), horizon N = 220 for every method, NO budget, iteration cap --cap
(default 1000), pair penalty Rc 0.8 / Wc 500 / wc 0.1, true cost and per-fine-step minimum distance by
mpc_multi_int.apply_step_int, run until every agent is at its goal or --maxcyc cycles (default 5000 = 100 s).

Methods (each a copy of its N-sweep runner, N_sweep/mpc_testbed_N_sweep.py, with the joint model swapped in):
    PMP_Shooting       N-sweep pmp_shooting_solve_counted on the coupled shooting object build_ss_int
    Vanilla_Hopf_Lax   mpc_solvers.vanilla_hopf_lax_solve on build_ss_int (exact 7K x 7K Hessian), tol 1e-4
    DDP                N-sweep ddp_run_counted on mpc_baselines_int.build_ddp_int (joint DDP)
    MPPI_12288         mpc_baselines_int.MPPIInt with 12288 samples (joint CUDA kernel)
    MPPI_20480         the same with 20480 samples
The single-agent DDP_YSHIFT wall nudge of the moving-wall testbed is not used (frozen scene).

USAGE
    python mpc_testbed_baselines_int.py --selftest
    python mpc_testbed_baselines_int.py --method PMP_Shooting,DDP,Vanilla_Hopf_Lax,MPPI_12288,MPPI_20480 --K 2
"""
from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ("core", "closed_loop", "N_sweep", "multi_agent")]
import mpc_core as C
import mpc_solvers as S
import mpc_testbed as T                   # CF compile flag
import mpc_multi as MU
import mpc_testbed_multi as TM
import mpc_testbed_N_sweep as NS          # counted PMP / DDP solver copies
import mpc_multi_int as MI
import mpc_baselines_int as BI

PKL = "results_baselines_int.pkl"
N_DEFAULT = 220
CF = T.CF
METHODS = ["PMP_Shooting", "DDP", "Vanilla_Hopf_Lax", "MPPI_12288", "MPPI_20480"]
VANILLA_TOL = 1e-4


def make_env(cap, maxcyc=None):
    P = TM.make_env(cap, maxcyc)                                       # scene parameters, no budget, lm_maxit
    P.ilqr_Kmax = int(cap)
    return P


# ======================================================================================================
#  METHOD ADAPTERS: init(x, t) compiles outside the timed loop; step(x, t, tc) is the timed solve and
#  returns (u (3K,), p0 (7K,) or None, iters, status, extras dict)
# ======================================================================================================
def make_method(method, K, N, P, M, JX, MK, col):
    n = M.n

    def xr2(x, t):
        return MU.ref_window_multi(x, N, MK, M, P, 2, t)

    def xr1(x, t):
        return BI.joint_xref(MU.ref_window_multi(x, N, MK, M, P, 1, t), n)

    if method in ("PMP_Shooting", "Vanilla_Hopf_Lax"):
        Sh = MI.build_ss_int(M, K, N, P.dt, JX, col=col, napply=P.napply)
        st = SimpleNamespace(seed=np.zeros(K * n))

        def init(x):
            Xr0 = xr2(x, 0.0)
            Sh.warm(x, Xr0)
            if method == "Vanilla_Hopf_Lax":
                S.ensure_jbatch(Sh)(np.zeros((7, K * n)), x, Xr0)
                Hf = S.ensure_hessian(Sh)
                np.asarray(Hf(np.zeros(K * n), x, Xr0))

        def step(x, t, tc):
            Xr = xr2(x, t)
            ts = time.perf_counter()
            if method == "PMP_Shooting":
                p0, itk, nr, status, ntr, mu = NS.pmp_shooting_solve_counted(Sh, st.seed, x, Xr, P, tc)
                ex = dict(nres=nr, ntrial=ntr, mu=mu)
            else:
                PP = SimpleNamespace(tol=VANILLA_TOL, maxit=P.lm_maxit)
                p0, itk, info = S.vanilla_hopf_lax_solve(Sh, st.seed, x, Xr, PP, tc, P.budget)
                status = info.status
                ex = dict(ngrad=info.norm_grad, nacc=info.nacc, nrej=info.nrej, t_hess=info.t_hess, lminH=info.lminH)
            ex["_tsolve"] = time.perf_counter() - ts
            u = MI.ustar_multi(x, p0, M, JX)
            st.seed = np.asarray(Sh.roll_costate(p0, x, Xr))[:, P.napply]
            return u, p0, itk, status, ex

    elif method == "DDP":
        DP = BI.build_ddp_int(M, K, N, P.dt, JX, col)
        st = SimpleNamespace(U=None)

        def init(x):
            DP.warm(x, xr1(x, 0.0), C.mobs_center(M.mobs, 0.0))

        def step(x, t, tc):
            Xref = xr1(x, t)
            mc = C.mobs_center(M.mobs, t)
            Uinit = None if st.U is None else S.shift_U(st.U, P.napply)
            ts = time.perf_counter()
            p0, st.U, itk, info = NS.ddp_run_counted(DP, x, Xref, mc, Uinit, P, tc)
            tsol = time.perf_counter() - ts
            u = MI.ustar_multi(x, p0, M, JX)
            return u, p0, itk, info.status, dict(nbad=info.nbad, nrej=info.nrej, mu=info.mu, dV1=info.dV1, J=info.J,
                                                 _tsolve=tsol)

    elif method.startswith("MPPI_"):
        ctrl = BI.MPPIInt(M, P, N=N, K=int(method[5:]), KA=K, col=col, lam=P.lam_mppi, alpha=P.alpha_mppi,
                          iters=P.iters_mppi, seed=0, ess_target=P.ess_target_mppi)

        def init(x):
            ctrl.warm(x, xr1(x, 0.0), C.mobs_center(M.mobs, 0.0))

        def step(x, t, tc):
            Xref, mc = xr1(x, t), C.mobs_center(M.mobs, t)
            ts = time.perf_counter()
            u, U, ess = ctrl.solve(x, Xref, mc)
            tsol = time.perf_counter() - ts
            ctrl.shift(P.napply)
            return np.asarray(u), None, int(P.iters_mppi), "ok", dict(ess=float(ess), lam=float(ctrl.lam), _tsolve=tsol)
    else:
        raise SystemExit(f"unknown method {method!r}; use {', '.join(METHODS)}")
    return init, step


# ======================================================================================================
#  CLOSED LOOP (common to every method; records as mpc_testbed_multi_int.run_int)
# ======================================================================================================
def run_method(method, K, N, P, col, every=100):
    P, M, JX, MK = MI.make_multi_scene_int(P, K)
    n = M.n
    label = f"{method} K={K}"
    init, step = make_method(method, K, N, P, M, JX, MK, col)
    x = MK.x0.copy()
    t = 0.0
    t0 = time.perf_counter()
    init(x)
    print(f"[{label}] compiled + warmed in {time.perf_counter() - t0:.1f} s", flush=True)
    Jreal, Jpen, Jpair = np.zeros(K), np.zeros(K), np.zeros((K, K))
    treach = np.full(K, np.nan)
    XL, UL, PL, TL = [x.copy()], [], [], []
    TC, TS, RC, ITS, ST, DMIN, DPAIR = [], [], [], [], [], [], []
    EX = {}
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            tc = time.perf_counter()
            u, p0, itk, status, ex = step(x, t, tc)
            TC.append(time.perf_counter() - tc)                          # whole cycle (reference window + solve + u)
            TS.append(ex.pop("_tsolve"))                                 # solver call only (as run_int)
            RC.append(CF.fired)
            ITS.append(int(itk)); ST.append(status); TL.append(t)
            for kk, vv in ex.items():
                EX.setdefault(kk, []).append(vv)
            if not np.all(np.isfinite(u)):
                raise RuntimeError("non-finite control")
            if c % every == 1:
                its = np.asarray(ITS); ms = 1e3 * np.asarray(TS)
                print(f"[{label}] c{c:4d} t{t:5.2f} | agents xy {np.round(x.reshape(K, n)[:, :2], 2).tolist()} | dmin so far "
                      f"{(min(DMIN) if DMIN else np.inf):5.3f} | iters mean {its.mean():6.2f} max {its.max():4d} | solve ms "
                      f"mean {ms.mean():7.2f} max {ms.max():7.1f} | non-conv {sum(s not in ('converged', 'ok') for s in ST)}",
                      flush=True)
            x, Jreal, Jpen, Jpair, at_goal, t, dmin, pmin = MI.apply_step_int(
                x, u, MU.ref_window_multi(x, N, MK, M, P, 1, t), Jreal, Jpen, Jpair, M, MK, P, t, JX, col,
                done=~np.isnan(treach))
            DMIN.append(dmin); DPAIR.append(pmin if pmin is not None else (-1, -1))
            XL.append(x.copy()); UL.append(np.asarray(u).copy()); PL.append(np.zeros(K * n) if p0 is None else np.asarray(p0).copy())
            newly = at_goal & np.isnan(treach)
            treach[newly] = t
            if np.all(at_goal):
                break
        except KeyboardInterrupt:
            raise
        except Exception as e_:                                       # noqa: BLE001
            aborted, err = True, repr(e_)
            print(f"[{label}] ABORTED at cycle {c}: {err}", flush=True)
            break
    its = np.asarray(ITS, dtype=int); ts = np.asarray(TS)
    dm = np.asarray(DMIN)
    e = dict(method=method, K=K, N=N, label=label, col=vars(col).copy(), starts=np.array(MK.starts),
             goals=np.array(MK.goals), X=np.array(XL).T.reshape(K, n, -1),
             U=np.array(UL).T.reshape(K, M.m, -1) if UL else np.zeros((K, M.m, 0)),
             Pc=np.array(PL).T.reshape(K, n, -1) if PL else np.zeros((K, n, 0)),
             tlog=np.asarray(TL), tcyc=np.asarray(TC), tsolve=ts, recompile=np.asarray(RC, dtype=bool),
             iters=its, status=list(ST), extra={k: np.asarray(v) for k, v in EX.items()},
             dmin=dm, dpair=np.asarray(DPAIR, dtype=int), Jreal=Jreal, Jpen=Jpen, Jpair=Jpair, Jtrue=Jreal + Jpen,
             Jpair_total=float(np.sum(Jpair)), t_reach=treach, all_reached=bool(np.all(~np.isnan(treach))),
             ncyc=len(ITS), aborted=aborted, error=err, wall_s=time.perf_counter() - t_start,
             stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
             meta=dict(dt=P.dt, dt_apply=P.dt_apply, napply=P.napply, maxcyc=P.maxcyc, budget=P.budget,
                       lm_tol=P.lm_tol, lm_maxit=P.lm_maxit, ilqr_tol=P.ilqr_tol, ilqr_Kmax=P.ilqr_Kmax,
                       vanilla_tol=VANILLA_TOL, reachtol=P.reachtol, layout="columns-mirrored-goals",
                       ess_target_mppi=P.ess_target_mppi, alpha_mppi=P.alpha_mppi, iters_mppi=P.iters_mppi),
             paths=[dict(pts=p.pts, s=p.s, th=p.th, len=p.len) for p in MK.paths],
             obs=dict(center=M.obs.center, ax=M.obs.ax, ay=M.obs.ay), mobs=dict(c0=M.mobs.c0, ax=M.mobs.ax, ay=M.mobs.ay))
    pos = its[1:] > 0
    print(f"[{label}] done: {len(ITS)} cycles, all reached {e['all_reached']}, t_reach {np.round(treach, 2).tolist()}, "
          f"J per agent {np.round(e['Jtrue'], 3).tolist()} + pair {e['Jpair_total']:.3f} | min distance "
          f"{dm.min() if dm.size else np.inf:.3f} m | iters mean {its.mean():.2f} | solve ms mean {1e3 * ts[1:].mean():.2f} "
          f"| ms/iter {1e3 * np.mean(ts[1:][pos] / its[1:][pos]) if pos.any() else np.nan:.2f} | status "
          f"{ {s: ST.count(s) for s in sorted(set(ST))} } | wall {e['wall_s']:.0f} s", flush=True)
    return e


def print_table(entries):
    print(f"\n{'run':<24}{'cyc':>6}{'reached':>8}{'t_last':>7}{'sumJ':>9}{'Jpair':>9}{'dmin':>7}{'it mean':>8}{'it max':>7}"
          f"{'%conv':>7}{'solve ms':>9}{'max ms':>9}{'ms/iter':>8}")
    print("-" * 112)
    for e in entries:
        its, ts = e["iters"][1:], e["tsolve"][1:]
        pos = its > 0
        conv = 100 * np.mean([s in ("converged", "ok") for s in e["status"]])
        tl = np.nanmax(e["t_reach"]) if e["all_reached"] else np.nan
        reached = "all" if e["all_reached"] else f"{int(np.sum(~np.isnan(e['t_reach'])))}/{e['K']}"
        print(f"{e['label']:<24}{e['ncyc']:>6d}{reached:>8}{tl:>7.2f}{e['Jtrue'].sum():>9.3f}{e['Jpair_total']:>9.3f}"
              f"{e['dmin'].min() if e['dmin'].size else np.inf:>7.3f}{its.mean():>8.2f}{its.max():>7d}{conv:>7.1f}"
              f"{1e3 * ts.mean():>9.2f}{1e3 * ts.max():>9.1f}{(1e3 * np.mean(ts[pos] / its[pos]) if pos.any() else np.nan):>8.2f}"
              f"{' ABORTED' if e['aborted'] else ''}")
    print("%conv: converged cycles (MPPI: 'ok', one sampling update per cycle); timing: cycle 1 excluded")


# ======================================================================================================
#  SELF-TEST
# ======================================================================================================
def selftest(P, N=220):
    ok = True
    rng = np.random.default_rng(1)

    def check(name, a, b, tol, rel=False):
        nonlocal ok
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            d = np.inf
        else:
            d = float(np.max(np.abs(a - b))) if a.size else 0.0
            if rel:
                d /= max(1.0, float(np.max(np.abs(b))))
        good = a.shape == b.shape and d <= tol
        tag = "IDENTICAL" if (a.shape == b.shape and np.array_equal(a, b)) else f"max|diff|{'(rel)' if rel else ''} {d:.2e}"
        print(f"  {name:<66s} {tag} -> {'OK' if good else 'BAD'}")
        ok &= good

    col = MI.make_col()
    P.budget = np.inf

    def scene(K):
        return MI.make_multi_scene_int(make_env(1000), K)

    def real_state():
        """(x, applied costate, t, distance) at the closest approach of the recorded HL K=2 coupled run."""
        try:
            R = pickle.load(open("results_multi_int.pkl", "rb"))["runs"]
            e = R[(2, N, col.Rc, col.Wc, col.wc)]
        except Exception:                                             # noqa: BLE001
            return None
        c = int(np.argmin(e["dmin"]))
        return e["X"][:, :, c].reshape(-1), e["Pc"][:, :, c].reshape(-1), float(e["tlog"][c]), float(e["dmin"][c])

    def close_pair(MK, M, K, gap=0.5):
        X = MK.x0.reshape(K, M.n).copy()
        X[:, 0] = 6.0
        X[:, 1] = 0.25 + gap * (np.arange(K) - 0.5 * (K - 1))
        return X.reshape(-1)

    # ---------------- DDP ----------------
    print("== selftest 1: DDP joint model, K = 1 vs mpc_solvers.build_ddp (single agent)")
    P1, M, JX, MK = scene(1)
    D0 = S.build_ddp(M, N, P1.dt, JX)
    D1 = BI.build_ddp_int(M, 1, N, P1.dt, JX, col)
    x = MK.x0.copy()
    Xref = BI.joint_xref(MU.ref_window_multi(x, N, MK, M, P1, 1, 0.0), M.n)
    mc = C.mobs_center(M.mobs, 0.0)
    U = rng.normal(size=(3, N)) * 0.5
    Xa, Ja = D0.rollout_cost(x, U, Xref, mc); Xb, Jb = D1.rollout_cost(x, U, Xref, mc)
    check("rollout X", Xb, Xa, 1e-12); check("cost J", Jb, Ja, 1e-10, rel=True)
    ba = D0.backward(Xa, U, Xref, mc, 1e-3); bb = D1.backward(Xb, U, Xref, mc, 1e-3)
    check("backward kff", bb[0], ba[0], 1e-9, rel=True); check("backward K", bb[1], ba[1], 1e-9, rel=True)
    check("backward dV1, dV2", [bb[2], bb[3]], [ba[2], ba[3]], 1e-9, rel=True)
    check("costate0", D1.costate0(Xb, U, Xref, mc), D0.costate0(Xa, U, Xref, mc), 1e-10, rel=True)
    P1.budget = np.inf
    ra = NS.ddp_run_counted(D0, x, Xref, mc, None, P1, time.perf_counter())
    rb = NS.ddp_run_counted(D1, x, Xref, mc, None, P1, time.perf_counter())
    check(f"ddp_run p0 (it {ra[2]} vs {rb[2]})", rb[0], ra[0], 1e-8, rel=True)
    ok &= ra[2] == rb[2]

    print("== selftest 2: DDP K = 2 with Wc = 0: block-diagonal gains equal the two solo backward passes")
    P2, M, JX, MK2 = scene(2)
    Dz = BI.build_ddp_int(M, 2, N, P2.dt, JX, MI.make_col(Wc=0.0))
    x2 = MK2.x0.copy()
    Xr2 = BI.joint_xref(MU.ref_window_multi(x2, N, MK2, M, P2, 1, 0.0), M.n)
    U2 = rng.normal(size=(6, N)) * 0.5
    X2, _ = Dz.rollout_cost(x2, U2, Xr2, mc)
    kf2, K2, _, _, _ = Dz.backward(X2, U2, Xr2, mc, 1e-3)
    for k in range(2):
        Xs, _ = D0.rollout_cost(x2[7 * k:7 * k + 7], U2[3 * k:3 * k + 3], Xr2[7 * k:7 * k + 7], mc)
        kfs, Ks, _, _, _ = D0.backward(Xs, U2[3 * k:3 * k + 3], Xr2[7 * k:7 * k + 7], mc, 1e-3)
        check(f"agent {k}: kff block", np.asarray(kf2)[3 * k:3 * k + 3], kfs, 1e-9, rel=True)
        check(f"agent {k}: K diagonal block", np.asarray(K2)[:, 3 * k:3 * k + 3, 7 * k:7 * k + 7], Ks, 1e-9, rel=True)
    check("K off-diagonal blocks == 0", np.concatenate([np.asarray(K2)[:, 0:3, 7:14].ravel(), np.asarray(K2)[:, 3:6, 0:7].ravel()]),
          np.zeros(2 * N * 21), 1e-14)

    print("== selftest 3: DDP K = 2 derivatives with ACTIVE coupling (agents 0.5 m apart)")
    D2 = BI.build_ddp_int(M, 2, N, P2.dt, JX, col)
    xc = close_pair(MK2, M, 2)
    Xrc = BI.joint_xref(MU.ref_window_multi(xc, N, MK2, M, P2, 1, 0.0), M.n)
    posf = jnp.asarray(xc[D2.pidx])
    pen, g = D2.obs_all(posf, mc)
    gad = jax.grad(lambda p: D2.obs_all(p, mc)[0])(posf)
    print(f"  pair distance {np.linalg.norm(xc[0:3] - xc[7:10]):.2f} m, barrier+pair penalty {float(pen):.2f}")
    check("analytic barrier+pair gradient == jax.grad", g, gad, 1e-12, rel=True)
    Uc = rng.normal(size=(6, N)) * 0.3
    Xc, _ = D2.rollout_cost(xc, Uc, Xrc, mc)
    p0 = np.asarray(D2.costate0(Xc, Uc, Xrc, mc))
    dJdx0 = np.asarray(jax.grad(lambda x0: D2.rollout_cost(x0, Uc, Xrc, mc)[1])(jnp.asarray(xc)))
    check("costate0 == dJ/dx0 (autodiff through rollout + cost)", p0, dJdx0, 1e-6, rel=True)
    # feedback-gain coupling: at a REAL coupled state, the closest approach of the recorded Hopf_Lax_MPC K = 2
    # run (results_multi_int.pkl), with mu raised x4 until Quu is PD (as ddp_run does)
    st_real = real_state()
    if st_real is not None:
        xg, pg, tg, dg = st_real
        Xrg = BI.joint_xref(MU.ref_window_multi(xg, N, MK2, M, P2, 1, tg), M.n)
        Ug = np.zeros((6, N))
        Xg, _ = D2.rollout_cost(xg, Ug, Xrg, mc)
        mu = 1e-3
        for _ in range(30):
            kf, Kc, dV1, dV2, bad = D2.backward(Xg, Ug, Xrg, mc, mu)
            if not bool(bad):
                break
            mu *= 4
        Kc = np.asarray(Kc)
        o = float(np.linalg.norm(Kc[:, 0:3, 7:14])) + float(np.linalg.norm(Kc[:, 3:6, 0:7]))
        Dzb = BI.build_ddp_int(M, 2, N, P2.dt, JX, MI.make_col(Wc=0.0))
        Kz = np.asarray(Dzb.backward(Xg, Ug, Xrg, mc, mu)[1])
        oz = float(np.linalg.norm(Kz[:, 0:3, 7:14])) + float(np.linalg.norm(Kz[:, 3:6, 0:7]))
        good = (not bool(bad)) and o > 1e-6 and oz == 0.0
        print(f"  backward at the HL closest approach ({dg:.2f} m, t {tg:.2f} s), mu {mu:.1e}: |off-diagonal gains| "
              f"{o:.3e} (Wc=0: {oz:.1e}) -> {'OK' if good else 'BAD'}")
        ok &= good

    # ---------------- MPPI ----------------
    if S.HAVE_GPU:
        import cupy as cp
        print("== selftest 4: MPPI joint kernel: K = 1 vs the single-agent kernel; K = 2 vs a NumPy reference")
        ctrl0 = S.MPPI(M, P1, N=N, K=256, seed=0)
        ctrl1 = BI.MPPIInt(M, P1, N=N, K=256, KA=1, col=col, seed=0)
        eps = cp.asarray(rng.normal(size=(256, N, 3)) * 0.5)
        U1 = rng.normal(size=(3, N)) * 0.3
        x1 = MK.x0.copy()
        Xr1 = BI.joint_xref(MU.ref_window_multi(x1, N, MK, M, P1, 1, 0.0), M.n)
        s0 = cp.asnumpy(ctrl0._sample_costs(x1, Xr1, mc, U1, eps)).copy()
        s1 = cp.asnumpy(ctrl1._sample_costs(x1, Xr1, mc, U1, eps)).copy()
        check("K=1 sample costs == single-agent kernel (rel)", s1, s0, 1e-12, rel=True)

        def ref_cost(xf, U, E, Xref, gs, cl, KA, nunb, kidx):
            """NumPy reference of the joint sample cost (the kernel's arithmetic, in float64)."""
            xx = xf.reshape(KA, 7).copy()
            Sk = 0.0
            for t in range(N):
                v = E[t] if kidx < nunb else U[:, t] + E[t]
                un = np.clip(v, -np.tile(M.umax, KA), np.tile(M.umax, KA))
                Sk += float(np.sum(gs * U[:, t] * v))
                q = 0.0
                for a in range(KA):
                    xx[a] = np.asarray(JX.rk4_state(xx[a], un[3 * a:3 * a + 3], P2.dt))
                    d = xx[a] - Xref[7 * a:7 * a + 7, t + 1]
                    q += float(d @ M.Q @ d + un[3 * a:3 * a + 3] @ M.R @ un[3 * a:3 * a + 3])
                    q += float(JX.obstacle(xx[a][:3])[0]) + float(JX.mobs_pen(xx[a][:3], mc)[0])
                q += MI.pair_penalty_np(xx[:, :3], cl)[0]
                Sk += q * P2.dt
            d = xx - Xref[:, N].reshape(KA, 7)
            return Sk + float(np.sum((d @ M.QT) * d))

        for cl, nm in ((col, "Wc=500 (pair active)"), (MI.make_col(Wc=0.0), "Wc=0")):
            ctrl2 = BI.MPPIInt(M, P2, N=N, K=256, KA=2, col=cl, seed=0)
            E2 = rng.normal(size=(256, N, 6)) * 0.5
            Uk = rng.normal(size=(6, N)) * 0.3
            s2 = cp.asnumpy(ctrl2._sample_costs(xc, Xrc, mc, Uk, cp.asarray(E2))).copy()
            gs = cp.asnumpy(ctrl2.d_gsig)
            for kidx in (0, 7, 200):
                sr = ref_cost(xc, Uk, E2[kidx], Xrc, gs, cl, 2, ctrl2.n_unbiased, kidx)
                check(f"K=2 {nm}: sample {kidx} cost == NumPy reference (rel)", s2[kidx], sr, 1e-10, rel=True)
    else:
        print("== selftest 4: MPPI skipped (no GPU)")

    # ---------------- PMP / Vanilla on the coupled shooting object ----------------
    print("== selftest 5: PMP_Shooting and Vanilla_Hopf_Lax on build_ss_int at the HL closest approach (K = 2)")
    if st_real is None:
        print("  results_multi_int.pkl has no K=2 coupled run: skipped")
    else:
        Sh = MI.build_ss_int(M, 2, N, P2.dt, JX, col=col, napply=P2.napply)
        Xr = MU.ref_window_multi(xg, N, MK2, M, P2, 2, tg)
        Sh.warm(xg, Xr)
        P2.budget = np.inf
        PP = SimpleNamespace(tol=P2.lm_tol, maxit=P2.lm_maxit)
        PV = SimpleNamespace(tol=VANILLA_TOL, maxit=P2.lm_maxit)
        ph, ith, _ = S.hopf_lax_mpc_solve(Sh, pg.copy(), xg, Xr, PP, P2.hopf_lax_mpc_epsc, S.HOPF_LAX_MPC_ETA_S,
                                          P2.hopf_lax_mpc_alpha, S.HOPF_LAX_MPC_DELTA, time.perf_counter(), np.inf)
        rh, Jh, Jrh, Sxh = [np.asarray(a) for a in Sh.all(ph, xg, Xr)]
        gh = -(Sxh.T @ rh)
        print(f"  HL re-solve from its applied costate: it {ith}, |res| {np.linalg.norm(rh):.1e}, |grad| {np.linalg.norm(gh):.1e}, "
              f"|offdiag Jr| {float(np.linalg.norm(Jrh[:7, 7:]) + np.linalg.norm(Jrh[7:, :7])):.2e}")
        pp, itp, nrp, stp, _, _ = NS.pmp_shooting_solve_counted(Sh, ph.copy(), xg, Xr, P2, time.perf_counter())
        pv, itv, inv = S.vanilla_hopf_lax_solve(Sh, ph.copy(), xg, Xr, PV, time.perf_counter(), np.inf)
        print(f"  warm start at the HL solution: PMP it {itp} {stp} |res| {nrp:.1e}; Vanilla it {itv} {inv.status} "
              f"|grad| {inv.norm_grad:.1e}")
        good = stp == "converged" and inv.status == "converged"
        print(f"  both accept the HL stationary point -> {'OK' if good else 'BAD'}")
        ok &= good
        check("PMP from the HL solution stays there (rel)", pp, ph, 1e-4, rel=True)
        check("Vanilla from the HL solution stays there (rel)", pv, ph, 1e-4, rel=True)
        pp0, itp0, nrp0, stp0, _, _ = NS.pmp_shooting_solve_counted(Sh, np.zeros(14), xg, Xr, P2, time.perf_counter())
        pv0, itv0, inv0 = S.vanilla_hopf_lax_solve(Sh, np.zeros(14), xg, Xr, PV, time.perf_counter(), np.inf)
        print(f"  (info, not asserted) cold start p=0: PMP it {itp0} {stp0} |dp to HL| {np.linalg.norm(pp0 - ph):.1e}; "
              f"Vanilla it {itv0} {inv0.status} |dp to HL| {np.linalg.norm(pv0 - ph):.1e}")

    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=",".join(METHODS))
    ap.add_argument("--K", default="2")
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--Rc", type=float, default=MI.RC_DEFAULT)
    ap.add_argument("--Wc", type=float, default=MI.WC_DEFAULT)
    ap.add_argument("--wc", type=float, default=MI.WSH_DEFAULT)
    ap.add_argument("--maxcyc", type=int, default=5000)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--out", default=PKL)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv[1:])
    if args.selftest:
        return selftest(make_env(args.cap, args.maxcyc), args.N)
    col = MI.make_col(args.Rc, args.Wc, args.wc)
    D = {}
    if os.path.exists(args.out):
        with open(args.out, "rb") as f:
            D = pickle.load(f)
    D.setdefault("runs", {})
    D["config"] = dict(N=args.N, cap=args.cap, maxcyc=args.maxcyc, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                       host=platform.node())
    done = []
    for K in [int(s) for s in args.K.split(",")]:
        for method in args.method.split(","):
            key = (method, K, args.N, col.Rc, col.Wc, col.wc)
            if key in D["runs"] and not args.redo:
                print(f"-- {key} already in {args.out}; skipping (--redo to recompute)")
                done.append(D["runs"][key]); continue
            print(f"\n#### {method} K={K} N={args.N}  ({time.strftime('%H:%M:%S')})", flush=True)
            e = run_method(method, K, args.N, make_env(args.cap, args.maxcyc), col, every=args.print_every)
            D["runs"][key] = e
            tmp = args.out + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(D, f)
            os.replace(tmp, args.out)
            done.append(e)
            print(f"#### saved {args.out}", flush=True)
    print_table(done)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

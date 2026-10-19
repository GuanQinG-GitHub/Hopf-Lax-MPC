"""
mpc_testbed_multi_int.py -- Hopf-Lax-MPC on K INTERACTING agents (pairwise collision penalty) as ONE
coupled problem.

Protocol (per K): the frozen-obstacle scene and agent layout of multi_agent (mpc_multi.make_multi_scene),
horizon N = 220, NO per-cycle budget, iteration cap --cap (default 1000), the paper's solver
mpc_solvers.hopf_lax_mpc_solve run UNCHANGED on the coupled shooting object
(mpc_multi_int.build_ss_int), closed loop until every agent is at its goal or --maxcyc cycles.  Arrived
agents stay in the problem (they keep flying and remain obstacles for the others).  On top of the
multi_agent records: per-cycle minimum pairwise distance (over every fine step) and closest pair, the
true pair-collision cost, and the size of the off-diagonal (inter-agent) Jacobian blocks.  Results merge
into results_multi_int.pkl, key (K, N, Rc, Wc, wc) (Wc = 0 is the decoupled baseline, --nocol).

USAGE
    python mpc_testbed_multi_int.py --selftest             # reductions, derivation and generality checks
    python mpc_testbed_multi_int.py --K 2                  # coupled two-agent run (Rc 0.8, Wc 500, wc 0.1)
    python mpc_testbed_multi_int.py --K 2 --nocol          # same code with Wc = 0 (decoupled baseline)
    python mpc_testbed_multi_int.py --K 3,4 --Rc 0.8       # generality
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
import numpy as np

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ("core", "closed_loop", "N_sweep", "multi_agent")]
import mpc_core as C
import mpc_solvers as S
import mpc_testbed as T                   # CF compile flag (+ piecewise mobs_center, inert with vel = 0)
import mpc_multi as MU
import mpc_testbed_multi as TM            # make_env (same scene parameters, no budget)
import mpc_multi_int as MI
from mpc_testbed_N_sweep import hopf_lax_mpc_status

PKL = "results_multi_int.pkl"
N_DEFAULT = 220
CF = T.CF
make_env = TM.make_env


def offdiag_norm(A, K, n):
    """Frobenius norm of the inter-agent blocks of a (Kn, Kn) matrix."""
    B = np.asarray(A).copy()
    for k in range(K):
        B[k * n:(k + 1) * n, k * n:(k + 1) * n] = 0.0
    return float(np.linalg.norm(B))


def solve(Sh, v, x, Xr, P, PP, trace=None):
    return S.hopf_lax_mpc_solve(Sh, v, x, Xr, PP, P.hopf_lax_mpc_epsc, S.HOPF_LAX_MPC_ETA_S, P.hopf_lax_mpc_alpha,
                                S.HOPF_LAX_MPC_DELTA, time.perf_counter(), P.budget, trace=trace)


def run_int(K, N, P, col, every=50, label=None):
    P, M, JX, MK = MU.make_multi_scene(P, K)
    n = M.n
    label = label or f"K={K} Rc={col.Rc:g} Wc={col.Wc:g}"
    Sh = MI.build_ss_int(M, K, N, P.dt, JX, col=col, napply=P.napply)
    x = MK.x0.copy()
    t = 0.0
    Xr0 = MU.ref_window_multi(x, N, MK, M, P, 2, 0.0)
    t0 = time.perf_counter()
    Sh.warm(x, Xr0)                                                    # compile OUTSIDE the timed loop
    Jb = S.ensure_jbatch(Sh)
    Jb(np.zeros((2, K * n)), x, Xr0)
    Jb(np.zeros((7, K * n)), x, Xr0)
    print(f"[{label}] coupled shooting object (dim {K * n}) compiled + warmed in {time.perf_counter() - t0:.1f} s")
    PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
    seed = np.zeros(K * n)
    Jreal, Jpen, Jpair = np.zeros(K), np.zeros(K), np.zeros((K, K))
    treach = np.full(K, np.nan)
    XL, UL, PL, TL = [x.copy()], [], [], []
    TC, TS, RC, ITS, ST, NG = [], [], [], [], [], []
    NDESC, NESC, NKICK, NEIG, LM, NR, OFFJ, DMIN, DPAIR = [], [], [], [], [], [], [], [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            tc = time.perf_counter()
            XrK = MU.ref_window_multi(x, N, MK, M, P, 2, t)
            tr = []
            ts = time.perf_counter()
            p0, itk, info = S.hopf_lax_mpc_solve(Sh, seed, x, XrK, PP, P.hopf_lax_mpc_epsc, S.HOPF_LAX_MPC_ETA_S,
                                                 P.hopf_lax_mpc_alpha, S.HOPF_LAX_MPC_DELTA, tc, P.budget, trace=tr)
            TS.append(time.perf_counter() - ts)
            st = hopf_lax_mpc_status(tr, itk, PP.maxit)
            u = MI.ustar_multi(x, p0, M, JX)
            Pc = np.asarray(Sh.roll_costate(p0, x, XrK))
            seed = Pc[:, P.napply]
            TC.append(time.perf_counter() - tc)
            RC.append(CF.fired)
            ITS.append(itk); ST.append(st); TL.append(t)
            NG.append(float(tr[-1].get("norm_grad", np.nan)) if tr else np.nan)
            NDESC.append(info.ndesc); NESC.append(info.nesc); NKICK.append(info.nkick); NEIG.append(info.neig)
            # ---- untimed diagnostics at the applied iterate: surrogate lmin, |res|, inter-agent coupling ----
            rd, _, Jrd, Sxd = Sh.all(p0, x, XrK)
            Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
            LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
            NR.append(float(np.linalg.norm(np.asarray(rd))))
            OFFJ.append(offdiag_norm(Jrd, K, n))
            if c % every == 1 or c == 1:
                its = np.asarray(ITS); ms = 1e3 * np.asarray(TS)
                pos = x.reshape(K, n)[:, :2]
                print(f"[{label}] c{c:4d} t{t:5.2f} | agents xy {np.round(pos, 2).tolist()} | dmin so far "
                      f"{(min(DMIN) if DMIN else np.inf):5.3f} | iters mean {its.mean():5.2f} max {its.max():3d} | "
                      f"solve ms mean {ms.mean():7.2f} max {ms.max():7.1f} | ms/iter {np.mean(ms[its > 0] / its[its > 0]):6.2f} | "
                      f"non-conv {sum(s != 'converged' for s in ST)}", flush=True)
            x, Jreal, Jpen, Jpair, at_goal, t, dmin, pmin = MI.apply_step_int(
                x, u, XrK, Jreal, Jpen, Jpair, M, MK, P, t, JX, col, done=~np.isnan(treach))
            DMIN.append(dmin); DPAIR.append(pmin if pmin is not None else (-1, -1))
            XL.append(x.copy()); UL.append(u.copy()); PL.append(p0.copy())
            newly = at_goal & np.isnan(treach)
            treach[newly] = t
            if np.all(at_goal):
                break
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                       # noqa: BLE001
            aborted, err = True, repr(ex)
            print(f"[{label}] ABORTED at cycle {c}: {err}")
            break
    its = np.asarray(ITS, dtype=int); ts = np.asarray(TS); tt = np.asarray(TC)
    pos = its[1:] > 0
    dm = np.asarray(DMIN)
    e = dict(K=K, N=N, label=label, col=vars(col).copy(), agents=MK.agents,
             starts=np.array(MK.starts), goals=np.array(MK.goals),
             X=np.array(XL).T.reshape(K, n, -1), U=np.array(UL).T.reshape(K, M.m, -1) if UL else np.zeros((K, M.m, 0)),
             Pc=np.array(PL).T.reshape(K, n, -1) if PL else np.zeros((K, n, 0)),
             tlog=np.asarray(TL), tcyc=tt, tsolve=ts, recompile=np.asarray(RC, dtype=bool),
             iters=its, status=list(ST), ngrad=np.asarray(NG), ndesc=np.asarray(NDESC), nesc=np.asarray(NESC),
             nkick=np.asarray(NKICK), neig=np.asarray(NEIG), lmin=np.asarray(LM), nres=np.asarray(NR),
             offJr=np.asarray(OFFJ), dmin=dm, dpair=np.asarray(DPAIR, dtype=int),
             Jreal=Jreal, Jpen=Jpen, Jpair=Jpair, Jtrue=Jreal + Jpen, Jpair_total=float(np.sum(Jpair)),
             t_reach=treach, all_reached=bool(np.all(~np.isnan(treach))),
             ncyc=len(ITS), aborted=aborted, error=err, wall_s=time.perf_counter() - t_start,
             stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
             meta=dict(dt=P.dt, dt_apply=P.dt_apply, napply=P.napply, maxcyc=P.maxcyc, budget=P.budget,
                       lm_tol=P.lm_tol, lm_maxit=P.lm_maxit, hopf_lax_mpc_epsc=P.hopf_lax_mpc_epsc,
                       hopf_lax_mpc_alpha=P.hopf_lax_mpc_alpha, eta_s=S.HOPF_LAX_MPC_ETA_S, ridge=S.HOPF_LAX_MPC_RIDGE,
                       delta=S.HOPF_LAX_MPC_DELTA, reachtol=P.reachtol, y_spread=MU.NAGENT_Y_SPREAD, frozen_mobs=True),
             paths=[dict(pts=p.pts, s=p.s, th=p.th, len=p.len) for p in MK.paths],
             obs=dict(center=M.obs.center, ax=M.obs.ax, ay=M.obs.ay), mobs=dict(c0=M.mobs.c0, ax=M.mobs.ax, ay=M.mobs.ay))
    k = int(np.argmin(dm)) if dm.size else 0
    print(f"[{label}] done: {len(ITS)} cycles, all reached {e['all_reached']}, t_reach {np.round(treach, 2).tolist()}, "
          f"J per agent {np.round(e['Jtrue'], 3).tolist()} + pair {e['Jpair_total']:.3f} | min distance "
          f"{dm.min() if dm.size else np.inf:.3f} m (pair {tuple(e['dpair'][k]) if dm.size else None}, t {k * P.dt_apply:.2f} s) | "
          f"iters mean {its.mean():.2f} | solve ms mean {1e3 * ts[1:].mean():.2f} | ms/iter mean "
          f"{1e3 * np.mean(ts[1:][pos] / its[1:][pos]):.2f} | descent {int(np.sum(NDESC))} escape {int(np.sum(NESC))} "
          f"kicks {int(np.sum(NKICK))} eigen reads {int(np.sum(NEIG))} | wall {e['wall_s']:.0f} s")
    return e


def print_table(entries):
    print(f"\n{'run':<26}{'dim':>5}{'cyc':>5}{'reached':>8}{'t_last':>7}{'sumJ':>9}{'Jpair':>8}{'dmin':>7}{'it mean':>8}"
          f"{'it max':>7}{'%conv':>7}{'solve ms':>9}{'max ms':>8}{'ms/iter':>8}{'kicks':>6}{'esc':>5}")
    print("-" * 130)
    for e in entries:
        its, ts = e["iters"][1:], e["tsolve"][1:]
        pos = its > 0
        pi = ts[pos] / its[pos]
        conv = 100 * np.mean([s == "converged" for s in e["status"]])
        tl = np.nanmax(e["t_reach"]) if e["all_reached"] else np.nan
        reached = "all" if e["all_reached"] else f"{int(np.sum(~np.isnan(e['t_reach'])))}/{e['K']}"
        print(f"{e['label']:<26}{e['K'] * 7:>5d}{e['ncyc']:>5d}{reached:>8}{tl:>7.2f}{e['Jtrue'].sum():>9.3f}"
              f"{e['Jpair_total']:>8.3f}{e['dmin'].min() if e['dmin'].size else np.inf:>7.3f}{its.mean():>8.2f}{its.max():>7d}"
              f"{conv:>7.1f}{1e3 * ts.mean():>9.2f}{1e3 * ts.max():>8.1f}{1e3 * pi.mean():>8.2f}"
              f"{int(e['nkick'].sum()):>6d}{int(e['nesc'].sum()):>5d}{' ABORTED' if e['aborted'] else ''}")
    print("sumJ = tracking + obstacle cost summed over agents; Jpair = true pair-collision cost; dmin = minimum "
          "pairwise distance [m] over every fine step;\ntiming: cycle 1 excluded; ms/iter = solve time / iterations "
          "per cycle (cycles with 0 iterations excluded)")


# ======================================================================================================
#  SELF-TEST
# ======================================================================================================
def selftest(P, N=220):
    ok = True
    rng = np.random.default_rng(0)

    def check(name, a, b, tol=0.0, rel=False):
        nonlocal ok
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            d = np.inf
        else:
            d = float(np.max(np.abs(a - b))) if a.size else 0.0
            if rel:
                d /= max(1.0, float(np.max(np.abs(b))))
        good = a.shape == b.shape and (np.array_equal(a, b) if tol == 0.0 else d <= tol)
        tag = "IDENTICAL" if (a.shape == b.shape and np.array_equal(a, b)) else f"max|diff|{'(rel)' if rel else ''} {d:.2e}"
        print(f"  {name:<62s} {tag} -> {'OK' if good else 'BAD'}")
        ok &= good

    def scene(K):
        Pk, M, JX, MK = MU.make_multi_scene(make_env(1000), K)
        return Pk, M, JX, MK

    def clustered(MK, M, K, spread):
        """Layout states moved into a cluster around (6, 0.5) so that every pair interacts."""
        X = MK.x0.reshape(K, M.n).copy()
        X[:, 0] = 6.0 + spread * rng.uniform(-1, 1, K)
        X[:, 1] = 0.5 + spread * rng.uniform(-1, 1, K)
        return X.reshape(-1)

    PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
    col = MI.make_col()

    print("== selftest 1: exact reductions (Wc = 0 for K = 2, 3; K = 1 with the penalty ON) vs mpc_multi.build_ss_multi")
    for K, cc in ((2, MI.make_col(Wc=0.0)), (3, MI.make_col(Wc=0.0)), (1, col)):
        Pk, M, JX, MK = scene(K)
        Sa = MU.build_ss_multi(M, K, N, Pk.dt, JX, napply=Pk.napply)
        Sb = MI.build_ss_int(M, K, N, Pk.dt, JX, col=cc, napply=Pk.napply)
        x = MK.x0.copy()
        Xr = MU.ref_window_multi(x, N, MK, M, Pk, 2, 0.0)
        Sa.warm(x, Xr); Sb.warm(x, Xr); S.ensure_jbatch(Sa); S.ensure_jbatch(Sb)
        v = rng.normal(size=7 * K) * 0.3
        ra, Ja, Jra, Sxa = [np.asarray(a) for a in Sa.all(v, x, Xr)]
        rb, Jb_, Jrb, Sxb = [np.asarray(a) for a in Sb.all(v, x, Xr)]
        check(f"K={K} Wc={cc.Wc:g}: res", rb, ra, 1e-12); check(f"K={K}: J", Jb_, Ja, 1e-10)
        check(f"K={K}: Jr", Jrb, Jra, 1e-9); check(f"K={K}: Sx", Sxb, Sxa, 1e-9)
        check(f"K={K}: roll_costate", np.asarray(Sb.roll_costate(v, x, Xr)), np.asarray(Sa.roll_costate(v, x, Xr)), 1e-12)
        check(f"K={K}: roll_pred", np.asarray(Sb.roll_pred(v, x, Xr)), np.asarray(Sa.roll_pred(v, x, Xr)), 1e-12)
        pa, ia, _ = solve(Sa, np.zeros(7 * K), x, Xr, Pk, PP)
        pb, ib, _ = solve(Sb, np.zeros(7 * K), x, Xr, Pk, PP)
        check(f"K={K}: hopf_lax_mpc_solve costate (it {ia} vs {ib})", pb, pa, 1e-8)
        ok &= ia == ib

    print("== selftest 2: analytic pair gradient G vs jax.grad of sum_{i<j} C (pairs inside and outside Rc)")
    for K in (2, 3, 5):
        Q = np.column_stack([rng.uniform(0, 1.2, K), rng.uniform(0, 1.2, K), np.full(K, 0.5) + 0.05 * rng.normal(size=K)])
        pen, G = MI.pair_penalty(jax.numpy.asarray(Q), col)
        Gad = jax.grad(lambda q: MI.pair_penalty(q, col)[0])(jax.numpy.asarray(Q))
        pen_np, _ = MI.pair_penalty_np(Q, col)
        dmin, _ = MI.pair_distances(Q)
        print(f"  K={K}: min distance {dmin:.3f} m (Rc {col.Rc}), penalty {float(pen):.3f}")
        check(f"K={K}: G == grad (rel)", G, Gad, 1e-12, rel=True)
        check(f"K={K}: JAX penalty == NumPy penalty", float(pen), pen_np, 1e-12)
    check("K=1: penalty and G are exactly zero", np.asarray(MI.pair_penalty(jax.numpy.ones((1, 3)), col)[1]), np.zeros((1, 3)))

    print("== selftest 3: hand-derived costate equation == -dH/dx of the joint Hamiltonian (u held at u*)")
    for K in (2, 4):
        Pk, M, JX, MK = scene(K)
        Sb = MI.build_ss_int(M, K, N, Pk.dt, JX, col=col, napply=Pk.napply)
        x = clustered(MK, M, K, 0.4)
        Xr = MU.ref_window_multi(x, N, MK, M, Pk, 2, 0.0)
        p = rng.normal(size=7 * K)
        u = MI.ustar_multi(x, p, M, JX)
        Xa = Xr[:, :, 0]
        gH = jax.grad(Sb.hamiltonian, argnums=0)(x, p, u, Xa)
        pd = np.asarray(Sb.costate_rhs(x, p, Xa))
        print(f"  K={K}: min distance {MI.pair_distances(x.reshape(K, 7)[:, :3])[0]:.3f} m, |pair part of pdot| "
              f"{np.linalg.norm(pd - np.asarray(MI.build_ss_int(M, K, N, Pk.dt, JX, col=MI.make_col(Wc=0.0)).costate_rhs(x, p, Xa))):.2e}")
        check(f"K={K}: pdot == -dH/dx (rel)", pd, -np.asarray(gH), 1e-12, rel=True)

    print("== selftest 4: gradient identity -Sx' rho == jax.grad(Phi) with active coupling")
    for K in (2, 3):
        Pk, M, JX, MK = scene(K)
        Sb = MI.build_ss_int(M, K, N, Pk.dt, JX, col=col, napply=Pk.napply)
        x = clustered(MK, M, K, 0.3)
        Xr = MU.ref_window_multi(x, N, MK, M, Pk, 2, 0.0)
        Sb.warm(x, Xr)
        for i, v in enumerate([np.zeros(7 * K), rng.normal(size=7 * K) * 0.3]):
            r, J, Jr, Sx = [np.asarray(a) for a in Sb.all(v, x, Xr)]
            g_ad = np.asarray(Sb.g(v, x, Xr))
            g_hl = -(Sx.T @ r)
            rel = np.linalg.norm(g_hl - g_ad) / (1.0 + np.linalg.norm(g_ad))
            print(f"  K={K} v{i}: Phi {J:.4f}, |grad| {np.linalg.norm(g_ad):.3e}, |offdiag Jr| {offdiag_norm(Jr, K, 7):.3e}, "
                  f"rel |(-Sx'rho) - grad| {rel:.2e} -> {'OK' if rel < 1e-6 else 'SOFT-FAIL'}")
            ok &= rel < 1e-6

    print("== selftest 5: permutation equivariance (K = 3): permuting the agents permutes rho and x(T), keeps Phi")
    Pk, M, JX, MK = scene(3)
    Sb = MI.build_ss_int(M, 3, N, Pk.dt, JX, col=col, napply=Pk.napply)
    x = clustered(MK, M, 3, 0.3)
    Xr = MU.ref_window_multi(x, N, MK, M, Pk, 2, 0.0)
    v = rng.normal(size=21) * 0.3
    perm = np.array([2, 0, 1])
    pf = lambda a: a.reshape(3, 7)[perm].reshape(-1)
    r1, x1 = [np.asarray(a) for a in Sb.resx(v, x, Xr)]
    r2, x2 = [np.asarray(a) for a in Sb.resx(pf(v), pf(x), Xr[perm])]
    check("res(perm) == perm(res)", r2, pf(r1), 1e-12)
    check("xN(perm) == perm(xN)", x2, pf(x1), 1e-12)
    check("Phi(perm) == Phi", float(Sb.J(pf(v), pf(x), Xr[perm])), float(Sb.J(v, x, Xr)), 1e-10)

    print("== selftest 6: coupling is real: inter-agent Jacobian blocks, close vs far apart (K = 2)")
    Pk, M, JX, MK = scene(2)
    Sb = MI.build_ss_int(M, 2, N, Pk.dt, JX, col=col, napply=Pk.napply)
    xf = MK.x0.copy()                                                  # layout: 4 m apart
    xc = xf.reshape(2, 7).copy(); xc[:, 0] = 6.0; xc[:, 1] = [0.2, 0.7]; xc = xc.reshape(-1)
    for nm, xx in (("far (4 m)", xf), ("close (0.5 m)", xc)):
        Xr = MU.ref_window_multi(xx, N, MK, M, Pk, 2, 0.0)
        _, _, Jr, Sx = [np.asarray(a) for a in Sb.all(np.zeros(14), xx, Xr)]
        o = offdiag_norm(Jr, 2, 7)
        good = (o < 1e-10) if nm.startswith("far") else (o > 1e-3)
        print(f"  {nm:<14s} |offdiag Jr| {o:.3e}  |offdiag Sx| {offdiag_norm(Sx, 2, 7):.3e} -> {'OK' if good else 'BAD'}")
        ok &= good

    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", default="2", help="comma-separated agent counts")
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--Rc", type=float, default=MI.RC_DEFAULT, help="collision safety radius [m]")
    ap.add_argument("--Wc", type=float, default=MI.WC_DEFAULT, help="collision penalty weight")
    ap.add_argument("--wc", type=float, default=MI.WSH_DEFAULT, help="collision penalty shell width")
    ap.add_argument("--nocol", action="store_true", help="Wc = 0: the decoupled baseline through the same code")
    ap.add_argument("--maxcyc", type=int, default=None)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--out", default=PKL)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv[1:])
    P = make_env(args.cap, args.maxcyc)
    if args.selftest:
        return selftest(P, args.N)
    col = MI.make_col(args.Rc, 0.0 if args.nocol else args.Wc, args.wc)
    Ks = [int(s) for s in args.K.split(",")]
    D = {}
    if os.path.exists(args.out):
        with open(args.out, "rb") as f:
            D = pickle.load(f)
    D.setdefault("runs", {})
    D["config"] = dict(N=args.N, cap=args.cap, maxcyc=P.maxcyc, started=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node())
    done = []
    for K in Ks:
        key = (K, args.N, col.Rc, col.Wc, col.wc)
        if key in D["runs"] and not args.redo:
            print(f"-- {key} already in {args.out}; skipping (--redo to recompute)")
            done.append(D["runs"][key]); continue
        print(f"\n#### K={K} N={args.N} Rc={col.Rc:g} Wc={col.Wc:g} wc={col.wc:g}  ({time.strftime('%H:%M:%S')})", flush=True)
        e = run_int(K, args.N, make_env(args.cap, args.maxcyc), col, every=args.print_every)
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

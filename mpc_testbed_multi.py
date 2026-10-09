"""
mpc_testbed_multi.py -- Hopf-Lax-MPC on K decoupled agents as ONE stacked problem: computation scaling.

Protocol (per K): the frozen-obstacle scene of mpc_multi.make_multi_scene, K agents with their own
start / goal / reference, horizon N = 220, NO per-cycle budget, iteration cap --cap (default 1000),
the paper's solver mpc_solvers.chlqn_solve run UNCHANGED on the stacked shooting object
(mpc_multi.build_ss_multi), closed loop until every agent is at its goal or --maxcyc cycles.
Per cycle: solve time, iteration count, termination status, |grad|, descent / escape / kick / eigen-read
counts; per agent: trajectory, true tracking and obstacle cost, time of arrival.  Per-iteration average
cost = solve time / iterations per cycle.  Results merge into results_multi.pkl (one entry per K).

USAGE
    python mpc_testbed_multi.py --selftest                 # K=1 == mpc_core.build_ss; K=2 == two solo solves
    python mpc_testbed_multi.py --K 2                      # the two-agent correctness run
    python mpc_testbed_multi.py --K 2 --solo 0             # agent 0 of the 2-agent layout alone (K_eff = 1)
    python mpc_testbed_multi.py --K 1,2,3,4,6,8            # scaling sweep
    python mpc_testbed_multi.py --K 2 --maxcyc 60          # pipeline check
"""
from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

import mpc_core as C
import mpc_solvers as S
import mpc_testbed as T                   # CF compile flag (+ piecewise mobs_center, inert with vel = 0)
import mpc_tuned_params
import mpc_multi as MU
from mpc_testbed_N_sweep import m1v2_status

PKL = "results_multi.pkl"
N_DEFAULT = 220
CF = T.CF


def make_env(cap, maxcyc=None):
    P = C.make_P()
    P = mpc_tuned_params.apply(P)                                     # same scene parameters as the testbed
    P.budget = np.inf                                                 # NO real-time budget
    P.lm_maxit = int(cap)
    if maxcyc is not None:
        P.maxcyc = int(maxcyc)
    return P


def run_multi(K, N, P, solo=None, every=50, label=None):
    P, M, JX, MK = MU.make_multi_scene(P, K, solo=solo)
    Keff, n = MK.K, M.n
    label = label or (f"K={K}" if solo is None else f"K={K} solo agent {solo}")
    Sh = MU.build_ss_multi(M, Keff, N, P.dt, JX, napply=P.napply)
    x = MK.x0.copy()
    t = 0.0
    Xr0 = MU.ref_window_multi(x, N, MK, M, P, 2, 0.0)
    t0 = time.perf_counter()
    Sh.warm(x, Xr0)                                                    # compile OUTSIDE the timed loop
    Jb = S.ensure_jbatch(Sh)
    Jb(np.zeros((2, Keff * n)), x, Xr0)
    Jb(np.zeros((7, Keff * n)), x, Xr0)
    print(f"[{label}] stacked shooting object (dim {Keff * n}) compiled + warmed in {time.perf_counter() - t0:.1f} s")
    PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
    seed = np.zeros(Keff * n)
    Jreal, Jpen = np.zeros(Keff), np.zeros(Keff)
    treach = np.full(Keff, np.nan)
    XL, UL, PL, TL = [x.copy()], [], [], []
    TC, TS, RC, ITS, ST, NG = [], [], [], [], [], []
    NDESC, NESC, NKICK, NEIG, LM, NR = [], [], [], [], [], []
    t_start = time.perf_counter()
    aborted, err = False, None
    for c in range(1, P.maxcyc + 1):
        try:
            CF.fired = False
            tc = time.perf_counter()
            XrK = MU.ref_window_multi(x, N, MK, M, P, 2, t)
            tr = []
            ts = time.perf_counter()
            p0, itk, info = S.chlqn_solve(Sh, seed, x, XrK, PP, P.M1_epsc, S.M1V2_ETA_S, P.M1_alpha,
                                          S.M1V2_DELTA, tc, P.budget, trace=tr)
            TS.append(time.perf_counter() - ts)
            st = m1v2_status(tr, itk, PP.maxit)
            u = MU.ustar_multi(x, p0, M, JX)
            Pc = np.asarray(Sh.roll_costate(p0, x, XrK))
            seed = Pc[:, P.napply]
            TC.append(time.perf_counter() - tc)
            RC.append(CF.fired)
            ITS.append(itk); ST.append(st); TL.append(t)
            NG.append(float(tr[-1].get("norm_grad", np.nan)) if tr else np.nan)
            NDESC.append(info.ndesc); NESC.append(info.nesc); NKICK.append(info.nkick); NEIG.append(info.neig)
            # ---- untimed diagnostics at the applied iterate: surrogate lmin, |res| (stacked) ----
            rd, _, Jrd, Sxd = Sh.all(p0, x, XrK)
            Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
            LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
            NR.append(float(np.linalg.norm(np.asarray(rd))))
            if c % every == 1 or c == 1:
                its = np.asarray(ITS); ms = 1e3 * np.asarray(TS)
                pos = x.reshape(Keff, n)[:, :2]
                print(f"[{label}] c{c:4d} t{t:5.2f} | agents xy {np.round(pos, 2).tolist()} | iters mean {its.mean():5.2f} "
                      f"max {its.max():3d} | solve ms mean {ms.mean():7.2f} max {ms.max():7.1f} | ms/iter "
                      f"{np.mean(ms[its > 0] / its[its > 0]):6.2f} | non-conv {sum(s != 'converged' for s in ST)}", flush=True)
            x, Jreal, Jpen, at_goal, t = MU.apply_step_multi(x, u, XrK, Jreal, Jpen, M, MK, P, t, JX,
                                                             done=~np.isnan(treach))
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
    e = dict(K=Keff, K_layout=K, solo=solo, agents=MK.agents, N=N, label=label,
             starts=np.array(MK.starts), goals=np.array(MK.goals),
             X=np.array(XL).T.reshape(Keff, n, -1), U=np.array(UL).T.reshape(Keff, M.m, -1) if UL else np.zeros((Keff, M.m, 0)),
             Pc=np.array(PL).T.reshape(Keff, n, -1) if PL else np.zeros((Keff, n, 0)),
             tlog=np.asarray(TL), tcyc=tt, tsolve=ts, recompile=np.asarray(RC, dtype=bool),
             iters=its, status=list(ST), ngrad=np.asarray(NG), ndesc=np.asarray(NDESC), nesc=np.asarray(NESC),
             nkick=np.asarray(NKICK), neig=np.asarray(NEIG), lmin=np.asarray(LM), nres=np.asarray(NR),
             Jreal=Jreal, Jpen=Jpen, Jtrue=Jreal + Jpen, t_reach=treach, all_reached=bool(np.all(~np.isnan(treach))),
             ncyc=len(ITS), aborted=aborted, error=err, wall_s=time.perf_counter() - t_start,
             stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
             meta=dict(dt=P.dt, dt_apply=P.dt_apply, napply=P.napply, maxcyc=P.maxcyc, budget=P.budget,
                       lm_tol=P.lm_tol, lm_maxit=P.lm_maxit, M1_epsc=P.M1_epsc, M1_alpha=P.M1_alpha,
                       eta_s=S.M1V2_ETA_S, ridge=S.M1V2_RIDGE, delta=S.M1V2_DELTA, reachtol=P.reachtol,
                       y_spread=MU.NAGENT_Y_SPREAD, frozen_mobs=True),
             paths=[dict(pts=p.pts, s=p.s, th=p.th, len=p.len) for p in MK.paths],
             obs=dict(center=M.obs.center, ax=M.obs.ax, ay=M.obs.ay), mobs=dict(c0=M.mobs.c0, ax=M.mobs.ax, ay=M.mobs.ay))
    print(f"[{label}] done: {len(ITS)} cycles, all reached {e['all_reached']}, t_reach {np.round(treach, 2).tolist()}, "
          f"J per agent {np.round(e['Jtrue'], 3).tolist()} | iters mean {its.mean():.2f} | solve ms mean {1e3 * ts[1:].mean():.2f} "
          f"| ms/iter mean {1e3 * np.mean(ts[1:][pos] / its[1:][pos]):.2f} | descent {int(np.sum(NDESC))} escape {int(np.sum(NESC))} "
          f"kicks {int(np.sum(NKICK))} eigen reads {int(np.sum(NEIG))} | wall {e['wall_s']:.0f} s")
    return e


def print_table(entries):
    print(f"\n{'run':<22}{'dim':>5}{'cyc':>5}{'reached':>8}{'t_last':>7}{'sumJ':>9}{'it mean':>8}{'it max':>7}{'%conv':>7}"
          f"{'solve ms':>9}{'max ms':>8}{'ms/iter':>8}{'ms/it max':>10}{'kicks':>6}{'esc':>5}")
    print("-" * 126)
    for e in entries:
        its, ts = e["iters"][1:], e["tsolve"][1:]
        pos = its > 0
        pi = ts[pos] / its[pos]
        conv = 100 * np.mean([s == "converged" for s in e["status"]])
        tl = np.nanmax(e["t_reach"]) if e["all_reached"] else np.nan
        print(f"{e['label']:<22}{e['K'] * 7:>5d}{e['ncyc']:>5d}{('all' if e['all_reached'] else f'{int(np.sum(~np.isnan(e['t_reach'])))}/{e['K']}'):>8}"
              f"{tl:>7.2f}{e['Jtrue'].sum():>9.3f}{its.mean():>8.2f}{its.max():>7d}{conv:>7.1f}{1e3 * ts.mean():>9.2f}"
              f"{1e3 * ts.max():>8.1f}{1e3 * pi.mean():>8.2f}{1e3 * pi.max():>10.2f}{int(e['nkick'].sum()):>6d}{int(e['nesc'].sum()):>5d}"
              f"{' ABORTED' if e['aborted'] else ''}")
    print("timing: cycle 1 (cold start) excluded; ms/iter = solve time / iterations per cycle (cycles with 0 iterations excluded)")


def selftest(P, N=220):
    ok = True

    def check(name, a, b, tol=0.0):
        nonlocal ok
        a, b = np.asarray(a), np.asarray(b)
        d = float(np.max(np.abs(a - b))) if a.shape == b.shape else np.inf
        good = a.shape == b.shape and (np.array_equal(a, b) if tol == 0.0 else d <= tol)
        print(f"  {name:<52s} {'IDENTICAL' if (a.shape == b.shape and np.array_equal(a, b)) else f'max|diff| {d:.2e}'} -> {'OK' if good else 'BAD'}")
        ok &= good

    print("== selftest 1: stacked object with K=1 (agent 0 of the 1-agent layout) vs mpc_core.build_ss")
    P1, M, JX, MK = MU.make_multi_scene(P, 1)
    Sh1 = C.build_ss(M, N, P1.dt, JX, napply=P1.napply)
    ShK = MU.build_ss_multi(M, 1, N, P1.dt, JX, napply=P1.napply)
    x = MK.x0.copy()
    Xr = C.ref_window(x, N, MK.paths[0], M, P1, 2, 0.0)
    XrK = MU.ref_window_multi(x, N, MK, M, P1, 2, 0.0)
    check("ref_window_multi == ref_window", XrK[0], Xr)
    Sh1.warm(x, Xr); ShK.warm(x, XrK)
    rng = np.random.default_rng(0)
    for i, v in enumerate([np.zeros(7), rng.normal(size=7) * 0.3]):
        r1, J1, Jr1, Sx1 = [np.asarray(a) for a in Sh1.all(v, x, Xr)]
        rK, JK, JrK, SxK = [np.asarray(a) for a in ShK.all(v, x, XrK)]
        check(f"v{i}: res", r1, rK, 1e-12); check(f"v{i}: J", J1, JK, 1e-10)
        check(f"v{i}: Jr", Jr1, JrK, 1e-9); check(f"v{i}: Sx", Sx1, SxK, 1e-9)
        check(f"v{i}: roll_costate", np.asarray(Sh1.roll_costate(v, x, Xr)), np.asarray(ShK.roll_costate(v, x, XrK)), 1e-12)
    PP = SimpleNamespace(tol=P1.lm_tol, maxit=P1.lm_maxit)
    S.ensure_jbatch(Sh1); S.ensure_jbatch(ShK)
    p1, it1, _ = S.chlqn_solve(Sh1, np.zeros(7), x, Xr, PP, P1.M1_epsc, S.M1V2_ETA_S, P1.M1_alpha, S.M1V2_DELTA, time.perf_counter(), np.inf)
    pK, itK, _ = S.chlqn_solve(ShK, np.zeros(7), x, XrK, PP, P1.M1_epsc, S.M1V2_ETA_S, P1.M1_alpha, S.M1V2_DELTA, time.perf_counter(), np.inf)
    check(f"chlqn_solve costate (it {it1} vs {itK})", p1, pK, 1e-8)
    ok &= it1 == itK

    print("== selftest 2: K=2 stacked solve vs the two solo solves at the same states (decoupled => block equality)")
    P2, M, JX, MK2 = MU.make_multi_scene(make_env(1000), 2)
    Sh2 = MU.build_ss_multi(M, 2, N, P2.dt, JX, napply=P2.napply)
    x2 = MK2.x0.copy()
    XrK2 = MU.ref_window_multi(x2, N, MK2, M, P2, 2, 0.0)
    Sh2.warm(x2, XrK2); S.ensure_jbatch(Sh2)
    solos = []
    for k in range(2):
        Pk, Mk, JXk, MKk = MU.make_multi_scene(make_env(1000), 2, solo=k)
        Shk = MU.build_ss_multi(Mk, 1, N, Pk.dt, JXk, napply=Pk.napply)
        xk = MKk.x0.copy(); Xrk = MU.ref_window_multi(xk, N, MKk, Mk, Pk, 2, 0.0)
        Shk.warm(xk, Xrk); S.ensure_jbatch(Shk)
        solos.append((Shk, xk, Xrk, Pk))
    check("stacked x0 == [x0_solo0; x0_solo1]", x2, np.concatenate([solos[0][1], solos[1][1]]))
    check("stacked Xr == [Xr_solo0, Xr_solo1]", XrK2, np.stack([solos[0][2][0], solos[1][2][0]]))
    v2 = np.concatenate([rng.normal(size=7) * 0.3, rng.normal(size=7) * 0.3])
    r2, J2, Jr2, Sx2 = [np.asarray(a) for a in Sh2.all(v2, x2, XrK2)]
    parts = [[np.asarray(a) for a in solos[k][0].all(v2[7 * k:7 * k + 7], solos[k][1], solos[k][2])] for k in range(2)]
    check("res stacked == [res_0; res_1]", r2, np.concatenate([parts[0][0], parts[1][0]]), 1e-12)
    check("J stacked == J_0 + J_1", J2, parts[0][1] + parts[1][1], 1e-9)
    check("Jr diagonal blocks == Jr_k", np.stack([Jr2[:7, :7], Jr2[7:, 7:]]), np.stack([parts[0][2], parts[1][2]]), 1e-9)
    check("Jr off-diagonal blocks == 0", np.stack([Jr2[:7, 7:], Jr2[7:, :7]]), np.zeros((2, 7, 7)), 1e-14)
    check("Sx diagonal blocks == Sx_k", np.stack([Sx2[:7, :7], Sx2[7:, 7:]]), np.stack([parts[0][3], parts[1][3]]), 1e-9)
    PP = SimpleNamespace(tol=P2.lm_tol, maxit=P2.lm_maxit)
    p2, it2, info2 = S.chlqn_solve(Sh2, np.zeros(14), x2, XrK2, PP, P2.M1_epsc, S.M1V2_ETA_S, P2.M1_alpha, S.M1V2_DELTA, time.perf_counter(), np.inf)
    ps = [S.chlqn_solve(solos[k][0], np.zeros(7), solos[k][1], solos[k][2], PP, P2.M1_epsc, S.M1V2_ETA_S, P2.M1_alpha, S.M1V2_DELTA, time.perf_counter(), np.inf) for k in range(2)]
    print(f"  stacked solve: it={it2} ndesc={info2.ndesc} nkick={info2.nkick}; solo solves: it={ps[0][1]}, {ps[1][1]}")
    check("converged stacked costate == [p_solo0; p_solo1] (tol 1e-6)", p2, np.concatenate([ps[0][0], ps[1][0]]), 1e-6)
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", default="2", help="comma-separated agent counts")
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--solo", type=int, default=None, help="run only agent <solo> of the K-layout (correctness)")
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
    Ks = [int(s) for s in args.K.split(",")]
    D = {}
    if os.path.exists(args.out):
        with open(args.out, "rb") as f:
            D = pickle.load(f)
    D.setdefault("runs", {})
    D["config"] = dict(N=args.N, cap=args.cap, maxcyc=P.maxcyc, started=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node())
    done = []
    for K in Ks:
        key = (K, args.N) if args.solo is None else (K, args.N, "solo", args.solo)
        if key in D["runs"] and not args.redo:
            print(f"-- {key} already in {args.out}; skipping (--redo to recompute)")
            done.append(D["runs"][key]); continue
        print(f"\n#### K={K} N={args.N}{'' if args.solo is None else f' solo {args.solo}'}  ({time.strftime('%H:%M:%S')})", flush=True)
        e = run_multi(K, args.N, make_env(args.cap, args.maxcyc), solo=args.solo, every=args.print_every)
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

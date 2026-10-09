"""
mpc_testbed_disturb.py -- Hopf-Lax-MPC robustness test: bounded Gaussian process disturbance.

Same closed-loop testbed as mpc_testbed.py (scenario, hard 20 ms/cycle budget inside the solver,
warm start, tuned horizon N = P.N_m1 = 220, stop at goal) for the Hopf-Lax-MPC runner ONLY, but the
simulated plant is    xdot = f(x, u) + d    with d a bounded Gaussian disturbance redrawn at every
fine simulation step (dt = 5 ms; 4 fine steps per 20 ms control cycle):

    d_i ~ N(0, sigma^2),  clipped to [-b, b],  i = 1..n   (sigma = sigma_ratio * b, default b/2)

so |d| <= b componentwise in units of state-rate (m/s for positions, rad/s for the heading,
m/s^2 for the speed states).  The controller is unchanged and unaware of d (it predicts with the
clean model); d enters only in the true plant step  x <- rk4(x, u, dt) + d * dt.  b = 0 reproduces the
clean testbed exactly (the clean run is the b = 0 entry of the same script, so timing conditions match).

Results merge into results_disturb.pkl keyed by (bound, seed); each entry stores the closed-loop
outcome, per-cycle timing / iteration data and the applied disturbance sequence.

USAGE
    python mpc_testbed_disturb.py --bound 0                 # clean reference run (same session timing)
    python mpc_testbed_disturb.py --bound 0.5               # one disturbed run, seed 0
    python mpc_testbed_disturb.py --bound 0.25,0.5,1,2 --seeds 0,1,2     # sweep (later)
    python mpc_testbed_disturb.py --table                   # print the comparison table and exit
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np

import mpc_core as C
import mpc_solvers as S
import mpc_testbed as T                   # piecewise mobs_center, CF, mk_res, summary-style helpers
import mpc_tuned_params

PKL = "results_disturb.pkl"
CF = T.CF


def apply_step_dist(x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX, rng, b, sigma, DL):
    """mpc_testbed.apply_step with the disturbed plant  x <- rk4(x,u,dt) + d*dt,  d redrawn per fine step."""
    Uapp.append(np.asarray(u).copy())
    if p0 is not None:
        Pclog.append(np.asarray(p0).copy())
    dxk = x - Xref[:, 0]
    Jreal = Jreal + (dxk @ M.Q @ dxk + u @ M.R @ u) * P.dt_apply
    for s in range(1, P.napply + 1):
        x = np.asarray(JX.rk4_state(x, u, P.dt))
        if b > 0.0:
            d = np.clip(rng.normal(0.0, sigma, size=x.shape), -b, b)
            x = x + d * P.dt
        else:
            d = np.zeros_like(x)
        DL.append(d)
        X.append(x.copy())
        mc = C.mobs_center(M.mobs, t + s * P.dt)                       # TRUE position, not the prediction
        pen1, _ = JX.obstacle(x[M.posidx])
        pen2, _ = JX.mobs_pen(x[M.posidx], mc)
        Jpen = Jpen + (float(pen1) + float(pen2)) * P.dt
    t = t + P.napply * P.dt
    stop = np.linalg.norm(x[M.posidx] - P.pgoal[:len(M.posidx)]) < P.reachtol
    return x, X, Jreal, Jpen, Uapp, Pclog, 0, stop, t


def run_m1v2_dist(M, N, P, path, Sh, JX, b, sigma, seed, every=50):
    """Clone of mpc_testbed.run_m1v2_mpc (budgeted Hopf-Lax-MPC) on the disturbed plant; no viz logging."""
    rng = np.random.default_rng(seed)
    x = M.x0.copy()
    Xr0 = C.ref_window(x, N, path, M, P, 2, 0.0)
    Sh.warm(x, Xr0)                                                    # compile OUTSIDE the timed loop
    Jb = S.ensure_jbatch(Sh)
    Jb(np.zeros((2, M.n)), x, Xr0)
    Jb(np.zeros((7, M.n)), x, Xr0)
    X = [x.copy()]
    tcyc, Uapp, Pclog, DL, TL = [], [], [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    seed_v = np.zeros(M.n)
    t = 0.0
    PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
    ITS, NDESC, NESC, NKICK, NEIG, LM, NR, RC = [], [], [], [], [], [], [], []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xr2 = C.ref_window(x, N, path, M, P, 2, t)
        Xref = Xr2[:M.n, ::2]
        p0, itk, infok = S.chlqn_solve(Sh, seed_v, x, Xr2, PP, P.M1_epsc, S.M1V2_ETA_S, P.M1_alpha,
                                       S.M1V2_DELTA, tc, P.budget)
        u = np.asarray(JX.ustar(x, p0))
        ITS.append(itk); NDESC.append(infok.ndesc); NESC.append(infok.nesc)
        NKICK.append(infok.nkick); NEIG.append(infok.neig)
        Pc = np.asarray(Sh.roll_costate(p0, x, Xr2))
        seed_v = Pc[:, P.napply]
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        rd, _, Jrd, Sxd = Sh.all(p0, x, Xr2)                           # untimed diagnostics
        Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
        LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
        NR.append(float(np.linalg.norm(np.asarray(rd))))
        TL.append(t)
        if c % every == 1:
            print(f"M1v2 b={b:g} c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f} |res|{NR[-1]:9.2e} "
                  f"lmin{LM[-1]:10.2e} it{itk:3d}")
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = apply_step_dist(
            x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX, rng, b, sigma, DL)
        if stop:
            reached = 1
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    D = np.array(DL).T if DL else np.zeros((M.n, 0))
    tt = np.asarray(tcyc)[1:]
    e = dict(bound=float(b), sigma=float(sigma), seed=int(seed), N=int(N),
             Jreal=float(res.Jreal), Jpen=float(res.Jpen), Jtrue=float(res.Jtrue), reached=int(res.reached),
             t_reach=float(TL[-1] + P.dt_apply) if res.reached else np.nan, ncyc=int(res.ncyc),
             clear=float(res.clear), max_abs_y=float(np.abs(res.X[1]).max()),
             X=res.X.copy(), U=res.U.copy(), tlog=np.asarray(TL), tcyc=np.asarray(tcyc),
             recompile=np.asarray(RC, dtype=bool), iters=np.asarray(ITS), ndesc=np.asarray(NDESC),
             nesc=np.asarray(NESC), nkick=np.asarray(NKICK), neig=np.asarray(NEIG),
             lmin=np.asarray(LM), nres=np.asarray(NR), dist=D,
             ms_mean=1e3 * tt.mean(), ms_max=1e3 * tt.max(), ms_p99=1e3 * np.percentile(tt, 99),
             stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
             meta=dict(dt=P.dt, dt_apply=P.dt_apply, napply=P.napply, budget=P.budget, lm_tol=P.lm_tol,
                       lm_maxit=P.lm_maxit, maxcyc=P.maxcyc, eta_s=S.M1V2_ETA_S, ridge=S.M1V2_RIDGE))
    print(f"M1v2 b={b:g} seed={seed}: {'reached %.2f s' % e['t_reach'] if reached else 'TRAPPED'}  "
          f"J_track {e['Jreal']:.3f} J_obs {e['Jpen']:.3f} J_total {e['Jtrue']:.3f} | mean {e['ms_mean']:.2f} ms "
          f"p99 {e['ms_p99']:.1f} | iters mean {np.mean(ITS):.2f} max {max(ITS)} | kicks {sum(NKICK)} "
          f"escape {sum(NESC)} | max|d| applied {np.abs(D).max() if D.size else 0:.3f}")
    return e


ROW = ("{bound:>6} {seed:>4} {out:>8} {t_reach:>7} {Jreal:>8} {Jpen:>8} {Jtrue:>8} {cyc:>4} {ms:>8} {p99:>7} "
       "{mx:>7} {it:>6} {itmax:>5} {kick:>5} {esc:>4} {clear:>6} {maxy:>6} {nres:>9}")


def print_table(entries, ref=None):
    print(ROW.format(bound="bound", seed="seed", out="outcome", t_reach="t_reach", Jreal="J_track", Jpen="J_obs",
                     Jtrue="J_total", cyc="cyc", ms="mean ms", p99="p99 ms", mx="max ms", it="it/cyc", itmax="itmax",
                     kick="kicks", esc="esc", clear="clear", maxy="max|y|", nres="|res| med"))
    print("-" * 150)
    rows = []
    if ref is not None:
        rows.append(("recorded", "--", ref))
    rows += [(f"{e['bound']:g}", str(e["seed"]), e) for e in entries]
    for bnd, sd, e in rows:
        print(ROW.format(bound=bnd, seed=sd, out=("reached" if e["reached"] else "TRAPPED"),
                         t_reach=(f"{e['t_reach']:.2f}" if e["reached"] else "--"), Jreal=f"{e['Jreal']:.3f}",
                         Jpen=f"{e['Jpen']:.3f}", Jtrue=f"{e['Jtrue']:.3f}", cyc=e["ncyc"], ms=f"{e['ms_mean']:.2f}",
                         p99=f"{e['ms_p99']:.1f}", mx=f"{e['ms_max']:.1f}", it=f"{np.mean(e['iters']):.2f}",
                         itmax=int(np.max(e["iters"])), kick=int(np.sum(e["nkick"])), esc=int(np.sum(e["nesc"])),
                         clear=f"{e['clear']:.3f}", maxy=f"{e['max_abs_y']:.3f}", nres=f"{np.median(e['nres']):.1e}"))


def recorded_m1(base="results_testbed_v10.pkl"):
    """The published budgeted Hopf-Lax-MPC run, in the same row format."""
    if not os.path.exists(base):
        return None
    with open(base, "rb") as f:
        D = pickle.load(f)
    r = next(q for q in D["R"] if q.name == "M1_v2")
    P = D["P"]
    tt = np.asarray(r.res.tcyc)[1:]
    return dict(Jreal=r.res.Jreal, Jpen=r.res.Jpen, Jtrue=r.res.Jtrue, reached=r.res.reached,
                t_reach=(r.res.tlog[-1] + P.dt_apply) if r.res.reached else np.nan, ncyc=r.res.ncyc,
                ms_mean=1e3 * tt.mean(), ms_max=1e3 * tt.max(), ms_p99=1e3 * np.percentile(tt, 99),
                iters=r.res.m1_its, nkick=r.res.m1_nkick, nesc=r.res.m1_nesc, clear=r.res.clear,
                max_abs_y=float(np.abs(r.res.X[1]).max()), nres=r.res.nres)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bound", default="0.5", help="disturbance bound(s) b, comma-separated (0 = clean)")
    ap.add_argument("--sigma-ratio", type=float, default=0.5, help="sigma = ratio * b")
    ap.add_argument("--seeds", default="0", help="RNG seed(s), comma-separated")
    ap.add_argument("--N", type=int, default=None, help="horizon (default: the tuned P.N_m1 = 220)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nobudget", action="store_true")
    ap.add_argument("--out", default=PKL)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the table from the pickle and exit")
    args = ap.parse_args(argv[1:])

    D = {}
    if os.path.exists(args.out):
        with open(args.out, "rb") as f:
            D = pickle.load(f)
    D.setdefault("runs", {})
    if args.table:
        print_table([D["runs"][k] for k in sorted(D["runs"])], recorded_m1())
        return 0

    P = C.make_P(smoke=args.smoke)
    P = mpc_tuned_params.apply(P)
    if args.nobudget:
        P.budget = np.inf
    N = args.N or P.N_m1
    M = C.model_scn(P)
    JX = C.build_jax(M)
    path = C.plan_path(P.p0, P.pgoal, P.obs_plan, P)
    Sh = C.build_ss(M, N, P.dt, JX, napply=P.napply)
    bounds = [float(s) for s in args.bound.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== Hopf-Lax-MPC with bounded Gaussian disturbance: N={N}, budget {1e3 * P.budget:.0f} ms, "
          f"maxcyc {P.maxcyc}, bounds {bounds}, sigma = {args.sigma_ratio:g} b, seeds {seeds} -> {args.out}")
    done = []
    for b in bounds:
        for sd in seeds:
            key = (b, sd)
            if key in D["runs"] and not args.redo:
                print(f"-- {key} already in {args.out}; skipping")
                done.append(D["runs"][key])
                continue
            e = run_m1v2_dist(M, N, P, path, Sh, JX, b, args.sigma_ratio * b, sd)
            if not args.smoke:
                D["runs"][key] = e
                tmp = args.out + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(D, f)
                os.replace(tmp, args.out)
            done.append(e)
    print()
    print_table(done, recorded_m1())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

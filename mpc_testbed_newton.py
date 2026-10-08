"""
mpc_testbed_newton.py -- the Newton (exact Hessian) baseline in the BUDGETED closed-loop testbed.

Runs ONE closed-loop runner, the damped Newton baseline on the Hopf-Lax objective (mpc_newton.py),
under exactly the protocol of mpc_testbed.py (same scenario, hard 20 ms/cycle wall-clock cut inside
the solver, same warm start, viz logging for the animation) at a horizon chosen by the budgeted
horizon study (see README section 3), and APPENDS it to the recorded six-runner pickle so that
make_animations.py renders all seven runners together.  mpc_testbed.py itself is untouched.

USAGE
    python mpc_testbed_newton.py --N 60                                  # -> results_testbed_newton.pkl
    python mpc_testbed_newton.py --N 60 --base results_testbed_v10.pkl --out results_testbed_newton.pkl
    python mpc_testbed_newton.py --N 60 --smoke                          # 60 cycles
    python make_animations.py results_testbed_newton.pkl c figs/with_newton   # 7-runner animation
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
import mpc_testbed as T                   # piecewise mobs_center, CF, viz_*, apply_step, mk_res, runner, summary
import mpc_tuned_params
import mpc_newton as NW

NAME = "Newton-H"
COLOR = (0.0, 0.60, 0.60)
N_NEWTON_DEFAULT = 55                     # budgeted horizon study (README section 3): largest N with p99 inside 20 ms
CF = T.CF


def run_newton_mpc(M, N, P, path, Sh, JX):
    """Clone of mpc_testbed.run_m1v2_mpc with mpc_newton.newton_solve as the solver (budget enforced
    inside the solver via P.budget, like every other runner)."""
    x = M.x0.copy()
    Xr0 = C.ref_window(x, N, path, M, P, 2, 0.0)
    Sh.warm(x, Xr0)                                                    # compile OUTSIDE the timed loop
    Jb = S.ensure_jbatch(Sh)
    Jb(np.zeros((7, M.n)), x, Xr0)                                     # batched line search (B=7)
    t0 = time.perf_counter()
    Hf = NW.ensure_hessian(Sh, getattr(P, "hess_mode", "hessian"))
    np.asarray(Hf(np.zeros(M.n), x, Xr0))
    print(f"{NAME} N={N}: jax.hessian compiled in {time.perf_counter() - t0:.1f} s")
    X = [x.copy()]
    tcyc, Uapp, Pclog = [], [], []
    Jreal = Jpen = 0.0
    reached = 0
    seed = np.zeros(M.n)
    t = 0.0
    VZ = T.viz_init()
    PP = SimpleNamespace(tol=P.newton_tol, maxit=P.lm_maxit)
    ITS, ST, NG, NACC, NREJ, THESS, LMH, LM, NR, RC = [], [], [], [], [], [], [], [], [], []
    for c in range(1, P.maxcyc + 1):
        CF.fired = False
        tc = time.perf_counter()
        Xr2 = C.ref_window(x, N, path, M, P, 2, t)
        Xref = Xr2[:M.n, ::2]
        p0, itk, info = NW.newton_solve(Sh, seed, x, Xr2, PP, tc, P.budget)
        u = np.asarray(JX.ustar(x, p0))
        Pc = np.asarray(Sh.roll_costate(p0, x, Xr2))
        seed = Pc[:, P.napply]
        tcyc.append(time.perf_counter() - tc)
        RC.append(CF.fired)
        ITS.append(itk); ST.append(info.status); NG.append(info.norm_grad)
        NACC.append(info.nacc); NREJ.append(info.nrej); THESS.append(info.t_hess); LMH.append(info.lminH)
        # ---- untimed diagnostics at the applied iterate (as run_m1v2_mpc) ----
        rd, _, Jrd, Sxd = Sh.all(p0, x, Xr2)
        Hc = -(np.asarray(Sxd).T @ np.asarray(Jrd))
        LM.append(float(np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min()))
        NR.append(float(np.linalg.norm(np.asarray(rd))))
        if c % 10 == 1:
            print(f"NewtH c{c:3d} t{t:5.2f} px{x[0]:6.2f} py{x[1]:7.3f} |res|{NR[-1]:9.2e} lmin{LM[-1]:10.2e} "
                  f"it{itk:3d} {info.status}")
        T.viz_log(VZ, t, x, Sh.roll_pred(p0, x, Xr2), Xref, M)
        x, X, Jreal, Jpen, Uapp, Pclog, reached, stop, t = T.apply_step(
            x, u, p0, X, Xref, Jreal, Jpen, Uapp, Pclog, M, P, t, JX)
        if stop:
            reached = 1
            break
    res = T.mk_res(X, tcyc, Jreal, Jpen, reached, Uapp, Pclog, M)
    res.lmin, res.nres = np.array(LM), np.array(NR)
    res.recompile = np.array(RC)
    res.nw_its, res.nw_status, res.nw_ngrad = np.array(ITS), list(ST), np.array(NG)
    res.nw_nacc, res.nw_nrej, res.nw_thess, res.nw_lminH = np.array(NACC), np.array(NREJ), np.array(THESS), np.array(LMH)
    ncut = sum(s == "timeout" for s in ST)
    print(f"{NAME} iter stats: mean {np.mean(ITS):.2f} iters/cycle (max {max(ITS)}) | budget-cut cycles "
          f"{ncut}/{len(ITS)} ({100 * ncut / len(ITS):.1f}%) | converged {sum(s == 'converged' for s in ST)} | "
          f"Hessian share of cycle time {100 * np.sum(THESS) / np.sum(tcyc):.0f}%")
    return T.viz_pack(res, VZ)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=N_NEWTON_DEFAULT, help="horizon of the Newton-H runner")
    ap.add_argument("--base", default="results_testbed_v10.pkl", help="six-runner pickle to append to")
    ap.add_argument("--out", default="results_testbed_newton.pkl")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nobudget", action="store_true")
    ap.add_argument("--newton-tol", type=float, default=1e-4)
    args = ap.parse_args(argv[1:])

    P = C.make_P(smoke=args.smoke)
    P = mpc_tuned_params.apply(P)                                     # scenario exactly as mpc_testbed
    if args.nobudget:
        P.budget = np.inf
    P.newton_tol = float(args.newton_tol)
    P.N_newton = int(args.N)
    M = C.model_scn(P)
    JX = C.build_jax(M)
    path = C.plan_path(P.p0, P.pgoal, P.obs_plan, P)
    print(f"=== {NAME} in the budgeted testbed: N={args.N} (Tp={args.N * P.dt:.2f} s), budget "
          f"{1e3 * P.budget:.0f} ms, maxcyc {P.maxcyc}, tol {P.newton_tol:g}")
    Sh = C.build_ss(M, args.N, P.dt, JX, napply=P.napply)
    tt = time.perf_counter()
    r = T.runner(NAME, COLOR, "-", run_newton_mpc(M, args.N, P, path, Sh, JX), args.N)
    print(f"{NAME} done ({time.perf_counter() - tt:.0f} s)")

    if os.path.exists(args.base) and not args.smoke:
        with open(args.base, "rb") as f:
            D = pickle.load(f)
        R = [q for q in D["R"] if q.name != NAME] + [r]                # append (or replace) the Newton runner
        D["R"] = R
        D["P"].N_newton = args.N
        D["P"].newton_tol = P.newton_tol
        print(f"appended to {args.base}: runners = {[q.name for q in R]}")
    else:
        R = [r]
        D = {"R": R, "P": P, "M": M, "path": path}
        if args.smoke:
            print("(smoke: not merged into the base pickle)")
    T.summary(R, D["P"])
    out = args.out if not args.smoke else "results_testbed_newton_smoke.pkl"
    with open(out, "wb") as f:
        pickle.dump(D, f)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

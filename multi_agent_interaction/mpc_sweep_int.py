"""
mpc_sweep_int.py -- ONE (method, K) run of the all-method K-sweep on the interacting multi-agent problem.

Run exactly one method at one K per process (the timing must not be affected by other runs or by JIT caches
of earlier runs); run_sweep_int.sh loops over K = 1..10 and the six methods sequentially.

Protocol (all methods): horizon N = 200, NO per-cycle budget, iteration cap 1000, frozen scene and column
layout (mpc_multi_int.make_multi_scene_int), pair penalty Rc 0.8 / Wc 500 / wc 0.1.
  * run end:      every agent has reached its goal at least once (arrival = within reachtol), or 1500
                  cycles (30 s), or the WALL-TIME LIMIT (default 1800 s = 30 min) -> saved as incomplete.
                  The solver budget of every cycle is the remaining wall time, so a single slow
                  cycle cannot overrun the limit (normal cycles are far from it: no effect).
  * trajectory cost (full run, all terms): sum over agents of tracking + control + obstacle cost and the
                  pair collision cost, integrated along the WHOLE closed-loop trajectory until the run ends
                  (mpc_multi_int.apply_step_int with no agent masked).
  * per cycle:    solver-call time, whole-cycle time, iterations, status, method counters, minimum pairwise
                  distance; plus the trajectory, controls and costates -- everything in the result file.

Methods: Hopf_Lax_MPC, PMP_Shooting, DDP, Vanilla_Hopf_Lax, MPPI_12288, MPPI_20480
(adapters in mpc_testbed_baselines_int.make_method).

USAGE
    python mpc_sweep_int.py --method DDP --K 4        # -> results_sweep_int/DDP_K4.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time

import numpy as np

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ("core", "closed_loop", "N_sweep", "multi_agent")]
import mpc_multi as MU
import mpc_multi_int as MI
import mpc_testbed_baselines_int as TB

OUTDIR = "results_sweep_int"
METHODS = ["Hopf_Lax_MPC", "PMP_Shooting", "DDP", "Vanilla_Hopf_Lax", "MPPI_12288", "MPPI_20480"]


def run(method, K, N, maxcyc, wall_limit, col, every=100):
    P = TB.make_env(1000, maxcyc)
    P, M, JX, MK = MI.make_multi_scene_int(P, K)
    n = M.n
    label = f"{method} K={K}"
    init, step = TB.make_method(method, K, N, P, M, JX, MK, col)
    x = MK.x0.copy()
    t = 0.0
    t0 = time.perf_counter()
    init(x)
    t_compile = time.perf_counter() - t0
    print(f"[{label}] compiled + warmed in {t_compile:.1f} s", flush=True)
    Jreal, Jpen, Jpair = np.zeros(K), np.zeros(K), np.zeros((K, K))
    treach = np.full(K, np.nan)
    XL, UL, PL, TL = [x.copy()], [], [], []
    TC, TS, RC, ITS, ST, DMIN, DPAIR = [], [], [], [], [], [], []
    EX = {}
    t_start = time.perf_counter()
    deadline = t_start + wall_limit
    aborted, err, end = False, None, "maxcyc"
    for c in range(1, P.maxcyc + 1):
        try:
            TB.CF.fired = False
            tc = time.perf_counter()
            P.budget = max(deadline - tc, 0.0)                         # = remaining wall time: no effect on a normal
                                                                       # cycle, but a single long cycle (e.g. DDP's
                                                                       # cold start) is cut at the run's wall limit
            u, p0, itk, status, ex = step(x, t, tc)
            TC.append(time.perf_counter() - tc)
            TS.append(ex.pop("_tsolve"))
            RC.append(TB.CF.fired)
            ITS.append(int(itk)); ST.append(status); TL.append(t)
            for kk, vv in ex.items():
                EX.setdefault(kk, []).append(vv)
            if not np.all(np.isfinite(u)):
                raise RuntimeError("non-finite control")
            if c % every == 1:
                its = np.asarray(ITS); ms = 1e3 * np.asarray(TS)
                print(f"[{label}] c{c:4d} t{t:5.2f} | arrived {int(np.sum(~np.isnan(treach)))}/{K} | dmin so far "
                      f"{(min(DMIN) if DMIN else np.inf):5.3f} | iters mean {its.mean():6.2f} max {its.max():4d} | solve ms "
                      f"mean {ms.mean():8.2f} max {ms.max():8.1f} | wall {time.perf_counter() - t_start:6.0f} s", flush=True)
            x, Jreal, Jpen, Jpair, at_goal, t, dmin, pmin = MI.apply_step_int(
                x, u, MU.ref_window_multi(x, N, MK, M, P, 1, t), Jreal, Jpen, Jpair, M, MK, P, t, JX, col, done=None)
            DMIN.append(dmin); DPAIR.append(pmin if pmin is not None else (-1, -1))
            XL.append(x.copy()); UL.append(np.asarray(u).copy()); PL.append(np.zeros(K * n) if p0 is None else np.asarray(p0).copy())
            newly = at_goal & np.isnan(treach)
            treach[newly] = t
            if np.all(~np.isnan(treach)):
                end = "all-arrived"
                break
            if time.perf_counter() >= deadline:
                end = "wall-limit"
                break
        except KeyboardInterrupt:
            raise
        except Exception as e_:                                       # noqa: BLE001
            aborted, err, end = True, repr(e_), "aborted"
            print(f"[{label}] ABORTED at cycle {c}: {err}", flush=True)
            break
    its = np.asarray(ITS, dtype=int); ts = np.asarray(TS)
    dm = np.asarray(DMIN)
    J_full = float(Jreal.sum() + Jpen.sum() + Jpair.sum())
    e = dict(method=method, K=K, N=N, label=label, col=vars(col).copy(), starts=np.array(MK.starts), goals=np.array(MK.goals),
             X=np.array(XL).T.reshape(K, n, -1), U=np.array(UL).T.reshape(K, M.m, -1) if UL else np.zeros((K, M.m, 0)),
             Pc=np.array(PL).T.reshape(K, n, -1) if PL else np.zeros((K, n, 0)),
             tlog=np.asarray(TL), tcyc=np.asarray(TC), tsolve=ts, recompile=np.asarray(RC, dtype=bool),
             iters=its, status=list(ST), extra={k: np.asarray(v) for k, v in EX.items()},
             dmin=dm, dpair=np.asarray(DPAIR, dtype=int), Jreal=Jreal, Jpen=Jpen, Jpair=Jpair, Jtrue=Jreal + Jpen,
             Jpair_total=float(np.sum(Jpair)), J_full=J_full, t_reach=treach, all_reached=bool(np.all(~np.isnan(treach))),
             end=end, complete=(end == "all-arrived"), ncyc=len(ITS), aborted=aborted, error=err,
             wall_s=time.perf_counter() - t_start, t_compile=t_compile, stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
             host=platform.node(),
             meta=dict(dt=P.dt, dt_apply=P.dt_apply, napply=P.napply, maxcyc=P.maxcyc, wall_limit=wall_limit,
                       budget="remaining wall time of the run (no per-cycle budget)", lm_tol=P.lm_tol, lm_maxit=P.lm_maxit,
                       ilqr_tol=P.ilqr_tol, ilqr_Kmax=P.ilqr_Kmax, vanilla_tol=TB.VANILLA_TOL, reachtol=P.reachtol,
                       layout="columns-mirrored-goals", stop_rule="all agents arrived at least once",
                       cost="full run, all terms (no masking after arrival)"),
             paths=[dict(pts=p.pts, s=p.s, th=p.th, len=p.len) for p in MK.paths],
             obs=dict(center=M.obs.center, ax=M.obs.ax, ay=M.obs.ay), mobs=dict(c0=M.mobs.c0, ax=M.mobs.ax, ay=M.mobs.ay))
    print(f"[{label}] END ({end}): {len(ITS)} cycles, arrived {int(np.sum(~np.isnan(treach)))}/{K}, t_reach "
          f"{np.round(treach, 2).tolist()} | J_full {J_full:.3f} (track {Jreal.sum():.3f} obs {Jpen.sum():.3f} pair "
          f"{Jpair.sum():.3f}) | min distance {dm.min() if dm.size else np.inf:.3f} m | iters mean {its.mean():.2f} | "
          f"solve ms mean {1e3 * ts[1:].mean() if ts.size > 1 else np.nan:.2f} | status "
          f"{ {s: ST.count(s) for s in sorted(set(ST))} } | wall {e['wall_s']:.0f} s", flush=True)
    return e


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--maxcyc", type=int, default=1500)
    ap.add_argument("--wall-limit", type=float, default=1800.0)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(argv[1:])
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.method}_K{args.K}.pkl")
    if os.path.exists(out) and not args.redo:
        print(f"-- {out} exists; skipping (--redo to recompute)")
        return 0
    print(f"#### {args.method} K={args.K} N={args.N}  ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    e = run(args.method, args.K, args.N, args.maxcyc, args.wall_limit, MI.make_col())
    with open(out + ".tmp", "wb") as f:
        pickle.dump(e, f)
    os.replace(out + ".tmp", out)
    print(f"#### saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

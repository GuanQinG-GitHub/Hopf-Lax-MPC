"""
mpc_demo_int.py -- ONE (method, layout) closed-loop run on main's N-sweep course with K interacting agents.

Course: core/mpc_tuned_params.apply_variant("wall_saddle_bumps_rand") -- static obstacles (offset bump +
7 on-axis z-bumps) and MOVING bodies (the wall stops on the axis: the symmetric saddle), centres from the
piecewise mobs_center (closed_loop/mpc_testbed).  Agents: explicit starts / goals from LAYOUTS (tuned for
the demonstration); references from the N-sweep planner model P.obs_plan.  Pair penalty Rc 0.8 / Wc 500 /
wc 0.1.  Protocol: horizon N = 200, no budget, iteration caps 1000, SHARED stopping rule (core/shared_stop:
|dJ_ocp| <= 1e-6 |J| + 1e-10, J_ocp = the joint DDP rollout cost incl. the pair penalty) for every
iterative method; its untimed evaluation is subtracted from the solve time.  MPPI = vanilla IT-MPC
(lambda = 0.01, one update per cycle).  Run end: every agent arrived once, or 1500 cycles (30 s), or the
wall-time limit (default 30 min).

Methods: Hopf_Lax_MPC, PMP_Shooting, DDP_Box, Vanilla_Hopf_Lax, MPPI_12288, MPPI_20480, Collocation_IPOPT.

USAGE
    python mpc_demo_int.py --layout A_column --method Hopf_Lax_MPC     # -> results_demo_int/A_column/Hopf_Lax_MPC.pkl
    python mpc_demo_int.py --list
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

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ("core", "closed_loop", "N_sweep", "multi_agent")]
import mpc_core as C
import mpc_solvers as S
import mpc_testbed as T                   # piecewise mobs_center + CF compile flag + _BudgetCB
import mpc_multi as MU
import mpc_testbed_multi as TM
import mpc_testbed_N_sweep as NS          # counted PMP copy (with the shared stop), IPOPT status map
import shared_stop as SSTOP
import mpc_multi_int as MI
import mpc_baselines_int as BI

OUTDIR = "results_demo_int"
METHODS = ["Hopf_Lax_MPC", "PMP_Shooting", "DDP_Box", "Vanilla_Hopf_Lax", "MPPI_12288", "MPPI_20480",
           "Collocation_IPOPT"]
Z0 = 0.5                                                               # cruise altitude
# K = 2 layouts (start, goal per agent; x, y, z).  Families: A column on the axis, B mirror pair crossing at
# the wall, C one agent on the axis + one off-axis.  Tuned for the demonstration (see README).
LAYOUTS = {
    "A_column":     dict(starts=[(0.0, 0.0, Z0), (-1.5, 0.0, Z0)], goals=[(21.5, 0.0, Z0), (20.0, 0.0, Z0)]),
    "A_column_pm":  dict(starts=[(0.0, 0.0, Z0), (-1.5, 0.0, Z0)], goals=[(20.0, 0.67, Z0), (20.0, -0.67, Z0)]),
    "B_mirror_05":  dict(starts=[(0.0, -0.5, Z0), (0.0, 0.5, Z0)], goals=[(20.0, 0.5, Z0), (20.0, -0.5, Z0)]),
    "B_mirror_08":  dict(starts=[(0.0, -0.8, Z0), (0.0, 0.8, Z0)], goals=[(20.0, 0.8, Z0), (20.0, -0.8, Z0)]),
    "C_axis_side":  dict(starts=[(0.0, 0.0, Z0), (0.0, 2.0, Z0)], goals=[(20.0, 0.0, Z0), (20.0, -2.0, Z0)]),
    "K1_axis":      dict(starts=[(0.0, 0.0, Z0)], goals=[(20.0, 0.0, Z0)]),   # single-agent sanity (= N sweep)
}


REACHTOL = 0.8                                                         # arrival radius [m] (was 0.3): agents can settle
                                                                       # 0.3-0.7 m short of the goal (tracking equilibrium
                                                                       # with the cruise-speed reference at the path end)
COL_DX, SIDE_Y = 1.5, 2.0                                              # column spacing, side-agent lane


def family_layout(family, K):
    """The two demo families for any K (K = 2 reproduces LAYOUTS['A_column'] / ['C_axis_side'] exactly).
    A_column:    K agents in one column on the axis, agent k starts at (-1.5k, 0) and flies to the mirrored
                 goal (20 + 1.5(K-1-k), 0): every agent meets the wall saddle on the axis.
    C_axis_side: one axis agent (0,0) -> (20,0) plus K-1 side agents in +-y pairs on the lanes y = +-2
                 (agent k >= 1: column c = (k-1)//2 at x = -1.5c, y = +2 for odd k, -2 for even k), each
                 crossing to the mirrored lane at x = 20 + 1.5c."""
    if family == "A_column":
        starts = [(-COL_DX * k, 0.0, Z0) for k in range(K)]
        goals = [(20.0 + COL_DX * (K - 1 - k), 0.0, Z0) for k in range(K)]
    elif family == "C_axis_side":
        starts, goals = [(0.0, 0.0, Z0)], [(20.0, 0.0, Z0)]
        for k in range(1, K):
            c = (k - 1) // 2
            y = SIDE_Y if k % 2 == 1 else -SIDE_Y
            starts.append((-COL_DX * c, y, Z0))
            goals.append((20.0 + COL_DX * c, -y, Z0))
    elif family in ("C_lanes", "C_cross_after", "C_cross_before"):
        # C variants (axis agent + +-y side pairs) tuned to keep the axis agent's symmetric wall approach:
        #   C_lanes        side pairs on y = +-2.5 lanes, no crossing
        #   C_cross_after  crossing pairs, start (2 - 1.5c, +-2) -> (24 + 1.5c, -+2): crossing at x ~ 13, behind the wall
        #   C_cross_before crossing pairs, start (-10 - 1.5c, +-2) -> (20 + 1.5c, -+2): crossing at x ~ 5, after the
        #                  axis agent has passed and before the wall
        starts, goals = [(0.0, 0.0, Z0)], [(20.0, 0.0, Z0)]
        for k in range(1, K):
            c = (k - 1) // 2
            sgn = 1.0 if k % 2 == 1 else -1.0
            if family == "C_lanes":
                starts.append((-COL_DX * c, 2.5 * sgn, Z0)); goals.append((20.0 + COL_DX * c, 2.5 * sgn, Z0))
            elif family == "C_cross_after":
                starts.append((2.0 - COL_DX * c, SIDE_Y * sgn, Z0)); goals.append((24.0 + COL_DX * c, -SIDE_Y * sgn, Z0))
            else:
                starts.append((-10.0 - COL_DX * c, SIDE_Y * sgn, Z0)); goals.append((20.0 + COL_DX * c, -SIDE_Y * sgn, Z0))
    else:
        raise KeyError(family)
    return dict(starts=starts, goals=goals)


def get_layout(name):
    """LAYOUTS entry, or '<family>_K<K>' for the scalable families (e.g. A_column_K6, C_axis_side_K10)."""
    if name in LAYOUTS:
        return LAYOUTS[name]
    fam, k = name.rsplit("_K", 1)
    return family_layout(fam, int(k))


def make_env(cap, maxcyc, mppi_lambda=0.01):
    """Base parameters only: the course (apply_variant, which calls apply() itself) is added by make_scene_v2."""
    P = C.make_P()
    P.budget = np.inf
    P.lm_maxit = P.ilqr_Kmax = int(cap)
    P.maxcyc = int(maxcyc)
    P.reachtol = REACHTOL
    P.lam_mppi, P.ess_target_mppi = float(mppi_lambda), None
    return P


def make_method(method, K, N, P, M, JX, MK, col, sst):
    n = M.n
    xr2 = lambda x, t: MU.ref_window_multi(x, N, MK, M, P, 2, t)
    xr1K = lambda x, t: MU.ref_window_multi(x, N, MK, M, P, 1, t)
    st = SimpleNamespace(seed=np.zeros(K * n), U=None, W=None)
    spent = lambda: sst.t_spent if sst is not None else 0.0

    if method in ("Hopf_Lax_MPC", "PMP_Shooting", "Vanilla_Hopf_Lax"):
        Sh = MI.build_ss_int(M, K, N, P.dt, JX, col=col, napply=P.napply)

        def init(x):
            Xr0 = xr2(x, 0.0)
            Sh.warm(x, Xr0)
            Sh.roll_controls(np.zeros(K * n), x, Xr0)
            Jb = S.ensure_jbatch(Sh)
            Jb(np.zeros((2, K * n)), x, Xr0); Jb(np.zeros((7, K * n)), x, Xr0)
            if method == "Vanilla_Hopf_Lax":
                np.asarray(S.ensure_hessian(Sh)(np.zeros(K * n), x, Xr0))

        def step(x, t, tc):
            Xr = xr2(x, t)
            ts = time.perf_counter()
            if method == "Hopf_Lax_MPC":
                tr = []
                PP = SimpleNamespace(tol=P.lm_tol, maxit=P.lm_maxit)
                p0, itk, info = S.hopf_lax_mpc_solve(Sh, st.seed, x, Xr, PP, P.hopf_lax_mpc_epsc, S.HOPF_LAX_MPC_ETA_S,
                                                     P.hopf_lax_mpc_alpha, S.HOPF_LAX_MPC_DELTA, tc, P.budget, trace=tr,
                                                     stop=sst)
                status = NS.hopf_lax_mpc_status(tr, itk, PP.maxit)
                if status == "converged" and sst is not None and sst.fired:
                    status = "shared-stop"
                ex = dict(ndesc=info.ndesc, nesc=info.nesc, nkick=info.nkick, neig=info.neig)
            elif method == "PMP_Shooting":
                p0, itk, nr, status, ntr, mu = NS.pmp_shooting_solve_counted(Sh, st.seed, x, Xr, P, tc, stop=sst)
                ex = dict(nres=nr, ntrial=ntr, mu=mu)
            else:
                PP = SimpleNamespace(tol=TBVAN_TOL, maxit=P.lm_maxit)
                p0, itk, info = S.vanilla_hopf_lax_solve(Sh, st.seed, x, Xr, PP, tc, P.budget, stop=sst)
                status = "shared-stop" if (sst is not None and sst.fired) else info.status
                ex = dict(nacc=info.nacc, nrej=info.nrej, t_hess=info.t_hess, lminH=info.lminH)
            ex["_tsolve"] = time.perf_counter() - ts - spent()
            u = MI.ustar_multi(x, p0, M, JX)
            st.seed = np.asarray(Sh.roll_costate(p0, x, Xr))[:, P.napply]
            return u, p0, itk, status, ex

    elif method == "DDP_Box":
        DP = BI.build_ddp_int(M, K, N, P.dt, JX, col, box=True)

        def init(x):
            DP.warm(x, BI.joint_xref(xr1K(x, 0.0), n), C.mobs_center(M.mobs, 0.0))

        def step(x, t, tc):
            Xref, mc = BI.joint_xref(xr1K(x, t), n), C.mobs_center(M.mobs, t)
            Uinit = None if st.U is None else S.shift_U(st.U, P.napply)
            ts = time.perf_counter()
            p0, st.U, itk, info = BI.ddp_run_box(DP, x, Xref, mc, Uinit, P, tc, stop=sst)
            tsol = time.perf_counter() - ts - spent()
            u = MI.ustar_multi(x, p0, M, JX)
            return u, p0, itk, info.status, dict(nbad=info.nbad, nrej=info.nrej, n_eff=info.n_eff, mu=info.mu,
                                                 a_min=info.a_min, J=info.J, _tsolve=tsol)

    elif method.startswith("MPPI_"):
        ctrl = BI.MPPIInt(M, P, N=N, K=int(method[5:]), KA=K, col=col, lam=P.lam_mppi, alpha=P.alpha_mppi,
                          iters=P.iters_mppi, seed=0, ess_target=None)

        def init(x):
            ctrl.warm(x, BI.joint_xref(xr1K(x, 0.0), n), C.mobs_center(M.mobs, 0.0))

        def step(x, t, tc):
            Xref, mc = BI.joint_xref(xr1K(x, t), n), C.mobs_center(M.mobs, t)
            ts = time.perf_counter()
            u, U, ess = ctrl.solve(x, Xref, mc)
            tsol = time.perf_counter() - ts
            ctrl.shift(P.napply)
            h, wh = ctrl.sample_stats()                                # untimed
            return np.asarray(u), None, int(P.iters_mppi), "ok", dict(ess=float(ess), hit=h, whit=wh, _tsolve=tsol)

    elif method == "Collocation_IPOPT":
        coll = BI.build_collocation_int(M, K, N, P.dt, col, max_iter=P.lm_maxit, stop=sst)

        def init(x):
            if sst is not None:
                sst.reset(x.copy(), BI.joint_xref(xr1K(x, 0.0), n), C.mobs_center(M.mobs, 0.0))
            coll(x, xr1K(x, 0.0), None, time.perf_counter())         # warm IPOPT (discarded)

        def step(x, t, tc):
            ts = time.perf_counter()
            f, u0, w = coll(x, xr1K(x, t), st.W, tc)
            tsol = time.perf_counter() - ts - spent()
            st.W = S.shift_collocation_ipopt(w, coll.n, coll.m, N, P.napply)
            sts = coll.solver.stats()
            raw = str(sts.get("return_status", "?"))
            status = "shared-stop" if (sst is not None and sst.fired) else NS.COLLOCATION_IPOPT_STATUS.get(raw, raw)
            return np.asarray(u0), None, int(sts.get("iter_count", -1)), status, dict(ipopt_status=raw, f=f, _tsolve=tsol)
    else:
        raise SystemExit(f"unknown method {method!r}; use {', '.join(METHODS)}")
    return init, step


TBVAN_TOL = 1e-4


def run(method, layout, N, maxcyc, wall_limit, eps, every=100):
    L = get_layout(layout)
    col = MI.make_col()
    P = make_env(1000, maxcyc)
    P, M, JX, MK = MI.make_scene_v2(P, L["starts"], L["goals"])
    K, n = MK.K, M.n
    label = f"{method} {layout} K={K}"
    sst = None
    if eps is not None:
        DPc = BI.build_ddp_int(M, K, N, P.dt, JX, col)                  # the joint OCP cost (DDP rollout_cost)
        sst = SSTOP.SharedStop(lambda x0, U, Xref, mc: float(DPc.rollout_cost(x0, U, Xref, mc)[1]), eps, 1e-10)
    init, step = make_method(method, K, N, P, M, JX, MK, col, sst)
    x = MK.x0.copy(); t = 0.0
    if sst is not None:                                                # compile the shared evaluator outside the loop
        sst.reset(x.copy(), BI.joint_xref(MU.ref_window_multi(x, N, MK, M, P, 1, 0.0), n), C.mobs_center(M.mobs, 0.0))
        sst(np.zeros((3 * K, N)))
    t0 = time.perf_counter()
    init(x)
    t_compile = time.perf_counter() - t0
    print(f"[{label}] compiled + warmed in {t_compile:.1f} s", flush=True)
    Jreal, Jpen, Jpair = np.zeros(K), np.zeros(K), np.zeros((K, K))
    treach = np.full(K, np.nan)
    XL, UL, PL, TL = [x.copy()], [], [], []
    TC, TS, RC, ITS, ST, DMIN = [], [], [], [], [], []
    EX = {}
    t_start = time.perf_counter(); deadline = t_start + wall_limit
    aborted, err, end = False, None, "maxcyc"
    for c in range(1, P.maxcyc + 1):
        try:
            T.CF.fired = False
            if sst is not None:                                        # per cycle, BEFORE the timed section
                sst.reset(x.copy(), BI.joint_xref(MU.ref_window_multi(x, N, MK, M, P, 1, t), n), C.mobs_center(M.mobs, t))
            tc = time.perf_counter()
            P.budget = max(deadline - tc, 0.0)                         # = remaining wall time (no per-cycle budget)
            u, p0, itk, status, ex = step(x, t, tc)
            TC.append(time.perf_counter() - tc - (sst.t_spent if sst is not None else 0.0))
            TS.append(ex.pop("_tsolve"))
            RC.append(T.CF.fired)
            ITS.append(int(itk)); ST.append(status); TL.append(t)
            for kk, vv in ex.items():
                EX.setdefault(kk, []).append(vv)
            if not np.all(np.isfinite(u)):
                raise RuntimeError("non-finite control")
            if c % every == 1:
                its = np.asarray(ITS); ms = 1e3 * np.asarray(TS)
                print(f"[{label}] c{c:4d} t{t:5.2f} | xy {np.round(x.reshape(K, n)[:, :2], 2).tolist()} | arrived "
                      f"{int(np.sum(~np.isnan(treach)))}/{K} | it mean {its.mean():6.2f} max {its.max():4d} | solve ms "
                      f"mean {ms.mean():8.2f} max {ms.max():8.1f} | wall {time.perf_counter() - t_start:5.0f} s", flush=True)
            x, Jreal, Jpen, Jpair, at_goal, t, dmin, _ = MI.apply_step_int(
                x, u, MU.ref_window_multi(x, N, MK, M, P, 1, t), Jreal, Jpen, Jpair, M, MK, P, t, JX, col, done=None)
            DMIN.append(dmin)
            XL.append(x.copy()); UL.append(np.asarray(u).copy())
            PL.append(np.zeros(K * n) if p0 is None else np.asarray(p0).copy())
            newly = at_goal & np.isnan(treach); treach[newly] = t
            if np.all(~np.isnan(treach)):
                end = "all-arrived"; break
            if time.perf_counter() >= deadline:
                end = "wall-limit"; break
        except KeyboardInterrupt:
            raise
        except Exception as e_:                                       # noqa: BLE001
            aborted, err, end = True, repr(e_), "aborted"
            print(f"[{label}] ABORTED at cycle {c}: {err}", flush=True)
            break
    its = np.asarray(ITS, dtype=int); ts = np.asarray(TS); dm = np.asarray(DMIN)
    J_full = float(Jreal.sum() + Jpen.sum() + Jpair.sum())
    tgrid = np.arange(len(XL)) * P.dt_apply
    e = dict(method=method, layout=layout, K=K, N=N, label=label, col=vars(col).copy(), starts=np.array(MK.starts),
             goals=np.array(MK.goals), X=np.array(XL).T.reshape(K, n, -1),
             U=np.array(UL).T.reshape(K, M.m, -1) if UL else np.zeros((K, M.m, 0)),
             Pc=np.array(PL).T.reshape(K, n, -1) if PL else np.zeros((K, n, 0)),
             tlog=np.asarray(TL), tcyc=np.asarray(TC), tsolve=ts, recompile=np.asarray(RC, dtype=bool), iters=its,
             status=list(ST), extra={k: np.asarray(v) for k, v in EX.items()}, dmin=dm, Jreal=Jreal, Jpen=Jpen,
             Jpair=Jpair, Jtrue=Jreal + Jpen, Jpair_total=float(np.sum(Jpair)), J_full=J_full, t_reach=treach,
             all_reached=bool(np.all(~np.isnan(treach))), end=end, complete=(end == "all-arrived"), ncyc=len(ITS),
             aborted=aborted, error=err, wall_s=time.perf_counter() - t_start, t_compile=t_compile,
             stamp=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
             meta=dict(dt=P.dt, dt_apply=P.dt_apply, napply=P.napply, maxcyc=P.maxcyc, wall_limit=wall_limit,
                       shared_stop_eps=eps, lam_mppi=P.lam_mppi, scenario=MK.scenario, reachtol=P.reachtol,
                       lm_maxit=P.lm_maxit, ilqr_Kmax=P.ilqr_Kmax, vanilla_tol=TBVAN_TOL,
                       stop_rule="all agents arrived at least once", cost="full run, all terms"),
             paths=[dict(pts=p.pts, s=p.s, th=p.th, len=p.len) for p in MK.paths],
             obs=dict(center=M.obs.center, ax=M.obs.ax, ay=M.obs.ay, az=M.obs.az),
             mobs=dict(c0=M.mobs.c0, ax=M.mobs.ax, ay=M.mobs.ay, az=M.mobs.az),
             mobs_t=np.stack([C.mobs_center(M.mobs, tk) for tk in tgrid], axis=-1))
    nst = {s: ST.count(s) for s in sorted(set(ST))}
    print(f"[{label}] END ({end}): {len(ITS)} cycles, arrived {int(np.sum(~np.isnan(treach)))}/{K} t_reach "
          f"{np.round(treach, 2).tolist()} | J_full {J_full:.3f} (track {Jreal.sum():.3f} obs {Jpen.sum():.3f} pair "
          f"{Jpair.sum():.3f}) | min dist {dm.min() if dm.size and K > 1 else np.inf:.3f} | it mean {its.mean():.2f} | "
          f"solve ms mean {1e3 * ts[1:].mean() if ts.size > 1 else np.nan:.2f} | status {nst} | wall {e['wall_s']:.0f} s",
          flush=True)
    return e


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--layout", help="a LAYOUTS name or <family>_K<K> (families A_column, C_axis_side)")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--maxcyc", type=int, default=1500)
    ap.add_argument("--wall-limit", type=float, default=1800.0)
    ap.add_argument("--shared-stop", type=float, default=1e-6, help="eps of the shared rule; <= 0 disables it")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv[1:])
    if args.list:
        for k, v in LAYOUTS.items():
            print(f"{k:14s} starts {v['starts']}  goals {v['goals']}")
        return 0
    d = os.path.join(args.outdir, args.layout)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{args.method}.pkl")
    if os.path.exists(out) and not args.redo:
        print(f"-- {out} exists; skipping (--redo)")
        return 0
    print(f"#### {args.method} layout {args.layout} N={args.N}  ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    e = run(args.method, args.layout, args.N, args.maxcyc, args.wall_limit,
            args.shared_stop if args.shared_stop > 0 else None)
    with open(out + ".tmp", "wb") as f:
        pickle.dump(e, f)
    os.replace(out + ".tmp", out)
    print(f"#### saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

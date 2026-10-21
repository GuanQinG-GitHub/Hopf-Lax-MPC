"""
make_animation_sweep.py -- 3-D animation of ALL methods of the no-budget N sweep at one horizon N.

Renders with closed_loop/make_animations.py (same scene, camera, colours, legend names).  The sweep does
not log the per-cycle predicted horizons or the MPPI sample clouds, so the video shows each method's
driven trajectory (comet tail + current position) against the moving obstacles at their TRUE positions;
the predicted-path overlay of the testbed animations is not drawn.

USAGE (from N_sweep/)
    python make_animation_sweep.py --N 275 [--pkl results_N_sweep_shared_eps1e-06_wall_saddle_bumps_rand.pkl] [--out figs/anim_N275.mp4]
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)  # repository layout:
                 for d in ("core", "closed_loop")]                     # shared modules in sibling folders
import mpc_core as C                                                   # noqa: E402
import make_animations as MA                                           # noqa: E402

ORDER = ["PMP_Shooting", "DDP_Box", "DDP", "Collocation_IPOPT", "Hopf_Lax_MPC", "MPPI", "MPPI_20480", "Vanilla_Hopf_Lax"]
COL = {"Hopf_Lax_MPC": (.10, .55, .20), "PMP_Shooting": (.85, .15, .15), "Vanilla_Hopf_Lax": (0.0, .60, .60),
       "DDP_Box": (.65, .30, .0), "DDP": (.90, .45, .0), "Collocation_IPOPT": (0, .45, .85), "MPPI": (.60, .20, .70), "MPPI_20480": (.40, .10, .50)}


def runner_from_sweep(e, M):
    """Sweep run -> the runner layout make_animations.render reads (no predictions / clouds)."""
    X = np.asarray(e["X"])                                             # (n, ncyc+1), one column per cycle
    n = len(e["tlog"])
    pos = [X[M.posidx, min(k, X.shape[1] - 1)].reshape(3, 1) for k in range(n)]
    res = SimpleNamespace(X=X, ncyc=n, xlog=X[:, :n], Xpred=pos, Xrefw=pos, extra=[None] * n)
    return SimpleNamespace(name=e["method"], color=COL.get(e["method"], (.3, .3, .3)), style="-", res=res,
                           N=e["N"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--pkl", default="results_N_sweep_shared_eps1e-06_wall_saddle_bumps_rand.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        D = pickle.load(f)
    P = copy.copy(D["P"])
    P.napply = 1                                                       # sweep X is logged once per cycle
    M = C.model_scn(D["P"])
    R = [runner_from_sweep(D["runs"][(m, args.N)], M) for m in ORDER if (m, args.N) in D["runs"]]
    for r in R:
        e = D["runs"][(r.name, args.N)]
        print(f"  {r.name:<18} reached={e['reached']}  J={e['Jtrue']:8.2f}  cycles={r.res.ncyc}")
    out = args.out or f"figs/anim_all_methods_N{args.N}.mp4"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rule = D.get("config", {}).get("shared_stop_eps")
    ttl = args.title or (f"no budget, N = {args.N}" + (f", shared stop eps {rule:g}" if rule else ""))
    MA.render({"R": R, "P": P, "M": M, "path": D["path"], "obs_motion": D.get("obs_motion")},
              list(range(len(R))), out, ttl)


if __name__ == "__main__":
    main()

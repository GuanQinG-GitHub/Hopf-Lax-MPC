"""
plot_final_frames.py -- final frame (end of run) of one method for every K of a demo family, one panel per K.

Each panel: static obstacles, the moving bodies at their final position, references, the full closed-loop
trails, final positions (black edge) with the arrival circle (reach tolerance) around every goal, and a title
with arrivals and the cost.  Runs are read from <dir>/<family>_K<K>/<method>.pkl (K = 2 can come from a
second directory, e.g. the tuned K = 2 layouts).

USAGE
    python plot_final_frames.py --family A_column --dirs results_demo_screen,results_demo_int_tol03 [--method Hopf_Lax_MPC]
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Ellipse                                 # noqa: E402

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d) for _d in ("multi_agent",)]
import plot_multi as PM                                                # noqa: E402
from plot_demo_int import LBL, xlim_of                                 # noqa: E402

RED = (.85, .18, .18)


def find(dirs, family, K, method):
    for d in dirs:
        for name in (f"{family}_K{K}", family if K == 2 else None):
            if name and os.path.exists(os.path.join(d, name, method + ".pkl")):
                return pickle.load(open(os.path.join(d, name, method + ".pkl"), "rb"))
    return None


def panel(ax, e):
    K = e["K"]; cols = PM.agent_colors(K)
    so, mo, mt = e["obs"], e["mobs"], e["mobs_t"]
    for i in range(so["center"].shape[1]):
        ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=RED, alpha=.10))
        ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=RED, alpha=.35))
    for i in range(mo["c0"].shape[1]):
        c = mt[:2, i, -1]
        ax.add_patch(Ellipse(c, 2.5 * mo["ax"][i], 2.5 * mo["ay"][i], color=(.55, .1, .1), alpha=.12))
        ax.add_patch(Ellipse(c, 2 * mo["ax"][i], 2 * mo["ay"][i], color=(.55, .1, .1), alpha=.5))
    rt = e["meta"]["reachtol"]
    for k in range(K):
        X, p, g = e["X"][k], e["paths"][k], e["goals"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.5, alpha=0.35)
        ax.plot(X[0], X[1], "-", color=cols[k], lw=1.3)
        ax.plot(g[0], g[1], "*", color=cols[k], ms=8)
        ax.add_patch(plt.Circle(g[:2], rt, fill=False, ls=":", color=cols[k], lw=0.8))
        ax.plot(X[0, -1], X[1, -1], "o", color=cols[k], ms=5, mec="k", mew=0.7)
    arr = int(np.sum(~np.isnan(e["t_reach"])))
    tend = (e["X"].shape[2] - 1) * e["meta"]["dt_apply"]
    ax.set_title(f"K={K}: {arr}/{K} arrived (tol {rt:g} m), t_end {tend:.1f} s ({e['end']}), J {e['J_full']:.1f}",
                 fontsize=8.5, loc="left")
    ax.set_xlim(*xlim_of(e)); ax.set_ylim(-2.8, 2.8); ax.set_aspect("equal"); ax.grid(alpha=.2)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--dirs", default="results_demo_screen,results_demo_int_tol03")
    ap.add_argument("--method", default="Hopf_Lax_MPC")
    ap.add_argument("--Ks", default="2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dirs = args.dirs.split(",")
    E = [e for e in (find(dirs, args.family, int(k), args.method) for k in args.Ks.split(",")) if e is not None]
    fig, axs = plt.subplots(len(E), 1, figsize=(15, 2.35 * len(E)), dpi=120, squeeze=False)
    for ax, e in zip(axs[:, 0], E):
        panel(ax, e)
    fig.suptitle(f"{LBL[args.method]} -- family {args.family}, final frame of every K (course {E[0]['meta']['scenario']}, "
                 f"N = {E[0]['N']}; moving bodies at their final position; dotted circle = arrival radius; black-edged dot = "
                 f"final position)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = args.out or os.path.join("figs", "demo", f"final_frames_{args.family}_{args.method}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({len(E)} panels: K = {[e['K'] for e in E]})")


if __name__ == "__main__":
    main()

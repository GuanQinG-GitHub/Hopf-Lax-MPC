"""
plot_c_variants.py -- annotated trajectory figures of the C-family layouts (screening runs, Hopf-Lax-MPC vs PMP).

For each layout family (C_axis_side, C_lanes, C_cross_before, C_cross_after): a 2 x 2 grid, rows K = 4 / 7,
columns Hopf-Lax-MPC / PMP-Shooting.  Annotations: the stopped wall (the saddle), the axis agent, where each
side agent crosses the axis (y changes sign), agents trapped in front of the wall (ringed), agents that did not
arrive (arrival radius 0.8 m, evaluated on the trajectory), and per panel arrivals + trajectory cost.
Writes figs/c_variants/<family>.png.

USAGE
    python plot_c_variants.py [--dir results_demo_screen]
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

RED = (.85, .18, .18)
TOL = 0.8
FAMS = ["C_axis_side", "C_lanes", "C_cross_before", "C_cross_after"]
TITLE = {"C_axis_side": "original C_axis_side: side pairs start BESIDE the axis agent (y = +-2, same x) and cross at x ~ 12-15; "
                        "a side agent is 1.3 m from the axis agent when it nears the wall",
         "C_lanes": "C_lanes: side pairs stay on y = +-2.5 (no crossing); nearest side agent 2.5 m away when the axis agent nears the wall",
         "C_cross_before": "C_cross_before: side pairs start 10 m BEHIND and cross at x ~ 7-14 after the axis agent has met the wall; "
                           "nearest side agent 10.3 m away when the axis agent nears the wall",
         "C_cross_after": "C_cross_after: side pairs start 2 m AHEAD and cross at x ~ 6-18; a side agent is 1.2 m from the axis agent "
                          "when it nears the wall"}
MLBL = {"Hopf_Lax_MPC": "Hopf-Lax-MPC (ours)", "PMP_Shooting": "PMP-Shooting"}


def panel(ax, e):
    K = e["K"]; cols = PM.agent_colors(K); X = e["X"]
    so, mo, mt = e["obs"], e["mobs"], e["mobs_t"]
    for i in range(so["center"].shape[1]):
        ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=RED, alpha=.08))
        ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=RED, alpha=.28))
    for i in range(mo["c0"].shape[1]):
        c = mt[:2, i, -1]
        ax.add_patch(Ellipse(c, 2 * mo["ax"][i], 2 * mo["ay"][i], color=(.45, .05, .05), alpha=.55))
    wall = mt[:2, 0, -1]
    ax.annotate("wall (stops on the axis at t = 6.4 s)\n= the symmetric saddle", xy=wall, xytext=(wall[0] - 1.2, -3.15),
                fontsize=7, color=(.45, .05, .05), arrowprops=dict(arrowstyle="->", color=(.45, .05, .05), lw=0.8))
    dist = np.array([np.linalg.norm(X[k, :3, :] - e["goals"][k][:, None], axis=0).min() for k in range(K)])
    arrived = dist < TOL
    crossed = False
    for k in range(K):
        ax.plot(X[k, 0], X[k, 1], "-", color=cols[k], lw=1.4)
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=4)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=9)
        ax.plot(X[k, 0, -1], X[k, 1, -1], "o", color=cols[k], ms=5, mec="k", mew=0.7)
        if k > 0:                                                      # side agent: where it crosses the axis
            y = X[k, 1]
            s = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
            if s.size and abs(e["starts"][k][1]) > 0.5 and np.sign(e["starts"][k][1]) != np.sign(e["goals"][k][1]):
                i = s[0]
                ax.plot(X[k, 0, i], 0.0, "x", color=cols[k], ms=7, mew=1.6)
                if not crossed:
                    ax.annotate("side agents cross\nthe axis here (x)", xy=(X[k, 0, i], 0.0), xytext=(X[k, 0, i] - 3.0, 2.85),
                                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
                    crossed = True
        if not arrived[k]:
            q = X[k, :2, -1]
            trapped = q[0] < 10.0
            ax.add_patch(plt.Circle(q, 0.55, fill=False, color="k", lw=1.6, ls="-" if trapped else "--"))
            ax.annotate("TRAPPED at the wall saddle" if trapped else "not arrived", xy=q,
                        xytext=(q[0] - 4.5, q[1] + (1.6 if q[1] <= 0 else -1.8)), fontsize=8, fontweight="bold",
                        color="k", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("axis agent", xy=(e["starts"][0][0], 0.0), xytext=(e["starts"][0][0] - 2.6, -1.0), fontsize=7,
                color=cols[0], arrowprops=dict(arrowstyle="->", color=cols[0], lw=0.7))
    xs = [v[0] for v in e["starts"]] + [v[0] for v in e["goals"]]
    ax.set_xlim(min(min(xs), 0) - 3.2, max(max(xs), 20) + 1.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal"); ax.grid(alpha=.2); ax.tick_params(labelsize=7)
    ax.set_title(f"{MLBL[e['method']]}, K = {K}: {int(arrived.sum())}/{K} arrived, trajectory cost J = {e['J_full']:.1f}",
                 fontsize=9, loc="left", color=(0.8, 0.1, 0.1) if e["method"] == "Hopf_Lax_MPC" else (0.1, 0.3, 0.8))
    return int(arrived.sum()), e["J_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_demo_screen")
    ap.add_argument("--outdir", default="figs/c_variants")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for fam in FAMS:
        fig, axs = plt.subplots(2, 2, figsize=(17, 6.0), dpi=110)
        for r, K in enumerate((4, 7)):
            for c, m in enumerate(("Hopf_Lax_MPC", "PMP_Shooting")):
                e = pickle.load(open(os.path.join(args.dir, f"{fam}_K{K}", m + ".pkl"), "rb"))
                a, J = panel(axs[r, c], e)
                rows.append((fam, K, m, a, J))
        fig.suptitle(TITLE[fam] + "\n(dark = moving bodies at their final position, * = goal, black-edged dot = final "
                     "position, x = where a side agent crosses the axis)", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(args.outdir, f"{fam}.png")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)
    for r in rows:
        print(f"{r[0]:15s} K={r[1]} {r[2]:13s} arrived {r[3]}/{r[1]}  J {r[4]:.1f}")


if __name__ == "__main__":
    main()

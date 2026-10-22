"""
plot_e_cross.py -- annotated screening figures of the E_cross layouts (mirror pairs crossing the axis after obs-1 and
the first bump; Hopf-Lax-MPC vs PMP-Shooting).

One row per K, columns Hopf-Lax-MPC / PMP-Shooting.  Annotations: the planned crossing point (E_XC, 0), the references
(dashed), where each pair agent actually crossed the axis (x), the stopped wall, agents that did not arrive (arrival
radius 0.8 m), ringed and labelled DEADLOCK (stopped before the wall, off the axis: the symmetric crossing) or TRAPPED
at the wall (on the axis in front of it), and per panel arrivals + trajectory cost.  Writes figs/e_cross/<family>.png.

USAGE
    python plot_e_cross.py [--dir results_demo_screen_E] [--family E_cross] [--Ks 2,3,4,5]
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
import mpc_demo_int as D                                               # noqa: E402

RED = (.85, .18, .18)
TOL = 0.8
MLBL = {"Hopf_Lax_MPC": "Hopf-Lax-MPC (ours)", "PMP_Shooting": "PMP-Shooting"}
ALL = ["Hopf_Lax_MPC", "PMP_Shooting", "DDP_Box", "Vanilla_Hopf_Lax", "Collocation_IPOPT", "MPPI_12288", "MPPI_20480"]
ALBL = dict(MLBL, DDP_Box="Box-DDP", Vanilla_Hopf_Lax="Vanilla-Hopf-Lax", Collocation_IPOPT="Collocation-IPOPT",
            MPPI_12288="MPPI (12288)", MPPI_20480="MPPI (20480)")
ACOL = {"Hopf_Lax_MPC": (0.85, 0.10, 0.10), "PMP_Shooting": (0.10, 0.35, 0.80), "DDP_Box": (0.55, 0.35, 0.05),
        "Collocation_IPOPT": (0.20, 0.60, 0.20), "Vanilla_Hopf_Lax": (0.0, 0.60, 0.60), "MPPI_12288": (0.45, 0.25, 0.70),
        "MPPI_20480": (0.65, 0.45, 0.85)}


def panel(ax, e):
    K = e["K"]; cols = PM.agent_colors(K); X = e["X"]
    so, mo, mt = e["obs"], e["mobs"], e["mobs_t"]
    for i in range(so["center"].shape[1]):
        ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=RED, alpha=.08))
        ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=RED, alpha=.28))
    for i in range(mo["c0"].shape[1]):
        ax.add_patch(Ellipse(mt[:2, i, -1], 2 * mo["ax"][i], 2 * mo["ay"][i], color=(.45, .05, .05), alpha=.55))
    wall = mt[:2, 0, -1]
    ax.annotate("wall (stops at t = 6.4 s)", xy=wall, xytext=(wall[0] + 0.6, -3.2), fontsize=7, color=(.45, .05, .05),
                arrowprops=dict(arrowstyle="->", color=(.45, .05, .05), lw=0.8))
    ax.plot(D.E_XC, 0.0, "P", color="k", ms=9, mfc="none", mew=1.5, zorder=6)
    ax.annotate("planned crossing\n(mirror pair arrives together)", xy=(D.E_XC, 0.0), xytext=(D.E_XC - 6.0, 3.0),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    dist = np.array([np.linalg.norm(X[k, :3, :] - e["goals"][k][:, None], axis=0).min() for k in range(K)])
    arrived = dist < TOL
    labelled = set()
    for k in range(K):
        p = e["paths"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.7, alpha=0.6)
        ax.plot(X[k, 0], X[k, 1], "-", color=cols[k], lw=1.5)
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=4)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=9)
        ax.plot(X[k, 0, -1], X[k, 1, -1], "o", color=cols[k], ms=5, mec="k", mew=0.7)
        y = X[k, 1]
        s = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
        if abs(e["starts"][k][1]) > 0.5 and s.size:
            ax.plot(X[k, 0, s[0]], 0.0, "x", color=cols[k], ms=7, mew=1.6, zorder=6)
        if not arrived[k]:
            q = X[k, :2, -1]
            on_axis = abs(q[1]) < 0.3 and abs(e["starts"][k][1]) < 0.3
            kind = "TRAPPED at the wall" if on_axis else ("DEADLOCK at the crossing" if q[0] < wall[0] else "not arrived")
            ax.add_patch(plt.Circle(q, 0.55, fill=False, color="k", lw=1.6))
            if kind not in labelled:
                labelled.add(kind)
                ax.annotate(kind, xy=q, xytext=(q[0] + (1.5 if on_axis else -2.5), q[1] + (1.9 if q[1] >= 0 else -1.9)),
                            fontsize=8, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.0))
    xs = [v[0] for v in e["starts"]] + [v[0] for v in e["goals"]]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.0); ax.set_ylim(-3.6, 3.6)
    ax.set_aspect("equal"); ax.grid(alpha=.2); ax.tick_params(labelsize=7)
    its, ts = e["iters"][1:], e["tsolve"][1:]
    lb = " (stopped: lower bound)" if e["end"] == "deadlock" else ""
    ax.set_title(f"{ALBL[e['method']]}, K = {K}: {int(arrived.sum())}/{K} arrived, J = {e['J_full']:.1f}{lb}; "
                 f"{its.mean():.1f} it/cycle, {1e3 * ts.mean():.0f} ms/cycle", fontsize=8.5, loc="left", color=ACOL[e["method"]])
    return int(arrived.sum()), e["J_full"]


def per_k(d, family, K, outdir):
    """All methods of one K: one map per method + solve time per cycle vs t."""
    E = [pickle.load(open(os.path.join(d, f"{family}_K{K}", m + ".pkl"), "rb")) for m in ALL
         if os.path.exists(os.path.join(d, f"{family}_K{K}", m + ".pkl"))]
    n = len(E) + 1; nc = 2; nr = -(-n // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(17, 2.9 * nr), dpi=110, squeeze=False)
    for i, e in enumerate(E):
        panel(axs[i // nc, i % nc], e)
    a = axs[len(E) // nc, len(E) % nc]
    for e in E:
        a.semilogy(e["tlog"][1:], np.maximum(1e3 * e["tsolve"][1:], 1e-1), "-", lw=0.7, color=ACOL[e["method"]],
                   label=ALBL[e["method"]])
    a.set_title("solve time per cycle [ms] vs t (spikes: Box-DDP mu-max cycles, our saddle escapes)", fontsize=8.5, loc="left")
    a.set_xlabel("t [s]"); a.grid(alpha=.25); a.legend(fontsize=7, frameon=False, ncol=2)
    for j in range(n, nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.suptitle(f"{family}, K = {K}: all methods (dashed = reference, x = where an agent crossed the axis, black-edged dot = "
                 f"final position, ring = not arrived)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, f"{family}_K{K}_all.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-methods", action="store_true", help="one figure per K with every method")
    ap.add_argument("--dir", default="results_demo_screen_E")
    ap.add_argument("--family", default="E_cross")
    ap.add_argument("--Ks", default="2,3,4,5")
    ap.add_argument("--outdir", default="figs/e_cross")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.all_methods:
        for K in [int(k) for k in args.Ks.split(",")]:
            if os.path.isdir(os.path.join(args.dir, f"{args.family}_K{K}")):
                per_k(args.dir, args.family, K, args.outdir)
        return
    Ks = [int(k) for k in args.Ks.split(",")
          if all(os.path.exists(os.path.join(args.dir, f"{args.family}_K{k}", m + ".pkl")) for m in MLBL)]
    fig, axs = plt.subplots(len(Ks), 2, figsize=(17, 2.9 * len(Ks)), dpi=110, squeeze=False)
    rows = []
    for r, K in enumerate(Ks):
        for c, m in enumerate(MLBL):
            e = pickle.load(open(os.path.join(args.dir, f"{args.family}_K{K}", m + ".pkl"), "rb"))
            rows.append((K, m) + panel(axs[r, c], e))
    fig.suptitle(f"{args.family}: mirror pairs from near the origin cross the axis at x = {D.E_XC} (after obs-1 and the "
                 f"bump at 4.4), then take the mirrored lane y = -+{D.E_YL}\n(dashed = reference, x = where an agent "
                 f"actually crossed the axis, black-edged dot = final position, ring = not arrived)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.outdir, f"{args.family}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)
    for K, m, a, J in rows:
        print(f"K={K} {m:13s} arrived {a}/{K}  J {J:.1f}")


if __name__ == "__main__":
    main()

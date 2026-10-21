"""
plot_demo_int.py -- all methods of one K-agent demo layout (results_demo_int/<layout>/<method>.pkl, mpc_demo_int.py).

  figs/demo/<layout>/paths.png   one map panel per method: static obstacles, the moving bodies at their final
                                 (stopped) positions with their tracks dotted, references, closed-loop paths
  figs/demo/<layout>/time.png    solve time per cycle vs t (log) and minimum pairwise distance vs t
plus the printed table (arrival, J_full and its parts, iterations, DDP rejections / mu-max, solve ms, status).

USAGE
    python plot_demo_int.py --layout A_column
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Ellipse                                 # noqa: E402

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d) for _d in ("multi_agent",)]
import plot_multi as PM                                                # noqa: E402  agent_colors

ORDER = ["Hopf_Lax_MPC", "PMP_Shooting", "DDP_Box", "Collocation_IPOPT", "Vanilla_Hopf_Lax", "MPPI_12288", "MPPI_20480"]
LBL = {"Hopf_Lax_MPC": "Hopf-Lax-MPC (ours)", "PMP_Shooting": "PMP-Shooting", "DDP_Box": "Box-DDP",
       "Collocation_IPOPT": "Collocation-IPOPT", "Vanilla_Hopf_Lax": "Vanilla-Hopf-Lax", "MPPI_12288": "MPPI (12288)",
       "MPPI_20480": "MPPI (20480)"}
COL = {"Hopf_Lax_MPC": (0.85, 0.10, 0.10), "PMP_Shooting": (0.10, 0.35, 0.80), "DDP_Box": (0.55, 0.35, 0.05),
       "Collocation_IPOPT": (0.20, 0.60, 0.20), "Vanilla_Hopf_Lax": (0.0, 0.60, 0.60), "MPPI_12288": (0.45, 0.25, 0.70),
       "MPPI_20480": (0.65, 0.45, 0.85)}
OK_STATUS = ("converged", "shared-stop", "ok", "acceptable")


def xlim_of(e, pad=1.5):
    """x range covering every start and goal (the K-agent columns extend beyond the course)."""
    xs = [v[0] for v in e["starts"]] + [v[0] for v in e["goals"]]
    return min(min(xs), 0.0) - pad, max(max(xs), 20.0) + pad


def load(d):
    R = {}
    for f in glob.glob(os.path.join(d, "*.pkl")):
        e = pickle.load(open(f, "rb"))
        R[e["method"]] = e
    return [R[m] for m in ORDER if m in R]


def draw_scene(ax, e):
    so, mo, mt = e["obs"], e["mobs"], e["mobs_t"]
    red = (.85, .18, .18)
    for i in range(so["center"].shape[1]):
        ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=red, alpha=.10))
        ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=red, alpha=.35))
    for i in range(mo["c0"].shape[1]):
        tr = mt[:2, i, :]
        if np.all(np.abs(tr[1]) > 4):
            continue
        if np.ptp(tr[0]) + np.ptp(tr[1]) > 1e-6:
            ax.plot(tr[0], tr[1], ":", color=red, lw=0.8, alpha=0.6)
        c = tr[:, -1]
        ax.add_patch(Ellipse(c, 2.5 * mo["ax"][i], 2.5 * mo["ay"][i], color=(.6, .1, .1), alpha=.12))
        ax.add_patch(Ellipse(c, 2 * mo["ax"][i], 2 * mo["ay"][i], color=(.6, .1, .1), alpha=.45))


def stats(e):
    its, ts = e["iters"][1:], e["tsolve"][1:]
    pos = its > 0
    return dict(it=its.mean(), itmax=int(its.max()) if its.size else 0, ms=1e3 * ts.mean(), msmax=1e3 * ts.max(),
                pi=(1e3 * np.mean(ts[pos] / its[pos]) if pos.any() else np.nan),
                conv=100 * np.mean([s in OK_STATUS for s in e["status"]]))


def plot_paths(E, out):
    nm = len(E); ncol = 2; nrow = -(-nm // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow), dpi=120, squeeze=False)
    for i, e in enumerate(E):
        ax = axs[i // ncol, i % ncol]
        K = e["K"]; cols = PM.agent_colors(K)
        draw_scene(ax, e)
        for k in range(K):
            p = e["paths"][k]; X = e["X"][k]
            ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.6, alpha=0.4)
            ax.plot(X[0], X[1], "-", color=cols[k], lw=1.5)
            ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=4)
            ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=9)
            ax.plot(X[0, -1], X[1, -1], "o", color=cols[k], ms=6, mec="k", mew=0.6)
        s = stats(e)
        arr = f"{int(np.sum(~np.isnan(e['t_reach'])))}/{K} arrive"
        ax.set_title(f"{LBL[e['method']]}: {arr}, J {e['J_full']:.2f}, {s['it']:.1f} it/cyc, {s['ms']:.1f} ms/cyc",
                     fontsize=8.5, loc="left", color=COL[e["method"]])
        ax.set_xlim(*xlim_of(e)); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.25)
        ax.tick_params(labelsize=7)
    for j in range(nm, nrow * ncol):
        axs[j // ncol, j % ncol].axis("off")
    e0 = E[0]
    fig.suptitle(f"layout {e0['layout']}: K={e0['K']}, N={e0['N']}, course {e0['meta']['scenario']} (moving bodies at their final "
                 f"positions, dotted = their tracks), shared stop eps {e0['meta']['shared_stop_eps']:g}; dashed = reference, "
                 f"black-edged dot = final position", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_time(E, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), dpi=120)
    for e in E:
        ax[0].semilogy(e["tlog"], np.maximum(1e3 * e["tsolve"], 1e-2), "-", color=COL[e["method"]], lw=0.8,
                       label=f"{LBL[e['method']]}: mean {stats(e)['ms']:.1f} ms")
        if e["K"] > 1:
            ax[1].plot(np.arange(1, e["dmin"].size + 1) * e["meta"]["dt_apply"], e["dmin"], "-", color=COL[e["method"]],
                       lw=1.0, label=LBL[e["method"]])
    ax[0].set_title("solve time per cycle [ms] (shared-stop evaluation excluded)", fontsize=9, loc="left")
    ax[1].axhline(E[0]["col"]["Rc"], color="k", ls=":", lw=0.8)
    ax[1].set_title("minimum pairwise distance [m]", fontsize=9, loc="left")
    for a in ax:
        a.grid(alpha=.25); a.set_xlabel("t [s]"); a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def print_table(E):
    print(f"{'method':<20}{'end':>12}{'arrived':>8}{'t_last':>7}{'J_full':>9}{'track':>8}{'obs':>8}{'pair':>7}{'dmin':>6}"
          f"{'it':>7}{'it max':>7}{'ms/cyc':>8}{'max ms':>8}{'ms/it':>7}  status / extra")
    print("-" * 150)
    for e in E:
        s = stats(e)
        tl = np.nanmax(e["t_reach"]) if e["all_reached"] else np.nan
        nst = {k: e["status"].count(k) for k in sorted(set(e["status"]))}
        ex = ""
        if e["method"] == "DDP_Box":
            x = e["extra"]
            ex = f" | rejections {int(x['nrej'].sum())}, Quu-not-PD {int(x['nbad'].sum())}"
        if e["method"].startswith("MPPI"):
            ex = f" | ESS mean {np.mean(e['extra']['ess']):.0f}, hit {100 * np.nanmean(e['extra']['hit']):.1f}%"
        print(f"{LBL[e['method']]:<20}{e['end']:>12}{int(np.sum(~np.isnan(e['t_reach']))):>5d}/{e['K']:<2d}{tl:>7.2f}"
              f"{e['J_full']:>9.3f}{e['Jreal'].sum():>8.3f}{e['Jpen'].sum():>8.3f}{e['Jpair_total']:>7.3f}"
              f"{(e['dmin'].min() if (e['dmin'].size and e['K'] > 1) else np.nan):>6.2f}{s['it']:>7.2f}{s['itmax']:>7d}"
              f"{s['ms']:>8.2f}{s['msmax']:>8.1f}{s['pi']:>7.2f}  {nst}{ex}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--dir", default="results_demo_int")
    ap.add_argument("--outdir", default="figs/demo")
    args = ap.parse_args()
    E = load(os.path.join(args.dir, args.layout))
    if not E:
        raise SystemExit("no runs")
    d = os.path.join(args.outdir, args.layout)
    os.makedirs(d, exist_ok=True)
    plot_paths(E, os.path.join(d, "paths.png"))
    plot_time(E, os.path.join(d, "time.png"))
    print(f"wrote {d}/paths.png, time.png\n")
    print_table(E)


if __name__ == "__main__":
    main()

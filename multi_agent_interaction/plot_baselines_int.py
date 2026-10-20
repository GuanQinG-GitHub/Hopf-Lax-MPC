"""
plot_baselines_int.py -- all methods on the interacting multi-agent test, for one K.

Reads the baseline runs (results_baselines_int.pkl, mpc_testbed_baselines_int.py) and the Hopf-Lax-MPC run
of the same K, N and pair penalty (results_multi_int.pkl, mpc_testbed_multi_int.py) and writes
  figs/baselines/K<K>/paths.png   one map panel per method: closed-loop paths, starts / goals, closest approach
  figs/baselines/K<K>/dmin.png    minimum pairwise distance vs time (all methods) + solve time per cycle
plus the printed comparison table.

USAGE
    python plot_baselines_int.py --K 2 [--outdir figs/baselines]
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d) for _d in ("multi_agent",)]
import plot_multi as PM                                                # noqa: E402  draw_scene, agent_colors
from plot_multi_int import xlims                                       # noqa: E402

ORDER = ["Hopf_Lax_MPC", "PMP_Shooting", "DDP", "Vanilla_Hopf_Lax", "MPPI_12288", "MPPI_20480"]
LBL = {"Hopf_Lax_MPC": "Hopf-Lax-MPC (ours)", "PMP_Shooting": "PMP-Shooting", "DDP": "DDP",
       "Vanilla_Hopf_Lax": "Vanilla-Hopf-Lax", "MPPI_12288": "MPPI (12288)", "MPPI_20480": "MPPI (20480)"}
COL = {"Hopf_Lax_MPC": (0.85, 0.10, 0.10), "PMP_Shooting": (0.10, 0.35, 0.80), "DDP": (0.95, 0.55, 0.05),
       "Vanilla_Hopf_Lax": (0.0, 0.60, 0.60), "MPPI_12288": (0.45, 0.25, 0.70), "MPPI_20480": (0.65, 0.45, 0.85)}


def load(K, N, col_key):
    runs = {}
    if os.path.exists("results_baselines_int.pkl"):
        for k, e in pickle.load(open("results_baselines_int.pkl", "rb"))["runs"].items():
            if k[1:] == (K, N) + col_key:
                runs[k[0]] = e
    if os.path.exists("results_multi_int.pkl"):
        e = pickle.load(open("results_multi_int.pkl", "rb"))["runs"].get((K, N) + col_key)
        if e is not None:
            runs["Hopf_Lax_MPC"] = dict(e, method="Hopf_Lax_MPC")
    return [runs[m] for m in ORDER if m in runs]


def stats(e):
    its, ts = e["iters"][1:], e["tsolve"][1:]
    pos = its > 0
    conv = 100 * np.mean([s in ("converged", "ok") for s in e["status"]])
    return dict(it=its.mean(), itmax=int(its.max()), conv=conv, ms=1e3 * ts.mean(), msmax=1e3 * ts.max(),
                pi=(1e3 * np.mean(ts[pos] / its[pos]) if pos.any() else np.nan))


def plot_paths(E, out):
    nm = len(E)
    ncol = 2
    nrow = -(-nm // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 3.3 * nrow), dpi=120, squeeze=False)
    for i, e in enumerate(E):
        ax = axs[i // ncol, i % ncol]
        K = e["K"]; cols = PM.agent_colors(K); Rc = e["col"]["Rc"]
        PM.draw_scene(ax, e, labels=False)
        kc = int(np.argmin(e["dmin"]))
        for k in range(K):
            X = e["X"][k]
            ax.plot(X[0], X[1], "-", color=cols[k], lw=1.5)
            ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=5)
            ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=10)
            q = X[:2, min(kc, X.shape[1] - 1)]
            ax.add_patch(plt.Circle(q, Rc / 2, fill=False, color=cols[k], lw=1.0))
        s = stats(e)
        arr = "all arrive" if e["all_reached"] else f"{int(np.sum(~np.isnan(e['t_reach'])))}/{K} arrive"
        ax.set_title(f"{LBL[e['method']]}: {arr}"
                     f"{' (last %.2f s)' % np.nanmax(e['t_reach']) if e['all_reached'] else ''}, min dist "
                     f"{e['dmin'].min():.2f} m, J {e['Jtrue'].sum():.2f} + pair {e['Jpair_total']:.2f}, "
                     f"{s['it']:.1f} it/cyc", fontsize=8.5, loc="left", color=COL[e["method"]])
        ax.set_xlim(*xlims(e)); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.25)
        ax.tick_params(labelsize=7)
    for j in range(nm, nrow * ncol):
        axs[j // ncol, j % ncol].axis("off")
    fig.suptitle(f"K={E[0]['K']} interacting agents, N={E[0]['N']}, no budget, pair penalty Rc={E[0]['col']['Rc']:g} m, "
                 f"Wc={E[0]['col']['Wc']:g}: closed-loop paths (circles of radius Rc/2 at each run's closest approach)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_dmin(E, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), dpi=120)
    for e in E:
        dta = e["meta"]["dt_apply"]
        t = np.arange(1, e["dmin"].size + 1) * dta
        ax[0].plot(t, e["dmin"], "-", color=COL[e["method"]], lw=1.3,
                   label=f"{LBL[e['method']]}: min {e['dmin'].min():.2f} m")
        ax[1].semilogy(e["tlog"], np.maximum(1e3 * e["tsolve"], 1e-3), "-", color=COL[e["method"]], lw=0.9,
                       label=f"{LBL[e['method']]}: mean {stats(e)['ms']:.1f} ms")
    ax[0].axhline(E[0]["col"]["Rc"], color="k", ls=":", lw=0.9, label=f"Rc = {E[0]['col']['Rc']:g} m")
    ax[0].set_title("minimum pairwise distance [m] (every fine step)", fontsize=9, loc="left")
    ax[1].set_title("solve time per cycle [ms] (no budget)", fontsize=9, loc="left")
    for a in ax:
        a.grid(alpha=.25); a.set_xlabel("t [s]"); a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def print_table(E):
    print(f"{'method':<20}{'cyc':>6}{'reached':>8}{'t_last':>7}{'sumJ':>9}{'Jpair':>9}{'dmin':>7}{'it mean':>8}{'it max':>7}"
          f"{'%conv':>7}{'solve ms':>9}{'max ms':>9}{'ms/iter':>8}")
    print("-" * 114)
    for e in E:
        s = stats(e)
        tl = np.nanmax(e["t_reach"]) if e["all_reached"] else np.nan
        reached = "all" if e["all_reached"] else f"{int(np.sum(~np.isnan(e['t_reach'])))}/{e['K']}"
        print(f"{LBL[e['method']]:<20}{e['ncyc']:>6d}{reached:>8}{tl:>7.2f}{e['Jtrue'].sum():>9.3f}{e['Jpair_total']:>9.3f}"
              f"{e['dmin'].min():>7.3f}{s['it']:>8.2f}{s['itmax']:>7d}{s['conv']:>7.1f}{s['ms']:>9.2f}{s['msmax']:>9.1f}"
              f"{s['pi']:>8.2f}{' ABORTED' if e.get('aborted') else ''}")
    print("sumJ = tracking + obstacle cost summed over agents (each up to its own arrival); Jpair = true pair cost;\n"
          "%conv: converged cycles (MPPI: one sampling update per cycle, 'ok'); timing: solver call only, cycle 1 excluded")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--N", type=int, default=220)
    ap.add_argument("--Rc", type=float, default=0.8)
    ap.add_argument("--Wc", type=float, default=500.0)
    ap.add_argument("--wc", type=float, default=0.1)
    ap.add_argument("--outdir", default="figs/baselines")
    args = ap.parse_args()
    E = load(args.K, args.N, (args.Rc, args.Wc, args.wc))
    if not E:
        raise SystemExit("no runs found")
    d = os.path.join(args.outdir, f"K{args.K}")
    os.makedirs(d, exist_ok=True)
    plot_paths(E, os.path.join(d, "paths.png"))
    plot_dmin(E, os.path.join(d, "dmin.png"))
    print(f"wrote {d}/paths.png, {d}/dmin.png\n")
    print_table(E)


if __name__ == "__main__":
    main()

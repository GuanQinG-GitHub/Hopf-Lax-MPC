"""
plot_sweep_int.py -- the all-method K-sweep on the interacting multi-agent problem (mpc_sweep_int.py).

  figs/sweep/sweep_time_cost.png (+ .pdf)
      left axis  (solid, filled circles): average computation time per cycle [ms], log scale -- the mean
                  solver-call time over all cycles of the run except the first (cold start);
      right axis (dashed, squares):      final trajectory cost J_full -- tracking + control + obstacle cost of
                  every agent plus the pair collision cost, integrated along the whole closed-loop trajectory;
      (log scale), both against the number of agents K.  Hollow markers: not every agent arrived (30 s cycle
      cap, 30 min wall-time limit, or aborted) -- the time average covers the cycles that ran, the cost the
      trajectory that was flown (a wall-limited run flew only part of the mission: its cost is not comparable).
  results_sweep_int_table.txt  (printed): one row per (method, K)

USAGE
    python plot_sweep_int.py [--dir results_sweep_int] [--outdir figs/sweep]
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
from matplotlib.lines import Line2D                                    # noqa: E402

from plot_baselines_int import ORDER, LBL, COL                         # method order, labels, colours


def load(d):
    R = {}
    for f in glob.glob(os.path.join(d, "*_K*.pkl")):
        e = pickle.load(open(f, "rb"))
        R[(e["method"], e["K"])] = e
    return R


def tmean(e):
    ts = e["tsolve"][1:]
    return 1e3 * ts.mean() if ts.size else np.nan


def plot(R, out):
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=140)
    ax2 = ax.twinx()
    for m in ORDER:
        Ks = sorted(K for (mm, K) in R if mm == m)
        if not Ks:
            continue
        T = np.array([tmean(R[(m, K)]) for K in Ks])
        J = np.array([R[(m, K)]["J_full"] for K in Ks])
        done = np.array([R[(m, K)]["complete"] for K in Ks])
        c = COL[m]
        ax.plot(Ks, T, "-", color=c, lw=1.8)
        ax2.plot(Ks, J, "--", color=c, lw=1.3)
        for K, tv, jv, dn in zip(Ks, T, J, done):
            ax.plot(K, tv, "o", ms=7, color=c, mfc=c if dn else "white", mew=1.5)
            ax2.plot(K, jv, "s", ms=6, color=c, mfc=c if dn else "white", mew=1.3)
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("number of agents K  (state dimension 7K)")
    ax.set_ylabel("average computation time per cycle [ms]  (solid, circles; log scale)")
    ax2.set_ylabel("final trajectory cost J  (dashed, squares; log scale)")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"{k}\n({7 * k})" for k in range(1, 11)])
    ax.grid(alpha=.25, which="both")
    h = [Line2D([], [], color=COL[m], lw=2, label=LBL[m]) for m in ORDER if any(mm == m for (mm, _) in R)]
    h += [Line2D([], [], color="k", ls="-", marker="o", label="computation time (left)"),
          Line2D([], [], color="k", ls="--", marker="s", label="trajectory cost (right)"),
          Line2D([], [], color="k", ls="none", marker="o", mfc="white", label="not all agents arrived (30 s cap or 30 min limit)")]
    ax.legend(handles=h, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5)
    ax.set_title("Interacting multi-agent test: all methods, N = 200, no budget, pair penalty Rc = 0.8 m; "
                 "one (method, K) run at a time", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)


def print_table(R):
    print(f"{'method':<20}{'K':>3}{'end':>13}{'arrived':>8}{'t_last':>7}{'cyc':>6}{'ms/cycle':>10}{'max ms':>10}"
          f"{'it mean':>8}{'%conv':>7}{'J_full':>12}{'track':>12}{'obs':>11}{'pair':>10}{'dmin':>7}{'wall s':>8}")
    print("-" * 152)
    for m in ORDER:
        for K in sorted(K for (mm, K) in R if mm == m):
            e = R[(m, K)]
            ts = e["tsolve"][1:]
            conv = 100 * np.mean([s in ("converged", "ok") for s in e["status"]])
            tl = np.nanmax(e["t_reach"]) if e["all_reached"] else np.nan
            print(f"{LBL[m]:<20}{K:>3}{e['end']:>13}{int(np.sum(~np.isnan(e['t_reach']))):>5d}/{K:<2d}{tl:>7.2f}{e['ncyc']:>6d}"
                  f"{tmean(e):>10.2f}{(1e3 * ts.max() if ts.size else np.nan):>10.1f}{e['iters'].mean():>8.2f}{conv:>7.1f}"
                  f"{e['J_full']:>12.3f}{e['Jreal'].sum():>12.3f}{e['Jpen'].sum():>11.3f}{e['Jpair_total']:>10.3f}"
                  f"{(e['dmin'].min() if e['dmin'].size else np.inf):>7.3f}{e['wall_s']:>8.0f}")
    print("ms/cycle: mean solver-call time per cycle (cycle 1 excluded); J_full: whole closed-loop trajectory, all terms;\n"
          "end: all-arrived | wall-limit (30 min) | maxcyc (30 s) | aborted")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results_sweep_int")
    ap.add_argument("--outdir", default="figs/sweep")
    args = ap.parse_args()
    R = load(args.dir)
    os.makedirs(args.outdir, exist_ok=True)
    plot(R, os.path.join(args.outdir, "sweep_time_cost"))
    print(f"wrote {args.outdir}/sweep_time_cost.png/.pdf ({len(R)} runs)\n")
    print_table(R)


if __name__ == "__main__":
    main()

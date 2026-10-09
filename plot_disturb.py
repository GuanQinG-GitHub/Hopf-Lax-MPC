"""
plot_disturb.py -- task cost of budgeted Hopf-Lax-MPC versus the disturbance level b.

Reads the pickle written by mpc_testbed_disturb.py and, for one (scale, post_y) scene, plots per
disturbance knob b (median over seeds, inter-quartile band, individual seeds as dots):
  (a) J_total,  (b) J_track and J_obs,  (c) time to reach the goal and fraction of runs that reached,
  (d) saddle kicks per run and mean cycle time.
Prints a per-b statistics table.

USAGE
    python plot_disturb.py [--pkl results_disturb_posty-0.1.pkl] [--scale practical] [--post-y -0.1]
                           [--out figs/disturb_sweep.png]
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

GREEN = (.10, .55, .20)
GREY = (.35, .35, .35)


def band(ax, b, vals, color, label=None, marker="o", lw=2.2):
    med = np.array([np.median(v) for v in vals])
    q1 = np.array([np.percentile(v, 25) for v in vals])
    q3 = np.array([np.percentile(v, 75) for v in vals])
    ax.fill_between(b, q1, q3, color=color, alpha=0.18, lw=0)
    for bi, v in zip(b, vals):
        ax.plot(np.full(len(v), bi), v, ".", color=color, alpha=0.45, ms=5)
    ax.plot(b, med, "-" + marker, color=color, lw=lw, ms=6, label=label)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results_disturb_posty-0.1.pkl")
    ap.add_argument("--scale", default="practical")
    ap.add_argument("--post-y", type=float, default=-0.1)
    ap.add_argument("--out", default="figs/disturb_sweep.png")
    ap.add_argument("--bs", default=None, help="comma-separated b levels to plot (default: all in the pickle)")
    ap.add_argument("--exclude-seeds", default="", help="comma-separated seeds to drop (scenario-dependent failures)")
    ap.add_argument("--seeds", default="", help="comma-separated seeds to KEEP (overrides --exclude-seeds)")
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        runs = pickle.load(f)["runs"]
    excl = {int(v) for v in args.exclude_seeds.split(",") if v.strip()}
    keep = {int(v) for v in args.seeds.split(",") if v.strip()}
    sel = {k: e for k, e in runs.items()
           if len(k) == 4 and k[0] == args.scale and abs(k[1] - args.post_y) < 1e-9
           and (k[3] in keep if keep else k[3] not in excl)}
    bs = sorted({k[2] for k in sel})
    if args.bs:
        want = [float(v) for v in args.bs.split(",")]
        bs = [b for b in bs if any(abs(b - w) < 1e-9 for w in want)]
    by_b = {b: [sel[k] for k in sorted(sel) if k[2] == b] for b in bs}
    nseed = {b: len(v) for b, v in by_b.items()}
    e0 = sel[sorted(sel)[0]]
    s = e0["scale"]

    # ---- table ----
    print(f"scene: scale={args.scale} s={np.round(s, 3)}, fork post y={args.post_y:g}, N={e0['N']}, "
          f"budget {1e3 * e0['meta']['budget']:.0f} ms;  per-axis bound = b * s, sigma = {e0['sigma_ratio']:g} * bound")
    if keep:
        print(f"seeds used: {sorted(keep)}")
    elif excl:
        print(f"excluded seeds: {sorted(excl)}")
    print(f"{'b':>4} {'n':>3} {'reached':>8} {'t_reach med':>11} {'J_total med':>11} {'[q1, q3]':>16} {'max':>8} "
          f"{'J_track med':>11} {'J_obs med':>10} {'J_obs max':>10} {'clips':>6} {'kicks med':>9} {'kicks max':>9} "
          f"{'ms mean':>8} {'p99 ms':>7} {'it/cyc':>7} {'x_end (not reached)':>20}")
    print("-" * 182)
    for b in bs:
        E = by_b[b]
        J = np.array([e["Jtrue"] for e in E]); Jt = np.array([e["Jreal"] for e in E]); Jo = np.array([e["Jpen"] for e in E])
        rc = np.array([e["reached"] for e in E], dtype=bool)
        tr = np.array([e["t_reach"] for e in E if e["reached"]])
        kk = np.array([int(np.sum(e["nkick"])) for e in E])
        ms = np.array([e["ms_mean"] for e in E]); p99 = np.array([e["ms_p99"] for e in E])
        it = np.array([np.mean(e["iters"]) for e in E])
        xend = [e["X"][0, -1] for e in E if not e["reached"]]
        xtxt = ("" if not xend else " ".join(f"{v:.1f}" for v in sorted(xend)))
        print(f"{b:>4g} {len(E):>3} {rc.mean() * 100:>7.0f}% {(np.median(tr) if tr.size else np.nan):>11.2f} "
              f"{np.median(J):>11.3f} [{np.percentile(J, 25):>6.2f}, {np.percentile(J, 75):>6.2f}] {J.max():>8.2f} "
              f"{np.median(Jt):>11.3f} {np.median(Jo):>10.3f} {Jo.max():>10.2f} {int(np.sum(Jo > 5)):>6} "
              f"{np.median(kk):>9.0f} {kk.max():>9} {ms.mean():>8.2f} {p99.mean():>7.1f} {it.mean():>7.2f} {xtxt:>20}")
    print("clips = runs with J_obs > 5 (an obstacle shell was entered); band in the figure = inter-quartile range")

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.8), dpi=140,
                            gridspec_kw=dict(wspace=0.26, hspace=0.34, left=0.07, right=0.98, bottom=0.08, top=0.9))
    (a, bx), (c, d) = axs
    JT = [np.array([e["Jtrue"] for e in by_b[b]]) for b in bs]
    band(a, bs, JT, GREEN, "J_total (median, IQR, seeds)")
    a.set_ylabel("closed-loop total cost J_total  (log)"); a.set_title("(a) task cost vs disturbance level", loc="left", fontsize=10)
    a.set_yscale("log")
    band(bx, bs, [np.array([e["Jreal"] for e in by_b[b]]) for b in bs], (0, .45, .85), "J_track")
    band(bx, bs, [np.array([e["Jpen"] for e in by_b[b]]) for b in bs], (.85, .15, .15), "J_obs", marker="s")
    bx.set_ylabel("cost"); bx.set_title("(b) tracking and obstacle cost", loc="left", fontsize=10); bx.set_yscale("log")
    TR = [np.array([e["t_reach"] for e in by_b[b] if e["reached"]]) for b in bs]
    TRv = [t if t.size else np.array([np.nan]) for t in TR]
    band(c, bs, TRv, GREEN, "t_reach (reached runs)")
    c2 = c.twinx()
    frac = [100 * np.mean([e["reached"] for e in by_b[b]]) for b in bs]
    c2.plot(bs, frac, "--^", color=GREY, lw=1.4, ms=5, label="reached [%]")
    c2.set_ylim(-2, 102); c2.set_ylabel("runs that reached the goal [%]", color=GREY)
    c.set_ylabel("time to reach the goal [s]"); c.set_title("(c) completion", loc="left", fontsize=10)
    h1, l1 = c.get_legend_handles_labels(); h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    band(d, bs, [np.array([float(np.sum(e["nkick"])) for e in by_b[b]]) for b in bs], (.60, .20, .70), "saddle kicks per run")
    d2 = d.twinx()
    band(d2, bs, [np.array([e["ms_mean"] for e in by_b[b]]) for b in bs], GREY, "mean cycle time [ms]", marker="s", lw=1.4)
    d2.axhline(20, color=GREY, ls=":", lw=1); d2.set_ylabel("mean cycle time [ms]", color=GREY); d2.set_ylim(bottom=0)
    d.set_ylabel("kicks per run"); d.set_title("(d) solver activity and compute", loc="left", fontsize=10); d.set_ylim(bottom=0)
    h1, l1 = d.get_legend_handles_labels(); h2, l2 = d2.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    for ax in (a, bx, c, d):
        ax.set_xlabel("disturbance level b  (per-axis bound = b x [%s] on [px py pz th v w vz] rates)"
                      % " ".join(f"{v:g}" for v in s), fontsize=8)
        ax.grid(alpha=0.25); ax.set_xticks(bs)
    for ax in (a, bx):
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Hopf-Lax-MPC (N={e0['N']}, 20 ms budget) under bounded Gaussian process disturbance -- "
                 f"{max(nseed.values())} seeds per level, fork post y={args.post_y:g}"
                 + (f", seeds {sorted(keep)}" if keep else (f", seeds {sorted(excl)} excluded (scenario-dependent wall side flip)" if excl else "")), fontsize=11)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

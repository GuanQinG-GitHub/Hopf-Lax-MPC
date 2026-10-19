"""
plot_N_sweep.py -- renders the no-budget closed-loop horizon sweep from results_N_sweep.pkl
(written by mpc_testbed_N_sweep.py) in the style of plot_complexity_maxiter.py.

Panels (2 x 2):
  (a) closed-loop total cost J_total vs N        filled marker = reached the goal, hollow = TRAPPED
  (b) per-cycle computation time vs N            mean (solid) and max (dashed); 20 ms line = the
                                                 real-time budget of the published closed loop
  (c) solver iterations per cycle vs N           mean (solid) and max (dashed); MPPI omitted (1 update
                                                 per cycle by protocol); dotted line = iteration cap
  (d) cycles not converged [%] vs N              status not in {converged, acceptable}; MPPI omitted
Second figure (per-iteration time):
  per-iteration solve time vs N  = (solver time of a cycle) / (iterations of that cycle), averaged
  over the cycles of a run (solid) and its max over cycles (dashed).  Cycles with 0 iterations (warm
  start already converged) are excluded.  MPPI: 1 update per cycle, so this equals its cycle time.
  Dotted: the open-loop STRUCTURAL worst-case per-iteration benchmark of the complexity study
  (results_complexity.pkl, D["iterbench"]) for Hopf-Lax-MPC / PMP / DDP, as a cross-check.
Timing/iteration statistics exclude cycle 1 (cold start) and recompile-tagged cycles, as the
testbed summary does.  A full per-(method, N) table is printed.

USAGE
    python plot_N_sweep.py [--pkl results_N_sweep.pkl] [--out figs/N_sweep_nobudget.png]
                           [--out2 figs/N_sweep_periter.png] [--complexity results_complexity.pkl]
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

COL = {"m1v2": (.10, .55, .20), "pmp": (.85, .15, .15), "newton": (0.0, .60, .60), "ddp": (.90, .45, .0),
       "coll": (0, .45, .85), "mppi12288": (.60, .20, .70), "mppi20480": (.25, .05, .35)}
LBL = {"m1v2": "Hopf-Lax-MPC", "pmp": "PMP", "newton": "Newton (exact Hessian)", "ddp": "DDP",
       "coll": "Collocation", "mppi12288": "MPPI-(K=12288)", "mppi20480": "MPPI-(K=20480)"}
ORDER = ["m1v2", "pmp", "newton", "ddp", "coll", "mppi12288", "mppi20480"]
MK = {"newton": "s"}                                                   # marker overrides (default o / ^ for MPPI)
CONV_OK = ("converged", "acceptable", "ok")


def keep_mask(e):
    tt = np.asarray(e["tcyc"])
    keep = np.ones(tt.size, dtype=bool)
    keep[0] = False
    rc = np.asarray(e["recompile"], dtype=bool)
    nrc = 0
    if rc.size == tt.size:
        nrc = int(np.sum(rc[1:]))
        keep &= ~rc
    if not keep.any():
        keep[:] = True
    return keep, nrc


def stats(e):
    keep, nrc = keep_mask(e)
    tt = 1e3 * np.asarray(e["tcyc"])[keep]
    ts = 1e3 * np.asarray(e["tsolve"])[keep]
    its = np.asarray(e["iters"])[keep]
    st = [s for s, k in zip(e["status"], keep) if k]
    n = max(len(st), 1)
    pos = its > 0                                                      # per-iteration time needs >= 1 iteration
    pi = ts[pos] / its[pos] if pos.any() else np.array([np.nan])
    return dict(ms_mean=tt.mean(), ms_max=tt.max(), ms_p99=np.percentile(tt, 99),
                solve_mean=ts.mean(), solve_max=ts.max(),
                pi_mean=float(np.mean(pi)), pi_max=float(np.max(pi)), pi_med=float(np.median(pi)),
                pi_std=float(np.std(pi)),
                it_mean=float(its.mean()), it_max=int(its.max()),
                pconv=100 * sum(s in CONV_OK for s in st) / n,
                pcap=100 * sum(s == "cap" for s in st) / n,
                pstall=100 * sum(s in ("stalled", "esc-stalled") for s in st) / n,
                pnonconv=100 * sum(s not in CONV_OK for s in st) / n, nrc=nrc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results_N_sweep.pkl")
    ap.add_argument("--out", default="figs/N_sweep_nobudget.png")
    ap.add_argument("--out2", default="figs/N_sweep_periter.png", help="per-iteration time figure")
    ap.add_argument("--out3", default="figs/N_sweep_periter_mean.png",
                    help="per-iteration time figure, mean only (linear axis)")
    ap.add_argument("--out4", default="figs/N_sweep_periter_max.png",
                    help="per-iteration time figure, max only (linear axis)")
    ap.add_argument("--complexity", default="results_complexity.pkl",
                    help="complexity-study pickle whose iterbench is overlaid (skipped if missing)")
    ap.add_argument("--out5", default="figs/N_sweep_periter_mean_band.png",
                    help="per-iteration time: mean line with a +-3 sigma band over cycles")
    ap.add_argument("--Nmax", type=int, default=None, help="only plot horizons N <= Nmax")
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        D = pickle.load(f)
    P = D["P"]
    runs = D["runs"]
    if args.Nmax is not None:
        runs = {k: v for k, v in runs.items() if k[1] <= args.Nmax}
    cap = D.get("config", {}).get("cap", P.lm_maxit)
    methods = [m for m in ORDER if any(k[0] == m for k in runs)]
    methods += sorted({k[0] for k in runs} - set(methods))             # any unexpected method last

    # ---- table ----
    print(f"{'method':<24}{'N':>5}{'Tp[s]':>6}{'mean ms':>9}{'max ms':>9}{'p99 ms':>9}{'solve mean':>11}"
          f"{'solve max':>10}{'it mean':>8}{'it max':>7}{'ms/it mean':>11}{'ms/it max':>10}"
          f"{'%conv':>7}{'%cap':>6}{'%stall':>7}"
          f"{'J_track':>9}{'J_obs':>9}{'J_total':>9}{'reached':>9}{'t_reach':>8}{'cyc':>5}{'rcmp':>5}")
    print("-" * 199)
    for m in methods:
        for N in sorted(k[1] for k in runs if k[0] == m):
            e = runs[(m, N)]
            s = stats(e)
            tr = f"{e['t_reach']:.2f}" if e["reached"] else "--"
            flag = " ABORTED" if e.get("aborted") else ""
            print(f"{LBL.get(m, m):<24}{N:>5d}{N * P.dt:>6.2f}{s['ms_mean']:>9.2f}{s['ms_max']:>9.1f}"
                  f"{s['ms_p99']:>9.1f}{s['solve_mean']:>11.2f}{s['solve_max']:>10.1f}{s['it_mean']:>8.1f}"
                  f"{s['it_max']:>7d}{s['pi_mean']:>11.3f}{s['pi_max']:>10.2f}"
                  f"{s['pconv']:>7.1f}{s['pcap']:>6.1f}{s['pstall']:>7.1f}{e['Jreal']:>9.3f}"
                  f"{e['Jpen']:>9.3f}{e['Jtrue']:>9.3f}{('yes' if e['reached'] else 'TRAPPED'):>9}{tr:>8}"
                  f"{e['ncyc']:>5d}{s['nrc']:>5d}{flag}")
    print("timing: whole-cycle ms; cycle 1 and recompile-tagged cycles excluded; 'solve' = solver call only;"
          " 'ms/it' = solve time / iterations per cycle (cycles with 0 iterations excluded)")

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.6), dpi=140,
                            gridspec_kw=dict(wspace=0.24, hspace=0.32, left=0.06, right=0.985,
                                             bottom=0.07, top=0.91))
    (ax_j, ax_t), (ax_i, ax_c) = axs
    for m in methods:
        Ns = sorted(k[1] for k in runs if k[0] == m)
        es = [runs[(m, N)] for N in Ns]
        ss = [stats(e) for e in es]
        col, lbl = COL.get(m, (.3, .3, .3)), LBL.get(m, m)
        mk = MK.get(m, "^" if m.startswith("mppi") else "o")
        lw, ms = (3.0, 8) if m == "m1v2" else (1.6, 5)
        # (a) J_total, reached vs trapped marker fill
        J = np.array([e["Jtrue"] for e in es])
        reached = np.array([bool(e["reached"]) for e in es])
        ax_j.plot(Ns, J, "-", color=col, lw=lw, label=lbl)
        ax_j.plot(np.array(Ns)[reached], J[reached], mk, color=col, ms=ms)
        ax_j.plot(np.array(Ns)[~reached], J[~reached], mk, color=col, ms=ms, mfc="white", mew=1.8)
        # (b) per-cycle time
        ax_t.plot(Ns, [s["ms_mean"] for s in ss], "-" + mk, color=col, lw=lw, ms=ms, label=lbl)
        ax_t.plot(Ns, [s["ms_max"] for s in ss], "--" + mk, color=col, lw=0.6 * lw, ms=0.7 * ms, alpha=0.75)
        if not m.startswith("mppi"):
            # (c) iterations per cycle
            ax_i.plot(Ns, [s["it_mean"] for s in ss], "-" + mk, color=col, lw=lw, ms=ms, label=lbl)
            ax_i.plot(Ns, [s["it_max"] for s in ss], "--" + mk, color=col, lw=0.6 * lw, ms=0.7 * ms, alpha=0.75)
            # (d) non-converged fraction
            ax_c.plot(Ns, [s["pnonconv"] for s in ss], "-" + mk, color=col, lw=lw, ms=ms, label=lbl)

    ax_j.set_yscale("log")
    ax_j.set_title("closed-loop total cost  J_total  (filled = reached, hollow = TRAPPED)", loc="left", fontsize=10)
    ax_j.set_xlabel("horizon N"); ax_j.set_ylabel("J_total = J_track + J_obs")
    h, l = ax_j.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color="0.3", ls="", ms=6, label="reached"),
          Line2D([], [], marker="o", color="0.3", ls="", ms=6, mfc="white", mew=1.8, label="TRAPPED")]
    ax_j.legend(handles=h, frameon=False, fontsize=8)

    ax_t.set_yscale("log")
    ax_t.axhline(20.0, color="0.35", ls=":", lw=1.0, alpha=0.8)
    ax_t.text(ax_t.get_xlim()[0], 20.0, " 20 ms budget of the published closed loop", fontsize=7,
              color="0.35", va="bottom")
    ax_t.set_title("per-cycle computation time  (solid = mean, dashed = max)", loc="left", fontsize=10)
    ax_t.set_xlabel("horizon N"); ax_t.set_ylabel("cycle time [ms]")
    ax_t.legend(frameon=False, fontsize=8)

    ax_i.set_yscale("log")
    ax_i.axhline(cap, color="0.35", ls=":", lw=1.0, alpha=0.8)
    ax_i.text(ax_i.get_xlim()[0], cap, f" iteration cap {cap}", fontsize=7, color="0.35", va="top")
    ax_i.set_title("solver iterations per cycle  (solid = mean, dashed = max; MPPI: 1 update by protocol)",
                   loc="left", fontsize=10)
    ax_i.set_xlabel("horizon N"); ax_i.set_ylabel("iterations / cycle")
    ax_i.legend(frameon=False, fontsize=8)

    ax_c.set_ylim(-2, 102)
    ax_c.set_title("cycles not converged  (cap / stalled / diverged / timeout)", loc="left", fontsize=10)
    ax_c.set_xlabel("horizon N"); ax_c.set_ylabel("non-converged cycles [%]")
    ax_c.legend(frameon=False, fontsize=8)

    for ax in axs.ravel():
        ax.grid(alpha=0.25)
    budget = D.get("config", {}).get("cycle_timeout", np.inf)
    fig.suptitle(f"Closed loop WITHOUT real-time budget vs horizon -- "
                 f"{'no per-cycle cut' if not np.isfinite(budget) else f'{budget:g} s safety cut'}, "
                 f"iteration cap {cap}, {P.maxcyc} cycles / stop at goal", fontsize=12)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")

    # ---- figure 2: per-iteration solve time ----
    bench = {}
    try:
        with open(args.complexity, "rb") as f:
            bench = pickle.load(f).get("iterbench", {})
    except (OSError, pickle.UnpicklingError):
        pass
    fig2, ax = plt.subplots(figsize=(7.2, 5.4), dpi=140)
    for m in methods:
        Ns = sorted(k[1] for k in runs if k[0] == m)
        ss = [stats(runs[(m, N)]) for N in Ns]
        col, lbl = COL.get(m, (.3, .3, .3)), LBL.get(m, m)
        mk = MK.get(m, "^" if m.startswith("mppi") else "o")
        lw, ms = (3.0, 8) if m == "m1v2" else (1.6, 5)
        ax.plot(Ns, [s["pi_mean"] for s in ss], "-" + mk, color=col, lw=lw, ms=ms, label=lbl)
        ax.plot(Ns, [s["pi_max"] for s in ss], "--" + mk, color=col, lw=0.6 * lw, ms=0.7 * ms, alpha=0.75)
        bn = sorted(k[1] for k in bench if k[0] == m)
        if bn:
            ax.plot(bn, [bench[(m, N)] for N in bn], ":", color=col, lw=1.2, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("horizon N"); ax.set_ylabel("solver time per iteration [ms]")
    ax.set_title("per-iteration solve time, closed loop without budget\n"
                 "(solid = mean over cycles, dashed = max over cycles"
                 + (", dotted = open-loop structural worst-case benchmark)" if bench else ")"),
                 loc="left", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig2.savefig(args.out2, bbox_inches="tight")
    print(f"saved {args.out2}")

    # ---- figures 3 / 4: per-iteration solve time, MEAN only and MAX only (linear axis, no overlay) ----
    for key, ylab, word, out in [("pi_mean", "Average per-iteration computation time [ms]", "mean", args.out3),
                                 ("pi_max", "max solver time per iteration [ms]", "max", args.out4)]:
        figk, ax = plt.subplots(figsize=(7.2, 5.4), dpi=140)
        for m in methods:
            Ns = sorted(k[1] for k in runs if k[0] == m)
            ss = [stats(runs[(m, N)]) for N in Ns]
            col, lbl = COL.get(m, (.3, .3, .3)), LBL.get(m, m)
            mk = MK.get(m, "^" if m.startswith("mppi") else "o")
            lw, ms = (3.0, 8) if m == "m1v2" else (1.6, 5)
            ax.plot(Ns, [s[key] for s in ss], "-" + mk, color=col, lw=lw, ms=ms, label=lbl)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("horizon N"); ax.set_ylabel(ylab)
        ax.set_title(f"per-iteration solve time, closed loop without budget ({word} over cycles)",
                     loc="left", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        figk.savefig(out, bbox_inches="tight")
        print(f"saved {out}")

    # ---- figure 5: per-iteration solve time, mean line with a +-3 sigma band (sigma over the cycles) ----
    fig5, ax = plt.subplots(figsize=(7.2, 5.4), dpi=140)
    for m in methods:
        Ns = sorted(k[1] for k in runs if k[0] == m)
        ss = [stats(runs[(m, N)]) for N in Ns]
        col, lbl = COL.get(m, (.3, .3, .3)), LBL.get(m, m)
        mk = MK.get(m, "^" if m.startswith("mppi") else "o")
        lw, ms = (3.0, 8) if m == "m1v2" else (1.6, 5)
        mu = np.array([s["pi_mean"] for s in ss]); sd = np.array([s["pi_std"] for s in ss])
        ax.fill_between(Ns, np.maximum(mu - 3 * sd, 0.0), mu + 3 * sd, color=col, alpha=0.15, lw=0)
        ax.plot(Ns, mu, "-" + mk, color=col, lw=lw, ms=ms, label=lbl)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("horizon N"); ax.set_ylabel("Average per-iteration computation time [ms]")
    ax.set_title("per-iteration solve time, closed loop without budget\n"
                 "(line = mean over cycles, band = mean +- 3 sigma over cycles)", loc="left", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig5.savefig(args.out5, bbox_inches="tight")
    print(f"saved {args.out5}")


if __name__ == "__main__":
    main()

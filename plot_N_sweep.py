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
Timing/iteration statistics exclude cycle 1 (cold start) and recompile-tagged cycles, as the
testbed summary does.  A full per-(method, N) table is printed.

USAGE
    python plot_N_sweep.py [--pkl results_N_sweep.pkl] [--out figs/N_sweep_nobudget.png]
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

COL = {"m1v2": (.10, .55, .20), "pmp": (.85, .15, .15), "ddp": (.90, .45, .0),
       "coll": (0, .45, .85), "mppi12288": (.60, .20, .70), "mppi20480": (.25, .05, .35)}
LBL = {"m1v2": "Hopf-Lax-MPC", "pmp": "PMP", "ddp": "DDP", "coll": "Collocation",
       "mppi12288": "MPPI-(K=12288)", "mppi20480": "MPPI-(K=20480)"}
ORDER = ["m1v2", "pmp", "ddp", "coll", "mppi12288", "mppi20480"]
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
    return dict(ms_mean=tt.mean(), ms_max=tt.max(), ms_p99=np.percentile(tt, 99),
                solve_mean=ts.mean(), solve_max=ts.max(),
                it_mean=float(its.mean()), it_max=int(its.max()),
                pconv=100 * sum(s in CONV_OK for s in st) / n,
                pcap=100 * sum(s == "cap" for s in st) / n,
                pstall=100 * sum(s in ("stalled", "esc-stalled") for s in st) / n,
                pnonconv=100 * sum(s not in CONV_OK for s in st) / n, nrc=nrc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results_N_sweep.pkl")
    ap.add_argument("--out", default="figs/N_sweep_nobudget.png")
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        D = pickle.load(f)
    P = D["P"]
    runs = D["runs"]
    cap = D.get("config", {}).get("cap", P.lm_maxit)
    methods = [m for m in ORDER if any(k[0] == m for k in runs)]
    methods += sorted({k[0] for k in runs} - set(methods))             # any unexpected method last

    # ---- table ----
    print(f"{'method':<16}{'N':>5}{'Tp[s]':>6}{'mean ms':>9}{'max ms':>9}{'p99 ms':>9}{'solve mean':>11}"
          f"{'solve max':>10}{'it mean':>8}{'it max':>7}{'%conv':>7}{'%cap':>6}{'%stall':>7}"
          f"{'J_track':>9}{'J_obs':>9}{'J_total':>9}{'reached':>9}{'t_reach':>8}{'cyc':>5}{'rcmp':>5}")
    print("-" * 170)
    for m in methods:
        for N in sorted(k[1] for k in runs if k[0] == m):
            e = runs[(m, N)]
            s = stats(e)
            tr = f"{e['t_reach']:.2f}" if e["reached"] else "--"
            flag = " ABORTED" if e.get("aborted") else ""
            print(f"{LBL.get(m, m):<16}{N:>5d}{N * P.dt:>6.2f}{s['ms_mean']:>9.2f}{s['ms_max']:>9.1f}"
                  f"{s['ms_p99']:>9.1f}{s['solve_mean']:>11.2f}{s['solve_max']:>10.1f}{s['it_mean']:>8.1f}"
                  f"{s['it_max']:>7d}{s['pconv']:>7.1f}{s['pcap']:>6.1f}{s['pstall']:>7.1f}{e['Jreal']:>9.3f}"
                  f"{e['Jpen']:>9.3f}{e['Jtrue']:>9.3f}{('yes' if e['reached'] else 'TRAPPED'):>9}{tr:>8}"
                  f"{e['ncyc']:>5d}{s['nrc']:>5d}{flag}")
    print("timing: whole-cycle ms; cycle 1 and recompile-tagged cycles excluded; 'solve' = solver call only")

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
        mk = "^" if m.startswith("mppi") else "o"
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


if __name__ == "__main__":
    main()

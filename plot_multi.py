"""
plot_multi.py -- figures for the multi-agent (stacked, decoupled) Hopf-Lax-MPC scaling study.

  figs/multi_paths_K<K>.png   x-y paths of the K agents in the frozen scene (+ their RRT references)
  figs/multi_scaling.png      vs K: per-iteration solve time (mean / max), solve time per cycle,
                              iterations per cycle, non-converged %; dotted: ideal linear / cubic scaling
  stacked-vs-solo check       for every (K, solo) pair in the pickle: max |dU|, |dX| per agent

USAGE
    python plot_multi.py [--pkl results_multi.pkl] [--paths-K 2]
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Ellipse                                 # noqa: E402

GREEN = (.10, .55, .20)


def stats(e):
    its, ts, tt = e["iters"][1:], e["tsolve"][1:], e["tcyc"][1:]
    pos = its > 0
    pi = ts[pos] / its[pos]
    return dict(it_mean=its.mean(), it_max=int(its.max()), ms_mean=1e3 * ts.mean(), ms_max=1e3 * ts.max(),
                pi_mean=1e3 * pi.mean(), pi_med=1e3 * np.median(pi), pi_max=1e3 * pi.max(),
                pnonconv=100 * np.mean([s != "converged" for s in e["status"]]), cyc=1e3 * tt.mean())


def plot_paths(e, out):
    K = e["K"]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
    for i in range(e["obs"]["center"].shape[1]):
        ax.add_patch(Ellipse(e["obs"]["center"][:2, i], 2 * e["obs"]["ax"][i], 2 * e["obs"]["ay"][i], color=(.85, .18, .18), alpha=.25))
    for i in range(e["mobs"]["c0"].shape[1]):
        ax.add_patch(Ellipse(e["mobs"]["c0"][:2, i], 2 * e["mobs"]["ax"][i], 2 * e["mobs"]["ay"][i], color=(.85, .18, .18), alpha=.25))
        ax.text(e["mobs"]["c0"][0, i], e["mobs"]["c0"][1, i], str(i), fontsize=7, ha="center", va="center")
    cmap = plt.get_cmap("tab10")
    for k in range(K):
        p = e["paths"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cmap(k), lw=0.8, alpha=0.6)
        X = e["X"][k]
        tr = e["t_reach"][k]
        ax.plot(X[0], X[1], "-", color=cmap(k), lw=1.6,
                label=f"agent {k}: J={e['Jtrue'][k]:.2f}, {'reached %.2f s' % tr if np.isfinite(tr) else 'not reached'}")
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cmap(k), ms=7)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cmap(k), ms=12)
    ax.set_xlim(-1, 21); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_title(f"{K} decoupled agents, stacked Hopf-Lax-MPC, N={e['N']}, no budget -- frozen obstacles "
                 f"(o start, * goal, dashed = RRT reference)", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results_multi.pkl")
    ap.add_argument("--paths-K", type=int, default=None, help="draw the paths figure for this K (default: all K)")
    args = ap.parse_args()
    D = pickle.load(open(args.pkl, "rb"))
    runs = D["runs"]
    full = {k[0]: e for k, e in runs.items() if len(k) == 2}
    # ---- table ----
    print(f"{'run':<24}{'dim':>5}{'cyc':>5}{'reached':>8}{'sumJ':>9}{'it mean':>8}{'it max':>7}{'%nonconv':>9}"
          f"{'solve ms':>9}{'max ms':>8}{'ms/it mean':>11}{'ms/it med':>10}{'ms/it max':>10}{'cycle ms':>9}")
    print("-" * 132)
    for key in sorted(runs, key=lambda k: (k[0], len(k), k[-1] if len(k) > 2 else -1)):
        e = runs[key]; s = stats(e)
        print(f"{e['label']:<24}{7 * e['K']:>5d}{e['ncyc']:>5d}{('all' if e['all_reached'] else str(int(np.sum(np.isfinite(e['t_reach']))))):>8}"
              f"{e['Jtrue'].sum():>9.3f}{s['it_mean']:>8.2f}{s['it_max']:>7d}{s['pnonconv']:>9.1f}{s['ms_mean']:>9.2f}{s['ms_max']:>8.1f}"
              f"{s['pi_mean']:>11.2f}{s['pi_med']:>10.2f}{s['pi_max']:>10.2f}{s['cyc']:>9.2f}")
    # ---- stacked vs solo check ----
    for key, e in runs.items():
        if len(key) == 4 and key[0] in full:
            K, N, _, k = key; f = full[K]; ai = f["agents"].index(k) if k in f["agents"] else k
            n = min(e["U"].shape[2], f["U"].shape[2])
            dU = np.abs(e["U"][0, :, :n] - f["U"][ai, :, :n]).max()
            dX = np.abs(e["X"][0, :, :n + 1] - f["X"][ai, :, :n + 1]).max()
            print(f"stacked K={K} agent {k} vs solo: max|dU| {dU:.2e} max|dX| {dX:.2e} over {n} cycles | "
                  f"J {f['Jtrue'][ai]:.4f} vs {e['Jtrue'][0]:.4f} | t_reach {f['t_reach'][ai]:.2f} vs {e['t_reach'][0]:.2f} | "
                  f"iters/cycle {f['iters'].mean():.2f} vs {e['iters'].mean():.2f}")
    # ---- paths figures ----
    Ks = [args.paths_K] if args.paths_K else sorted(full)
    for K in Ks:
        if K in full:
            plot_paths(full[K], f"figs/multi_paths_K{K}.png")
    # ---- scaling figure ----
    if len(full) >= 2:
        Ks = sorted(full); ss = [stats(full[K]) for K in Ks]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), dpi=130)
        (a1, a2), (a3, a4) = axs
        a1.plot(Ks, [s["pi_mean"] for s in ss], "-o", color=GREEN, lw=2.4, ms=7, label="mean over cycles")
        a1.plot(Ks, [s["pi_max"] for s in ss], "--o", color=GREEN, lw=1.2, ms=5, alpha=.7, label="max over cycles")
        base = ss[0]["pi_mean"]
        a1.plot(Ks, [base * K / Ks[0] for K in Ks], ":", color="0.4", lw=1, label="linear in K")
        a1.plot(Ks, [base * (K / Ks[0]) ** 2 for K in Ks], ":", color="0.6", lw=1, label="quadratic in K")
        a1.plot(Ks, [base * (K / Ks[0]) ** 3 for K in Ks], ":", color="0.8", lw=1, label="cubic in K")
        a1.set_yscale("log"); a1.set_xscale("log"); a1.set_title("per-iteration solve time vs number of agents", loc="left", fontsize=10)
        a1.set_xlabel("K agents (stacked dimension 7K)"); a1.set_ylabel("ms per iteration"); a1.legend(frameon=False, fontsize=8)
        a2.plot(Ks, [s["ms_mean"] for s in ss], "-o", color=GREEN, lw=2.4, ms=7, label="mean")
        a2.plot(Ks, [s["ms_max"] for s in ss], "--o", color=GREEN, lw=1.2, ms=5, alpha=.7, label="max")
        a2.set_yscale("log"); a2.set_xscale("log"); a2.set_title("solve time per cycle", loc="left", fontsize=10)
        a2.set_xlabel("K agents"); a2.set_ylabel("ms per cycle"); a2.legend(frameon=False, fontsize=8)
        a3.plot(Ks, [s["it_mean"] for s in ss], "-o", color=GREEN, lw=2.4, ms=7, label="mean")
        a3.plot(Ks, [s["it_max"] for s in ss], "--o", color=GREEN, lw=1.2, ms=5, alpha=.7, label="max")
        a3.set_title("iterations per cycle", loc="left", fontsize=10); a3.set_xlabel("K agents"); a3.set_ylabel("iterations"); a3.legend(frameon=False, fontsize=8)
        a4.plot(Ks, [s["pnonconv"] for s in ss], "-o", color=GREEN, lw=2.4, ms=7)
        a4.set_ylim(-2, 102); a4.set_title("cycles not converged [%]", loc="left", fontsize=10); a4.set_xlabel("K agents")
        for a in axs.ravel():
            a.grid(alpha=.25, which="both")
        fig.suptitle(f"Hopf-Lax-MPC on K decoupled agents as one stacked problem -- N={full[Ks[0]]['N']}, no budget, frozen obstacles", fontsize=11)
        fig.tight_layout(); fig.savefig("figs/multi_scaling.png", bbox_inches="tight"); print("saved figs/multi_scaling.png")


if __name__ == "__main__":
    main()

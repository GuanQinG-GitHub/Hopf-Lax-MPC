"""
plot_multi.py -- figures for the multi-agent (stacked, decoupled) Hopf-Lax-MPC scaling study.

  figs/multi/K<K>/paths.png     x-y paths of the K agents in the frozen scene (+ RRT references)
  figs/multi/K<K>/review.png    paths with 1 s ticks + y(t), v(t), heading, control, goal distance, solver
  figs/multi/scaling.png        per-iteration average solve time vs K (x ticks labelled with the stacked
                                state dimension 7K), with linear / quadratic / cubic references, plus
                                solve time per cycle, iterations per cycle and non-converged %
  stacked-vs-solo check         for every (K, solo) pair in the pickle: max |dU|, |dX| per agent

USAGE
    python plot_multi.py [--pkl results_multi.pkl] [--outdir figs/multi] [--only-K 2]
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

GREEN = (.10, .55, .20)
RED = (.85, .18, .18)
MOBS_NAMES = ["wall", "blocker", "door", "shoulder", "fork post", "stage-2", "companion"]


def stats(e):
    its, ts, tt = e["iters"][1:], e["tsolve"][1:], e["tcyc"][1:]
    pos = its > 0
    pi = ts[pos] / its[pos]
    return dict(it_mean=its.mean(), it_max=int(its.max()), ms_mean=1e3 * ts.mean(), ms_max=1e3 * ts.max(),
                pi_mean=1e3 * pi.mean(), pi_med=1e3 * np.median(pi), pi_max=1e3 * pi.max(),
                pnonconv=100 * np.mean([s != "converged" for s in e["status"]]), cyc=1e3 * tt.mean())


def draw_scene(ax, e, labels=True):
    so, mo = e["obs"], e["mobs"]
    for i in range(so["center"].shape[1]):
        ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=RED, alpha=.12))
        ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=RED, alpha=.45))
        if labels:
            ax.text(so["center"][0, i], so["center"][1, i], "obs-1", fontsize=7, ha="center", va="center")
    for i in range(mo["c0"].shape[1]):
        c = mo["c0"][:2, i]
        if abs(c[1]) > 4:
            continue
        ax.add_patch(Ellipse(c, 2.5 * mo["ax"][i], 2.5 * mo["ay"][i], color=RED, alpha=.12))
        ax.add_patch(Ellipse(c, 2 * mo["ax"][i], 2 * mo["ay"][i], color=RED, alpha=.45))
        if labels:
            ax.text(c[0], c[1] + mo["ay"][i] + 0.12, MOBS_NAMES[i] if i < len(MOBS_NAMES) else str(i),
                    fontsize=7, ha="center", va="bottom", color=(.55, .1, .1))


def agent_colors(K):
    cmap = plt.get_cmap("tab10")
    return [cmap(k % 10) for k in range(K)]


def plot_paths(e, out):
    K = e["K"]; cols = agent_colors(K)
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=130)
    draw_scene(ax, e)
    for k in range(K):
        p = e["paths"][k]; X = e["X"][k]; tr = e["t_reach"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.8, alpha=0.55)
        ax.plot(X[0], X[1], "-", color=cols[k], lw=1.5,
                label=f"agent {k}: J={e['Jtrue'][k]:.2f}, {'%.2f s' % tr if np.isfinite(tr) else 'not reached'}")
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=6)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=11)
    ax.set_xlim(-0.8, 20.8); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"K={K} decoupled agents (state dim {7 * K}), stacked Hopf-Lax-MPC, N={e['N']}, no budget, "
                 f"obstacles frozen at t=0  (o start, * goal, dashed = RRT reference)", loc="left", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(K, 5))
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_review(e, out):
    K = e["K"]; cols = agent_colors(K); dt_apply = e["meta"]["dt_apply"]; tl = e["tlog"]
    fig = plt.figure(figsize=(14, 10.5), dpi=130)
    gs = fig.add_gridspec(3, 3, height_ratios=[2.6, 1, 1], hspace=0.35, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    draw_scene(ax, e)
    for k in range(K):
        p = e["paths"][k]; X = e["X"][k]; tr = e["t_reach"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.8, alpha=0.5)
        ax.plot(X[0], X[1], "-", color=cols[k], lw=1.8,
                label=f"agent {k} (J={e['Jtrue'][k]:.3f}, {'arrives %.2f s' % tr if np.isfinite(tr) else 'not reached'})")
        tt = np.arange(X.shape[1]) * dt_apply
        for s in range(1, int(tt[-1]) + 1):
            i = int(round(s / dt_apply))
            if i < X.shape[1]:
                ax.plot(X[0, i], X[1, i], "o", color=cols[k], ms=3, mfc="white", mew=1.0)
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=8)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=13)
        ax.add_patch(plt.Circle(e["goals"][k][:2], e["meta"]["reachtol"], fill=False, ls=":", color=cols[k], lw=0.8))
    ax.set_xlim(-0.8, 20.8); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"K={K} decoupled agents (state dim {7 * K}), stacked Hopf-Lax-MPC (N={e['N']}, no budget), obstacles frozen at t=0"
                 f" -- solid ellipse = barrier core, light = shell; white dots = 1 s ticks; dotted circle = reach tolerance",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(K, 5))
    a = [fig.add_subplot(gs[r, c]) for r in (1, 2) for c in (0, 1, 2)]
    for k in range(K):
        X = e["X"][k]; tt = np.arange(X.shape[1]) * dt_apply; U = e["U"][k]; tu = np.arange(U.shape[1]) * dt_apply
        a[0].plot(tt, X[1], color=cols[k], lw=1.2); a[1].plot(tt, X[4], color=cols[k], lw=1.2)
        a[2].plot(tt, np.degrees(X[3]), color=cols[k], lw=1.2); a[3].plot(tu, U[0], color=cols[k], lw=0.8)
        a[4].plot(tt, np.linalg.norm(X[:2] - e["goals"][k][:2, None], axis=0), color=cols[k], lw=1.2)
    a[0].set_title("lateral position y(t) [m]", fontsize=9, loc="left"); a[1].set_title("speed v(t) [m/s]", fontsize=9, loc="left")
    a[2].set_title("heading [deg]", fontsize=9, loc="left"); a[3].set_title("acceleration command u1 [m/s^2]", fontsize=9, loc="left")
    a[4].set_title("distance to own goal [m]", fontsize=9, loc="left"); a[4].set_yscale("log")
    a[5].plot(tl, e["iters"], ".", color=GREEN, ms=3)
    a5b = a[5].twinx(); a5b.plot(tl, 1e3 * e["tsolve"], "-", color="0.4", lw=0.8, alpha=0.8); a5b.set_ylabel("solve ms", color="0.4", fontsize=8)
    a[5].set_title("solver: iterations per cycle (dots), solve ms (grey)", fontsize=9, loc="left")
    for ax_ in a:
        ax_.grid(alpha=.25); ax_.set_xlabel("t [s]", fontsize=8); ax_.tick_params(labelsize=8)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_scaling(full, out):
    Ks = sorted(full); ss = [stats(full[K]) for K in Ks]
    dims = [7 * K for K in Ks]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9), dpi=130)
    (a1, a2), (a3, a4) = axs
    base = ss[0]["pi_mean"]
    a1.plot(Ks, [s["pi_mean"] for s in ss], "-o", color=GREEN, lw=2.6, ms=7, label="mean over cycles", zorder=5)
    a1.plot(Ks, [s["pi_max"] for s in ss], "--o", color=GREEN, lw=1.2, ms=4.5, alpha=.7, label="max over cycles")
    for p, lab, c in ((1, "linear in K", "0.35"), (2, "quadratic in K", "0.55"), (3, "cubic in K", "0.75")):
        a1.plot(Ks, [base * (K / Ks[0]) ** p for K in Ks], ":", color=c, lw=1.1, label=lab)
    a1.set_yscale("log"); a1.set_title("per-iteration average solve time", loc="left", fontsize=10)
    a1.set_ylabel("ms per iteration"); a1.legend(frameon=False, fontsize=8)
    a2.plot(Ks, [s["ms_mean"] for s in ss], "-o", color=GREEN, lw=2.4, ms=6, label="mean")
    a2.plot(Ks, [s["ms_max"] for s in ss], "--o", color=GREEN, lw=1.2, ms=4.5, alpha=.7, label="max")
    a2.set_yscale("log"); a2.set_title("solve time per cycle", loc="left", fontsize=10); a2.set_ylabel("ms per cycle"); a2.legend(frameon=False, fontsize=8)
    a3.plot(Ks, [s["it_mean"] for s in ss], "-o", color=GREEN, lw=2.4, ms=6, label="mean")
    a3.plot(Ks, [s["it_max"] for s in ss], "--o", color=GREEN, lw=1.2, ms=4.5, alpha=.7, label="max")
    a3.set_title("iterations per cycle", loc="left", fontsize=10); a3.set_ylabel("iterations"); a3.legend(frameon=False, fontsize=8)
    a4.plot(Ks, [s["pnonconv"] for s in ss], "-o", color=GREEN, lw=2.4, ms=6)
    a4.set_ylim(-2, 102); a4.set_title("cycles not converged [%]", loc="left", fontsize=10)
    for ax_ in axs.ravel():
        ax_.set_xticks(Ks); ax_.set_xticklabels([f"{K}\n{d}" for K, d in zip(Ks, dims)], fontsize=8)
        ax_.set_xlabel("agents K (top)  /  stacked state dimension 7K (bottom)", fontsize=9); ax_.grid(alpha=.25, which="both")
    # the fused Jacobian pass batches 7K x K = 7K^2 rollouts; the XLA CPU kernel changes regime near 200 (measured)
    thr = float(np.sqrt(200 / 7))
    a1.axvline(thr, color="0.5", ls="--", lw=0.9)
    a1.text(thr + 0.08, a1.get_ylim()[0] * 1.6, "7K^2 = 200 batched rollouts:\nfused-Jacobian kernel\nregime change (measured)",
            fontsize=7.5, color="0.35", va="bottom")
    e0 = full[Ks[0]]
    fig.suptitle(f"Hopf-Lax-MPC on K decoupled agents solved as ONE stacked problem -- N={e0['N']}, no budget, "
                 f"obstacles frozen; same algorithm, no block-structure exploitation", fontsize=11)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")  # vector copy for the paper
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results_multi.pkl")
    ap.add_argument("--outdir", default="figs/multi")
    ap.add_argument("--only-K", type=int, default=None)
    args = ap.parse_args()
    D = pickle.load(open(args.pkl, "rb"))
    runs = D["runs"]
    full = {k[0]: e for k, e in runs.items() if len(k) == 2}
    print(f"{'run':<22}{'dim':>5}{'cyc':>5}{'reached':>8}{'sumJ':>9}{'it mean':>8}{'it max':>7}{'%nonconv':>9}"
          f"{'solve ms':>9}{'max ms':>8}{'ms/it mean':>11}{'ms/it med':>10}{'ms/it max':>10}{'cycle ms':>9}")
    print("-" * 130)
    for key in sorted(runs, key=lambda k: (k[0], len(k), k[-1] if len(k) > 2 else -1)):
        e = runs[key]; s = stats(e)
        print(f"{e['label']:<22}{7 * e['K']:>5d}{e['ncyc']:>5d}{('all' if e['all_reached'] else str(int(np.sum(np.isfinite(e['t_reach']))))):>8}"
              f"{e['Jtrue'].sum():>9.3f}{s['it_mean']:>8.2f}{s['it_max']:>7d}{s['pnonconv']:>9.1f}{s['ms_mean']:>9.2f}{s['ms_max']:>8.1f}"
              f"{s['pi_mean']:>11.2f}{s['pi_med']:>10.2f}{s['pi_max']:>10.2f}{s['cyc']:>9.2f}")
    for key, e in runs.items():
        if len(key) == 4 and key[0] in full:
            K, N, _, k = key; f = full[K]; ai = f["agents"].index(k) if k in f["agents"] else k
            n = min(e["U"].shape[2], f["U"].shape[2])
            dU = np.abs(e["U"][0, :, :n] - f["U"][ai, :, :n]).max(); dX = np.abs(e["X"][0, :, :n + 1] - f["X"][ai, :, :n + 1]).max()
            print(f"stacked K={K} agent {k} vs solo: max|dU| {dU:.2e} max|dX| {dX:.2e} over {n} cycles | J {f['Jtrue'][ai]:.4f} vs {e['Jtrue'][0]:.4f}")
    Ks = [args.only_K] if args.only_K else sorted(full)
    for K in Ks:
        if K in full:
            d = os.path.join(args.outdir, f"K{K}"); os.makedirs(d, exist_ok=True)
            plot_paths(full[K], os.path.join(d, "paths.png")); plot_review(full[K], os.path.join(d, "review.png"))
            print(f"saved {d}/paths.png, {d}/review.png")
    if len(full) >= 2:
        os.makedirs(args.outdir, exist_ok=True)
        plot_scaling(full, os.path.join(args.outdir, "scaling.png")); print(f"saved {args.outdir}/scaling.png")


if __name__ == "__main__":
    main()

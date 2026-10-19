"""
plot_multi_int.py -- figures and table for the interacting multi-agent Hopf-Lax-MPC runs.

For every coupled run (Wc > 0) in the pickle, with its decoupled baseline (same K, N, Rc, wc, Wc = 0)
when present:
  figs/K<K>_Rc<Rc>_Wc<Wc>/compare.png   map: coupled paths (solid) vs decoupled (dashed), safety discs
                                        (radius Rc/2 around each agent, touching = distance Rc) at the
                                        coupled run's closest approach; minimum pairwise distance vs time;
                                        lateral position y(t); inter-agent Jacobian coupling |offdiag Jr|;
                                        iterations and solve time per cycle
plus the printed results table of every run.

USAGE
    python plot_multi_int.py [--pkl results_multi_int.pkl] [--outdir figs]
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
import plot_multi as PM                                                # draw_scene, agent_colors, stats (reused)

GREY = "0.45"


def run_dir(e):
    c = e["col"]
    return f"K{e['K']}_Rc{c['Rc']:g}_Wc{c['Wc']:g}"


def plot_compare(e, b, out):
    """e: coupled run, b: decoupled baseline (or None)."""
    K = e["K"]; cols = PM.agent_colors(K); dta = e["meta"]["dt_apply"]; Rc = e["col"]["Rc"]
    fig = plt.figure(figsize=(14, 10.5), dpi=130)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.6, 1, 1], hspace=0.38, wspace=0.2)
    ax = fig.add_subplot(gs[0, :])
    PM.draw_scene(ax, e)
    kc = int(np.argmin(e["dmin"]))                                    # coupled closest approach (cycle index)
    for k in range(K):
        X = e["X"][k]; tr = e["t_reach"][k]
        if b is not None:
            ax.plot(b["X"][k][0], b["X"][k][1], "--", color=cols[k], lw=1.0, alpha=0.6)
        ax.plot(X[0], X[1], "-", color=cols[k], lw=1.8,
                label=f"agent {k}: J={e['Jtrue'][k]:.3f}, {'arrives %.2f s' % tr if np.isfinite(tr) else 'not reached'}")
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=7)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=12)
        q = X[:2, min(kc, X.shape[1] - 1)]
        ax.add_patch(plt.Circle(q, Rc / 2, fill=False, color=cols[k], lw=1.2))
        ax.plot(*q, "o", color=cols[k], ms=3)
    if b is not None:
        kb = int(np.argmin(b["dmin"]))
        for k in range(K):
            q = b["X"][k][:2, min(kb, b["X"][k].shape[1] - 1)]
            ax.add_patch(plt.Circle(q, Rc / 2, fill=False, color=cols[k], lw=0.8, ls="--", alpha=0.6))
    ax.set_xlim(-0.8, 20.8); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ttl = (f"K={K} interacting agents, coupled Hopf-Lax-MPC (N={e['N']}, no budget), pair penalty Rc={Rc:g} m, "
           f"Wc={e['col']['Wc']:g}, wc={e['col']['wc']:g}.  Solid = coupled, dashed = decoupled (Wc=0); circles of "
           f"radius Rc/2 at each run's closest approach (touching = distance Rc)")
    ax.set_title(ttl, loc="left", fontsize=8.5, wrap=True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(K, 4))

    a = [fig.add_subplot(gs[r, c]) for r in (1, 2) for c in (0, 1)]
    for r, ls, lw, lab in ((e, "-", 1.5, "coupled"), (b, "--", 1.1, "decoupled")):
        if r is None:
            continue
        t = np.arange(1, r["dmin"].size + 1) * dta
        a[0].plot(t, r["dmin"], ls, color="k", lw=lw, label=f"{lab}: min {r['dmin'].min():.3f} m")
        for k in range(K):
            X = r["X"][k]
            a[1].plot(np.arange(X.shape[1]) * dta, X[1], ls, color=cols[k], lw=lw * 0.8)
        a[2].semilogy(r["tlog"], np.maximum(r["offJr"], 1e-16), ls, color="k", lw=lw * 0.8, label=lab)
        a[3].plot(r["tlog"], r["iters"], ".", ms=3 if r is e else 2, color=PM.GREEN if r is e else GREY,
                  label=f"{lab}: {r['iters'][1:].mean():.2f} it/cycle, {PM.stats(r)['pi_mean']:.1f} ms/it")
    a[0].axhline(Rc, color=PM.RED, lw=0.9, ls=":", label=f"Rc = {Rc:g} m")
    a[0].set_title("minimum pairwise distance [m] (over every fine step)", fontsize=9, loc="left")
    a[1].set_title("lateral position y(t) [m]", fontsize=9, loc="left")
    a[2].set_title("inter-agent coupling |offdiag Jr| at the applied iterate", fontsize=9, loc="left")
    a[3].set_title("iterations per cycle", fontsize=9, loc="left")
    for i in (0, 2, 3):
        a[i].legend(fontsize=7, frameon=False)
    for ax_ in a:
        ax_.grid(alpha=.25); ax_.set_xlabel("t [s]", fontsize=8); ax_.tick_params(labelsize=8)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def print_table(entries):
    print(f"{'run':<26}{'dim':>5}{'cyc':>5}{'reached':>8}{'t_last':>7}{'sumJ':>9}{'Jpair':>8}{'dmin':>7}{'it mean':>8}"
          f"{'it max':>7}{'%conv':>7}{'solve ms':>9}{'max ms':>8}{'ms/iter':>8}{'kicks':>6}{'esc':>5}")
    print("-" * 130)
    for e in entries:
        s = PM.stats(e)
        tl = np.nanmax(e["t_reach"]) if e["all_reached"] else np.nan
        reached = "all" if e["all_reached"] else f"{int(np.sum(~np.isnan(e['t_reach'])))}/{e['K']}"
        print(f"{e['label']:<26}{e['K'] * 7:>5d}{e['ncyc']:>5d}{reached:>8}{tl:>7.2f}{e['Jtrue'].sum():>9.3f}"
              f"{e['Jpair_total']:>8.3f}{e['dmin'].min():>7.3f}{s['it_mean']:>8.2f}{s['it_max']:>7d}"
              f"{100 - s['pnonconv']:>7.1f}{s['ms_mean']:>9.2f}{s['ms_max']:>8.1f}{s['pi_mean']:>8.2f}"
              f"{int(e['nkick'].sum()):>6d}{int(e['nesc'].sum()):>5d}{' ABORTED' if e['aborted'] else ''}")
    print("sumJ = tracking + obstacle cost summed over agents (each up to its own arrival); Jpair = true pair-collision "
          "cost; dmin = minimum pairwise distance [m] over every fine step;\ntiming: cycle 1 excluded; ms/iter = solve "
          "time / iterations per cycle (cycles with 0 iterations excluded)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="results_multi_int.pkl")
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        R = pickle.load(f)["runs"]
    keys = sorted(R, key=lambda k: (k[0], k[1], k[2], k[4], k[3]))
    for key in keys:
        e = R[key]
        if e["col"]["Wc"] == 0.0:
            continue
        K, N, Rc, Wc, wc = key
        b = R.get((K, N, Rc, 0.0, wc))
        if b is None:                                                  # any decoupled run of this K, N
            b = next((R[k] for k in keys if k[0] == K and k[1] == N and k[3] == 0.0), None)
        d = os.path.join(args.outdir, run_dir(e))
        os.makedirs(d, exist_ok=True)
        plot_compare(e, b, os.path.join(d, "compare.png"))
        print(f"wrote {d}/compare.png")
    print()
    print_table([R[k] for k in keys])


if __name__ == "__main__":
    main()

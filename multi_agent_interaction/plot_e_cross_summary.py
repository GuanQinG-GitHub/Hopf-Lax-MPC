"""
plot_e_cross_summary.py -- all-method comparison of the E_cross screening (results_demo_screen_E/E_cross_K<K>/<method>.pkl).

  figs/e_cross/summary.png    vs K: trajectory cost J_full (log), mean solve time per cycle, time per iteration, iterations
                              per cycle; one line per method; trapped runs (not every agent arrived) drawn hollow
  figs/e_cross/table.html     HTML table fragment (one row per K x method, ratios to Hopf-Lax-MPC at the same K)
plus the same table printed as text.

USAGE
    python plot_e_cross_summary.py [--dir results_demo_screen_E] [--Ks 2,3,4,5]
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

import plot_demo_int as PD                                             # noqa: E402  ORDER, LBL, COL, stats


def load(d, Ks):
    R = {}
    for K in Ks:
        for m in PD.ORDER:
            f = os.path.join(d, f"E_cross_K{K}", m + ".pkl")
            if os.path.exists(f):
                R[(K, m)] = pickle.load(open(f, "rb"))
    return R


def row(e):
    s = PD.stats(e)
    x = e["extra"]
    note = ""
    if e["method"] == "DDP_Box":
        st = e["status"]
        note = f"mu-max {100 * st.count('mu-max') / len(st):.0f}% of cycles, {int(x['nrej'].sum())} rejected steps"
    elif e["method"] == "Hopf_Lax_MPC":
        note = f"{int(np.sum(x['nkick']))} saddle escapes"
    elif e["method"] == "Vanilla_Hopf_Lax":
        note = f"Hessian {1e3 * np.mean(x['t_hess']):.0f} ms/cycle" if "t_hess" in x else ""
    nst = {k: e["status"].count(k) for k in sorted(set(e["status"]))}
    return dict(arr=int(np.sum(~np.isnan(e["t_reach"]))), K=e["K"], end=e["end"], J=e["J_full"], it=s["it"], itmax=s["itmax"],
                ms=s["ms"], msmax=s["msmax"], p95=1e3 * np.percentile(e["tsolve"][1:], 95), pi=s["pi"], note=note, status=nst,
                tsim=e["ncyc"] * e["meta"]["dt_apply"], wall=e["wall_s"])


def plot_summary(R, Ks, out):
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2), dpi=120)
    keys = [("J", "trajectory cost J_full (log)", True), ("ms", "mean solve time per cycle [ms] (log)", True),
            ("pi", "time per iteration [ms]", False), ("it", "iterations per cycle (mean)", False)]
    for m in PD.ORDER:
        ks = [K for K in Ks if (K, m) in R]
        if not ks:
            continue
        rr = [row(R[(K, m)]) for K in ks]
        for a, (k, t, lg) in zip(ax, keys):
            v = [r[k] for r in rr]
            a.plot(ks, v, "-", color=PD.COL[m], lw=1.4, label=PD.LBL[m])
            for K, r, vv in zip(ks, rr, v):
                full = r["arr"] == r["K"]
                a.plot(K, vv, "o", color=PD.COL[m], mfc=PD.COL[m] if full else "white", ms=7, mew=1.5)
            a.set_title(t, fontsize=9, loc="left")
            if lg:
                a.set_yscale("log")
    for a in ax:
        a.set_xticks(Ks); a.set_xlabel("K (agents)"); a.grid(alpha=.25)
    ax[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("E_cross screening (parallel batches of 4 on one machine: timings comparable within a K, not timing-grade); "
                 "hollow marker = not every agent arrived (trapped / deadlocked)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def table(R, Ks, out):
    hdr = ["K", "method", "arrived", "end", "J_full", "J / ours", "it/cycle mean (max)", "solve ms/cycle mean", "p95 ms",
           "ms/cycle / ours", "ms/iter", "notes"]
    html = ["<table>", "<tr>" + "".join(f"<th>{h}</th>" for h in hdr) + "</tr>"]
    txt = []
    for K in Ks:
        if (K, "Hopf_Lax_MPC") not in R:
            continue
        r0 = row(R[(K, "Hopf_Lax_MPC")])
        for m in PD.ORDER:
            if (K, m) not in R:
                continue
            r = row(R[(K, m)])
            cls = ' class="ok"' if r["arr"] == K else ' class="bad"'
            jr = r["J"] / r0["J"]; tr = r["ms"] / r0["ms"]
            lb = " (lower bound)" if r["end"] == "deadlock" else ""
            cells = [str(K), PD.LBL[m], f"<span{cls}>{r['arr']}/{K}</span>", r["end"], f"{r['J']:.1f}{lb}",
                     "1" if m == "Hopf_Lax_MPC" else f"<b>{jr:.1f}×</b>", f"{r['it']:.1f} ({r['itmax']})", f"{r['ms']:.0f}",
                     f"{r['p95']:.0f}", "1" if m == "Hopf_Lax_MPC" else f"<b>{tr:.1f}×</b>", f"{r['pi']:.1f}", r["note"]]
            html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            txt.append(f"K={K} {m:18s} {r['arr']}/{K} {r['end']:11s} J {r['J']:8.1f} ({jr:5.1f}x) it {r['it']:5.1f} ({r['itmax']:4d}) "
                       f"ms {r['ms']:7.0f} p95 {r['p95']:6.0f} ({tr:4.1f}x) ms/it {r['pi']:6.1f} | {r['note']} | {r['status']}")
    html.append("</table>")
    open(out, "w").write("\n".join(html) + "\n")
    print("\n".join(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_demo_screen_E")
    ap.add_argument("--Ks", default="2,3,4,5")
    ap.add_argument("--outdir", default="figs/e_cross")
    args = ap.parse_args()
    Ks = [int(k) for k in args.Ks.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    R = load(args.dir, Ks)
    plot_summary(R, Ks, os.path.join(args.outdir, "summary.png"))
    table(R, Ks, os.path.join(args.outdir, "table.html"))
    print("wrote", os.path.join(args.outdir, "summary.png"), os.path.join(args.outdir, "table.html"))


if __name__ == "__main__":
    main()

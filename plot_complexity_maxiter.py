"""
plot_complexity_maxiter.py -- variant of plot_complexity.py where the TRACE-BASED per-iteration
estimate (collocation / MPPI, the methods without a structural iterbench) uses the MAX observed
inter-iteration time instead of the median: IPOPT iterations have unfixed content (extra inertia
corrections / KKT refactorizations in some iterations), which made the median line non-monotone in
N; the max shows the genuinely worst observed iteration.  Per rep the first record's dt is still
excluded (per-solve setup); the per-rep max is then medianed ACROSS reps to reject one-off OS
scheduling noise.  m1v2/pmp/ddp keep their structural worst-case benchmarks (already fixed-content).

USAGE:  python plot_complexity_maxiter.py    -> figs/complexity_vs_N_maxiter.png + printed table
"""
from __future__ import annotations

import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

# Optimality band.  2% (was 1%): the measured cross-family floor is the shooting methods'
# certificate-stop + ZOH control-extraction offset under the common rollout_cost metric -- up to
# +1.35% at N=250..290 with |res| < 1e-10 (fully converged on their own functional) -- so a 1% band
# sits below the measurement floor at those horizons and mislabels converged solves as DNF.
EPS = 2e-2
PKL = "results_complexity.pkl"
NMAX = 300         # paper range: N > 300 excluded (shooting sensitivity regime, reported separately)


def per_iter_ms(reps):
    """WORST observed iteration: per rep, the MAX inter-record time with the first record's dt
    excluded when possible (per-solve setup + occasional first-rep retrace); median across reps
    (one-off OS scheduling spikes should not define the estimate, a repeatable worst iteration
    should).  Replaces the median-of-dts estimate, whose value drifted non-monotonically in N for
    collocation because IPOPT iterations have unfixed content."""
    vals = []
    for rep in reps:
        tr = [r for r in rep["trace"] if r["k"] != "final"]
        if not tr:
            continue
        ts = np.array([r["t"] for r in tr])
        dts = np.diff(np.concatenate([[0.0], ts]))
        use = dts[1:] if dts.size >= 2 else dts
        vals.append(np.max(use))
    return 1e3 * float(np.median(vals)) if vals else np.nan

COL = {"m1v2": (.10, .55, .20), "pmp": (.85, .15, .15), "ddp": (.90, .45, .0),
       "coll": (0, .45, .85), "mppi8192": (.75, .45, .85), "mppi12288": (.60, .20, .70),
       "mppi16384": (.40, .10, .50), "mppi20480": (.25, .05, .35)}
# Display names aligned with the closed-loop testbed figures (v11 conventions).
LBL = {"m1v2": "Hopf–Lax-MPC", "pmp": "PMP", "ddp": "DDP", "coll": "Collocation",
       "mppi8192": "MPPI-(K=8192)", "mppi12288": "MPPI-(K=12288)",
       "mppi16384": "MPPI-(K=16384)", "mppi20480": "MPPI-(K=20480)"}
HIDE = {"mppi8192", "mppi16384"}                                       # not drawn; still in J*(N)


def main():
    with open(PKL, "rb") as f:
        D = pickle.load(f)
    runs = {k: v for k, v in D["runs"].items() if k[1] <= NMAX}
    methods = sorted({m for m, _ in runs}, key=lambda m: list(COL).index(m) if m in COL else 99)
    methods = [m for m in methods if m not in HIDE]                    # J*(N) still uses ALL runs
    Ns_all = sorted({N for _, N in runs})

    # ---- J*(N): best final J across CONVERGED runs at that N (a stalled/diverged final must not
    #      define the optimum).  Horizons with no converged run at all -> every entry is DNF. ----
    Jstar = {}
    for (m, N), e in runs.items():
        for rep in e["reps"]:
            Jf = rep["trace"][-1]["J"]
            if np.isfinite(Jf) and rep["status"] == "converged":
                Jstar[N] = min(Jstar.get(N, np.inf), Jf)

    rows = []
    series = {m: dict(N=[], t=[], dnf=[], iters=[], mspi=[]) for m in methods}
    for m in methods:
        for N in Ns_all:
            e = runs.get((m, N))
            if e is None:
                continue
            band = Jstar.get(N, np.nan) * (1 + EPS)
            touts, dnf_rep = [], False
            for rep in e["reps"]:
                tr = rep["trace"]
                hit = None
                if np.isfinite(band):
                    js = np.array([r["J"] for r in tr])
                    ts = np.array([r["t"] for r in tr])
                    okmask = np.isfinite(js) & (js <= band)
                    # ENTER AND STAY: earliest record from which every later J stays inside the band
                    # (a transient dip through the band -- possible for noisy/non-monotone traces --
                    # does not count as having reached the optimum).
                    stay = np.logical_and.accumulate(okmask[::-1])[::-1]
                    if stay.any():
                        hit = float(ts[np.argmax(stay)])
                if hit is None or rep["status"] in ("diverged", "stalled", "esc-stalled"):
                    dnf_rep = True
                else:
                    touts.append(hit)
            nit = len(e["reps"][0]["trace"]) - 1
            # per-iteration cost: WORST-CASE-content structural benchmark where available
            # (m1v2/pmp/ddp; big-O-consistent), else uniform-content trace median (coll/MPPI).
            ib = D.get("iterbench", {}).get((m, N))
            mspi = float(ib) if ib is not None else per_iter_ms(e["reps"])
            dnf = dnf_rep or not touts
            tval = float(np.median(touts)) if touts else np.nan
            status = e["reps"][0]["status"]
            series[m]["N"].append(N)
            series[m]["t"].append(tval)
            series[m]["dnf"].append(dnf)
            series[m]["iters"].append(nit)
            series[m]["mspi"].append(mspi)
            rows.append((m, N, tval, nit, mspi, status, Jstar.get(N, np.nan),
                         e["reps"][0]["trace"][-1]["J"], e["reps"][0]["trace"][-1].get("nr")))

    # ---- table ----
    print(f"{'method':<11}{'N':>5}{'t_opt[ms]':>11}{'iters':>7}{'ms/iter':>9}{'status':>12}"
          f"{'J*':>12}{'J_final':>12}{'|res|_f':>10}")
    print("-" * 90)
    for m, N, tval, nit, mspi, status, Js, Jf, nrf in rows:
        tv = f"{1e3 * tval:.2f}" if np.isfinite(tval) else "DNF"
        nr_s = f"{nrf:.1e}" if nrf is not None else "--"
        print(f"{m:<11}{N:>5}{tv:>11}{nit:>7}{mspi:>9.2f}{status:>12}{Js:>12.5f}{Jf:>12.5f}{nr_s:>10}")

    # ---- figure: per-iteration cost panel + converged-optimality panel ----
    fig = plt.figure(figsize=(13.0, 5.6), dpi=140)
    gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.06, right=0.985, bottom=0.11, top=0.88)
    ax_c = fig.add_subplot(gs[0])
    ax_o = fig.add_subplot(gs[1])

    for m in methods:
        s, c = series[m], COL.get(m, (.3, .3, .3))
        ax_c.plot(s["N"], s["mspi"], "-o", color=c, ms=4, lw=1.6, label=LBL.get(m, m))
    ax_c.set_xlabel("horizon N"); ax_c.set_ylabel("worst-case iteration cost [ms]")
    ax_c.set_title("per-iteration cost (worst-case content, structural benchmark)",
                   loc="left", fontsize=10)
    ax_c.legend(frameon=False, fontsize=8)
    ax_c.grid(True, alpha=0.25)

    # converged optimality J*(N)/J_final in (0,1]: 1 = optimal.  (The signed form 1 - gap/J* goes
    # negative past 100% gap -- DDP stalls would leave the axis -- so the bounded reciprocal is
    # used; identical reading and ordering.)  MPPI: best-of-budget J, per its no-iteration role.
    for m in methods:
        c = COL.get(m, (.3, .3, .3))
        xs, os_ = [], []
        for N in series[m]["N"]:
            js = np.array([r["J"] for r in runs[(m, N)]["reps"][0]["trace"]])
            js = js[np.isfinite(js) & (js > 0)]
            if js.size and N in Jstar:
                xs.append(N)
                os_.append(Jstar[N] / (js.min() if m.startswith("mppi") else js[-1]))
        # m1v2 and coll overlap at 1.0; m1v2 is drawn first (underneath) with a wider line and
        # larger markers so it stays visible as a rim under collocation's thinner line on top
        lw, ms = (3.4, 8.0) if m == "m1v2" else (1.4, 3.5) if m == "coll" else (1.6, 4)
        ax_o.plot(xs, os_, "-^" if m.startswith("mppi") else "-o", color=c, ms=ms, lw=lw,
                  label=LBL.get(m, m))
    ax_o.axhline(1 / (1 + EPS), color="0.35", ls="--", lw=1.0, alpha=0.8)
    ax_o.set_xlabel("horizon N")
    ax_o.set_ylabel("converged optimality  J*(N) / J_final")
    ax_o.set_ylim(0, 1.05)
    ax_o.set_title("optimality of the converged solution (1 = optimal)", loc="left", fontsize=10)
    ax_o.legend(frameon=False, fontsize=8, loc="lower left")
    ax_o.grid(True, alpha=0.25)
    fig.suptitle("Complexity vs horizon -- single solve at the obs-2 saddle, symmetric init",
                 fontsize=12)
    fig.savefig("figs/complexity_vs_N_maxiter.png", bbox_inches="tight")
    print("\nsaved figs/complexity_vs_N_maxiter.png")


if __name__ == "__main__":
    main()

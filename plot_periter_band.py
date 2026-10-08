"""
plot_periter_band.py -- standalone plot: mean per-iteration computation time vs horizon N, with a
+- 3 sigma band, for every method of the no-budget N sweep (results_N_sweep.pkl).

This is a self-contained version of figure 5 of plot_N_sweep.py, written for manual tuning: every
styling choice is a named constant at the top of the file, and every step is commented.

DATA MODEL (what results_N_sweep.pkl contains, as written by mpc_testbed_N_sweep.py)
    D["runs"][(method, N)] -> one closed-loop run (up to 800 cycles) with per-cycle arrays:
        e["tsolve"]    (ncyc,)  solver wall time of each cycle [s]   (the solver call only)
        e["iters"]     (ncyc,)  solver iterations in that cycle
        e["tcyc"]      (ncyc,)  whole-cycle wall time [s]            (used only for the recompile mask)
        e["recompile"] (ncyc,)  bool, True when JAX recompiled inside that cycle (excluded from timing)
    method tokens: "m1v2" (Hopf-Lax-MPC), "pmp", "newton", "ddp", "coll", "mppi12288", "mppi20480"

PER-ITERATION TIME (the quantity plotted)
    for each cycle c with iters[c] > 0:   pi[c] = tsolve[c] / iters[c]          [s -> ms]
    line  = mean_c pi[c]
    band  = mean_c pi[c] -+ BAND_K * std_c pi[c]       (lower edge clipped at 0)
    Cycle 1 (cold start) and recompile-tagged cycles are excluded, as in the testbed summary.

USAGE
    python plot_periter_band.py                                   # -> figs/N300_sweep/N_sweep_periter_mean_band.png
    python plot_periter_band.py --pkl results_N_sweep.pkl --out my.png --Nmax 200 --k 2
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")                                                  # headless backend: write PNG, no window
import matplotlib.pyplot as plt                                        # noqa: E402

# ======================================================================================================
#  STYLE CONSTANTS -- edit these to tune the plot
# ======================================================================================================
# Methods to draw, in legend / z-order (first = drawn first = bottom).  Remove a token to hide it.
ORDER = ["m1v2", "pmp", "newton", "ddp", "coll", "mppi12288", "mppi20480"]

# Legend label per method token.
LBL = {"m1v2": "Hopf-Lax-MPC", "pmp": "PMP", "newton": "Newton (exact Hessian)", "ddp": "DDP",
       "coll": "Collocation", "mppi12288": "MPPI-(K=12288)", "mppi20480": "MPPI-(K=20480)"}

# Line / marker / band colour per method (RGB in 0..1).  The band uses the same colour, lighter.
COL = {"m1v2": (.10, .55, .20), "pmp": (.85, .15, .15), "newton": (0.0, .60, .60), "ddp": (.90, .45, .0),
       "coll": (0, .45, .85), "mppi12288": (.60, .20, .70), "mppi20480": (.25, .05, .35)}

# Marker symbol per method ("o" circle, "s" square, "^" triangle, "D" diamond, ...).
MK = {"m1v2": "o", "pmp": "o", "newton": "s", "ddp": "o", "coll": "o", "mppi12288": "^", "mppi20480": "^"}

# Line width and marker size per method; the paper's method is drawn heavier so it stays visible.
LW = {"m1v2": 3.0}                                                     # default LW_DEFAULT for the rest
MS = {"m1v2": 8}                                                       # default MS_DEFAULT for the rest
LW_DEFAULT, MS_DEFAULT = 1.6, 5

BAND_K = 3.0                                                           # band half-width in sigmas
BAND_ALPHA = 0.15                                                      # band transparency (0 = invisible)
CLIP_LOWER_AT_ZERO = True                                              # a time cannot be negative

FIGSIZE = (7.2, 5.4)                                                   # inches
DPI = 140
YLABEL = "Average per-iteration computation time [ms]"
XLABEL = "horizon N"
TITLE = ("per-iteration solve time, closed loop without budget\n"
         "(line = mean over cycles, band = mean +- {k:g} sigma over cycles)")   # {k} -> BAND_K
TITLE_FONTSIZE = 10
LEGEND_FONTSIZE = 8
GRID_ALPHA = 0.25
YLIM_BOTTOM = 0.0                                                      # None = let matplotlib choose
YLIM_TOP = None                                                        # None = let matplotlib choose
LOG_Y = False                                                          # True = logarithmic y axis


# ======================================================================================================
#  DATA HELPERS
# ======================================================================================================
def keep_mask(e):
    """Boolean mask of the cycles that enter the statistics.

    Excludes cycle 1 (JIT cold start) and every cycle tagged by the recompile detector, exactly as the
    testbed summary table does.  If nothing survives (tiny smoke runs) all cycles are kept."""
    n = len(e["tcyc"])
    keep = np.ones(n, dtype=bool)
    keep[0] = False                                                    # cycle 1: cold start
    rc = np.asarray(e["recompile"], dtype=bool)
    if rc.size == n:
        keep &= ~rc                                                    # drop recompile-tagged cycles
    if not keep.any():
        keep[:] = True
    return keep


def per_iteration_times_ms(e):
    """Per-cycle per-iteration solver time in ms for one run: tsolve / iters over the kept cycles.

    Cycles with 0 iterations (the warm start already satisfied the stopping test) carry no
    per-iteration information and are dropped."""
    keep = keep_mask(e)
    ts = np.asarray(e["tsolve"])[keep]                                 # solver seconds per cycle
    its = np.asarray(e["iters"])[keep]                                 # iterations per cycle
    pos = its > 0
    if not pos.any():
        return np.array([np.nan])
    return 1e3 * ts[pos] / its[pos]                                    # ms per iteration, one value per cycle


def method_curve(runs, m, Nmax=None):
    """For method token m: sorted horizons and, per horizon, mean and std of the per-iteration time.

    Returns (Ns, mean, std) as numpy arrays; horizons above Nmax are skipped when Nmax is given."""
    Ns = sorted(k[1] for k in runs if k[0] == m and (Nmax is None or k[1] <= Nmax))
    mean, std = [], []
    for N in Ns:
        pi = per_iteration_times_ms(runs[(m, N)])
        mean.append(float(np.mean(pi)))
        std.append(float(np.std(pi)))                                  # population std over cycles
    return np.array(Ns), np.array(mean), np.array(std)


# ======================================================================================================
#  PLOT
# ======================================================================================================
def draw(runs, out, Nmax=None, k=BAND_K):
    """Draw the figure for all methods in ORDER that exist in `runs` and save it to `out`."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for m in ORDER:
        if not any(key[0] == m for key in runs):                       # method absent from the pickle
            continue
        Ns, mu, sd = method_curve(runs, m, Nmax)
        col = COL.get(m, (.3, .3, .3))                                 # grey fallback for unknown tokens
        lo = mu - k * sd
        hi = mu + k * sd
        if CLIP_LOWER_AT_ZERO:
            lo = np.maximum(lo, 0.0)
        # band first (lower z-order), then the mean line with markers on top
        ax.fill_between(Ns, lo, hi, color=col, alpha=BAND_ALPHA, lw=0)
        ax.plot(Ns, mu, "-" + MK.get(m, "o"), color=col,
                lw=LW.get(m, LW_DEFAULT), ms=MS.get(m, MS_DEFAULT), label=LBL.get(m, m))
    # ---- axes cosmetics ----
    if LOG_Y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=YLIM_BOTTOM, top=YLIM_TOP)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE.format(k=k), loc="left", fontsize=TITLE_FONTSIZE)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def main():
    """CLI: pickle path, output path, optional horizon cap and band width."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="results_N_sweep.pkl", help="sweep pickle (mpc_testbed_N_sweep.py)")
    ap.add_argument("--out", default="figs/N300_sweep/N_sweep_periter_mean_band.png", help="output PNG")
    ap.add_argument("--Nmax", type=int, default=None, help="only plot horizons N <= Nmax")
    ap.add_argument("--k", type=float, default=BAND_K, help="band half-width in sigmas (default 3)")
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        runs = pickle.load(f)["runs"]                                  # {(method, N): run dict}
    draw(runs, args.out, args.Nmax, args.k)


if __name__ == "__main__":
    main()

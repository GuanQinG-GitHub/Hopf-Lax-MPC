"""
plot_cycle_band.py -- standalone plot: mean TOTAL computation time per control cycle vs horizon N,
optionally with a +- 3 sigma band, for every method of the no-budget N sweep, and (--overlay-cost) the
final closed-loop trajectory cost on a right axis.  The quantity is the whole solve of a cycle, i.e. all
of its iterations.

MAIN DELIVERABLE (shared stopping rule, N <= 200)
    python plot_cycle_band.py --pkl results_N_sweep_shared_eps1e-06.pkl --log --no-band --mark-reached         --overlay-cost --Nmax 200 --exclude DDP --out figs/N_sweep_time_cost.png
    (--exclude DDP: the plain-DDP runs kept in that pickle; the DDP baseline is DDP_Box)

DATA MODEL (what results_N_sweep.pkl contains, as written by mpc_testbed_N_sweep.py)
    D["runs"][(method, N)] -> one closed-loop run (up to 800 cycles) with per-cycle arrays:
        e["tsolve"]    (ncyc,)  solver wall time of each cycle [s]   (the solver call: all its iterations)
        e["tcyc"]      (ncyc,)  whole-cycle wall time [s]            (used only for the recompile mask)
        e["recompile"] (ncyc,)  bool, True when JAX recompiled inside that cycle (excluded from timing)
    method tokens: "Hopf_Lax_MPC", "PMP_Shooting", "Vanilla_Hopf_Lax", "DDP", "Collocation_IPOPT", "MPPI", "MPPI_20480"

TOTAL TIME PER CYCLE (the quantity plotted)
    for each cycle c:   tc[c] = tsolve[c]          [s -> ms]
    line  = mean_c tc[c]
    band  = mean_c tc[c] -+ BAND_K * std_c tc[c]       (lower edge clipped at 0, or at YLOG_FLOOR on a log axis)
    Cycle 1 (cold start) and recompile-tagged cycles are excluded, as in the testbed summary.

USAGE
    python plot_cycle_band.py                                     # -> figs/N_sweep_cycle_mean_band.png (+ .pdf)
    python plot_cycle_band.py --log                               # -> figs/N_sweep_cycle_mean_band_log.png (+ .pdf)
    python plot_cycle_band.py --log --no-band                     # -> figs/N_sweep_cycle_mean_log.png (+ .pdf), lines only
    python plot_cycle_band.py --pkl results_N_sweep.pkl --out my.png --Nmax 200 --k 2 --no-pdf
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")                                                  # headless backend: write PNG, no window
import matplotlib.pyplot as plt                                        # noqa: E402

# PDF export settings: Type 42 (TrueType) fonts keep text editable/searchable in the PDF.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ======================================================================================================
#  STYLE CONSTANTS -- edit these to tune the plot
# ======================================================================================================
ORDER = ["Hopf_Lax_MPC", "PMP_Shooting", "Vanilla_Hopf_Lax", "DDP_Box", "DDP", "Collocation_IPOPT", "MPPI", "MPPI_20480"]
LBL = {m: m for m in ORDER}                                            # legend = method name
COL = {"Hopf_Lax_MPC": (.10, .55, .20), "PMP_Shooting": (.85, .15, .15), "Vanilla_Hopf_Lax": (0.0, .60, .60),
       "DDP_Box": (.65, .30, .0), "DDP": (.90, .45, .0), "Collocation_IPOPT": (0, .45, .85), "MPPI": (.60, .20, .70), "MPPI_20480": (.25, .05, .35)}
MK = {"Hopf_Lax_MPC": "o", "PMP_Shooting": "o", "Vanilla_Hopf_Lax": "s", "DDP_Box": "D", "DDP": "o", "Collocation_IPOPT": "o",
      "MPPI": "^", "MPPI_20480": "^"}
LW = {"Hopf_Lax_MPC": 3.0}                                             # default LW_DEFAULT for the rest
MS = {"Hopf_Lax_MPC": 8}                                               # default MS_DEFAULT for the rest
LW_DEFAULT, MS_DEFAULT = 1.6, 5

BAND_K = 3.0                                                           # band half-width in sigmas
BAND_ALPHA = 0.15                                                      # band transparency (0 = invisible)
CLIP_LOWER_AT_ZERO = True                                              # a time cannot be negative

FIGSIZE = (7.2, 5.4)                                                   # inches (also the PDF page size)
DPI = 140                                                              # PNG raster resolution (CLI --dpi)
EXPORT_PDF = True                                                      # also write <out>.pdf unless --no-pdf
YLABEL = "Average computation time per control cycle [ms]"
XLABEL = "horizon N"
TITLE = ("total solve time per control cycle (all iterations), closed loop without budget\n"
         "(line = mean over cycles, band = mean +- {k:g} sigma over cycles)")   # {k} -> BAND_K
TITLE_NOBAND = ("total solve time per control cycle (all iterations), closed loop without budget\n"
                "(line = mean over cycles)")                           # --no-band
TITLE_FONTSIZE = 10
LEGEND_FONTSIZE = 8
GRID_ALPHA = 0.25
YLIM_BOTTOM = 0.0                                                      # linear axis; None = matplotlib's choice
YLIM_TOP = None
YLOG_FLOOR = 0.1                                                       # [ms] lower axis / band edge on a log axis
PEN_THRESHOLD = 500.0                                                  # obstacle cost above which a "reached" run
                                                                       # counts as having driven through an obstacle


# ======================================================================================================
#  DATA HELPERS
# ======================================================================================================
def keep_mask(e):
    """Cycles that enter the statistics: all except cycle 1 (JIT cold start) and recompile-tagged
    cycles, exactly as the testbed summary table.  If nothing survives, all cycles are kept."""
    n = len(e["tcyc"])
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    rc = np.asarray(e["recompile"], dtype=bool)
    if rc.size == n:
        keep &= ~rc
    if not keep.any():
        keep[:] = True
    return keep


def cycle_times_ms(e):
    """Per-cycle total solver time in ms (all iterations of that cycle) over the kept cycles."""
    return 1e3 * np.asarray(e["tsolve"])[keep_mask(e)]


def method_curve(runs, m, Nmax=None):
    """For method token m: sorted horizons and, per horizon, mean and std of the per-cycle time."""
    Ns = sorted(k[1] for k in runs if k[0] == m and (Nmax is None or k[1] <= Nmax))
    mean, std = [], []
    for N in Ns:
        tc = cycle_times_ms(runs[(m, N)])
        mean.append(float(np.mean(tc)))
        std.append(float(np.std(tc)))                                  # population std over cycles
    return np.array(Ns), np.array(mean), np.array(std)


# ======================================================================================================
#  PLOT
# ======================================================================================================
def draw(runs, out, Nmax=None, k=BAND_K, pdf=None, dpi=DPI, log_y=False, exclude=(), ymax=YLIM_TOP, band=True,
         mark_reached=False, pen_threshold=PEN_THRESHOLD, overlay_cost=False):
    """mark_reached: filled marker where the run reached the goal without driving through an obstacle
    (e["reached"] and obstacle cost e["Jpen"] <= pen_threshold), hollow otherwise.
    overlay_cost: right axis (log) = realised closed-loop cost of the whole driven trajectory, e["Jtrue"]
    = tracking + obstacle cost integrated along the applied trajectory with the TRUE obstacle positions,
    from t = 0 to the goal (or to the end of the run if it never got there); dashed lines, same colours."""
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] + (1.2 if overlay_cost else 0.0), FIGSIZE[1]), dpi=dpi)
    ax2 = ax.twinx() if overlay_cost else None
    for m in ORDER:
        if m in exclude or not any(key[0] == m for key in runs):
            continue
        Ns, mu, sd = method_curve(runs, m, Nmax)
        col = COL.get(m, (.3, .3, .3))
        lo, hi = mu - k * sd, mu + k * sd
        if log_y:
            lo = np.maximum(lo, YLOG_FLOOR)                            # no zero / negative edge on a log axis
        elif CLIP_LOWER_AT_ZERO:
            lo = np.maximum(lo, 0.0)
        if band:
            ax.fill_between(Ns, lo, hi, color=col, alpha=BAND_ALPHA, lw=0)
        mk, lw, ms = MK.get(m, "o"), LW.get(m, LW_DEFAULT), MS.get(m, MS_DEFAULT)
        if ax2 is not None:                                            # realised trajectory cost, right axis
            Jt = np.array([runs[(m, N)]["Jtrue"] for N in Ns])
            okc = np.array([bool(runs[(m, N)]["reached"]) and runs[(m, N)]["Jpen"] <= pen_threshold for N in Ns])
            ax2.plot(Ns, Jt, "--", color=col, lw=1.0, alpha=0.85)
            ax2.plot(Ns[okc], Jt[okc], "x", color=col, ms=5, mew=1.4, ls="none")
            ax2.plot(Ns[~okc], Jt[~okc], "x", color=col, ms=5, mew=0.8, ls="none", alpha=0.6)
        if not mark_reached:
            ax.plot(Ns, mu, "-" + mk, color=col, lw=lw, ms=ms, label=LBL.get(m, m))
            continue
        ok = np.array([bool(runs[(m, N)]["reached"]) and runs[(m, N)]["Jpen"] <= pen_threshold for N in Ns])
        ax.plot(Ns, mu, "-", color=col, lw=lw, label=None)
        ax.plot(Ns[ok], mu[ok], mk, color=col, ms=ms, ls="none", label=LBL.get(m, m))       # reached: filled
        ax.plot(Ns[~ok], mu[~ok], mk, color=col, ms=ms, ls="none", mfc="white", mew=1.3)   # not reached: hollow
    if mark_reached:                                                   # legend entries for the marker fill
        ax.plot([], [], "o", color="0.35", ms=6, ls="none", label="filled: reached the goal")
        ax.plot([], [], "o", color="0.35", ms=6, ls="none", mfc="white", mew=1.3, label=f"hollow: not reached, or obstacle\ncost > {pen_threshold:g} (drove through)")
    if log_y:
        ax.set_yscale("log")
        if band:
            ax.set_ylim(bottom=YLOG_FLOOR)                             # band edges are clipped at the floor
    else:
        ax.set_ylim(bottom=YLIM_BOTTOM, top=ymax)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE.format(k=k) if band else TITLE_NOBAND, loc="left", fontsize=TITLE_FONTSIZE)
    ax.grid(alpha=GRID_ALPHA, which="both" if log_y else "major")
    if ax2 is not None:
        ax2.set_yscale("log")
        ax2.set_ylabel("final trajectory cost (closed loop, dashed)")
        ax.set_ylabel(YLABEL + " (solid)")
        ax.grid(True, which="major", alpha=0.55, lw=0.8)               # clearly visible grid for both axes
        ax.grid(True, which="minor", alpha=0.25, lw=0.5)
        ax.xaxis.grid(True, which="major", alpha=0.55, lw=0.8)
        ax2.grid(True, which="major", axis="y", ls=":", alpha=0.5, lw=0.8)
        ax.set_title((TITLE.format(k=k) if band else TITLE_NOBAND)
                     + "\nright axis: final trajectory cost J_track + J_obs of the driven trajectory",
                     loc="left", fontsize=TITLE_FONTSIZE)
        ax.plot([], [], "-", color="0.35", lw=1.6, label="solid: computation time (left)")
        ax.plot([], [], "--x", color="0.35", lw=1.0, ms=5, label="dashed x: trajectory cost (right)")
    if mark_reached:                                                   # 9 entries: outside, right of the axes
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="center left",
                  bbox_to_anchor=(1.12 if ax2 is not None else 1.01, 0.5))
    else:
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"saved {out}")
    if pdf:
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        print(f"saved {pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="results_N_sweep.pkl", help="sweep pickle (mpc_testbed_N_sweep.py)")
    ap.add_argument("--out", default=None, help="output PNG (default figs/N_sweep_cycle_mean_band[_log].png)")
    ap.add_argument("--log", action="store_true", help="logarithmic y axis")
    ap.add_argument("--no-band", action="store_true", help="mean lines only, no sigma band")
    ap.add_argument("--mark-reached", action="store_true", help="filled markers = reached the goal, hollow = not")
    ap.add_argument("--pen-threshold", type=float, default=PEN_THRESHOLD,
                    help="obstacle cost above which a reached run is drawn hollow (default 500)")
    ap.add_argument("--overlay-cost", action="store_true",
                    help="right axis: final closed-loop trajectory cost Jtrue (dashed)")
    ap.add_argument("--exclude", default="", help="comma-separated methods to leave out, e.g. DDP,Vanilla_Hopf_Lax")
    ap.add_argument("--ymax", type=float, default=YLIM_TOP, help="top of the linear y axis [ms] (bands above are cut)")
    ap.add_argument("--Nmax", type=int, default=None, help="only plot horizons N <= Nmax")
    ap.add_argument("--k", type=float, default=BAND_K, help="band half-width in sigmas (default 3)")
    ap.add_argument("--dpi", type=int, default=DPI, help="PNG resolution in dots per inch")
    ap.add_argument("--pdf", default=None, help="PDF output path (default: the PNG path with .pdf)")
    ap.add_argument("--no-pdf", action="store_true", help="write the PNG only")
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        runs = pickle.load(f)["runs"]
    out = args.out or ("figs/N_sweep_cycle_mean" + ("" if args.no_band else "_band")
                       + ("_log" if args.log else "") + ".png")
    pdf = None
    if not args.no_pdf and (EXPORT_PDF or args.pdf):
        pdf = args.pdf or (out.rsplit(".", 1)[0] + ".pdf")
    draw(runs, out, args.Nmax, args.k, pdf, args.dpi, log_y=args.log,
         exclude=tuple(m for m in args.exclude.split(",") if m), ymax=args.ymax, band=not args.no_band,
         mark_reached=args.mark_reached, pen_threshold=args.pen_threshold,
         overlay_cost=args.overlay_cost)


if __name__ == "__main__":
    main()

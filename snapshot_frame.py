"""
snapshot_frame.py -- still frames of the closed-loop animation, publication-styled.

Draws exactly the scene that make_animations.py renders at a chosen time (or frame index) -- the
obstacles at their true positions, the RRT reference, every selected runner's trajectory so far with
its comet tail, current position, predicted trajectory (dashed) and reference window, the MPPI sample
cloud -- and saves it as PNG and/or PDF at a chosen resolution.  Differences from the video frame,
all requested for the paper:
  * a full 3-D box frame around the axes (MATLAB `box on`),
  * axis labels p_x [m], p_y [m], p_z [m],
  * Arial for every number and label (including the math labels),
  * PDF export with an explicit dpi (e.g. 450) for the rasterised surfaces; lines and text stay vector.

The scene geometry, camera, limits, colours and line styles are imported from make_animations.py, so
a snapshot matches the video.  make_animations.py and the pickle are not modified.

USAGE (from the repo root)
    python snapshot_frame.py --t 5.4                                  # all runners, t = 5.4 s -> figs/snapshots/frame_t5.40.png/.pdf
    python snapshot_frame.py --t 5.4,10.0,13.9 --dpi 450              # several frames, 450 dpi
    python snapshot_frame.py --frame 270                              # by frame index (cycle, 20 ms each)
    python snapshot_frame.py --t 10 --methods M1_v2,PMP,Newton-H      # subset of runners (pickle names)
    python snapshot_frame.py --t 10 --pkl results_testbed_newton.pkl --outdir figs/snapshots --no-cloud
    python snapshot_frame.py --t 10 --elev 30 --azim -55 --no-pred --no-title
Runner names in the pickle: PMP, DDP, Collocation, M1_v2, MPPI, MPPI-20480, Newton-H.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")                                                  # headless: files only

# ---- fonts: Arial everywhere, including the math-mode axis labels ($p_x$ etc.) ----
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Arial"
matplotlib.rcParams["mathtext.it"] = "Arial:italic"
matplotlib.rcParams["mathtext.bf"] = "Arial:bold"
matplotlib.rcParams["pdf.fonttype"] = 42                               # embed TrueType (editable text in PDF)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt                                        # noqa: E402

# Scene recipe shared with the video (limits, obstacle colour/alpha, ellipsoid drawer, motion law,
# display names, method colours).  Importing make_animations has no side effect beyond rcParams.
from make_animations import XL, YL, ZL, OBSCOL, OBSALPHA, ellipsoid, mobs_at, disp_name, mcolor  # noqa: E402

# ======================================================================================================
#  STYLE CONSTANTS -- edit to tune
# ======================================================================================================
FIGSIZE = (12.8, 7.2)                                                  # inches, as the video frame
AXES_RECT = [-0.02, -0.04, 0.90, 1.06]                                 # 3-D axes carry large internal margins
BOX_ASPECT = (15 / 2.2, 5.8, 2.6)                                      # x compressed 2.2:1 vs y, z (as video)
ELEV, AZIM = 24, -62                                                   # camera (as video)
AXLBL = 15                                                             # axis-label font size
LABELPAD = 14                                                          # gap between tick numbers and axis label
TICK_FONTSIZE = 11
TITLE_FONTSIZE = 13
LEGEND_FONTSIZE = 10
BOX_COLOR, BOX_LW = "k", 0.9                                           # the `box on` frame
PANE_FILL = False                                                      # True = light grey panes behind the data
TRAJ_LW, TAIL_LW, TAIL_CYCLES = 2.2, 3.6, 20                           # trajectory, comet tail (last 20 cycles)
PRED_LW, WINDOW_LW, WINDOW_ALPHA = 1.5, 6, 0.30                        # dashed prediction, RRT cost window
CLOUD_LW, CLOUD_ALPHA = 0.5, 0.13                                      # MPPI sample cloud
MARK_MS, START_MS, GOAL_MS = 9, 8, 16


# ======================================================================================================
#  DRAWING HELPERS
# ======================================================================================================
def draw_box(ax, elev, azim):
    """MATLAB-style box frame, BACK edges only.

    The cuboid has 12 edges; the three that meet at the corner nearest the camera lie in front of
    the data and would cross the trajectories, so they are omitted and the 9 edges of the three
    back faces are drawn (the faces matplotlib uses for its panes).  The nearest corner follows
    from the camera direction: the camera sits on the +x side when cos(azim) > 0, on the +y side
    when sin(azim) > 0, and above the box when elev > 0."""
    x0, x1 = XL; y0, y1 = YL; z0, z1 = ZL
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    near = (x1 if np.cos(az) > 0 else x0, y1 if np.sin(az) > 0 else y0, z1 if el > 0 else z0)
    edges = [
        ((x0, y0, z0), (x1, y0, z0)), ((x0, y1, z0), (x1, y1, z0)), ((x0, y0, z1), (x1, y0, z1)), ((x0, y1, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y1, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x0, y1, z0), (x0, y1, z1)), ((x1, y1, z0), (x1, y1, z1)),
    ]
    for a, b in edges:
        if a == near or b == near:                                     # one of the three front edges
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "-", color=BOX_COLOR, lw=BOX_LW, zorder=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):                        # panes: no fill, thin edge
        axis.pane.fill = PANE_FILL
        axis.pane.set_edgecolor(BOX_COLOR)
        axis.pane.set_alpha(1.0)


def setup_axes(ax, elev, azim):
    """Limits, aspect, camera, labels (with units), tick font, grid and the box frame."""
    ax.set_xlim(XL); ax.set_ylim(YL); ax.set_zlim(ZL)
    ax.set_box_aspect(BOX_ASPECT)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$p_x$ [m]", fontsize=AXLBL, labelpad=LABELPAD)
    ax.set_ylabel("$p_y$ [m]", fontsize=AXLBL, labelpad=LABELPAD)
    ax.set_zlabel("$p_z$ [m]", fontsize=AXLBL, labelpad=LABELPAD)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.25)
    draw_box(ax, elev, azim)


def draw_scene(ax, D, k):
    """Static scene at frame k: obstacles at their TRUE positions, reference path, start and goal."""
    P, M, path, OM = D["P"], D["M"], D["path"], D.get("obs_motion")
    t = k * P.dt_apply
    for i in range(M.obs.center.shape[1]):                             # static obstacles
        oc = M.obs.center[:, i]
        ellipsoid(ax, oc, M.obs.ax[i], M.obs.ay[i], M.obs.az[i], OBSCOL, OBSALPHA)
    mc = (OM["centers_t"][:, :, min(k, OM["centers_t"].shape[2] - 1)]  # moving obstacles: recorded
          if OM is not None else mobs_at(M, t))                        # positions, else the motion law
    for kk in range(mc.shape[1]):
        ellipsoid(ax, mc[:, kk], M.mobs.ax[kk], M.mobs.ay[kk], M.mobs.az[kk], OBSCOL, OBSALPHA)
    ax.plot(path.pts[0], path.pts[1], np.full(path.pts.shape[1], ZL[0]),
            "-", color=(.75, .75, .78), lw=1.0)                        # ground shadow (depth cue)
    ax.plot(path.pts[0], path.pts[1], path.pts[2], ":", color=(.15, .15, .15), lw=1.4, label="RRT reference")
    ax.plot([P.p0[0]], [P.p0[1]], [P.p0[2]], "o", mfc=(.2, .75, .3), mec="k", ms=START_MS)
    ax.plot([P.pgoal[0]], [P.pgoal[1]], [P.pgoal[2]], "*", mfc=(1, .85, .1), mec="k", ms=GOAL_MS)


def draw_runner(ax, D, R_i, k, cloud=True, pred=True):
    """One runner at frame k: trajectory so far, comet tail, current position, and -- while the run is
    still live -- its predicted trajectory, reference window and (MPPI) sample cloud."""
    P, M = D["P"], D["M"]
    r, c = R_i.res, mcolor(R_i)
    kc = min(k, r.ncyc - 1)                                            # last available cycle for this runner
    live = k < r.ncyc                                                  # still running at this frame?
    ie = min(kc * P.napply + 1, r.X.shape[1])                          # fine-step index of the current cycle
    Xt = r.X[M.posidx][:, :ie]
    ax.plot(Xt[0], Xt[1], Xt[2], "-", color=c, lw=TRAJ_LW, label=disp_name(R_i.name, P))
    it = max(0, ie - TAIL_CYCLES * P.napply)
    ax.plot(Xt[0, it:], Xt[1, it:], Xt[2, it:], "-", color=c, lw=TAIL_LW)             # comet tail
    xc = r.xlog[M.posidx, kc]
    ax.plot([xc[0]], [xc[1]], [xc[2]], "o", mfc=c, mec="w", ms=MARK_MS)               # current position
    if live:
        if cloud and R_i.name.startswith("MPPI") and r.extra[kc] is not None:
            for cl in r.extra[kc]:                                     # sampled rollouts (thinned)
                ax.plot(cl[0], cl[1], cl[2], "-", color=c, lw=CLOUD_LW, alpha=CLOUD_ALPHA)
        if pred:
            Xp, Xw = r.Xpred[kc], r.Xrefw[kc]
            cl = tuple(1 - (1 - np.array(c)) * 0.55)                   # lighter shade of the method colour
            ax.plot(Xp[0], Xp[1], Xp[2], "--", color=cl, lw=PRED_LW)   # predicted trajectory
            ax.plot(Xw[0], Xw[1], Xw[2], "-", color=c, lw=WINDOW_LW, alpha=WINDOW_ALPHA)  # RRT cost window


def snapshot(D, sel, k, outbase, dpi, elev, azim, cloud, pred, title, png=True, pdf=True):
    """Compose one frame for the runner indices `sel` at frame k and save PNG/PDF at `dpi`."""
    P = D["P"]
    t = k * P.dt_apply
    fig = plt.figure(figsize=FIGSIZE, dpi=100)
    ax = fig.add_axes(AXES_RECT, projection="3d")
    setup_axes(ax, elev, azim)
    draw_scene(ax, D, k)
    for i in sel:
        draw_runner(ax, D, D["R"][i], k, cloud, pred)
    if title:
        ax.set_title(f"t = {t:5.2f} s", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.95), fontsize=LEGEND_FONTSIZE, frameon=False)
    if png:
        fig.savefig(outbase + ".png", dpi=dpi, bbox_inches="tight")
        print(f"saved {outbase}.png  ({dpi} dpi)")
    if pdf:
        fig.savefig(outbase + ".pdf", dpi=dpi, bbox_inches="tight")    # surfaces rasterised at dpi,
        print(f"saved {outbase}.pdf  ({dpi} dpi)")                     # lines and text stay vector
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="results_testbed_newton.pkl", help="testbed pickle (with viz logs)")
    ap.add_argument("--t", default=None, help="time(s) in seconds, comma-separated (e.g. 5.4,10)")
    ap.add_argument("--frame", default=None, help="frame index/indices, comma-separated (cycle = 20 ms)")
    ap.add_argument("--methods", default=None, help="comma-separated runner names (default: all)")
    ap.add_argument("--outdir", default="figs/snapshots")
    ap.add_argument("--prefix", default="frame", help="file name prefix: <prefix>_t<time>.png/.pdf")
    ap.add_argument("--dpi", type=int, default=450)
    ap.add_argument("--elev", type=float, default=ELEV)
    ap.add_argument("--azim", type=float, default=AZIM)
    ap.add_argument("--no-cloud", action="store_true", help="hide the MPPI sample cloud")
    ap.add_argument("--no-pred", action="store_true", help="hide predicted trajectories and cost windows")
    ap.add_argument("--no-title", action="store_true", help="omit the 't = ... s' title")
    ap.add_argument("--png-only", action="store_true")
    ap.add_argument("--pdf-only", action="store_true")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        D = pickle.load(f)
    P, R = D["P"], D["R"]
    names = [r.name for r in R]
    if args.methods:
        want = [m.strip() for m in args.methods.split(",")]
        bad = [m for m in want if m not in names]
        if bad:
            raise SystemExit(f"unknown runner(s) {bad}; available: {names}")
        sel = [names.index(m) for m in want]
    else:
        sel = list(range(len(R)))
    if args.frame:
        frames = [int(s) for s in args.frame.split(",")]
    elif args.t:
        frames = [int(round(float(s) / P.dt_apply)) for s in args.t.split(",")]
    else:
        raise SystemExit("give --t (seconds) or --frame (index)")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"loaded {args.pkl}: runners {names}; drawing {[names[i] for i in sel]}")
    for k in frames:
        t = k * P.dt_apply
        base = os.path.join(args.outdir, f"{args.prefix}_t{t:.2f}")
        snapshot(D, sel, k, base, args.dpi, args.elev, args.azim, not args.no_cloud, not args.no_pred,
                 not args.no_title, png=not args.pdf_only, pdf=not args.png_only)


if __name__ == "__main__":
    main()

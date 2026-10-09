"""
snapshot_disturb.py -- publication still frames of a disturbance run (mpc_testbed_disturb.py),
in the style of snapshot_frame.py (figs/snapshots/frame_t16.00.png).

Draws exactly what make_animations_disturb.py renders at the chosen time -- obstacles at their true
positions (static and zero-velocity bodies slate grey, moving bodies red), the RRT reference, the
trajectory so far with its comet tail, the vehicle, and the per-tick arrows for f(x,u) (blue, model
velocity) and d (red, disturbance velocity) -- with the paper styling of snapshot_frame.py: Arial,
box frame, axis labels with units, legend with static/moving patches, PNG + PDF at a chosen dpi.

USAGE (from the repo root)
    python snapshot_disturb.py --bound 30 --seed 0 --t 16                    # last frame (clamped)
    python snapshot_disturb.py --bound 30 --seed 6 --t 6.8,9.4 --dpi 450
    python snapshot_disturb.py --bound 30 --seed 0 --t 9 --hud --no-title
Output: figs/snapshots/disturb_b<b>_seed<s>_t<time>.png/.pdf
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import snapshot_frame as SF                                            # Arial rcParams, box frame, style constants
import matplotlib.pyplot as plt                                        # noqa: E402
from make_animations_disturb import draw_frame, load_run               # noqa: E402


def snapshot(e, P, M, path, k, outbase, dpi, elev, azim, hud, title, png=True, pdf=True, Sh=None):
    fig = plt.figure(figsize=SF.FIGSIZE, dpi=100)
    ax = fig.add_axes(SF.AXES_RECT, projection="3d")
    SF.setup_axes(ax, elev, azim)
    t = e["tlog"][k]
    ttl = f"t = {t:5.2f} s" if title else None
    draw_frame(ax, e, P, M, path, k, hud=hud, legend=True, title=None, traj_lw=SF.TRAJ_LW, tail_lw=SF.TAIL_LW, Sh=Sh)
    if ttl:
        ax.set_title(ttl, fontsize=SF.TITLE_FONTSIZE, fontweight="bold")
    if png:
        fig.savefig(outbase + ".png", dpi=dpi, bbox_inches="tight")
        print(f"saved {outbase}.png  ({dpi} dpi)")
    if pdf:
        fig.savefig(outbase + ".pdf", dpi=dpi, bbox_inches="tight")
        print(f"saved {outbase}.pdf  ({dpi} dpi)")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="results_disturb_posty-0.1.pkl")
    ap.add_argument("--scale", default="practical")
    ap.add_argument("--post-y", type=float, default=-0.1)
    ap.add_argument("--bound", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--t", default=None, help="time(s) in seconds, comma-separated; clamped to the last cycle")
    ap.add_argument("--frame", default=None, help="cycle index/indices, comma-separated")
    ap.add_argument("--outdir", default="figs/snapshots")
    ap.add_argument("--dpi", type=int, default=450)
    ap.add_argument("--elev", type=float, default=SF.ELEV)
    ap.add_argument("--azim", type=float, default=SF.AZIM)
    ap.add_argument("--hud", action="store_true", help="also print the HUD text (time, |f|, |d|, u, iters)")
    ap.add_argument("--no-pred", action="store_true", help="hide the predicted path and the x_ref window")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--png-only", action="store_true")
    ap.add_argument("--pdf-only", action="store_true")
    args = ap.parse_args()

    e, P, M, path, Sh = load_run(args.pkl, args.scale, args.post_y, args.bound, args.seed, with_pred=not args.no_pred)
    if Sh is None and not args.no_pred:
        print("note: this run has no stored costates (Pc); predicted path and x_ref window are not drawn. "
              "Runs recorded after the Pc change (e.g. results_disturb_viz.pkl) have them.")
    last = e["U"].shape[1] - 1
    if args.frame:
        frames = [int(s) for s in args.frame.split(",")]
    elif args.t:
        frames = [int(round(float(s) / P.dt_apply)) for s in args.t.split(",")]
    else:
        frames = [last]
    os.makedirs(args.outdir, exist_ok=True)
    for k in frames:
        k = int(np.clip(k, 0, last))
        tag = os.path.splitext(os.path.basename(args.pkl))[0].replace("results_disturb", "").strip("_")
        base = os.path.join(args.outdir, f"disturb{('_' + tag) if tag else ''}_b{args.bound:g}_seed{args.seed}_t{e['tlog'][k]:.2f}")
        snapshot(e, P, M, path, k, base, args.dpi, args.elev, args.azim, args.hud, not args.no_title,
                 png=not args.pdf_only, pdf=not args.png_only, Sh=Sh)


if __name__ == "__main__":
    main()

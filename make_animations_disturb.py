"""
make_animations_disturb.py -- 3-D animation of one disturbance run (mpc_testbed_disturb.py) with
per-tick arrows for the model velocity f(x,u) and the disturbance d at the vehicle.

Scene, camera, limits, obstacle style and comet trail follow make_animations.py.  At every control
tick (20 ms) two arrows start at the vehicle position:
    blue   f(x,u) position part = (v cos th, v sin th, vz)   -- where the clean model is taking it
    red    d position part      = (d_x, d_y, d_z)             -- the disturbance velocity injected
           (mean of the 4 fine-step draws of that tick; units m/s drawn at ARROW_SCALE m per m/s)
The acceleration parts of f and d (on v, omega, vz) are printed in the HUD instead of drawn.
The solver's predicted trajectory is not stored by the disturbance runner and is therefore not shown.

USAGE
    python make_animations_disturb.py --pkl results_disturb_posty-0.1.pkl --bound 30 --seed 0 \
           --post-y -0.1 --out figs/disturb/anim_b30_seed0_success.mp4
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:                                                     # noqa: BLE001
    pass
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.animation import FFMpegWriter                          # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

import mpc_core as C                                                   # noqa: E402
import mpc_testbed as T                                                # noqa: E402  (piecewise mobs_center)
import mpc_tuned_params                                                # noqa: E402
from make_animations import XL, YL, ZL, OBSCOL, OBSALPHA, AXLBL, ellipsoid   # noqa: E402

GREEN = (.10, .55, .20)
FCOL = (0.0, .35, .85)
DCOL = (.85, .15, .15)
ARROW_SCALE = 1.0                                                      # metres of arrow per m/s


def build_scene(post_y):
    P = mpc_tuned_params.apply(C.make_P())
    if post_y != 0.0:
        P.mobs.c0[1, 4] = float(post_y)
    M = C.model_scn(P)
    path = C.plan_path(P.p0, P.pgoal, P.obs_plan, P)
    return P, M, path


def render(e, P, M, path, outfile, label):
    X, U, D, tl = e["X"], e["U"], e["dist"], e["tlog"]
    nf = U.shape[1]
    nap = P.napply
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([-0.02, -0.04, 0.90, 1.06], projection="3d")
    writer = FFMpegWriter(fps=int(round(1 / P.dt_apply)), bitrate=6000)
    print(f"rendering {outfile}  ({nf} frames)")
    with writer.saving(fig, outfile, dpi=100):
        for k in range(nf):
            t = tl[k]
            ax.cla()
            ax.set_xlim(XL); ax.set_ylim(YL); ax.set_zlim(ZL)
            ax.set_box_aspect((15 / 2.2, 5.8, 2.6))
            ax.view_init(elev=24, azim=-62)
            ax.set_xlabel("$p_x$", fontsize=AXLBL); ax.set_ylabel("$p_y$", fontsize=AXLBL)
            ax.set_zlabel("$p_z$", fontsize=AXLBL)
            ax.grid(True, alpha=0.25)
            for i in range(M.obs.center.shape[1]):
                oc = M.obs.center[:, i]
                ellipsoid(ax, oc, M.obs.ax[i], M.obs.ay[i], M.obs.az[i], OBSCOL, OBSALPHA)
            mc = C.mobs_center(M.mobs, t)                              # true (piecewise, tstop-aware) positions
            for kk in range(mc.shape[1]):
                ellipsoid(ax, mc[:, kk], M.mobs.ax[kk], M.mobs.ay[kk], M.mobs.az[kk], OBSCOL, OBSALPHA)
            ax.plot(path.pts[0], path.pts[1], np.full(path.pts.shape[1], ZL[0]), "-", color=(.75, .75, .78), lw=1.0)
            ax.plot(path.pts[0], path.pts[1], path.pts[2], ":", color=(.15, .15, .15), lw=1.4)
            ax.plot([P.p0[0]], [P.p0[1]], [P.p0[2]], "o", mfc=(.2, .75, .3), mec="k", ms=8)
            ax.plot([P.pgoal[0]], [P.pgoal[1]], [P.pgoal[2]], "*", mfc=(1, .85, .1), mec="k", ms=16)

            ie = k * nap + 1
            Xt = X[M.posidx][:, :ie]
            ax.plot(Xt[0], Xt[1], Xt[2], "-", color=GREEN, lw=2.2)
            it = max(0, ie - 20 * nap)
            ax.plot(Xt[0, it:], Xt[1, it:], Xt[2, it:], "-", color=GREEN, lw=3.6)
            x = X[:, k * nap]
            u = U[:, k]
            d = D[:, k * nap:(k + 1) * nap].mean(axis=1) if D.shape[1] >= (k + 1) * nap else np.zeros(7)
            pos = x[M.posidx]
            inbox = (XL[0] <= pos[0] <= XL[1]) and (YL[0] <= pos[1] <= YL[1]) and (ZL[0] <= pos[2] <= ZL[1])
            if inbox:
                ax.plot([pos[0]], [pos[1]], [pos[2]], "o", mfc=GREEN, mec="w", ms=9)
                fpos = np.array([x[4] * np.cos(x[3]), x[4] * np.sin(x[3]), x[6]])
                dpos = d[:3]
                ax.quiver(pos[0], pos[1], pos[2], *(ARROW_SCALE * fpos), color=FCOL, lw=2.2,
                          arrow_length_ratio=0.25)
                ax.quiver(pos[0], pos[1], pos[2], *(ARROW_SCALE * dpos), color=DCOL, lw=2.2,
                          arrow_length_ratio=0.25)
            else:
                ax.text2D(0.03, 0.80, "vehicle outside the view", transform=ax.transAxes, fontsize=11,
                          color=DCOL, fontweight="bold")
            hud = (f"t = {t:5.2f} s\n"
                   f"|f_pos| = {np.linalg.norm([x[4] * np.cos(x[3]), x[4] * np.sin(x[3]), x[6]]):4.2f} m/s   "
                   f"|d_pos| = {np.linalg.norm(d[:3]):4.2f} m/s\n"
                   f"u = [{u[0]:+.2f} {u[1]:+.2f} {u[2]:+.2f}]   d_acc = [{d[4]:+.2f} {d[5]:+.2f} {d[6]:+.2f}]   d_th = {d[3]:+.2f}\n"
                   f"iters {int(e['iters'][k])}   kick {int(e['nkick'][k])}   |res| {e['nres'][k]:.1e}")
            ax.text2D(0.03, 0.97, hud, transform=ax.transAxes, fontsize=10, va="top", family="monospace")
            ax.set_title(label, fontsize=13, fontweight="bold")
            handles = [Line2D([], [], color=GREEN, lw=2.5, label="Hopf-Lax-MPC trajectory"),
                       Line2D([], [], color=FCOL, lw=2.5, label="f(x,u): model velocity (m/s, x%g)" % ARROW_SCALE),
                       Line2D([], [], color=DCOL, lw=2.5, label="d: disturbance velocity (m/s, x%g)" % ARROW_SCALE),
                       Line2D([], [], color=(.15, .15, .15), ls=":", lw=1.4, label="RRT reference")]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 0.95), fontsize=9, frameon=False)
            writer.grab_frame()
            if (k + 1) % 100 == 0:
                print(f"   frame {k + 1}/{nf}")
    plt.close(fig)
    print(f"   done -> {outfile}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="results_disturb_posty-0.1.pkl")
    ap.add_argument("--scale", default="practical")
    ap.add_argument("--post-y", type=float, default=-0.1)
    ap.add_argument("--bound", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.pkl, "rb") as f:
        runs = pickle.load(f)["runs"]
    e = runs[(args.scale, args.post_y, args.bound, args.seed)]
    P, M, path = build_scene(args.post_y)
    out = args.out or f"figs/disturb/anim_b{args.bound:g}_seed{args.seed}.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    outcome = f"reached at {e['t_reach']:.2f} s" if e["reached"] else "NOT reached"
    label = (f"Hopf-Lax-MPC, N={e['N']}, 20 ms budget, disturbance b={args.bound:g} (seed {args.seed}): "
             f"{outcome}, J_total {e['Jtrue']:.1f}")
    render(e, P, M, path, out, label)


if __name__ == "__main__":
    main()

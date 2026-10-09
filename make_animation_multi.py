"""
make_animation_multi.py -- top-down animation of a multi-agent (stacked, decoupled) Hopf-Lax-MPC run.

Reads results_multi.pkl (written by mpc_testbed_multi.py) and renders, for one K, the frozen scene,
each agent's RRT reference, its closed-loop trail, its current predicted horizon (re-rolled from the
stored costate with the same stacked shooting object) and a HUD with time, arrivals and the solver's
iterations / solve time of the current cycle.  Real-time playback: one frame per MPC cycle (20 ms).

USAGE
    python make_animation_multi.py --K 10 [--pkl results_multi.pkl] [--out figs/multi/K10/anim.mp4] [--stride 1]
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

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
from matplotlib.patches import Ellipse                                 # noqa: E402

import mpc_multi as MU                                                 # noqa: E402
import mpc_testbed_multi as TM                                         # noqa: E402

RED = (.85, .18, .18)
MOBS_NAMES = ["wall", "blocker", "door", "shoulder", "fork post", "stage-2", "companion"]


def predictions(e):
    """Re-roll the predicted horizon of every cycle from the stored stacked costates: (ncyc, K, 2, N+1)."""
    K, N = e["K"], e["N"]
    P = TM.make_env(e["meta"]["lm_maxit"])
    P, M, JX, MK = MU.make_multi_scene(P, e["K_layout"], solo=e["solo"])
    Sh = MU.build_ss_multi(M, K, N, P.dt, JX, napply=P.napply)
    X, Pc, tl = e["X"], e["Pc"], e["tlog"]
    ncyc = Pc.shape[2]
    out = np.zeros((ncyc, K, 3, N + 1))
    t0 = time.perf_counter()
    for c in range(ncyc):
        x = X[:, :, c].reshape(-1)
        Xr = MU.ref_window_multi(x, N, MK, M, P, 2, tl[c])
        Xp = np.asarray(Sh.roll_pred(Pc[:, :, c].reshape(-1), x, Xr))   # (K, n, N+1)
        out[c] = Xp[:, :3, :]
    print(f"re-rolled {ncyc} predicted horizons for K={K} in {time.perf_counter() - t0:.1f} s")
    return out


def render3d(e, pred, out, stride, snapshot=None, snapshot_dpi=450):
    """3-D view in the style of make_animations.py (same box aspect, camera, clipped ellipsoids).
    snapshot: None -> write the video; 'last' or a cycle index -> write that single frame as a PNG at
    snapshot_dpi (out should then be a .png path)."""
    from make_animations import ellipsoid, OBSCOL, OBSALPHA, XL, YL, ZL
    K, N = e["K"], e["N"]
    X, dt_apply = e["X"], e["meta"]["dt_apply"]
    ncyc = X.shape[2] - 1
    cols = [plt.get_cmap("tab10")(k % 10) for k in range(K)]
    if snapshot is not None:
        # publication still: the snapshot_frame.py style (Arial, p_x [m] labels, MATLAB `box on` frame)
        import snapshot_frame as SF                                    # sets the Arial rcParams on import
        fig = plt.figure(figsize=SF.FIGSIZE, dpi=100)
        ax = fig.add_axes([0.0, -0.04, 0.84, 1.06], projection="3d")   # narrower than SF.AXES_RECT: room for p_z
        SF.setup_axes(ax, SF.ELEV, SF.AZIM)
    else:
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(XL); ax.set_ylim(YL); ax.set_zlim(ZL)
        ax.set_box_aspect((15 / 2.2, 5.8, 2.6))
        ax.view_init(elev=24, azim=-62)
        ax.set_xlabel("$p_x$", fontsize=13); ax.set_ylabel("$p_y$", fontsize=13); ax.set_zlabel("$p_z$", fontsize=13)
    so, mo = e["obs"], e["mobs"]
    for i in range(so["center"].shape[1]):
        ellipsoid(ax, so["center"][:, i], so["ax"][i] * 1.25, so["ay"][i] * 1.25, so.get("az", np.full(len(so["ax"]), 0.75))[i] * 1.25, OBSCOL, OBSALPHA)
    for i in range(mo["c0"].shape[1]):
        ellipsoid(ax, mo["c0"][:, i], mo["ax"][i] * 1.25, mo["ay"][i] * 1.25, 5.0, OBSCOL, OBSALPHA)
    trails, heads, preds = [], [], []
    for k in range(K):
        p = e["paths"][k]
        ax.plot(p["pts"][0], p["pts"][1], p["pts"][2], "--", color=cols[k], lw=0.7, alpha=0.4)
        ax.plot([e["goals"][k][0]], [e["goals"][k][1]], [e["goals"][k][2]], "*", color=cols[k], ms=11)
        trails.append(ax.plot([], [], [], "-", color=cols[k], lw=1.6)[0])
        preds.append(ax.plot([], [], [], "-", color=cols[k], lw=1.0, alpha=0.35)[0])
        heads.append(ax.plot([], [], [], "o", color=cols[k], ms=7, mec="k", mew=0.6)[0])
    hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    fig.suptitle(f"K={K} decoupled agents (state dim {7 * K}) -- stacked Hopf-Lax-MPC, N={N}, no budget, obstacles frozen",
                 fontsize=11)
    def draw(c):
        t = c * dt_apply
        for k in range(K):
            trails[k].set_data_3d(X[k, 0, :c + 1], X[k, 1, :c + 1], X[k, 2, :c + 1])
            heads[k].set_data_3d([X[k, 0, c]], [X[k, 1, c]], [X[k, 2, c]])
            if pred is not None and c < pred.shape[0]:
                preds[k].set_data_3d(pred[c, k, 0, ::4], pred[c, k, 1, ::4], pred[c, k, 2, ::4])
            elif pred is not None:
                preds[k].set_data_3d([], [], [])
        arrived = int(np.sum(e["t_reach"] <= t + 1e-9))
        cc = min(c, len(e["iters"]) - 1)
        hud.set_text(f"t = {t:5.2f} s   arrived {arrived}/{K}   cycle {c:4d}: {e['iters'][cc]} iters, "
                     f"{1e3 * e['tsolve'][cc]:6.1f} ms solve")

    if snapshot is not None:
        c = ncyc if snapshot == "last" else int(snapshot)
        draw(c)
        hud.set_visible(False)                                          # paper still: no HUD / suptitle,
        fig.suptitle("")                                                # just the time, as snapshot_frame.py
        ax.set_title(f"t = {c * dt_apply:5.2f} s", fontsize=SF.TITLE_FONTSIZE, fontweight="bold")
        # matplotlib places the 3-D z label inside the plot for this camera, where it is hidden:
        # draw it explicitly to the right of the z tick numbers instead.
        from mpl_toolkits.mplot3d import proj3d
        fig.canvas.draw()
        ax.set_zlabel("")
        # screen position of the right vertical box edge (where the z ticks sit for this camera)
        xs, ys = [], []
        for z in (ZL[0], ZL[1]):
            px, py, _ = proj3d.proj_transform(XL[1], YL[1], z, ax.get_proj())
            sx, sy = ax.transData.transform((px, py))
            xs.append(sx); ys.append(sy)
        fw, fh = fig.bbox.width, fig.bbox.height
        fig.text((max(xs) + 62) / fw, 0.5 * (ys[0] + ys[1]) / fh, "$p_z$ [m]", rotation=90, va="center",
                 ha="left", fontsize=SF.AXLBL)
        base = os.path.splitext(out)[0]
        fig.savefig(base + ".png", dpi=snapshot_dpi, bbox_inches="tight")
        fig.savefig(base + ".pdf", dpi=snapshot_dpi, bbox_inches="tight")   # surfaces rasterised at dpi,
        print(f"snapshot of cycle {c} (t = {c * dt_apply:.2f} s) -> {base}.png/.pdf at dpi {snapshot_dpi}")  # lines/text vector
        return 0
    fps = int(round(1.0 / (dt_apply * stride)))
    writer = FFMpegWriter(fps=fps, bitrate=6000)
    frames = list(range(0, ncyc + 1, stride))
    t0 = time.perf_counter()
    with writer.saving(fig, out, dpi=100):
        for fi, c in enumerate(frames):
            draw(c)
            writer.grab_frame()
            if fi % 100 == 0:
                print(f"   frame {fi}/{len(frames)}", flush=True)
    print(f"done -> {out}  ({len(frames)} frames at {fps} fps, {time.perf_counter() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--pkl", default="results_multi.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1, help="render every <stride>-th cycle")
    ap.add_argument("--no-pred", action="store_true", help="skip the predicted horizons")
    ap.add_argument("--three-d", action="store_true", help="3-D view (make_animations.py style) -> anim3d.mp4")
    ap.add_argument("--snapshot", default=None, help="3-D only: 'last' or a cycle index -> single PNG frame instead of a video")
    ap.add_argument("--snapshot-dpi", type=int, default=450)
    args = ap.parse_args()
    D = pickle.load(open(args.pkl, "rb"))
    e = D["runs"][(args.K, 220)] if (args.K, 220) in D["runs"] else D["runs"][[k for k in D["runs"] if k[0] == args.K and len(k) == 2][0]]
    K, N = e["K"], e["N"]
    if args.out:
        out = args.out
    elif args.three_d and args.snapshot is not None:
        out = f"figs/multi/K{K}/anim3d_{'last' if args.snapshot == 'last' else 'c' + str(args.snapshot)}.png"
    else:
        out = f"figs/multi/K{K}/{'anim3d' if args.three_d else 'anim'}.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X, tl, dt_apply = e["X"], e["tlog"], e["meta"]["dt_apply"]
    ncyc = X.shape[2] - 1
    pred = None if args.no_pred else predictions(e)
    if args.three_d:
        return render3d(e, pred, out, args.stride, snapshot=args.snapshot, snapshot_dpi=args.snapshot_dpi)
    cols = [plt.get_cmap("tab10")(k % 10) for k in range(K)]

    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=110)
    so, mo = e["obs"], e["mobs"]
    for i in range(so["center"].shape[1]):
        ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=RED, alpha=.12))
        ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=RED, alpha=.45))
    for i in range(mo["c0"].shape[1]):
        c = mo["c0"][:2, i]
        if abs(c[1]) > 4:
            continue
        ax.add_patch(Ellipse(c, 2.5 * mo["ax"][i], 2.5 * mo["ay"][i], color=RED, alpha=.12))
        ax.add_patch(Ellipse(c, 2 * mo["ax"][i], 2 * mo["ay"][i], color=RED, alpha=.45))
        ax.text(c[0], c[1] + mo["ay"][i] + 0.1, MOBS_NAMES[i] if i < len(MOBS_NAMES) else str(i), fontsize=6.5,
                ha="center", va="bottom", color=(.55, .1, .1))
    trails, heads, preds = [], [], []
    for k in range(K):
        p = e["paths"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.7, alpha=0.4)
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=5, alpha=0.6)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=11)
        ax.add_patch(plt.Circle(e["goals"][k][:2], e["meta"]["reachtol"], fill=False, ls=":", color=cols[k], lw=0.7))
        trails.append(ax.plot([], [], "-", color=cols[k], lw=1.6)[0])
        preds.append(ax.plot([], [], "-", color=cols[k], lw=1.0, alpha=0.35)[0])
        heads.append(ax.plot([], [], "o", color=cols[k], ms=7, mec="k", mew=0.6)[0])
    ax.set_xlim(-0.8, 20.8); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"K={K} decoupled agents (state dim {7 * K}) -- stacked Hopf-Lax-MPC, N={N}, no budget, obstacles frozen"
                 f"   (trail = closed loop, faint = predicted horizon, dashed = RRT reference)", loc="left", fontsize=9)
    hud = ax.text(0.01, 0.97, "", transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    fps = int(round(1.0 / (dt_apply * args.stride)))
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    frames = list(range(0, ncyc + 1, args.stride))
    t0 = time.perf_counter()
    with writer.saving(fig, out, dpi=110):
        for fi, c in enumerate(frames):
            t = c * dt_apply
            for k in range(K):
                trails[k].set_data(X[k, 0, :c + 1], X[k, 1, :c + 1])
                heads[k].set_data([X[k, 0, c]], [X[k, 1, c]])
                if pred is not None and c < pred.shape[0]:
                    preds[k].set_data(pred[c, k, 0, ::4], pred[c, k, 1, ::4])
            arrived = int(np.sum(e["t_reach"] <= t + 1e-9))
            cc = min(c, len(e["iters"]) - 1)
            hud.set_text(f"t = {t:5.2f} s   arrived {arrived}/{K}   cycle {c:4d}: {e['iters'][cc]} iters, "
                         f"{1e3 * e['tsolve'][cc]:6.1f} ms solve ({1e3 * e['tsolve'][cc] / max(e['iters'][cc], 1):5.1f} ms/iter)")
            writer.grab_frame()
            if fi % 100 == 0:
                print(f"   frame {fi}/{len(frames)}", flush=True)
    print(f"done -> {out}  ({len(frames)} frames at {fps} fps, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()

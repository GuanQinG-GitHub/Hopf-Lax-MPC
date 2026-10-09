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
    out = np.zeros((ncyc, K, 2, N + 1))
    t0 = time.perf_counter()
    for c in range(ncyc):
        x = X[:, :, c].reshape(-1)
        Xr = MU.ref_window_multi(x, N, MK, M, P, 2, tl[c])
        Xp = np.asarray(Sh.roll_pred(Pc[:, :, c].reshape(-1), x, Xr))   # (K, n, N+1)
        out[c] = Xp[:, :2, :]
    print(f"re-rolled {ncyc} predicted horizons for K={K} in {time.perf_counter() - t0:.1f} s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--pkl", default="results_multi.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1, help="render every <stride>-th cycle")
    ap.add_argument("--no-pred", action="store_true", help="skip the predicted horizons")
    args = ap.parse_args()
    D = pickle.load(open(args.pkl, "rb"))
    e = D["runs"][(args.K, 220)] if (args.K, 220) in D["runs"] else D["runs"][[k for k in D["runs"] if k[0] == args.K and len(k) == 2][0]]
    K, N = e["K"], e["N"]
    out = args.out or f"figs/multi/K{K}/anim.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X, tl, dt_apply = e["X"], e["tlog"], e["meta"]["dt_apply"]
    ncyc = X.shape[2] - 1
    pred = None if args.no_pred else predictions(e)
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

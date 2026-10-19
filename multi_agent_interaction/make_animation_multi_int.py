"""
make_animation_multi_int.py -- top-down animation of an interacting multi-agent Hopf-Lax-MPC run.

Copy of multi_agent/make_animation_multi.py (top-down view) for results_multi_int.pkl: frozen scene, RRT
references, closed-loop trails, the predicted horizon of every cycle (re-rolled from the stored costate
with the same COUPLED shooting object), plus a safety disc of radius Rc/2 around each agent (two discs
touch exactly at distance Rc; a disc turns red while its agent is closer than Rc to another one) and a
HUD with time, arrivals, the current minimum pairwise distance and the solver's iterations / solve time.
Real-time playback: one frame per MPC cycle (20 ms).

USAGE
    python make_animation_multi_int.py --K 2 [--Rc 0.8 --Wc 500 --wc 0.1] [--pkl results_multi_int.pkl]
                                       [--out figs/K2_Rc0.8_Wc500/anim.mp4] [--stride 1] [--no-pred]
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

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ("core", "closed_loop", "N_sweep", "multi_agent")]
import mpc_multi as MU                                                 # noqa: E402
import mpc_testbed_multi as TM                                         # noqa: E402
import mpc_multi_int as MI                                             # noqa: E402
import plot_multi as PM                                                # noqa: E402  (draw_scene)
from plot_multi_int import run_dir                                     # noqa: E402


def predictions(e):
    """Re-roll the predicted horizon of every cycle from the stored costates: (ncyc, K, 3, N+1)."""
    K, N = e["K"], e["N"]
    P = TM.make_env(e["meta"]["lm_maxit"])
    P, M, JX, MK = MU.make_multi_scene(P, K)
    c = e["col"]
    Sh = MI.build_ss_int(M, K, N, P.dt, JX, col=MI.make_col(c["Rc"], c["Wc"], c["wc"], c["eps"]), napply=P.napply)
    X, Pc, tl = e["X"], e["Pc"], e["tlog"]
    ncyc = Pc.shape[2]
    out = np.zeros((ncyc, K, 3, N + 1))
    t0 = time.perf_counter()
    for i in range(ncyc):
        x = X[:, :, i].reshape(-1)
        Xr = MU.ref_window_multi(x, N, MK, M, P, 2, tl[i])
        out[i] = np.asarray(Sh.roll_pred(Pc[:, :, i].reshape(-1), x, Xr))[:, :3, :]
    print(f"re-rolled {ncyc} predicted horizons for K={K} in {time.perf_counter() - t0:.1f} s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--N", type=int, default=220)
    ap.add_argument("--Rc", type=float, default=MI.RC_DEFAULT)
    ap.add_argument("--Wc", type=float, default=MI.WC_DEFAULT)
    ap.add_argument("--wc", type=float, default=MI.WSH_DEFAULT)
    ap.add_argument("--pkl", default="results_multi_int.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1, help="render every <stride>-th cycle")
    ap.add_argument("--no-pred", action="store_true", help="skip the predicted horizons")
    args = ap.parse_args()
    D = pickle.load(open(args.pkl, "rb"))
    e = D["runs"][(args.K, args.N, args.Rc, args.Wc, args.wc)]
    K, N, Rc = e["K"], e["N"], e["col"]["Rc"]
    out = args.out or os.path.join("figs", run_dir(e), "anim.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X, dt_apply = e["X"], e["meta"]["dt_apply"]
    ncyc = X.shape[2] - 1
    pred = None if args.no_pred else predictions(e)
    cols = PM.agent_colors(K)

    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=110)
    PM.draw_scene(ax, e)
    trails, heads, preds, discs = [], [], [], []
    for k in range(K):
        p = e["paths"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.7, alpha=0.4)
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=5, alpha=0.6)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=11)
        trails.append(ax.plot([], [], "-", color=cols[k], lw=1.6)[0])
        preds.append(ax.plot([], [], "-", color=cols[k], lw=1.0, alpha=0.35)[0])
        heads.append(ax.plot([], [], "o", color=cols[k], ms=6, mec="k", mew=0.6)[0])
        discs.append(ax.add_patch(plt.Circle((0, 0), Rc / 2, fill=False, color=cols[k], lw=1.1)))
    ax.set_xlim(-0.8, 20.8); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"K={K} interacting agents (state dim {7 * K}) -- coupled Hopf-Lax-MPC, N={N}, no budget, pair penalty "
                 f"Rc={Rc:g} m, Wc={e['col']['Wc']:g}   (circles: radius Rc/2, red while closer than Rc; faint = predicted "
                 f"horizon)", loc="left", fontsize=8.5)
    hud = ax.text(0.01, 0.97, "", transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    fps = int(round(1.0 / (dt_apply * args.stride)))
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    frames = list(range(0, ncyc + 1, args.stride))
    t0 = time.perf_counter()
    with writer.saving(fig, out, dpi=110):
        for fi, c in enumerate(frames):
            t = c * dt_apply
            Q = X[:, :3, c]
            D_ = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1) + np.diag(np.full(K, np.inf))
            for k in range(K):
                trails[k].set_data(X[k, 0, :c + 1], X[k, 1, :c + 1])
                heads[k].set_data([X[k, 0, c]], [X[k, 1, c]])
                discs[k].center = (X[k, 0, c], X[k, 1, c])
                discs[k].set_color(PM.RED if D_[k].min() < Rc else cols[k])
                if pred is not None and c < pred.shape[0]:
                    preds[k].set_data(pred[c, k, 0, ::4], pred[c, k, 1, ::4])
            arrived = int(np.sum(e["t_reach"] <= t + 1e-9))
            cc = min(c, len(e["iters"]) - 1)
            hud.set_text(f"t = {t:5.2f} s   arrived {arrived}/{K}   min distance {D_.min():5.2f} m   cycle {c:4d}: "
                         f"{e['iters'][cc]} iters, {1e3 * e['tsolve'][cc]:6.1f} ms solve")
            writer.grab_frame()
            if fi % 100 == 0:
                print(f"   frame {fi}/{len(frames)}", flush=True)
    print(f"done -> {out}  ({len(frames)} frames at {fps} fps, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()

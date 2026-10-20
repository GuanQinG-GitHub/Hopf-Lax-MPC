"""
make_animation_sweep_int.py -- top-down animations of the all-method K-sweep runs (results_sweep_int/*.pkl).

Each video shows (1) the environment layout: frozen obstacles (core + shell), RRT references, starts and goals;
(2) the running closed-loop state: trails, current positions with a safety disc of radius Rc/2 (red while
the agent is closer than Rc to another one), HUD with time, arrivals, minimum distance, iterations and solve
time; (3) each agent's MPC predicted trajectory at that cycle.  Real-time playback: one frame per MPC cycle.

Predicted trajectory per method (only what the recorded data supports):
    Hopf_Lax_MPC, PMP_Shooting, Vanilla_Hopf_Lax   EXACT: the applied costate of the cycle re-rolled through the
                                                   same coupled canonical system (mpc_multi_int.build_ss_int).
    DDP                                            APPROXIMATE: DDP's plan (its control sequence) was not saved,
                                                   only its adjoint costate p0 -> the canonical rollout from p0.
    MPPI_*                                         NONE: the nominal control sequence was not recorded.

USAGE
    python make_animation_sweep_int.py                      # every run in results_sweep_int -> figs/sweep/anim/
    python make_animation_sweep_int.py --only DDP_K4,Hopf_Lax_MPC_K10 [--stride 2] [--redo]
"""
from __future__ import annotations

import argparse
import glob
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
import plot_multi as PM                                                # noqa: E402
from plot_multi_int import xlims                                       # noqa: E402
from plot_baselines_int import LBL                                     # noqa: E402

PRED_KIND = {"Hopf_Lax_MPC": "exact", "PMP_Shooting": "exact", "Vanilla_Hopf_Lax": "exact", "DDP": "approx"}


def predictions(e):
    """(ncyc, K, 3, N+1) predicted positions, or None (MPPI)."""
    if e["method"] not in PRED_KIND:
        return None
    K, N = e["K"], e["N"]
    P = TM.make_env(e["meta"]["lm_maxit"])
    P, M, JX, MK = MI.make_multi_scene_int(P, K)
    c = e["col"]
    Sh = MI.build_ss_int(M, K, N, P.dt, JX, col=MI.make_col(c["Rc"], c["Wc"], c["wc"], c["eps"]), napply=P.napply)
    X, Pc, tl = e["X"], e["Pc"], e["tlog"]
    ncyc = Pc.shape[2]
    out = np.zeros((ncyc, K, 3, N + 1))
    for i in range(ncyc):
        x = X[:, :, i].reshape(-1)
        Xr = MU.ref_window_multi(x, N, MK, M, P, 2, tl[i])
        out[i] = np.asarray(Sh.roll_pred(Pc[:, :, i].reshape(-1), x, Xr))[:, :3, :]
    return out


def render(e, pred, out, stride=1):
    K, N, Rc = e["K"], e["N"], e["col"]["Rc"]
    m = e["method"]
    X, dt_apply = e["X"], e["meta"]["dt_apply"]
    ncyc = X.shape[2] - 1
    cols = PM.agent_colors(K)
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=110)
    PM.draw_scene(ax, e)
    trails, heads, preds, discs = [], [], [], []
    for k in range(K):
        p = e["paths"][k]
        ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.7, alpha=0.4)
        ax.plot(e["starts"][k][0], e["starts"][k][1], "o", color=cols[k], ms=5, alpha=0.6)
        ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=11)
        trails.append(ax.plot([], [], "-", color=cols[k], lw=1.6)[0])
        preds.append(ax.plot([], [], "-", color=cols[k], lw=1.2, alpha=0.45)[0])
        heads.append(ax.plot([], [], "o", color=cols[k], ms=6, mec="k", mew=0.6)[0])
        discs.append(ax.add_patch(plt.Circle((0, 0), Rc / 2, fill=False, color=cols[k], lw=1.1)))
    ax.set_xlim(*xlims(e)); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    pk = PRED_KIND.get(m)
    ptxt = {"exact": "faint = MPC predicted trajectory", "approx": "faint = predicted trajectory re-rolled from DDP's adjoint "
            "costate (approximation of its plan)", None: "MPC prediction not recorded for MPPI"}[pk]
    ax.set_title(f"{LBL[m]}, K={K} interacting agents (state dim {7 * K}), N={N}, no budget, pair penalty Rc={Rc:g} m. "
                 f"Dashed = RRT reference, circles = radius Rc/2 (red while closer than Rc), {ptxt}",
                 loc="left", fontsize=8.2, wrap=True)
    hud = ax.text(0.01, 0.97, "", transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    fps = int(round(1.0 / (dt_apply * stride)))
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    frames = list(range(0, ncyc + 1, stride))
    with writer.saving(fig, out, dpi=110):
        for c in frames:
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
                elif pred is not None:
                    preds[k].set_data([], [])
            arrived = int(np.sum(e["t_reach"] <= t + 1e-9))
            cc = min(c, len(e["iters"]) - 1)
            dmin = D_.min() if K > 1 else np.inf
            hud.set_text(f"t = {t:5.2f} s   arrived {arrived}/{K}   min distance {dmin:5.2f} m   cycle {c:4d}: "
                         f"{e['iters'][cc]} iters, {1e3 * e['tsolve'][cc]:8.1f} ms solve, {e['status'][cc]}")
            writer.grab_frame()
    plt.close(fig)
    return len(frames), fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results_sweep_int")
    ap.add_argument("--outdir", default="figs/sweep/anim")
    ap.add_argument("--only", default=None, help="comma list of <method>_K<K>")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.dir, "*_K*.pkl")),
                   key=lambda f: (int(f.rsplit("_K", 1)[1][:-4]), os.path.basename(f)))
    if args.only:
        want = set(args.only.split(","))
        files = [f for f in files if os.path.basename(f)[:-4] in want]
    for f in files:
        name = os.path.basename(f)[:-4]
        out = os.path.join(args.outdir, name + ".mp4")
        if os.path.exists(out) and not args.redo:
            print(f"-- {out} exists; skipping", flush=True)
            continue
        t0 = time.perf_counter()
        e = pickle.load(open(f, "rb"))
        pred = predictions(e)
        t1 = time.perf_counter()
        nf, fps = render(e, pred, out, args.stride)
        print(f"{name}: {nf} frames at {fps} fps -> {out}  (predictions {t1 - t0:.0f} s, render "
              f"{time.perf_counter() - t1:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

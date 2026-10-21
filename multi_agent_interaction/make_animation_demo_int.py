"""
make_animation_demo_int.py -- top-down animations of the demo runs (results_demo_int/<layout>/<method>.pkl).

Each video: (1) the course -- static obstacles and the MOVING bodies drawn at their true position of the frame
(the wall slides in and stops on the axis), references, starts and goals; (2) the running closed-loop state:
trails, current positions with safety discs of radius Rc/2 (red while closer than Rc), HUD with time,
arrivals, minimum distance, iterations, solve time and status; (3) each agent's predicted trajectory:
    Hopf_Lax_MPC, PMP_Shooting, Vanilla_Hopf_Lax   exact (applied costate re-rolled through the coupled canonical system)
    DDP_Box                                        approximate (canonical rollout of DDP's adjoint costate p0)
    MPPI_*, Collocation_IPOPT                      not recorded (none shown)
One frame per MPC cycle (real time, 50 fps).

USAGE
    python make_animation_demo_int.py --layout A_column [--only Hopf_Lax_MPC,DDP_Box] [--stride 1] [--redo]
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
from matplotlib.patches import Ellipse                                 # noqa: E402

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ("core", "closed_loop", "N_sweep", "multi_agent")]
import mpc_testbed as T                                                # noqa: E402,F401  piecewise mobs_center
import mpc_multi as MU                                                 # noqa: E402
import mpc_multi_int as MI                                             # noqa: E402
import mpc_demo_int as DM                                              # noqa: E402
import plot_multi as PM                                                # noqa: E402
from plot_demo_int import LBL, xlim_of                                          # noqa: E402

PRED_KIND = {"Hopf_Lax_MPC": "exact", "PMP_Shooting": "exact", "Vanilla_Hopf_Lax": "exact", "DDP_Box": "approx"}
RED = (.85, .18, .18)


def predictions(e):
    if e["method"] not in PRED_KIND:
        return None
    L = DM.get_layout(e["layout"])
    P = DM.make_env(1000, 1500)
    P, M, JX, MK = MI.make_scene_v2(P, L["starts"], L["goals"])
    c = e["col"]
    K, N = e["K"], e["N"]
    Sh = MI.build_ss_int(M, K, N, P.dt, JX, col=MI.make_col(c["Rc"], c["Wc"], c["wc"], c["eps"]), napply=P.napply)
    X, Pc, tl = e["X"], e["Pc"], e["tlog"]
    out = np.zeros((Pc.shape[2], K, 3, N + 1))
    for i in range(Pc.shape[2]):
        x = X[:, :, i].reshape(-1)
        Xr = MU.ref_window_multi(x, N, MK, M, P, 2, tl[i])
        out[i] = np.asarray(Sh.roll_pred(Pc[:, :, i].reshape(-1), x, Xr))[:, :3, :]
    return out


def render(e, pred, out, stride=1):
    K, N, Rc, m = e["K"], e["N"], e["col"]["Rc"], e["method"]
    X, dta = e["X"], e["meta"]["dt_apply"]
    ncyc = X.shape[2] - 1
    cols = PM.agent_colors(K)
    so, mo, mt = e["obs"], e["mobs"], e["mobs_t"]
    fig, ax = plt.subplots(figsize=(13.5, 5.4), dpi=110)
    for i in range(so["center"].shape[1]):                             # static obstacles
        ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=RED, alpha=.10))
        ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=RED, alpha=.35))
    mov = []                                                           # moving bodies (updated per frame)
    for i in range(mo["c0"].shape[1]):
        sh = ax.add_patch(Ellipse(mt[:2, i, 0], 2.5 * mo["ax"][i], 2.5 * mo["ay"][i], color=(.55, .1, .1), alpha=.12))
        co = ax.add_patch(Ellipse(mt[:2, i, 0], 2 * mo["ax"][i], 2 * mo["ay"][i], color=(.55, .1, .1), alpha=.5))
        mov.append((sh, co))
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
    ax.set_xlim(*xlim_of(e)); ax.set_ylim(-3.4, 3.6); ax.set_aspect("equal"); ax.grid(alpha=.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ptxt = {"exact": "faint = MPC predicted trajectory", "approx": "faint = predicted trajectory re-rolled from DDP's adjoint costate "
            "(approximation)", None: "MPC prediction not recorded for this method"}[PRED_KIND.get(m)]
    ax.set_title(f"{LBL[m]} -- layout {e['layout']}, K={K}, N={N}, course {e['meta']['scenario']} (dark = moving bodies), "
                 f"shared stop eps {e['meta']['shared_stop_eps']:g}.  Circles: radius Rc/2 = {Rc / 2:g} m; {ptxt}",
                 loc="left", fontsize=8.0, wrap=True)
    hud = ax.text(0.01, 0.97, "", transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    fps = int(round(1.0 / (dta * stride)))
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    frames = list(range(0, ncyc + 1, stride))
    with writer.saving(fig, out, dpi=110):
        for c in frames:
            t = c * dta
            for i, (sh, co) in enumerate(mov):
                ctr = mt[:2, i, min(c, mt.shape[2] - 1)]
                sh.center = ctr; co.center = ctr
            Q = X[:, :3, c]
            D_ = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1) + np.diag(np.full(K, np.inf))
            for k in range(K):
                trails[k].set_data(X[k, 0, :c + 1], X[k, 1, :c + 1])
                heads[k].set_data([X[k, 0, c]], [X[k, 1, c]])
                discs[k].center = (X[k, 0, c], X[k, 1, c])
                discs[k].set_color(RED if D_[k].min() < Rc else cols[k])
                if pred is not None and c < pred.shape[0]:
                    preds[k].set_data(pred[c, k, 0, ::4], pred[c, k, 1, ::4])
                elif pred is not None:
                    preds[k].set_data([], [])
            cc = min(c, len(e["iters"]) - 1)
            dmin = D_.min() if K > 1 else np.inf
            hud.set_text(f"t = {t:5.2f} s   arrived {int(np.sum(e['t_reach'] <= t + 1e-9))}/{K}   min dist {dmin:5.2f} m   "
                         f"cycle {c:4d}: {e['iters'][cc]:3d} it, {1e3 * e['tsolve'][cc]:7.1f} ms, {e['status'][cc]}")
            writer.grab_frame()
    plt.close(fig)
    return len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--dir", default="results_demo_int")
    ap.add_argument("--outdir", default="figs/demo")
    ap.add_argument("--only", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    d = os.path.join(args.outdir, args.layout, "anim")
    os.makedirs(d, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.dir, args.layout, "*.pkl"))):
        name = os.path.basename(f)[:-4]
        if args.only and name not in args.only.split(","):
            continue
        out = os.path.join(d, name + ".mp4")
        if os.path.exists(out) and not args.redo:
            print(f"-- {out} exists; skipping"); continue
        t0 = time.perf_counter()
        e = pickle.load(open(f, "rb"))
        nf = render(e, predictions(e), out, args.stride)
        print(f"{name}: {nf} frames -> {out} ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

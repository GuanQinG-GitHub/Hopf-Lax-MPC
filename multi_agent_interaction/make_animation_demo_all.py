"""
make_animation_demo_all.py -- ONE video per demo layout with ALL methods side by side (same clock).

Grid of 7 map panels (one per method, results_demo_int/<layout>/<method>.pkl) plus an 8th panel with every
method's solve time per cycle up to the current time (log scale).  Each map panel: static obstacles, the
moving bodies at their true position of the frame, references, goals, trails, current positions with safety
discs (radius Rc/2, red while closer than Rc) and the predicted trajectory where it was recorded (exact for
Hopf-Lax-MPC / PMP / Vanilla, re-rolled from the adjoint costate for Box-DDP, none for MPPI / IPOPT); the
panel title shows arrivals, the iterations and solve time of the current cycle and the cost so far is in the
end-of-run label.  A run that has ended (all agents arrived, or the 30 s cap) is frozen at its last frame.

USAGE
    python make_animation_demo_all.py --layout A_column [--stride 1]      # -> figs/demo/A_column/anim_all_methods.mp4
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

import make_animation_demo_int as AD                                   # noqa: E402  predictions()
import plot_multi as PM                                                # noqa: E402
from plot_demo_int import ORDER, LBL, COL, xlim_of                              # noqa: E402

RED = (.85, .18, .18)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--dir", default="results_demo_int")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    out = args.out or os.path.join("figs", "demo", args.layout, "anim_all_methods.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.perf_counter()
    E = [pickle.load(open(os.path.join(args.dir, args.layout, m + ".pkl"), "rb")) for m in ORDER
         if os.path.exists(os.path.join(args.dir, args.layout, m + ".pkl"))]
    PR = [AD.predictions(e) for e in E]
    print(f"loaded {len(E)} runs + predictions in {time.perf_counter() - t0:.0f} s", flush=True)
    e0 = E[0]
    K, Rc, dta = e0["K"], e0["col"]["Rc"], e0["meta"]["dt_apply"]
    cols = PM.agent_colors(K)
    nfr = max(e["X"].shape[2] for e in E)
    fig, axs = plt.subplots(4, 2, figsize=(16, 12.5), dpi=90)
    axs = axs.ravel()
    panels = []
    for j, e in enumerate(E):
        ax = axs[j]
        so, mo, mt = e["obs"], e["mobs"], e["mobs_t"]
        for i in range(so["center"].shape[1]):
            ax.add_patch(Ellipse(so["center"][:2, i], 2.5 * so["ax"][i], 2.5 * so["ay"][i], color=RED, alpha=.10))
            ax.add_patch(Ellipse(so["center"][:2, i], 2 * so["ax"][i], 2 * so["ay"][i], color=RED, alpha=.35))
        mov = [(ax.add_patch(Ellipse(mt[:2, i, 0], 2.5 * mo["ax"][i], 2.5 * mo["ay"][i], color=(.55, .1, .1), alpha=.12)),
                ax.add_patch(Ellipse(mt[:2, i, 0], 2 * mo["ax"][i], 2 * mo["ay"][i], color=(.55, .1, .1), alpha=.5)))
               for i in range(mo["c0"].shape[1])]
        tr, hd, pr, dc = [], [], [], []
        for k in range(K):
            p = e["paths"][k]
            ax.plot(p["pts"][0], p["pts"][1], "--", color=cols[k], lw=0.6, alpha=0.35)
            ax.plot(e["goals"][k][0], e["goals"][k][1], "*", color=cols[k], ms=9)
            tr.append(ax.plot([], [], "-", color=cols[k], lw=1.4)[0])
            pr.append(ax.plot([], [], "-", color=cols[k], lw=1.1, alpha=0.45)[0])
            hd.append(ax.plot([], [], "o", color=cols[k], ms=5, mec="k", mew=0.5)[0])
            dc.append(ax.add_patch(plt.Circle((0, 0), Rc / 2, fill=False, color=cols[k], lw=0.9)))
        ax.set_xlim(*xlim_of(e)); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal"); ax.grid(alpha=.2)
        ax.tick_params(labelsize=7)
        ttl = ax.set_title("", loc="left", fontsize=9, color=COL[e["method"]], fontweight="bold")
        panels.append(dict(e=e, pred=PR[j], mov=mov, tr=tr, hd=hd, pr=pr, dc=dc, ttl=ttl, ax=ax))
    # 8th panel: solve time per cycle, all methods, up to the current time
    axT = axs[len(E)] if len(E) < 8 else None
    lines = []
    if axT is not None:
        for e in E:
            lines.append(axT.semilogy([], [], "-", color=COL[e["method"]], lw=0.9, label=LBL[e["method"]])[0])
        tmax = max(e["tlog"][-1] for e in E) + 0.5
        lo = min(1e3 * np.min(e["tsolve"][e["tsolve"] > 0]) for e in E) * 0.7
        hi = max(1e3 * np.max(e["tsolve"]) for e in E) * 1.4
        axT.set_xlim(0, tmax); axT.set_ylim(lo, hi); axT.grid(alpha=.25, which="both")
        axT.set_title("solve time per cycle [ms] (log)", loc="left", fontsize=9)
        axT.set_xlabel("t [s]", fontsize=8); axT.tick_params(labelsize=7)
        axT.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    for j in range(len(E) + 1, 8):
        axs[j].axis("off")
    sup = fig.suptitle("", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fps = max(1, int(round(1.0 / (dta * args.stride))))
    writer = FFMpegWriter(fps=fps, bitrate=4000)
    frames = list(range(0, nfr, args.stride))
    t1 = time.perf_counter()
    with writer.saving(fig, out, dpi=90):
        for fi, c in enumerate(frames):
            t = c * dta
            sup.set_text(f"layout {e0['layout']}: K={K} agents, N={e0['N']}, course {e0['meta']['scenario']} (dark = moving "
                         f"bodies), shared stop eps {e0['meta']['shared_stop_eps']:g}  --  t = {t:5.2f} s")
            for P_ in panels:
                e = P_["e"]; X = e["X"]
                cc = min(c, X.shape[2] - 1)
                done = c >= X.shape[2] - 1
                for i, (sh, co) in enumerate(P_["mov"]):
                    ctr = e["mobs_t"][:2, i, min(cc, e["mobs_t"].shape[2] - 1)]
                    sh.center = ctr; co.center = ctr
                Q = X[:, :3, cc]
                D_ = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1) + np.diag(np.full(K, np.inf))
                for k in range(K):
                    P_["tr"][k].set_data(X[k, 0, :cc + 1], X[k, 1, :cc + 1])
                    P_["hd"][k].set_data([X[k, 0, cc]], [X[k, 1, cc]])
                    P_["dc"][k].center = (X[k, 0, cc], X[k, 1, cc])
                    P_["dc"][k].set_color(RED if D_[k].min() < Rc else cols[k])
                    pd = P_["pred"]
                    if pd is not None and cc < pd.shape[0] and not done:
                        P_["pr"][k].set_data(pd[cc, k, 0, ::4], pd[cc, k, 1, ::4])
                    else:
                        P_["pr"][k].set_data([], [])
                ic = min(cc, len(e["iters"]) - 1)
                arr = int(np.sum(e["t_reach"] <= min(t, e["tlog"][-1] + dta) + 1e-9))
                if done:
                    state = (f"ENDED at {e['tlog'][-1] + dta:.2f} s ({'all arrived' if e['all_reached'] else '30 s cap'}), "
                             f"J {e['J_full']:.1f}")
                else:
                    state = f"{e['iters'][ic]} it, {1e3 * e['tsolve'][ic]:.1f} ms"
                P_["ttl"].set_text(f"{LBL[e['method']]}: arrived {arr}/{K} | {state}")
            for ln, e in zip(lines, E):
                n = min(c + 1, len(e["tlog"]))
                ln.set_data(e["tlog"][:n], 1e3 * e["tsolve"][:n])
            writer.grab_frame()
            if fi % 200 == 0:
                print(f"   frame {fi}/{len(frames)}", flush=True)
    plt.close(fig)
    print(f"done -> {out} ({len(frames)} frames at {fps} fps, {time.perf_counter() - t1:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

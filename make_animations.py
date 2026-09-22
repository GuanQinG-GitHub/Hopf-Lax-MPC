"""
make_animations.py -- Python port of make_animations.m, extended to five methods.

Reads a results pickle written by mpc_compare.py / mpc_compare_tuned.py and renders, into figs/:

    anim_all_methods.mp4                     all methods together
    anim_PMP.mp4 / anim_M1.mp4 / ...         one per method, same scene, same camera, same limits

Per frame (= one MPC cycle, so 20 fps plays back in real time) each method shows:
    solid line     travelled path        dashed line  its MPC predicted trajectory over its own horizon
    filled marker  current state         thick band   the slice of the RRT path inside its cost window
MPPI additionally shows a translucent cloud of its top-weighted sampled rollouts -- its most legible
visual signature, and the qualitative twin of the ESS number in the summary table.

Scene: static obstacle, both moving obstacles at their true positions, the full RRT reference, goal.

AXES NOTE: the corridor is 13 long while the manoeuvres are O(0.5) in y and z, so at equal aspect the
z-climb is invisible. x is compressed 2.2:1 against y and z (which stay equal to each other); the corner
key states this.

USAGE
    python make_animations.py [results_python.pkl]
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
# matplotlib does not find an ffmpeg binary on Windows unless one is on PATH; imageio-ffmpeg ships one,
# so point matplotlib at it rather than requiring a system install.
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.animation import FFMpegWriter                          # noqa: E402

XL, YL, ZL = (-1, 21), (-2.9, 2.9), (-0.9, 1.7)
# v11 style: ALL obstacles (static and moving) share ONE colour -- the obs-1 outer-shell red-pink
# -- drawn as the translucent outer region only (no solid inner core).
OBSCOL = (.85, .18, .18)
OBSALPHA = 0.16
AXLBL = 15                                                             # axis-label fontsize


def mobs_color(k):
    return OBSCOL


def disp_name(name, P):
    """Display names for legends/titles (runner names stay as file names)."""
    if name in ("M1_v2", "M1"):
        return "Hopf–Lax-MPC"
    if name == "MPPI":
        return f"MPPI-(K={getattr(P, 'K_mppi', '')})"
    if name.startswith("MPPI-"):
        return f"MPPI-(K={getattr(P, 'K_mppi2', name.split('-')[1])})"
    return name


def mobs_at(M, t):
    """True moving-obstacle centres at time t.  Honours the optional per-body stop times introduced
    by the v6 scenarios (P.mobs.tstop: centre(t) = c0 + vel*min(t, tstop) -- linear until tstop, then
    frozen).  Pickles without tstop render exactly as before."""
    ts = getattr(M.mobs, "tstop", None)
    te = t if ts is None else np.minimum(t, ts)
    return M.mobs.c0 + M.mobs.vel * te


def mcolor(r):
    """Method colours as in the MATLAB figure, with iLQR's mid-grey darkened for contrast."""
    return (.32, .32, .36) if r.name == "iLQR" else tuple(r.color)


def horizon(P, r):
    """Per-runner horizon: v7+ pickles carry it on the runner (r.N -- required once MPPI appears
    twice with different N); older pickles fall back to the name->P lookup."""
    N = getattr(r, "N", 0)
    if N:
        return N
    return {"PMP": P.N_pmp, "iLQR": P.N_ilqr, "DDP": getattr(P, "N_ddp", 0), "Collocation": P.N_coll,
            "M1": P.N_m1, "M1_v2": P.N_m1, "MPPI": P.N_mppi}.get(r.name, 0)


def ellipsoid(ax, c, a, b, cc, color, alpha, n=24):
    """Draw an ellipsoid, CLIPPED to the axes box.

    Unlike MATLAB, matplotlib's 3D backend does not clip surfaces to the axis limits: an obstacle whose
    centre lies outside the box is still drawn, which both wrecks the composition and silently rescales
    the viewer's sense of the scene. Two guards below: skip anything entirely outside the box, and clamp
    the mesh to the limits (which renders a tall barrier as a wall flush with the box face -- exactly how
    it should read).
    """
    if (c[1] - b > YL[1] or c[1] + b < YL[0] or c[0] - a > XL[1] or c[0] + a < XL[0]):
        return None                                                    # entirely outside: don't draw
    u, v = np.mgrid[0:2 * np.pi:n * 1j, 0:np.pi:n * 1j]
    X = np.clip(c[0] + a * np.cos(u) * np.sin(v), *XL)
    Y = np.clip(c[1] + b * np.sin(u) * np.sin(v), *YL)
    Z = np.clip(c[2] + cc * np.cos(v), *ZL)
    return ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, edgecolor="none",
                           antialiased=True, shade=True, rstride=1, cstride=1)


def render(D, sel, outfile, ttl):
    R, P, M, path = D["R"], D["P"], D["M"], D["path"]
    OM = D.get("obs_motion")                                           # explicit motion record (v6+ pickles)
    nm = len(sel)
    nf = max(R[i].res.ncyc for i in sel)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([-0.02, -0.04, 0.90, 1.06], projection="3d")  # 3D axes carry large internal margins
    ax.set_xlim(XL); ax.set_ylim(YL); ax.set_zlim(ZL)
    ax.set_box_aspect((15 / 2.2, 5.8, 2.6))                            # x compressed 2.2:1 vs y,z
    ax.view_init(elev=24, azim=-62)
    ax.set_xlabel("$p_x$", fontsize=AXLBL); ax.set_ylabel("$p_y$", fontsize=AXLBL)
    ax.set_zlabel("$p_z$", fontsize=AXLBL)

    writer = FFMpegWriter(fps=int(round(1 / P.dt_apply)), bitrate=6000)
    print(f"rendering {outfile}  ({nf} frames)")
    with writer.saving(fig, outfile, dpi=100):
        for k in range(nf):
            t = k * P.dt_apply
            ax.cla()
            ax.set_xlim(XL); ax.set_ylim(YL); ax.set_zlim(ZL)
            ax.set_box_aspect((15 / 2.2, 5.8, 2.6))
            ax.view_init(elev=24, azim=-62)
            ax.set_xlabel("$p_x$", fontsize=AXLBL); ax.set_ylabel("$p_y$", fontsize=AXLBL)
            ax.set_zlabel("$p_z$", fontsize=AXLBL)
            ax.grid(True, alpha=0.25)

            # ---- static scene (outer shell region only, one colour for every obstacle) ----
            for i in range(M.obs.center.shape[1]):
                oc = M.obs.center[:, i]
                ellipsoid(ax, oc, M.obs.ax[i], M.obs.ay[i], M.obs.az[i], OBSCOL, OBSALPHA)
            mc = (OM["centers_t"][:, :, min(k, OM["centers_t"].shape[2] - 1)]
                  if OM is not None else mobs_at(M, t))                # true positions (pkl record or law)
            for kk in range(mc.shape[1]):
                ellipsoid(ax, mc[:, kk], M.mobs.ax[kk], M.mobs.ay[kk], M.mobs.az[kk], OBSCOL, OBSALPHA)
            ax.plot(path.pts[0], path.pts[1], np.full(path.pts.shape[1], ZL[0]),
                    "-", color=(.75, .75, .78), lw=1.0)                # ground shadow (depth cue)
            ax.plot(path.pts[0], path.pts[1], path.pts[2], ":", color=(.15, .15, .15), lw=1.4,
                    label="RRT reference")
            ax.plot([P.p0[0]], [P.p0[1]], [P.p0[2]], "o", mfc=(.2, .75, .3), mec="k", ms=8)
            ax.plot([P.pgoal[0]], [P.pgoal[1]], [P.pgoal[2]], "*", mfc=(1, .85, .1), mec="k", ms=16)

            for i in sel:
                r, c = R[i].res, mcolor(R[i])
                kc = min(k, r.ncyc - 1)
                live = k < r.ncyc
                ie = min(kc * P.napply + 1, r.X.shape[1])
                Xt = r.X[M.posidx][:, :ie]
                ax.plot(Xt[0], Xt[1], Xt[2], "-", color=c, lw=2.2, label=disp_name(R[i].name, P))
                it = max(0, ie - 20 * P.napply)
                ax.plot(Xt[0, it:], Xt[1, it:], Xt[2, it:], "-", color=c, lw=3.6)   # comet tail
                xc = r.xlog[M.posidx, kc]
                ax.plot([xc[0]], [xc[1]], [xc[2]], "o", mfc=c, mec="w", ms=9)
                if live:
                    # MPPI's sampled cloud: shows directly whether the samples straddle the obstacle or
                    # have collapsed (the visual form of the ESS diagnostic).
                    if R[i].name.startswith("MPPI") and r.extra[kc] is not None:
                        for cl in r.extra[kc]:
                            ax.plot(cl[0], cl[1], cl[2], "-", color=c, lw=0.5, alpha=0.13)
                    Xp, Xw = r.Xpred[kc], r.Xrefw[kc]
                    cl = tuple(1 - (1 - np.array(c)) * 0.55)
                    ax.plot(Xp[0], Xp[1], Xp[2], "--", color=cl, lw=1.5)
                    ax.plot(Xw[0], Xw[1], Xw[2], "-", color=c, lw=6, alpha=0.30)     # RRT cost window

            ax.set_title(f"{ttl}      t = {t:5.2f} s", fontsize=13, fontweight="bold")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.95), fontsize=9, frameon=False)
            writer.grab_frame()
            if (k + 1) % 50 == 0:
                print(f"   frame {k + 1}/{nf}")
    plt.close(fig)
    print(f"   done -> {outfile}")


# def main():
#     src = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results_python.pkl"
#     with open(src, "rb") as f:
#         D = pickle.load(f)
#     print(f"loaded {src}: {[r.name for r in D['R']]}")
#     render(D, list(range(len(D["R"]))), "figs/anim_all_methods.mp4",
#            " vs ".join(r.name for r in D["R"]))
#     for i, r in enumerate(D["R"]):
#         render(D, [i], f"figs/anim_{r.name}.mp4", r.name)
#     print("\nall animations written to figs/")
#     return 0

def main():
    src = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results_python.pkl"
    with open(src, "rb") as f:
        D = pickle.load(f)
    names = [r.name for r in D["R"]]
    print(f"loaded {src}: {names}")

    # ---- output directory: 3rd positional arg (e.g. figs/v11), default figs/ ----
    import os
    outdir = sys.argv[3] if len(sys.argv) > 3 else "figs"
    os.makedirs(outdir, exist_ok=True)

    # ---- choose what to render: CLI arg (2nd positional) or interactive prompt ----
    choice = sys.argv[2] if len(sys.argv) > 2 else None
    if choice is None:
        print("what to render?")
        print("  0 : ALL (combined video + one per method)")
        print("  c : combined all-methods video only")
        for i, nm in enumerate(names, start=1):
            print(f"  {i} : {nm} only")
        choice = input("choice [0]: ").strip() or "0"

    dnames = [disp_name(n, D["P"]) for n in names]
    if choice == "0" or choice.lower() == "all":
        render(D, list(range(len(names))), f"{outdir}/anim_all_methods.mp4", " vs ".join(dnames))
        for i, r in enumerate(D["R"]):
            render(D, [i], f"{outdir}/anim_{r.name}.mp4", dnames[i])
    elif choice.lower() in ("c", "combined"):
        render(D, list(range(len(names))), f"{outdir}/anim_all_methods.mp4", " vs ".join(dnames))
    else:
        # accept a 1-based number or a method name (case-insensitive)
        if choice.isdigit():
            i = int(choice) - 1
            if not (0 <= i < len(names)):
                raise SystemExit(f"index out of range: {choice} (1..{len(names)})")
        else:
            low = [n.lower() for n in names]
            if choice.lower() not in low:
                raise SystemExit(f"unknown method '{choice}'; available: {names}")
            i = low.index(choice.lower())
        render(D, [i], f"{outdir}/anim_{names[i]}.mp4", dnames[i])

    print("\ndone; output in figs/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

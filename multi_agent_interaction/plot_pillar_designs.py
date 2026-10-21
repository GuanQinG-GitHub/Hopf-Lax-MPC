"""
plot_pillar_designs.py -- design review of the saddle-pillar layouts (geometry only, NO closed loop).

For each layout: the course, the references, the saddles (stopped wall, pillars), the crossings, and the
cruise-speed SCHEDULE s_k(t) = vcr t of every agent at the moment each saddle is reached (the quantity
that decided the C-variant screening: is the agent ALONE at its saddle?).  Prints, per saddle, the nearest
non-lane-mate agent over the approach window and the course clearance (min barrier coordinate rho).

  figs/pillars/<layout>.png   one annotated map per layout
USAGE
    python plot_pillar_designs.py [layouts ...]      (default: the designs of figs/pillars/pillar_designs.html)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Circle, Ellipse                         # noqa: E402

import mpc_demo_int as D                                               # noqa: E402
import mpc_multi_int as MI                                             # noqa: E402
import mpc_core as C                                                   # noqa: E402

OUT = os.path.join("figs", "pillars")
DESIGNS = ["C_cross_lane_K4+P", "D_cross_first_K5", "D_sym_cross_K5", "C_cross_lane_K7+P", "D_cross_first_K7"]
COLS = plt.get_cmap("tab10").colors


def at(p, s):
    return np.vstack([np.interp(np.clip(s, 0, p.len), p.s, p.pts[r]) for r in range(3)])


def course_rho(q, P, t):
    """min barrier coordinate over q (3, m) of every course body (static, moving at t and stopped)."""
    o, mo = P.obs, P.mobs
    nst = o.center.shape[1] - len(getattr(P, "_pillars", []))
    best = np.inf
    for cen, ax, ay, az in [(o.center[:, :nst], o.ax[:nst], o.ay[:nst], o.az[:nst]),
                            (C.mobs_center(mo, t), mo.ax, mo.ay, mo.az), (C.mobs_center(mo, 1e3), mo.ax, mo.ay, mo.az)]:
        d = q[:, :, None] - cen[:, None, :]
        best = min(best, float(np.min(np.sqrt((d[0] / ax) ** 2 + (d[1] / ay) ** 2 + (d[2] / az) ** 2))))
    return best


def saddles(MK, P):
    """[(kind, centre xy, first agent k, arclength of k at the centre, lane mates)] -- wall + pillars."""
    out = []
    wall = C.mobs_center(P.mobs, 1e3)[:2, 0]
    cands = [("wall", wall)] + [("pillar", np.asarray(c)) for c in MK.pillars]
    for kind, c in cands:
        users = []
        for k, p in enumerate(MK.paths):
            d = np.linalg.norm(p.pts[:2] - c[:, None], axis=0)
            if d.min() < 0.06:
                users.append((p.s[int(np.argmin(d))], k))
        if users:
            users.sort(key=lambda u: u[0] / P.vcr)
            out.append((kind, c, users))
    return out


def crossings(MK, P):
    """Pairwise reference crossings (2-D segment intersection, first per pair) with scheduled times."""
    res = []
    for i in range(MK.K):
        for j in range(i + 1, MK.K):
            a, b = MK.paths[i].pts[:2], MK.paths[j].pts[:2]
            d = np.linalg.norm(a[:, ::4, None] - b[:, None, ::4], axis=0)
            ii, jj = np.unravel_index(np.argmin(d), d.shape)
            dth = abs(np.angle(np.exp(1j * (MK.paths[i].th[4 * ii] - MK.paths[j].th[4 * jj]))))
            if d[ii, jj] < 0.1 and dth > np.deg2rad(15.0):
                si, sj = MK.paths[i].s[4 * ii], MK.paths[j].s[4 * jj]
                res.append((i, j, a[:, 4 * ii], si / P.vcr, sj / P.vcr))
    return res


def analyse(name):
    L = D.get_layout(name)
    P = D.make_env(1000, 1500)
    P, M, JX, MK = MI.make_scene_v2(P, L["starts"], L["goals"], scenario=L.get("scenario", MI.SCENARIO_DEFAULT),
                                    pillars=L.get("pillars"), waypoints=L.get("waypoints"))
    P._pillars = MK.pillars
    rows = []
    for kind, c, users in saddles(MK, P):
        for s_k, k in users:
            sw = np.linspace(s_k - D.APPROACH[0], s_k + D.APPROACH[1], 41)
            q, tw = at(MK.paths[k], sw), sw / P.vcr
            u = np.array([np.cos(np.interp(s_k, MK.paths[k].s, MK.paths[k].th)),
                          np.sin(np.interp(s_k, MK.paths[k].s, MK.paths[k].th))])
            near, who = np.inf, None
            for j in range(MK.K):
                if j == k or D._offline(MK.paths[j], c, u) <= D.LANE_TOL:
                    continue
                dj = np.min(np.linalg.norm(at(MK.paths[j], P.vcr * tw)[:2] - q[:2], axis=0))
                if dj < near:
                    near, who = dj, j
            rows.append(dict(kind=kind, c=c, k=k, t=s_k / P.vcr, near=near, who=who,
                             rho=course_rho(q, P, s_k / P.vcr) if kind == "pillar" else np.nan))
    return L, P, M, MK, rows, crossings(MK, P)


def draw(name, ax):
    L, P, M, MK, rows, xs = analyse(name)
    o, nst = M.obs, M.obs.center.shape[1] - len(MK.pillars)
    for i in range(nst):
        ax.add_patch(Ellipse(o.center[:2, i], 2 * o.ax[i], 2 * o.ay[i], color=(0.9, 0.45, 0.45), alpha=0.35, lw=0))
    mc = C.mobs_center(M.mobs, 1e3)
    for i in range(mc.shape[1]):
        if abs(mc[1, i]) < 4 and -14 < mc[0, i] < 25:
            ax.add_patch(Ellipse(mc[:2, i], 2 * M.mobs.ax[i], 2 * M.mobs.ay[i], color=(0.5, 0.1, 0.1), alpha=0.75, lw=0))
    for k, p in enumerate(MK.paths):
        ax.plot(p.pts[0], p.pts[1], color=COLS[k % 10], lw=1.4, zorder=3)
        ax.plot(*MK.starts[k][:2], "o", color=COLS[k % 10], ms=5, zorder=4)
        ax.plot(*MK.goals[k][:2], "*", color=COLS[k % 10], ms=10, zorder=4)
        ax.annotate(str(k), MK.starts[k][:2], xytext=(-9, 3), textcoords="offset points", fontsize=8,
                    color=COLS[k % 10], weight="bold")
    for c in MK.pillars:
        ax.add_patch(Circle(c, MI.PILLAR_R, color="k", zorder=5))
        ax.add_patch(Circle(c, MI.PILLAR_R * (1 + 3 * MI.PILLAR_SHELL), fill=False, ls=":", color="k", zorder=5))
    # schedule snapshot at each saddle's FIRST arrival: every agent's planned position (hollow markers)
    done = set()
    for r in rows:
        key = (r["kind"], tuple(np.round(r["c"], 2)))
        if key in done:
            continue
        done.add(key)
        lab = ("WALL saddle" if r["kind"] == "wall" else "PILLAR saddle") + \
              f"\nagent {r['k']} at t={r['t']:.1f}s\nnearest off-line agent {r['near']:.1f} m"
        dy = 1.35 if r["c"][1] >= 0 else -1.35
        ax.annotate(lab, r["c"], xytext=(r["c"][0], np.clip(r["c"][1] + dy, -3.6, 3.6)), fontsize=7, ha="center",
                    va="center", arrowprops=dict(arrowstyle="->", lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.6), zorder=7)
    shown = []
    for i, j, xy, ti, tj in xs:
        if any(np.linalg.norm(xy - s) < 1.0 for s in shown):
            continue
        shown.append(xy)
        ax.plot(*xy, "x", color="k", ms=9, mew=2.2, zorder=6)
        ax.annotate(f"crossing\nt={min(ti, tj):.1f}s", xy, xytext=(xy[0], xy[1] + (1.5 if xy[1] <= 0 else -1.5)),
                    fontsize=7, ha="center", va="center", arrowprops=dict(arrowstyle="->", lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff8e0", ec="0.6", lw=0.6), zorder=7)
    ax.set_aspect("equal")
    ax.set_ylim(-3.9, 3.9)
    xa = [s[0] for s in MK.starts] + [g[0] for g in MK.goals]
    ax.set_xlim(min(xa) - 1, max(xa) + 1)
    ax.grid(alpha=0.25)
    ax.set_title(name, fontsize=10, loc="left")
    return rows, xs, MK, P


def offline_gap(MK, P, tgrid):
    """Scheduled distance of each agent to its nearest OFF-LINE neighbour (lateral offset from the agent's
    heading > LANE_TOL; an in-line lane mate ahead / behind does not break the symmetry)."""
    pos = np.stack([at(p, P.vcr * tgrid)[:2] for p in MK.paths])            # (K, 2, T)
    th = np.stack([np.interp(np.clip(P.vcr * tgrid, 0, p.len), p.s, p.th) for p in MK.paths])
    out = np.full((MK.K, tgrid.size), np.inf)
    for k in range(MK.K):
        u = np.stack([np.cos(th[k]), np.sin(th[k])])
        for j in range(MK.K):
            if j == k:
                continue
            d = pos[j] - pos[k]
            lat = np.abs(u[0] * d[1] - u[1] * d[0])
            dist = np.linalg.norm(d, axis=0)
            out[k] = np.where(lat > D.LANE_TOL, np.minimum(out[k], dist), out[k])
    return out


def draw_gap(ax, rows, xs, MK, P):
    T = max(p.len for p in MK.paths) / P.vcr
    tg = np.linspace(0, T, 1500)
    g = offline_gap(MK, P, tg)
    ax.axhspan(0, 0.8, color=(0.85, 0.2, 0.2), alpha=0.10, lw=0)
    ax.axhspan(0.8, 1.6, color=(0.95, 0.6, 0.2), alpha=0.10, lw=0)
    ax.text(0.2, 0.4, "inside Rc = 0.8 m: strong pair push", fontsize=7, color=(0.6, 0.1, 0.1), va="center")
    ax.text(0.2, 1.2, "0.8-1.6 m: pair-penalty tail", fontsize=7, color=(0.6, 0.35, 0.05), va="center")
    ax.axhline(D.ALONE_D, color="0.4", ls="--", lw=0.8)
    ax.text(T, D.ALONE_D + 0.1, f"alone >= {D.ALONE_D} m (penalty < 1e-6)", fontsize=7, color="0.3", ha="right")
    for k in range(MK.K):
        ax.plot(tg, np.minimum(g[k], 9.5), color=COLS[k % 10], lw=1.3, label=f"agent {k}")
    for r in rows:
        y = min(float(np.interp(r["t"], tg, g[r["k"]])), 9.5)
        ax.plot(r["t"], y, "s" if r["kind"] == "pillar" else "D", color=COLS[r["k"] % 10], mec="k", ms=8, zorder=5)
    for i, j, xy, ti, tj in xs:
        ax.axvline(min(ti, tj), color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_ylim(0, 10); ax.set_xlim(0, T)
    ax.set_xlabel("scheduled time t [s] (cruise 1.5 m/s along the reference)")
    ax.set_ylabel("nearest off-line\nneighbour [m]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=MK.K, loc="upper right")
    ax.set_title("squares = pillar saddle reached, diamond = wall saddle reached, dotted = reference crossings",
                 fontsize=8, loc="left")


def main(argv):
    names = argv[1:] or DESIGNS
    os.makedirs(OUT, exist_ok=True)
    for nm in names:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 7.6), gridspec_kw=dict(height_ratios=[1.25, 1]))
        rows, xs, MK, P = draw(nm, ax)
        ax.text(0.005, 0.02, "o start (number = agent)   * goal   x reference crossing   black disc = pillar "
                "(dotted: barrier shell)   light red = static obstacles   dark red = moving bodies at their stop",
                transform=ax.transAxes, fontsize=7, color="0.3")
        draw_gap(ax2, rows, xs, MK, P)
        fig.tight_layout()
        f = os.path.join(OUT, nm.replace("+P", "_P") + ".png")
        fig.savefig(f, dpi=110)
        plt.close(fig)
        print(f"== {nm} -> {f}")
        print(f"   pillars {np.round(MK.pillars, 2).tolist()}")
        for r in rows:
            print(f"   {r['kind']:6s} ({r['c'][0]:6.2f},{r['c'][1]:5.2f}) agent {r['k']} t {r['t']:5.2f} s | nearest other-lane "
                  f"agent {r['who']} at {r['near']:5.2f} m | course rho {r['rho']:.2f}")
        for i, j, xy, ti, tj in xs:
            print(f"   crossing {i}-{j} at ({xy[0]:6.2f},{xy[1]:5.2f}) t {ti:5.2f} / {tj:5.2f} s (dt {abs(ti - tj):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""
mpc_tuned_params.py -- ALL scenario tuning in one module (flattened from the design-iteration chain
mpc_tuned_params -> _v5 -> _v6 -> _v7 -> _v10; every value below is the FINAL value that chain produced,
verified field-for-field against the original modules on 2026-09-22).

Two entry points, matching the two published results:

    apply_wall_course(P)   the v6 CORRIDOR course -- moving obs-2 wall that STOPS, static lane blocker,
                           closing door, -y shoulder blocker.  This is the scene the open-loop
                           complexity-vs-horizon study reduces to "obs-2 only" (complexity_vs_horizon.py,
                           figs/complexity_vs_N_maxiter.png).
    apply(P)               the v10 UNIFIED-ROUTE course used by the closed-loop testbed (mpc_testbed.py,
                           results_testbed_v10.pkl, figs/anim_all_methods.mp4): the wall course plus a
                           downstream fork (on-lane post, arriving hot companion, stage-2 blocker) and
                           re-calibrated horizons / MPPI pins.

Nothing here touches solver numerics: mpc_core / mpc_solvers stay verified against the MATLAB reference.

---------------------------------------------------------------------------------------------------
GEOMETRY (moving-obstacle bodies, in column order of P.mobs; all half-axes az = 5 (2-D problem))
---------------------------------------------------------------------------------------------------
 col  body                       centre c0            vel          ax    ay    W       w     tstop
  0   obs-2 head-on WALL         (12.0, 0.00, 0.5)  (-0.25,0,0)   0.80  0.20  500     0.25  6.4
  1   +y lane BLOCKER (static)   (11.8, 0.55, 0.5)   0            0.50  0.20  900     0.05  inf
  2   closing DOOR post          (13.8, 3.33, 0.5)  (0,-0.25,0)   0.40  0.80  900     0.05  10.13
  3   -y SHOULDER blocker        (10.4,-0.55, 0.5)   0            0.70  0.30  900*    0.08  inf
 --- apply() adds (v10 fork) ---
  4   fork on-lane post          (16.2, 0.00, 0.5)   0            0.30  0.30  900     0.07  inf
  5   stage-2 +y blocker         (17.4, 0.60, 0.5)   0            0.35  0.30  900     0.08  inf
  6   fork COMPANION (arriving)  (16.2,-10.6, 0.5)  (0,+1.0,0)    0.80  0.40  50000   0.08  10.0
 * apply() raises the shoulder W 900 -> 20000 (deterministic +y wall kick for Hopf_Lax_MPC).

Per-body stop time semantics (v6): centre_k(t) = c0_k + vel_k * min(t, tstop_k) -- linear until tstop,
frozen after.  mpc_core.mobs_center is linear; the testbed installs the piecewise replacement at import.

Design rationale (condensed from the original module headers):
  * WALL start 12.0 / stop 6.4 s: measured stall onsets PMP 6.44 / Coll 6.50 / DDP 6.58 -> stopping at
    the earliest stall freezes the wall at x = 10.4 so trapped methods are never pushed backward.
  * BLOCKER at (11.8, 0.55): sits in MPPI's measured approach lane (|y| ~ 0.43) and above Hopf_Lax_MPC's
    (|y| ~ 0.28); frozen-obstacle prediction sees it exactly (static) so the reroute is priced early.
  * DOOR at 13.8, closing 0.25 m/s from y0 = 0.80 + 0.22 + 0.25*9.25 = 3.33: gap 0.22 when Hopf_Lax_MPC passes
    (t ~ 9.25), frozen shut at t = (3.33-0.80)/0.25 = 10.13 (no endless chasing of ejected methods;
    the original v6 module comment said 9.92 -- its formula, kept here, gives 10.13).
  * SHOULDER blocker overlapping the wall's -y shoulder (core from x ~ 9.7): prices the -y dodge AT the
    saddle split for every method; thin shell -> saturated-zero on axis, so PMP/collocation traps and the
    symmetric-saddle certification are untouched to machine precision.
  * FORK (v10): every method dodges the wall +y; the DDP trap is the downstream fork.  The companion
    ARRIVES from -y and freezes at (16.2, -0.60) at t = 10.0 -- absent at DDP's fork commitment
    (~9.3-9.6 s, commits -y blind), present and hot (W = 5e4, thick ax .80 ay .40 so it cannot be
    punched through) for Hopf_Lax_MPC's fork kick (~10.3 s) -> deterministic +y slalom.  Stage-2 blocker denies
    DDP's late wide +y arc.

---------------------------------------------------------------------------------------------------
HORIZONS (20 ms/cycle budget; dt = 0.005)
---------------------------------------------------------------------------------------------------
                    wall course        unified-route course (apply)
  PMP      N_pmp_shooting        220                 260   largest clean-trap horizon (cut 3.9%; erratic >= 290)
  Hopf_Lax_MPC    N_hopf_lax_mpc         180                 220   largest horizon with p99 inside 20 ms (fair-N: PMP is
                                                  cheaper per iteration -> gets the longer horizon)
  iLQR     N_ilqr       200                 200   (not run in the final testbeds)
  DDP      N_ddp        150                 300   J-optimal on its own closed-loop sweep
  Coll     N_collocation_ipopt        80                  80
  MPPI #1  N/K       280 / 16384         340 / 12288   one wave, 19.45 ms measured
  MPPI #2  N2/K2       (none)            175 / 20480   two waves, 19.79 ms measured
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

# ---- wall course (v5/v6 constants) -------------------------------------------------------------------
X2_0 = 12.0            # obs-2 wall start x
T2_STOP = 6.4          # wall stop time [s]  (MEASURED, see header)
XB2, YB = 11.8, 0.55   # +y lane blocker station / lane centre
XD2 = 13.8             # door station
AXP, AYP = 0.40, 0.80  # door-post half-axes
VC = 0.25              # door closing speed [m/s]
GAP_HOPF_LAX_MPC = 0.22          # door gap when Hopf_Lax_MPC passes (t = 9.25 s)
Y0_2 = AYP + GAP_HOPF_LAX_MPC + VC * 9.25                                      # = 3.33: door post start y
T3_STOP = (Y0_2 - AYP) / VC                                          # = 10.13: door frozen shut
WSH = 0.05             # thin shells (blocker, door)
WAMP = 900.0
XS, YS = 10.4, -0.55   # -y shoulder blocker centre
AXS, AYS = 0.70, 0.30  # shoulder-blocker half-axes
WSS = 0.08             # shoulder-blocker shell

# ---- unified-route additions (v10 constants) ---------------------------------------------------------
#                (x,    y,      z,   ax,   ay,   az,  W,       w)
FORK_STATIC = [(16.2, 0.00, 0.5, 0.30, 0.30, 5.0, 900.0, 0.07),      # on-lane post
               (17.4, +0.60, 0.5, 0.35, 0.30, 5.0, 900.0, 0.08)]     # stage-2 +y blocker
COMPANION = (16.2, -10.60, 0.5, 0.80, 0.40, 5.0, 50000.0, 0.08, +1.0, 10.0)  # (..., vy, tstop)
SHOULDER_W = 20000.0                                                 # wall -y shoulder (body 3)


def apply_wall_course(P):
    """v6 corridor course: wall (stops) + blocker + closing door + -y shoulder.  Used by the
    complexity study (reduced there to the wall alone) and as the base of apply()."""
    # horizons (base calibration + v6 re-calibration of DDP and the MPPI pin)
    P.N_pmp_shooting = 220
    P.N_hopf_lax_mpc = 180
    P.N_ilqr = 200
    P.N_ddp = 150
    P.N_collocation_ipopt = 80
    P.N_mppi, P.K_mppi = 280, 16384
    # moving set: column 0 = obs-2 wall from mpc_core.make_P (ax/ay/W/w untouched -- the saddle trap)
    m = P.mobs
    P.mobs = SimpleNamespace(
        c0=np.array([[X2_0, XB2, XD2, XS],
                     [m.c0[1, 0], YB, Y0_2, YS],
                     [m.c0[2, 0], 0.5, 0.5, 0.5]]),
        vel=np.array([[m.vel[0, 0], 0.0, 0.0, 0.0],
                      [m.vel[1, 0], 0.0, -VC, 0.0],
                      [m.vel[2, 0], 0.0, 0.0, 0.0]]),
        ax=np.array([m.ax[0], 0.50, AXP, AXS]),
        ay=np.array([m.ay[0], 0.20, AYP, AYS]),
        az=np.array([m.az[0], 5.0, 5.0, 5.0]),
        W=np.array([m.W[0], WAMP, WAMP, 900.0]),
        w=np.array([m.w[0], WSH, WSH, WSS]),
        eps=m.eps,
    )
    P.mobs.tstop = np.array([T2_STOP, np.inf, T3_STOP, np.inf])
    return P


def apply(P):
    """v10 unified-route course (closed-loop testbed): wall course + fork, re-calibrated horizons."""
    P = apply_wall_course(P)
    P.N_pmp_shooting = 260
    P.N_hopf_lax_mpc = 220
    P.N_ddp = 300
    P.N_mppi, P.K_mppi = 340, 12288
    P.N_mppi2, P.K_mppi2 = 175, 20480
    P.mobs.W[3] = SHOULDER_W
    m = P.mobs
    s = COMPANION
    allb = FORK_STATIC + [s[:8]]
    vel = np.zeros((3, len(allb)))
    vel[1, -1] = s[8]
    P.mobs = SimpleNamespace(
        c0=np.hstack([m.c0] + [np.array([[b[0]], [b[1]], [b[2]]]) for b in allb]),
        vel=np.hstack([m.vel, vel]),
        ax=np.append(m.ax, [b[3] for b in allb]), ay=np.append(m.ay, [b[4] for b in allb]),
        az=np.append(m.az, [b[5] for b in allb]), W=np.append(m.W, [b[6] for b in allb]),
        w=np.append(m.w, [b[7] for b in allb]), eps=m.eps,
    )
    P.mobs.tstop = np.append(m.tstop, [np.inf, np.inf, s[9]])
    return P


# ---- N-sweep course variants (scenario only; no solver is touched) -----------------------------------
#  In apply() the -y shoulder blocker (body 3) leaves a 5 cm gap to the stopped wall, so a -y escape from
#  the wall's saddle ends in a dead-end pocket: at short horizons Hopf_Lax_MPC leaves the saddle (as
#  designed) but parks in the pocket, at a higher cost than PMP_Shooting parked ON the saddle.  The
#  variants below open that pocket; the wall (the symmetric saddle trap) is unchanged.
#
#  WHY THE CURRENT N-SWEEP LAYOUT IS "wall_saddle_bumps_rand" (notes from the layout tuning):
#  * -y shoulder blocker (10.4, -0.55, W 20000): in apply() it forces a deterministic +y kick at N = 220 in
#    the 20 ms testbed, but its 5 cm gap is a -y dead-end pocket where Hopf_Lax_MPC got trapped at short N
#    (a true local minimum, lambda > 0).  Moved to y = -0.90 (~0.4 m gap): the pocket is open.
#  * fork post, stage-2 blocker, companion: built for the 20 ms testbed (plain-DDP traps); in the sweep
#    they only added extra local minima (e.g. in front of the post at N = 125, lambda = +1.8e-4), so they
#    are moved out of the course (post_y / stage2_y far away, companion_tstop = 0).
#  * static z-bump obs-1: these notes were written with obs-1 OFFSET at (3, 0.25); since 2026-10-22 obs-1
#    is ON the axis (3, 0) in mpc_core.make_P (the only mode), so every variant below is mirror-symmetric
#    around it.  With the offset, at short N the single solution drifted -y and every optimizer passed;
#    beyond a turning point (fold / imperfect pitchfork) the on-axis wall saddle existed, PMP_Shooting
#    converged to it and was trapped, while the descent methods slid off (results recorded before the
#    change were made with the offset).
#  * extra on-axis z-bumps: Box-DDP's quadratic model of the stiff barrier fails where the vehicle
#    enters / leaves a barrier tail (climb EDGES), giving damping-exhausted (mu-max) cycles of ~27
#    iterations; its normal cycles are cheaper per iteration than ours.  So many SHORT bumps (more edges)
#    rather than one long ridge (wall_saddle_ridge did not help).  Irregular sizes / spacing avoid an
#    artificial periodic course.  Bumps sit on the axis (no lateral push: the wall saddle is untouched),
#    are taller than cruise altitude (every bump forces a climb) and are kept out of x = 6.6 .. 12.2
#    (the wall's decision zone and passage).  Result (N = 100..175): ~27-32% mu-max DDP_Box cycles,
#    DDP_Box ~1.7-1.8x our mean cycle time; PMP_Shooting trapped at the saddle from N = 150 on.
NSWEEP_VARIANTS = {
    "v10": {},                                                        # = apply(), the published course
    "open_shoulder": dict(shoulder_y=-0.90),                          # blocker moved down: ~0.4 m -y gap
    # + the stage-2 +y blocker (body 5, only there to deny plain DDP a late wide arc in the 20 ms testbed)
    #   moved out of the course: the fork post + arriving companion stay, so +y around the post is open
    "open_shoulder_no_stage2": dict(shoulder_y=-0.90, stage2_y=8.0),
    # + the arriving companion (body 6, a blind trap for plain DDP in the 20 ms testbed) frozen far away at
    #   its start: the fork post on the reference is a second symmetric saddle with BOTH sides open
    "open_post": dict(shoulder_y=-0.90, stage2_y=8.0, companion_tstop=0.0),
    # + the fork post (body 4) moved out too: the stopped wall is the single symmetric saddle of the course
    "wall_saddle": dict(shoulder_y=-0.90, stage2_y=8.0, companion_tstop=0.0, post_y=-8.0),
    # wall_saddle + extra STATIC z-bumps (copies of obs-1's shape, centred ON the axis y = 0, below cruise
    # altitude) away from the wall's decision zone: climbing them keeps the vehicle in a stiff barrier tail
    # (where DDP's quadratic model fails) without any lateral push (the wall saddle is untouched)
    "wall_saddle_bumps3": dict(shoulder_y=-0.90, stage2_y=8.0, companion_tstop=0.0, post_y=-8.0,
                               static_add=[(x, 0.0, 0.0, 0.75, 1.50, 0.75, 200.0, 0.10) for x in (5.5, 15.0, 17.5)]),
    # one bump before the wall + a long ridge after it (x ~ 13 .. 18): longer climbs in a stiff tail
    "wall_saddle_ridge": dict(shoulder_y=-0.90, stage2_y=8.0, companion_tstop=0.0, post_y=-8.0,
                              static_add=[(5.5, 0.0, 0.0, 0.75, 1.50, 0.75, 200.0, 0.10),
                                          (15.5, 0.0, 0.0, 2.50, 1.50, 0.75, 200.0, 0.10)]),
    # short bumps (1 m long) every 1.4 m: DDP's model breaks at the climb EDGES, so more edges, more failures
    "wall_saddle_bumps7": dict(shoulder_y=-0.90, stage2_y=8.0, companion_tstop=0.0, post_y=-8.0,
                               static_add=[(x, 0.0, 0.0, 0.50, 1.50, 0.75, 200.0, 0.10)
                                           for x in (4.6, 6.0, 12.6, 14.0, 15.4, 16.8, 18.2)]),
    # irregular version of bumps7 (hand-picked, deterministic): uneven spacing 1.2-1.8 m, length +-0.45-0.70,
    # width +-1.2-1.8, top 0.62-0.85 (> cruise 0.5: every bump forces a climb), weight 150-300; all on the
    # axis (no lateral push) and none in x = 6.6 .. 12.2 (the wall's decision zone and passage)
    #                     (x,     y,   z,   ax,   ay,   az,   W,     w)
    "wall_saddle_bumps_rand": dict(shoulder_y=-0.90, stage2_y=8.0, companion_tstop=0.0, post_y=-8.0,
                                   static_add=[(4.4,  0.0, 0.0, 0.55, 1.35, 0.70, 180.0, 0.10),
                                               (6.0,  0.0, 0.0, 0.45, 1.70, 0.62, 260.0, 0.10),
                                               (12.7, 0.0, 0.0, 0.60, 1.50, 0.80, 150.0, 0.10),
                                               (14.3, 0.0, 0.0, 0.50, 1.20, 0.66, 300.0, 0.10),
                                               (15.5, 0.0, 0.0, 0.70, 1.80, 0.85, 200.0, 0.10),
                                               (17.3, 0.0, 0.0, 0.45, 1.40, 0.72, 240.0, 0.10),
                                               (18.6, 0.0, 0.0, 0.55, 1.60, 0.64, 170.0, 0.10)]),
}


def apply_variant(P, name="v10"):
    """apply() plus an N-sweep layout variant (see NSWEEP_VARIANTS): shoulder_y / shoulder_W override the
    -y shoulder blocker's centre y and barrier weight; companion_tstop its arrival (stop) time."""
    P = apply(P)
    v = NSWEEP_VARIANTS[name]
    if "shoulder_y" in v:
        P.mobs.c0[1, 3] = v["shoulder_y"]
    if "shoulder_W" in v:
        P.mobs.W[3] = v["shoulder_W"]
    if "static_add" in v:                                             # extra static obstacles (x,y,z,ax,ay,az,W,w)
        o, add = P.obs, np.array(v["static_add"], float).T
        P.obs = SimpleNamespace(center=np.hstack([o.center, add[0:3]]),
                                rphys=np.append(o.rphys, np.minimum(add[3], add[4])),
                                ax=np.append(o.ax, add[3]), ay=np.append(o.ay, add[4]), az=np.append(o.az, add[5]),
                                W=np.append(o.W, add[6]), w=np.append(o.w, add[7]), eps=o.eps)
    if "post_y" in v:
        P.mobs.c0[1, 4] = v["post_y"]
    if "stage2_y" in v:
        P.mobs.c0[1, 5] = v["stage2_y"]
    if "companion_tstop" in v:
        P.mobs.tstop[6] = v["companion_tstop"]
    P.scenario = name
    return P

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
 * apply() raises the shoulder W 900 -> 20000 (deterministic +y wall kick for M1_v2).

Per-body stop time semantics (v6): centre_k(t) = c0_k + vel_k * min(t, tstop_k) -- linear until tstop,
frozen after.  mpc_core.mobs_center is linear; the testbed installs the piecewise replacement at import.

Design rationale (condensed from the original module headers):
  * WALL start 12.0 / stop 6.4 s: measured stall onsets PMP 6.44 / Coll 6.50 / DDP 6.58 -> stopping at
    the earliest stall freezes the wall at x = 10.4 so trapped methods are never pushed backward.
  * BLOCKER at (11.8, 0.55): sits in MPPI's measured approach lane (|y| ~ 0.43) and above M1's
    (|y| ~ 0.28); frozen-obstacle prediction sees it exactly (static) so the reroute is priced early.
  * DOOR at 13.8, closing 0.25 m/s from y0 = 0.80 + 0.22 + 0.25*9.25 = 3.33: gap 0.22 when M1 passes
    (t ~ 9.25), frozen shut at t = (3.33-0.80)/0.25 = 10.13 (no endless chasing of ejected methods;
    the original v6 module comment said 9.92 -- its formula, kept here, gives 10.13).
  * SHOULDER blocker overlapping the wall's -y shoulder (core from x ~ 9.7): prices the -y dodge AT the
    saddle split for every method; thin shell -> saturated-zero on axis, so PMP/collocation traps and the
    symmetric-saddle certification are untouched to machine precision.
  * FORK (v10): every method dodges the wall +y; the DDP trap is the downstream fork.  The companion
    ARRIVES from -y and freezes at (16.2, -0.60) at t = 10.0 -- absent at DDP's fork commitment
    (~9.3-9.6 s, commits -y blind), present and hot (W = 5e4, thick ax .80 ay .40 so it cannot be
    punched through) for M1's fork kick (~10.3 s) -> deterministic +y slalom.  Stage-2 blocker denies
    DDP's late wide +y arc.

---------------------------------------------------------------------------------------------------
HORIZONS (20 ms/cycle budget; dt = 0.005)
---------------------------------------------------------------------------------------------------
                    wall course        unified-route course (apply)
  PMP      N_pmp        220                 260   largest clean-trap horizon (cut 3.9%; erratic >= 290)
  M1_v2    N_m1         180                 220   largest horizon with p99 inside 20 ms (fair-N: PMP is
                                                  cheaper per iteration -> gets the longer horizon)
  iLQR     N_ilqr       200                 200   (not run in the final testbeds)
  DDP      N_ddp        150                 300   J-optimal on its own closed-loop sweep
  Coll     N_coll        80                  80
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
GAP_M1 = 0.22          # door gap when M1 passes (t = 9.25 s)
Y0_2 = AYP + GAP_M1 + VC * 9.25                                      # = 3.33: door post start y
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
    P.N_pmp = 220
    P.N_m1 = 180
    P.N_ilqr = 200
    P.N_ddp = 150
    P.N_coll = 80
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
    P.N_pmp = 260
    P.N_m1 = 220
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

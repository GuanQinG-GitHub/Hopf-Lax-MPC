"""
mpc_multi.py -- K identical, DECOUPLED aerial vehicles as ONE stacked shooting problem.

Scaling study of Hopf-Lax-MPC with respect to the number of agents: the K agents share the scene
(all moving obstacles FROZEN at their t=0 positions), have their own start / goal / RRT reference, and
do not interact (no inter-agent cost, no coupling in the dynamics).  The algorithm is unchanged; only
the problem is stacked: the costate v = [p_1; ...; p_K] (7K), the state x = [x_1; ...; x_K] (7K), the
reference window Xr = [Xr_1, ..., Xr_K] (K, n+3*nm, 2N+1), the residual rho = [rho_1; ...; rho_K] and
the objective Phi = sum_k Phi_k.  `build_ss_multi` returns an object with exactly the API of
mpc_core.build_ss (res, J, all, Jr, Sx, g, roll_pred, roll_costate, warm), so mpc_solvers.chlqn_solve
and mpc_solvers.ensure_jbatch run on it UNCHANGED.  The Jacobians Jr, Sx are formed by the same fused
forward-mode pass over the full 7K-dimensional costate (no exploitation of the block-diagonal
structure): this is the honest "stack the states and run the algorithm" cost.

Scene layout (make_multi_scene): agent k spawns at (x0, y_k, 0.5) and its goal is the opposite corner
(x_goal, -y_k, 0.5), with y_k spread uniformly over [-Y, +Y]; the RRT planner sees the static obstacle
(as in the single-agent testbed) PLUS the frozen moving bodies (spherical influence radii), so every
reference is a feasible route through the frozen scene.  Existing modules are not modified.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import mpc_core as C

NAGENT_Y_SPREAD = 2.0                      # spawn lateral positions y_k in [-2, 2]


# ======================================================================================================
#  SCENE
# ======================================================================================================
def freeze_mobs(P):
    """All moving obstacles frozen at their t=0 centres (velocity zero; any stop times irrelevant)."""
    m = P.mobs
    P.mobs = SimpleNamespace(c0=m.c0.copy(), vel=np.zeros_like(m.vel), ax=m.ax.copy(), ay=m.ay.copy(),
                             az=m.az.copy(), W=m.W.copy(), w=m.w.copy(), eps=m.eps)
    if hasattr(m, "tstop"):
        P.mobs.tstop = np.full(m.c0.shape[1], np.inf)
    return P


def planner_obstacles(P, margin=0.25):
    """RRT obstacle model for the frozen scene: the real static obstacle(s) plus every frozen moving
    body as a sphere of radius max(ax, ay) + margin (the planner is no longer deliberately wrong)."""
    cs = [P.obs.center]
    rs = [np.maximum(P.obs.ax, P.obs.ay) + margin]
    cs.append(P.mobs.c0)
    rs.append(np.maximum(P.mobs.ax, P.mobs.ay) + margin)
    return SimpleNamespace(center=np.hstack(cs), rinfl=np.concatenate(rs))


def agent_layout(K, P, y_spread=NAGENT_Y_SPREAD):
    """Start / goal per agent: (x0, y_k) -> (x_goal, -y_k), y_k uniform in [-y_spread, y_spread]."""
    ys = np.linspace(-y_spread, y_spread, K) if K > 1 else np.array([0.0])
    starts = [np.array([P.p0[0], y, P.p0[2]]) for y in ys]
    goals = [np.array([P.pgoal[0], -y, P.pgoal[2]]) for y in ys]
    return starts, goals


def make_multi_scene(P, K, solo=None):
    """Frozen scene, K agents (or ONE agent `solo` of the K-layout, for correctness checks).
    Returns (P, M, JX, MK) with MK = SimpleNamespace(K, starts, goals, paths, x0 (7K,), obs_plan)."""
    P = freeze_mobs(P)
    M = C.model_scn(P)
    JX = C.build_jax(M)
    starts, goals = agent_layout(K, P)
    idx = list(range(K)) if solo is None else [int(solo)]
    obs_plan = planner_obstacles(P)
    paths, x0s = [], []
    for k in idx:
        paths.append(C.plan_path(starts[k], goals[k], obs_plan, P))
        x0s.append(np.array([starts[k][0], starts[k][1], starts[k][2], 0.0, P.vcr, 0.0, 0.0]))
    MK = SimpleNamespace(K=len(idx), agents=idx, starts=[starts[k] for k in idx], goals=[goals[k] for k in idx],
                         paths=paths, x0=np.concatenate(x0s), obs_plan=obs_plan, n=M.n, m=M.m)
    return P, M, JX, MK


def ref_window_multi(xf, N, MK, M, P, sub, t):
    """Stack the per-agent reference windows: (K, n+3*nm, sub*N+1)."""
    K, n = MK.K, M.n
    X = xf.reshape(K, n)
    return np.stack([C.ref_window(X[k], N, MK.paths[k], M, P, sub, t) for k in range(K)], axis=0)


# ======================================================================================================
#  STACKED SHOOTING OBJECT  (API-identical to mpc_core.build_ss; flat 7K vectors, 3-D Xr)
# ======================================================================================================
def build_ss_multi(M, K, N, dt, JX=None, napply=10):
    JX = C.build_jax(M) if JX is None else JX
    n = M.n
    QT = JX.QT

    def _shoot1(p0, xc, Xr):
        za0 = jnp.concatenate([xc, p0, jnp.zeros(1)])
        RA = Xr[:, 0:2 * N:2].T
        RM = Xr[:, 1:2 * N:2].T
        RB = Xr[:, 2:2 * N + 1:2].T

        def body(za, tr):
            return JX.rk4_canon_aug(za, tr[0], tr[1], tr[2], dt), None

        zaN, _ = lax.scan(body, za0, (RA, RM, RB))
        return zaN

    def _outs1(p0, xc, Xr):
        za = _shoot1(p0, xc, Xr)
        xN, pN, cc = za[:n], za[n:2 * n], za[2 * n]
        dxT = xN - Xr[:n, -1]
        Jc = cc + dxT @ QT @ dxT
        res = pN - 2.0 * QT @ dxT
        return res, xN, Jc

    def _outs(p0f, xcf, XrK):
        """Stacked outputs: res (7K,), xN (7K,), Phi = sum_k Phi_k (scalar)."""
        P0, XC = p0f.reshape(K, n), xcf.reshape(K, n)
        res, xN, Jc = jax.vmap(_outs1)(P0, XC, XrK)
        return res.reshape(-1), xN.reshape(-1), jnp.sum(Jc)

    @jax.jit
    def S_res(p0, xc, Xr):
        return _outs(p0, xc, Xr)[0]

    @jax.jit
    def S_J(p0, xc, Xr):
        return _outs(p0, xc, Xr)[2]

    @jax.jit
    def S_resx(p0, xc, Xr):
        res, xN, _ = _outs(p0, xc, Xr)
        return res, xN

    @jax.jit
    def S_all(p0, xc, Xr):
        """Fused [res, J, Jr, Sx] from ONE vmapped forward pass over the FULL 7K costate (as build_ss)."""
        nK = K * n

        def f(p):
            res, xN, Jc = _outs(p, xc, Xr)
            return jnp.concatenate([res, xN, jnp.array([Jc])])

        val, jac = C._value_and_jacfwd(f, p0)                        # jac: (2nK+1, nK)
        return val[:nK], val[2 * nK], jac[:nK, :], jac[nK:2 * nK, :]

    @jax.jit
    def S_Jr(p0, xc, Xr):
        return jax.jacfwd(lambda p: _outs(p, xc, Xr)[0])(p0)

    @jax.jit
    def S_Sx(p0, xc, Xr):
        return jax.jacfwd(lambda p: _outs(p, xc, Xr)[1])(p0)

    @jax.jit
    def S_g(p0, xc, Xr):
        return jax.grad(lambda p: _outs(p, xc, Xr)[2])(p0)

    def _roll_pred1(p0, xc, Xr):
        za0 = jnp.concatenate([xc, p0, jnp.zeros(1)])
        RA, RM = Xr[:, 0:2 * N:2].T, Xr[:, 1:2 * N:2].T
        RB = Xr[:, 2:2 * N + 1:2].T

        def body(za, tr):
            za2 = JX.rk4_canon_aug(za, tr[0], tr[1], tr[2], dt)
            return za2, za2[:n]

        _, Xs = lax.scan(body, za0, (RA, RM, RB))
        return jnp.vstack([xc[None, :], Xs]).T                       # (n, N+1)

    @jax.jit
    def roll_pred(p0, xc, Xr):
        """Predicted state trajectories per agent: (K, n, N+1)."""
        return jax.vmap(_roll_pred1)(p0.reshape(K, n), xc.reshape(K, n), Xr)

    def _roll_costate1(p0, xc, Xr):
        z0 = jnp.concatenate([xc, p0])
        RA = Xr[:, 0:2 * napply:2].T
        RM = Xr[:, 1:2 * napply:2].T
        RB = Xr[:, 2:2 * napply + 1:2].T

        def body(z, tr):
            z2 = JX.rk4_canon(z, tr[0], tr[1], tr[2], dt)
            return z2, z2[n:]

        _, Ps = lax.scan(body, z0, (RA, RM, RB))
        return jnp.vstack([p0[None, :], Ps]).T                       # (n, napply+1)

    @jax.jit
    def roll_costate(p0, xc, Xr):
        """Stacked costate over the first napply sub-steps: (7K, napply+1); column napply = next seed."""
        Pc = jax.vmap(_roll_costate1)(p0.reshape(K, n), xc.reshape(K, n), Xr)   # (K, n, napply+1)
        return Pc.reshape(K * n, napply + 1)

    def warm(xc, Xr):
        p = np.zeros(K * n)
        S_res(p, xc, Xr); S_J(p, xc, Xr); S_all(p, xc, Xr); S_resx(p, xc, Xr)
        S_Jr(p, xc, Xr); S_Sx(p, xc, Xr); S_g(p, xc, Xr)
        roll_pred(p, xc, Xr); roll_costate(p, xc, Xr)

    return SimpleNamespace(res=S_res, J=S_J, all=S_all, resx=S_resx, Jr=S_Jr, Sx=S_Sx, g=S_g,
                           roll_pred=roll_pred, roll_costate=roll_costate, warm=warm, N=N, dt=dt, JX=JX, K=K)


# ======================================================================================================
#  CLOSED-LOOP STEP (per agent, true cost; decoupled)
# ======================================================================================================
def apply_step_multi(xf, uf, XrK, Jreal, Jpen, M, MK, P, t, JX, done=None):
    """Apply u_k for napply fine steps per agent; integrate the TRUE per-agent cost (tracking + static
    and frozen-moving barriers).  Agents flagged in `done` (already arrived) keep moving under their
    control but accrue NO further cost, so each agent's J is its cost up to its own arrival -- exactly
    what a solo run of that agent records.  Returns the new stacked state, updated per-agent cost
    arrays, per-agent 'at goal' flags and the new time."""
    K, n = MK.K, M.n
    X = xf.reshape(K, n).copy()
    U = uf.reshape(K, M.m)
    mc = C.mobs_center(M.mobs, t)                                     # frozen: = c0
    at_goal = np.zeros(K, dtype=bool)
    done = np.zeros(K, dtype=bool) if done is None else done
    for k in range(K):
        x, u = X[k], U[k]
        dxk = x - XrK[k, :n, 0]
        if not done[k]:
            Jreal[k] += (dxk @ M.Q @ dxk + u @ M.R @ u) * P.dt_apply
        for _ in range(P.napply):
            x = np.asarray(JX.rk4_state(x, u, P.dt))
            if not done[k]:
                pen1, _ = JX.obstacle(x[M.posidx])
                pen2, _ = JX.mobs_pen(x[M.posidx], mc)
                Jpen[k] += (float(pen1) + float(pen2)) * P.dt
        X[k] = x
        at_goal[k] = np.linalg.norm(x[M.posidx] - MK.goals[k][:len(M.posidx)]) < P.reachtol
    return X.reshape(-1), Jreal, Jpen, at_goal, t + P.napply * P.dt


def ustar_multi(xf, pf, M, JX):
    K, n = pf.size // M.n, M.n
    X, Pp = xf.reshape(K, n), pf.reshape(K, n)
    return np.concatenate([np.asarray(JX.ustar(X[k], Pp[k])) for k in range(K)])

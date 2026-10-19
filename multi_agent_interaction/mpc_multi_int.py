"""
mpc_multi_int.py -- K INTERACTING aerial vehicles as ONE coupled shooting problem.

Extension of multi_agent/mpc_multi.py: the scene, agent layout and reference windows are reused
unchanged, and a soft pairwise collision penalty is added to the running cost,

    L = sum_k L_k(x_k, u_k, t) + sum_{i<j} C(q_i, q_j),
    C(q_i, q_j) = (Wc/2) (1 - tanh((r_ij - 1)/wc)),   r_ij = sqrt(|q_i - q_j|^2 / Rc^2 + eps),

with q_k = x_k[0:3].  C does not depend on u, so u*_k is the unchanged per-agent clipped formula; the
only new term of the canonical system is in the costate equation,

    pdot_k = -dHdx7_k(x_k, p_k) - [ sum_{j != k} g_kj ; 0_4 ],   g_kj = grad_{q_k} C(q_k, q_j),

so the K rollouts are no longer independent: `build_ss_int` integrates ONE joint rollout of all agents.
It returns an object with exactly the API of mpc_multi.build_ss_multi (flat agent-major 7K vectors, 3-D
reference window), so mpc_solvers.hopf_lax_mpc_solve and ensure_jbatch run on it UNCHANGED.  With
Wc = 0 (or K = 1) it reproduces build_ss_multi.  Existing modules are not modified.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d) for _d in ("core", "multi_agent")]
import mpc_core as C
import mpc_multi as MU                                                 # scene, ref_window_multi, ustar_multi (reused)

RC_DEFAULT, WC_DEFAULT, WSH_DEFAULT = 0.8, 500.0, 0.1                 # safety radius [m], weight, shell width
PAIR_EPS = 1e-10                                                      # as the obstacle barriers


def make_col(Rc=RC_DEFAULT, Wc=WC_DEFAULT, wc=WSH_DEFAULT, eps=PAIR_EPS):
    return SimpleNamespace(Rc=float(Rc), Wc=float(Wc), wc=float(wc), eps=float(eps))


# ======================================================================================================
#  PAIRWISE COLLISION PENALTY
# ======================================================================================================
def pair_penalty(Q, col):
    """Q (K, 3) agent positions -> (sum_{i<j} C(q_i, q_j), G (K, 3) with G_k = grad_{q_k} of that sum).

    Vectorised over the K x K difference tensor with the diagonal masked out (K = 1 -> exactly 0); eps
    keeps the masked diagonal sqrt finite and differentiable.  G_k = sum_{j != k} g_kj with
    g_kj = -(Wc / (2 wc)) (1 - tanh^2 z_kj) (q_k - q_j) / (Rc^2 r_kj), z_kj = (r_kj - 1)/wc."""
    K = Q.shape[0]
    D = Q[:, None, :] - Q[None, :, :]                                 # (K, K, 3): q_i - q_j
    r = jnp.sqrt(jnp.sum(D ** 2, axis=-1) / col.Rc ** 2 + col.eps)
    th = jnp.tanh((r - 1.0) / col.wc)
    mask = 1.0 - jnp.eye(K, dtype=Q.dtype)
    pen = 0.5 * jnp.sum(mask * (0.5 * col.Wc) * (1.0 - th))          # each unordered pair once
    coef = mask * (-0.5 * col.Wc / col.wc) * (1.0 - th ** 2) / (col.Rc ** 2 * r)
    G = jnp.sum(coef[:, :, None] * D, axis=1)
    return pen, G


def pair_penalty_np(Q, col):
    """NumPy twin of pair_penalty for the closed-loop true cost: (sum_{i<j} C, per-pair C (K, K))."""
    D = Q[:, None, :] - Q[None, :, :]
    r = np.sqrt(np.sum(D ** 2, axis=-1) / col.Rc ** 2 + col.eps)
    Cij = 0.5 * col.Wc * (1.0 - np.tanh((r - 1.0) / col.wc))
    np.fill_diagonal(Cij, 0.0)
    return 0.5 * float(np.sum(Cij)), Cij


def pair_distances(Q):
    """Q (K, 3) -> (min distance over pairs, (i, j) of the closest pair); (inf, None) for K = 1."""
    K = Q.shape[0]
    if K < 2:
        return np.inf, None
    D = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
    D[np.diag_indices(K)] = np.inf
    i, j = np.unravel_index(np.argmin(D), D.shape)
    return float(D[i, j]), (int(min(i, j)), int(max(i, j)))


# ======================================================================================================
#  COUPLED SHOOTING OBJECT  (API-identical to mpc_multi.build_ss_multi)
# ======================================================================================================
def build_ss_int(M, K, N, dt, JX=None, col=None, napply=10):
    """One JOINT canonical rollout of the K agents.  Carry: ZA (K, 2n+1) = per agent [x_k, p_k, c_k]
    (c_k = integral of L_k, exactly as the decoupled object) and cp = integral of sum_{i<j} C.
    Per-agent RHS = jax.vmap(JX.fz_aug) (core dynamics, -dH/dx, L_k unchanged); coupling: the position
    entries of pdot_k get -G_k and cp' = sum_{i<j} C.  Phi = sum_k (c_k + phi_k) + cp."""
    JX = C.build_jax(M) if JX is None else JX
    col = make_col() if col is None else col
    n = M.n
    QT = JX.QT
    pos_i = JX.pos_i
    fz_aug_K = jax.vmap(JX.fz_aug)                                     # (K, 2n+1), (K, rows) -> (K, 2n+1)
    fz_K = jax.vmap(JX.fz_track)                                       # (K, 2n),   (K, rows) -> (K, 2n)

    def rhs_aug(ZA, XA):
        dZ = fz_aug_K(ZA, XA)
        pen, G = pair_penalty(ZA[:, pos_i], col)
        dZ = dZ.at[:, n + pos_i].add(-G)                               # pdot_k[0:3] -= G_k
        return dZ, pen

    def rk4_aug(carry, XAa, XAm, XAb):
        ZA, cp = carry
        k1, c1 = rhs_aug(ZA, XAa)
        k2, c2 = rhs_aug(ZA + dt / 2 * k1, XAm)
        k3, c3 = rhs_aug(ZA + dt / 2 * k2, XAm)
        k4, c4 = rhs_aug(ZA + dt * k3, XAb)
        return ZA + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), cp + dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4)

    def rhs(Z, XA):                                                    # canonical system without the cost
        dZ = fz_K(Z, XA)
        _, G = pair_penalty(Z[:, pos_i], col)
        return dZ.at[:, n + pos_i].add(-G)

    def rk4(Z, XAa, XAm, XAb):
        k1 = rhs(Z, XAa); k2 = rhs(Z + dt / 2 * k1, XAm)
        k3 = rhs(Z + dt / 2 * k2, XAm); k4 = rhs(Z + dt * k3, XAb)
        return Z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _stages(XrK, nst):
        """Sub-stage reference columns, time-major: (nst, K, rows) each (as the core's RA/RM/RB)."""
        tr = lambda a: jnp.transpose(a, (2, 0, 1))
        return tr(XrK[:, :, 0:2 * nst:2]), tr(XrK[:, :, 1:2 * nst:2]), tr(XrK[:, :, 2:2 * nst + 1:2])

    def _shoot(p0f, xcf, XrK):
        P0, XC = p0f.reshape(K, n), xcf.reshape(K, n)
        ZA0 = jnp.concatenate([XC, P0, jnp.zeros((K, 1), dtype=P0.dtype)], axis=1)

        def body(carry, tr):
            return rk4_aug(carry, tr[0], tr[1], tr[2]), None

        (ZAN, cp), _ = lax.scan(body, (ZA0, jnp.zeros((), dtype=P0.dtype)), _stages(XrK, N))
        return ZAN, cp

    def _outs(p0f, xcf, XrK):
        """Stacked outputs: res (7K,), xN (7K,), Phi = sum_k Phi_k + integral of the pair penalty."""
        ZAN, cp = _shoot(p0f, xcf, XrK)
        xN, pN, cc = ZAN[:, :n], ZAN[:, n:2 * n], ZAN[:, 2 * n]
        dxT = xN - XrK[:, :n, -1]
        Jc = cc + jnp.einsum("ki,ij,kj->k", dxT, QT, dxT)
        res = pN - 2.0 * dxT @ QT.T
        return res.reshape(-1), xN.reshape(-1), jnp.sum(Jc) + cp

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
        """Fused [res, J, Jr, Sx] from ONE vmapped forward pass over the full 7K costate (as build_ss)."""
        nK = K * n

        def f(p):
            res, xN, Jc = _outs(p, xc, Xr)
            return jnp.concatenate([res, xN, jnp.array([Jc])])

        val, jac = C._value_and_jacfwd(f, p0)
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

    @jax.jit
    def roll_pred(p0, xc, Xr):
        """Predicted state trajectories per agent: (K, n, N+1)."""
        Z0 = jnp.concatenate([xc.reshape(K, n), p0.reshape(K, n)], axis=1)

        def body(Z, tr):
            Z2 = rk4(Z, tr[0], tr[1], tr[2])
            return Z2, Z2[:, :n]

        _, Xs = lax.scan(body, Z0, _stages(Xr, N))                     # (N, K, n)
        return jnp.concatenate([xc.reshape(K, n)[None], Xs], axis=0).transpose(1, 2, 0)

    @jax.jit
    def roll_costate(p0, xc, Xr):
        """Stacked costate over the first napply sub-steps: (7K, napply+1); column napply = next seed."""
        Z0 = jnp.concatenate([xc.reshape(K, n), p0.reshape(K, n)], axis=1)

        def body(Z, tr):
            Z2 = rk4(Z, tr[0], tr[1], tr[2])
            return Z2, Z2[:, n:]

        _, Ps = lax.scan(body, Z0, _stages(Xr, napply))               # (napply, K, n)
        Pc = jnp.concatenate([p0.reshape(K, n)[None], Ps], axis=0)    # (napply+1, K, n)
        return Pc.transpose(1, 2, 0).reshape(K * n, napply + 1)

    @jax.jit
    def hamiltonian(xf, pf, uf, Xa):
        """Joint H(x, p, u) at one reference column Xa (K, rows): sum_k [L_k + p_k'f_k] + sum_{i<j} C.
        For the self-test of the hand-derived costate equation (dH/dx with u held fixed)."""
        X, Pm, U = xf.reshape(K, n), pf.reshape(K, n), uf.reshape(K, -1)

        def H1(x, p, u, xa):
            xr, mc = xa[:n], xa[n:].reshape((JX.Kmob, 3)).T
            po, _ = JX.obstacle(x[pos_i])
            pm, _ = JX.mobs_pen(x[pos_i], mc)
            return (x - xr) @ JX.Q @ (x - xr) + u @ JX.R @ u + po + pm + p @ JX.dyn7(x, u)

        return jnp.sum(jax.vmap(H1)(X, Pm, U, Xa)) + pair_penalty(X[:, pos_i], col)[0]

    @jax.jit
    def costate_rhs(xf, pf, Xa):
        """Joint pdot = -dH/dx as integrated by the rollout (flat 7K)."""
        Z = jnp.concatenate([xf.reshape(K, n), pf.reshape(K, n)], axis=1)
        return rhs(Z, Xa)[:, n:].reshape(-1)

    def warm(xc, Xr):
        p = np.zeros(K * n)
        S_res(p, xc, Xr); S_J(p, xc, Xr); S_all(p, xc, Xr); S_resx(p, xc, Xr)
        S_Jr(p, xc, Xr); S_Sx(p, xc, Xr); S_g(p, xc, Xr)
        roll_pred(p, xc, Xr); roll_costate(p, xc, Xr)

    return SimpleNamespace(res=S_res, J=S_J, all=S_all, resx=S_resx, Jr=S_Jr, Sx=S_Sx, g=S_g,
                           roll_pred=roll_pred, roll_costate=roll_costate, warm=warm, N=N, dt=dt, JX=JX, K=K,
                           col=col, hamiltonian=hamiltonian, costate_rhs=costate_rhs)


# ======================================================================================================
#  CLOSED-LOOP STEP (decoupled plants, coupled true cost)
# ======================================================================================================
def apply_step_int(xf, uf, XrK, Jreal, Jpen, Jpair, M, MK, P, t, JX, col, done=None):
    """Apply u_k for napply fine steps per agent (the plants do not interact); integrate the TRUE cost:
    per agent tracking + barriers until its own arrival (as mpc_multi.apply_step_multi), and per pair
    the collision penalty while at least one of the two agents has not arrived (Jpair (K, K), upper
    triangle).  Also returns the minimum pairwise distance over the fine steps and its pair."""
    K, n = MK.K, M.n
    X = xf.reshape(K, n).copy()
    U = uf.reshape(K, M.m)
    mc = C.mobs_center(M.mobs, t)
    done = np.zeros(K, dtype=bool) if done is None else done
    live = np.triu(~np.logical_and.outer(done, done), 1)               # pair still accrues cost
    for k in range(K):
        if not done[k]:
            dxk = X[k] - XrK[k, :n, 0]
            Jreal[k] += (dxk @ M.Q @ dxk + U[k] @ M.R @ U[k]) * P.dt_apply
    dmin, pmin = pair_distances(X[:, M.posidx])
    for _ in range(P.napply):
        for k in range(K):
            X[k] = np.asarray(JX.rk4_state(X[k], U[k], P.dt))
            if not done[k]:
                pen1, _ = JX.obstacle(X[k][M.posidx])
                pen2, _ = JX.mobs_pen(X[k][M.posidx], mc)
                Jpen[k] += (float(pen1) + float(pen2)) * P.dt
        if K > 1:
            _, Cij = pair_penalty_np(X[:, M.posidx], col)
            Jpair += np.where(live, Cij, 0.0) * P.dt
            d, pr = pair_distances(X[:, M.posidx])
            if d < dmin:
                dmin, pmin = d, pr
    at_goal = np.array([np.linalg.norm(X[k][M.posidx] - MK.goals[k][:len(M.posidx)]) < P.reachtol for k in range(K)])
    return X.reshape(-1), Jreal, Jpen, Jpair, at_goal, t + P.napply * P.dt, dmin, pmin


def ustar_multi(xf, pf, M, JX):
    return MU.ustar_multi(xf, pf, M, JX)

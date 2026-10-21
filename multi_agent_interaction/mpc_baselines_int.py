"""
mpc_baselines_int.py -- the baseline MPC solvers extended to K INTERACTING agents (pair collision penalty).

The coupled problem is the one of mpc_multi_int.py: K agents with decoupled dynamics (one dyn7 vehicle each)
and the running cost

    L = sum_k [ (x_k - xr_k)' Q (x_k - xr_k) + u_k' R u_k + B_obs(q_k) ] + sum_{i<j} C(q_i, q_j),

terminal cost sum_k (x_k - xr_k)' QT (x_k - xr_k).  Per baseline:

  PMP_Shooting, Vanilla_Hopf_Lax   only use the shooting object (res, Jr / all, J); they run UNCHANGED on
                                   mpc_multi_int.build_ss_int (nothing in this module).
  DDP                              build_ddp_int: copy of mpc_solvers.build_ddp on the JOINT model
                                   (state 7K, control 3K, stacked dynamics, joint barrier + pair cost); same
                                   API, so mpc_solvers.ddp_run (and the counted N-sweep copy) run unchanged.
  MPPI                             MPPIInt: subclass of mpc_solvers.MPPI with a new CUDA kernel in which one
                                   thread rolls out ALL K agents of one sample and adds the pair penalty.

Existing modules are not modified.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import os as _os, sys as _sys                                         # repository layout: shared modules in sibling folders
_sys.path[:0] = [_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), _d)
                 for _d in ("core", "multi_agent")]
import mpc_solvers as S
import mpc_multi_int as MI

DDP_ALPHAS = S.DDP_ALPHAS


def joint_xref(XrK1, n):
    """Per-agent sub=1 reference windows (K, n+3nm, N+1) -> joint tracking reference (K*n, N+1), agent-major
    (the layout of the flat joint state x = [x_1; ...; x_K])."""
    return np.ascontiguousarray(XrK1[:, :n, :].reshape(-1, XrK1.shape[2]))


def pos_index(K, pos_i, n):
    """Flat indices of all agents' positions in the joint state: (3K,), agent-major."""
    return (np.arange(K)[:, None] * n + np.asarray(pos_i)[None, :]).ravel()


# ======================================================================================================
#  BOX-DDP pieces (control-limited DDP, Tassa, Mansard & Todorov, ICRA 2014)
#
#  The plain DDP backward pass takes the unconstrained step k = -Quu^-1 Qu and the bound |u| <= umax only
#  enters through the clip in the forward pass: with controls on the bound the step points out of the box,
#  the clipped rollout disagrees with the quadratic model and the line search fails.  Control-limited DDP
#  instead solves, per stage,
#        min_du  1/2 du' Quu du + Qu' du    s.t.   -umax - u <= du <= umax - u
#  by projected Newton (Tassa's boxQP), and sets the feedback rows of the CLAMPED channels to zero:
#        K_free = -Quu_ff^-1 Qux_f,   K_clamped = 0.
#  The value-function update keeps the general form (Vx = Qx + K'Quu k + K'Qu + Qux'k, ...).
# ======================================================================================================
BOXQP_MAXIT = 50
BOXQP_ARMIJO = 0.1
BOXQP_STEPS = 0.6 ** jnp.arange(22)                                   # Tassa: step *= 0.6 until Armijo
BOXQP_GTOL = 1e-8


def masked_solve(H, B, free):
    """Solve H_ff X_f = B_f on the free index set, zero rows on the clamped set (fixed shapes for jit)."""
    f = free.astype(H.dtype)
    Hm = H * jnp.outer(f, f) + jnp.diag(1.0 - f)
    L = jnp.linalg.cholesky(Hm)
    fb = f[:, None] if B.ndim == 2 else f
    return jax.scipy.linalg.cho_solve((L, True), B * fb) * fb


def box_qp(H, g, lo, hi):
    """Projected-Newton box QP (port of Tassa's boxQP): min 1/2 x'Hx + g'x, lo <= x <= hi, start x = clip(0).
    Returns (x, free mask at the solution).  H must be PD (the DDP 'bad' flag already checks the full Quu)."""
    def val(x):
        return x @ g + 0.5 * x @ H @ x

    def clamped_set(x, grad):
        return ((x <= lo) & (grad > 0)) | ((x >= hi) & (grad < 0))

    def cond(st):
        x, it, done = st
        return (~done) & (it < BOXQP_MAXIT)

    def body(st):
        x, it, done = st
        grad = g + H @ x
        clamped = clamped_set(x, grad)
        free = ~clamped
        f = free.astype(H.dtype)
        gnorm = jnp.linalg.norm(grad * f)
        gc = g + H @ (x * (1.0 - f))                                   # gradient with the free part of x zeroed
        search = masked_solve(H, -gc, free) - x * f                    # Newton step on the free subspace
        sdotg = search @ grad
        stop = jnp.all(clamped) | (gnorm < BOXQP_GTOL) | (sdotg >= 0.0)
        xs = jnp.clip(x[None, :] + BOXQP_STEPS[:, None] * search[None, :], lo, hi)
        vals = xs @ g + 0.5 * jnp.einsum("si,ij,sj->s", xs, H, xs)
        ok = (vals - val(x)) / (BOXQP_STEPS * sdotg) >= BOXQP_ARMIJO
        anyok = jnp.any(ok)
        xn = jnp.where(stop | ~anyok, x, xs[jnp.argmax(ok)])
        return xn, it + 1, stop | ~anyok

    x0 = jnp.clip(jnp.zeros_like(g), lo, hi)
    x, _, _ = lax.while_loop(cond, body, (x0, 0, False))
    free = ~clamped_set(x, g + H @ x)
    return x, free


# ======================================================================================================
#  DDP on the joint model  (copy of mpc_solvers.build_ddp; changes marked  <<K, Box-DDP option  <<BOX)
# ======================================================================================================
def build_ddp_int(M, K, N, dt, JX, col, box=False, psd_barrier=None):
    """box=True: control-limited Box-DDP; psd_barrier: project the joint barrier + pair Hessian to PSD before the
    backward pass -- default (as main's core build_ddp): True for plain DDP, False for Box-DDP (EXACT Hessian)."""
    if psd_barrier is None:
        psd_barrier = not box
    n0, m0 = M.n, M.m
    n, m = K * n0, K * m0                                              # <<K joint dimensions
    Q = jnp.kron(jnp.eye(K), JX.Q)                                     # <<K block-diagonal weights
    QT = jnp.kron(jnp.eye(K), JX.QT)
    R = jnp.kron(jnp.eye(K), JX.R)
    umax = jnp.tile(JX.umax, K)
    pidx = jnp.asarray(pos_index(K, JX.pos_i, n0))                     # <<K all agents' positions (3K,)

    def obs_all(posf, mc):
        """<<K joint barrier: per-agent static + frozen-moving barriers at q_k, plus sum_{i<j} C(q_i, q_j).
        posf: (3K,) agent-major positions.  Returns (penalty, gradient (3K,))."""
        Qp = posf.reshape(K, 3)
        p1, g1 = jax.vmap(JX.obstacle)(Qp)
        p2, g2 = jax.vmap(lambda q: JX.mobs_pen(q, mc))(Qp)
        pp, G = MI.pair_penalty(Qp, col)
        return jnp.sum(p1 + p2) + pp, (g1 + g2 + G).reshape(-1)

    def _F(x, u):                                                      # <<K stacked per-agent RK4
        return jax.vmap(lambda xx, uu: JX.rk4_state(xx, uu, dt))(x.reshape(K, n0), u.reshape(K, m0)).reshape(-1)

    def _Fz(z, xr_unused=None):
        return _F(z[:n], z[n:])

    def _rollout(x0, U):
        def body(x, u):
            x2 = _F(x, u)
            return x2, x2
        _, Xs = lax.scan(body, x0, U.T)
        return jnp.vstack([x0[None, :], Xs]).T

    def _traj_cost(X, U, Xref, mc):
        def body(c, k):
            dxk = X[:, k] - Xref[:, k]
            pen, _ = obs_all(X[pidx, k], mc)
            return c + (dxk @ Q @ dxk + U[:, k] @ R @ U[:, k] + pen) * dt, None
        Jc, _ = lax.scan(body, 0.0, jnp.arange(N))
        dxT = X[:, N] - Xref[:, N]
        return Jc + dxT @ QT @ dxT

    @jax.jit
    def rollout_cost(x0, U, Xref, mc):
        X = _rollout(x0, U)
        return X, _traj_cost(X, U, Xref, mc)

    def _psd(H):
        w, V = jnp.linalg.eigh(0.5 * (H + H.T))
        return (V * jnp.maximum(w, 0.0)) @ V.T

    @jax.jit
    def backward(X, U, Xref, mc, mu):
        """Full second-order sweep on the joint model. Returns (kff, Kk, dV1, dV2, bad)."""
        Z = jnp.concatenate([X[:, :N].T, U.T], axis=1)
        Fx = jax.vmap(lambda x, u: jax.jacfwd(lambda xx: _F(xx, u))(x))(X[:, :N].T, U.T)
        Fu = jax.vmap(lambda x, u: jax.jacfwd(lambda uu: _F(x, uu))(u))(X[:, :N].T, U.T)
        H = jax.vmap(lambda z: jax.jacfwd(jax.jacfwd(_Fz))(z))(Z)     # dense (N, n, n+m, n+m)
        Fxx = H[:, :, :n, :n]
        Fuu = H[:, :, n:, n:]
        Fux = H[:, :, n:, :n]
        # <<K joint barrier + pair Hessian over all 3K positions, PSD-projected (as the per-agent 3x3 one)
        _bproj = _psd if psd_barrier else (lambda H: 0.5 * (H + H.T))  # Box-DDP: exact (indefinite) Hessian
        Bpos = jax.vmap(lambda x: _bproj(jax.hessian(lambda p: obs_all(p, mc)[0])(x[pidx])))(X[:, :N].T)

        def body(carry, k):
            Vx, Vxx, dV1, dV2, bad = carry
            fx, fu = Fx[k], Fu[k]
            x, u = X[:, k], U[:, k]
            _, og = obs_all(x[pidx], mc)
            gx = jnp.zeros(n).at[pidx].set(og)                         # <<K scatter into position entries
            lxx = 2 * Q * dt
            lxx = lxx.at[jnp.ix_(pidx, pidx)].add(Bpos[k] * dt)
            lx = (2 * Q @ (x - Xref[:, k]) + gx) * dt
            lu = 2 * R @ u * dt
            Vxx_r = Vxx + mu * jnp.eye(n)
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx + jnp.einsum('i,ijk->jk', Vx, Fxx[k])
            Quu = 2 * R * dt + fu.T @ Vxx_r @ fu + jnp.einsum('i,ijk->jk', Vx, Fuu[k])
            Qux = fu.T @ Vxx_r @ fx + jnp.einsum('i,ijk->jk', Vx, Fux[k])
            L = jnp.linalg.cholesky(Quu)
            bad = bad | jnp.any(jnp.isnan(L))
            if box:                                                    # <<BOX control-limited DDP (Tassa 2014)
                kf, free = box_qp(Quu, Qu, -umax - u, umax - u)
                Kk = -masked_solve(Quu, Qux, free)                     # clamped channels: zero feedback rows
            else:
                kf = -jax.scipy.linalg.cho_solve((L, True), Qu)
                Kk = -jax.scipy.linalg.cho_solve((L, True), Qux)
            dV1 = dV1 + kf @ Qu
            dV2 = dV2 + 0.5 * kf @ Quu @ kf
            Vx2 = Qx + Kk.T @ Quu @ kf + Kk.T @ Qu + Qux.T @ kf
            Vxx2 = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
            Vxx2 = 0.5 * (Vxx2 + Vxx2.T)
            return (Vx2, Vxx2, dV1, dV2, bad), (kf, Kk)

        Vx0 = 2 * QT @ (X[:, N] - Xref[:, N])
        (_, _, dV1, dV2, bad), (kff, Kk) = lax.scan(
            body, (Vx0, 2 * QT, 0.0, 0.0, False), jnp.arange(N - 1, -1, -1))
        return kff[::-1].T, Kk[::-1], dV1, dV2, bad

    @jax.jit
    def forward(x0, X, U, kff, Kk, Xref, mc, alphas):
        def one(a):
            def body(xn, k):
                uk = U[:, k] + a * kff[:, k] + Kk[k] @ (xn - X[:, k])
                un = jnp.clip(uk, -umax, umax)
                return _F(xn, un), (xn, un)
            xN, (Xs, Us) = lax.scan(body, x0, jnp.arange(N))
            Xn = jnp.vstack([Xs, xN[None, :]]).T
            Un = Us.T
            return Xn, Un, _traj_cost(Xn, Un, Xref, mc)
        return jax.vmap(one)(alphas)

    @jax.jit
    def costate0(X, U, Xref, mc):
        """Adjoint sweep -> joint p0 (7K), handed to the shared u = ustar(x_k, p_k) applicator."""
        Fx = jax.vmap(lambda x, u: jax.jacfwd(lambda xx: _F(xx, u))(x))(X[:, :N].T, U.T)

        def body(p, k):
            _, og = obs_all(X[pidx, k], mc)
            gx = jnp.zeros(n).at[pidx].set(og)
            return (2 * Q @ (X[:, k] - Xref[:, k]) + gx) * dt + Fx[k].T @ p, None

        pN = 2 * QT @ (X[:, N] - Xref[:, N])
        p0, _ = lax.scan(body, pN, jnp.arange(N - 1, -1, -1))
        return p0

    def warm(x0, Xref, mc):
        U = np.zeros((m, N))
        X, _ = rollout_cost(x0, U, Xref, mc)
        kff, Kk, _, _, _ = backward(X, U, Xref, mc, 1e-3)
        forward(x0, X, U, kff, Kk, Xref, mc, DDP_ALPHAS)
        costate0(X, U, Xref, mc)

    return SimpleNamespace(rollout_cost=rollout_cost, backward=backward, forward=forward, costate0=costate0,
                           warm=warm, n=n, m=m, N=N, dt=dt, K=K, obs_all=jax.jit(obs_all), pidx=np.asarray(pidx),
                           box=box, psd_barrier=psd_barrier)


# ======================================================================================================
#  MPPI on the joint model: one CUDA thread = one sample = the rollout of ALL K agents
# ======================================================================================================
_KERNEL_INT = r"""
#define KA __KA__
#define NX (7*KA)
#define NU (3*KA)

__device__ __forceinline__ void dyn7(const double* X, const double* u, double* F) {
    F[0] = X[4]*cos(X[3]); F[1] = X[4]*sin(X[3]); F[2] = X[6];
    F[3] = X[5]; F[4] = u[0]; F[5] = u[1]; F[6] = u[2];
}

extern "C" __global__
void mppi_rollout_int(
    const double* __restrict__ x0,        // (NX,)  joint state, agent-major
    const double* __restrict__ Unom,      // (NU*N,) nominal joint control, [t*NU + j]
    const double* __restrict__ eps,       // (K*N*NU,) noise [k*N*NU + t*NU + j]
    const double* __restrict__ Xref,      // (NX*(N+1),) joint tracking reference [t*NX + i]
    const double* __restrict__ oc,
    const double* __restrict__ oax, const double* __restrict__ oay, const double* __restrict__ oaz,
    const double* __restrict__ oW,  const double* __restrict__ ow,
    const double* __restrict__ mc,
    const double* __restrict__ max_, const double* __restrict__ may_, const double* __restrict__ maz_,
    const double* __restrict__ mW_,  const double* __restrict__ mw_,
    const double* __restrict__ Qd,        // (7,) diag(Q), same for every agent
    const double* __restrict__ QTd,       // (7,)
    const double* __restrict__ Rd,        // (3,)
    const double* __restrict__ umax,      // (NU,)
    const double* __restrict__ gsig,      // (NU,)
    double* __restrict__ Sout,            // (K,)
    int* __restrict__ Hit,                // (K,) OUT: 1 if any agent's rollout entered an obstacle ellipse
    const int K, const int N, const int Ko, const int Km,
    const double dt, const double eps_barrier, const int n_unbiased,
    const double Rc, const double Wc, const double wc, const double eps_pair)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    double x[NX];
    for (int i = 0; i < NX; ++i) x[i] = x0[i];
    double S = 0.0;
    int hit = 0;

    for (int t = 0; t < N; ++t) {
        double un[NU];
        for (int j = 0; j < NU; ++j) {
            const double e = eps[(size_t)k * N * NU + (size_t)t * NU + j];
            const double u = Unom[t * NU + j];
            const double v = (k < n_unbiased) ? e : (u + e);
            un[j] = fmin(fmax(v, -umax[j]), umax[j]);
            S += gsig[j] * u * v;
        }
        double q = 0.0;
        for (int a = 0; a < KA; ++a) {
            double* xa = x + 7 * a;
            const double* ua = un + 3 * a;
            // ---- RK4 step of dyn7 for agent a (decoupled dynamics) ----
            double k1[7], k2[7], k3[7], k4[7], xt[7];
            dyn7(xa, ua, k1);
            for (int i = 0; i < 7; ++i) xt[i] = xa[i] + 0.5*dt*k1[i];
            dyn7(xt, ua, k2);
            for (int i = 0; i < 7; ++i) xt[i] = xa[i] + 0.5*dt*k2[i];
            dyn7(xt, ua, k3);
            for (int i = 0; i < 7; ++i) xt[i] = xa[i] + dt*k3[i];
            dyn7(xt, ua, k4);
            for (int i = 0; i < 7; ++i) xa[i] += dt/6.0*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
            // ---- agent a stage cost at t+1: tracking + control + barriers (as the single-agent kernel) ----
            for (int i = 0; i < 7; ++i) { const double d = xa[i] - Xref[(t+1)*NX + 7*a + i]; q += Qd[i]*d*d; }
            for (int j = 0; j < 3; ++j) q += Rd[j]*ua[j]*ua[j];
            for (int i = 0; i < Ko; ++i) {
                const double dx = xa[0]-oc[0*Ko+i], dy = xa[1]-oc[1*Ko+i], dz = xa[2]-oc[2*Ko+i];
                const double rho = sqrt((dx/oax[i])*(dx/oax[i]) + (dy/oay[i])*(dy/oay[i])
                                      + (dz/oaz[i])*(dz/oaz[i]) + eps_barrier);
                q += oW[i]*0.5*(1.0 - tanh((rho-1.0)/ow[i]));
                hit |= (rho < 1.0);
            }
            for (int i = 0; i < Km; ++i) {
                const double dx = xa[0]-mc[0*Km+i], dy = xa[1]-mc[1*Km+i], dz = xa[2]-mc[2*Km+i];
                const double rho = sqrt((dx/max_[i])*(dx/max_[i]) + (dy/may_[i])*(dy/may_[i])
                                      + (dz/maz_[i])*(dz/maz_[i]) + eps_barrier);
                q += mW_[i]*0.5*(1.0 - tanh((rho-1.0)/mw_[i]));
                hit |= (rho < 1.0);
            }
        }
        // ---- pair collision penalty over i < j (positions after the step, like the barriers) ----
        for (int a = 0; a < KA; ++a)
            for (int b = a + 1; b < KA; ++b) {
                const double dx = x[7*a]-x[7*b], dy = x[7*a+1]-x[7*b+1], dz = x[7*a+2]-x[7*b+2];
                const double r = sqrt((dx*dx + dy*dy + dz*dz)/(Rc*Rc) + eps_pair);
                q += 0.5*Wc*(1.0 - tanh((r-1.0)/wc));
            }
        S += q * dt;
    }
    for (int a = 0; a < KA; ++a)
        for (int i = 0; i < 7; ++i) { const double d = x[7*a+i] - Xref[N*NX + 7*a + i]; S += QTd[i]*d*d; }
    Sout[k] = S;
    Hit[k] = hit;
}
"""


class MPPIInt(S.MPPI):
    """mpc_solvers.MPPI on the joint model of KA agents (sample count `K`, as in the parent).  The parent's
    lambda tuning, weighted update, Savitzky-Golay filter, shift and warm start are reused unchanged on the
    3KA-dimensional control; only the sample-cost kernel is replaced.  `_tune_lambda` uses the wide
    log10-lambda bracket [-6, 9] of the N-sweep runner (N_sweep/mpc_testbed_N_sweep.run_mppi)."""

    def __init__(self, M, P, N, K, KA, col, **kw):
        import cupy as cp
        Mj = SimpleNamespace(**vars(M))                                # joint control view of the model
        Mj.m = 3 * KA
        Mj.umax = np.tile(np.asarray(M.umax, float), KA)
        super().__init__(Mj, P, N, K, **kw)
        self.KA, self.col = KA, col
        self.kern = cp.RawKernel(_KERNEL_INT.replace("__KA__", str(int(KA))), "mppi_rollout_int")

    def _tune_lambda(self, Sc, rho):
        import cupy as cp
        nsub = min(self.K, 2048)
        d_h = cp.asnumpy(Sc[:nsub] - rho)
        lo, hi = -6.0, 9.0
        target = self.ess_target * nsub
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            w = np.exp(-d_h / (10.0 ** mid))
            w = w / w.sum()
            ess = 1.0 / np.sum(w ** 2)
            if ess < target:
                lo = mid
            else:
                hi = mid
        self.lam = 10.0 ** (0.5 * (lo + hi))
        self.gamma = self.lam * (1.0 - self.alpha)
        self.d_gsig = cp.asarray(np.ascontiguousarray(self.gamma / self.sigma ** 2, dtype=np.float64))
        return self.lam

    def _sample_costs(self, x, Xref, mc, U, eps):
        import cupy as cp
        c = self.col
        self.kern((self.grid,), (self.block,), (
            cp.asarray(x), cp.asarray(np.ascontiguousarray(U.T.ravel())), eps,
            cp.asarray(np.ascontiguousarray(Xref[:, :self.N + 1].T.ravel())),
            self.d_oc, self.d_oax, self.d_oay, self.d_oaz, self.d_oW, self.d_ow,
            cp.asarray(np.ascontiguousarray(mc.ravel())),
            self.d_max, self.d_may, self.d_maz, self.d_mW, self.d_mw,
            self.d_Q, self.d_QT, self.d_R, self.d_umax, self.d_gsig, self.d_S, self.d_Hit,
            np.int32(self.K), np.int32(self.N), np.int32(self.Ko), np.int32(self.Km),
            np.float64(self.P.dt), np.float64(self.eps_barrier), np.int32(self.n_unbiased),
            np.float64(c.Rc), np.float64(c.Wc), np.float64(c.wc), np.float64(c.eps)))
        return self.d_S


# ======================================================================================================
#  Host loop for Box-DDP: N-sweep ddp_run_counted + a damping cap (Tassa: lambdaMax = 1e10)
# ======================================================================================================
DDP_MU_MAX = 1e10


def ddp_run_box(IL, x, Xref, mc, Uinit, P, tc, mu0=1e-3, mu_max=DDP_MU_MAX, trace=None, stop=None):
    """Copy of N_sweep/mpc_testbed_N_sweep.ddp_run_counted (same mu schedule, acceptance and stopping rules)
    with ONE addition: the damping is capped -- if mu exceeds mu_max the cycle ends with status 'mu-max'
    instead of multiplying mu up to inf (which made the plain-DDP cycles burn the iteration cap).
    `trace` (list) records every iteration.  Returns (p0, U, it, info) like ddp_run_counted."""
    import time as _time
    N, m = IL.N, IL.m
    U = np.zeros((m, N)) if Uinit is None else Uinit
    X, J = IL.rollout_cost(x, U, Xref, mc)
    X, J = np.asarray(X), float(J)
    mu, dX = mu0, np.inf
    alphas = DDP_ALPHAS
    it = 0
    nbad = nrej = since_acc = 0
    dV1 = np.nan
    a_last, a_min = np.nan, np.nan
    status, conv_by = "cap", ""
    for it in range(1, P.ilqr_Kmax + 1):
        if it > 1 and (_time.perf_counter() - tc - (stop.t_spent if stop is not None else 0.0)) >= P.budget:
            status = "timeout"
            break
        if mu > mu_max:                                                # <<BOX damping cap
            status = "mu-max"
            break
        if stop is not None and stop(U):                               # SHARED rule (untimed; only on a new U)
            status, conv_by = "shared-stop", "dJ"
            break
        kff, Kk, dV1, Dv2, bad = IL.backward(X, U, Xref, mc, mu)
        if bool(bad):
            nbad += 1
            since_acc += 1
            if trace is not None:
                trace.append(dict(it=it, mu=mu, bad=True))
            mu = mu * 4
            continue
        dV1, Dv2 = float(dV1), float(Dv2)
        if stop is None and abs(dV1) < P.ilqr_tol:
            status, conv_by = "converged", "dV"
            if trace is not None:
                trace.append(dict(it=it, mu=mu, bad=False, dV1=dV1, dV2=Dv2, J=J, stop="dV1"))
            break
        Xn, Un, Jn = IL.forward(x, X, U, kff, Kk, Xref, mc, alphas)
        Jn = np.asarray(Jn)
        rec = dict(it=it, mu=mu, bad=False, dV1=dV1, dV2=Dv2, J=J, Jn=Jn.copy()) if trace is not None else None
        improved = False
        a_acc = np.nan
        for i in range(len(alphas)):
            if Jn[i] < J:
                a = float(alphas[i])
                Xa, Ua = np.asarray(Xn[i]), np.asarray(Un[i])
                dX = np.linalg.norm(Xa - X) / (1 + np.linalg.norm(X))
                if rec is not None:
                    rec.update(alpha=a, dJ=J - float(Jn[i]), expected=-(a * dV1 + a * a * Dv2), dX=dX)
                X, U, J = Xa, Ua, float(Jn[i])
                improved, a_acc = True, a
                break
        if rec is not None:
            trace.append(rec)
        if improved:
            since_acc = 0
            a_last = a_acc
            a_min = a_acc if not np.isfinite(a_min) else min(a_min, a_acc)
            mu = mu * 2.0 if a_acc < 0.5 else max(mu * 0.7, 1e-9)
            if stop is None and dX < P.ilqr_tol:
                status, conv_by = "converged", "dX"
                if rec is not None:
                    rec["stop"] = "dX"
                break
        else:
            nrej += 1
            since_acc += 1
            mu = mu * 4
    if not np.isfinite(J):
        status = "diverged"
    elif status == "cap" and since_acc >= 50:
        status = "stalled"
    p0 = np.asarray(IL.costate0(X, U, Xref, mc))
    info = SimpleNamespace(status=status, conv_by=conv_by, nbad=nbad, nrej=nrej, n_eff=it - nbad,
                           mu=float(mu), dV1=float(dV1) if np.isfinite(dV1) else np.nan,
                           dX=float(dX) if np.isfinite(dX) else np.nan, a_last=a_last, a_min=a_min, J=J)
    return p0, U, it, info


# ======================================================================================================
#  COLLOCATION-IPOPT on the joint model (copy of N_sweep build_collocation_ipopt_nobudget; changes <<K)
# ======================================================================================================
def build_collocation_int(M, K, N, dt, col, max_iter, stop=None):
    """RK4 multiple-shooting NLP of the joint problem: X (7K x N+1), U (3K x N), per-agent dynamics defects,
    cost = sum_k [tracking + u'Ru + static/moving barriers] dt + terminal + sum_{i<j} C(q_i, q_j) dt   <<K
    (the moving centres per stage are carried in rows n0.. of each agent's reference window, as the
    shooting methods).  Box bounds on U.  stop: core/shared_stop.SharedStop through the IPOPT iteration
    callback (IPOPT's own test pushed out of reach), as main.  call(xc, XrK, W0, tc) with XrK the sub=1
    per-agent windows (K, n0+3nm, N+1) -> (f, u0 (3K,), w)."""
    import casadi as ca
    import mpc_testbed as T
    n0, m0 = M.n, M.m
    n, m = K * n0, K * m0
    nm = M.mobs.c0.shape[1]
    Q, QT, R = ca.DM(M.Q), ca.DM(M.QT), ca.DM(M.R)

    def dyn(x, u):
        th = x[3]
        return ca.vertcat(x[4] * ca.cos(th), x[4] * ca.sin(th), x[6], x[5], u[0], u[1], u[2])

    def rk4(x, u):
        k1 = dyn(x, u); k2 = dyn(x + dt / 2 * k1, u)
        k3 = dyn(x + dt / 2 * k2, u); k4 = dyn(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def barrier(pos, Cc, ax, ay, az, W, w, eps):
        pen = 0
        for i in range(Cc.shape[1]):
            dd = pos - Cc[:, i]
            rho = ca.sqrt((dd[0] / ax[i]) ** 2 + (dd[1] / ay[i]) ** 2 + (dd[2] / az[i]) ** 2 + eps)
            pen = pen + W[i] * 0.5 * (1 - ca.tanh((rho - 1) / w[i]))
        return pen

    xk, uk = ca.SX.sym("x", n0), ca.SX.sym("u", m0)
    FP = ca.Function("FP", [xk, uk], [rk4(xk, uk)])
    par = ca.MX.sym("par", n)
    Xr = ca.MX.sym("Xr", K * (n0 + 3 * nm), N + 1)                    # <<K agent blocks stacked row-wise
    X = ca.MX.sym("X", n, N + 1)
    U = ca.MX.sym("U", m, N)
    o = M.obs
    J = 0
    D = []
    for a in range(K):                                                 # <<K per-agent cost and defects
        Xa, Ua = X[a * n0:(a + 1) * n0, :], U[a * m0:(a + 1) * m0, :]
        Ra = Xr[a * (n0 + 3 * nm):(a + 1) * (n0 + 3 * nm), :]
        D.append(Xa[:, 1:] - FP.map(N)(Xa[:, :N], Ua))
        for k in range(N):
            dxk = Xa[:, k] - Ra[:n0, k]
            mck = ca.reshape(Ra[n0:, k], 3, nm)
            pen = (barrier(Xa[:3, k], ca.DM(o.center), o.ax, o.ay, o.az, o.W, o.w, o.eps)
                   + barrier(Xa[:3, k], mck, M.mobs.ax, M.mobs.ay, M.mobs.az, M.mobs.W, M.mobs.w, M.mobs.eps))
            J = J + (dxk.T @ Q @ dxk + Ua[:, k].T @ R @ Ua[:, k] + pen) * dt
        dxT = Xa[:, N] - Ra[:n0, N]
        J = J + dxT.T @ QT @ dxT
    for k in range(N):                                                 # <<K pair collision penalty
        for i in range(K):
            for j in range(i + 1, K):
                dq = X[i * n0:i * n0 + 3, k] - X[j * n0:j * n0 + 3, k]
                r = ca.sqrt(ca.sumsqr(dq) / col.Rc ** 2 + col.eps)
                J = J + 0.5 * col.Wc * (1 - ca.tanh((r - 1) / col.wc)) * dt
    G = ca.vertcat(X[:, 0] - par, *[ca.reshape(Dd, -1, 1) for Dd in D])
    nx = n * (N + 1) + m * N
    nG = n * (N + 1)
    npar = n + K * (n0 + 3 * nm) * (N + 1)
    cb = T._BudgetCB("budget_cb_int", nx, nG, npar)
    opts = {"ipopt.max_iter": int(max_iter), "ipopt.tol": 1e-8, "ipopt.print_level": 0,
            "print_time": 0, "error_on_fail": False, "iteration_callback": cb}
    if stop is not None:
        cb.stop, cb.ulay = stop, (n, N, m)
        opts["ipopt.tol"] = 1e-14
        opts["ipopt.acceptable_iter"] = 0
    s = ca.nlpsol("s", "ipopt", {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
                                 "f": J, "g": G, "p": ca.vertcat(par, ca.reshape(Xr, -1, 1))}, opts)
    umax = np.tile(np.asarray(M.umax, float), K)
    lbx = np.concatenate([-np.inf * np.ones(n * (N + 1)), np.tile(-umax, N)])
    ubx = np.concatenate([np.inf * np.ones(n * (N + 1)), np.tile(umax, N)])

    def call(xc, XrK, W0, tc):
        if W0 is None:
            X0 = np.tile(xc.reshape(-1, 1), (1, N + 1))
            W0 = np.concatenate([X0.flatten(order="F"), np.zeros(m * N)])
        cb.t0, cb.deadline, cb.calls = tc, np.inf, 0
        Xrv = XrK.reshape(-1, XrK.shape[2])                            # (K*(n0+3nm), N+1), agent blocks
        r = s(x0=W0, p=np.concatenate([xc, Xrv.flatten(order="F")]),
              lbg=np.zeros(nG), ubg=np.zeros(nG), lbx=lbx, ubx=ubx)
        w = np.asarray(r["x"]).ravel()
        Uo = w[n * (N + 1):].reshape((m, N), order="F")
        return float(r["f"]), Uo[:, 0], w

    call.cb, call.solver, call.n, call.m, call.N = cb, s, n, m, N
    return call

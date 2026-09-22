"""
mpc_core.py -- Python port of the model / dynamics / cost / shooting layer of mpc_compare.m.

WHY THIS FILE EXISTS SEPARATELY FROM mpc_compare.py
    mpc_compare.py is the runnable experiment (setup + runners + main). This module is everything the
    experiment is built OUT of: the 7-state vehicle, the tanh obstacle barriers, the RRT reference, and
    the JAX shooting object. verify_vs_matlab.py imports THIS module to check each piece against MATLAB
    fixtures one function at a time, which is only possible if the pieces are importable on their own.

PORTING CONTRACT (read before editing)
    Every function here is a line-by-line port of its MATLAB twin in mpc_compare.m and must stay
    numerically identical to it. Where MATLAB writes an expression in a roundabout way (e.g. ustar's
    branch-free min/max), the roundabout form is reproduced verbatim rather than "cleaned up", because
    the clean form can differ in the last bits and parity is the whole point of the port.

BACKEND
    JAX on CPU, float64 (jax_enable_x64). CPU is not a compromise: the state is 7-dimensional, so a GPU
    would lose to kernel-launch overhead. The reason JAX is here at all is that a plain NumPy RK4
    rollout of 190 steps costs ~2.3 ms -- slower than MATLAB -- while the same rollout fused into one
    jit'd lax.scan is ~100x cheaper. The rule that follows: the ENTIRE rollout must live inside one jit.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

# x64 must be set before jax.numpy is touched; without it every parity check below fails at ~1e-8.
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax                                                          # noqa: E402
jax.config.update("jax_enable_x64", True)                           # float64 everywhere, like MATLAB
jax.config.update("jax_platform_name", "cpu")                       # see BACKEND note above
import jax.numpy as jnp                                             # noqa: E402
from jax import lax                                                 # noqa: E402


# ======================================================================================================
#  PARAMETERS  (mpc_compare.m lines ~35-86).  Values are copied verbatim; comments explain intent only
#  where the MATLAB source explains it, so this file does not drift from the original's reasoning.
# ======================================================================================================
def make_P(smoke: bool = False) -> SimpleNamespace:
    P = SimpleNamespace()
    P.rt_hz = 50
    P.dt = 0.005
    P.dt_apply = 1.0 / P.rt_hz
    P.napply = round(P.dt_apply / P.dt)                              # 10 fine steps per applied control
    P.umax = np.array([3.0, 3.0, 3.0])
    P.budget = 0.020                                                 # HARD 20 ms/cycle wall-clock cut

    # Per-method horizons (Tp = N*dt), calibrated in MATLAB so mean ms/cycle ~ 20. The Python port is
    # faster, so these are NOT the right horizons here -- recalibrating them is a tuning task, done in
    # mpc_compare_tuned.py. They are kept at the MATLAB values so parity can be checked first.
    P.N_pmp, P.N_m1, P.N_ilqr, P.N_coll = 190, 120, 85, 45
    P.N_ddp = 60                                                     # DDP: dearer per iteration (Hessian
    # tensors), so a shorter horizon than iLQR; recalibrated in calibrate_horizons.py / mpc_tuned_params.

    # ---- MPPI (GPU) ----
    # Sized from the measured (K,N) sweep on this RTX 5070, NOT guessed: K is nearly free below ~8192
    # (K=256 and K=8192 both cost ~3 ms at N=100 -- the GPU is not saturated), while cost grows linearly
    # in N because the horizon is sequential. K=8192/N=200 measures 6.2 ms; K=8192/N=400 measures 12.6 ms.
    # N=200 is the default so there is headroom; raise it toward 400 when tuning.
    P.K_mppi, P.N_mppi = 8192, 200
    # lam is scaled to the SPREAD of the sample costs (std(S_k) ~ 20 here), not their magnitude: at
    # lam=1 the softmax collapses to ESS=1 (random shooting), at lam=100 it is 96% uniform (no
    # selectivity). lam=20 measures ~4% ESS, inside the healthy 1-10% band.
    P.lam_mppi = 20.0                                                # only used if ess_target is disabled
    P.ess_target_mppi = 0.05                                         # auto-tune lam to 5% ESS per cycle
    P.alpha_mppi = 0.01                                              # ~1% zero-mean reset samples
    P.iters_mppi = 1                                                 # 1 = canonical MPPI (warm start carries)

    P.p0 = np.array([0.0, 0.0, 0.5])
    P.pgoal = np.array([20.0, 0.0, 0.5])
    P.reachtol = 0.30
    P.vcr = 1.5                                                      # cruise speed the reference is drawn at

    # ---- STATIC obstacle: obs-1, the on-axis z-bump ----
    # Centre is BELOW the cruise altitude (cz=0 vs z=0.5) so the push is definitely upward (no z-saddle);
    # ay=1.5 spans the corridor so a y-dodge is expensive and the cheap escape is the z-climb. The (z,vz)
    # channel is fully decoupled from (x,y,theta,v,omega), so climbing perturbs y/theta/omega by exactly
    # zero -- the vehicle reaches obs-2 with its antisymmetric state pristine, which is what lets PMP
    # re-lock onto the symmetric root and trap there.
    P.obs = SimpleNamespace(
        center=np.array([[3.0], [0.25], [0.0]]),
        rphys=np.array([0.45]),
        ax=np.array([0.75]), ay=np.array([1.50]), az=np.array([0.75]),
        W=np.array([200.0]), w=np.array([0.10]), eps=1e-10,
    )

    # ---- MOVING obstacles (columns): #1 = obs-2 head-on wall, #2 = obs-3 rising gate ----
    # obs-2 is y-symmetric at EVERY t, so the warm-started on-axis root stays the tracked root until the
    # closing gap enters the horizon -- the index-1 saddle then forms directly UNDER the current iterate.
    # That is capture by construction, and it is the centrepiece of the demo.
    P.mobs = SimpleNamespace(
        c0=np.array([[13.0, 13.0], [0.0, -10.5], [0.5, 0.5]]),
        vel=np.array([[-0.25, 0.0], [0.0, 0.9], [0.0, 0.0]]),
        ax=np.array([0.8, 0.6]), ay=np.array([0.2, 0.9]), az=np.array([5.0, 5.0]),
        W=np.array([500.0, 450.0]), w=np.array([0.25, 0.12]), eps=1e-10,
    )

    # ---- PLANNER obstacle model (RRT only) ----
    # Deliberately WRONG about obs-2 (placed off-axis at (8,1.2)) so the RRT reference runs straight
    # through the true wall, and carries a planner-only gate at x=6 that forces the reference back onto
    # y=0 before obs-2. The controllers never see these; only the planner does.
    P.obs_plan = SimpleNamespace(
        center=np.array([[3.0, 8.0, 10.0, 6.0, 6.0],
                         [0.90, 1.2, -1.5, 0.62, -0.62],
                         [0.5, 0.5, 0.5, 0.5, 0.5]]),
        rinfl=np.array([0.80, 0.60, 0.60, 0.56, 0.56]),
    )

    P.seed = 1
    P.rrt = SimpleNamespace(step=0.5, gb=0.15, goalrad=0.5, maxit=40000,
                            spacing=0.05, collstep=0.05, pad=np.array([1.5, 2.5, 1.0]))
    P.ilqr_Kmax, P.ilqr_tol = 50, 1e-4
    P.lm_tol, P.lm_maxit = 1e-4, 50
    # epsc must sit BELOW the cruise lmin (+1.9e-3, so convex reads stay convex) and ABOVE the saddle
    # depth. At Tp=0.6 the pop-up saddle is shallow, so the MATLAB default of 0.5 could never trigger.
    P.M1_epsc, P.M1_alpha = 1e-3, 0.30
    P.maxcyc = 60 if smoke else 800
    return P


def model_scn(P: SimpleNamespace) -> SimpleNamespace:
    """Port of model_scn(). Q_y=0.4 makes a lateral dodge affordable; Q_z=2 is kept high on purpose --
    a cheap Q_z lets every method climb-escape obs-2 and destroys the trap."""
    M = SimpleNamespace()
    M.name, M.n, M.m = "SCN", 7, 3
    M.posidx = np.array([0, 1, 2])                                   # MATLAB [1 2 3], 0-based here
    M.umax = P.umax
    M.Q = np.diag([2.0, 0.4, 2.0, 0.5, 0.3, 0.3, 0.3])
    M.QT = np.diag([10.0, 2.0, 10.0, 2.0, 1.0, 1.0, 1.0])
    M.R = 0.05 * np.eye(3)
    M.vcr = P.vcr
    M.obs, M.mobs = P.obs, P.mobs
    M.x0 = np.array([P.p0[0], P.p0[1], P.p0[2], 0.0, P.vcr, 0.0, 0.0])
    return M


def mobs_center(mobs: SimpleNamespace, t: float) -> np.ndarray:
    """Port of mobs_center(). Centres translate linearly in t; obs-3's gap to the y=0 reference SHRINKS
    with time, so a method delayed at obs-2 arrives to find the gate closed."""
    return mobs.c0 + mobs.vel * t


def augobs(M: SimpleNamespace, t: float) -> SimpleNamespace:
    """Port of augobs(): static obstacles + the moving ones frozen at their CURRENT position. This is the
    'static-at-current-position' prediction every method gets -- iLQR via this combined set, the shooting
    methods via the moving-centre rows carried inside Xr. Fair by construction: nobody is told the
    obstacle's velocity."""
    mc = mobs_center(M.mobs, t)
    return SimpleNamespace(
        center=np.hstack([M.obs.center, mc]),
        ax=np.concatenate([M.obs.ax, M.mobs.ax]),
        ay=np.concatenate([M.obs.ay, M.mobs.ay]),
        az=np.concatenate([M.obs.az, M.mobs.az]),
        W=np.concatenate([M.obs.W, M.mobs.W]),
        w=np.concatenate([M.obs.w, M.mobs.w]),
        eps=M.obs.eps,
        rphys=getattr(M.obs, "rphys", None),
    )


# ======================================================================================================
#  JAX PRIMITIVES.  build_jax() closes over the numeric constants and returns pure functions, mirroring
#  how the MATLAB code bakes M into its CasADi Functions. Everything below is jit-able and differentiable.
# ======================================================================================================
def build_jax(M: SimpleNamespace, obs=None):
    """Return a namespace of pure JAX functions for the model M, with barriers from `obs`
    (defaults to M.obs). iLQR passes an augmented obstacle set here each cycle."""
    obs = M.obs if obs is None else obs
    n, m = M.n, M.m
    Q, QT, R = jnp.asarray(M.Q), jnp.asarray(M.QT), jnp.asarray(M.R)
    umax = jnp.asarray(M.umax)
    rdiag = jnp.diag(jnp.asarray(M.R))
    pos_i = jnp.asarray(M.posidx)

    # static barrier constants
    oc, oax, oay = jnp.asarray(obs.center), jnp.asarray(obs.ax), jnp.asarray(obs.ay)
    oaz, oW, ow = jnp.asarray(obs.az), jnp.asarray(obs.W), jnp.asarray(obs.w)
    oeps = obs.eps
    # moving barrier constants (shape/weight only; centres arrive per-call)
    mp = M.mobs
    max_, may_, maz_ = jnp.asarray(mp.ax), jnp.asarray(mp.ay), jnp.asarray(mp.az)
    mW_, mw_, meps_ = jnp.asarray(mp.W), jnp.asarray(mp.w), mp.eps
    Kmob = mp.c0.shape[1]

    def dyn7(x, u):
        """f = [v cos th, v sin th, vz, omega, u1, u2, u3]. The (z,vz) channel is decoupled from the
        horizontal one -- that decoupling is load-bearing for the demo (see make_P obs-1 note)."""
        th = x[3]
        return jnp.array([x[4] * jnp.cos(th), x[4] * jnp.sin(th), x[6], x[5], u[0], u[1], u[2]])

    def ustar(x, p):
        """Port of ustar(). The branch-free min/max form is copied verbatim from MATLAB rather than
        replaced by clip(): identical FP ops are what keep the parity check exact."""
        uu = -p[4:7] / (2.0 * rdiag)
        mn = 0.5 * ((uu + umax) - jnp.abs(uu - umax))                # min(uu, umax)
        return 0.5 * ((mn - umax) + jnp.abs(mn + umax))              # max(mn, -umax)

    def obstacle(pos):
        """Static ellipsoidal tanh barriers -> (penalty, dpenalty/dpos). Branch-free by construction,
        which is also why it is cheap inside the MPPI CUDA kernel later."""
        dd = pos[:, None] - oc                                       # (3,K)
        rho = jnp.sqrt((dd[0] / oax) ** 2 + (dd[1] / oay) ** 2 + (dd[2] / oaz) ** 2 + oeps)
        zz = (rho - 1.0) / ow
        pen = jnp.sum(oW * 0.5 * (1.0 - jnp.tanh(zz)))
        s2 = 1.0 - jnp.tanh(zz) ** 2
        grho = jnp.stack([dd[0] / oax ** 2, dd[1] / oay ** 2, dd[2] / oaz ** 2]) / rho
        g = jnp.sum((-0.5 * oW * s2 / ow) * grho, axis=1)
        return pen, g

    def mobs_pen(pos, C):
        """Same tanh barrier as obstacle(), but centres C (3,K) are supplied per stage."""
        dd = pos[:, None] - C
        rho = jnp.sqrt((dd[0] / max_) ** 2 + (dd[1] / may_) ** 2 + (dd[2] / maz_) ** 2 + meps_)
        zz = (rho - 1.0) / mw_
        pen = jnp.sum(mW_ * 0.5 * (1.0 - jnp.tanh(zz)))
        s2 = 1.0 - jnp.tanh(zz) ** 2
        grho = jnp.stack([dd[0] / max_ ** 2, dd[1] / may_ ** 2, dd[2] / maz_ ** 2]) / rho
        g = jnp.sum((-0.5 * mW_ * s2 / mw_) * grho, axis=1)
        return pen, g

    def dHdx7(x, p, xr, mc):
        """Costate RHS: dH/dx. Port of dHdx7()."""
        th, v = x[3], x[4]
        bg = 2.0 * Q @ (x - xr)
        _, g = obstacle(x[pos_i])
        _, gm = mobs_pen(x[pos_i], mc)
        gt = g + gm
        extra = jnp.array([0.0, 0.0, 0.0,
                           v * (-p[0] * jnp.sin(th) + p[1] * jnp.cos(th)),
                           p[0] * jnp.cos(th) + p[1] * jnp.sin(th),
                           p[3], p[2]])
        return bg + jnp.concatenate([gt, jnp.zeros(4)]) + extra

    def rk4_state(x, u, dt):
        k1 = dyn7(x, u); k2 = dyn7(x + dt / 2 * k1, u)
        k3 = dyn7(x + dt / 2 * k2, u); k4 = dyn7(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _split_ref(xa):
        """Xr column -> (tracking ref, moving-obstacle centres). MATLAB reshape(xa(n+1:end),3,K) is
        column-major, so the (K,3).T form here is the exact equivalent."""
        return xa[:n], xa[n:].reshape((Kmob, 3)).T

    def fz_track(z, xa):
        x, p = z[:n], z[n:]
        xr, mc = _split_ref(xa)
        return jnp.concatenate([dyn7(x, ustar(x, p)), -dHdx7(x, p, xr, mc)])

    def rk4_canon(z, ra, rm, rb, dt):
        k1 = fz_track(z, ra); k2 = fz_track(z + dt / 2 * k1, rm)
        k3 = fz_track(z + dt / 2 * k2, rm); k4 = fz_track(z + dt * k3, rb)
        return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def fz_aug(za, xa):
        """Canonical RHS augmented with the running-cost accumulator (last element), so one rollout
        yields the state, the costate AND the cost integral -- this is what makes S.all a single pass."""
        x, p = za[:n], za[n:2 * n]
        xr, mc = _split_ref(xa)
        u = ustar(x, p)
        fdyn = dyn7(x, u)
        po, _ = obstacle(x[pos_i])
        pm, _ = mobs_pen(x[pos_i], mc)
        Lhl = (x - xr) @ Q @ (x - xr) + u @ R @ u + po + pm          # Hctrl - p'f, i.e. the running cost
        return jnp.concatenate([fdyn, -dHdx7(x, p, xr, mc), jnp.array([Lhl])])

    def rk4_canon_aug(za, ra, rm, rb, dt):
        k1 = fz_aug(za, ra); k2 = fz_aug(za + dt / 2 * k1, rm)
        k3 = fz_aug(za + dt / 2 * k2, rm); k4 = fz_aug(za + dt * k3, rb)
        return za + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    J = SimpleNamespace(n=n, m=m, dyn7=dyn7, ustar=ustar, obstacle=obstacle, mobs_pen=mobs_pen,
                        dHdx7=dHdx7, rk4_state=rk4_state, fz_track=fz_track, rk4_canon=rk4_canon,
                        fz_aug=fz_aug, rk4_canon_aug=rk4_canon_aug, Q=Q, QT=QT, R=R, umax=umax,
                        pos_i=pos_i, Kmob=Kmob)
    return J


# ======================================================================================================
#  SHOOTING OBJECT  (port of build_ss).  Xr has n+3K rows: 1..n tracking ref, then the K moving-obstacle
#  centres per sub-stage.  Returns res/J/Jr/Sx and the fused `all`, matching the MATLAB CasADi object.
# ======================================================================================================
def _value_and_jacfwd(f, x):
    """Value AND full forward-mode Jacobian in ONE vmapped pass.

    jax.jacfwd would recompute the primal and throw it away; here the primal comes out of the same jvp
    sweep. This is the direct analogue of MATLAB's fused S.all (one rollout -> res, J, Jr, Sx), and the
    reason M1's hot loop costs ~1 rollout per iteration instead of 4-5.
    """
    basis = jnp.eye(x.shape[0], dtype=x.dtype)
    y, jac_cols = jax.vmap(lambda v: jax.jvp(f, (x,), (v,)), out_axes=(None, 0))(basis)
    return y, jac_cols.T                                             # jac_cols[i] = J @ e_i -> transpose


def build_ss(M: SimpleNamespace, N: int, dt: float, JX=None, napply: int = 10):
    """Shooting object for horizon N. Mirrors build_ss() in mpc_compare.m.

    `napply` is the number of sub-steps the applied control is held for; it sizes roll_costate (the
    warm-start seed) and MUST match P.napply. It is a build-time constant because it fixes a scan length.
    """
    JX = build_jax(M) if JX is None else JX
    n = M.n
    QT = JX.QT

    def _shoot(p0, xc, Xr):
        """One canonical rollout over the horizon, fused inside a single lax.scan.

        scan (not a Python loop) because a 190-step unroll compiles for tens of seconds; scan compiles in
        ~1 s and, on CPU where there are no kernel launches, costs essentially nothing at runtime.
        """
        za0 = jnp.concatenate([xc, p0, jnp.zeros(1)])
        RA = Xr[:, 0:2 * N:2].T                                      # sub-stage a: cols 0,2,...,2N-2
        RM = Xr[:, 1:2 * N:2].T                                      # sub-stage m: cols 1,3,...,2N-1
        RB = Xr[:, 2:2 * N + 1:2].T                                  # sub-stage b: cols 2,4,...,2N

        def body(za, tr):
            return JX.rk4_canon_aug(za, tr[0], tr[1], tr[2], dt), None

        zaN, _ = lax.scan(body, za0, (RA, RM, RB))
        return zaN

    def _outs(p0, xc, Xr):
        za = _shoot(p0, xc, Xr)
        xN, pN, cc = za[:n], za[n:2 * n], za[2 * n]
        dxT = xN - Xr[:n, -1]
        Jc = cc + dxT @ QT @ dxT                                     # Mayer + Lagrange
        res = pN - 2.0 * QT @ dxT                                    # transversality residual: the PMP root
        return res, xN, Jc

    @jax.jit
    def S_res(p0, xc, Xr):
        return _outs(p0, xc, Xr)[0]

    @jax.jit
    def S_J(p0, xc, Xr):
        return _outs(p0, xc, Xr)[2]

    @jax.jit
    def S_resx(p0, xc, Xr):
        """(res, xN) from ONE rollout, NO Jacobians -- the cheap primitive for the quasi-Newton variant's
        Broyden updates (needs the residual AND the terminal state, but not Sx/Jr). ~1 rollout vs S.all's
        ~7. New output; the verified res/J/Jr/Sx/all functions are untouched, so parity is unaffected."""
        res, xN, _ = _outs(p0, xc, Xr)
        return res, xN

    @jax.jit
    def S_all(p0, xc, Xr):
        """Fused [res, J, Jr, Sx] from ONE rollout -- the hot path for PMP and M1."""
        def f(p):
            res, xN, Jc = _outs(p, xc, Xr)
            return jnp.concatenate([res, xN, jnp.array([Jc])])
        val, jac = _value_and_jacfwd(f, p0)                          # jac: (2n+1, n)
        return val[:n], val[2 * n], jac[:n, :], jac[n:2 * n, :]      # res, J, Jr, Sx

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
        """Predicted STATE trajectory over the horizon (animation logging; mirrors roll_pred in MATLAB)."""
        za0 = jnp.concatenate([xc, p0, jnp.zeros(1)])
        RA, RM = Xr[:, 0:2 * N:2].T, Xr[:, 1:2 * N:2].T
        RB = Xr[:, 2:2 * N + 1:2].T

        def body(za, tr):
            za2 = JX.rk4_canon_aug(za, tr[0], tr[1], tr[2], dt)
            return za2, za2[:n]

        _, Xs = lax.scan(body, za0, (RA, RM, RB))
        return jnp.vstack([xc[None, :], Xs]).T                       # (n, N+1)

    @jax.jit
    def roll_costate(p0, xc, Xr):
        """Costate over the first `napply` sub-steps -- the warm-start seed for the next cycle.

        This runs INSIDE the timed section, so its length is not cosmetic. It rolls napply steps (10),
        NOT the full horizon: only the costate at step napply is ever read (`seed = Pc[:, napply]`),
        because the vehicle only advances napply sub-steps per cycle. An earlier version of this port
        rolled all N steps and then threw ~94% of the result away -- at N=180 that is 18x the necessary
        work, charged to PMP and M1 on every cycle while MATLAB's roll_costate (ns=napply) never paid it.

        Uses rk4_canon (the 14-dim z=[x;p]) rather than rk4_canon_aug (15-dim): the extra component is
        the running-cost accumulator, which never feeds back into x or p, so dropping it is bit-identical
        for the seed and skips a barrier-penalty evaluation per RK4 stage. This matches MATLAB, which
        also calls rk4_canon here.
        """
        z0 = jnp.concatenate([xc, p0])
        RA = Xr[:, 0:2 * napply:2].T                                 # cols 0,2,...,2*napply-2
        RM = Xr[:, 1:2 * napply:2].T                                 # cols 1,3,...,2*napply-1
        RB = Xr[:, 2:2 * napply + 1:2].T                             # cols 2,4,...,2*napply

        def body(z, tr):
            z2 = JX.rk4_canon(z, tr[0], tr[1], tr[2], dt)
            return z2, z2[n:]

        _, Ps = lax.scan(body, z0, (RA, RM, RB))
        return jnp.vstack([p0[None, :], Ps]).T                       # (n, napply+1)

    def warm(xc, Xr):
        """Force-compile every jit'd entry point BEFORE the timed loop.

        This is not a micro-optimisation, it is a correctness requirement for the experiment: JAX
        compiles lazily on first call per (function, shape), so an un-warmed solver pays a ~0.7 s compile
        INSIDE a cycle. That both destroys the timing statistics and blows the 20 ms budget, causing the
        cut to fire on a compile rather than on real solver work.
        """
        p = np.zeros(n)
        S_res(p, xc, Xr); S_J(p, xc, Xr); S_all(p, xc, Xr); S_resx(p, xc, Xr)
        S_Jr(p, xc, Xr); S_Sx(p, xc, Xr); S_g(p, xc, Xr)
        roll_pred(p, xc, Xr); roll_costate(p, xc, Xr)

    return SimpleNamespace(res=S_res, J=S_J, all=S_all, resx=S_resx, Jr=S_Jr, Sx=S_Sx, g=S_g,
                           roll_pred=roll_pred, roll_costate=roll_costate, warm=warm, N=N, dt=dt, JX=JX)


# ======================================================================================================
#  REFERENCE WINDOW + RRT  (ports of ref_window / plan_path and helpers).  Host-side NumPy: these run
#  once per cycle on small arrays, so jit would only add dispatch overhead.
# ======================================================================================================
def ref_window(xcur, N, path, M, P, sub, t):
    """Rows 0..n-1: tracking ref along the path at cruise vcr. Rows n.. : the K moving-obstacle centres,
    HELD at their position at the current cycle time t across the whole horizon (static-at-current
    prediction -- nobody gets the obstacle's velocity)."""
    npos = len(M.posidx)
    pos = xcur[M.posidx]
    d = np.sum((path.pts[:npos, :] - pos[:, None]) ** 2, axis=0)
    i0 = int(np.argmin(d))                                            # MATLAB min -> first index on ties
    s0 = path.s[i0]
    nm = M.mobs.c0.shape[1]
    Ns = sub * N
    h = P.dt / sub
    Xr = np.zeros((M.n + 3 * nm, Ns + 1))
    ds = max(2 * P.rrt.spacing, 1e-3)
    mc = mobs_center(M.mobs, t)
    j = np.arange(Ns + 1)
    sk = np.minimum(s0 + j * P.vcr * h, path.len)
    skp = np.minimum(sk + ds, path.len)
    pk = np.vstack([np.interp(sk, path.s, path.pts[r, :]) for r in range(npos)])
    thk = np.interp(sk, path.s, path.th)
    thp = np.interp(skp, path.s, path.th)
    om = P.vcr * (thp - thk) / np.maximum(skp - sk, 1e-9)
    z2 = np.interp(skp, path.s, path.pts[2, :])
    vz = P.vcr * (z2 - pk[2]) / np.maximum(skp - sk, 1e-9)
    Xr[0, :], Xr[1, :], Xr[2, :] = pk[0], pk[1], pk[2]                # build_ref: [pos; th; vcr; om; vz]
    Xr[3, :], Xr[4, :], Xr[5, :], Xr[6, :] = thk, P.vcr, om, vz
    Xr[M.n:, :] = mc.reshape(-1, 1, order="F")                        # column-major, matching mc(:)
    return Xr


def _seg_free(a, b, obs, hstep):
    L = np.linalg.norm(b - a)
    ns = max(2, int(np.ceil(L / hstep)))
    for j in range(ns + 1):
        q = a + (b - a) * (j / ns)
        if np.any(np.linalg.norm(q[:, None] - obs.center, axis=0) <= obs.rinfl):
            return False
    return True


def _shortcut(wp, obs, P):
    changed, passes = True, 0
    while changed and passes < 20 and wp.shape[1] > 2:
        changed, passes = False, passes + 1
        i = 0
        while i <= wp.shape[1] - 3:
            if _seg_free(wp[:, i], wp[:, i + 2], obs, P.rrt.collstep):
                wp = np.delete(wp, i + 1, axis=1); changed = True
            else:
                i += 1
    return wp


def _smooth_path(pts, obs, iters):
    K = obs.center.shape[1]
    for _ in range(iters):
        q = pts.copy()
        for i in range(1, pts.shape[1] - 1):
            cand = 0.5 * pts[:, i] + 0.25 * pts[:, i - 1] + 0.25 * pts[:, i + 1]
            ok = True
            for kk in range(K):
                if np.linalg.norm(cand - obs.center[:, kk]) <= obs.rinfl[kk] + 0.05:
                    ok = False; break
            if ok:
                q[:, i] = cand
        pts = q
    return pts


def _resample(wp, sp):
    d = np.sqrt(np.sum(np.diff(wp, axis=1) ** 2, axis=0))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-9:
        return wp
    sq = np.arange(0, s[-1], sp)
    if sq[-1] < s[-1]:
        sq = np.append(sq, s[-1])
    return np.vstack([np.interp(sq, s, wp[r, :]) for r in range(wp.shape[0])])


def _headings(pts):
    Mn = pts.shape[1]
    th = np.zeros(Mn)
    for k in range(Mn):
        if k == 0:
            dv = pts[:, 1] - pts[:, 0]
        elif k == Mn - 1:
            dv = pts[:, Mn - 1] - pts[:, Mn - 2]
        else:
            dv = pts[:, k + 1] - pts[:, k - 1]
        th[k] = np.arctan2(dv[1], dv[0])
    return np.unwrap(th)


def plan_path(p_init, p_goal, obs, P):
    """RRT on the (partly inaccurate) planner model.

    NOTE ON DETERMINISM: MATLAB's rng(1) Mersenne Twister stream does NOT match NumPy's, so this planner
    produces a DIFFERENT path from the MATLAB one even at the same seed. That is expected and harmless
    for production, but it means parity checks must load MATLAB's path instead of re-planning -- see
    verify_vs_matlab.py.
    """
    rng = np.random.default_rng(P.seed)
    dim = p_init.size
    lo, hi = np.minimum(p_init, p_goal) - P.rrt.pad, np.maximum(p_init, p_goal) + P.rrt.pad
    V = p_init.reshape(-1, 1)
    parent = [0]
    goal_idx = -1
    for _ in range(P.rrt.maxit):
        xs = p_goal if rng.random() < P.rrt.gb else lo + (hi - lo) * rng.random(dim)
        d = np.sum((V - xs[:, None]) ** 2, axis=0)
        ni = int(np.argmin(d))
        xn = V[:, ni]
        dirv = xs - xn
        L = np.linalg.norm(dirv)
        if L < 1e-9:
            continue
        xnew = xn + min(P.rrt.step, L) * dirv / L
        if _seg_free(xn, xnew, obs, P.rrt.collstep):
            V = np.hstack([V, xnew[:, None]]); parent.append(ni)
            if np.linalg.norm(xnew - p_goal) < P.rrt.goalrad and _seg_free(xnew, p_goal, obs, P.rrt.collstep):
                V = np.hstack([V, p_goal[:, None]]); parent.append(V.shape[1] - 2)
                goal_idx = V.shape[1] - 1
                break
    if goal_idx < 0:
        raise RuntimeError("RRT failed")
    idx, wp = goal_idx, []
    while idx > 0:
        wp.insert(0, V[:, idx]); idx = parent[idx]
    wp.insert(0, V[:, 0])
    wp = np.array(wp).T
    wp = _shortcut(wp, obs, P)
    pts = _resample(wp, P.rrt.spacing)
    pts = _smooth_path(pts, obs, 30)
    pts = _resample(pts, P.rrt.spacing)
    d = np.sqrt(np.sum(np.diff(pts, axis=1) ** 2, axis=0))
    s = np.concatenate([[0], np.cumsum(d)])
    return SimpleNamespace(pts=pts, th=_headings(pts), s=s, len=s[-1])


def path_from_matlab(matfile):
    """Load MATLAB's exact RRT path. Parity checks use this so the two ports are compared on IDENTICAL
    input; re-planning in Python would change the path and make every downstream diff meaningless."""
    import scipy.io as sio
    d = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)
    p = d["path"]
    return SimpleNamespace(pts=np.asarray(p.pts, dtype=float), th=np.asarray(p.th, dtype=float),
                           s=np.asarray(p.s, dtype=float), len=float(p.len))

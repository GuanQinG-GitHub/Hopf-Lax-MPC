"""
mpc_solvers.py -- the four MPC solvers, ported from mpc_compare.m.

    resnewton_solve   PMP: plain Levenberg-Marquardt on the transversality residual.
    guarded_rn_fast   M1: the same LM step, but GUARDED -- it reads the curvature of the shooting
                      Hessian model and, when it finds itself at an index-1 saddle, kicks off it.
                      This is the method the whole comparison exists to demonstrate.
    ilqr_run          iLQR: second-order DDP-style baseline.
    build_coll        Direct collocation NLP through CasADi/IPOPT.

PORTING CONTRACT
    The control flow is reproduced statement-for-statement, including the parts that look redundant.
    Two examples that matter and must NOT be "improved":
      * The line search tries alphas in a fixed order and takes the FIRST that improves. Taking the
        best instead would change the iterate and break parity.
      * The budget check is `if k>1 and elapsed>=budget: break`, so every solver is guaranteed at least
        one step even when the budget is already blown. Removing that guard would let a solver return
        its warm start unmodified.

WHERE THE HOST LOOP STAYS IN PYTHON
    The budget cut is wall-clock, so control must return to the host between iterations to read the
    clock. Each iteration therefore calls a handful of jit'd kernels rather than living inside one
    lax.while_loop. That costs ~20 us of dispatch per iteration and buys the exact MATLAB semantics.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import casadi as ca
import jax
import jax.numpy as jnp
from jax import lax


# ======================================================================================================
#  PMP  --  resnewton_solve
# ======================================================================================================
def resnewton_solve(S, p, xc, Xr, P, tc):
    """Levenberg-Marquardt on res(p)=0. No curvature guard: if the root it converges to happens to be a
    saddle of J, PMP converges to it happily and sits there -- that is exactly the obs-2 trap."""
    n = p.size
    mu = 1e-6
    rr = np.asarray(S.res(p, xc, Xr))
    nr = np.linalg.norm(rr)
    for k in range(1, P.lm_maxit + 1):
        if nr < P.lm_tol:
            break
        if k > 1 and (time.perf_counter() - tc) >= P.budget:          # HARD wall-clock cut (>=1 step done)
            break
        Jr = np.asarray(S.Jr(p, xc, Xr))
        A = Jr.T @ Jr
        b = -Jr.T @ rr
        acc = False
        for _ in range(20):                                           # LM damping search
            step = np.linalg.solve(A + mu * np.eye(n), b)
            pt = p + step
            rt = np.asarray(S.res(pt, xc, Xr))
            if np.linalg.norm(rt) < nr:
                p, rr, nr = pt, rt, np.linalg.norm(rt)
                mu = max(mu / 3, 1e-12)
                acc = True
                break
            mu = mu * 5
        if not acc:
            break
    return p


# ======================================================================================================
#  M1  --  guarded_rn_fast  (the saddle-escaping method)
# ======================================================================================================
def guarded_rn_fast(S, p, xc, Xr, PP, gmode, epsc, alpha, tc, budget, trace=None):
    """Speed-fused M1.

    One fused S.all per iteration (single rollout -> res, J, Jr, Sx); the gradient is then the free
    mat-vec g = -Sx'res, and the guard reuses Sx,Jr from the same call.

    The logic in one line: take cheap LM steps while the curvature model M = -Sx'Jr says "convex", and
    when its minimum eigenvalue drops below -epsc, stop trusting the root and escape.

    THE KICK is the part that matters. At the obs-2 wall the geometry is y-symmetric, so the gradient
    becomes orthogonal to the unstable eigenvector (|g'vmin|/|g| < 1e-3): the iterate sits ON the saddle
    ridge and a gradient method has nothing to descend. Detecting that, M1 steps a fixed distance alpha
    ALONG +/-vmin, picking the side by explicitly comparing J -- which is also why the arbitrary sign of
    an eigenvector never matters here.

    `latched` is a speed hack: once converged and convex twice in a row, skip the eigendecomposition and
    only re-check every 3rd iteration, dropping back to the full guard if curvature drifts negative.
    """
    n = p.size
    rho, mu = 1e-6, 1e-6
    it = ncheap = nexp = nkick = conv = lc = 0
    latched = False
    Vc = Dc = None

    for k in range(1, PP.maxit + 1):
        if k > 1 and (time.perf_counter() - tc) >= budget:            # HARD wall-clock cut (>=1 step done)
            break
        res, J, Jr, Sx = S.all(p, xc, Xr)                             # ONE fused rollout
        res, J = np.asarray(res), float(J)
        Jr, Sx = np.asarray(Jr), np.asarray(Sx)
        nr = np.linalg.norm(res)
        g = -Sx.T @ res                                               # R1 identity: free gradient
        ng = np.linalg.norm(g)
        if trace is not None:                                         # DIAGNOSTIC only; no effect on numerics
            trace.append(dict(nr=nr, ng=ng))                         # residual sequence -> shows Newton convergence

        if latched:
            lc += 1
            drift = False
            if lc % 3 == 1:                                           # periodic re-check, not every iter
                Hc = -(Sx.T @ Jr)
                if np.linalg.eigvalsh(0.5 * (Hc + Hc.T)).min() < -epsc:
                    drift = True
            if drift:
                latched, conv = False, 0
            elif ng < PP.tol * (1 + abs(J)):
                break
            else:
                A = Jr.T @ Jr
                b = -Jr.T @ res
                ac = False
                for _ in range(20):
                    step = np.linalg.solve(A + mu * np.eye(n), b)
                    pt = p + step
                    if np.linalg.norm(np.asarray(S.res(pt, xc, Xr))) < nr:
                        p, mu, ac = pt, max(mu / 3, 1e-12), True
                        break
                    mu = mu * 5
                ncheap += 1
                if not ac:
                    latched = False
                it += 1
                continue

        Hc = -(Sx.T @ Jr)
        Hc = 0.5 * (Hc + Hc.T)                                        # Gauss-Newton curvature model
        Dv, Vc = np.linalg.eigh(Hc)                                   # ascending, like MATLAB's eig(sym)
        Dc = Dv
        im = int(np.argmin(Dv))
        lmin, vmin = Dv[im], Vc[:, im]

        if ng < PP.tol * (1 + abs(J)) and lmin >= -epsc:              # converged AND convex -> done
            break

        if lmin >= -epsc:                                             # convex read: cheap LM step
            conv += 1
            if conv >= 2:
                latched = True
            A = Jr.T @ Jr
            b = -Jr.T @ res
            ac = False
            for _ in range(20):
                step = np.linalg.solve(A + mu * np.eye(n), b)
                pt = p + step
                if np.linalg.norm(np.asarray(S.res(pt, xc, Xr))) < nr:
                    p, mu, ac = pt, max(mu / 3, 1e-12), True
                    break
                mu = mu * 5
            ncheap += 1
            if not ac:
                break
        else:                                                         # SADDLE: certified index-1
            conv = 0
            if abs(g @ vmin) / max(ng, np.finfo(float).eps) < 1e-3:   # gradient blind to the escape dir
                s = 1.0
                if float(S.J(p - alpha * vmin, xc, Xr)) < float(S.J(p + alpha * vmin, xc, Xr)):
                    s = -1.0                                          # pick the downhill side explicitly
                p = p + alpha * s * vmin
                nkick += 1
                res, J, Jr, Sx = S.all(p, xc, Xr)                     # refresh at the kicked iterate
                res, J = np.asarray(res), float(J)
                Jr, Sx = np.asarray(Jr), np.asarray(Sx)
                g = -Sx.T @ res
                Hc = -(Sx.T @ Jr)
                Hc = 0.5 * (Hc + Hc.T)
                Dv, Vc = np.linalg.eigh(Hc)
                Dc = Dv
            ev = np.abs(Dc)                                           # eigenvalue-floored Newton step
            flo = max(rho * ev.max(), 1e-14)
            dd = np.maximum(ev, flo)
            step = -Vc @ ((Vc.T @ g) / dd)
            ac = False
            for a in (1, 0.5, 0.25, 0.1, 0.03, 0.01):                 # FIRST improving alpha wins
                pt = p + a * step
                Jt = float(S.J(pt, xc, Xr))
                if np.isfinite(Jt) and Jt < J - 1e-9 * abs(J):
                    p, ac = pt, True
                    break
            nexp += 1
            if not ac:
                break
        it += 1

    info = SimpleNamespace(ncheap=ncheap, nexp=nexp, nkick=nkick, latched=latched)
    return p, it, info


# ======================================================================================================
#  iLQR
# ======================================================================================================
def build_ilqr(M, N, dt, JX):
    """jit'd iLQR kernels. The moving-obstacle centres `mc` are a runtime ARGUMENT, never a closure
    constant -- baking them in would retrigger a JAX recompile on every cycle (they move every cycle)."""
    n, m = M.n, M.m
    Q, QT, R = JX.Q, JX.QT, JX.R
    umax = JX.umax
    pos_i = JX.pos_i

    def obs_all(pos, mc):
        """augobs() in functional form: static barriers + the moving ones at their current position."""
        p1, g1 = JX.obstacle(pos)
        p2, g2 = JX.mobs_pen(pos, mc)
        return p1 + p2, g1 + g2

    def _rollout(x0, U):
        def body(x, u):
            x2 = JX.rk4_state(x, u, dt)
            return x2, x2
        _, Xs = lax.scan(body, x0, U.T)
        return jnp.vstack([x0[None, :], Xs]).T                        # (n, N+1)

    def _traj_cost(X, U, Xref, mc):
        def body(c, k):
            dxk = X[:, k] - Xref[:, k]
            pen, _ = obs_all(X[pos_i, k], mc)
            return c + (dxk @ Q @ dxk + U[:, k] @ R @ U[:, k] + pen) * dt, None
        Jc, _ = lax.scan(body, 0.0, jnp.arange(N))
        dxT = X[:, N] - Xref[:, N]
        return Jc + dxT @ QT @ dxT

    @jax.jit
    def rollout_cost(x0, U, Xref, mc):
        X = _rollout(x0, U)
        return X, _traj_cost(X, U, Xref, mc)

    @jax.jit
    def backward(X, U, Xref, mc, mu):
        """Riccati sweep. Returns (kff, Kk, bad); `bad` mirrors MATLAB's chol failure flag -- JAX's
        cholesky yields NaN on a non-PD Quu, which we detect and hand back so the host can raise mu."""
        Fx = jax.vmap(lambda x, u: jax.jacfwd(lambda xx: JX.rk4_state(xx, u, dt))(x))(X[:, :N].T, U.T)
        Fu = jax.vmap(lambda x, u: jax.jacfwd(lambda uu: JX.rk4_state(x, uu, dt))(u))(X[:, :N].T, U.T)

        def body(carry, k):
            Vx, Vxx, bad = carry
            fx, fu = Fx[k], Fu[k]
            x, u = X[:, k], U[:, k]
            _, og = obs_all(x[pos_i], mc)
            gx = jnp.concatenate([og, jnp.zeros(n - 3)])
            lx = (2 * Q @ (x - Xref[:, k]) + gx) * dt
            lu = 2 * R @ u * dt
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = 2 * Q * dt + fx.T @ Vxx @ fx
            Quu = 2 * R * dt + fu.T @ Vxx @ fu + mu * jnp.eye(m)
            Qux = fu.T @ Vxx @ fx
            L = jnp.linalg.cholesky(Quu)
            bad = bad | jnp.any(jnp.isnan(L))
            kf = -jax.scipy.linalg.cho_solve((L, True), Qu)
            Kk = -jax.scipy.linalg.cho_solve((L, True), Qux)
            Vx2 = Qx + Kk.T @ Quu @ kf + Kk.T @ Qu + Qux.T @ kf
            Vxx2 = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
            Vxx2 = 0.5 * (Vxx2 + Vxx2.T)
            return (Vx2, Vxx2, bad), (kf, Kk)

        Vx0 = 2 * QT @ (X[:, N] - Xref[:, N])
        (_, _, bad), (kff, Kk) = lax.scan(body, (Vx0, 2 * QT, False), jnp.arange(N - 1, -1, -1))
        return kff[::-1].T, Kk[::-1], bad                             # scan ran backwards -> restore order

    @jax.jit
    def forward(x0, X, U, kff, Kk, Xref, mc, alphas):
        """All four line-search alphas evaluated at once (vmap), so one dispatch replaces four. The host
        still takes the FIRST improving alpha, matching MATLAB's break-on-first semantics exactly."""
        def one(a):
            def body(xn, k):
                uk = U[:, k] + a * kff[:, k] + Kk[k] @ (xn - X[:, k])
                un = jnp.clip(uk, -umax, umax)
                return JX.rk4_state(xn, un, dt), (xn, un)
            xN, (Xs, Us) = lax.scan(body, x0, jnp.arange(N))
            Xn = jnp.vstack([Xs, xN[None, :]]).T
            Un = Us.T
            return Xn, Un, _traj_cost(Xn, Un, Xref, mc)
        return jax.vmap(one)(alphas)

    @jax.jit
    def costate0(X, U, Xref, mc):
        """Costate at stage 0 by sweeping the adjoint backwards -- this is what hands iLQR's solution to
        the SAME u = ustar(x,p) applicator the shooting methods use, keeping the comparison fair."""
        Fx = jax.vmap(lambda x, u: jax.jacfwd(lambda xx: JX.rk4_state(xx, u, dt))(x))(X[:, :N].T, U.T)

        def body(p, k):
            _, og = obs_all(X[pos_i, k], mc)
            gx = jnp.concatenate([og, jnp.zeros(n - 3)])
            return (2 * Q @ (X[:, k] - Xref[:, k]) + gx) * dt + Fx[k].T @ p, None

        pN = 2 * QT @ (X[:, N] - Xref[:, N])
        p0, _ = lax.scan(body, pN, jnp.arange(N - 1, -1, -1))
        return p0

    def warm(x0, Xref, mc):
        """Compile every iLQR kernel before the timed loop -- see the note on build_ss.warm()."""
        U = np.zeros((m, N))
        X, _ = rollout_cost(x0, U, Xref, mc)
        kff, Kk, _ = backward(X, U, Xref, mc, 1e-3)
        forward(x0, X, U, kff, Kk, Xref, mc, jnp.array([1.0, 0.5, 0.2, 0.05]))
        costate0(X, U, Xref, mc)

    return SimpleNamespace(rollout_cost=rollout_cost, backward=backward, forward=forward,
                           costate0=costate0, warm=warm, n=n, m=m, N=N, dt=dt)


def ilqr_run(IL, x, Xref, mc, Uinit, P, tc, trace=None, mu0=1e-3):
    """Host loop: mirrors ilqr_run() including the mu schedule and the dX convergence test.

    `trace` (a list, or None) and `mu0` are DIAGNOSTIC-ONLY: with trace=None and mu0=1e-3 the numerics
    are byte-identical to the verified version, so parity is unaffected. trace records per-iteration
    (mu, bad, alpha, J, dX); mu0 lets decompose.py test whether the free-space iteration count is a
    regularisation artifact by starting the damping lower.
    """
    N, m = IL.N, IL.m
    U = np.zeros((m, N)) if Uinit is None else Uinit
    X, J = IL.rollout_cost(x, U, Xref, mc)
    X, J = np.asarray(X), float(J)
    mu, dX = mu0, np.inf
    alphas = jnp.array([1.0, 0.5, 0.2, 0.05])
    it = 0
    for it in range(1, P.ilqr_Kmax + 1):
        if it > 1 and (time.perf_counter() - tc) >= P.budget:
            break
        kff, Kk, bad = IL.backward(X, U, Xref, mc, mu)
        if bool(bad):                                                 # Quu not PD -> raise damping, retry
            if trace is not None:
                trace.append(dict(mu=mu, bad=True, alpha=np.nan, J=J, dX=np.nan))
            mu = mu * 4
            continue
        Xn, Un, Jn = IL.forward(x, X, U, kff, Kk, Xref, mc, alphas)
        Jn = np.asarray(Jn)
        improved = False
        a_acc = np.nan
        for i in range(len(alphas)):                                  # FIRST improving alpha, as in MATLAB
            if Jn[i] < J:
                Xa, Ua = np.asarray(Xn[i]), np.asarray(Un[i])
                dX = np.linalg.norm(Xa - X) / (1 + np.linalg.norm(X))
                X, U, J = Xa, Ua, float(Jn[i])
                improved = True
                a_acc = float(alphas[i])
                break
        if trace is not None:
            trace.append(dict(mu=mu, bad=False, alpha=a_acc, J=J, dX=(dX if improved else np.nan)))
        if improved:
            mu = max(mu * 0.7, 1e-9)
            if dX < P.ilqr_tol:
                break
        else:
            mu = mu * 4
    p0 = np.asarray(IL.costate0(X, U, Xref, mc))
    return p0, U, it


def shift_U(U, ns):
    return np.hstack([U[:, ns:], np.repeat(U[:, -1:], ns, axis=1)])


# ======================================================================================================
#  DDP  --  full second-order Differential Dynamic Programming.
#
#  The linearization-based method taken to second order: iLQR (build_ilqr) drops the dynamics Hessian
#  (Vx.Fxx) and the barrier cost Hessian, keeping only the Gauss-Newton terms -> LINEAR convergence. DDP
#  keeps them -> QUADRATIC convergence, at the price of forming/contracting the Hessian tensors each
#  iteration. This is the fair, strongest linearization baseline against the (linearization-free) M1.
#
#  Backward Q-blocks (DDP adds the terms iLQR omits, marked <<):
#    Qxx = 2Q.dt + barrier_Hxx.dt + fx'Vxx fx + Vx.Fxx           << barrier_Hxx, Vx.Fxx
#    Quu = 2R.dt              + fu'(Vxx+muI) fu + Vx.Fuu          << Vx.Fuu  (mu on Vxx: Tassa reg)
#    Qux =                     + fu'(Vxx+muI) fx + Vx.Fux         << Vx.Fux
#
#  EFFICIENCY: Fxx/Fuu/Fux and barrier_Hxx do NOT depend on Vx, so they are precomputed for ALL stages in
#  parallel (vmap) BEFORE the sequential backward sweep; the scan then does only cheap tensor-vector
#  contractions einsum('i,ijk->jk', Vx, F..) + Riccati algebra. Expensive AD is kept off the sequential
#  critical path -- the standard efficient-DDP structure.
# ======================================================================================================
# Finer geometric line search than iLQR's: the tanh barrier is stiff, so the full Newton step often
# overshoots and only a small alpha reduces J. Shared by warm() and ddp_run() so forward() compiles once.
DDP_ALPHAS = jnp.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01])


def build_ddp(M, N, dt, JX):
    n, m = M.n, M.m
    Q, QT, R = JX.Q, JX.QT, JX.R
    umax = JX.umax
    pos_i = JX.pos_i

    def obs_all(pos, mc):
        p1, g1 = JX.obstacle(pos)
        p2, g2 = JX.mobs_pen(pos, mc)
        return p1 + p2, g1 + g2

    def _F(x, u):
        return JX.rk4_state(x, u, dt)

    def _Fz(z, xr_unused=None):                                       # F over the stacked z=(x,u)
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
            pen, _ = obs_all(X[pos_i, k], mc)
            return c + (dxk @ Q @ dxk + U[:, k] @ R @ U[:, k] + pen) * dt, None
        Jc, _ = lax.scan(body, 0.0, jnp.arange(N))
        dxT = X[:, N] - Xref[:, N]
        return Jc + dxT @ QT @ dxT

    @jax.jit
    def rollout_cost(x0, U, Xref, mc):
        X = _rollout(x0, U)
        return X, _traj_cost(X, U, Xref, mc)

    @jax.jit
    def backward(X, U, Xref, mc, mu):
        """Full second-order sweep. Returns (kff, Kk, dV1, dV2, bad)."""
        Z = jnp.concatenate([X[:, :N].T, U.T], axis=1)                # (N, n+m) stacked states+controls
        Fx = jax.vmap(lambda x, u: jax.jacfwd(lambda xx: _F(xx, u))(x))(X[:, :N].T, U.T)
        Fu = jax.vmap(lambda x, u: jax.jacfwd(lambda uu: _F(x, uu))(u))(X[:, :N].T, U.T)
        # dynamics Hessian tensors, Vx-independent -> computed for all stages in parallel
        H = jax.vmap(lambda z: jax.jacfwd(jax.jacfwd(_Fz))(z))(Z)     # (N, n_out, n+m, n+m)
        Fxx = H[:, :, :n, :n]                                          # (N, n, n, n)
        Fuu = H[:, :, n:, n:]                                          # (N, n, m, m)
        Fux = H[:, :, n:, :n]                                          # (N, n, m, n)
        # barrier cost Hessian d2(pen)/dpos2 (3x3), embedded into the n-block; Vx-independent.
        # PSD-PROJECT it (clip negative eigenvalues to 0): the tanh barrier is nonconvex -- its saturating
        # tail has an indefinite Hessian that destabilizes full DDP near obstacles. Projection keeps the
        # convex avoidance bowl and drops the concave part, the standard treatment of nonconvex costs in
        # second-order trajectory optimization. It is still genuine 2nd-order barrier curvature.
        def _psd(H):
            w, V = jnp.linalg.eigh(0.5 * (H + H.T))
            return (V * jnp.maximum(w, 0.0)) @ V.T
        Bpos = jax.vmap(lambda x: _psd(jax.hessian(lambda p: obs_all(p, mc)[0])(x[pos_i])))(X[:, :N].T)

        def body(carry, k):
            Vx, Vxx, dV1, dV2, bad = carry
            fx, fu = Fx[k], Fu[k]
            x, u = X[:, k], U[:, k]
            _, og = obs_all(x[pos_i], mc)
            gx = jnp.concatenate([og, jnp.zeros(n - 3)])
            lxx = 2 * Q * dt
            lxx = lxx.at[jnp.ix_(pos_i, pos_i)].add(Bpos[k] * dt)     # << barrier cost Hessian
            lx = (2 * Q @ (x - Xref[:, k]) + gx) * dt
            lu = 2 * R @ u * dt
            Vxx_r = Vxx + mu * jnp.eye(n)                             # << Tassa state regularization
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx + jnp.einsum('i,ijk->jk', Vx, Fxx[k])       # << Vx.Fxx
            Quu = 2 * R * dt + fu.T @ Vxx_r @ fu + jnp.einsum('i,ijk->jk', Vx, Fuu[k])   # << Vx.Fuu
            Qux = fu.T @ Vxx_r @ fx + jnp.einsum('i,ijk->jk', Vx, Fux[k])           # << Vx.Fux
            L = jnp.linalg.cholesky(Quu)
            bad = bad | jnp.any(jnp.isnan(L))
            kf = -jax.scipy.linalg.cho_solve((L, True), Qu)
            Kk = -jax.scipy.linalg.cho_solve((L, True), Qux)
            dV1 = dV1 + kf @ Qu                                       # expected reduction, 1st order
            dV2 = dV2 + 0.5 * kf @ Quu @ kf                           # expected reduction, 2nd order
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
        """Adjoint sweep -> p0, handing DDP's solution to the shared u=ustar(x,p) applicator (as iLQR)."""
        Fx = jax.vmap(lambda x, u: jax.jacfwd(lambda xx: _F(xx, u))(x))(X[:, :N].T, U.T)

        def body(p, k):
            _, og = obs_all(X[pos_i, k], mc)
            gx = jnp.concatenate([og, jnp.zeros(n - 3)])
            return (2 * Q @ (X[:, k] - Xref[:, k]) + gx) * dt + Fx[k].T @ p, None

        pN = 2 * QT @ (X[:, N] - Xref[:, N])
        p0, _ = lax.scan(body, pN, jnp.arange(N - 1, -1, -1))
        return p0

    def warm(x0, Xref, mc):
        U = np.zeros((m, N))
        X, _ = rollout_cost(x0, U, Xref, mc)
        kff, Kk, _, _, _ = backward(X, U, Xref, mc, 1e-3)
        forward(x0, X, U, kff, Kk, Xref, mc, DDP_ALPHAS)             # MUST match ddp_run's alpha count,
        costate0(X, U, Xref, mc)                                     # else forward recompiles mid-run (~250ms)

    return SimpleNamespace(rollout_cost=rollout_cost, backward=backward, forward=forward,
                           costate0=costate0, warm=warm, n=n, m=m, N=N, dt=dt)


def ddp_run(IL, x, Xref, mc, Uinit, P, tc, trace=None, mu0=1e-3):
    """Host loop for DDP; mirrors ilqr_run (mu schedule, dX-tol, wall-clock cut). `trace` records per
    iteration (mu, alpha, J, dX, and the expected/actual reduction ratio) for verify_ddp.py."""
    N, m = IL.N, IL.m
    U = np.zeros((m, N)) if Uinit is None else Uinit
    X, J = IL.rollout_cost(x, U, Xref, mc)
    X, J = np.asarray(X), float(J)
    mu, dX = mu0, np.inf
    alphas = DDP_ALPHAS                                              # shared with warm() so forward() is compiled
    it = 0
    for it in range(1, P.ilqr_Kmax + 1):
        if it > 1 and (time.perf_counter() - tc) >= P.budget:
            break
        kff, Kk, dV1, Dv2, bad = IL.backward(X, U, Xref, mc, mu)
        if bool(bad):                                                 # Quu not PD -> raise damping, retry
            if trace is not None:
                trace.append(dict(mu=mu, bad=True, alpha=np.nan, J=J, dX=np.nan, ratio=np.nan))
            mu = mu * 4
            continue
        dV1, Dv2 = float(dV1), float(Dv2)
        if abs(dV1) < P.ilqr_tol:                                     # expected reduction ~ 0 -> converged
            break
        Xn, Un, Jn = IL.forward(x, X, U, kff, Kk, Xref, mc, alphas)
        Jn = np.asarray(Jn)
        improved = False
        a_acc, ratio = np.nan, np.nan
        for i in range(len(alphas)):                                  # FIRST improving alpha (as iLQR)
            if Jn[i] < J:
                a = float(alphas[i])
                expected = a * dV1 + a * a * Dv2                      # DDP expected reduction at alpha
                ratio = (J - float(Jn[i])) / expected if abs(expected) > 1e-14 else np.nan
                Xa, Ua = np.asarray(Xn[i]), np.asarray(Un[i])
                dX = np.linalg.norm(Xa - X) / (1 + np.linalg.norm(X))
                X, U, J = Xa, Ua, float(Jn[i])
                improved, a_acc = True, a
                break
        if trace is not None:
            trace.append(dict(mu=mu, bad=False, alpha=a_acc, J=J, dX=(dX if improved else np.nan),
                              ratio=ratio))
        if improved:
            # alpha-based regularization (standard DDP heuristic): a SMALL accepted alpha means the
            # (undamped) Newton step overshot -- the stiff barrier curvature was under-damped -- so raise
            # mu to better-scale the next step; a full step (alpha~1) means the model is good, so relax mu.
            # Without this the damping shrinks while crawling and DDP stalls at stiff-barrier states.
            mu = mu * 2.0 if a_acc < 0.5 else max(mu * 0.7, 1e-9)
            if dX < P.ilqr_tol:
                break
        else:
            mu = mu * 4
    p0 = np.asarray(IL.costate0(X, U, Xref, mc))
    return p0, U, it


# ======================================================================================================
#  COLLOCATION  (CasADi + IPOPT)  --  direct port of build_coll
# ======================================================================================================
def build_coll(M, N, dt, budget):
    """Collocation NLP. Xr carries n+3K rows (tracking ref + K moving-obstacle centres per stage), so the
    solver call signature is identical to the shooting methods'."""
    n, m = M.n, M.m
    nm = M.mobs.c0.shape[1]
    Q, QT, R = ca.DM(M.Q), ca.DM(M.QT), ca.DM(M.R)

    def dyn(x, u):
        th = x[3]
        return ca.vertcat(x[4] * ca.cos(th), x[4] * ca.sin(th), x[6], x[5], u[0], u[1], u[2])

    def rk4(x, u):
        k1 = dyn(x, u); k2 = dyn(x + dt / 2 * k1, u)
        k3 = dyn(x + dt / 2 * k2, u); k4 = dyn(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def barrier(pos, C, ax, ay, az, W, w, eps):
        pen = 0
        for i in range(C.shape[1]):
            dd = pos - C[:, i]
            rho = ca.sqrt((dd[0] / ax[i]) ** 2 + (dd[1] / ay[i]) ** 2 + (dd[2] / az[i]) ** 2 + eps)
            pen = pen + W[i] * 0.5 * (1 - ca.tanh((rho - 1) / w[i]))
        return pen

    xk, uk = ca.SX.sym("x", n), ca.SX.sym("u", m)
    FP = ca.Function("FP", [xk, uk], [rk4(xk, uk)])
    par = ca.MX.sym("par", n)
    Xr = ca.MX.sym("Xr", n + 3 * nm, N + 1)
    X = ca.MX.sym("X", n, N + 1)
    U = ca.MX.sym("U", m, N)
    Xn = FP.map(N)(X[:, :N], U)
    J = 0
    o = M.obs
    for k in range(N):
        dxk = X[:, k] - Xr[:n, k]
        mck = ca.reshape(Xr[n:, k], 3, nm)
        pen = (barrier(X[:3, k], ca.DM(o.center), o.ax, o.ay, o.az, o.W, o.w, o.eps)
               + barrier(X[:3, k], mck, M.mobs.ax, M.mobs.ay, M.mobs.az, M.mobs.W, M.mobs.w, M.mobs.eps))
        J = J + (dxk.T @ Q @ dxk + U[:, k].T @ R @ U[:, k] + pen) * dt
    dxT = X[:, N] - Xr[:n, N]
    J = J + dxT.T @ QT @ dxT
    D = X[:, 1:] - Xn
    G = ca.vertcat(X[:, 0] - par, ca.reshape(D, -1, 1))
    opts = {"ipopt.max_iter": 100, "ipopt.tol": 1e-8, "ipopt.print_level": 0,
            "print_time": 0, "error_on_fail": False}
    if np.isfinite(budget):                                           # IPOPT rejects a non-finite limit
        opts["ipopt.max_wall_time"] = float(budget)
    s = ca.nlpsol("s", "ipopt", {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
                                 "f": J, "g": G, "p": ca.vertcat(par, ca.reshape(Xr, -1, 1))}, opts)
    lbx = np.concatenate([-np.inf * np.ones(n * (N + 1)), np.tile(-M.umax, N)])
    ubx = np.concatenate([np.inf * np.ones(n * (N + 1)), np.tile(M.umax, N)])
    nG = n * (N + 1)

    def call(xc, Xrv, W0):
        if W0 is None:
            X0 = np.tile(xc.reshape(-1, 1), (1, N + 1))
            W0 = np.concatenate([X0.flatten(order="F"), np.zeros(m * N)])
        r = s(x0=W0, p=np.concatenate([xc, Xrv.flatten(order="F")]),
              lbg=np.zeros(nG), ubg=np.zeros(nG), lbx=lbx, ubx=ubx)
        w = np.asarray(r["x"]).ravel()
        Uo = w[n * (N + 1):].reshape((m, N), order="F")
        return float(r["f"]), Uo[:, 0], w

    return call


def shift_coll(w, n, m, N, ns):
    X = w[:n * (N + 1)].reshape((n, N + 1), order="F")
    U = w[n * (N + 1):].reshape((m, N), order="F")
    Xs = np.hstack([X[:, ns:], np.repeat(X[:, -1:], ns, axis=1)])
    Us = np.hstack([U[:, ns:], np.repeat(U[:, -1:], ns, axis=1)])
    return np.concatenate([Xs.flatten(order="F"), Us.flatten(order="F")])

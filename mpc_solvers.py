"""
mpc_solvers.py -- the MPC solvers compared by mpc_testbed.py

    resnewton_solve   PMP: plain Levenberg-Marquardt on the transversality residual.
    chlqn_solve       M1_v2: paper Algorithm 1, Certified Hopf-Lax Quasi-Newton.  The same cheap
                      shooting rollout, but it reads the curvature of the free Hessian model and,
                      when it certifies an index-1 saddle, kicks off it.  This is the method the
                      whole comparison exists to demonstrate.
    ilqr_run          iLQR: first-order-dynamics (Gauss-Newton) baseline.
    ddp_run           DDP: full second-order baseline.
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
    """PMP single shooting: Levenberg-Marquardt on the transversality residual res(p) = 0.
    
    S: Shooting problem

    DECISION VARIABLE.  `p` is the INITIAL COSTATE p0 in R^n (n = 7). The control is recovered pointwise from the Hamiltonian minimiser u = ustar(x, p) inside the canonical
    RK4 step, so one forward rollout of z = [x; p] from (xc, p) pins down the entire trajectory.

    WHAT IS BEING SOLVED.

        res(p) = p_N - 2 Q_T (x_N - x_N^ref) = 0,                                          (transversality)

    the free-endpoint boundary condition of the PMP two-point BVP.

    WHY THIS IS THE TRAP.  Root-finding cannot distinguish a minimum from a saddle from a maximum --
    all three satisfy res = 0.

    HOW THIS IS SOLVED.  Levenberg-Marquardt on ||res(p)||, linearising about the current p:

        res(p + d) ~= res(p) + Jr d,            Jr = d res / d p   (7x7, jacfwd)

        normal equation:   (Jr^T Jr)          d = -Jr^T res(p)
        LM damped:         (Jr^T Jr + mu I)   d = -Jr^T res(p)
    """
    n = p.size                                                        # n = 7, the costate dimension
    mu = 1e-6                                                         # LM damping; small = trust the GN step
    rr = np.asarray(S.res(p, xc, Xr))                                 # residual at the warm start (1 rollout)
    nr = np.linalg.norm(rr)                                           # merit value: the ONLY thing minimised
    for k in range(1, P.lm_maxit + 1):                                # P.lm_maxit = 50 (mpc_core.make_P)
        if nr < P.lm_tol:                                             # P.lm_tol = 1e-4: converged on the root
            break
        if k > 1 and (time.perf_counter() - tc) >= P.budget:          # HARD wall-clock cut (>=1 step done)
            break                                                     # k>1 guard: never return the warm start
        Jr = np.asarray(S.Jr(p, xc, Xr))                              # Jr = d res / d p, (7,7), by jacfwd
        A = Jr.T @ Jr                                                 # Gauss-Newton normal matrix, PSD by
        b = -Jr.T @ rr                                                # construction; b = -grad of ||res||^2/2
        acc = False                                                   # did any damping level get accepted?
        for _ in range(20):                                           # LM damping search
            GN_step = np.linalg.solve(A + mu * np.eye(n), b)          # (A + mu I) GN_d = b
            pt = p + GN_step                                          # GN step (mu->0) and gradient (mu->inf)
            rt = np.asarray(S.res(pt, xc, Xr))                        # trial rollout: 1 extra shoot per try
            if np.linalg.norm(rt) < nr:                               # ACCEPT on residual decrease alone.
                p, rr, nr = pt, rt, np.linalg.norm(rt)                # Note: J is never consulted, so a step
                mu = max(mu / 3, 1e-12)                               # toward a saddle is as welcome as one
                acc = True                                            # toward a minimum -- the trap, in one
                break                                                 # line of code.
            mu = mu * 5                                               # reject -> damp harder, shorten the step
        if not acc:                                                   # 20 levels (mu x 5^20 ~ 1e14) all failed
            break                                                     # -> stationary in the residual; give up
    return p                                                          # converged/cut costate; caller maps it
                                                                      # to u via JX.ustar(x, p)


# ======================================================================================================
#  M1  --  chlqn_solve  (paper Algorithm 1: Certified Hopf-Lax Quasi-Newton; the saddle-escaping method)
#
#  THE TWO OBJECTS THIS METHOD IS BUILT ON.  One fused S.all per iteration yields (res, J, Jr, Sx) from
#  a SINGLE rollout, where Jr = d res / d v and Sx = d x_N / d v.  From them:
#
#    (R1)  the gradient of the COST is free, no extra rollout and AD:
#              grad_v Phi = Sx' (2 Q_T (x_N - x_N^ref) - p_N) = -Sx' res
#          
#    (R2)  the curvature model is likewise free, as a single 7x7 mat-mat:
#              M := -Sx' Jr  ~=  Hess_v Phi,
#
#  THE KICK is the part that matters.  At the obs-2 wall the geometry is y-symmetric, so the gradient is
#  orthogonal to the unstable eigenvector (|g'vmin| / |g| below M1V2_RIDGE): the iterate sits ON the
#  saddle ridge, where every gradient-based method -- PMP, DDP, collocation, MPPI -- has nothing to
#  descend.  cHLQN detects that configuration and steps a fixed distance alpha ALONG +/-vmin, choosing
#  the side by explicitly comparing Phi on both.  That explicit comparison is also why the arbitrary sign
#  an eigensolver assigns to vmin never matters.
#
#  CONTRAST WITH DDP.  Both methods meet the same indefinite curvature.  DDP treats it as a numerical
#  obstacle and damps it away (Tassa mu on Vxx, PSD projection of the barrier Hessian).  cHLQN treats the negative eigenvalue as the INFORMATION it needs: vmin names
#  the direction in which the two branches separate, and the |M|-floor in the escape step turns that
#  direction into a descent direction instead of deleting it.

# ======================================================================================================
# Solver parameters (eta = P.lm_tol, eps = P.M1_epsc, alpha = P.M1_alpha come from mpc_core.make_P).

M1V2_ETA_S = 0.29      # detection gate (relative measure r = |grad|/(1+|Phi|)).  Measured: sign
                       # agreement is 100% for r <= 0.1 (1001 pts), first disagreement at r = 0.586;
                       # 0.29 = half that (safety margin).  Beyond it the ONLY observed failure mode
                       # is missed-saddle (M convex, H saddle) -- zero false alarms at any r, so a
                       # detection inside this envelope can never trigger a wrong escape.
M1V2_RIDGE = 0.1       # kick on-ridge threshold |grad.vmin|/|grad| (was 1e-3, the paper draft's
                       # symmetric-manifold value).  Measured at detection-eligible points
                       # (r <= eta_s, lmin(M) < -eps): first-detection ridges span 8e-4..9.2e-2
                       # (descent pollutes exact symmetry), while escape-mode iterates cluster at
                       # ~1.0 (gradient already along vmin -- must NOT kick).  0.1 separates them.
M1V2_DELTA = 1e-8      # absolute eigenvalue floor of the escape step (paper input `delta`;
                       # the old guarded-LM M1 floored relative to the spectrum instead).

# BATCHED-ROLLOUT improvement (labelled engineering change, adopted 2026-07-28): the kick's +/-alpha
# side comparison and the escape-step Phi line search are independent canonical rollouts, so they are
# evaluated in ONE vmapped dispatch each (batch-2 and batch-7) -- the same pattern iLQR/DDP use for
# their alpha batch. 
M1V2_ESC_ALPHAS = 2.0 ** -np.arange(7)                                # 1, 1/2, ..., 1/64


def ensure_jbatch(S):
    """Attach (once) a vmapped batch evaluator of the Hopf-Lax objective to a shooting object."""
    if not hasattr(S, "Jbatch"):
        S.Jbatch = jax.jit(jax.vmap(S.J, in_axes=(0, None, None)))
    return S.Jbatch


def chlqn_solve(S, v, xc, Xr, PP, epsc, eta_s, alpha, delta, tc, budget, trace=None):
    """cHLQN: certifed Hopf–Lax Quasi-Newton.

    ARGUMENTS
        S       shooting object from mpc_core.build_ss (res / J / Jr / Sx / all, all jit'd)
        v       decision variable: initial costate p0 in R^7
        xc      current state;
        Xr      reference window, (n + 3K, 2N+1), sub-stage columns
        PP      SimpleNamespace(tol=P.lm_tol=1e-4, maxit=P.lm_maxit=50)
        epsc    negative-curvature threshold (P.M1_epsc = 1e-3): lmin < -epsc certifies an index-1 saddle
        eta_s   detection gate on the relative gradient (M1V2_ETA_S)
        alpha   kick length along the unstable eigenvector (P.M1_alpha = 0.30)
        delta   absolute eigenvalue floor of the escape step (M1V2_DELTA)
        tc      cycle start time;
        budget  wall-clock budget in seconds (20 ms in the testbed)
        trace   optional list; diagnostics only, never affects the numerics

    ALGORITHM FLOW (one iteration)
    1)   ONE fused rollout -> res, Phi, Jr, Sx.  From them grad = -Sx'res (R1) and the curvature model
         M = -Sx'Jr (R2), both FREE: no extra rollout, no AD, just 7x7 algebra.

    2)   GATED CURVATURE READ.  Eigendecompose M when already escaping, or when the relative gradient
         |grad|/(1+|Phi|) <= eta_s -- Thm. 1 makes sign(lmin) trustworthy only near a root.  Then set
         escape := (lmin < -epsc).  Hysteresis: entered only through this gate, left on any convex read.

    3)   CERTIFIED STOP.  Return if |grad|/(1+|Phi|) <= tol AND the last read was convex (Prop. 3).
         Stationarity alone is what traps PMP; the curvature half is what makes the answer a minimiser.

    4-1) NOT ESCAPING -- Newton descent on the residual.  Solve Jr d = -res (eq. (8), the factored form
         of -M^-1 grad), backtrack a in {1, ..., 1/128} on ||res||.  If every a fails, take an UNGATED
         steering read: a saddle flips to escape, a convex stall returns v.

    4-2) ESCAPING -- minimise Phi, not ||res||, since leaving a root must raise ||res||.  On the ridge
         (|cos(grad, vmin)| < RIDGE) first KICK a fixed alpha along +/-vmin, side picked by comparing
         Phi on both, then re-shoot.  Then the modified-Newton step -|M|^-1 grad -- abs() turns the
         unstable mode into a descent one -- with a batched 7-alpha line search on Phi.

    """
    n = v.size                                                        # COSTATE dimension
    escape = False                                                    # MODE FLAG
                                                                      # res = 0.  True: a saddle has been
                                                                      # certified, minimise Phi instead.

    it = ndesc = nesc = nkick = neig = 0                              # counters -> iterations, descent
                                                                      # steps, escape steps, kicks, eigen reads

    Dv = Vc = vmin = None                                             # last eigen read: eigenvalues,
                                                                      # eigenvectors, vmin = Vc[:, argmin Dv].
                                                                      
    for k in range(1, PP.maxit + 1):
        if k > 1 and (time.perf_counter() - tc) >= budget:            # HARD wall-clock cut (>=1 step done)
            break
        res, J, Jr, Sx = S.all(v, xc, Xr)                             # ONE fused rollout -> res, Phi, Jr = dres/dv,
        res, J, Jr, Sx = np.asarray(res), float(J), np.asarray(Jr), np.asarray(Sx)

        nr = np.linalg.norm(res)                                      # ||res||: merit for the DESCENT line search
                                                                      
        grad = -Sx.T @ res                                            # analytical grad_v Phi

        norm_grad = np.linalg.norm(grad)                              # ||grad||: drives the detection gate AND the
                                                                      # certified-minimiser stop below

        rec = None                                                    # per-iteration diagnostic record

        if trace is not None:                                         # DIAGNOSTIC only; no effect on numerics
            rec = dict(k=k, v=v.copy(), nr=nr, norm_grad=norm_grad, J=J, gate=False, lmin=None, escape_in=escape,
                       branch=None, ridge=None, kick=False, a=None, ac=None)
            trace.append(rec)

        Hc = -(Sx.T @ Jr)                                             # curvature surrogate M ~= Hess_v Phi, one
                                                                      # free 7x7 mat-mat.  THIS is what PMP lacks.

        Hc = 0.5 * (Hc + Hc.T)                                        # symmetrise: the product is symmetric only
                                                                      # up to rollout round-off

        # read curvature of M when (a) already escaping or (b) the RELATIVE gradient is small
        if escape or norm_grad <= eta_s * (1 + abs(J)):               

            Dv, Vc = np.linalg.eigh(Hc)                               # symmetric eigendecomposition

            neig += 1                                                 # count the eigen reads: the only O(n^3) work
                                                                      # M1 adds over PMP (reported in info/cost)

            im = int(np.argmin(Dv))                                   # index of lmin -- always 0 for np.linalg.eigh

            vmin = Vc[:, im]                                          # the UNSTABLE direction.  Its sign is
                                                                      # arbitrary (eigensolver's choice) -- the
                                                                      # kick compares both sides

            escape = bool(Dv[im] < -epsc)                             # HYSTERESIS in one line: enters escape only
                                                                      # through the gate above, but LEAVES it on any
                                                                      # convex read, whatever the gradient size
            if rec is not None:
                rec["gate"], rec["lmin"] = True, float(Dv[im])

        if norm_grad <= PP.tol * (1 + abs(J)) and not escape:         # CERTIFIED minimiser (Prop. 3): stationary
            break                                                     # AND the last curvature read said convex.
        
        
        if not escape:                                                
        # ---------------- DESCENT branch ----------------
            try:
                d = np.linalg.solve(Jr, -res)                         # FACTORED Newton step: since
                                                                      # M = -Sx'Jr and grad = -Sx'res,
                                                                      # -M^-1 grad = -Jr^-1 res exactly -- so Jr
                                                                      # alone suffices, and we skip forming and
                                                                      # inverting M (cheaper, better conditioned).
                                                                      
            except np.linalg.LinAlgError:
                d = None                                              # Jr exactly singular -> fall through

            if d is None or not np.all(np.isfinite(d)):
                d = np.linalg.lstsq(Jr, -res, rcond=None)[0]          # FALLBACK: minimum-norm least-squares solve
                                                                      # of Jr d ~= -res via SVD, tiny singular
                                                                      # values truncated.  Returns a usable
                                                                      # direction where `solve` gives inf/NaN.

            ac, a = False, 1.0                                        # ac = "accepted": did ANY alpha improve?
                                                                      # a = step length, starting at the full step

            for _ in range(8):                                        # backtracking: a in {1, 1/2, ..., 1/128}
                vt = v + a * d                                        # trial iterate
                if np.linalg.norm(np.asarray(S.res(vt, xc, Xr))) < nr:# accept the FIRST alpha that lowers ||res||
                    v, ac = vt, True                                  
                    break
                a *= 0.5

            ndesc += 1 # record the number of descent steps
            if rec is not None:
                rec["branch"], rec["a"], rec["ac"] = "desc", a, ac
            if not ac:                                                # STALL: 8 halvings, ||res|| never fell.
                Dv, Vc = np.linalg.eigh(Hc)                           # STEERING read -- UNGATED, because a stall is
                neig += 1                                             # itself evidence of being near-stationary
                im = int(np.argmin(Dv))
                vmin = Vc[:, im]
                escape = bool(Dv[im] < -epsc)                         # stalled ON a saddle -> escape next iteration
                if rec is not None:
                    rec["lmin"] = float(Dv[im])
                if not escape:
                    break                                             # convex stall -> nothing left to try; return v
        else:
            # ------------- ESCAPE branch: index-1 saddle -------------                                                         
            ridge = abs(grad @ vmin) / max(norm_grad, np.finfo(float).eps)
                                                                      # RIDGE = |cos(grad, vmin)| in [0, 1]: how much
                                                                      # of the gradient points ALONG the unstable
                                                                      # direction.  ~0 means grad _|_ vmin -- the
                                                                      # iterate sits on the saddle RIDGE and every
                                                                      # gradient-based method is blind to the escape
                                                                      # direction.  ~1 means already sliding down
                                                                      # vmin, so no kick is needed.  (eps guards
                                                                      # against a 0/0 at an exact stationary point.)

            if rec is not None:
                rec["branch"], rec["ridge"] = "esc", float(ridge)

            if ridge < M1V2_RIDGE:                                    # ON-RIDGE (< 0.1): gradient carries no usable
                                                                      # information, so take an explicit finite step

                Jb = ensure_jbatch(S)                                 # vmapped Phi evaluator, built once per S

                Jpm = np.asarray(Jb(np.stack([v - alpha * vmin, v + alpha * vmin]), xc, Xr))
                                                                      # Phi on BOTH sides, ONE batch-2 dispatch

                s = -1.0 if Jpm[0] < Jpm[1] else 1.0                  # downhill side, chosen by explicit comparison
                                                                      # -- this is why vmin's arbitrary sign is moot

                v = v + alpha * s * vmin                              # THE KICK: fixed length alpha (0.30) off the
                                                                      # ridge onto one branch.

                nkick += 1

                if rec is not None:
                    rec["kick"] = True

                res, J, Jr, Sx = S.all(v, xc, Xr)                     # RE-SHOOT at the kicked iterate: grad, M and
                res, J, Jr, Sx = np.asarray(res), float(J), np.asarray(Jr), np.asarray(Sx)    
                
                grad = -Sx.T @ res
                Hc = -(Sx.T @ Jr)
                Hc = 0.5 * (Hc + Hc.T)
                Dv, Vc = np.linalg.eigh(Hc)
                neig += 1

            # not Newton method here which drives to grad=0 --> go back to the index-1 saddle
            # use the Modified Newton to escape from the saddle

            dd = np.maximum(np.abs(Dv), delta)                        # MODIFIED NEWTON in |M| = Vc diag|Dv| Vc'.
                                                                      # abs() FLIPS the negative eigenvalue, turning
                                                                      # the saddle's ascent mode into a descent one
                                                                      # (DDP instead damps it away, discarding it);
                                                                      # delta floors near-null modes so the step
                                                                      # cannot blow up.

            step = -Vc @ ((Vc.T @ grad) / dd)                         # -|M|^-1 grad, done in the eigenbasis:
                                                                      # project, scale, project back.  Guaranteed
                                                                      # a descent direction for Phi.

            Jb = ensure_jbatch(S)
            Vt = v[None, :] + M1V2_ESC_ALPHAS[:, None] * step[None, :]# 7 trial iterates, alpha = 1 ... 1/64

            Jts = np.asarray(Jb(Vt, xc, Xr))                          # ALL alphas, one batched dispatch

            ok = np.isfinite(Jts) & (Jts < J - 1e-9 * abs(J))         # ESCAPE JUDGES ON THE COST Phi, not ||res||:
                                                                      # the point is to LEAVE a root, where ||res||
                                                                      # necessarily grows. The relative margin
                                                                      # rejects round-off-sized "improvements".

            ac = bool(ok.any())

            a = float(M1V2_ESC_ALPHAS[int(np.argmax(ok))]) if ac else float(M1V2_ESC_ALPHAS[-1])
                                                                      # argmax on a bool array = FIRST True index

            if ac:
                v = Vt[int(np.argmax(ok))]                            # FIRST improving alpha
            
            nesc += 1
            
            if rec is not None:
                rec["a"], rec["ac"] = a, ac
                rec["stepnorm"] = float(np.linalg.norm(step))
                rec["spec"] = Dv.copy()
            
            if not ac:
                break                                                 # no alpha lowered Phi -> stop this cycle
        it += 1                                                       # counts only iterations that completed a
                                                                      # step (not the budget/convergence breaks)

    info = SimpleNamespace(ndesc=ndesc, nesc=nesc, nkick=nkick, neig=neig, escape=escape)
    return v, it, info


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

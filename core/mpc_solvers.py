"""
mpc_solvers.py -- all MPC solvers compared by the testbeds (one module)

    hopf_lax_mpc_solve       Hopf_Lax_MPC: paper Algorithm 1, Certified Hopf-Lax Quasi-Newton.  The same
                             cheap shooting rollout as PMP_Shooting, but it reads the curvature of the
                             free Hessian model and, when it certifies an index-1 saddle, kicks off it.
                             This is the method the whole comparison exists to demonstrate.
    pmp_shooting_solve       PMP_Shooting: plain Levenberg-Marquardt on the transversality residual.
    vanilla_hopf_lax_solve   Vanilla_Hopf_Lax: damped Newton on the same Hopf-Lax objective with the
                             EXACT Hessian (jax.hessian), no curvature certificate, no escape.
    ddp_run                  DDP: full second-order baseline (ilqr_run: its Gauss-Newton variant).
    build_collocation_ipopt  Collocation_IPOPT: direct collocation NLP through CasADi/IPOPT.
    MPPI                     MPPI: GPU sampling MPC (CuPy RawKernel; the module still imports without CuPy).

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
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.linalg
import casadi as ca
import jax
import jax.numpy as jnp
from jax import lax

try:
    with warnings.catch_warnings():                                    # CuPy's 'CUDA path' notice is noise
        warnings.simplefilter("ignore", UserWarning)                   # for the CPU-only solvers here
        import cupy as cp                                              # GPU MPPI only
    HAVE_GPU = True
except Exception:                                                      # keep the module importable on CPU
    cp = None
    HAVE_GPU = False


# ======================================================================================================
#  PMP  --  pmp_shooting_solve
# ======================================================================================================
def pmp_shooting_solve(S, p, xc, Xr, P, tc):
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
#  HOPF_LAX_MPC  --  hopf_lax_mpc_solve  (paper Algorithm 1: Certified Hopf-Lax Quasi-Newton; the saddle-escaping method)
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
#  orthogonal to the unstable eigenvector (|g'vmin| / |g| below HOPF_LAX_MPC_RIDGE): the iterate sits ON the
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
# Solver parameters (eta = P.lm_tol, eps = P.hopf_lax_mpc_epsc, alpha = P.hopf_lax_mpc_alpha come from mpc_core.make_P).

HOPF_LAX_MPC_ETA_S = 0.29      # detection gate (relative measure r = |grad|/(1+|Phi|)).  Measured: sign
                       # agreement is 100% for r <= 0.1 (1001 pts), first disagreement at r = 0.586;
                       # 0.29 = half that (safety margin).  Beyond it the ONLY observed failure mode
                       # is missed-saddle (M convex, H saddle) -- zero false alarms at any r, so a
                       # detection inside this envelope can never trigger a wrong escape.
HOPF_LAX_MPC_RIDGE = 0.1       # kick on-ridge threshold |grad.vmin|/|grad| (was 1e-3, the paper draft's
                       # symmetric-manifold value).  Measured at detection-eligible points
                       # (r <= eta_s, lmin(M) < -eps): first-detection ridges span 8e-4..9.2e-2
                       # (descent pollutes exact symmetry), while escape-mode iterates cluster at
                       # ~1.0 (gradient already along vmin -- must NOT kick).  0.1 separates them.
HOPF_LAX_MPC_DELTA = 1e-8      # absolute eigenvalue floor of the escape step (paper input `delta`;
                       # the old guarded-LM Hopf_Lax_MPC floored relative to the spectrum instead).

# BATCHED-ROLLOUT improvement (labelled engineering change, adopted 2026-07-28): the kick's +/-alpha
# side comparison and the escape-step Phi line search are independent canonical rollouts, so they are
# evaluated in ONE vmapped dispatch each (batch-2 and batch-7) -- the same pattern iLQR/DDP use for
# their alpha batch. 
HOPF_LAX_MPC_ESC_ALPHAS = 2.0 ** -np.arange(7)                                # 1, 1/2, ..., 1/64


def ensure_jbatch(S):
    """Attach (once) a vmapped batch evaluator of the Hopf-Lax objective to a shooting object."""
    if not hasattr(S, "Jbatch"):
        S.Jbatch = jax.jit(jax.vmap(S.J, in_axes=(0, None, None)))
    return S.Jbatch


def hopf_lax_mpc_solve(S, v, xc, Xr, PP, epsc, eta_s, alpha, delta, tc, budget, trace=None):
    """cHLQN: certifed Hopf–Lax Quasi-Newton.

    ARGUMENTS
        S       shooting object from mpc_core.build_ss (res / J / Jr / Sx / all, all jit'd)
        v       decision variable: initial costate p0 in R^7
        xc      current state;
        Xr      reference window, (n + 3K, 2N+1), sub-stage columns
        PP      SimpleNamespace(tol=P.lm_tol=1e-4, maxit=P.lm_maxit=50)
        epsc    negative-curvature threshold (P.hopf_lax_mpc_epsc = 1e-3): lmin < -epsc certifies an index-1 saddle
        eta_s   detection gate on the relative gradient (HOPF_LAX_MPC_ETA_S)
        alpha   kick length along the unstable eigenvector (P.hopf_lax_mpc_alpha = 0.30)
        delta   absolute eigenvalue floor of the escape step (HOPF_LAX_MPC_DELTA)
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
                                                                      # Hopf_Lax_MPC adds over PMP (reported in info/cost)

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

            if ridge < HOPF_LAX_MPC_RIDGE:                                    # ON-RIDGE (< 0.1): gradient carries no usable
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
            Vt = v[None, :] + HOPF_LAX_MPC_ESC_ALPHAS[:, None] * step[None, :]# 7 trial iterates, alpha = 1 ... 1/64

            Jts = np.asarray(Jb(Vt, xc, Xr))                          # ALL alphas, one batched dispatch

            ok = np.isfinite(Jts) & (Jts < J - 1e-9 * abs(J))         # ESCAPE JUDGES ON THE COST Phi, not ||res||:
                                                                      # the point is to LEAVE a root, where ||res||
                                                                      # necessarily grows. The relative margin
                                                                      # rejects round-off-sized "improvements".

            ac = bool(ok.any())

            a = float(HOPF_LAX_MPC_ESC_ALPHAS[int(np.argmax(ok))]) if ac else float(HOPF_LAX_MPC_ESC_ALPHAS[-1])
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
#  iteration. This is the fair, strongest linearization baseline against the (linearization-free) Hopf_Lax_MPC.
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
#  COLLOCATION  (CasADi + IPOPT)  --  direct port of build_collocation_ipopt
# ======================================================================================================
def build_collocation_ipopt(M, N, dt, budget):
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


def shift_collocation_ipopt(w, n, m, N, ns):
    X = w[:n * (N + 1)].reshape((n, N + 1), order="F")
    U = w[n * (N + 1):].reshape((m, N), order="F")
    Xs = np.hstack([X[:, ns:], np.repeat(X[:, -1:], ns, axis=1)])
    Us = np.hstack([U[:, ns:], np.repeat(U[:, -1:], ns, axis=1)])
    return np.concatenate([Xs.flatten(order="F"), Us.flatten(order="F")])


# ======================================================================================================
#  MPPI  --  GPU sampling MPC (formerly mpc_mppi.py)
# ======================================================================================================
#  MPPI (Williams et al., information-theoretic MPC) as a CuPy RawKernel on the GPU.
#
#  WHY A HAND-WRITTEN CUDA KERNEL AND NOT vmap/BATCHED TENSORS
#      The K samples parallelise, but the N horizon steps are inherently SEQUENTIAL. A batched-tensor
#      formulation (PyTorch/TorchRL-style) dispatches kernels per timestep and measures ~25-30 ms per
#      solve REGARDLESS of K -- it is dispatch-bound, and would blow the entire 20 ms budget doing no
#      useful work. Published benchmarks (MPPI-Generic, arXiv:2409.07563, RTX 3080, N=100): a fused CUDA
#      kernel does K=8192 in 0.36 ms vs TorchRL's 28.7 ms. That is the ~100x gap this file exists to avoid.
#
#      Here one thread owns one sample and runs the whole horizon in-thread: ONE kernel launch per control
#      cycle, the 7-dim state living in registers and never touching global memory.
#
#  FAIRNESS
#      The cost is the SAME cost the other four methods minimise: quadratic tracking + the identical tanh
#      ellipsoid barriers (static + moving-at-current-position) + the same terminal QT. Controls are
#      clamped INSIDE the rollout, so a sample's cost reflects what the actuator would really do. The
#      nominal sequence is warm-started by shifting, exactly like the other methods' warm starts.
#
#  TWO PROPERTIES OF THIS PROBLEM WORTH KNOWING WHEN READING THE RESULTS
#      * The barriers peak at W=500 against an O(1) tracking cost, so sample costs span ~3 decades. lambda
#        must be scaled to the SPREAD of S_k, not its magnitude, or the softmax collapses onto one sample.
#        ESS is logged every cycle to make that visible rather than silent.
#      * tanh SATURATES. Deep inside a barrier every sample gets the same cost, and MPPI -- being
#        zeroth-order -- then has no signal about which way is out. This is a real limitation of MPPI here,
#        not a bug to tune away.

# ======================================================================================================
#  CUDA C KERNEL.  One thread = one sample = one full RK4 rollout over the horizon.
#  Compiled once by NVRTC at first call and cached by CuPy; warm() below pays that cost up front so it
#  never lands inside a timed control cycle.
# ======================================================================================================
_KERNEL = r"""
extern "C" __global__
void mppi_rollout(
    const double* __restrict__ x0,        // (7,)   current state
    const double* __restrict__ Unom,      // (m*N,) nominal control sequence, row-major [t*m + j]
    const double* __restrict__ eps,       // (K*N*m,) pre-generated noise [k*N*m + t*m + j]
    const double* __restrict__ Xref,      // (7*(N+1),) tracking reference [t*7 + i]
    const double* __restrict__ oc,        // (3*Ko,) static obstacle centres  [i*Ko + j] column-major
    const double* __restrict__ oax, const double* __restrict__ oay, const double* __restrict__ oaz,
    const double* __restrict__ oW,  const double* __restrict__ ow,
    const double* __restrict__ mc,        // (3*Km,) moving obstacle centres at the CURRENT time
    const double* __restrict__ max_, const double* __restrict__ may_, const double* __restrict__ maz_,
    const double* __restrict__ mW_,  const double* __restrict__ mw_,
    const double* __restrict__ Qd,        // (7,) diag(Q)
    const double* __restrict__ QTd,       // (7,) diag(QT)
    const double* __restrict__ Rd,        // (3,) diag(R)
    const double* __restrict__ umax,      // (3,)
    const double* __restrict__ gsig,      // (3,) gamma * Sigma^-1 = gamma/sigma_j^2, folded per channel
    double* __restrict__ Sout,            // (K,) OUT: sample cost
    const int K, const int N, const int Ko, const int Km,
    const double dt, const double eps_barrier, const int n_unbiased)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    // ---- state in registers for the whole rollout: never spills to global memory ----
    double x[7];
    #pragma unroll
    for (int i = 0; i < 7; ++i) x[i] = x0[i];

    double S = 0.0;
    const int m = 3;

    for (int t = 0; t < N; ++t) {
        // ---- v = u + eps  (importance-sampled), or v = eps for the unbiased "reset" samples ----
        double v[3], un[3];
        #pragma unroll
        for (int j = 0; j < m; ++j) {
            const double e = eps[(size_t)k * N * m + (size_t)t * m + j];
            const double u = Unom[t * m + j];
            v[j] = (k < n_unbiased) ? e : (u + e);
            // clamp INSIDE the rollout: the cost must describe the control the actuator would apply
            un[j] = fmin(fmax(v[j], -umax[j]), umax[j]);
            // control cost gamma * u_t^T Sigma^-1 v_t; Sigma is diagonal so it folds per channel
            S += gsig[j] * u * v[j];
        }

        // ---- RK4 step of dyn7: [v cos th, v sin th, vz, omega, u1, u2, u3] ----
        double k1[7], k2[7], k3[7], k4[7], xt[7];
        #define DYN(SS, XX) { \
            SS[0] = XX[4]*cos(XX[3]); SS[1] = XX[4]*sin(XX[3]); SS[2] = XX[6]; \
            SS[3] = XX[5]; SS[4] = un[0]; SS[5] = un[1]; SS[6] = un[2]; }
        DYN(k1, x)
        #pragma unroll
        for (int i = 0; i < 7; ++i) xt[i] = x[i] + 0.5*dt*k1[i];
        DYN(k2, xt)
        #pragma unroll
        for (int i = 0; i < 7; ++i) xt[i] = x[i] + 0.5*dt*k2[i];
        DYN(k3, xt)
        #pragma unroll
        for (int i = 0; i < 7; ++i) xt[i] = x[i] + dt*k3[i];
        DYN(k4, xt)
        #pragma unroll
        for (int i = 0; i < 7; ++i) x[i] += dt/6.0*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        #undef DYN

        // ---- stage cost at t+1: tracking + control + barriers (branch-free, so no warp divergence) ----
        double q = 0.0;
        #pragma unroll
        for (int i = 0; i < 7; ++i) { const double d = x[i] - Xref[(t+1)*7 + i]; q += Qd[i]*d*d; }
        #pragma unroll
        for (int j = 0; j < m; ++j) q += Rd[j]*un[j]*un[j];

        for (int i = 0; i < Ko; ++i) {                       // static ellipsoid barriers
            const double dx = x[0]-oc[0*Ko+i], dy = x[1]-oc[1*Ko+i], dz = x[2]-oc[2*Ko+i];
            const double rho = sqrt((dx/oax[i])*(dx/oax[i]) + (dy/oay[i])*(dy/oay[i])
                                  + (dz/oaz[i])*(dz/oaz[i]) + eps_barrier);
            q += oW[i]*0.5*(1.0 - tanh((rho-1.0)/ow[i]));
        }
        for (int i = 0; i < Km; ++i) {                       // moving barriers, frozen at current pos
            const double dx = x[0]-mc[0*Km+i], dy = x[1]-mc[1*Km+i], dz = x[2]-mc[2*Km+i];
            const double rho = sqrt((dx/max_[i])*(dx/max_[i]) + (dy/may_[i])*(dy/may_[i])
                                  + (dz/maz_[i])*(dz/maz_[i]) + eps_barrier);
            q += mW_[i]*0.5*(1.0 - tanh((rho-1.0)/mw_[i]));
        }
        S += q * dt;
    }

    // ---- terminal cost: same QT the other methods use ----
    #pragma unroll
    for (int i = 0; i < 7; ++i) { const double d = x[i] - Xref[N*7 + i]; S += QTd[i]*d*d; }
    Sout[k] = S;
}
"""


class MPPI:
    """MPPI controller. K samples, horizon N, on the GPU.

    Parameters follow Williams et al.:
        lam    temperature. Scaled to the SPREAD of the sample costs, not their magnitude.
        sigma  exploration std per control channel (~0.3 x control range is the published ballpark).
        alpha  fraction of zero-mean "reset" samples (~1%), which guarantee recovery if the nominal
               sequence becomes garbage.
        gamma  = lam*(1-alpha); decouples control-authority penalty from temperature.
        iters  optimisation iterations per cycle. Default 1 = canonical MPPI, which relies on the
               warm-start shift to accumulate progress across cycles.
    """

    def __init__(self, M, P, N, K, lam=None, sigma=None, alpha=0.01, iters=1, seed=0, sg_window=9,
                 ess_target=0.05):
        if not HAVE_GPU:
            raise RuntimeError("cupy unavailable -- MPPI needs the GPU")
        self.M, self.P, self.N, self.K = M, P, N, K
        self.m = M.m
        self.iters = iters
        self.alpha = alpha
        # sigma ~ 0.3 x control range (range is 2*umax since bounds are symmetric)
        self.sigma = np.asarray(sigma if sigma is not None else 0.3 * (2 * M.umax), float)
        # ess_target != None  -> lambda is auto-tuned per cycle to hit this effective-sample-size
        # fraction. THIS IS NECESSARY HERE, not a refinement: the cost spread varies by ~3 decades along
        # the run (tanh barriers peak at W=500 against an O(1) tracking cost in free space), so ONE fixed
        # lambda cannot serve both regimes. Measured: lam=20 gives 97% ESS in open space -- weights go
        # uniform, the update averages zero-mean noise to ~0, and MPPI simply stops steering.
        # Auto-tuning the temperature to a fixed ESS is the standard fix in the ESS literature.
        self.ess_target = ess_target
        self.lam = float(lam) if lam is not None else 20.0
        self.gamma = self.lam * (1.0 - alpha)
        self.n_unbiased = int(round(alpha * K))                # first n_unbiased samples ignore the nominal
        self.rng = cp.random.default_rng(seed)
        self.U = np.zeros((self.m, N))                         # nominal sequence (host copy)
        self.kern = cp.RawKernel(_KERNEL, "mppi_rollout")
        self.block = 256                                       # multiple of 32 (warp size)
        self.grid = (K + self.block - 1) // self.block
        self.ess_log = []

        # ---- device-side constants (uploaded once) ----
        d = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float64))
        o, mo = M.obs, M.mobs
        self.Ko, self.Km = o.center.shape[1], mo.c0.shape[1]
        self.d_oc = d(o.center.ravel())                        # row-major (3,Ko) -> [i*Ko+j]
        self.d_oax, self.d_oay, self.d_oaz = d(o.ax), d(o.ay), d(o.az)
        self.d_oW, self.d_ow = d(o.W), d(o.w)
        self.d_max, self.d_may, self.d_maz = d(mo.ax), d(mo.ay), d(mo.az)
        self.d_mW, self.d_mw = d(mo.W), d(mo.w)
        self.d_Q, self.d_QT, self.d_R = d(np.diag(M.Q)), d(np.diag(M.QT)), d(np.diag(M.R))
        self.d_umax = d(M.umax)
        self.d_gsig = d(self.gamma / self.sigma ** 2)                   # gamma * Sigma^-1, per channel
        self.eps_barrier = o.eps
        self.d_S = cp.zeros(K, dtype=cp.float64)
        self._sg = _sg_coeffs(sg_window, 3) if sg_window else None

    def _tune_lambda(self, S, rho):
        """Bisect lambda so the weights hit `ess_target` x K effective samples.

        ESS(lambda) = 1/sum_k w_k^2 is monotone increasing in lambda: lambda->0 collapses onto the single
        best sample (ESS->1, i.e. random shooting), lambda->inf makes the weights uniform (ESS->K, i.e.
        no selectivity). So a plain bisection on log10(lambda) is well-posed and converges in ~30 cheap
        GPU reductions (~0.2 ms at K=8192 -- affordable inside a 20 ms budget).
        """
        # The bisection runs on the HOST over a SUBSAMPLE of the costs. Both choices are for latency:
        # iterating on the device costs one device->host sync per step (~30 syncs measured at ~4.6 ms,
        # nearly as much as the rollout kernel itself), and the ESS *fraction* is a distributional
        # statistic that a few thousand samples estimate perfectly well. The final weights are still
        # formed on-device from the full K samples, so nothing downstream is approximated.
        nsub = min(self.K, 2048)
        d_h = cp.asnumpy(S[:nsub] - rho)
        lo, hi = -6.0, 6.0                                     # log10 lambda search bracket
        target = self.ess_target * nsub
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            w = np.exp(-d_h / (10.0 ** mid))
            w = w / w.sum()
            ess = 1.0 / np.sum(w ** 2)
            if ess < target:
                lo = mid                                       # too selective -> raise lambda
            else:
                hi = mid
        self.lam = 10.0 ** (0.5 * (lo + hi))
        self.gamma = self.lam * (1.0 - self.alpha)
        self.d_gsig = cp.asarray(np.ascontiguousarray(self.gamma / self.sigma ** 2, dtype=np.float64))
        return self.lam

    def _sample_costs(self, x, Xref, mc, U, eps):
        """One kernel launch -> S_k for every sample."""
        self.kern((self.grid,), (self.block,), (
            cp.asarray(x), cp.asarray(np.ascontiguousarray(U.T.ravel())), eps,
            cp.asarray(np.ascontiguousarray(Xref[:, :self.N + 1].T.ravel())),
            self.d_oc, self.d_oax, self.d_oay, self.d_oaz, self.d_oW, self.d_ow,
            cp.asarray(np.ascontiguousarray(mc.ravel())),
            self.d_max, self.d_may, self.d_maz, self.d_mW, self.d_mw,
            self.d_Q, self.d_QT, self.d_R, self.d_umax, self.d_gsig, self.d_S,
            np.int32(self.K), np.int32(self.N), np.int32(self.Ko), np.int32(self.Km),
            np.float64(self.P.dt), np.float64(self.eps_barrier),
            np.int32(self.n_unbiased)))
        return self.d_S

    def solve(self, x, Xref, mc):
        """One control cycle. Returns (u0, U, ess)."""
        U = self.U
        ess = np.nan
        for _ in range(self.iters):
            # Noise is generated ON DEVICE: a host->device transfer of K*N*m doubles would cost ~1 ms and
            # dwarf the kernel itself. It is kept because the weighted update needs eps again below.
            eps = self.rng.standard_normal((self.K, self.N, self.m), dtype=cp.float64)
            eps = eps * cp.asarray(self.sigma)[None, None, :]
            S = self._sample_costs(x, Xref, mc, U, cp.ascontiguousarray(eps))
            # min-shift is mandatory numerics: without it exp(-S/lam) underflows to 0 for EVERY sample
            # (our S spans ~1e3) and the update becomes 0/0.
            rho = S.min()
            lam = self._tune_lambda(S, rho) if self.ess_target else self.lam
            wgt = cp.exp(-(S - rho) / lam)
            wgt = wgt / wgt.sum()
            dU = cp.einsum("k,ktj->jt", wgt, eps)             # sum_k w_k eps_t^k
            U = U + cp.asnumpy(dU)
            ess = float(1.0 / cp.asnumpy(cp.sum(wgt ** 2)))   # effective sample size: the starvation gauge
            self._last_eps, self._last_w = eps, wgt           # kept for the animation's sample cloud
            if self._sg is not None:
                U = _sg_filter(U, self._sg)                   # Savitzky-Golay along time (Williams)
            U = np.clip(U, -self.M.umax[:, None], self.M.umax[:, None])
        self.U = U
        self.ess_log.append(ess)
        return U[:, 0].copy(), U, ess

    def shift(self, ns):
        """Warm start: slide the sequence down and repeat the tail, matching the other methods."""
        self.U = np.hstack([self.U[:, ns:], np.repeat(self.U[:, -1:], ns, axis=1)])

    def top_sample_controls(self, nsel=24):
        """The nsel highest-weight sampled control sequences, as (nsel, m, N) on the host.

        For the animation only, and called OUTSIDE the timed section: the sample cloud is MPPI's most
        legible visual signature -- it shows directly whether the cloud straddles an obstacle or has
        collapsed onto one sample (which is what ESS reports numerically).
        """
        if getattr(self, "_last_eps", None) is None:
            return np.zeros((0, self.m, self.N))
        nsel = min(nsel, self.K)
        idx = cp.argsort(self._last_w)[::-1][:nsel]
        eps = cp.asnumpy(self._last_eps[idx])                 # (nsel, N, m)
        V = self.U[None, :, :] + eps.transpose(0, 2, 1)       # nominal + noise -> (nsel, m, N)
        return np.clip(V, -self.M.umax[None, :, None], self.M.umax[None, :, None])

    def warm(self, x, Xref, mc):
        """Pay the one-time NVRTC compile (~40 ms) before any timed cycle."""
        U0 = self.U.copy()
        self.solve(x, Xref, mc)
        self.U = U0
        self.ess_log.clear()
        cp.cuda.Stream.null.synchronize()


def _sg_coeffs(window, order):
    """Savitzky-Golay smoothing coefficients (local polynomial least squares)."""
    if window % 2 == 0:
        window += 1
    half = window // 2
    A = np.vander(np.arange(-half, half + 1), order + 1, increasing=True)
    return np.linalg.pinv(A.T @ A) @ A.T                      # row 0 = smoothing weights


def _sg_filter(U, C):
    """Apply the SGF along the time axis. MPPI's update is a Monte-Carlo average and is therefore noisy
    in time; this is Williams' original fix for the resulting control chatter."""
    w = C[0]
    half = len(w) // 2
    Up = np.pad(U, ((0, 0), (half, half)), mode="edge")
    return np.stack([np.convolve(Up[j], w[::-1], mode="valid") for j in range(U.shape[0])])


# ======================================================================================================
#  VANILLA HOPF-LAX  --  damped Newton with the exact Hessian (formerly mpc_newton.py)
# ======================================================================================================
#  Vanilla_Hopf_Lax baseline: damped Newton on the Hopf-Lax objective with the EXACT Hessian.
#
#  The paper's method (mpc_solvers.hopf_lax_mpc_solve, "Hopf-Lax-MPC") minimises Phi(p0) = S.J(p0, xc, Xr)
#  with a Hessian SURROGATE M = -Sx'Jr that falls out of the one fused forward pass, a gated eigen read
#  and a saddle escape.  This module is the obvious naive alternative: the same objective, the exact
#  second derivative grad^2 Phi from jax.hessian at every iteration, Levenberg damping (H + mu I) d = -g,
#  a batched Armijo line search on Phi, and NO curvature certificate / NO escape -- it stops at any
#  stationary point, saddles included.
#
#  Design notes
#    * gradient: grad Phi = -Sx' rho from S.all (identical formula and identical stopping test
#      |grad| <= tol (1 + |Phi|) as hopf_lax_mpc_solve, so iteration counts are comparable).  S.g (jax.grad of
#      the discrete objective) agrees with -Sx'rho up to RK4 discretisation; the sweep --selftest prints
#      the difference.
#    * Hessian: S.H = jax.jit(jax.hessian(S.J)) attached once per shooting object (ensure_hessian), the
#      exact pattern of the 2026-07 verify_signread.py study.  Symmetrised in numpy.  jax.hessian is
#      forward-over-reverse through the lax.scan rollout; `fwdfwd` / `fwdgrad` are fallbacks.
#    * damping: smallest mu >= current mu with H + mu I positive definite (Cholesky attempt, mu x10 on
#      failure).  A rejected line search raises mu x10 and retries the SAME iterate with the cached
#      (Phi, grad, H).  A full accepted step (alpha = 1) relaxes mu / 3.
#    * line search: ONE batched dispatch of S.Jbatch over alpha in {1, 1/2, ..., 1/64} (the same ladder
#      and the same batching as the escape line search of hopf_lax_mpc_solve); the FIRST alpha satisfying
#      Armijo  Phi(v + a d) <= Phi(v) + c a grad.d  is taken.
#    * status: converged | cap | stalled (mu exhausted, or an immeasurable accepted step) | diverged
#      (non-finite Phi / grad / H) | timeout (only with a finite budget).  Never raises.
VANILLA_HOPF_LAX_ALPHAS = 2.0 ** -np.arange(7)                                 # 1, 1/2, ..., 1/64 (as HOPF_LAX_MPC_ESC_ALPHAS)


def ensure_hessian(S, mode: str = "hessian"):
    """Attach (once) a jitted exact Hessian of the Hopf-Lax objective to a shooting object."""
    if not hasattr(S, "H"):
        if mode == "hessian":
            S.H = jax.jit(jax.hessian(S.J))                          # = jacfwd(jacrev): forward-over-reverse
        elif mode == "fwdfwd":
            S.H = jax.jit(jax.jacfwd(jax.jacfwd(S.J)))               # no reverse pass through the scan
        elif mode == "fwdgrad":
            S.H = jax.jit(jax.jacfwd(S.g))                           # forward over the existing jitted grad
        else:
            raise ValueError(f"unknown hessian mode {mode!r}")
        S.H_mode = mode
    return S.H


def vanilla_hopf_lax_solve(S, v, xc, Xr, PP, tc, budget, trace=None, mu0=1e-6, mu_min=1e-8, mu_max=1e8,
                 c_armijo=1e-4, step_tol=1e-10, alphas=VANILLA_HOPF_LAX_ALPHAS):
    """Damped Newton on Phi(v) with the exact Hessian.  PP = SimpleNamespace(tol, maxit).

    Returns (v, it, info); it = accepted + rejected iterations (a rejected iteration re-uses the cached
    Hessian, so info.t_hess / info.nacc is the Hessian cost per fresh iterate)."""
    Hf = ensure_hessian(S)
    Jb = ensure_jbatch(S)
    n = v.size
    I = np.eye(n)
    v = np.array(v, dtype=float)
    mu = mu0
    it = nacc = nrej = nchol = 0
    t_hess = 0.0
    J = norm_grad = lminH = lminM = np.nan
    grad = H = None
    status, fresh = "cap", True
    for k in range(1, PP.maxit + 1):
        if k > 1 and (time.perf_counter() - tc) >= budget:            # HARD wall-clock cut (>=1 step done)
            status = "timeout"
            break
        if fresh:
            res, J, Jr, Sx = S.all(v, xc, Xr)                         # one fused rollout: rho, Phi, Jr, Sx
            res, J = np.asarray(res), float(J)
            Jr, Sx = np.asarray(Jr), np.asarray(Sx)
            grad = -Sx.T @ res                                        # grad Phi (Prop. 1)
            norm_grad = float(np.linalg.norm(grad))
            if not (np.isfinite(J) and np.all(np.isfinite(grad))):
                status = "diverged"
                break
            if norm_grad <= PP.tol * (1 + abs(J)):                    # stationarity ONLY (no curvature half)
                status = "converged"
                break
            t0 = time.perf_counter()
            H = np.asarray(Hf(v, xc, Xr))                             # exact grad^2 Phi, 7x7
            t_hess += time.perf_counter() - t0
            if not np.all(np.isfinite(H)):
                status = "diverged"
                break
            H = 0.5 * (H + H.T)
            lminH = float(np.linalg.eigvalsh(H)[0])                   # diagnostics (7x7, negligible)
            Mc = -(Sx.T @ Jr)
            lminM = float(np.linalg.eigvalsh(0.5 * (Mc + Mc.T))[0])
            fresh = False
        # ---- direction: smallest mu >= current with H + mu I positive definite ----
        cf = None
        while mu <= mu_max:
            try:
                cf = scipy.linalg.cho_factor(H + mu * I)
                break
            except np.linalg.LinAlgError:
                nchol += 1
                mu *= 10.0
        if cf is None:
            status = "stalled"
            break
        d = scipy.linalg.cho_solve(cf, -grad)
        gd = float(grad @ d)                                          # < 0 (PD system)
        # ---- batched Armijo line search: ONE dispatch, FIRST alpha that passes ----
        Vt = v[None, :] + alphas[:, None] * d[None, :]
        Jts = np.asarray(Jb(Vt, xc, Xr))
        ok = np.isfinite(Jts) & (Jts <= J + c_armijo * alphas * gd)
        it += 1
        rec = None
        if trace is not None:
            rec = dict(k=k, v=v.copy(), J=J, norm_grad=norm_grad, mu=mu, lminH=lminH, lminM=lminM,
                       gd=gd, a=None, ac=bool(ok.any()))
            trace.append(rec)
        if ok.any():
            i = int(np.argmax(ok))
            a = float(alphas[i])
            v = np.array(Vt[i])
            nacc += 1
            fresh = True
            if i == 0:
                mu = max(mu / 3.0, mu_min)                            # full step accepted: relax damping
            if rec is not None:
                rec["a"] = a
            if a * np.linalg.norm(d) <= step_tol * (1.0 + np.linalg.norm(v)):
                status = "stalled"                                    # accepted but immeasurable move
                break
        else:
            nrej += 1
            mu *= 10.0                                                # retry the same iterate, more damping
            if mu > mu_max:
                status = "stalled"
                break
    info = SimpleNamespace(status=status, nacc=nacc, nrej=nrej, nchol=nchol, mu=float(mu),
                           lminH=float(lminH), lminM=float(lminM), t_hess=float(t_hess),
                           norm_grad=float(norm_grad), J=float(J))
    return v, it, info

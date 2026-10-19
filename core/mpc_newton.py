"""
mpc_newton.py -- baseline: damped Newton on the Hopf-Lax objective with the EXACT Hessian.

The paper's method (mpc_solvers.chlqn_solve, "Hopf-Lax-MPC") minimises Phi(p0) = S.J(p0, xc, Xr)
with a Hessian SURROGATE M = -Sx'Jr that falls out of the one fused forward pass, a gated eigen read
and a saddle escape.  This module is the obvious naive alternative: the same objective, the exact
second derivative grad^2 Phi from jax.hessian at every iteration, Levenberg damping (H + mu I) d = -g,
a batched Armijo line search on Phi, and NO curvature certificate / NO escape -- it stops at any
stationary point, saddles included.

Design notes
  * gradient: grad Phi = -Sx' rho from S.all (identical formula and identical stopping test
    |grad| <= tol (1 + |Phi|) as chlqn_solve, so iteration counts are comparable).  S.g (jax.grad of
    the discrete objective) agrees with -Sx'rho up to RK4 discretisation; the sweep --selftest prints
    the difference.
  * Hessian: S.H = jax.jit(jax.hessian(S.J)) attached once per shooting object (ensure_hessian), the
    exact pattern of the 2026-07 verify_signread.py study.  Symmetrised in numpy.  jax.hessian is
    forward-over-reverse through the lax.scan rollout; `fwdfwd` / `fwdgrad` are fallbacks.
  * damping: smallest mu >= current mu with H + mu I positive definite (Cholesky attempt, mu x10 on
    failure).  A rejected line search raises mu x10 and retries the SAME iterate with the cached
    (Phi, grad, H).  A full accepted step (alpha = 1) relaxes mu / 3.
  * line search: ONE batched dispatch of S.Jbatch over alpha in {1, 1/2, ..., 1/64} (the same ladder
    and the same batching as the escape line search of chlqn_solve); the FIRST alpha satisfying
    Armijo  Phi(v + a d) <= Phi(v) + c a grad.d  is taken.
  * status: converged | cap | stalled (mu exhausted, or an immeasurable accepted step) | diverged
    (non-finite Phi / grad / H) | timeout (only with a finite budget).  Never raises.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import jax
import numpy as np
import scipy.linalg

from mpc_solvers import ensure_jbatch

NEWTON_ALPHAS = 2.0 ** -np.arange(7)                                 # 1, 1/2, ..., 1/64 (as M1V2_ESC_ALPHAS)


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


def newton_solve(S, v, xc, Xr, PP, tc, budget, trace=None, mu0=1e-6, mu_min=1e-8, mu_max=1e8,
                 c_armijo=1e-4, step_tol=1e-10, alphas=NEWTON_ALPHAS):
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

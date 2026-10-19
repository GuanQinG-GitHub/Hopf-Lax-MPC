"""
shared_stop.py -- ONE stopping rule for every iterative solver (branch `shared-stop`).

At every iteration the current iterate is turned into its control sequence U (m x N) and the cost of the
SAME discretized OCP is evaluated:

    J_ocp(U) = sum_k [ (x_k - xr_k)' Q (x_k - xr_k) + u_k' R u_k + barriers(x_k) ] dt + (x_N - xr_N)' QT (x_N - xr_N)

    with x_{k+1} = RK4(x_k, u_k, dt) (zero-order hold), the static barrier plus the moving barriers held at
    the current cycle's centres, and the sub=1 reference window -- i.e. exactly DDP's `rollout_cost`
    (mpc_solvers.build_ddp), which is reused here so the evaluator is literally the same function.

The solver stops as soon as   |J_k - J_{k-1}| <= eps * (1 + |J_k|)   (relative; eps = 1e-3 by default).
The first comparison is after one iteration (the warm start is J_0), so every solver does >= 1 step.

The evaluation is NOT part of a method's cost: `SharedStop` times itself (`t_spent`), the solvers subtract
it from their wall-clock budget check and the runners subtract it from the recorded solve time.

How each method supplies U:
    DDP                                  its control sequence U
    Collocation_IPOPT                    the U block of the primal iterate (iteration callback)
    Hopf_Lax_MPC, Vanilla_Hopf_Lax, PMP  `check_costate`: U = ustar along the canonical rollout of the costate
                                         iterate (mpc_core.build_ss -> roll_controls)
"""
from __future__ import annotations

import time

import numpy as np


def build_ocp_cost(M, N, dt, JX):
    """J_ocp(x0, U, Xref, mc) -> float: DDP's rollout_cost for horizon N (jitted on first call)."""
    import mpc_solvers as S                                            # local import: no cycle at module load
    DP = S.build_ddp(M, N, dt, JX)
    rc = DP.rollout_cost

    def cost(x0, U, Xref, mc):
        return float(rc(x0, U, Xref, mc)[1])
    return cost


class SharedStop:
    """Shared cost-change stopping test.  Call `reset(...)` once per control cycle, then the solver calls
    `__call__(U)` (or `check_costate(...)`) at the top of every iteration; True means 'stop now'."""

    def __init__(self, cost, eps=1e-3):
        self.cost, self.eps = cost, float(eps)
        self.reset(None, None, None)

    def reset(self, x, Xref, mc):
        self.x, self.Xref, self.mc = x, Xref, mc
        self.J_prev = None
        self.J_hist = []
        self.t_spent = 0.0                                             # seconds spent evaluating (untimed)
        self.fired = False
        self._last = None                                              # last evaluated iterate

    def _unchanged(self, it):
        """A rejected step leaves the iterate as it was; comparing it with itself would give dJ = 0 and a
        spurious stop, so the test only runs on a NEW iterate (consecutive accepted iterates)."""
        it = np.asarray(it)
        if self._last is not None and self._last.shape == it.shape and np.array_equal(self._last, it):
            return True
        self._last = it.copy()
        return False

    def _test(self, J):
        stop = self.J_prev is not None and abs(J - self.J_prev) <= self.eps * (1.0 + abs(J))
        self.J_prev = J
        self.J_hist.append(J)
        self.fired = self.fired or stop
        return stop

    def __call__(self, U):
        t0 = time.perf_counter()
        if self._unchanged(U):
            self.t_spent += time.perf_counter() - t0
            return False
        J = self.cost(self.x, np.asarray(U, dtype=float), self.Xref, self.mc)
        stop = self._test(J)
        self.t_spent += time.perf_counter() - t0
        return stop

    def check_costate(self, Sh, p, xc, Xr):
        """Costate methods: U from the canonical rollout of the iterate p, then the shared test."""
        t0 = time.perf_counter()
        if self._unchanged(p):
            self.t_spent += time.perf_counter() - t0
            return False
        U = np.asarray(Sh.roll_controls(p, xc, Xr))
        J = self.cost(self.x, U, self.Xref, self.mc)
        stop = self._test(J)
        self.t_spent += time.perf_counter() - t0
        return stop

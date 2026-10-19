"""
mpc_mppi.py -- MPPI (Williams et al., information-theoretic MPC) as a CuPy RawKernel on the GPU.

WHY A HAND-WRITTEN CUDA KERNEL AND NOT vmap/BATCHED TENSORS
    The K samples parallelise, but the N horizon steps are inherently SEQUENTIAL. A batched-tensor
    formulation (PyTorch/TorchRL-style) dispatches kernels per timestep and measures ~25-30 ms per
    solve REGARDLESS of K -- it is dispatch-bound, and would blow the entire 20 ms budget doing no
    useful work. Published benchmarks (MPPI-Generic, arXiv:2409.07563, RTX 3080, N=100): a fused CUDA
    kernel does K=8192 in 0.36 ms vs TorchRL's 28.7 ms. That is the ~100x gap this file exists to avoid.

    Here one thread owns one sample and runs the whole horizon in-thread: ONE kernel launch per control
    cycle, the 7-dim state living in registers and never touching global memory.

FAIRNESS
    The cost is the SAME cost the other four methods minimise: quadratic tracking + the identical tanh
    ellipsoid barriers (static + moving-at-current-position) + the same terminal QT. Controls are
    clamped INSIDE the rollout, so a sample's cost reflects what the actuator would really do. The
    nominal sequence is warm-started by shifting, exactly like the other methods' warm starts.

TWO PROPERTIES OF THIS PROBLEM WORTH KNOWING WHEN READING THE RESULTS
    * The barriers peak at W=500 against an O(1) tracking cost, so sample costs span ~3 decades. lambda
      must be scaled to the SPREAD of S_k, not its magnitude, or the softmax collapses onto one sample.
      ESS is logged every cycle to make that visible rather than silent.
    * tanh SATURATES. Deep inside a barrier every sample gets the same cost, and MPPI -- being
      zeroth-order -- then has no signal about which way is out. This is a real limitation of MPPI here,
      not a bug to tune away.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

try:
    import cupy as cp
    HAVE_GPU = True
except Exception:                                                      # keep the module importable on CPU
    cp = None
    HAVE_GPU = False


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

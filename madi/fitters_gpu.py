"""
madi.fitters_gpu — numba.cuda kernels for GPU-accelerated MADI voxel fitting
=============================================================================

Mirrors ``madi.walker_gpu``'s ``HAS_CUDA`` / CPU-fallback pattern: one CUDA
thread per voxel, no new dependency (numba is already required).

MAP (``match_voxels_batch``/``_fits0``) and Bayes (``bayes_fit``) are, on the
CPU, a single ``measured @ lib_m.T`` matmul followed by a row-wise reduction
(argmin, or softmax + weighted sum). The GPU kernels reproduce that *exact*
math per voxel without ever materializing the ``(n_vox, n_lib)`` distance
matrix — which would blow past a few GB of VRAM for realistic whole-brain
voxel counts against a several-thousand-entry library. Each thread streams
over the library rows for its own voxel, so device memory scales with
``n_vox + n_lib``, not their product, and no voxel chunking is required.
Output should match the CPU path to float64 precision.

AMICO (``amico_fit``) replaces the CPU's serial per-voxel
``scipy.optimize.nnls`` loop — the actual bottleneck in the CPU code — with
a per-thread FISTA (accelerated proximal/projected-gradient) solver for the
same elastic-net objective, positivity enforced by clamping to zero each
step. This is necessarily an *approximation* of exact active-set NNLS: any
solver that runs all voxels in parallel has to be iterative. AMICO-GPU
output is therefore close to, but not bit-identical to, AMICO-CPU output —
see ``docs/fitting_methods.md``. Each voxel's weight vector lives in
per-thread local memory (backed by device global memory, not registers),
sized to the actual candidate-library length via a small kernel-factory
cache, so voxels are processed in chunks mainly to bound host<->device
transfer size and give progress feedback, not because of a hard memory
ceiling.
"""

from __future__ import annotations

import functools
import math

import numpy as np

try:
    from numba import cuda
    import numba
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False

if not HAS_CUDA:
    import sys as _sys
    _sys.stderr.write(
        "WARNING: madi.fitters_gpu — CUDA not available, GPU fitters "
        "cannot run (CPU path will be used instead).\n")


_THREADS_PER_BLOCK = 128
_AMICO_THREADS_PER_BLOCK = 64

DEFAULT_GPU_CHUNK_VOXELS = 20_000
# NOTE on these two: convergence speed for AMICO's per-voxel FISTA solve
# depends strongly on lambda2. The ridge term isn't just a regularizer here
# — it's what conditions the (otherwise highly correlated / ill-conditioned)
# MADI library design matrix for a first-order solver. At the library's
# default lambda2=0.01 a few hundred iterations comfortably converge; as
# lambda2 -> 0, the number of iterations needed to actually reach the
# sparse, active-set-consistent optimum that CPU's exact NNLS finds
# directly can grow by 2-3 orders of magnitude. If you push --lambda2
# toward 0 and get suspiciously large/flat n_eff, raise --amico-gpu-iters
# substantially (tens of thousands) or use --device cpu for exact behavior.
DEFAULT_AMICO_ITERS = 2000
DEFAULT_AMICO_TOL = 1e-8


def _grid(n, threads_per_block=_THREADS_PER_BLOCK):
    return (n + threads_per_block - 1) // threads_per_block


def _weighted_mean_std_np(w, kios, rhos, Vs):
    """Same math as fitters._weighted_mean_std; kept local to avoid a
    fitters_gpu <-> fitters circular import (fitters.py dispatches into
    this module)."""
    out = {}
    for name, vals in (("kio", kios), ("rho", rhos), ("V", Vs)):
        mean = w @ vals
        mean2 = w @ (vals ** 2)
        var = np.maximum(mean2 - mean ** 2, 0.0)
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = np.sqrt(var)
    return out


# ===========================================================================
# MAP kernels (fixed-S0 and free-S0) — one thread per voxel
# ===========================================================================

if HAS_CUDA:

    @cuda.jit
    def _map_kernel_fixed(measured, lib, lib_l2, best_idx_out, best_resid_out):
        tid = cuda.grid(1)
        if tid >= measured.shape[0]:
            return
        n_feat = measured.shape[1]
        n_lib = lib.shape[0]

        m2 = 0.0
        for f in range(n_feat):
            m2 += measured[tid, f] * measured[tid, f]

        best_score = math.inf
        best_j = 0
        for j in range(n_lib):
            dot = 0.0
            for f in range(n_feat):
                dot += measured[tid, f] * lib[j, f]
            score = lib_l2[j] - 2.0 * dot
            if score < best_score:
                best_score = score
                best_j = j

        best_idx_out[tid] = best_j
        best_resid_out[tid] = m2 + best_score

    @cuda.jit
    def _map_kernel_free_s0(raw, lib, lib_rr, best_idx_out, best_resid_out,
                             best_s0_out):
        tid = cuda.grid(1)
        if tid >= raw.shape[0]:
            return
        n_feat = raw.shape[1]
        n_lib = lib.shape[0]

        mm = 0.0
        for f in range(n_feat):
            mm += raw[tid, f] * raw[tid, f]

        best_resid = math.inf
        best_j = 0
        best_s0 = 0.0
        for j in range(n_lib):
            mr = 0.0
            for f in range(n_feat):
                mr += raw[tid, f] * lib[j, f]
            rr = lib_rr[j]
            s0 = mr / rr
            resid = mm - (mr * mr) / rr
            if s0 <= 0.0:
                resid = math.inf
            if resid < best_resid:
                best_resid = resid
                best_j = j
                best_s0 = s0

        best_idx_out[tid] = best_j
        best_resid_out[tid] = best_resid
        best_s0_out[tid] = best_s0

    @cuda.jit
    def _bayes_kernel_fixed(measured, lib, lib_l2, kios, rhos, Vs,
                             inv_two_sigma2,
                             out_sumw, out_sumw_kio, out_sumw_kio2,
                             out_sumw_rho, out_sumw_rho2,
                             out_sumw_V, out_sumw_V2,
                             out_sumw_resid, out_sumw2):
        tid = cuda.grid(1)
        if tid >= measured.shape[0]:
            return
        n_feat = measured.shape[1]
        n_lib = lib.shape[0]

        m2 = 0.0
        for f in range(n_feat):
            m2 += measured[tid, f] * measured[tid, f]

        # Pass 1: min residual (numerical-stability shift, mirrors the CPU's
        # per-voxel max(log_w) subtraction).
        min_resid = math.inf
        for j in range(n_lib):
            dot = 0.0
            for f in range(n_feat):
                dot += measured[tid, f] * lib[j, f]
            resid = m2 + lib_l2[j] - 2.0 * dot
            if resid < min_resid:
                min_resid = resid

        # Pass 2: accumulate weighted sums.
        sw = 0.0; swk = 0.0; swk2 = 0.0
        swr = 0.0; swr2 = 0.0
        swv = 0.0; swv2 = 0.0
        swres = 0.0; sw2 = 0.0
        for j in range(n_lib):
            dot = 0.0
            for f in range(n_feat):
                dot += measured[tid, f] * lib[j, f]
            resid = m2 + lib_l2[j] - 2.0 * dot
            w = math.exp(-(resid - min_resid) * inv_two_sigma2)
            sw += w
            swk += w * kios[j]; swk2 += w * kios[j] * kios[j]
            swr += w * rhos[j]; swr2 += w * rhos[j] * rhos[j]
            swv += w * Vs[j];   swv2 += w * Vs[j] * Vs[j]
            swres += w * resid
            sw2 += w * w

        out_sumw[tid] = sw
        out_sumw_kio[tid] = swk; out_sumw_kio2[tid] = swk2
        out_sumw_rho[tid] = swr; out_sumw_rho2[tid] = swr2
        out_sumw_V[tid] = swv;   out_sumw_V2[tid] = swv2
        out_sumw_resid[tid] = swres
        out_sumw2[tid] = sw2

    @cuda.jit
    def _bayes_kernel_free_s0(raw, lib, lib_rr, kios, rhos, Vs,
                               inv_two_sigma2,
                               out_sumw, out_sumw_kio, out_sumw_kio2,
                               out_sumw_rho, out_sumw_rho2,
                               out_sumw_V, out_sumw_V2,
                               out_sumw_resid, out_sumw2, out_sumw_s0):
        tid = cuda.grid(1)
        if tid >= raw.shape[0]:
            return
        n_feat = raw.shape[1]
        n_lib = lib.shape[0]

        mm = 0.0
        for f in range(n_feat):
            mm += raw[tid, f] * raw[tid, f]

        # `resid` (below) is on raw signal units -- it scales like s0^2 --
        # but inv_two_sigma2 = 1/(2 sigma_m^2) is calibrated for the
        # normalized S/S0 scale (same convention as the fixed-S0 kernel).
        # Weighting directly by `resid` made the exponent astronomically
        # large relative to sigma_m whenever S0 ~ thousands (resid ~ S0^2
        # ~ millions), underflowing exp() to exactly 0 for every candidate
        # but the single best one -- collapsing the posterior to a one-hot
        # MAP-equivalent pick (n_eff=1, std=0) regardless of sigma_m.
        # `resid_w = resid / s0^2` converts it back to the same
        # normalized-signal residual the fixed-S0 kernel uses
        # (resid/s0^2 = ||raw/s0 - lib_j||^2); `resid` itself (raw units)
        # is kept for the reported residual output, matching the MAP
        # free-S0 matcher's convention.
        min_resid = math.inf
        for j in range(n_lib):
            mr = 0.0
            for f in range(n_feat):
                mr += raw[tid, f] * lib[j, f]
            rr = lib_rr[j]
            s0 = mr / rr
            resid = mm - (mr * mr) / rr
            if s0 <= 0.0:
                resid_w = math.inf
            else:
                resid_w = resid / (s0 * s0)
            if resid_w < min_resid:
                min_resid = resid_w

        sw = 0.0; swk = 0.0; swk2 = 0.0
        swr = 0.0; swr2 = 0.0
        swv = 0.0; swv2 = 0.0
        swres = 0.0; sw2 = 0.0; sws0 = 0.0
        # Degenerate-voxel guard: if every candidate got masked to +inf above
        # (no library entry gives a positive fitted S0 for this voxel — e.g.
        # background/low-SNR after Rician correction), min_resid is +inf too,
        # and `resid_w - min_resid` below would be `inf - inf = nan`, which
        # would poison every weighted sum for this voxel. Skip the loop
        # entirely in that case and leave all sums at their initialized 0
        # (same "no support" convention amico_fit already uses).
        if min_resid < math.inf:
            for j in range(n_lib):
                mr = 0.0
                for f in range(n_feat):
                    mr += raw[tid, f] * lib[j, f]
                rr = lib_rr[j]
                s0 = mr / rr
                resid = mm - (mr * mr) / rr
                if s0 <= 0.0:
                    resid_w = math.inf
                else:
                    resid_w = resid / (s0 * s0)
                # exp(-inf) == 0.0: masked candidates naturally get zero weight.
                w = math.exp(-(resid_w - min_resid) * inv_two_sigma2)
                sw += w
                swk += w * kios[j]; swk2 += w * kios[j] * kios[j]
                swr += w * rhos[j]; swr2 += w * rhos[j] * rhos[j]
                swv += w * Vs[j];   swv2 += w * Vs[j] * Vs[j]
                if resid == resid and resid != math.inf:  # finite check
                    swres += w * resid
                sw2 += w * w
                sws0 += w * s0

        out_sumw[tid] = sw
        out_sumw_kio[tid] = swk; out_sumw_kio2[tid] = swk2
        out_sumw_rho[tid] = swr; out_sumw_rho2[tid] = swr2
        out_sumw_V[tid] = swv;   out_sumw_V2[tid] = swv2
        out_sumw_resid[tid] = swres
        out_sumw2[tid] = sw2
        out_sumw_s0[tid] = sws0


# ===========================================================================
# AMICO: per-thread FISTA kernel, specialized (compiled once) per
# (n_lib, n_feat) via a small factory cache — local arrays need
# compile-time-constant shapes.
# ===========================================================================

@functools.lru_cache(maxsize=8)
def _build_amico_kernel(n_lib, n_feat):
    f64 = numba.float64

    @cuda.jit
    def _amico_kernel(D, M, lam1, lam2, step_size, n_iters, tol, x_out):
        tid = cuda.grid(1)
        if tid >= M.shape[0]:
            return

        x = cuda.local.array(n_lib, dtype=f64)
        y = cuda.local.array(n_lib, dtype=f64)
        r = cuda.local.array(n_feat, dtype=f64)

        for j in range(n_lib):
            x[j] = 0.0
            y[j] = 0.0

        t = 1.0
        prev_obj = math.inf
        stable_count = 0
        for _it in range(n_iters):
            # r = D @ y - m
            for f in range(n_feat):
                acc = 0.0
                for j in range(n_lib):
                    acc += D[f, j] * y[j]
                r[f] = acc - M[tid, f]

            # Scalar FISTA momentum coefficients (pure sequence, independent
            # of x/y values) — computed once per iteration, applied per-j
            # below without needing a third buffer for the previous x.
            t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) * 0.5
            beta = (t - 1.0) / t_new
            t = t_new

            reg = 0.0
            for j in range(n_lib):
                gradj = 0.0
                for f in range(n_feat):
                    gradj += D[f, j] * r[f]
                gradj = 2.0 * gradj + 2.0 * lam2 * y[j] + lam1

                x_old_j = x[j]
                new_val = y[j] - step_size * gradj
                if new_val < 0.0:
                    new_val = 0.0

                x[j] = new_val
                y[j] = new_val + beta * (new_val - x_old_j)
                reg += lam1 * y[j] + lam2 * y[j] * y[j]

            # Convergence check: relative change in the elastic-net
            # objective (evaluated at y, using the r already computed
            # above), not the per-coordinate step size. A per-coordinate
            # check is unreliable here: MADI library entries are highly
            # correlated (ill-conditioned D^T D), so most coordinates move
            # in tiny steps from iteration 1 regardless of how far the
            # overall solution is from converged — that check declares
            # "converged" almost immediately and freezes a diffuse,
            # non-sparse solution independent of lambda2. Require several
            # consecutive small-relative-change iterations (FISTA's
            # objective can wobble slightly step to step) before exiting.
            data_term = 0.0
            for f in range(n_feat):
                data_term += r[f] * r[f]
            obj = data_term + reg

            if prev_obj < math.inf:
                denom = obj if obj > 1e-30 else 1e-30
                rel_change = abs(prev_obj - obj) / denom
                if rel_change < tol:
                    stable_count += 1
                    if stable_count >= 20:
                        break
                else:
                    stable_count = 0
            prev_obj = obj

        for j in range(n_lib):
            x_out[tid, j] = x[j]

    return _amico_kernel


def _estimate_lipschitz(D, lambda2, n_power_iter=50, seed=0):
    """Power-iteration estimate of the Lipschitz constant of the smooth
    part's gradient (``2 D^T D + 2 lambda2 I``), for the FISTA step size.
    D is (n_feat, n_lib); cost is trivial at these library sizes."""
    n_lib = D.shape[1]
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n_lib)
    nrm = np.linalg.norm(v)
    v = v / nrm if nrm > 1e-30 else v
    lam_max = 0.0
    for _ in range(n_power_iter):
        w = D.T @ (D @ v)
        nrm = np.linalg.norm(w)
        if nrm < 1e-30:
            break
        v = w / nrm
        lam_max = nrm
    L = 2.0 * lam_max + 2.0 * lambda2
    return float(max(L, 1e-8))


# ===========================================================================
# Host-side wrappers — signatures/return shapes mirror the CPU functions in
# madi.library / madi.fitters.
# ===========================================================================

def map_match_gpu(measured_batch, lib_mat, kios_arr, rhos_arr, Vs_arr):
    """GPU counterpart of ``library.match_voxels_batch``'s core matmul.

    ``measured_batch``/``lib_mat`` should already have any ``log_space``
    transform applied by the caller (the kernel is transform-agnostic).
    """
    measured = np.ascontiguousarray(measured_batch, dtype=np.float64)
    lib = np.ascontiguousarray(lib_mat, dtype=np.float64)
    lib_l2 = np.sum(lib * lib, axis=1)

    n_vox = measured.shape[0]
    d_measured = cuda.to_device(measured)
    d_lib = cuda.to_device(lib)
    d_lib_l2 = cuda.to_device(lib_l2)
    d_best_idx = cuda.device_array(n_vox, dtype=np.int64)
    d_best_resid = cuda.device_array(n_vox, dtype=np.float64)

    _map_kernel_fixed[_grid(n_vox), _THREADS_PER_BLOCK](
        d_measured, d_lib, d_lib_l2, d_best_idx, d_best_resid)
    cuda.synchronize()

    best_idx = d_best_idx.copy_to_host()
    best_resid = d_best_resid.copy_to_host()
    return (kios_arr[best_idx], rhos_arr[best_idx], Vs_arr[best_idx],
            best_resid)


def map_match_fits0_gpu(raw_signal, lib_mat, kios_arr, rhos_arr, Vs_arr):
    """GPU counterpart of ``library.match_voxels_batch_fits0``."""
    raw = np.ascontiguousarray(raw_signal, dtype=np.float64)
    lib = np.ascontiguousarray(lib_mat, dtype=np.float64)
    lib_rr = np.maximum(np.sum(lib * lib, axis=1), 1e-30)

    n_vox = raw.shape[0]
    d_raw = cuda.to_device(raw)
    d_lib = cuda.to_device(lib)
    d_lib_rr = cuda.to_device(lib_rr)
    d_best_idx = cuda.device_array(n_vox, dtype=np.int64)
    d_best_resid = cuda.device_array(n_vox, dtype=np.float64)
    d_best_s0 = cuda.device_array(n_vox, dtype=np.float64)

    _map_kernel_free_s0[_grid(n_vox), _THREADS_PER_BLOCK](
        d_raw, d_lib, d_lib_rr, d_best_idx, d_best_resid, d_best_s0)
    cuda.synchronize()

    best_idx = d_best_idx.copy_to_host()
    best_resid = d_best_resid.copy_to_host()
    best_s0 = d_best_s0.copy_to_host()
    return (kios_arr[best_idx], rhos_arr[best_idx], Vs_arr[best_idx],
            best_resid, best_s0)


def bayes_fit_gpu(measured_batch, lib_mat, kios_arr, rhos_arr, Vs_arr,
                   sigma_m, fit_s0=False):
    """GPU counterpart of ``fitters.bayes_fit``.

    ``measured_batch`` is the (already log-transformed if requested)
    normalized signal for ``fit_s0=False``, or the raw un-normalized signal
    for ``fit_s0=True`` — same convention as the CPU function.
    """
    inv_two_sigma2 = 1.0 / (2.0 * sigma_m ** 2)
    n_vox = measured_batch.shape[0]
    M = np.ascontiguousarray(measured_batch, dtype=np.float64)
    lib = np.ascontiguousarray(lib_mat, dtype=np.float64)
    d_M = cuda.to_device(M)
    d_lib = cuda.to_device(lib)
    d_kios = cuda.to_device(np.ascontiguousarray(kios_arr, dtype=np.float64))
    d_rhos = cuda.to_device(np.ascontiguousarray(rhos_arr, dtype=np.float64))
    d_Vs = cuda.to_device(np.ascontiguousarray(Vs_arr, dtype=np.float64))

    outs = [cuda.device_array(n_vox, dtype=np.float64) for _ in range(9)]
    grid = _grid(n_vox)

    if fit_s0:
        lib_rr = np.maximum(np.sum(lib * lib, axis=1), 1e-30)
        d_lib_rr = cuda.to_device(lib_rr)
        d_sumw_s0 = cuda.device_array(n_vox, dtype=np.float64)
        _bayes_kernel_free_s0[grid, _THREADS_PER_BLOCK](
            d_M, d_lib, d_lib_rr, d_kios, d_rhos, d_Vs, inv_two_sigma2,
            *outs, d_sumw_s0)
        cuda.synchronize()
        sumw_s0 = d_sumw_s0.copy_to_host()
    else:
        lib_l2 = np.sum(lib * lib, axis=1)
        d_lib_l2 = cuda.to_device(lib_l2)
        _bayes_kernel_fixed[grid, _THREADS_PER_BLOCK](
            d_M, d_lib, d_lib_l2, d_kios, d_rhos, d_Vs, inv_two_sigma2, *outs)
        cuda.synchronize()
        sumw_s0 = None

    (sumw, sumw_kio, sumw_kio2, sumw_rho, sumw_rho2,
     sumw_V, sumw_V2, sumw_resid, sumw2) = [o.copy_to_host() for o in outs]

    W = np.maximum(sumw, 1e-300)
    kio_mean = sumw_kio / W
    rho_mean = sumw_rho / W
    V_mean = sumw_V / W
    kio_std = np.sqrt(np.maximum(sumw_kio2 / W - kio_mean ** 2, 0.0))
    rho_std = np.sqrt(np.maximum(sumw_rho2 / W - rho_mean ** 2, 0.0))
    V_std = np.sqrt(np.maximum(sumw_V2 / W - V_mean ** 2, 0.0))
    residual = sumw_resid / W
    n_eff = (W * W) / np.maximum(sumw2, 1e-300)

    out = dict(kio_mean=kio_mean, rho_mean=rho_mean, V_mean=V_mean,
               kio_std=kio_std, rho_std=rho_std, V_std=V_std,
               residual=residual, n_eff=n_eff)
    if fit_s0:
        out["s0_fit"] = sumw_s0 / W
    return out


def amico_fit_gpu(M, lib_mat, kios_arr, rhos_arr, Vs_arr,
                   lambda1=0.0, lambda2=0.01,
                   n_iters=DEFAULT_AMICO_ITERS, tol=DEFAULT_AMICO_TOL,
                   gpu_chunk_voxels=DEFAULT_GPU_CHUNK_VOXELS,
                   verbose=True):
    """GPU counterpart of ``fitters.amico_fit``.

    Replaces the CPU's exact per-voxel ``scipy.optimize.nnls`` solve with a
    per-thread FISTA solve of the same elastic-net objective — an
    approximation (see module docstring), not a bit-identical result.
    """
    M = np.ascontiguousarray(M, dtype=np.float64)
    lib = np.ascontiguousarray(lib_mat, dtype=np.float64)
    D = np.ascontiguousarray(lib.T, dtype=np.float64)  # (n_feat, n_lib)
    n_vox, n_feat = M.shape
    n_lib = lib.shape[0]

    step_size = 1.0 / _estimate_lipschitz(D, lambda2)
    kernel = _build_amico_kernel(n_lib, n_feat)
    d_D = cuda.to_device(D)

    kio_mean = np.zeros(n_vox); rho_mean = np.zeros(n_vox); V_mean = np.zeros(n_vox)
    kio_std = np.zeros(n_vox);  rho_std = np.zeros(n_vox);  V_std = np.zeros(n_vox)
    residual = np.zeros(n_vox); n_eff = np.zeros(n_vox);    s0_fit = np.zeros(n_vox)

    n_chunks = (n_vox + gpu_chunk_voxels - 1) // gpu_chunk_voxels
    for ci in range(n_chunks):
        lo = ci * gpu_chunk_voxels
        hi = min(lo + gpu_chunk_voxels, n_vox)
        M_chunk = M[lo:hi]
        n_chunk = M_chunk.shape[0]

        d_M = cuda.to_device(M_chunk)
        d_x_out = cuda.device_array((n_chunk, n_lib), dtype=np.float64)

        kernel[_grid(n_chunk, _AMICO_THREADS_PER_BLOCK), _AMICO_THREADS_PER_BLOCK](
            d_D, d_M, lambda1, lambda2, step_size, n_iters, tol, d_x_out)
        cuda.synchronize()

        X = d_x_out.copy_to_host()  # (n_chunk, n_lib)
        s = X.sum(axis=1)
        s0_fit[lo:hi] = s
        residual[lo:hi] = np.sum((X @ lib - M_chunk) ** 2, axis=1)

        valid = s > 0
        w = np.zeros_like(X)
        w[valid] = X[valid] / s[valid, None]
        n_eff[lo:hi] = np.where(valid, 1.0 / np.maximum(np.sum(w ** 2, axis=1), 1e-300), 0.0)

        stats = _weighted_mean_std_np(w, kios_arr, rhos_arr, Vs_arr)
        kio_mean[lo:hi] = stats["kio_mean"]; kio_std[lo:hi] = stats["kio_std"]
        rho_mean[lo:hi] = stats["rho_mean"]; rho_std[lo:hi] = stats["rho_std"]
        V_mean[lo:hi] = stats["V_mean"];     V_std[lo:hi] = stats["V_std"]

        if verbose:
            print(f"    AMICO-GPU: {hi}/{n_vox} voxels "
                  f"(chunk {ci+1}/{n_chunks})", flush=True)

    return dict(
        kio_mean=kio_mean, rho_mean=rho_mean, V_mean=V_mean,
        kio_std=kio_std, rho_std=rho_std, V_std=V_std,
        residual=residual, n_eff=n_eff, s0_fit=s0_fit,
    )

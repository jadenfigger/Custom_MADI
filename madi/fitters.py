"""
madi.fitters — pluggable voxel-fitting methods for MADI
=======================================================

The original point-estimate matcher (``match_voxels_batch`` /
``match_voxels_batch_fits0`` in :mod:`madi.library`) picks the single
library entry with the smallest residual per voxel — a maximum-a-posteriori
(MAP) estimate under a flat prior.  This module adds two alternatives that
combine *all* library entries per voxel:

``BayesFitter``
    Soft posterior over the library.  Weight each entry by a Gaussian
    likelihood ``w_i ∝ exp(-||m - s_i||² / (2 σ_m²))``, then report the
    posterior mean and standard deviation of each parameter.

``AMICOFitter``
    AMICO-style constrained regression.  Solve
    ``x* = argmin_{x≥0} ||D x - m||² + λ₁||x||₁ + λ₂||x||²``, normalize the
    weights, and report weighted parameter means / stds plus an effective
    number of atoms ``n_eff = 1/Σ w_i²``.

All three share the same candidate-filtering / (Δ,b)-subsetting logic
(``_build_candidate_lib_matrix`` in :mod:`madi.library`) and the same
free-S₀ semantics as the MAP matcher, so their outputs are directly
comparable.  See ``docs/fitting_methods.md`` for the math and guidance.

The fitters return ``dict``s of flat per-voxel maps so that
``scripts/fit_data.py`` can dispatch on ``--method`` and write whichever
maps a given method produces.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np

from .library import _build_candidate_lib_matrix


# Default residual-noise std on the normalized signal, used when the user
# gives neither --sigma-m nor enough info to auto-estimate it.  2% of the
# S/S0 signal is a deliberately rough placeholder.
DEFAULT_SIGMA_M = 0.02

# Default elastic-net penalties for AMICO: pure ridge (no L1) with a light
# L2 shrinkage.  Chosen to be a gentle, well-conditioned starting point.
DEFAULT_LAMBDA1 = 0.0
DEFAULT_LAMBDA2 = 0.01


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _weighted_mean_std(w, kios, rhos, Vs):
    """Posterior/weighted mean and std of each parameter.

    Parameters
    ----------
    w : (n_vox, n_lib) row-normalized weights (each row sums to 1).
    kios, rhos, Vs : (n_lib,) parameter values per library entry.

    Returns
    -------
    dict with kio_mean/rho_mean/V_mean and kio_std/rho_std/V_std,
    each (n_vox,).
    """
    out = {}
    for name, vals in (("kio", kios), ("rho", rhos), ("V", Vs)):
        mean = w @ vals
        mean2 = w @ (vals ** 2)
        var = np.maximum(mean2 - mean ** 2, 0.0)   # clip tiny negatives
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = np.sqrt(var)
    return out


def estimate_sigma_m(sigma_rician, s0_median, mean_n_dir):
    """Propagate Rician σ through shell averaging and S₀ normalization.

    Each shell's measured value is a mean over ``mean_n_dir`` directions
    (so its noise std shrinks by ``sqrt(mean_n_dir)``) and is then divided
    by S₀ (median ``s0_median``).  This gives the approximate residual
    noise std on the normalized S/S₀ signal:

        σ_m ≈ σ_rician / (S0_median · sqrt(mean_n_dir))
    """
    if sigma_rician is None or s0_median is None or s0_median <= 0:
        return None
    mean_n_dir = max(float(mean_n_dir), 1.0)
    return float(sigma_rician) / (float(s0_median) * np.sqrt(mean_n_dir))


# ---------------------------------------------------------------------------
# Method 1: Bayesian posterior-mean fitting
# ---------------------------------------------------------------------------

def bayes_fit(
    measured_batch,
    library,
    *,
    sigma_m,
    fit_deltas=None, lib_deltas=None, n_b=4,
    vi_min=0.5, vi_max=0.95, rho_max=None,
    fit_pairs=None, lib_b_values=None,
    log_space=False, s_floor=1e-3,
    fit_s0=False, raw_signal=None,
):
    """Bayesian posterior-mean fit over the full candidate library.

    Weights ``w_i ∝ exp(-r_i / (2 σ_m²))`` where ``r_i`` is the squared
    residual of library entry ``i`` (fixed-S₀: ``||m - s_i||²``; free-S₀:
    ``||m||² - (m·s_i)²/(s_i·s_i)`` with negative-S₀ candidates excluded).
    Computed in log-space with a per-voxel max subtraction for stability.

    Returns
    -------
    dict with keys kio_mean, rho_mean, V_mean, kio_std, rho_std, V_std,
    residual (posterior-weighted mean residual), and — when ``fit_s0`` —
    s0_fit (posterior-mean fitted S₀).
    """
    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, fit_deltas, lib_deltas, n_b, vi_min, vi_max, rho_max,
        fit_pairs=fit_pairs, lib_b_values=lib_b_values)

    if sigma_m <= 0:
        raise ValueError(f"sigma_m must be > 0, got {sigma_m}")

    s0_mean = None
    if fit_s0:
        if raw_signal is None:
            raise ValueError("fit_s0=True requires raw_signal.")
        M = raw_signal.astype(np.float64)            # (n_vox, n_feat)
        R = lib_mat.astype(np.float64)               # (n_lib, n_feat)
        rr = np.maximum(np.sum(R * R, axis=1), 1e-30)
        mm = np.sum(M * M, axis=1)
        MR = M @ R.T                                 # (n_vox, n_lib)
        S0_cand = MR / rr[None, :]
        resid = mm[:, None] - (MR ** 2) / rr[None, :]
        # Forbid negative S0 (flipped signal): give it zero posterior mass.
        resid = np.where(S0_cand > 0, resid, np.inf)
    else:
        if log_space:
            measured = np.log(np.clip(measured_batch, s_floor, 1.0))
            lib_m = np.log(np.clip(lib_mat, s_floor, 1.0))
        else:
            measured = measured_batch.astype(np.float64)
            lib_m = lib_mat.astype(np.float64)
        m2 = np.sum(measured ** 2, axis=1, keepdims=True)
        l2 = np.sum(lib_m ** 2, axis=1, keepdims=True).T
        resid = np.maximum(m2 + l2 - 2.0 * measured @ lib_m.T, 0.0)

    # Posterior weights in log-space, stabilized by per-voxel max subtraction.
    log_w = -resid / (2.0 * sigma_m ** 2)
    log_w -= np.max(log_w, axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= np.sum(w, axis=1, keepdims=True)

    out = _weighted_mean_std(w, kios_arr, rhos_arr, Vs_arr)

    # Posterior-weighted mean residual (inf entries carry zero weight; guard
    # the 0·inf → nan that would otherwise appear).
    resid_finite = np.where(np.isfinite(resid), resid, 0.0)
    out["residual"] = np.sum(w * resid_finite, axis=1)

    if fit_s0:
        out["s0_fit"] = np.sum(w * S0_cand, axis=1)

    return out


# ---------------------------------------------------------------------------
# Method 2: AMICO-style NNLS with elastic-net regularization
# ---------------------------------------------------------------------------

def _make_nnls_solver(D, lambda2, lambda1):
    """Return a per-voxel solver for pure NNLS + ridge (λ₁ folded in).

    Solves ``argmin_{x≥0} ||D x - m||² + λ₁·1ᵀx + λ₂||x||²`` via an
    augmented least-squares system passed to ``scipy.optimize.nnls``.

    The ridge term augments the design with ``sqrt(λ₂)·I``.  The (linear,
    for x≥0) L1 term is folded in by completing the square:
    ``λ₂||x||² + λ₁·1ᵀx = λ₂||x - c||² + const`` with
    ``c = -λ₁/(2 λ₂) · 1``; this only works when ``λ₂ > 0``.
    """
    from scipy.optimize import nnls

    n_feat, n_lib = D.shape
    if lambda2 <= 0:
        # Pure NNLS (λ₁ must be 0 here; callers guarantee it).
        A = D
        b_lower = None
    else:
        A = np.vstack([D, np.sqrt(lambda2) * np.eye(n_lib)])
        if lambda1 > 0:
            c = -(lambda1 / (2.0 * lambda2)) * np.ones(n_lib)
            b_lower = np.sqrt(lambda2) * c
        else:
            b_lower = np.zeros(n_lib)

    def solve(m):
        if b_lower is None:
            b = m
        else:
            b = np.concatenate([m, b_lower])
        x, _ = nnls(A, b)
        return x

    return solve


def _make_elasticnet_solver(D, lambda1, lambda2):
    """Return a per-voxel solver using sklearn's positive ElasticNet.

    Maps our objective ``||D x - m||² + λ₁||x||₁ + λ₂||x||²`` onto
    sklearn's ``(1/2n)||y-Xw||² + α·l1·||w||₁ + (α(1-l1)/2)||w||²`` with
    ``n = n_feat`` samples:

        a = λ₁/(2n),  b = λ₂/n,  α = a + b,  l1_ratio = a/(a+b)
    """
    from sklearn.linear_model import ElasticNet

    n_feat = D.shape[0]
    a = lambda1 / (2.0 * n_feat)
    b = lambda2 / n_feat
    alpha = a + b
    l1_ratio = a / (a + b) if alpha > 0 else 0.0
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True,
                       fit_intercept=False, max_iter=10000, tol=1e-5)

    def solve(m):
        model.fit(D, m)
        return np.maximum(model.coef_, 0.0)

    return solve


def amico_fit(
    measured_batch,
    library,
    *,
    lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2,
    fit_deltas=None, lib_deltas=None, n_b=4,
    vi_min=0.5, vi_max=0.95, rho_max=None,
    fit_pairs=None, lib_b_values=None,
    fit_s0=False, raw_signal=None,
    verbose=True, progress_every=2000,
):
    """AMICO-style elastic-net NNLS fit.

    Per voxel, solve ``argmin_{x≥0} ||D x - m||² + λ₁||x||₁ + λ₂||x||²``
    where ``D`` has the (subset) library vectors as columns, then normalize
    ``w = x/Σx`` and report weighted parameter means/stds.

    With ``fit_s0`` the un-normalized signal is used as ``m``; because the
    weights ``x`` are unconstrained in magnitude they absorb the per-voxel
    amplitude, so ``Σx`` plays the role of the fitted S₀ (returned as
    ``s0_fit``) — the same "S₀ as a free linear parameter" semantics as the
    MAP free-S₀ matcher.

    Returns
    -------
    dict with keys kio_mean, rho_mean, V_mean, kio_std, rho_std, V_std,
    residual (``||D x - m||²`` at the solution), n_eff (``1/Σ w²``), and —
    when ``fit_s0`` — s0_fit (``Σx``).
    """
    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, fit_deltas, lib_deltas, n_b, vi_min, vi_max, rho_max,
        fit_pairs=fit_pairs, lib_b_values=lib_b_values)

    D = np.ascontiguousarray(lib_mat.T, dtype=np.float64)  # (n_feat, n_lib)

    if fit_s0:
        if raw_signal is None:
            raise ValueError("fit_s0=True requires raw_signal.")
        M = raw_signal.astype(np.float64)
    else:
        M = measured_batch.astype(np.float64)

    n_vox = M.shape[0]
    n_lib = D.shape[1]

    if lambda1 > 0.0:
        if lambda2 <= 0.0:
            solve = _make_elasticnet_solver(D, lambda1, lambda2)
        else:
            solve = _make_nnls_solver(D, lambda2, lambda1)
    else:
        solve = _make_nnls_solver(D, lambda2, 0.0)

    kio_mean = np.zeros(n_vox); rho_mean = np.zeros(n_vox); V_mean = np.zeros(n_vox)
    kio_std = np.zeros(n_vox);  rho_std = np.zeros(n_vox);  V_std = np.zeros(n_vox)
    residual = np.zeros(n_vox); n_eff = np.zeros(n_vox);    s0_fit = np.zeros(n_vox)

    t0 = time.time()
    for v in range(n_vox):
        m = M[v]
        x = solve(m)
        s = x.sum()
        residual[v] = float(np.sum((D @ x - m) ** 2))
        s0_fit[v] = s
        if s <= 0:
            # No support (e.g. an air voxel) — leave means/stds at 0.
            continue
        w = x / s
        kio_mean[v] = w @ kios_arr
        rho_mean[v] = w @ rhos_arr
        V_mean[v]   = w @ Vs_arr
        kio_std[v] = np.sqrt(max(w @ (kios_arr ** 2) - kio_mean[v] ** 2, 0.0))
        rho_std[v] = np.sqrt(max(w @ (rhos_arr ** 2) - rho_mean[v] ** 2, 0.0))
        V_std[v]   = np.sqrt(max(w @ (Vs_arr ** 2)   - V_mean[v] ** 2, 0.0))
        n_eff[v] = 1.0 / np.sum(w ** 2)

        if verbose and progress_every and (v + 1) % progress_every == 0:
            el = time.time() - t0
            rate = (v + 1) / el
            eta = (n_vox - v - 1) / rate if rate > 0 else float("nan")
            print(f"    AMICO: {v+1}/{n_vox} voxels "
                  f"({rate:.0f} vox/s, ETA {eta:.0f}s)", flush=True)

    out = dict(
        kio_mean=kio_mean, rho_mean=rho_mean, V_mean=V_mean,
        kio_std=kio_std, rho_std=rho_std, V_std=V_std,
        residual=residual, n_eff=n_eff,
    )
    if fit_s0:
        out["s0_fit"] = s0_fit
    return out

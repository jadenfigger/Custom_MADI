"""
DWI signal computation for multi-Δ SDE acquisition.

Given encoding-moment differences dM (n_walkers, n_deltas, 3), computes
S(b)/S₀ for each Δ and b-value.
"""

from __future__ import annotations

import numpy as np
from .config import SimConfig, GAMMA_RAD, BVALS_UNIQUE, BVALS_UNIQUE_INT
from .walker_gpu import WalkResult


def G_from_b(b_s_mm2: float, delta_ms: float, Delta_ms: float) -> float:
    """Gradient strength [T/m] for given b [s/mm²], δ [ms], Δ [ms]."""
    b_si = b_s_mm2 * 1e6        # s/m²  (1 s/mm² = 1e6 s/m²)
    d_si = delta_ms * 1e-3       # s
    tD_si = (Delta_ms - delta_ms / 3.0) * 1e-3
    if b_si <= 0 or tD_si <= 0:
        return 0.0
    return np.sqrt(b_si / ((GAMMA_RAD * d_si) ** 2 * tD_si))


def compute_signals(
    wr: WalkResult,
    cfg: SimConfig | None = None,
    b_values_s_mm2: np.ndarray | None = None,
) -> dict:
    """Compute S(b)/S₀ for each Δ and b-value.

    Parameters
    ----------
    wr : WalkResult
        Contains dM (n_walkers, n_deltas, 3).
    cfg : SimConfig
    b_values_s_mm2 : array, optional
        b-values [s/mm²].  Default: [1000, 2500, 4000, 6000].

    Returns
    -------
    dict with keys:
        'b_values'  : ndarray (n_b,) in s/mm²
        'Deltas'    : list of Δ values [ms]
        'signals'   : ndarray (n_deltas, n_b)  — S(b)/S₀
    """
    if cfg is None:
        cfg = SimConfig()
    if b_values_s_mm2 is None:
        b_values_s_mm2 = BVALS_UNIQUE.astype(float)

    bvals = np.asarray(b_values_s_mm2)
    Deltas = wr.deltas
    n_deltas = len(Deltas)
    n_b = len(bvals)

    # dM: (N, n_deltas, 3) in μm·ms → convert to SI (m·s)
    dM_si = wr.dM * 1e-6 * 1e-3

    signals = np.zeros((n_deltas, n_b))

    for di, Delta in enumerate(Deltas):
        delta = cfg.delta
        for bi, b in enumerate(bvals):
            G = G_from_b(b, delta, Delta)
            # Phase for each walker, averaged over 3 axes (isotropic)
            cos_sum = 0.0
            for ax in range(3):
                phases = GAMMA_RAD * G * dM_si[:, di, ax]
                cos_sum += np.mean(np.cos(phases))
            signals[di, bi] = cos_sum / 3.0

    return {
        'b_values': bvals,
        'Deltas': Deltas,
        'signals': signals,
    }


def compute_adc(b_values, signal_1d, b_adc=1000.0):
    """ADC [μm²/ms] from two-point slope at b_adc [s/mm²]."""
    idx = np.argmin(np.abs(b_values - b_adc))
    S = signal_1d[idx]
    b_int = b_values[idx] / 1e6   # s/mm² → ms/μm²
    if S <= 0 or b_int <= 0:
        return 0.0
    return -np.log(S) / b_int


def signals_to_flat(result: dict) -> np.ndarray:
    """Flatten signals dict to a 1-D feature vector for library matching.

    Returns shape (n_deltas × n_b,) — concatenated across Δ values.
    """
    return result['signals'].ravel()

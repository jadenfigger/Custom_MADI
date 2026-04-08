"""
DWI signal computation for multi-Δ SDE acquisition.

Given the new `WalkResult.dM_per_axis` (three independent per-axis dM
arrays from `run_simulation`), compute S(b)/S₀ for each Δ and b-value.

The phase for walker k in the axis-ax ensemble at gradient strength G is
    φ_k = γ · G · dM_k,ax
where dM_k,ax is the scalar first-moment difference for that walker along
the ax direction (stored in dM_per_axis[ax]).  The axis-ax signal is
    S_ax(b) = mean_k cos(φ_k)
and the powder-averaged signal is (S_x + S_y + S_z) / 3.
"""

from __future__ import annotations

import numpy as np
from .config import SimConfig, GAMMA_RAD, BVALS_UNIQUE
from .walker_gpu import WalkResult


def G_from_b(b_s_mm2: float, delta_ms: float, Delta_ms: float) -> float:
    """Gradient strength [T/m] for given b [s/mm²], δ [ms], Δ [ms]."""
    b_si = b_s_mm2 * 1e6        # s/m²
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
    """Compute S(b)/S₀ for each Δ and b-value, powder-averaged over the
    three statistically independent axis ensembles.

    Parameters
    ----------
    wr : WalkResult
        Must have `dM_per_axis` list of three arrays, each shape
        (N_ax, n_deltas) in units of μm·ms.
    cfg : SimConfig
    b_values_s_mm2 : array, optional
        b-values [s/mm²].  Default: BVALS_UNIQUE = [1000, 2500, 4000, 6000].

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

    # Convert per-axis dM from [μm·ms] to SI [m·s]
    dM_per_axis_si = [arr * 1e-6 * 1e-3 for arr in wr.dM_per_axis]

    signals = np.zeros((n_deltas, n_b))

    for di, Delta in enumerate(Deltas):
        for bi, b in enumerate(bvals):
            G = G_from_b(b, cfg.delta, Delta)

            # Independent axis-x, axis-y, axis-z averages
            s_x = np.mean(np.cos(GAMMA_RAD * G * dM_per_axis_si[0][:, di]))
            s_y = np.mean(np.cos(GAMMA_RAD * G * dM_per_axis_si[1][:, di]))
            s_z = np.mean(np.cos(GAMMA_RAD * G * dM_per_axis_si[2][:, di]))

            signals[di, bi] = (s_x + s_y + s_z) / 3.0

    return {
        'b_values': bvals,
        'Deltas': Deltas,
        'signals': signals,
    }


def compute_adc(b_values, signal_1d, b_adc=1000.0):
    """ADC [μm²/ms] from two-point slope at b_adc [s/mm²]."""
    idx = np.argmin(np.abs(b_values - b_adc))
    S = signal_1d[idx]
    b_int = b_values[idx] / 1e6
    if S <= 0 or b_int <= 0:
        return 0.0
    return -np.log(S) / b_int


def signals_to_flat(result: dict) -> np.ndarray:
    """Flatten signals dict to a 1-D feature vector for library matching.

    Returns shape (n_deltas × n_b,) — concatenated across Δ values.
    """
    return result['signals'].ravel()

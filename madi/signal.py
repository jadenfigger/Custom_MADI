"""
DWI signal computation for the (δ, Δ, b)-universal MADI library.

`walker_gpu.py` is physics-agnostic below the walk: it reduces per-walker
Y(t) to Σcos(phase)/Σsin(phase) for arbitrary "columns" of
`(j_delta, j_Delta, j_sum, phase_coef)`. This module is where physical
(δ,Δ,b) triples become those columns, and where the library's stored S/S₀
block gets assembled.

The phase convention is UNCHANGED from the old fixed-δ pipeline:
    dM(δ,Δ) = Y(δ) + Y(Δ) − Y(Δ+δ)      [μm·ms]  (== old kernel's m1 − m2)
    phase   = γ · G(b,δ,Δ) · dM_si      [rad]     (dM_si = dM · 1e-9)
    S(b;δ,Δ) = ⟨cos(phase)⟩             (REAL part — never |S|, see below)

No interpolation is performed anywhere in this module: `build_columns`
builds the exact set of (δ,Δ,b) columns the caller asks for (by default,
the library's stored grid from `cfg`), and downstream matching picks the
nearest stored column rather than interpolating between them.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .config import SimConfig, GAMMA_RAD, grid_time_index
from .walker_gpu import (run_simulation_reduced, run_simulation_multi_kio_reduced,
                          ReducedResult)


def G_from_b(b_s_mm2: float, delta_ms: float, Delta_ms: float) -> float:
    """Gradient strength [T/m] for given b [s/mm²], δ [ms], Δ [ms]."""
    b_si = b_s_mm2 * 1e6        # s/m²
    d_si = delta_ms * 1e-3       # s
    tD_si = (Delta_ms - delta_ms / 3.0) * 1e-3
    if b_si <= 0 or tD_si <= 0:
        return 0.0
    return np.sqrt(b_si / ((GAMMA_RAD * d_si) ** 2 * tD_si))


# ---------------------------------------------------------------------------
# Column construction
# ---------------------------------------------------------------------------

@dataclass
class ColumnGrid:
    """Flattened (δ,Δ,b) column arrays consumed by
    `walker_gpu.run_simulation_reduced`. Row-major: column index =
    pair_index * n_b + b_index, matching the S[n_pairs, n_b] reshape."""
    delta_pairs: List[Tuple[float, float]]   # length n_pairs
    b_values:    np.ndarray                   # (n_b,)
    j_delta:     np.ndarray                   # (n_cols,) int32
    j_Delta:     np.ndarray                   # (n_cols,) int32
    j_sum:       np.ndarray                   # (n_cols,) int32
    phase_coef:  np.ndarray                   # (n_cols,) float64 — γ·G·1e-9
    n_pairs:     int
    n_b:         int


def build_columns(
    cfg: Optional[SimConfig] = None,
    delta_pairs: Optional[List[Tuple[float, float]]] = None,
    b_values: Optional[np.ndarray] = None,
) -> ColumnGrid:
    """Build the (δ,Δ,b) column grid.

    Defaults to `cfg`'s stored library grid (`cfg.delta_pairs()` ×
    `cfg.b_values`). Pass explicit `delta_pairs`/`b_values` for a custom
    (e.g. validation-suite) grid — every δ, Δ, Δ+δ must still land exactly
    on the h_ms grid or `grid_time_index` raises.
    """
    if cfg is None:
        cfg = SimConfig()
    pairs = delta_pairs if delta_pairs is not None else cfg.delta_pairs()
    bvals = np.asarray(b_values if b_values is not None else cfg.b_values,
                        dtype=float)
    n_pairs = len(pairs)
    n_b = len(bvals)
    n_cols = n_pairs * n_b

    j_delta = np.empty(n_cols, dtype=np.int32)
    j_Delta = np.empty(n_cols, dtype=np.int32)
    j_sum   = np.empty(n_cols, dtype=np.int32)
    phase_coef = np.empty(n_cols, dtype=np.float64)

    for pi, (delta, Delta) in enumerate(pairs):
        jd = grid_time_index(delta, cfg.h_ms)
        jD = grid_time_index(Delta, cfg.h_ms)
        js = grid_time_index(Delta + delta, cfg.h_ms)
        for bi, b in enumerate(bvals):
            col = pi * n_b + bi
            j_delta[col] = jd
            j_Delta[col] = jD
            j_sum[col] = js
            G = G_from_b(float(b), delta, Delta)
            phase_coef[col] = GAMMA_RAD * G * 1e-9

    return ColumnGrid(delta_pairs=pairs, b_values=bvals,
                       j_delta=j_delta, j_Delta=j_Delta, j_sum=j_sum,
                       phase_coef=phase_coef, n_pairs=n_pairs, n_b=n_b)


# ---------------------------------------------------------------------------
# Orchestration: walk + reduce → S[δ,Δ,b]
# ---------------------------------------------------------------------------

def _assemble(res: ReducedResult, columns: ColumnGrid) -> dict:
    n_eff = res.n_eff
    S = (res.cos_sum / n_eff).reshape(columns.n_pairs, columns.n_b)
    S_imag = (res.sin_sum / n_eff).reshape(columns.n_pairs, columns.n_b)
    return {
        'delta_pairs': columns.delta_pairs,
        'b_values':    columns.b_values,
        'S':           S,
        'S_imag':      S_imag,
        'n_eff':       n_eff,
        'n_escaped':   res.n_escaped,
    }


def compute_signals(
    rho: float, V: float, kio: float,
    cfg: Optional[SimConfig] = None,
    columns: Optional[ColumnGrid] = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run one (ρ,V,kio) library entry's walk and reduce it to S[δ,Δ,b].

    Returns dict with keys: 'delta_pairs', 'b_values', 'S' (n_pairs, n_b),
    'S_imag' (imaginary-part check — should be ~0 everywhere), 'n_eff',
    'n_escaped'.
    """
    if cfg is None:
        cfg = SimConfig()
    if columns is None:
        columns = build_columns(cfg)

    res = run_simulation_reduced(
        rho, V, kio,
        columns.j_delta, columns.j_Delta, columns.j_sum, columns.phase_coef,
        cfg, seed=seed, verbose=verbose)
    return _assemble(res, columns)


def compute_signals_multi_kio(
    rho: float, V: float, kios: list,
    cfg: Optional[SimConfig] = None,
    columns: Optional[ColumnGrid] = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Ensemble-reuse across a kio sweep — see
    `walker_gpu.run_simulation_multi_kio_reduced`. Returns dict[kio] -> the
    same per-entry dict `compute_signals` returns."""
    if cfg is None:
        cfg = SimConfig()
    if columns is None:
        columns = build_columns(cfg)

    results = run_simulation_multi_kio_reduced(
        rho, V, kios,
        columns.j_delta, columns.j_Delta, columns.j_sum, columns.phase_coef,
        cfg, seed=seed, verbose=verbose)
    return {kio: _assemble(res, columns) for kio, res in results.items()}


def compute_adc(b_values, signal_1d, b_adc=1000.0):
    """ADC [μm²/ms] from two-point slope at b_adc [s/mm²]."""
    idx = np.argmin(np.abs(np.asarray(b_values) - b_adc))
    S = signal_1d[idx]
    b_int = b_values[idx] / 1e6
    if S <= 0 or b_int <= 0:
        return 0.0
    return -np.log(S) / b_int


def signals_to_flat(result: dict) -> np.ndarray:
    """Flatten the S[n_pairs, n_b] block to a 1-D vector for library storage
    (row-major: pair-major then b)."""
    return result['S'].ravel()

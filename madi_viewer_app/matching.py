"""Live top-N matching for the viewer.

Unlike the batch matcher in ``madi/library.py``, this returns *all* top-N
entries with both linear- and log-space residuals computed every time
(cheap: one extra inner product), so the user can flip the sort at will.
Also supports fitted-S0 mode (returns the optimal S0 per candidate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import ProfileData


@dataclass
class MatchRow:
    rank:    int
    idx:     int       # row in the *candidate* matrix
    kio:     float
    rho:     float
    V:       float
    vi:      float
    sse_lin: float
    sse_log: float
    pred:    np.ndarray
    s0_fit:  Optional[float] = None   # only populated when fit_s0=True


def _vi_rho_mask(kios, rhos, Vs, vi_min, vi_max, rho_max):
    vi = (rhos / 1e9) * (Vs * 1e3)
    m = (vi >= vi_min) & (vi <= vi_max)
    if rho_max is not None:
        m &= rhos <= float(rho_max)
    return m, vi


def find_top_matches(
    pd: ProfileData,
    vx: int, vy: int, sl: int, axis: int,
    top_n: int = 10,
    sort_by: str = "lin",
    vi_min: float = 0.5,
    vi_max: float = 0.95,
    rho_max: Optional[float] = None,
    s_floor: float = 1e-3,
    fit_s0: bool = False,
) -> tuple[Optional[list[MatchRow]], Optional[np.ndarray]]:
    """Return (top matches, measured signal vector) for a voxel.

    sort_by ∈ {"lin", "log", "kio", "rho", "V", "vi"}.
    """
    measured = pd.signal_at(vx, vy, sl, axis)
    if measured is None or np.all(measured < 1e-10):
        return None, measured

    if not pd.ensure_lib_sub(pd.fit_deltas or pd.profile.delta_ms_list()):
        return None, measured

    bundle = pd.lib_bundle
    kios, rhos, Vs = bundle["kios"], bundle["rhos"], bundle["Vs"]
    lib_sub = pd.lib_sub

    mask, vi = _vi_rho_mask(kios, rhos, Vs, vi_min, vi_max, rho_max)
    if not mask.any():
        return None, measured

    # -------------------------------------------------------------
    # Build the vector the batch matcher would actually see:
    #   fit_s0=False → S/S0 normalised ratios (same units as library)
    #   fit_s0=True  → raw per-shell means (in scanner intensity)
    # This is the vector used for *ranking* and for computing s0_fit.
    # `measured` returned to the caller stays normalised so plotting
    # defaults (S/S0) keep their semantics; the plot's "raw-signal"
    # mode rescales by s0_measured / s0_fit on the fly.
    # -------------------------------------------------------------
    if fit_s0:
        raw = pd.raw_signal_at(vx, vy, sl, axis)
        if raw is None:
            return None, measured
        M_match = raw.astype(np.float64)
    else:
        M_match = measured.astype(np.float64)

    # Linear-space SSE (either normalised ratios, or with free S0)
    dists_lin = np.full(len(lib_sub), np.inf, dtype=np.float64)
    s0_cand = np.full(len(lib_sub), np.nan, dtype=np.float64)

    if fit_s0:
        R = lib_sub[mask].astype(np.float64)
        rr = np.maximum(np.sum(R * R, axis=1), 1e-30)
        mr = R @ M_match
        s0 = mr / rr
        resid = float(np.sum(M_match * M_match)) - (mr ** 2) / rr
        resid = np.where(s0 > 0, resid, np.inf)
        dists_lin[mask] = resid
        s0_cand[mask] = s0
    else:
        diff = lib_sub[mask] - M_match[None, :]
        dists_lin[mask] = np.sum(diff * diff, axis=1)

    # Log-space SSE — always computed (cheap) so table can show both.
    # In fit-S0 mode we compare log(M_raw) vs log(s0 * R); otherwise
    # log(M/S0) vs log(R) (both already in [s_floor, 1]).
    dists_log = np.full(len(lib_sub), np.inf, dtype=np.float64)
    if fit_s0:
        # Use the per-candidate scaled library curve.
        Rsel = lib_sub[mask].astype(np.float64)
        s0_sel = s0_cand[mask].reshape(-1, 1)
        floor = max(s_floor, 1e-6)
        scaled = np.clip(Rsel * s0_sel, floor, None)
        meas_floor = float(np.max(M_match)) * s_floor
        meas_floor = max(meas_floor, 1e-6)
        log_meas = np.log(np.clip(M_match, meas_floor, None))
        log_scal = np.log(scaled)
        dlog = log_scal - log_meas[None, :]
        dists_log[mask] = np.sum(dlog * dlog, axis=1)
    else:
        meas_log = np.log(np.clip(M_match, s_floor, 1.0))
        lib_log = np.log(np.clip(lib_sub[mask], s_floor, 1.0))
        diff_log = lib_log - meas_log[None, :]
        dists_log[mask] = np.sum(diff_log * diff_log, axis=1)

    # Pick ranking key
    if sort_by == "log":
        key = dists_log
    elif sort_by == "kio":
        key = np.where(mask, kios, np.inf)
    elif sort_by == "rho":
        key = np.where(mask, rhos, np.inf)
    elif sort_by == "V":
        key = np.where(mask, Vs, np.inf)
    elif sort_by == "vi":
        key = np.where(mask, vi, np.inf)
    else:
        key = dists_lin

    finite_mask = np.isfinite(key)
    if not finite_mask.any():
        return None, measured
    n = int(min(top_n, int(finite_mask.sum())))
    order = np.argsort(key)[:n]

    # Always populate s0_fit (L2-optimal scalar multiplier for this match
    # applied to the normalised library curve). When fit_s0=True this is the
    # same number used by the ranker and lives in physical scanner units
    # (matching s0_fit_map). Otherwise it's an L2 projection of the already-
    # normalised measured vector onto the library curve (≈1 for a good fit)
    # and is shown mainly as a sanity diagnostic.
    rows = []
    for rank, idx in enumerate(order):
        R = lib_sub[idx].astype(np.float64)
        rr_i = max(float(np.sum(R * R)), 1e-30)
        s0_i = float(R @ M_match) / rr_i
        rows.append(MatchRow(
            rank=rank + 1,
            idx=int(idx),
            kio=float(kios[idx]),
            rho=float(rhos[idx]),
            V=float(Vs[idx]),
            vi=float(vi[idx]),
            sse_lin=float(dists_lin[idx]),
            sse_log=float(dists_log[idx]),
            pred=lib_sub[idx].copy(),
            s0_fit=s0_i,
        ))
    return rows, measured

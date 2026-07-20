"""
Cramer-Rao / Fisher Information analysis of MADI parameter identifiability.

WHY THIS EXISTS
---------------
The MADI matcher picks (or blends) library entries by comparing a measured
signal against simulated signals S(kio, rho, V).  Some regions of
(kio, rho, V) space are intrinsically hard to pin down from a given
acquisition -- e.g. if dS/drho and dS/dV point in nearly the same direction
in measurement space, no amount of matching cleverness can separate rho
from V; the acquisition itself doesn't carry the information.  The Fisher
Information Matrix (FIM) makes this precise and lets us compare candidate
acquisitions (different (Delta, b) subsets) without re-running the fitter.

CRITICAL CAVEAT -- READ THIS BEFORE TRUSTING A NUMBER
-------------------------------------------------------
The Cramer-Rao bound assumes (1) a continuous, differentiable forward
model and (2) an unbiased estimator.  MADI's library is a DISCRETE grid of
Monte-Carlo-simulated signals, and its matcher (nearest-neighbour / soft
nearest-neighbour) is neither continuous nor unbiased.  Two consequences:

  1. Derivatives here are FINITE DIFFERENCES across neighbouring library
     grid points, not exact analytic derivatives.  The library grid is
     IRREGULAR (non-uniform rho/V spacing, valid only where
     rho*V <= vi_max), so every derivative is computed with the actual,
     generally unequal, forward/backward spacing -- see
     ``_axis_derivative`` below.  Do not assume a regular grid anywhere in
     this module.
  2. Because of (1) and (2), *absolute* CRLB values are only approximate.
     The valid use of this module is RELATIVE: comparing one acquisition
     against another, and finding WHERE in (kio, rho, V) space a given
     acquisition is least informative.  Do not report an absolute CRLB as
     "the variance you will get" -- report it as "the CRLB of acquisition
     A relative to acquisition B".

A future "universal"/spectral library (storing an underlying diffusion
spectrum per entry from which S(b) can be reconstructed analytically for
any (Delta, b)) would allow exact autodiff CRLBs. This module is the
finite-difference stand-in until that exists.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .library import LibraryEntry, _pair_indices


PARAM_NAMES = ("kio", "rho", "V")


# ---------------------------------------------------------------------------
# Rounding keys -- must match madi.library._entry_key's rounding so that
# "same rho" / "same V" / "same kio" grouping is robust to float noise.
# ---------------------------------------------------------------------------

def _rkio(k: float) -> float: return round(k, 4)
def _rrho(r: float) -> float: return round(r, 1)
def _rV(v: float) -> float:   return round(v, 6)


# ---------------------------------------------------------------------------
# Finite-difference derivatives on the irregular (kio, rho, V) grid
# ---------------------------------------------------------------------------

@dataclass
class EntryDerivatives:
    """Finite-difference derivative vectors for one library entry.

    Each ``d_*`` is a full-length vector (n_deltas * n_b), or ``None`` if
    the entry is an isolated point along that axis (no neighbour at all --
    e.g. only one kio value present for that (rho, V), which shouldn't
    happen but is guarded against). ``*_central`` is True iff a proper
    central difference was used (neighbours on both sides); False means a
    one-sided (grid-edge) difference was used.
    """
    idx: int
    kio: float
    rho: float
    V: float
    d_kio: Optional[np.ndarray]
    d_rho: Optional[np.ndarray]
    d_V: Optional[np.ndarray]
    kio_central: bool
    rho_central: bool
    V_central: bool


def _axis_derivative(sorted_group: List[Tuple[float, np.ndarray]], pos: int):
    """Finite-difference derivative at position ``pos`` of a 1-D axis scan.

    Parameters
    ----------
    sorted_group : list of (param_value, vector), sorted ascending by
        param_value.  These are the OTHER entries that vary only along
        this one axis (same value of the other two parameters).
    pos : index of the entry itself within ``sorted_group``.

    Returns
    -------
    (deriv, is_central) where deriv is None if fewer than 2 points exist
    (can't take any derivative -- isolated point).

    Stencil
    -------
    Central (neighbours on both sides at values val_minus < val0 < val_plus,
    spacings h_minus = val0 - val_minus, h_plus = val_plus - val0):

        dS/dtheta ~= (S(val_plus) - S(val_minus)) / (h_minus + h_plus)

    This is the simple non-uniform-spacing central difference requested --
    NOT the higher-order weighted-stencil version -- so it is only exact to
    O(h) when h_minus != h_plus (it is O(h^2) accurate only for equal
    spacing). Given the MC simulation noise floor in each library entry,
    this simpler and more transparent stencil is preferred over a
    higher-order fit.

    One-sided (grid edge, only one neighbour available):

        dS/dtheta ~= (S(neighbour) - S(val0)) / (neighbour - val0)
    """
    n = len(sorted_group)
    if n < 2:
        return None, False

    val0, vec0 = sorted_group[pos]

    if 0 < pos < n - 1:
        val_m, vec_m = sorted_group[pos - 1]
        val_p, vec_p = sorted_group[pos + 1]
        h_m = val0 - val_m
        h_p = val_p - val0
        deriv = (vec_p - vec_m) / (h_m + h_p)
        return deriv, True
    elif pos == 0:
        val_p, vec_p = sorted_group[pos + 1]
        h_p = val_p - val0
        deriv = (vec_p - vec0) / h_p
        return deriv, False
    else:  # pos == n - 1
        val_m, vec_m = sorted_group[pos - 1]
        h_m = val0 - val_m
        deriv = (vec0 - vec_m) / h_m
        return deriv, False


def _build_axis_groups(library: List[LibraryEntry]):
    """Group entry indices by the OTHER two parameters, for each axis.

    Because the library is built as a full kio cross-product on top of a
    (possibly irregular, vi-filtered) (rho, V) grid, kio-neighbours are
    always the full set of kios sharing that (rho, V); rho-neighbours are
    the set of rho values sharing that (kio, V) that also survived the vi
    filter; V-neighbours analogously. No axis is assumed regular.
    """
    kio_groups: Dict[Tuple[float, float], List[Tuple[float, int]]] = defaultdict(list)
    rho_groups: Dict[Tuple[float, float], List[Tuple[float, int]]] = defaultdict(list)
    V_groups:   Dict[Tuple[float, float], List[Tuple[float, int]]] = defaultdict(list)

    for i, e in enumerate(library):
        k, r, v = _rkio(e.kio), _rrho(e.rho), _rV(e.V)
        kio_groups[(r, v)].append((e.kio, i))
        rho_groups[(k, v)].append((e.rho, i))
        V_groups[(k, r)].append((e.V, i))

    for grp in (kio_groups, rho_groups, V_groups):
        for key in grp:
            grp[key].sort(key=lambda t: t[0])

    return kio_groups, rho_groups, V_groups


def compute_finite_diff_derivatives(
    library: List[LibraryEntry],
) -> List[EntryDerivatives]:
    """Compute dS/dkio, dS/drho, dS/dV (full-length vectors) for every entry.

    Returns a list aligned 1:1 with ``library`` (same indices).
    """
    kio_groups, rho_groups, V_groups = _build_axis_groups(library)

    # Pre-fetch each group's vectors once (vectors, not just indices), keyed
    # by the same group key, to avoid repeated indexing into `library`.
    def _materialize(groups):
        out = {}
        for key, pairs in groups.items():
            out[key] = [(val, library[i].vector) for val, i in pairs]
        return out

    kio_vecs = _materialize(kio_groups)
    rho_vecs = _materialize(rho_groups)
    V_vecs   = _materialize(V_groups)

    # Position of each entry within its own group (by identity of index).
    kio_pos: Dict[int, Tuple[Tuple[float, float], int]] = {}
    for key, pairs in kio_groups.items():
        for pos, (_, i) in enumerate(pairs):
            kio_pos[i] = (key, pos)
    rho_pos: Dict[int, Tuple[Tuple[float, float], int]] = {}
    for key, pairs in rho_groups.items():
        for pos, (_, i) in enumerate(pairs):
            rho_pos[i] = (key, pos)
    V_pos: Dict[int, Tuple[Tuple[float, float], int]] = {}
    for key, pairs in V_groups.items():
        for pos, (_, i) in enumerate(pairs):
            V_pos[i] = (key, pos)

    results = []
    for i, e in enumerate(library):
        k_key, k_pos = kio_pos[i]
        r_key, r_pos = rho_pos[i]
        v_key, v_pos = V_pos[i]

        d_kio, kio_central = _axis_derivative(kio_vecs[k_key], k_pos)
        d_rho, rho_central = _axis_derivative(rho_vecs[r_key], r_pos)
        d_V,   V_central   = _axis_derivative(V_vecs[v_key], v_pos)

        results.append(EntryDerivatives(
            idx=i, kio=e.kio, rho=e.rho, V=e.V,
            d_kio=d_kio, d_rho=d_rho, d_V=d_V,
            kio_central=kio_central, rho_central=rho_central, V_central=V_central,
        ))
    return results


# ---------------------------------------------------------------------------
# Fisher Information Matrix + CRLB diagnostics
# ---------------------------------------------------------------------------

def compute_fim(
    d_kio: np.ndarray, d_rho: np.ndarray, d_V: np.ndarray,
    col_idx: np.ndarray,
    sigma_m,
) -> np.ndarray:
    """3x3 Fisher Information Matrix over (kio, rho, V) for one entry.

        F_jk = sum_f  (dS_j/df) * (dS_k/df) / sigma_m[f]^2

    Parameters
    ----------
    d_kio, d_rho, d_V : full-length (n_deltas*n_b,) derivative vectors.
    col_idx : (n_feat,) int array selecting which (Delta, b) columns of the
        full vector belong to the acquisition being analyzed -- this is
        where "which measurements do I have" enters the calculation.
    sigma_m : float or (n_feat,) array. Per-measurement noise std on the
        normalized S/S0 signal. If an array, sigma_m[f] pairs with
        col_idx[f] (i.e. must already be in the acquisition's column
        order), allowing heteroscedastic (per-shell) noise.

    Returns
    -------
    F : (3, 3) ndarray, order (kio, rho, V).
    """
    D = np.stack([d_kio[col_idx], d_rho[col_idx], d_V[col_idx]], axis=0)  # (3, n_feat)
    sigma_m = np.atleast_1d(np.asarray(sigma_m, dtype=float))
    if sigma_m.size == 1:
        return (D @ D.T) / (sigma_m[0] ** 2)
    if sigma_m.size != D.shape[1]:
        raise ValueError(
            f"sigma_m has {sigma_m.size} entries but acquisition has "
            f"{D.shape[1]} (Delta,b) columns.")
    Dw = D / sigma_m[None, :]
    return Dw @ Dw.T


@dataclass
class FimDiagnostics:
    F: np.ndarray                 # (3,3)
    Finv: np.ndarray               # (3,3), pseudo-inverse if F singular
    crlb: np.ndarray               # (3,) diag(Finv) -- kio, rho, V
    trace_Finv: float
    rhoV_det: float                 # det of 2x2 rho-V submatrix of F
    rhoV_cond: float                # condition number of rho-V submatrix of F
    rhoV_corr: float                # corr(dS/drho, dS/dV), unweighted, in [-1,1]
    eigvals: np.ndarray             # (3,) ascending
    eigvecs: np.ndarray             # (3,3) columns = eigenvectors, same order
    dominant_param_idx: int         # index (0,1,2) with largest |component|
                                     # in the smallest-eigenvalue eigenvector
    min_eig_negative: bool          # True if smallest eigenvalue is
                                     # significantly negative (FD error flag)


def crlb_diagnostics(
    F: np.ndarray,
    d_rho_sub: np.ndarray,
    d_V_sub: np.ndarray,
    psd_atol_rel: float = 1e-6,
) -> FimDiagnostics:
    """Invert the FIM and compute the rho-V conditioning diagnostics.

    ``d_rho_sub``/``d_V_sub`` are the ALREADY-column-subset (acquisition-
    restricted) derivative vectors -- used only for the plain (unweighted)
    correlation coefficient, per the definition
        corr = (dS/drho . dS/dV) / (||dS/drho|| ||dS/dV||)
    which is independent of sigma_m by construction (unlike F itself, which
    is sigma-weighted and, under heteroscedastic sigma_m, would give a
    different, sigma-weighted "effective" correlation).
    """
    eigvals, eigvecs = np.linalg.eigh(F)  # ascending eigenvalues, symmetric F

    scale = max(abs(eigvals[-1]), 1e-30)
    min_eig_negative = eigvals[0] < -psd_atol_rel * scale

    try:
        Finv = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        Finv = np.linalg.pinv(F)
    crlb = np.diag(Finv).copy()
    trace_Finv = float(np.trace(Finv))

    sub = F[1:, 1:]
    rhoV_det = float(np.linalg.det(sub))
    sub_eigs = np.linalg.eigvalsh(sub)
    rhoV_cond = float(sub_eigs[-1] / sub_eigs[0]) if sub_eigs[0] > 0 else float("inf")

    denom = np.linalg.norm(d_rho_sub) * np.linalg.norm(d_V_sub)
    rhoV_corr = float(np.dot(d_rho_sub, d_V_sub) / denom) if denom > 0 else 0.0
    rhoV_corr = float(np.clip(rhoV_corr, -1.0, 1.0))

    dominant_param_idx = int(np.argmax(np.abs(eigvecs[:, 0])))

    return FimDiagnostics(
        F=F, Finv=Finv, crlb=crlb, trace_Finv=trace_Finv,
        rhoV_det=rhoV_det, rhoV_cond=rhoV_cond, rhoV_corr=rhoV_corr,
        eigvals=eigvals, eigvecs=eigvecs,
        dominant_param_idx=dominant_param_idx,
        min_eig_negative=min_eig_negative,
    )


# ---------------------------------------------------------------------------
# Acquisition column selection
# ---------------------------------------------------------------------------

def resolve_acquisition_columns(
    fit_pairs: Sequence[Tuple[float, float]],
    lib_deltas: Sequence[float],
    lib_b_values: Sequence[float],
    n_b: int,
) -> np.ndarray:
    """Column indices into a flat library vector for the requested (Delta, b)
    pairs -- thin wrapper around ``madi.library._pair_indices`` so that
    adding a new Delta later is a one-line change to ``fit_pairs``.
    """
    return _pair_indices(fit_pairs, lib_deltas, lib_b_values, n_b)


# ---------------------------------------------------------------------------
# Full per-library-entry analysis
# ---------------------------------------------------------------------------

@dataclass
class IdentifiabilityResult:
    rows: List[dict]
    summary: dict
    n_kio_central: int
    n_kio_edge: int
    n_rho_central: int
    n_rho_edge: int
    n_V_central: int
    n_V_edge: int
    n_skipped_isolated: int
    n_nonpsd: int


def analyze_library(
    library: List[LibraryEntry],
    lib_deltas: Sequence[float],
    lib_b_values: Sequence[float],
    n_b: int,
    fit_pairs: Sequence[Tuple[float, float]],
    sigma_m,
    degenerate_corr_threshold: float = 0.9,
    derivatives: Optional[List[EntryDerivatives]] = None,
) -> IdentifiabilityResult:
    """Run the full FIM/CRLB analysis over every library entry for one
    acquisition (a choice of (Delta, b) columns, ``fit_pairs``).

    Parameters
    ----------
    sigma_m : float or (len(fit_pairs),) array-like.
    derivatives : precomputed ``compute_finite_diff_derivatives(library)``
        output, if the caller wants to reuse it across multiple acquisition
        comparisons (avoids recomputing the same finite differences).
    """
    col_idx = resolve_acquisition_columns(fit_pairs, lib_deltas, lib_b_values, n_b)
    sigma_arr = np.atleast_1d(np.asarray(sigma_m, dtype=float))
    if sigma_arr.size not in (1, len(fit_pairs)):
        raise ValueError(
            f"sigma_m must be scalar or length {len(fit_pairs)} "
            f"(one per (Delta,b) in fit_pairs); got length {sigma_arr.size}.")

    if derivatives is None:
        derivatives = compute_finite_diff_derivatives(library)

    rows = []
    n_kio_central = n_kio_edge = 0
    n_rho_central = n_rho_edge = 0
    n_V_central = n_V_edge = 0
    n_skipped = 0
    n_nonpsd = 0

    for d in derivatives:
        if d.d_kio is None or d.d_rho is None or d.d_V is None:
            n_skipped += 1
            continue

        if d.kio_central: n_kio_central += 1
        else:             n_kio_edge += 1
        if d.rho_central: n_rho_central += 1
        else:             n_rho_edge += 1
        if d.V_central:   n_V_central += 1
        else:             n_V_edge += 1

        F = compute_fim(d.d_kio, d.d_rho, d.d_V, col_idx, sigma_arr)
        diag = crlb_diagnostics(F, d.d_rho[col_idx], d.d_V[col_idx])

        if diag.min_eig_negative:
            n_nonpsd += 1
            warnings.warn(
                f"Entry idx={d.idx} (kio={d.kio}, rho={d.rho}, V={d.V}): "
                f"FIM has a significantly negative eigenvalue "
                f"({diag.eigvals[0]:.3e}) -- likely a finite-difference "
                f"artifact (check grid-edge / noisy neighbour entries).")

        rows.append(dict(
            kio=d.kio, rho=d.rho, V=d.V,
            CRLB_kio=diag.crlb[0], CRLB_rho=diag.crlb[1], CRLB_V=diag.crlb[2],
            trace_Finv=diag.trace_Finv,
            rhoV_correlation=diag.rhoV_corr,
            rhoV_condition_number=diag.rhoV_cond,
            rhoV_det=diag.rhoV_det,
            smallest_eigenvalue=float(diag.eigvals[0]),
            dominant_param_of_least_identifiable_direction=PARAM_NAMES[diag.dominant_param_idx],
            kio_central=d.kio_central, rho_central=d.rho_central, V_central=d.V_central,
            psd_ok=not diag.min_eig_negative,
        ))

    if not rows:
        raise RuntimeError(
            "No library entries yielded a full 3x3 FIM (every entry was an "
            "isolated grid point along at least one axis). Check that the "
            "library actually has kio/rho/V neighbours.")

    crlb_kio = np.array([r["CRLB_kio"] for r in rows])
    crlb_rho = np.array([r["CRLB_rho"] for r in rows])
    crlb_V   = np.array([r["CRLB_V"] for r in rows])
    trace_Finv = np.array([r["trace_Finv"] for r in rows])
    corr = np.array([r["rhoV_correlation"] for r in rows])

    def _med_iqr(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return dict(median=float("nan"), q25=float("nan"), q75=float("nan"))
        return dict(median=float(np.median(x)),
                    q25=float(np.percentile(x, 25)),
                    q75=float(np.percentile(x, 75)))

    summary = dict(
        n_entries_total=len(library),
        n_entries_analyzed=len(rows),
        n_skipped_isolated=n_skipped,
        n_nonpsd=n_nonpsd,
        sigma_m=(float(sigma_arr[0]) if sigma_arr.size == 1 else sigma_arr.tolist()),
        fit_pairs=[list(p) for p in fit_pairs],
        CRLB_kio=_med_iqr(crlb_kio),
        CRLB_rho=_med_iqr(crlb_rho),
        CRLB_V=_med_iqr(crlb_V),
        trace_Finv=_med_iqr(trace_Finv),
        rhoV_correlation_median=float(np.median(corr)),
        degenerate_fraction=float(np.mean(np.abs(corr) > degenerate_corr_threshold)),
        degenerate_corr_threshold=degenerate_corr_threshold,
        derivative_stencil_counts=dict(
            kio_central=n_kio_central, kio_edge=n_kio_edge,
            rho_central=n_rho_central, rho_edge=n_rho_edge,
            V_central=n_V_central, V_edge=n_V_edge,
        ),
    )

    return IdentifiabilityResult(
        rows=rows, summary=summary,
        n_kio_central=n_kio_central, n_kio_edge=n_kio_edge,
        n_rho_central=n_rho_central, n_rho_edge=n_rho_edge,
        n_V_central=n_V_central, n_V_edge=n_V_edge,
        n_skipped_isolated=n_skipped, n_nonpsd=n_nonpsd,
    )


# ---------------------------------------------------------------------------
# Gaussian (mono-exponential ADC) sanity check
# ---------------------------------------------------------------------------

def gaussian_crlb_sanity_check(
    b_values_s_mm2: np.ndarray,
    adc_um2_ms: float,
    sigma_m: float,
) -> dict:
    """Cross-check: CRLB(ADC) for a hypothetical pure mono-exponential
    (Gaussian-diffusion) signal S(b) = exp(-b * ADC), evaluated on the
    given b-grid [s/mm^2] (ADC in the same units as MADI's internal
    b_values_s_mm2 -> convert to um^2/ms consistently, so b*ADC is
    dimensionless: b_internal = b_s_mm2 * 1e-3 [ms/um^2], ADC [um^2/ms]).

    This does NOT touch the MADI library at all -- it's a fully analytic,
    independent sanity check that single-shell CRLB(ADC) behaves the way
    the literature says it should (optimal b ~ 1/ADC; see Farooq et al.
    2026, Mukherjee et al.): for a single b-shell, the per-b Fisher
    information for ADC is
        I(b) = (b^2 * S(b)^2) / sigma_m^2 ,  S(b) = exp(-b*ADC)
    which is maximized (equivalently CRLB(ADC) = 1/I(b) minimized) at
    b* = 1/ADC (found by d/db [b^2 exp(-2 b ADC)] = 0).

    Returns
    -------
    dict with the b-grid, per-b CRLB(ADC), the grid's best b, and the
    theoretical optimum 1/ADC, for printing/comparison.
    """
    b_internal = np.asarray(b_values_s_mm2, dtype=float) * 1e-3  # s/mm2 -> ms/um2
    S = np.exp(-b_internal * adc_um2_ms)
    dS_dADC = -b_internal * S
    fisher_info = (dS_dADC ** 2) / (sigma_m ** 2)
    crlb_adc = np.where(fisher_info > 0, 1.0 / fisher_info, np.inf)

    best_idx = int(np.argmin(crlb_adc))
    b_star_theory = 1.0 / adc_um2_ms   # ms/um^2 units
    b_star_theory_s_mm2 = b_star_theory * 1e3

    return dict(
        b_values_s_mm2=b_values_s_mm2,
        crlb_adc=crlb_adc,
        best_b_on_grid_s_mm2=float(b_values_s_mm2[best_idx]),
        theoretical_optimal_b_s_mm2=float(b_star_theory_s_mm2),
        adc_um2_ms=adc_um2_ms,
    )

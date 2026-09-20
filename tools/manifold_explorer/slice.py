"""Slice the model manifold: which entries still agree on the measured columns.

The geometry, in plain terms
---------------------------
Every library entry is one point in a 31,125-dimensional prediction space.
Pick a few columns and you are projecting that space down onto a few axes.
"Slicing" means: take a reference point, and keep the entries whose projection
lands within a noise-sized ball of the reference's projection.  The survivors
are the entries a measurement at those columns could not tell apart.

    chi2(entry) = sum over measured columns of ((s_entry - s_ref) / sigma)^2

Two things read off a slice:

* How wide the survivors are in (rho, V, k_io) -- that is the identifiability
  question.  A long thin smear in (rho, V) is the known degeneracy.
* How wide they are on columns you did NOT measure -- that is the prediction
  question.  Narrow means the unmeasured column is implied by the measured
  ones and adds nothing.

Conventions are taken from the project's fitters, not invented here:

* Library vectors are already S/S0 (the b=0 column of every (delta, Delta)
  pair is exactly 1.0 by construction), which is exactly what
  ``madi.library.match_voxels_batch`` expects as input.  No renormalization.
* Fixed-S0 chi2 is the exponent of the ``bayes`` fitter's Gaussian weight,
  ``w ~ exp(-||m - s||^2 / (2 sigma_m^2))`` (docs/fitting_methods.md), so a
  slice at threshold T is exactly the set of entries whose Bayes weight
  relative to the reference exceeds exp(-T/2).
* Free-S0 chi2 mirrors ``madi.library.match_voxels_batch_fits0``: profile out
  the amplitude analytically, then divide the residual by that entry's own
  fitted amplitude squared so sigma stays on the S/S0 scale (the same
  normalization the ``--fit-s0`` bayes path applies).
* The candidate filter is ``madi.library.candidate_selection_mask`` itself,
  called on lightweight label-only entries, so the explorer's v_i band /
  rho_max / free-water rules cannot drift from the fitters'.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from madi.library import LibraryEntry, candidate_selection_mask

from .columns import N_ENSEMBLES, LibraryLabels

# The fitters' placeholder residual noise on the normalized signal when
# nothing better is known (docs/fitting_methods.md, sigma_m rule 4).
DEFAULT_SIGMA = 0.02


def candidate_mask(labels: LibraryLabels, vi_min: float = 0.40, vi_max: float = 0.99,
                   rho_max: float | None = None,
                   include_free_water: bool = False) -> np.ndarray:
    """Which entries are eligible, using the fitters' own filter definition.

    Builds label-only ``LibraryEntry`` objects (no signal vectors, so nothing
    large is read) and hands them to ``madi.library.candidate_selection_mask``.
    """
    empty = np.empty(0)
    entries = [
        LibraryEntry(kio=float(k), rho=float(r), V=float(v), vector=empty,
                     vi=float(vi), is_free_water=bool(free))
        for k, r, v, vi, free in zip(labels.kios, labels.rhos, labels.Vs,
                                     labels.vis, labels.is_free_water)
    ]
    return candidate_selection_mask(entries, vi_min, vi_max, rho_max,
                                    include_free_water=include_free_water)


def column_sigma(sigma_measurement: float, variance_block: np.ndarray | None = None,
                 n_ensembles: int = N_ENSEMBLES) -> np.ndarray | float:
    """Per-column noise scale, optionally with the library's own MC error.

    ``sigma_measurement`` is the measurement noise the user sets, in S/S0
    units.  The library value at a column is itself a Monte-Carlo mean whose
    standard error is ``sqrt(signal_variance / n_ensembles)`` -- the contract
    stated in the artifact's build metadata.  Adding the two in quadrature
    stops the slice from being narrower than the library's own precision.
    """
    if variance_block is None:
        return float(sigma_measurement)
    monte_carlo = np.sqrt(np.asarray(variance_block, dtype=float) / float(n_ensembles))
    return np.sqrt(float(sigma_measurement) ** 2 + monte_carlo ** 2)


def chi2_fixed_s0(block: np.ndarray, reference: np.ndarray,
                  sigma: np.ndarray | float) -> np.ndarray:
    """Sum of squared standardized residuals, amplitude fixed at 1.

    ``block`` is (n_entries, n_columns) of S/S0, ``reference`` is (n_columns,).
    This is the exponent of the bayes fitter's weight, times 2.
    """
    residual = (block - reference[None, :]) / sigma
    return np.einsum("ij,ij->i", residual, residual)


def chi2_free_s0(block: np.ndarray, reference: np.ndarray,
                 sigma: np.ndarray | float) -> np.ndarray:
    """Same, with a free per-entry amplitude profiled out analytically.

    Mirrors ``madi.library.match_voxels_batch_fits0``: for reference m and
    candidate curve s, the least-squares amplitude and residual are

        a*   = (m . s) / (s . s)
        ||m - a* s||^2 = ||m||^2 - (m . s)^2 / (s . s)

    and, as in the ``--fit-s0`` bayes path, the residual is divided by a*^2 so
    that ``sigma`` keeps its meaning on the S/S0 scale.  Entries that would
    need a negative amplitude are rejected outright (chi2 = inf), exactly as
    that matcher does.

    Weighting: with per-column sigma the inner products are taken on the
    sigma-scaled vectors, which is the same estimator with a diagonal metric.
    """
    scaled_block = block / sigma
    scaled_reference = np.asarray(reference, dtype=float) / sigma
    if scaled_reference.ndim == 2:
        # A per-entry sigma makes the reference entry-dependent too.
        ss = np.einsum("ij,ij->i", scaled_block, scaled_block)
        ms = np.einsum("ij,ij->i", scaled_reference, scaled_block)
        mm = np.einsum("ij,ij->i", scaled_reference, scaled_reference)
    else:
        ss = np.einsum("ij,ij->i", scaled_block, scaled_block)
        ms = scaled_block @ scaled_reference
        mm = float(scaled_reference @ scaled_reference)
    ss = np.maximum(ss, 1e-300)
    amplitude = ms / ss
    residual = mm - (ms ** 2) / ss
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = residual / np.maximum(amplitude, 1e-300) ** 2
    return np.where(amplitude > 0.0, normalized, np.inf)


@dataclass
class SliceResult:
    """Survivors of one slice, plus the widths that answer the question."""

    reference_signal: np.ndarray          # (n_measured,)
    measured_columns: np.ndarray          # (n_measured,)
    chi2: np.ndarray                      # (n_eligible,) chi2 of each candidate
    survivors: np.ndarray                 # (n_eligible,) bool
    eligible_rows: np.ndarray             # (n_eligible,) index into the library
    threshold: float
    sigma_measurement: float
    s0_mode: str
    parameter_stats: dict = field(default_factory=dict)
    display_stats: dict = field(default_factory=dict)

    @property
    def n_survivors(self) -> int:
        return int(self.survivors.sum())

    @property
    def survivor_rows(self) -> np.ndarray:
        """Library row indices of the surviving entries."""
        return self.eligible_rows[self.survivors]


def _spread(values: np.ndarray, log: bool = False) -> dict:
    """min / max / std of a parameter across survivors, plus a log-ratio width.

    ``log_width`` is max/min, the natural width on a log-spaced axis: 1.0 means
    the survivors sit at a single grid value, 10 means they span a decade.

    Zero and negative values have no log width.  The free-water atom is exactly
    that case -- it carries rho = V = 0 -- so the ratio is taken over the
    strictly positive values only and ``n_nonpositive`` records how many were
    set aside.  Letting one zero turn the whole width into NaN made the width
    curve vanish whenever free water was included.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"),
                "std": float("nan"), "log_width": float("nan"),
                "n_nonpositive": 0}
    positive = finite[finite > 0.0]
    record = {
        "n": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "std": float(finite.std()),
        "log_width": float("nan"),
        "n_nonpositive": int(finite.size - positive.size),
    }
    if log and positive.size > 0:
        record["log_width"] = float(positive.max() / positive.min())
    return record


def rmse_to_reference(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Root-mean-square distance to the reference, per entry, in S/S0 units.

    Deliberately noise-free: unlike chi2 it does not divide by sigma, so it
    ranks entries by raw prediction mismatch and does not move when the noise
    slider does.  Zero at the reference entry itself.
    """
    residual = np.asarray(block, dtype=float) - np.asarray(reference, dtype=float)
    return np.sqrt(np.einsum("ij,ij->i", residual, residual) / residual.shape[1])


def slice_manifold(block: np.ndarray, reference: np.ndarray, labels: LibraryLabels,
                   eligible_rows: np.ndarray, threshold: float,
                   sigma_measurement: float = DEFAULT_SIGMA,
                   variance_block: np.ndarray | None = None,
                   s0_mode: str = "fixed",
                   display_values: np.ndarray | None = None,
                   display_names: list[str] | None = None) -> SliceResult:
    """Keep entries whose prediction matches ``reference`` on the given columns.

    Parameters
    ----------
    block : (n_eligible, n_measured) S/S0 of the eligible entries.
    reference : (n_measured,) the point being sliced around.
    eligible_rows : library row index of each row of ``block``.
    threshold : chi2 cut.  ``n_measured`` means "reduced chi2 = 1".
    variance_block : matching between-ensemble variance, or None to ignore
        the library's own Monte-Carlo error.
    """
    sigma = column_sigma(sigma_measurement, variance_block)
    if s0_mode == "fixed":
        chi2 = chi2_fixed_s0(block, reference, sigma)
    elif s0_mode == "free":
        chi2 = chi2_free_s0(block, reference, sigma)
    else:
        raise ValueError(f"unknown s0_mode {s0_mode!r}; use 'fixed' or 'free'")

    survivors = chi2 <= float(threshold)
    rows = eligible_rows[survivors]

    result = SliceResult(
        reference_signal=np.asarray(reference, dtype=float),
        measured_columns=np.asarray([], dtype=np.int64),
        chi2=chi2, survivors=survivors, eligible_rows=eligible_rows,
        threshold=float(threshold), sigma_measurement=float(sigma_measurement),
        s0_mode=s0_mode,
    )
    result.parameter_stats = {
        "rho": _spread(labels.nominal_rhos[rows], log=True),
        "V": _spread(labels.nominal_Vs[rows], log=True),
        "k_io": _spread(labels.kios[rows]),
        "vi": _spread(labels.vis[rows]),
    }
    if display_values is not None and display_names is not None:
        # Display axes may be collapsed groups rather than single columns, so
        # they arrive already reduced to one number per entry per axis.
        result.display_stats = {
            name: _spread(np.asarray(display_values)[survivors, j])
            for j, name in enumerate(display_names)
        }
    return result


def widths_versus_measured_count(block: np.ndarray, reference: np.ndarray,
                                 labels: LibraryLabels, eligible_rows: np.ndarray,
                                 sigma_measurement: float = DEFAULT_SIGMA,
                                 variance_block: np.ndarray | None = None,
                                 s0_mode: str = "fixed",
                                 reduced_threshold: float = 1.0) -> list[dict]:
    """Add measured columns one at a time and watch the parameter spreads shrink.

    Columns are used in the order they appear in ``block``.  The threshold is
    kept at ``reduced_threshold * k`` for k measured columns, so each step asks
    the same question ("agrees to about one sigma per column") rather than a
    progressively harsher one.
    """
    answer = []
    for k in range(1, block.shape[1] + 1):
        variance_k = None if variance_block is None else variance_block[:, :k]
        step = slice_manifold(
            block[:, :k], reference[:k], labels, eligible_rows,
            threshold=reduced_threshold * k,
            sigma_measurement=sigma_measurement,
            variance_block=variance_k, s0_mode=s0_mode,
        )
        answer.append({
            "n_measured": k,
            "n_survivors": step.n_survivors,
            "rho_log_width": step.parameter_stats["rho"]["log_width"],
            "V_log_width": step.parameter_stats["V"]["log_width"],
            "kio_std": step.parameter_stats["k_io"]["std"],
            "rho_std": step.parameter_stats["rho"]["std"],
            "V_std": step.parameter_stats["V"]["std"],
            # Survivors with rho <= 0 (only the free-water atom) sit outside
            # any log ratio; the width above is taken over the rest.
            "n_nonpositive_rho": step.parameter_stats["rho"]["n_nonpositive"],
        })
    return answer

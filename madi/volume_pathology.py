"""Phase-4 arithmetic for the unrealistic-cell-volume pathology.

`fisher_crlb_analysis_plan.md` §7 asks why voxelwise MADI fits produce cell
volumes far above any plausible value — an exponential-like branch running to
about 180 pL/cell in the OHSU library, against MADI II medians of 6.0 pL
(cortical grey matter) and 0.91 pL (white matter) — and which of four
hypotheses accounts for it.

Everything here is a property of a fitted map, a measured signal, or a Fisher
matrix that has already been formed.  Nothing in this module knows about a
scanner, and nothing here decides which library columns exist: the trust floor
and the amplitude treatment enter Phase 4 as *experimental conditions switched
on and off* (§7, hypotheses H3 and H4), which is the correct use of a
conditional quantity and the opposite of the defect §2.8 exists to prevent.

The Fisher primitives Phase 4 needs — the contrast bound on `log v_i`, the
canonical-node join, and the fit-time trust floor — live in
`madi.fisher_crlb`, because they are Fisher/library arithmetic and must have
exactly one definition shared with Phases 1-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# The Jackson thesis handled the pathology with a hard 20 pL cutoff.  It is
# quoted here as the historical handling this phase is asked to replace
# (plan §7 and §4.5), not as a threshold this analysis invents: every reported
# comparison is also given as a function of the cutoff so no conclusion rests
# on the particular value.
THESIS_VOLUME_CUTOFF_PL = 20.0

# MADI II parameter medians, for scale.
MADI_II_CORTICAL_GM_VOLUME_PL = 6.0
MADI_II_WHITE_MATTER_VOLUME_PL = 0.91

# Free water at body temperature, as the ADC scale a CSF/partial-volume voxel
# approaches.  Used only to label an axis and to state where the free-water end
# of the ADC range is; no voxel is excluded on it.
FREE_WATER_ADC_UM2_PER_MS = 3.0


def apparent_diffusion_coefficient(measured: np.ndarray, b_values: np.ndarray, *,
                                   b_max: float | None = None,
                                   signal_floor: float = 1e-3) -> np.ndarray:
    """Per-voxel ADC [um^2/ms] from shell-averaged `S/S0`, by log-linear fit.

    `measured` is `(voxels, shells)` of `S/S0` and `b_values` the matching
    b-values in s/mm^2.  Because the signal is already normalized by its own
    `b = 0`, `ln(S/S0)` is fitted **through the origin**:

        ADC = - sum_c b_c ln(S_c/S0) / sum_c b_c^2

    which is the ordinary least-squares slope under that constraint.  A shell
    whose signal is at or below `signal_floor` carries no usable log and is
    dropped from that voxel's fit; a voxel left with no shell returns NaN.

    `b_max` restricts the fit to shells at or below a stated b.  A mono-
    exponential ADC is a good description only at low b — the same non-Gaussian
    decay MADI models is what makes a high-b ADC a biased slope — so the choice
    is a real one and is reported with the result rather than defaulted
    silently.  Units: b in s/mm^2 with the log slope gives mm^2/s, multiplied
    by 1e3 to reach the um^2/ms this project states diffusivities in.
    """
    measured = np.asarray(measured, dtype=float)
    b_values = np.asarray(b_values, dtype=float)
    if measured.ndim != 2 or measured.shape[1] != b_values.shape[0]:
        raise ValueError("measured must be (voxels, shells) matching b_values")
    keep_shell = b_values > 0
    if b_max is not None:
        keep_shell &= b_values <= float(b_max)
    if not keep_shell.any():
        raise ValueError("no diffusion-weighted shell survives the b_max restriction")
    b = b_values[keep_shell]
    signal = measured[:, keep_shell]
    usable = np.isfinite(signal) & (signal > signal_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_signal = np.where(usable, np.log(np.where(usable, signal, 1.0)), 0.0)
    numerator = -(log_signal * b[None, :] * usable).sum(axis=1)
    denominator = ((b[None, :] ** 2) * usable).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        adc = np.where(denominator > 0, numerator / denominator, np.nan)
    return adc * 1e3


def reachable_volume_ceiling(rho_cells_per_uL: np.ndarray, vi_max: float) -> np.ndarray:
    """Largest `V` [pL] the mask band allows at a given `rho`: `vi_max/(rho*1e-6)`.

    The library exists only inside `vi_min <= rho*V*1e-6 <= vi_max`, so a fit
    cannot report a volume above this curve however badly the data are
    explained.  Plan §4.4 makes the point that blow-up values landing *at* this
    boundary is not what a random fitting failure would produce.
    """
    rho = np.asarray(rho_cells_per_uL, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rho > 0, float(vi_max) / (rho * 1e-6), np.nan)


def band_position(rho_cells_per_uL: np.ndarray, volume_pL: np.ndarray,
                  vi_min: float, vi_max: float) -> np.ndarray:
    """Where a fitted `(rho, V)` sits across the mask band, on a 0-to-1 scale.

    0 is the lower `v_i` edge, 1 the upper.  Measured in **log** `v_i`, the
    coordinate the band is uniform in and the one the degeneracy runs along, so
    the scale is not distorted by the band's multiplicative width.  Values
    outside `[0, 1]` mean the point is outside the band, which a MAP estimate
    cannot be and a posterior mean can.
    """
    rho = np.asarray(rho_cells_per_uL, dtype=float)
    volume = np.asarray(volume_pL, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vi = rho * volume * 1e-6
        return ((np.log(np.where(vi > 0, vi, np.nan)) - np.log(vi_min))
                / (np.log(vi_max) - np.log(vi_min)))


def probability_of_superiority(inside: np.ndarray, outside: np.ndarray) -> float:
    """`P(X > Y) + 0.5 P(X = Y)` for two samples — a threshold-free effect size.

    Also called the common-language effect size or the Mann-Whitney AUC.  0.5
    means the two groups are indistinguishable, 1.0 that every member of the
    first exceeds every member of the second.  Used instead of a difference of
    means because the quantities Phase 4 compares — residuals, `kappa` — are
    heavy-tailed by several orders of magnitude, so a mean difference would be
    a statement about the tail rather than about the groups.
    """
    a = np.asarray(inside, dtype=float)
    b = np.asarray(outside, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="stable")
    ranks = np.empty(combined.size, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1, dtype=float)
    # Average ranks within ties, so an exact tie contributes 0.5 rather than 1.
    values = combined[order]
    start = 0
    for stop in np.flatnonzero(np.diff(values)) + 1:
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    if combined.size - start > 1:
        ranks[order[start:]] = ranks[order[start:]].mean()
    rank_sum = ranks[:a.size].sum()
    return float((rank_sum - a.size * (a.size + 1) / 2.0) / (a.size * b.size))


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {"count": int(finite.size), "min": float(finite.min()),
            "q05": float(np.quantile(finite, 0.05)), "median": float(np.median(finite)),
            "q95": float(np.quantile(finite, 0.95)), "max": float(finite.max())}


def stratified_comparison(values: np.ndarray, group: np.ndarray, *,
                          label: str = "") -> dict[str, Any]:
    """Distribution of `values` inside and outside a boolean `group`, plus effect size.

    This is the shape every Phase-4 hypothesis test takes: a quantity is
    compared between the voxels showing the pathology and the rest, and the
    comparison is reported as two distributions plus a threshold-free effect
    size rather than as a single difference.
    """
    values = np.asarray(values, dtype=float)
    group = np.asarray(group, dtype=bool)
    if values.shape != group.shape:
        raise ValueError("values and group must have the same shape")
    inside, outside = values[group], values[~group]
    return {
        "label": label,
        "inside": _quantiles(inside),
        "outside": _quantiles(outside),
        "median_ratio_inside_over_outside": (
            float(np.median(inside[np.isfinite(inside)]) / np.median(outside[np.isfinite(outside)]))
            if np.isfinite(inside).any() and np.isfinite(outside).any()
            and np.median(outside[np.isfinite(outside)]) != 0 else float("nan")),
        "probability_of_superiority_inside_over_outside": probability_of_superiority(inside, outside),
    }


def cutoff_sweep(volume_pL: np.ndarray, cutoffs: np.ndarray) -> list[dict[str, float]]:
    """Fraction of voxels above each of a range of volume cutoffs.

    Reported so that no Phase-4 conclusion depends on the thesis's particular
    20 pL choice: an effect that only exists at one cutoff is a property of the
    cutoff.
    """
    volume = np.asarray(volume_pL, dtype=float)
    finite = volume[np.isfinite(volume)]
    return [{"cutoff_pL": float(c),
             "fraction_above": float(np.mean(finite > c)) if finite.size else float("nan"),
             "count_above": int(np.count_nonzero(finite > c))}
            for c in np.asarray(cutoffs, dtype=float)]


@dataclass(frozen=True)
class KappaGatedReport:
    """Plan §4.5's principled replacement for a hard volume cutoff.

    `v_i` and its CRLB-derived error bar are reported everywhere, because the
    stiff combination is what the data determine.  `rho` and `V` are released
    separately only where their degeneracy-inflation factor is below the
    pre-registered threshold, which is a statement about information rather
    than about plausibility — the 20 pL cutoff it replaces discards a voxel for
    reporting an implausible number, whereas this one declines to report a
    number the acquisition never determined.
    """

    log_vi: np.ndarray
    log_vi_sigma: np.ndarray
    report_rho: np.ndarray
    report_volume: np.ndarray
    report_both: np.ndarray
    has_bound: np.ndarray
    kappa_threshold: float


def kappa_gated_report(rho_cells_per_uL: np.ndarray, volume_pL: np.ndarray,
                       kappa: np.ndarray, log_vi_sigma: np.ndarray,
                       kappa_threshold: float) -> KappaGatedReport:
    """Apply §4.5's reporting rule to a fitted map joined to its Fisher nodes.

    `kappa` is `(voxels, 3)` in PARAMETER_ORDER and `log_vi_sigma` the contrast
    bound from `madi.fisher_crlb.directional_crlb`.  A voxel whose node carries
    no Fisher matrix — the mask-band edges, where 136 of 369 `(rho, V)` pairs
    have no interior stencil — has no bound and is gated out of the
    parameter-level report rather than silently passed, since "no information
    was measured here" and "the information was good" must not look alike.
    """
    rho = np.asarray(rho_cells_per_uL, dtype=float)
    volume = np.asarray(volume_pL, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    sigma = np.asarray(log_vi_sigma, dtype=float)
    if kappa.ndim != 2 or kappa.shape[1] != 3:
        raise ValueError("kappa must be (voxels, 3) in PARAMETER_ORDER")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vi = np.log(np.where((rho > 0) & (volume > 0), rho * volume * 1e-6, np.nan))
    has_bound = np.isfinite(sigma) & np.isfinite(kappa).all(axis=1)
    report_rho = has_bound & (kappa[:, 0] < kappa_threshold)
    report_volume = has_bound & (kappa[:, 1] < kappa_threshold)
    return KappaGatedReport(
        log_vi=log_vi, log_vi_sigma=sigma,
        report_rho=report_rho, report_volume=report_volume,
        report_both=report_rho & report_volume, has_bound=has_bound,
        kappa_threshold=float(kappa_threshold))


def validate_quality_flag(posterior_sigma: np.ndarray, crlb_sigma: np.ndarray) -> dict[str, Any]:
    """Does the Bayes posterior spread track the Fisher matrix's prediction?

    Plan §4.5 asks for the posterior standard deviation to be *validated*
    against the CRLB, "since the Fisher matrix predicts where the ridge is worst
    before any fitting is done".  The comparison is reported as a rank
    correlation plus the distribution of their ratio, and deliberately not as a
    claim of equality: a posterior standard deviation over a bounded library is
    not an unbiased-estimator standard deviation, and where the CRLB is enormous
    the posterior is capped by the extent of the candidate set rather than by the
    data.  The rank correlation is the part that carries the claim; the ratio
    says which way the cap bites.
    """
    posterior = np.asarray(posterior_sigma, dtype=float)
    crlb = np.asarray(crlb_sigma, dtype=float)
    both = np.isfinite(posterior) & np.isfinite(crlb) & (posterior > 0) & (crlb > 0)
    if np.count_nonzero(both) < 3:
        return {"paired_voxels": int(np.count_nonzero(both))}
    a, b = posterior[both], crlb[both]

    def _rank(x):
        order = np.argsort(x, kind="stable")
        ranks = np.empty(x.size, dtype=float)
        ranks[order] = np.arange(1, x.size + 1, dtype=float)
        values = x[order]
        start = 0
        for stop in np.flatnonzero(np.diff(values)) + 1:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
        if x.size - start > 1:
            ranks[order[start:]] = ranks[order[start:]].mean()
        return ranks

    ra, rb = _rank(a), _rank(b)
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    ratio = a / b
    return {
        "paired_voxels": int(a.size),
        "spearman_rank_correlation": spearman,
        "pearson_on_logs": float(np.corrcoef(np.log(a), np.log(b))[0, 1]),
        "posterior_over_crlb_ratio": _quantiles(ratio),
        "posterior_sigma": _quantiles(a),
        "crlb_sigma": _quantiles(b),
    }


def expected_noise_residual(sigma_raw: float, s0_voxel: np.ndarray,
                            averages_per_column: np.ndarray) -> np.ndarray:
    """Residual a voxel would show from measurement noise alone, per voxel.

    A shell averaged over `n_c` volumes and normalized by the voxel's own `S0`
    has noise `sigma_c = sigma_raw / (S0 sqrt(n_c))`, so a model that fitted the
    signal exactly would still leave `sum_c sigma_c^2` of squared residual.  That
    is the scale a residual has to be read against: an absolute residual is
    smaller in a voxel whose `S0` is large for reasons that have nothing to do
    with how well the model fits.

    This is the zero-fitted-degrees-of-freedom expectation.  A three-parameter
    least-squares fit removes about three of the `K` columns' worth, and a grid
    MAP removes an ill-defined, spacing-dependent amount, so the ratio built on
    it is used to **compare groups of voxels**, not to calibrate an absolute
    goodness of fit.
    """
    s0 = np.asarray(s0_voxel, dtype=float)
    n = np.asarray(averages_per_column, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_column_variance_over_s0sq = np.sum(1.0 / n) * float(sigma_raw) ** 2
        return np.where(s0 > 0, per_column_variance_over_s0sq / s0 ** 2, np.nan)


def goodness_of_fit_ratio(residual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """`residual / expected_noise_residual`: how far a fit is above its noise floor.

    Plan §4.3's discriminator.  Hypothesis H1 (a degeneracy ridge) predicts the
    pathological voxels are fitted about as well as noise permits, because many
    entries along the ridge explain the data comparably and one was chosen
    arbitrarily.  Hypothesis H2 (out-of-model signal) predicts they sit far above
    it, because nothing in the library explains them.
    """
    residual = np.asarray(residual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(expected > 0, residual / expected, np.nan)


def log_displacement(rho_from: np.ndarray, volume_from: np.ndarray,
                     rho_to: np.ndarray, volume_to: np.ndarray) -> dict[str, np.ndarray]:
    """How a fitted `(rho, V)` moved between two fits of the same voxel.

    Plan §4.4's mechanism is MAP estimates *sliding along the ridge*.  If that is
    what happens, then when a fit is perturbed — a different amplitude treatment,
    a different column set — the estimate moves along the constant-`v_i`
    hyperbola: `log rho` and `log V` change by equal and opposite amounts and
    `log v_i` does not change at all.  This returns the displacement in
    `(log rho, log V)`, its acute angle to the hyperbola `(1, -1)`, and the
    separate changes in `log V` and `log v_i`, so that claim is measured rather
    than read off a picture.

    Voxels that did not move have no direction; their angle is NaN.
    """
    from .fisher_crlb import CONSTANT_VI_DIRECTION, direction_angle_deg

    with np.errstate(divide="ignore", invalid="ignore"):
        d_log_rho = np.log(np.asarray(rho_to, float)) - np.log(np.asarray(rho_from, float))
        d_log_volume = np.log(np.asarray(volume_to, float)) - np.log(np.asarray(volume_from, float))
    step = np.stack([d_log_rho, d_log_volume], axis=-1)
    moved = np.isfinite(step).all(axis=-1) & (np.linalg.norm(np.nan_to_num(step), axis=-1) > 0)
    reference = CONSTANT_VI_DIRECTION[:2] / np.linalg.norm(CONSTANT_VI_DIRECTION[:2])
    angle = np.full(step.shape[:-1], np.nan)
    if moved.any():
        angle[moved] = direction_angle_deg(step[moved], reference)
    return {
        "d_log_rho": d_log_rho, "d_log_volume": d_log_volume,
        "d_log_vi": d_log_rho + d_log_volume,
        "angle_to_hyperbola_deg": angle, "moved": moved,
    }


def permuted_displacement_null(rho_from: np.ndarray, volume_from: np.ndarray,
                               rho_to: np.ndarray, volume_to: np.ndarray, *,
                               seed: int = 0) -> dict[str, np.ndarray]:
    """What a move would look like if it had nothing to do with the voxel.

    The library's `(rho, V)` band is about 23 times longer along the
    constant-`v_i` hyperbola than across it, so a move between *any* two points
    inside it makes a small angle with the hyperbola: that is geometry, with no
    degeneracy involved.  A uniform 45-degree null therefore overstates how
    unusual a small angle is, and a ridge claim read against it would be partly
    a statement about the shape of the mask.

    This null keeps each voxel's starting estimate and pairs it with the perturbed
    estimate of a randomly chosen *other* voxel from the same pair of fits.  That
    preserves the band geometry and the fitted-parameter distribution exactly and
    breaks only the link between a voxel and its own refit.  A ridge mechanism
    predicts observed angles smaller than this null's, not merely smaller than 45
    degrees.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(rho_to).shape[0])
    return log_displacement(rho_from, volume_from, np.asarray(rho_to, dtype=float)[order],
                            np.asarray(volume_to, dtype=float)[order])

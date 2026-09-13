"""Synthetic tests for the Phase-4 unrealistic-volume arithmetic (`madi.volume_pathology`)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from madi.volume_pathology import (apparent_diffusion_coefficient, band_position, cutoff_sweep,
                                   expected_noise_residual, goodness_of_fit_ratio, kappa_gated_report,
                                   log_displacement, probability_of_superiority, reachable_volume_ceiling,
                                   stratified_comparison, validate_quality_flag)


def test_adc_recovers_an_exact_mono_exponential_and_honours_b_max() -> None:
    """Exact on a mono-exponential; and `b_max` really selects the low-b slope.

    A bi-exponential decay is convex in `ln S`, so its low-b slope is steeper than
    its all-shell slope.  That is why the ADC's b-range is a real choice and is
    reported rather than defaulted.
    """
    b = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    adc = np.array([0.7, 1.4, 3.0])
    signal = np.exp(-b[None, :] * adc[:, None] * 1e-3)
    assert np.allclose(apparent_diffusion_coefficient(signal, b), adc, rtol=1e-12)
    bi = 0.6 * np.exp(-b * 2.5e-3) + 0.4 * np.exp(-b * 0.3e-3)
    assert (apparent_diffusion_coefficient(bi[None, :], b, b_max=1000.0)[0]
            > apparent_diffusion_coefficient(bi[None, :], b)[0])


def test_adc_ignores_shells_at_or_below_the_signal_floor() -> None:
    b = np.array([0.0, 1000.0, 2000.0])
    assert np.isclose(apparent_diffusion_coefficient(np.array([[1.0, np.exp(-1.0), 0.0005]]), b)[0], 1.0)
    assert np.isnan(apparent_diffusion_coefficient(np.array([[1.0, 0.0, 0.0]]), b)[0])


def test_probability_of_superiority_is_the_mann_whitney_auc_including_ties() -> None:
    rng = np.random.default_rng(8)
    x = np.round(rng.normal(0.4, 1.0, 300), 1)
    y = np.round(rng.normal(0.0, 1.0, 350), 1)
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    assert np.isclose(probability_of_superiority(x, y), u / (x.size * y.size), rtol=0, atol=1e-12)
    assert probability_of_superiority(np.ones(4), np.ones(9)) == 0.5
    assert np.isnan(probability_of_superiority(np.array([np.nan]), np.ones(3)))


def test_the_reachable_volume_ceiling_is_the_plan_value_at_the_grid_minimum() -> None:
    """Plan §4.4: `V_max = 0.99 / (rho_min * 1e-6) = 99 pL` at `rho = 1e4`."""
    assert np.isclose(reachable_volume_ceiling(np.array([1e4]), 0.99)[0], 99.0)
    assert np.isnan(reachable_volume_ceiling(np.array([0.0]), 0.99)[0])


def test_band_position_is_zero_and_one_on_the_mask_edges() -> None:
    rho = np.array([2e5, 2e5, 2e5])
    volume = np.array([0.40, 0.99, np.sqrt(0.40 * 0.99)]) / (rho * 1e-6)
    assert np.allclose(band_position(rho, volume, 0.40, 0.99), [0.0, 1.0, 0.5], atol=1e-12)


def test_the_goodness_of_fit_ratio_averages_one_for_pure_noise() -> None:
    """The noise floor is calibrated: a residual of pure noise reads as ratio 1.

    Shell averaging and per-voxel `S0` both enter, exactly as they do for the
    measured acquisition this phase fits.
    """
    rng = np.random.default_rng(21)
    averages = np.array([6, 18, 24, 30, 36])
    sigma_raw = 170.0
    s0 = rng.uniform(1500.0, 6000.0, 40000)
    noise = rng.normal(size=(s0.size, averages.size)) * (sigma_raw / (s0[:, None] * np.sqrt(averages)))
    ratio = goodness_of_fit_ratio((noise ** 2).sum(axis=1), expected_noise_residual(sigma_raw, s0, averages))
    assert abs(np.mean(ratio) - 1.0) < 0.02


def test_a_displacement_along_the_hyperbola_leaves_v_i_unchanged() -> None:
    """The ridge signature, measured: equal and opposite `log rho` / `log V` moves."""
    step = log_displacement(np.array([1e5, 1e5, 1e5]), np.array([5.0, 5.0, 5.0]),
                            np.array([3e5, 1e5, 1e5]), np.array([5.0 / 3.0, 10.0, 5.0]))
    assert np.isclose(step["angle_to_hyperbola_deg"][0], 0.0, atol=1e-6)
    assert np.isclose(step["d_log_vi"][0], 0.0, atol=1e-12)
    assert np.isclose(step["angle_to_hyperbola_deg"][1], 45.0)
    assert np.isclose(step["d_log_vi"][1], np.log(2.0))
    assert not step["moved"][2] and np.isnan(step["angle_to_hyperbola_deg"][2])


def test_the_kappa_gate_withholds_parameters_the_acquisition_did_not_determine() -> None:
    """Plan §4.5: `rho`, `V` each released only below the pre-registered `kappa`.

    A voxel with no Fisher matrix at its node has no bound and is withheld, never
    passed through as if its information were good.
    """
    rho = np.full(4, 1e5)
    volume = np.full(4, 5.0)
    kappa = np.array([[2.0, 3.0, 1.5], [50.0, 3.0, 1.5], [2.0, 50.0, 1.5], [np.nan, np.nan, np.nan]])
    report = kappa_gated_report(rho, volume, kappa, np.array([0.1, 0.1, 0.1, np.nan]), 10.0)
    assert report.report_rho.tolist() == [True, False, True, False]
    assert report.report_volume.tolist() == [True, True, False, False]
    assert report.report_both.tolist() == [True, False, False, False]
    assert report.has_bound.tolist() == [True, True, True, False]
    assert np.allclose(report.log_vi, np.log(0.5))


def test_the_quality_flag_validation_recovers_a_monotone_relationship_and_rejects_noise() -> None:
    rng = np.random.default_rng(3)
    crlb = np.exp(rng.uniform(-3.0, 2.0, 5000))
    posterior = crlb * np.exp(rng.normal(0.0, 0.2, crlb.size))
    tracked = validate_quality_flag(posterior, crlb)
    assert tracked["spearman_rank_correlation"] > 0.95
    assert 0.9 < tracked["posterior_over_crlb_ratio"]["median"] < 1.1
    assert abs(validate_quality_flag(rng.permutation(posterior), crlb)["spearman_rank_correlation"]) < 0.05


def test_the_cutoff_sweep_never_increases_with_the_cutoff() -> None:
    volume = np.random.default_rng(5).lognormal(1.0, 1.2, 4000)
    rows = cutoff_sweep(volume, np.array([1.0, 5.0, 20.0, 90.0]))
    fractions = [row["fraction_above"] for row in rows]
    assert fractions == sorted(fractions, reverse=True)
    assert rows[2]["count_above"] == int(np.count_nonzero(volume > 20.0))


def test_stratified_comparison_reports_both_groups_and_their_effect_size() -> None:
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    group = np.array([False, False, False, True, True, True])
    out = stratified_comparison(values, group)
    assert out["inside"]["median"] == 11.0 and out["outside"]["median"] == 2.0
    assert out["probability_of_superiority_inside_over_outside"] == 1.0


def test_a_thin_band_makes_small_angles_by_geometry_alone() -> None:
    """The permutation null exists because the uniform null is the wrong reference.

    Points spread over a strip 23 times longer along the hyperbola than across it
    already give small angles to the hyperbola when paired at random.  A true
    along-ridge move must beat THAT, and here it does, while the random pairing
    still sits far below 45 degrees.
    """
    from madi.volume_pathology import permuted_displacement_null

    rng = np.random.default_rng(12)
    along = np.array([1.0, -1.0]) / np.sqrt(2.0)
    across = np.array([1.0, 1.0]) / np.sqrt(2.0)
    s = rng.uniform(0.0, 4.2, 3000)
    t = rng.uniform(0.0, 0.18, 3000)
    start = np.array([4.5, 1.2]) + s[:, None] * along + t[:, None] * across      # log10 (rho, V)
    moved = start + 0.3 * along                                                 # a pure along-ridge move
    rho0, vol0 = 10 ** start[:, 0], 10 ** start[:, 1]
    rho1, vol1 = 10 ** moved[:, 0], 10 ** moved[:, 1]

    observed = log_displacement(rho0, vol0, rho1, vol1)["angle_to_hyperbola_deg"]
    null = permuted_displacement_null(rho0, vol0, rho1, vol1, seed=3)["angle_to_hyperbola_deg"]
    assert np.nanmedian(observed) < 1e-6
    assert np.nanmedian(null) < 15.0                        # geometry alone, far below the uniform 45
    assert probability_of_superiority(null, observed) > 0.99
    again = permuted_displacement_null(rho0, vol0, rho1, vol1, seed=3)["angle_to_hyperbola_deg"]
    assert np.array_equal(null, again, equal_nan=True)      # the null is reproducible from its seed

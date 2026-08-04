"""Tier-A checks for the SI Eq. S2 full-facet contracted geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from madi.config import SimConfig
from madi.ensemble import (
    alpha_star_from_vi,
    create_ensemble,
    governing_mean_A_over_V,
    si_domain_side_um,
)
from madi.walker_gpu import _membrane_transition_counts

from ._si_test_support import si_test_config, write_uncertified_reference


pytestmark = pytest.mark.tier_a

VI_TARGETS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
RHO_TARGETS = (1.0e5, 5.0e5, 1.0e6)


def _s7a(x: float) -> float:
    return -0.570444 * x**3 + 10.6047 * x**2 - 5.81457 * x + 1.0


@pytest.mark.parametrize("vi", [0.40, 0.784, 0.90, 0.99])
def test_si_s7a_inversion_has_one_continuous_physical_branch(vi: float) -> None:
    """P0-A: cubic inversion replaces the old 25-row clamp completely."""
    rho = 5.0e5
    alpha = alpha_star_from_vi(vi, rho)
    x = (rho / 1.0e9) ** (1.0 / 3.0) * alpha
    assert _s7a(x) == pytest.approx(vi, abs=2e-12)
    assert alpha >= 0.0
    if vi == pytest.approx(0.784):
        assert alpha == pytest.approx(0.5048514755, abs=2e-8)


def test_s7a_high_vi_targets_are_distinct_and_never_clamped() -> None:
    rho = 5.0e5
    targets = np.asarray([0.80, 0.85, 0.90, 0.94, 0.99])
    values = np.asarray([alpha_star_from_vi(vi, rho) for vi in targets])
    assert np.all(np.diff(values) < 0.0)
    assert np.unique(np.round(values, 12)).size == len(values)


def test_governing_reference_is_untrimmed_and_scales_as_rho_one_third(tmp_path) -> None:
    """P0-B/SI §S.IV: Eq. 5 consumes the governing-cloud mean only."""
    ref = write_uncertified_reference(tmp_path)
    strict = SimConfig(
        geometry_reference_path=str(ref),
        small_deltas=[1.0],
        big_deltas=[1.0],
        T_max_ms=2.0,
    )
    with pytest.raises(RuntimeError, match="not SI-certified"):
        governing_mean_A_over_V(0.80, 5.0e5, strict)

    cfg = si_test_config(tmp_path)
    low, reference = governing_mean_A_over_V(0.80, 1.0e5, cfg)
    high, _ = governing_mean_A_over_V(0.80, 8.0e5, cfg)
    expected_low = (4.0 - 2.0 * 0.80) * (1.0e5 / 1.0e9) ** (1.0 / 3.0)
    # The fixture's table is linear in v_i.  This checks both interpolation
    # (0.80 lies between its 26 rows) and the SI rho^(1/3) rescaling.
    assert low == pytest.approx(expected_low, rel=1e-12)
    assert high / low == pytest.approx(2.0, rel=1e-12)
    assert reference.metadata["mean_estimator"] == "untrimmed_arithmetic_mean_A_over_V"
    assert reference.metadata["kappa"] == pytest.approx(0.90)


def test_si_domain_size_and_source_cube_follow_s8_for_128ms(tmp_path) -> None:
    """P0-C: W=554.256 µm and Ωsrc=0.4W at rho=5e5 / 128 ms."""
    cfg = si_test_config(tmp_path, L=None, T_max_ms=128.0)
    L_rw = math.sqrt(2.0 * cfg.D0 * cfg.tRW_max)
    expected = max(10.0 * L_rw, min(20.0 * L_rw, (8.0e5 / (5.0e5 / 1e9)) ** (1.0 / 3.0)))
    assert si_domain_side_um(5.0e5, cfg) == pytest.approx(expected)
    assert expected == pytest.approx(554.2562584, rel=1e-9)


@pytest.mark.parametrize("rho", RHO_TARGETS)
@pytest.mark.parametrize("target_vi", VI_TARGETS)
def test_full_facet_realised_labels_match_s7a_across_p0a_grid(
    target_vi: float, rho: float, tmp_path
) -> None:
    """P0-A Tier A: full facets pass the required 9-v_i × 3-rho grid."""
    cfg = si_test_config(
        tmp_path,
        L=None,
        T_max_ms=128.0,
        geometry_validation_points=20_000,
        geometry_vi_tolerance=0.005,
    )
    seed = 20_260_900 + int(round(target_vi * 100)) + int(rho // 1.0e5)
    ensemble = create_ensemble(rho, target_vi * 1.0e6 / rho, cfg, seed=seed)
    allowed = max(cfg.geometry_vi_tolerance, 4.0 * ensemble.geometry.realised_vi_se)
    assert abs(ensemble.vi - target_vi) <= allowed
    assert ensemble.rho * ensemble.V * 1e-6 == pytest.approx(ensemble.vi, abs=1e-12)
    assert ensemble.mean_AV == pytest.approx(governing_mean_A_over_V(target_vi, rho, cfg)[0])
    assert ensemble.geometry.population.s10_pass
    assert ensemble.geometry.population.s11_pass
    assert ensemble.source_hi - ensemble.source_lo == pytest.approx(0.4 * ensemble.L)


def _two_nearest_labels(ensemble, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic-only SI §S.IV.a shortcut; never used by the walker."""
    distances, indices = ensemble._tree().query(points, k=2)
    nearest = np.asarray(indices[:, 0], dtype=np.int32)
    second = np.asarray(indices[:, 1], dtype=np.int32)
    separation = np.linalg.norm(ensemble.seeds[second] - ensemble.seeds[nearest], axis=1)
    margin = (distances[:, 1] ** 2 - distances[:, 0] ** 2) / (2.0 * separation)
    return nearest, margin >= ensemble.annulus[nearest]


def _brute_force_full_facet_labels(ensemble, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference SI Eq. S2 evaluation over every seed, in bounded-memory blocks."""
    ids_out = np.empty(len(points), dtype=np.int32)
    inside_out = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), 128):
        stop = min(start + 128, len(points))
        block = points[start:stop]
        point_to_seed = ensemble.seeds[None, :, :] - block[:, None, :]
        d_sq = np.einsum("...i,...i->...", point_to_seed, point_to_seed)
        nearest = np.argmin(d_sq, axis=1).astype(np.int32)
        row = np.arange(len(block))
        d1_sq = d_sq[row, nearest]
        seed_to_nearest = ensemble.seeds[None, :, :] - ensemble.seeds[nearest, None, :]
        separation = np.linalg.norm(seed_to_nearest, axis=2)
        if np.any(separation[row, nearest] != 0.0):
            raise AssertionError("nearest seed must have zero self-separation")
        separation[row, nearest] = 1.0
        margin = (d_sq - d1_sq[:, None]) / (2.0 * separation)
        margin[row, nearest] = np.inf
        ids_out[start:stop] = nearest
        inside_out[start:stop] = np.all(margin >= ensemble.annulus[nearest, None], axis=1)
    return ids_out, inside_out


def test_cpu_classifier_matches_brute_force_all_facet_s2(tmp_path) -> None:
    """P0-D: adaptive-radius KD labels equal an all-seed SI Eq. S2 evaluation."""
    cfg = si_test_config(tmp_path)
    ensemble = create_ensemble(5.0e5, 1.4, cfg, seed=20_260_901, verify_vi=False)
    points = np.random.default_rng(20_260_902).uniform(0.0, ensemble.L, size=(2_000, 3))
    ids, inside = ensemble.classify_cpu(points)
    expected_ids, expected_inside = _brute_force_full_facet_labels(ensemble, points)
    assert np.array_equal(ids, expected_ids)
    assert np.array_equal(inside, expected_inside)
    assert not hasattr(ensemble, "grid_candidates")


def test_gpu_kd_tree_parent_links_cover_each_seed_once(tmp_path) -> None:
    """CUDA's stackless full-facet range query has a complete exact tree."""
    cfg = si_test_config(tmp_path)
    ensemble = create_ensemble(5.0e5, 1.4, cfg, seed=20_260_905, verify_vi=False)
    n = len(ensemble.seeds)
    assert len(ensemble.kd_node_seed) == n
    assert np.array_equal(np.sort(ensemble.kd_node_seed), np.arange(n, dtype=np.int32))
    assert ensemble.kd_node_parent[0] == -1
    for node in range(n):
        for child in (ensemble.kd_node_left[node], ensemble.kd_node_right[node]):
            if child >= 0:
                assert ensemble.kd_node_parent[child] == node


@pytest.mark.parametrize("target_vi", VI_TARGETS)
def test_two_nearest_discrepancy_curve_has_expected_one_sided_bias(target_vi: float, tmp_path) -> None:
    """P0-A: §S.IV.a moves gap points into cells; it cannot remove cell points."""
    cfg = si_test_config(tmp_path, L=None, T_max_ms=8.0, geometry_validation_points=4_000)
    rho = 5.0e5
    ensemble = create_ensemble(
        rho,
        target_vi * 1.0e6 / rho,
        cfg,
        seed=20_260_903 + int(target_vi * 100),
        verify_vi=False,
    )
    points = np.random.default_rng(20_260_904).uniform(0.0, ensemble.L, size=(20_000, 3))
    _, full_inside = ensemble.classify_cpu(points)
    _, two_inside = _two_nearest_labels(ensemble, points)
    assert np.all(~full_inside | two_inside)
    assert np.mean(two_inside) >= np.mean(full_inside)


def test_full_facet_limit_collapses_to_two_nearest_when_alpha_vanishes(tmp_path) -> None:
    """At alpha=0 the full conjunction is exactly the ordinary Voronoi cell."""
    cfg = si_test_config(tmp_path)
    ensemble = create_ensemble(
        5.0e5,
        0.99 * 1.0e6 / 5.0e5,
        cfg,
        seed=20_260_990,
        verify_vi=False,
    )
    points = np.random.default_rng(20_260_991).uniform(0.0, ensemble.L, size=(10_000, 3))
    original = ensemble.annulus.copy()
    ensemble.annulus.fill(0.0)
    try:
        full_ids, full_inside = ensemble.classify_cpu(points)
        two_ids, two_inside = _two_nearest_labels(ensemble, points)
    finally:
        ensemble.annulus[:] = original
    assert np.array_equal(full_ids, two_ids)
    assert np.array_equal(full_inside, two_inside)


def test_si_endpoint_only_membrane_counting_rule() -> None:
    old_cell = np.asarray([1, 1, 1, 1, 1], dtype=np.int32)
    old_inside = np.asarray([True, False, True, True, False])
    new_cell = np.asarray([1, 1, 2, 1, 1], dtype=np.int32)
    new_inside = np.asarray([False, True, True, True, False])
    assert np.array_equal(
        _membrane_transition_counts(old_cell, old_inside, new_cell, new_inside),
        np.asarray([1, 1, 2, 0, 0], dtype=np.int8),
    )

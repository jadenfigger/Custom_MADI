"""Tier-B independent geometry and detailed-balance diagnostics.

These are deliberately larger than the Tier-A P0 smoke tests, but remain
CPU-only.  They exercise the production periodic Poisson--Voronoi geometry
and its candidate-cache classifier against exact periodic KD-tree queries.
They replace the pre-remediation lookup-cache/XFAIL tests: no production
geometry may silently clamp a requested ``v_i`` or use a two-seed cache
without re-ranking candidates at the walker position.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from madi.config import SimConfig
from madi.ensemble import create_ensemble, estimate_vi
from madi.walker_gpu import _membrane_transition_counts, run_walk_Y


pytestmark = pytest.mark.tier_b


def _cfg(**overrides) -> SimConfig:
    values = dict(
        L=90.0,
        grid_spacing=0.75,
        classifier_candidates=8,
        geometry_calibration_points=80_000,
        geometry_validation_points=80_000,
        geometry_sample_cells=64,
        geometry_vi_tolerance=0.005,
        n_walkers=512,
        n_ensembles=1,
        T_max_ms=32.0,
        small_deltas=[2.0],
        big_deltas=[8.0, 16.0, 30.0],
        b_values=[0.0, 500.0, 1_000.0],
    )
    values.update(overrides)
    return SimConfig(**values)


def _ensemble(target_vi: float, rho: float = 5.0e5):
    cfg = _cfg()
    return create_ensemble(rho, target_vi * 1e6 / rho, cfg, seed=77_101)


def test_periodic_geometry_has_no_source_domain_or_boundary_population_assumption() -> None:
    """Production geometry tiles the primary domain instead of buffering it."""
    cfg = _cfg()
    assert cfg.boundary_mode == "periodic"
    assert cfg.grid_spacing < 1.0
    assert cfg.classifier_candidates >= 8


@pytest.mark.parametrize("target_vi", [0.40, 0.70, 0.90, 0.99])
def test_direct_volume_label_matches_requested_target_without_clamping(target_vi: float) -> None:
    ens = _ensemble(target_vi)
    measured = estimate_vi(ens, n=150_000, seed=77_102)
    se = np.sqrt(measured * (1.0 - measured) / 150_000)
    assert abs(measured - target_vi) <= 0.005 + 4.0 * se
    assert np.isclose(ens.rho * ens.V * 1e-6, ens.vi, rtol=0.0, atol=1e-12)


def test_high_vi_annuli_are_not_a_clamped_lookup_row() -> None:
    ensembles = [_ensemble(v) for v in (0.80, 0.85, 0.90, 0.94, 0.99)]
    alpha = np.asarray([e.alpha_star for e in ensembles])
    mean_av = np.asarray([e.mean_AV for e in ensembles])
    assert np.all(np.diff(alpha) < 0.0)
    assert np.unique(np.round(alpha, 8)).size == len(alpha)
    assert np.unique(np.round(mean_av, 8)).size == len(mean_av)


def test_candidate_cache_meets_compartment_and_crossing_error_budget() -> None:
    """P0-D acceptance bound on a larger independent point sample."""
    ens = _ensemble(0.90)
    cfg = _cfg()
    rng = np.random.default_rng(77_103)
    p0 = rng.uniform(0.0, cfg.L, size=(150_000, 3))
    p1 = np.mod(p0 + rng.normal(0.0, cfg.sigma, size=p0.shape), cfg.L)
    exact0 = ens.classify_exact_cpu(p0)
    exact1 = ens.classify_exact_cpu(p1)
    cached0 = ens.classify_cpu(p0)
    cached1 = ens.classify_cpu(p1)
    compartment_error = np.mean(exact0[1] != cached0[1])
    exact_m = _membrane_transition_counts(*exact0, *exact1)
    cached_m = _membrane_transition_counts(*cached0, *cached1)
    attempts = max(np.count_nonzero(exact_m), 1)
    crossing_error = np.count_nonzero(exact_m != cached_m) / attempts
    assert compartment_error < 5e-4
    assert crossing_error < 0.02


def test_exact_k2_and_k8_agree_for_periodic_voronoi_labelling() -> None:
    """The mathematical k=2 Voronoi fact is tested separately from caching."""
    ens = _ensemble(0.90)
    rng = np.random.default_rng(77_104)
    points = rng.uniform(0.0, ens.L, size=(100_000, 3))
    tree = cKDTree(ens.seeds, boxsize=ens.L)
    _, k2 = tree.query(points, k=2)
    _, k8 = tree.query(points, k=8)
    assert np.array_equal(k2, k8[:, :2])


@pytest.mark.parametrize("pp", [0.0, 0.01, 0.05])
def test_symmetric_pp_preserves_intracellular_occupancy_without_escape(pp: float) -> None:
    """Detailed balance is checked across p_p, with uniform-volume starts."""
    ens = _ensemble(0.70)
    cfg = _cfg(T_max_ms=24.0, n_walkers=384)
    _, escaped, telemetry = run_walk_Y(
        ens, 0.0, cfg, pp=pp, seed=77_105, verbose=False,
        return_telemetry=True,
    )
    occupancy = np.asarray(telemetry["occupancy_fraction"])
    se = np.sqrt(ens.vi * (1.0 - ens.vi) / cfg.n_walkers)
    assert escaped == 0
    assert np.max(np.abs(occupancy - ens.vi)) <= 5.0 * se + 0.01

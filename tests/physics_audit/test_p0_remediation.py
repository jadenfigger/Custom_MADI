"""P0 remediation acceptance tests.

Tier: A (local CPU, seconds to minutes).  These tests intentionally use
small, fixed-seed Poisson ensembles; production walker counts are larger, but
the assertions are formulated in terms of direct Monte-Carlo uncertainty.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from madi.config import GAMMA_RAD, SimConfig
from madi.ensemble import create_dummy_ensemble, create_ensemble, estimate_vi
from madi.fitters import bayes_fit
from madi.library import (
    LibraryEntry,
    _cell_widths,
    build_library_from_triplets,
    edge_railing_diagnostics,
    make_remediation_log_grid,
)
from madi.signal import G_from_b
from madi.walker_gpu import (
    PAPER_WALKS_PER_ENTRY,
    _membrane_transition_counts,
    pp_to_kio_eq5,
    run_walk_Y,
)


pytestmark = pytest.mark.tier_a


def _cfg(**overrides) -> SimConfig:
    values = dict(
        L=60.0,
        grid_spacing=0.75,
        classifier_candidates=8,
        geometry_calibration_points=30_000,
        geometry_validation_points=30_000,
        geometry_sample_cells=24,
        geometry_vi_tolerance=0.005,
        n_walkers=512,
        n_ensembles=1,
        T_max_ms=12.0,
        small_deltas=[1.0, 2.0],
        big_deltas=[2.0, 4.0, 8.0],
        b_values=[0.0, 500.0, 1_000.0, 2_000.0],
        exchange_calibration_walkers=512,
        exchange_calibration_ms=12.0,
        exchange_calibration_response_points=4,
        exchange_calibration_min_events=3,
        exchange_calibration_max_batches=2,
    )
    values.update(overrides)
    return SimConfig(**values)


@lru_cache(maxsize=32)
def _ensemble(target_vi: float, rho: float):
    cfg = _cfg()
    return create_ensemble(rho, target_vi * 1e6 / rho, cfg, seed=20_260_803)


def _event_rate(telemetry: dict) -> tuple[float, float]:
    metrics = telemetry["metrics"]
    time_ms = float(np.sum(metrics[:, 5]))
    events = float(np.sum(metrics[:, 6]))
    if time_ms == 0.0:
        return 0.0, 0.0
    rate = 1000.0 * events / time_ms
    se = 1000.0 * np.sqrt(events) / time_ms if events else 0.0
    return rate, se


def _finite_lobe_signal(Y: np.ndarray, cfg: SimConfig, delta: float, Delta: float, b: float) -> float:
    jd = int(round(delta / cfg.h_ms))
    jD = int(round(Delta / cfg.h_ms))
    js = int(round((delta + Delta) / cfg.h_ms))
    dM = Y[:, jd, :] + Y[:, jD, :] - Y[:, js, :]
    phase = GAMMA_RAD * G_from_b(b, delta, Delta) * dM * 1e-9
    return float(np.mean(np.cos(phase)))


def _measure_planar_permeability_gaussian(
    cfg: SimConfig,
    pp: float,
    *,
    n_samples: int = 2_000_000,
    slab_width_um: float = 1.0,
    seed: int = 20_260_820,
) -> tuple[float, float]:
    """Independent one-step flat-membrane flux measurement [um/ms].

    Starts are uniform in a slab on the intracellular side and only one
    Gaussian proposal is made.  Thus the estimator measures accepted flux
    per concentration without using the sphere's residence-time code.
    """
    rng = np.random.default_rng(seed)
    accepted = 0
    remaining = int(n_samples)
    while remaining:
        n = min(100_000, remaining)
        x = rng.uniform(0.0, slab_width_um, size=n)
        dx = rng.normal(0.0, cfg.sigma, size=n)
        u = rng.uniform(0.0, 1.0, size=n)
        accepted += int(np.count_nonzero((x + dx < 0.0) & (u < pp)))
        remaining -= n
    probability = accepted / n_samples
    permeability = slab_width_um * probability / cfg.ts
    se = slab_width_um * np.sqrt(max(probability * (1.0 - probability), 0.0) / n_samples) / cfg.ts
    return float(permeability), float(se)


def _measure_single_sphere_first_exit(
    cfg: SimConfig,
    radius_um: float,
    pp: float,
    *,
    n_walkers: int = 4096,
    T_ms: float = 16.0,
    seed: int = 20_260_821,
) -> tuple[float, float, float]:
    """Tagged first-exit sphere control using the production rejection rule."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n_walkers, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    position = direction * (radius_um * rng.uniform(size=n_walkers) ** (1.0 / 3.0))[:, None]
    alive = np.ones(n_walkers, dtype=bool)
    risk_time = 0.0
    exits = 0
    survivors = [n_walkers]
    steps = int(round(T_ms / cfg.ts))
    for step in range(steps):
        risk_time += float(np.count_nonzero(alive)) * cfg.ts
        increment = rng.normal(0.0, cfg.sigma, size=(n_walkers, 3))
        acceptance = rng.uniform(0.0, 1.0, size=n_walkers)
        proposal = position + increment
        outside = np.sum(proposal * proposal, axis=1) > radius_um ** 2
        first_exit = alive & outside & (acceptance < pp)
        exits += int(np.count_nonzero(first_exit))
        alive[first_exit] = False
        # Rejection is a literal reversion; accepted walkers are no longer
        # part of the tagged-starting-cell risk set.
        move = alive & ~outside
        position[move] = proposal[move]
        if (step + 1) % cfg.steps_per_h == 0:
            survivors.append(int(np.count_nonzero(alive)))
    event_rate = 1000.0 * exits / risk_time
    event_se = 1000.0 * np.sqrt(exits) / risk_time
    counts = np.asarray(survivors, dtype=float)
    t = np.arange(len(counts)) * cfg.h_ms
    keep = counts > 0.1 * n_walkers
    fit_rate = -1000.0 * np.polyfit(t[keep], np.log(counts[keep] / n_walkers), 1)[0]
    return float(event_rate), float(event_se), float(fit_rate)


@pytest.mark.parametrize("rho", [1.0e5, 5.0e5, 1.0e6])
@pytest.mark.parametrize("target_vi", [0.40, 0.70, 0.90, 0.99])
def test_p0a_realised_geometry_matches_every_requested_vi(target_vi: float, rho: float) -> None:
    """P0-A: no clamp; direct independent volume sampling validates labels."""
    ens = _ensemble(target_vi, rho)
    independent = estimate_vi(ens, n=80_000, seed=20_260_804)
    independent_se = np.sqrt(independent * (1.0 - independent) / 80_000)
    # 0.005 is the declared construction tolerance; the additional 4 SE is
    # solely the independent-volume-sampling uncertainty of this test.
    assert abs(independent - target_vi) <= ens.geometry.vi_se * 4.0 + 0.005 + independent_se * 4.0
    assert np.isclose(ens.rho * ens.V * 1e-6, ens.vi, rtol=0.0, atol=1e-12)
    assert ens.geometry.n_geometry_cells >= 0.90 * min(24, ens.geometry.n_primary_cells)
    assert ens.mean_AV > 0.0
    assert ens.geometry.annulus_min_um >= 0.0


def test_p0a_high_vi_targets_have_distinct_realised_annuli() -> None:
    """P0-A: the old 0.784 clamp cannot collapse 0.80--0.99 geometry."""
    ensembles = [_ensemble(v, 5.0e5) for v in (0.80, 0.85, 0.90, 0.94, 0.99)]
    alphas = np.array([e.alpha_star for e in ensembles])
    assert np.all(np.diff(alphas) < 0.0)
    assert np.unique(np.round(alphas, 8)).size == len(alphas)
    assert np.unique(np.round([e.mean_AV for e in ensembles], 8)).size == len(ensembles)


@pytest.mark.parametrize("target_vi,rho", [(0.40, 1.0e5), (0.70, 5.0e5), (0.99, 1.0e6)])
def test_p0d_candidate_classifier_meets_exact_kdtree_error_budget(target_vi: float, rho: float) -> None:
    """P0-D: cache errors are bounded against exact periodic KD labels."""
    ens = _ensemble(target_vi, rho)
    cfg = _cfg()
    rng = np.random.default_rng(20_260_805)
    p0 = rng.uniform(0.0, cfg.L, size=(60_000, 3))
    p1 = np.mod(p0 + rng.normal(0.0, cfg.sigma, size=p0.shape), cfg.L)
    exact0 = ens.classify_exact_cpu(p0)
    exact1 = ens.classify_exact_cpu(p1)
    cache0 = ens.classify_cpu(p0)
    cache1 = ens.classify_cpu(p1)
    compartment_error = np.mean(exact0[1] != cache0[1])
    exact_m = _membrane_transition_counts(*exact0, *exact1)
    cache_m = _membrane_transition_counts(*cache0, *cache1)
    attempted = max(int(np.count_nonzero(exact_m)), 1)
    crossing_error = np.count_nonzero(exact_m != cache_m) / attempted
    assert compartment_error < 5e-4
    assert crossing_error < 0.02


@pytest.mark.parametrize("target_vi,rho", [(0.40, 1.0e5), (0.70, 5.0e5), (0.99, 1.0e6)])
def test_p0d_cache_and_exact_walks_agree_on_direct_kio(target_vi: float, rho: float) -> None:
    """P0-D: classifier approximation changes direct k_io below MC error."""
    ens = _ensemble(target_vi, rho)
    cfg = _cfg(T_max_ms=4.0, small_deltas=[1.0], big_deltas=[2.0, 3.0], n_walkers=256)
    _, _, cached = run_walk_Y(ens, 0.0, cfg, pp=0.20, seed=20_260_806,
                              verbose=False, return_telemetry=True, classifier="cache")
    _, _, exact = run_walk_Y(ens, 0.0, cfg, pp=0.20, seed=20_260_806,
                             verbose=False, return_telemetry=True, classifier="exact")
    rc, sec = _event_rate(cached)
    re, see = _event_rate(exact)
    assert re > 0.0
    assert abs(rc - re) <= 4.0 * np.hypot(sec, see) + 0.5


def test_p0b_tagged_starting_cell_rate_is_linear_and_matches_survival_fit() -> None:
    """P0-B: low-p first-exit rate is linear over a decade within MC error."""
    ens = _ensemble(0.70, 5.0e5)
    cfg = _cfg(
        T_max_ms=8.0, small_deltas=[1.0], big_deltas=[2.0, 4.0, 6.0],
        n_walkers=512,
    )
    rates = []
    ses = []
    # This is the asymptotic regime in which a first-order permeability law
    # is expected.  The production response curve still measures/inverts the
    # full non-linear response rather than extrapolating this check to high p.
    pps = np.array((4.0e-4, 1.26491106e-3, 4.0e-3))
    for pp in pps:
        _, _, telemetry = run_walk_Y(
            ens, 0.0, cfg, pp=pp, seed=20_260_807, verbose=False,
            return_telemetry=True,
        )
        event, se = _event_rate(telemetry)
        counts = np.asarray(telemetry["start_survivor_fraction"]) * cfg.n_walkers
        initial = int(round(counts[0]))
        # Reconstruct the same fit independently, rather than trusting a
        # library label.  It validates the two requested estimators.
        t = np.arange(len(counts)) * cfg.h_ms
        frac = counts / max(initial, 1)
        keep = frac > 0.1
        fit = max(0.0, -1000.0 * np.polyfit(t[keep], np.log(frac[keep]), 1)[0])
        assert abs(event - fit) <= max(0.25 * event, 4.0 * se + 1.0)
        assert event > 0.0
        rates.append(event)
        ses.append(se)
    rates = np.asarray(rates)
    ses = np.asarray(ses)
    # Weighted fit through the physical origin.  The residual test, rather
    # than comparing three noisy rate/p ratios, explicitly reports the
    # Monte-Carlo tolerance of the one-decade first-order check.
    slope = np.sum(pps * rates / ses**2) / np.sum(pps**2 / ses**2)
    reduced_chi2 = np.sum(((rates - slope * pps) / ses) ** 2) / (len(pps) - 1)
    assert reduced_chi2 < 6.0


def test_p0b_single_sphere_matches_independently_measured_planar_permeability() -> None:
    """P0-B: a sphere gives k_io = (3/r) P_W using independent flux data."""
    cfg = _cfg(T_max_ms=16.0, small_deltas=[1.0], big_deltas=[2.0, 4.0, 8.0])
    radius_um = 10.0
    pp = 0.002
    P_w, P_w_se = _measure_planar_permeability_gaussian(cfg, pp)
    event, event_se, fit = _measure_single_sphere_first_exit(cfg, radius_um, pp)
    expected = 1000.0 * 3.0 * P_w / radius_um
    expected_se = 1000.0 * 3.0 * P_w_se / radius_um
    assert abs(event - fit) <= 4.0 * event_se + 0.15 * event
    assert abs(event - expected) <= 4.0 * np.hypot(event_se, expected_se) + 0.15 * expected


def test_p0b_measured_to_eq5_ratio_is_stored_as_a_smooth_geometry_function() -> None:
    """P0-B: Eq. 5 is a bounded comparator, not a hidden global label scale."""
    cfg = _cfg(T_max_ms=6.0, small_deltas=[1.0], big_deltas=[2.0, 4.0], n_walkers=256)
    ratios = []
    for target_vi in (0.40, 0.70, 0.99):
        ens = create_ensemble(5.0e5, target_vi * 2.0, cfg, seed=20_260_822)
        _, _, telemetry = run_walk_Y(
            ens, 0.0, cfg, pp=0.004, seed=20_260_823,
            verbose=False, return_telemetry=True,
        )
        measured, _ = _event_rate(telemetry)
        analytic = pp_to_kio_eq5(0.004, ens.mean_AV, cfg)
        assert measured > 0.0 and analytic > 0.0
        ratios.append(measured / analytic)
    ratios = np.asarray(ratios)
    # The direct label and Eq. 5 agree to a finite, geometry-dependent factor
    # in this smoke grid.  The retained per-entry response table—not a global
    # correction—is what captures that factor in production.
    assert np.all((ratios > 0.35) & (ratios < 2.5))
    assert ratios.max() / ratios.min() < 3.0


def test_p0b_library_labels_direct_start_cell_rate_and_retains_both_comparators() -> None:
    """P0-B: entry labels come from actual exits, never a target/lookup value."""
    cfg = _cfg(
        L=40.0, geometry_calibration_points=12_000,
        geometry_validation_points=12_000, geometry_sample_cells=16,
        T_max_ms=6.0, small_deltas=[1.0], big_deltas=[2.0, 4.0],
        b_values=[0.0, 1_000.0], n_walkers=192,
        exchange_calibration_walkers=384, exchange_calibration_ms=6.0,
    )
    lib = build_library_from_triplets(
        [(0.0, 5.0e5, 1.4), (10.0, 5.0e5, 1.4), (30.0, 5.0e5, 1.4)],
        cfg=cfg, seed=20_260_824, verbose=False,
    )
    cellular = [entry for entry in lib if entry.kio_nominal and entry.kio_nominal > 0.0]
    assert len(cellular) == 2
    for entry in cellular:
        exchange = entry.metadata["exchange"]
        calibration = exchange["calibration"][0]
        response = calibration["response"]
        assert entry.kio == pytest.approx(exchange["kio_measured_s_inv"])
        assert exchange["kio_survival_fit_s_inv"] >= 0.0
        assert exchange["kio_analytic_eq5_s_inv"] > 0.0
        assert response["pp_values"][0] == 0.0
        assert response["pp_values"][-1] == 1.0
        assert len(response["event_rates_s_inv"]) == len(response["pp_values"])


@pytest.mark.parametrize("pp", [0.0, 0.01, 0.05])
def test_p0b_symmetric_pp_preserves_intracellular_occupancy(pp: float) -> None:
    """P0-B: detailed balance holds across p_p, not at one hand-picked point."""
    ens = _ensemble(0.70, 5.0e5)
    cfg = _cfg(T_max_ms=8.0, small_deltas=[1.0], big_deltas=[2.0, 4.0, 6.0], n_walkers=256)
    _, n_escape, telemetry = run_walk_Y(
        ens, 0.0, cfg, pp=pp, seed=20_260_808, verbose=False,
        return_telemetry=True,
    )
    occ = np.asarray(telemetry["occupancy_fraction"])
    se = np.sqrt(ens.vi * (1.0 - ens.vi) / cfg.n_walkers)
    assert n_escape == 0
    assert np.max(np.abs(occ - ens.vi)) <= 5.0 * se + 0.01


def test_p0c_periodic_walk_has_no_escape_and_exposes_legacy_drop_bias() -> None:
    """P0-C: periodic production removes escape; legacy selection is measurable."""
    base = _cfg(
        L=30.0,
        buffer=0.0,
        grid_spacing=1.0,
        T_max_ms=16.0,
        small_deltas=[2.0],
        big_deltas=[8.0],
        b_values=[0.0, 1_000.0],
        n_walkers=512,
    )
    dummy = create_dummy_ensemble(base)
    y_periodic, escaped_periodic = run_walk_Y(
        dummy, 0.0, base, pp=0.0, seed=20_260_809, verbose=False
    )
    legacy = replace(base, boundary_mode="absorbing_legacy")
    y_legacy, escaped_legacy = run_walk_Y(
        dummy, 0.0, legacy, pp=0.0, seed=20_260_809, verbose=False
    )
    s_periodic = _finite_lobe_signal(y_periodic, base, 2.0, 8.0, 1_000.0)
    s_legacy = _finite_lobe_signal(y_legacy, legacy, 2.0, 8.0, 1_000.0)
    dm_periodic = y_periodic[:, 2, :] + y_periodic[:, 8, :] - y_periodic[:, 10, :]
    dm_legacy = y_legacy[:, 2, :] + y_legacy[:, 8, :] - y_legacy[:, 10, :]
    assert escaped_periodic == 0
    assert escaped_legacy > 0
    # Escape/drop-all conditions on a large unwrapped motion history.  It
    # narrows the finite-lobe phase-moment distribution; the signal direction
    # at one noisy b shell need not be monotone, but the selection bias is
    # unambiguous in the underlying moment variance.
    assert np.mean(dm_legacy ** 2) < 0.90 * np.mean(dm_periodic ** 2)
    assert not np.isclose(s_legacy, s_periodic, rtol=0.0, atol=1e-12)


def test_p0c_drop_all_library_can_select_a_low_vi_high_exchange_wrong_entry() -> None:
    """P0-C causal test: legacy selection biases a fixed synthetic decay.

    The old implementation froze escaped walkers and then removed them from
    *all* encoding columns.  Here a periodic synthetic voxel from the middle
    of the geometry box is compared with a deliberately small legacy library.
    The legacy argmin does not return the generating geometry: it picks a
    lower-v_i, lower-volume, higher-permeability candidate.  This is a
    compact, fixed-seed reproduction of the map-level bias hypothesis; it
    does not depend on CUDA or on the production artifact.
    """
    base = _cfg(
        L=30.0,
        buffer=0.0,
        grid_spacing=0.75,
        geometry_calibration_points=4_000,
        geometry_validation_points=4_000,
        geometry_sample_cells=10,
        geometry_vi_tolerance=0.016,
        n_walkers=64,
        T_max_ms=8.0,
        small_deltas=[1.0],
        big_deltas=[4.0],
        b_values=[0.0, 500.0, 1_000.0],
    )
    legacy = replace(base, boundary_mode="absorbing_legacy", max_escape_frac=1.0)

    def decay_and_rate(ens, cfg, pp):
        Y, escaped, telemetry = run_walk_Y(
            ens, 0.0, cfg, pp=pp, seed=222,
            verbose=False, return_telemetry=True,
        )
        dM = Y[:, 1, :] + Y[:, 4, :] - Y[:, 5, :]
        signal = []
        for b in base.b_values:
            phase = GAMMA_RAD * G_from_b(float(b), 1.0, 4.0) * dM * 1e-9
            signal.append(float(np.mean(np.cos(phase))))
        rate, _ = _event_rate(telemetry)
        return np.asarray(signal), int(escaped), rate

    # Generating periodic voxel: nominal v_i=0.70 and modest permeability.
    generating = create_ensemble(5.0e5, 1.4, base, seed=111)
    synthetic, n_escape_fixed, kio_fixed = decay_and_rate(generating, base, 0.05)
    assert n_escape_fixed == 0

    # Candidate 0 is the same physical entry under the broken boundary
    # handling; candidate 1 is a lower-rho, lower-v_i/high-exchange
    # alternative.  The latter becomes spuriously closer after
    # escape-selection removes the high-motion walkers from every column.
    candidate_specs = [
        (5.0e5, 1.4, 0.05),
        (3.0e5, 1.333, 0.20),
    ]
    broken = []
    for rho, volume, pp in candidate_specs:
        ens = create_ensemble(rho, volume, base, seed=111)
        vector, escaped, rate = decay_and_rate(ens, legacy, pp)
        broken.append((ens, vector, escaped, rate))
    residuals = np.asarray([np.sum((vector - synthetic) ** 2)
                            for _, vector, _, _ in broken])
    best = int(np.argmin(residuals))
    best_ens, _, best_escaped, best_kio = broken[best]

    assert all(item[2] > 0 for item in broken)
    assert best == 1
    assert best_ens.rho < generating.rho
    assert best_ens.vi < generating.vi - 0.20
    assert best_ens.V < generating.V
    assert best_kio > kio_fixed
    # The wrong candidate is decisively preferred, not a Monte-Carlo tie.
    assert residuals[best] < 0.10 * residuals[0]


def test_p0e_log_grid_weights_and_free_water_atom_are_explicit() -> None:
    """P0-E: transformed grid carries analytic quadrature weights."""
    grid = make_remediation_log_grid(n_rho=9, n_V=10, kios=np.array([0.0, 1.0, 5.0, 20.0, 80.0, 130.0]))
    triplets, weights = grid.triplets_and_weights()
    assert triplets[0] == (0.0, 0.0, 0.0)
    assert weights[("free_water", 0.0, 0.0)] > 0.0
    cellular = np.asarray([x for x in triplets if x[1] > 0.0])
    vi = cellular[:, 1] * cellular[:, 2] * 1e-6
    assert np.all((vi >= 0.40) & (vi <= 0.99))
    assert np.isclose(grid.rhos[0], 1.0e4)
    assert np.isclose(grid.rhos[-1], 1.0e7)
    assert np.isclose(grid.Vs[0], 0.01)
    assert np.isclose(grid.Vs[-1], 200.0)
    assert np.all(np.diff(np.log(grid.rhos)) > 0.0)
    assert np.all(np.diff(np.log(grid.Vs)) > 0.0)
    assert all(weights[(round(k, 4), round(r, 1), round(v, 6))] > 0.0
               for k, r, v in triplets[1:])


def test_p0e_quadrature_does_not_assign_mass_outside_declared_grid_bounds() -> None:
    """P0-E: first/last nodes are truncated at physical grid bounds."""
    widths = _cell_widths(
        np.asarray([0.0, 1.0, 5.0]), lower_bound=0.0, upper_bound=5.0,
    )
    assert np.allclose(widths, [0.5, 2.5, 2.0])
    assert np.isclose(widths.sum(), 5.0)


def test_p0e_bayesian_mode_rejects_unweighted_library() -> None:
    """P0-E: counting measure can no longer silently become a Bayesian prior."""
    library = [
        LibraryEntry(1.0, 1.0e6, 0.6, np.array([1.0, 0.8])),
        LibraryEntry(2.0, 1.0e6, 0.7, np.array([1.0, 0.7])),
    ]
    with pytest.raises(ValueError, match="quadrature weights"):
        bayes_fit(
            np.array([[1.0, 0.8]]), library, sigma_m=0.05,
            lib_delta_pairs=[(1.0, 2.0)], lib_b_values=[0.0, 500.0], n_b=2,
            fit_triples=[(1.0, 2.0, 0.0), (1.0, 2.0, 500.0)],
            vi_min=0.4, vi_max=0.99, use_gpu=False,
        )


def test_p0e_edge_railing_separates_free_water_from_cellular_boundaries() -> None:
    """P0-E: the free-water atom is not misreported as a rho/V floor rail."""
    library = [
        LibraryEntry(float("nan"), 0.0, 0.0, np.array([1.0, 0.9]),
                     vi=0.0, is_free_water=True),
        LibraryEntry(2.0, 1.0e4, 40.0, np.array([1.0, 0.8]), vi=0.4),
        LibraryEntry(30.0, 1.0e6, 0.7, np.array([1.0, 0.6]), vi=0.7),
        LibraryEntry(130.0, 1.0e7, 0.099, np.array([1.0, 0.4]), vi=0.99),
    ]
    diag = edge_railing_diagnostics(
        np.array([np.nan, 2.0, 130.0]),
        np.array([0.0, 1.0e4, 1.0e7]),
        np.array([0.0, 40.0, 0.099]),
        library, vi_min=0.4, vi_max=0.99, include_free_water=True,
    )
    assert diag["free_water_count"] == 1
    assert diag["rho"]["at_lower_count"] == 1
    assert diag["rho"]["at_upper_count"] == 1
    assert diag["kio"]["at_lower_count"] == 1
    assert diag["kio"]["at_upper_count"] == 1


def test_p0a_small_rebuild_has_no_duplicate_vectors_across_geometry_labels() -> None:
    """P0-A: formerly clamped high-v_i labels cannot produce tied vectors."""
    cfg = _cfg(
        L=50.0, geometry_calibration_points=16_000,
        geometry_validation_points=16_000, geometry_sample_cells=20,
        T_max_ms=8.0, small_deltas=[1.0], big_deltas=[2.0, 4.0],
        b_values=[0.0, 500.0, 1_000.0, 2_000.0], n_walkers=384,
    )
    triplets = [(0.0, 5.0e5, vi * 1e6 / 5.0e5) for vi in (0.70, 0.82, 0.90, 0.99)]
    lib = build_library_from_triplets(triplets, cfg=cfg, seed=20_260_810, verbose=False)
    vectors = np.asarray([entry.vector for entry in lib])
    pairwise = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    min_dist = float(np.min(pairwise[np.triu_indices(len(lib), 1)]))
    # This smoke library deliberately has only 1,152 axis-walks, so its raw
    # vector variance is not the production criterion.  At the production
    # design's 12M axis-walks, the bounded-cos conservative L2 MC scale is
    # sqrt(n_columns/N). The full held-out-ensemble estimator is exercised by
    # P2-M; this P0 test guards the former exact (zero-distance) geometry tie.
    production_bound = np.sqrt(vectors.shape[1] / PAPER_WALKS_PER_ENTRY)
    assert min_dist > 10.0 * production_bound

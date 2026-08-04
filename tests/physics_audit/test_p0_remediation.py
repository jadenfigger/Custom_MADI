"""Tier-A P0 checks retained after the MADI I SI clarification."""

from __future__ import annotations

from dataclasses import fields, replace
import inspect

import numpy as np
import pytest

from madi.config import SimConfig
from madi.ensemble import create_dummy_ensemble, si_domain_side_um
from madi.fitters import bayes_fit
from madi.library import (
    LibraryEntry,
    _cell_widths,
    _save_library,
    edge_railing_diagnostics,
    make_remediation_log_grid,
)
from madi import signal as sig
from madi.walker_gpu import (
    ReducedResult,
    WalkRandomStream,
    kio_to_pp,
    pp_to_kio_eq5,
    run_simulation_multi_kio_reduced,
    run_simulation_reduced,
    run_walk_Y,
)

from ._si_test_support import si_test_config


pytestmark = pytest.mark.tier_a


def test_eq5_inverse_uses_operational_rms_step_and_rejects_invalid_pp(tmp_path) -> None:
    """P0-B: Eq. 5 is the sole k_io calibration and never silently clamps."""
    cfg = si_test_config(tmp_path)
    mean_av = 0.50
    pp = 0.005
    kio = pp_to_kio_eq5(pp, mean_av, cfg)
    assert kio == pytest.approx(pp * np.sqrt(3.0 / 0.006) * mean_av * 1000.0)
    assert kio_to_pp(kio, mean_av, cfg) == pytest.approx(pp)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        kio_to_pp(1.0e7, mean_av, cfg)


def test_direct_residence_time_calibration_api_and_storage_are_absent(tmp_path) -> None:
    """The two withdrawn P0-B additions cannot contaminate new libraries."""
    assert "calibrate_exchange" not in inspect.signature(run_simulation_reduced).parameters
    assert "calibrate_exchange" not in inspect.signature(run_simulation_multi_kio_reduced).parameters
    assert "kio_measured_se" not in {field.name for field in fields(LibraryEntry)}

    cfg = si_test_config(tmp_path)
    out = tmp_path / "small_library.npz"
    _save_library(
        [LibraryEntry(10.0, 5.0e5, 1.4, np.asarray([1.0, 0.7]), vi=0.70)],
        str(out), cfg=cfg, columns=sig.build_columns(cfg),
    )
    with np.load(out, allow_pickle=False) as data:
        assert "kio_measured_se" not in data.files


def test_s8_boundary_escape_is_a_hard_error_not_survivor_selection() -> None:
    """P0-C/SI §S.III: any exit aborts; no per-column survival path exists."""
    cfg = SimConfig(
        L=2.0, n_walkers=1, n_ensembles=1, T_max_ms=2.0,
        small_deltas=[1.0], big_deltas=[1.0], b_values=[0.0],
    )
    steps = cfg.n_steps
    increments = np.zeros((steps, 1, 3), dtype=np.float64)
    increments[0, 0, 0] = 2.0
    stream = WalkRandomStream(
        initial_positions=np.asarray([[1.0, 1.0, 1.0]]),
        increments=increments,
        acceptance_uniforms=np.zeros((steps, 1), dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="fatal boundary condition"):
        run_walk_Y(
            create_dummy_ensemble(cfg), 0.0, cfg, pp=0.0, verbose=False,
            random_stream=stream, use_gpu=False,
        )


def test_successful_signal_payload_has_no_survivor_column_bookkeeping(tmp_path) -> None:
    """P0-C/SI §S.III: a successful entry has no escape-selection payload."""
    cfg = si_test_config(tmp_path)
    columns = sig.build_columns(cfg)
    result = ReducedResult(
        cos_sum=np.ones(columns.n_pairs * columns.n_b),
        sin_sum=np.zeros(columns.n_pairs * columns.n_b),
        n_walkers=8,
        n_escaped=0,
        occupancy_counts=np.zeros(cfg.n_grid, dtype=np.int64),
    )
    payload = sig._assemble(result, columns)
    assert payload["n_escaped"] == 0
    assert "surviving_walkers_by_checkpoint" not in payload


def test_s8_128ms_source_domain_has_no_free_water_escape() -> None:
    """P0-C Tier A: the prescribed 128-ms S8 domain contains its source walk."""
    cfg = SimConfig(
        L=None, n_walkers=128, n_ensembles=1, T_max_ms=128.0,
        small_deltas=[20.0], big_deltas=[50.0], b_values=[0.0],
    )
    ensemble = create_dummy_ensemble(cfg)
    assert ensemble.L == pytest.approx(554.2562584220407)
    assert ensemble.source_hi - ensemble.source_lo == pytest.approx(0.4 * ensemble.L)
    _, escaped = run_walk_Y(
        ensemble, 0.0, cfg, pp=0.0, seed=20_260_930,
        verbose=False, use_gpu=False,
    )
    assert escaped == 0


def test_s8_high_density_128ms_source_domain_has_no_free_water_escape() -> None:
    """P0-C Tier A: exercise the smallest S8 domain in the P0-E rho range.

    This deliberately uses a free-water control in the Eq. S8 box calculated
    for rho=1e7 cells/uL.  It isolates boundary geometry from membrane
    classification while covering the largest free-water escape risk in the
    planned dense grid.  Any escape is fatal, never a survivor-selection path.
    """
    base = SimConfig(
        L=None, n_walkers=512, n_ensembles=1, T_max_ms=128.0,
        small_deltas=[20.0], big_deltas=[50.0], b_values=[0.0],
    )
    side = si_domain_side_um(1.0e7, base)
    cfg = replace(base, L=side)
    ensemble = create_dummy_ensemble(cfg)
    assert side == pytest.approx(430.8869380063766)
    assert ensemble.source_lo == pytest.approx(0.3 * side)
    assert ensemble.source_hi == pytest.approx(0.7 * side)
    _, escaped = run_walk_Y(
        ensemble, 0.0, cfg, pp=0.0, seed=20_260_803,
        verbose=False, use_gpu=False,
    )
    assert escaped == 0


def test_p0e_log_grid_weights_and_free_water_atom_are_explicit() -> None:
    """P0-E survives unchanged: masked log grid carries analytic weights."""
    grid = make_remediation_log_grid(
        n_rho=9, n_V=10, kios=np.asarray([0.0, 1.0, 5.0, 20.0, 80.0, 130.0])
    )
    triplets, weights = grid.triplets_and_weights()
    assert triplets[0] == (0.0, 0.0, 0.0)
    assert weights[("free_water", 0.0, 0.0)] > 0.0
    cellular = np.asarray([row for row in triplets if row[1] > 0.0])
    vi = cellular[:, 1] * cellular[:, 2] * 1e-6
    assert np.all((vi >= 0.40) & (vi <= 0.99))
    assert np.all(np.diff(np.log(grid.rhos)) > 0.0)
    assert np.all(np.diff(np.log(grid.Vs)) > 0.0)
    assert all(weights[(round(k, 4), round(r, 1), round(v, 6))] > 0.0
               for k, r, v in triplets[1:])


def test_pilot_grid_exercises_the_extended_weighted_domain() -> None:
    """Tier A: lock the restricted Sol pilot to its documented 25 entries."""
    grid = make_remediation_log_grid(
        n_rho=8,
        n_V=12,
        kios=np.asarray([0.0, 20.0, 130.0]),
        rho_min=1.0e4,
        rho_max=1.0e7,
        V_min=0.01,
        V_max=200.0,
        vi_min=0.40,
        vi_max=0.99,
    )
    triplets, _ = grid.triplets_and_weights()
    pairs = sorted({(rho, volume) for _, rho, volume in triplets if rho > 0.0})
    vis = np.asarray([rho * volume * 1e-6 for rho, volume in pairs])
    assert len(pairs) == 8
    assert len(triplets) == 25  # 8 pairs × 3 k_io values plus free water.
    assert np.min(vis) == pytest.approx(0.4282828134809622)
    assert np.max(vis) == pytest.approx(0.9664172475849011)
    assert [
        sum(
            (rho > 0.0 and pairs.index((rho, volume)) % 4 == shard)
            or (shard == 0 and rho == 0.0 and volume == 0.0)
            for _, rho, volume in triplets
        )
        for shard in range(4)
    ] == [7, 6, 6, 6]


def test_p0e_quadrature_respects_declared_grid_bounds() -> None:
    assert np.allclose(
        _cell_widths(np.asarray([0.0, 1.0, 5.0]), lower_bound=0.0, upper_bound=5.0),
        [0.5, 2.5, 2.0],
    )


def test_p0e_bayesian_mode_rejects_unweighted_library() -> None:
    library = [
        LibraryEntry(1.0, 1.0e6, 0.6, np.asarray([1.0, 0.8])),
        LibraryEntry(2.0, 1.0e6, 0.7, np.asarray([1.0, 0.7])),
    ]
    with pytest.raises(ValueError, match="quadrature weights"):
        bayes_fit(
            np.asarray([[1.0, 0.8]]), library, sigma_m=0.05,
            lib_delta_pairs=[(1.0, 2.0)], lib_b_values=[0.0, 500.0], n_b=2,
            fit_triples=[(1.0, 2.0, 0.0), (1.0, 2.0, 500.0)],
            vi_min=0.4, vi_max=0.99, use_gpu=False,
        )


def test_p0e_edge_railing_separates_free_water_from_cellular_bounds() -> None:
    library = [
        LibraryEntry(float("nan"), 0.0, 0.0, np.asarray([1.0, 0.9]), vi=0.0, is_free_water=True),
        LibraryEntry(2.0, 1.0e4, 40.0, np.asarray([1.0, 0.8]), vi=0.4),
        LibraryEntry(30.0, 1.0e6, 0.7, np.asarray([1.0, 0.6]), vi=0.7),
        LibraryEntry(130.0, 1.0e7, 0.099, np.asarray([1.0, 0.4]), vi=0.99),
    ]
    diag = edge_railing_diagnostics(
        np.asarray([np.nan, 2.0, 130.0]),
        np.asarray([0.0, 1.0e4, 1.0e7]),
        np.asarray([0.0, 40.0, 0.099]),
        library, vi_min=0.4, vi_max=0.99, include_free_water=True,
    )
    assert diag["free_water_count"] == 1
    assert diag["rho"]["at_lower_count"] == 1
    assert diag["rho"]["at_upper_count"] == 1

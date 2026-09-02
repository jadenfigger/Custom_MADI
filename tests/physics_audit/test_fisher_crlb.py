"""Focused, synthetic tests for the Phase-0/1 Fisher framework."""
from __future__ import annotations

import numpy as np

from madi.fisher_crlb import (analytic_free_water_gate, build_grid_manifest, canonical_grid, derivative_variance,
                               feasibility_masks, fisher_diagnostics, fisher_matrix,
                               invalidated_stencils)


def test_missing_canonical_group_is_reported_and_invalidates_neighbours() -> None:
    rhos, volumes, _, retained = canonical_grid()
    # Choose a pair with four rho-neighbours inside the retained band.
    centre = next((ir, iv) for ir, iv in retained
                  if all((candidate, iv) in retained
                         for candidate in (ir - 2, ir - 1, ir + 1, ir + 2)))
    rows_rho, rows_V, free = [], [], []
    for ir, iv in retained - {centre}:
        rows_rho.extend([rhos[ir]] * 51)
        rows_V.extend([volumes[iv]] * 51)
        free.extend([False] * 51)
    manifest = build_grid_manifest(np.asarray(rows_rho), np.asarray(rows_V), np.asarray(free))
    assert centre in manifest.missing
    invalid = invalidated_stencils(manifest, {"rho": [1]})["rho"]["1"]
    assert any(row["rho_index"] == centre[0] - 1 and row["V_index"] == centre[1] for row in invalid)
    assert any(row["rho_index"] == centre[0] + 1 and row["V_index"] == centre[1] for row in invalid)


def test_central_variance_uses_aligned_crn_covariance() -> None:
    minus = np.asarray([0.1, 0.2, 0.3, 0.4])[:, None]
    plus = minus + 0.2
    result = derivative_variance(np.asarray([np.var(minus, ddof=1)]),
                                 np.asarray([np.var(plus, ddof=1)]), minus, plus,
                                 denominator=2.0, n_ensembles=4)
    assert np.allclose(result, 0.0, atol=1e-15)


def test_debiased_fisher_leaves_off_diagonal_unchanged() -> None:
    J = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    variance = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    raw = fisher_matrix(J, 2.0)
    corrected = fisher_matrix(J, 2.0, variance)
    assert corrected[0, 1] == raw[0, 1]
    assert corrected[1, 0] == raw[1, 0]
    assert corrected[0, 0] < raw[0, 0]
    diag = fisher_diagnostics(np.eye(3), 20.0)
    assert np.allclose(diag["kappa"], 1.0)


def test_feasibility_filters_are_hard_boolean_masks() -> None:
    signals = np.asarray([[1.0, 0.01], [1.0, 0.02]])
    masks = feasibility_masks(signals, np.asarray([10.0]), np.asarray([30.0]), np.asarray([0.0, 1000.0]),
                              sigma0=0.01, G_max=1.0)
    assert masks["trust_floor_per_entry"].dtype == bool
    assert not masks["trust_floor_per_entry"][0, 1]
    assert masks["combined_per_entry"].shape == signals.shape


def test_analytic_gate_is_synthetic_and_uses_the_predeclared_tolerance() -> None:
    report = analytic_free_water_gate(np.arange(0.0, 12_500.0, 500.0), tolerance=1e-12)
    assert report["pass"]
    assert report["derivative_max_abs_error"] == 0.0

"""Small, explicitly non-production fixtures for SI geometry unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from madi.config import SimConfig
from madi.ensemble import alpha_star_from_vi


def write_uncertified_reference(directory: Path) -> Path:
    """Create a schema-valid 26-row fixture, never a production reference.

    Geometry construction needs the reference only for the Eq.-5 permeability
    factor; these unit tests exercise its loader/scaling and never claim its
    deliberately tiny synthetic cloud is a scientific calibration artifact.
    """
    path = directory / "si_geometry_reference_test_fixture.npz"
    if path.exists():
        return path
    vi = np.linspace(0.40, 1.00, 26, dtype=np.float64)
    alpha_x = np.asarray([
        0.0 if value >= 1.0 else alpha_star_from_vi(value, 1.0e9)
        for value in vi
    ])
    metadata = {
        "schema": "madi-si-geometry-test-fixture-v1",
        "source": "unit-test fixture; not a certified SI reference cloud",
        "rho_reference": 1.0,
        "kappa": 0.90,
        "n_single_cells": 16,
        "n_alpha": 26,
        "alpha_spacing": "linear",
        "mean_estimator": "untrimmed_arithmetic_mean_A_over_V",
        "contraction_rule": "all_shifted_voronoi_facets_S2",
    }
    np.savez(
        path,
        vi=vi,
        alpha_x=alpha_x,
        mean_A_over_V_norm=4.0 - 2.0 * vi,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return path


def si_test_config(tmp_path: Path, **overrides) -> SimConfig:
    values = dict(
        L=80.0,
        geometry_reference_path=str(write_uncertified_reference(tmp_path)),
        allow_uncertified_geometry_reference=True,
        geometry_validation_points=8_000,
        geometry_vi_tolerance=0.04,
        n_walkers=96,
        n_ensembles=1,
        T_max_ms=4.0,
        small_deltas=[1.0],
        big_deltas=[2.0, 3.0],
        b_values=[0.0, 500.0, 1_000.0],
    )
    values.update(overrides)
    return SimConfig(**values)

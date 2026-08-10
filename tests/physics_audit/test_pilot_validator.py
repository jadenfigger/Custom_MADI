"""Tier-A self-test for the pre-written Sol remediation-pilot validator."""

from __future__ import annotations

import numpy as np

from madi.config import SimConfig
from madi.library import (
    LibraryEntry,
    _save_library,
    ensemble_mean_subset_column_indices,
    make_remediation_log_grid,
)
from madi import signal as sig
from scripts.validate_remediation_pilot import (
    PILOT_B_VALUES,
    PILOT_KIOS,
    PILOT_N_RHO,
    PILOT_N_ENSEMBLES,
    PILOT_N_SHARDS,
    PILOT_N_V,
    PILOT_PAIRS,
    PILOT_BIG_DELTAS,
    PILOT_RHO_MAX,
    PILOT_RHO_MIN,
    PILOT_SMALL_DELTAS,
    PILOT_VI_MAX,
    PILOT_VI_MIN,
    PILOT_V_MAX,
    PILOT_V_MIN,
    validate,
)


def _pilot_grid():
    return make_remediation_log_grid(
        n_rho=PILOT_N_RHO, n_V=PILOT_N_V, kios=PILOT_KIOS,
        rho_min=PILOT_RHO_MIN, rho_max=PILOT_RHO_MAX,
        V_min=PILOT_V_MIN, V_max=PILOT_V_MAX,
        vi_min=PILOT_VI_MIN, vi_max=PILOT_VI_MAX,
    )


def test_pilot_validator_accepts_a_schema_valid_complete_fixture(tmp_path) -> None:
    """The future returned pilot has a local, deterministic acceptance path."""
    cfg = SimConfig(
        T_max_ms=128.0, n_walkers=512, n_ensembles=PILOT_N_ENSEMBLES,
        geometry_reference_path="data/geometry_reference_si_kappa_0p9.npz",
        small_deltas=PILOT_SMALL_DELTAS, big_deltas=PILOT_BIG_DELTAS,
        b_values=PILOT_B_VALUES,
    )
    grid = _pilot_grid()
    triplets, weights = grid.triplets_and_weights()
    pairs = sorted(
        {(rho, volume) for _, rho, volume in triplets if rho > 0.0},
        key=lambda pair: pair[0] * pair[1],
    )
    columns = sig.build_columns(cfg)
    paths = []
    for shard_id in range(PILOT_N_SHARDS):
        entries = []
        for ordinal, (kio, rho, volume) in enumerate(triplets):
            selected = ((rho == 0.0 and volume == 0.0 and shard_id == 0)
                        or (rho > 0.0 and pairs.index((rho, volume)) % PILOT_N_SHARDS == shard_id))
            if not selected:
                continue
            matrix = np.tile(
                np.asarray([1.0, *[1.0 - b / 20000.0 for b in PILOT_B_VALUES[1:]]]),
                (len(PILOT_PAIRS), 1),
            )
            matrix[:, 1:] -= ordinal * 1e-5
            vector = matrix.ravel()
            ensemble_means = np.tile(vector, (PILOT_N_ENSEMBLES, 1)).astype(np.float32)
            is_free = rho == 0.0 and volume == 0.0
            vi = 0.0 if is_free else rho * volume * 1e-6
            metadata = {
                "imaginary_signal_check": {
                    "n_ensembles": PILOT_N_ENSEMBLES,
                    "degrees_of_freedom": PILOT_N_ENSEMBLES - 1,
                    "max_abs_standardized_deviation": 0.0,
                    "max_column": {
                        "delta_ms": PILOT_PAIRS[0][0],
                        "Delta_ms": PILOT_PAIRS[0][1],
                        "b_s_mm2": PILOT_B_VALUES[0],
                    },
                    "mean_imaginary_signal": 0.0,
                    "standard_error": 0.0,
                    "zero_standard_error_nonzero_count": 0,
                },
            }
            if not is_free:
                metadata.update({
                    "boundary": {"n_escaped": 0},
                    "realised_geometry": {"vi": vi},
                    "per_ensemble_geometry": [{"reference_metadata": {
                        "n_single_cells": 5_000_000,
                        "contraction_rule": "all_shifted_voronoi_facets_S2",
                        "kappa": 0.90,
                    }}],
                })
            entries.append(LibraryEntry(
                kio=float("nan") if is_free else kio,
                rho=rho, V=volume, vector=vector, vi=vi,
                signal_imag=np.zeros_like(vector, dtype=np.float32),
                signal_variance=np.var(ensemble_means, axis=0, ddof=1).astype(np.float32),
                ensemble_means_subset=ensemble_means[
                    :, ensemble_mean_subset_column_indices(columns)
                ],
                weight=weights[("free_water", 0.0, 0.0)] if is_free else weights[(round(kio, 4), round(rho, 1), round(volume, 6))],
                kio_nominal=kio, rho_nominal=rho, V_nominal=volume,
                pp=0.0 if is_free else 0.001,
                kio_analytic_eq5=kio,
                is_free_water=is_free,
                metadata=metadata,
            ))
        path = tmp_path / f"madi_remediation_pilot.shard{shard_id:03d}.npz"
        _save_library(entries, str(path), cfg=cfg, columns=columns, grid_metadata=grid.metadata())
        paths.append(path)

    report = validate(paths)
    assert report["pass"], report["errors"]
    assert report["total_entries"] == 25
    assert report["expected_shard_counts"] == {"0": 7, "1": 6, "2": 6, "3": 6}
    assert report["minimum_pairwise_l2"] > 0.0

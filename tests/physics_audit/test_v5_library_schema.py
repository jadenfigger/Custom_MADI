"""Focused Tier-A checks for the v5 Monte-Carlo diagnostic schema."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from madi import signal as sig
from madi.config import ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2, SimConfig
from madi.library import (
    ENSEMBLE_MEAN_SUBSET_N_COLUMNS,
    LibraryEntry,
    _save_library,
    _v5_diagnostics_from_result,
    ensemble_mean_subset_column_indices,
    make_remediation_log_grid,
)
from madi.walker_gpu import ReducedResult, _merge_results
from scripts.fit_data import load_remediation_entry_subset
from scripts.validate_remediation_pilot import PILOT_BIG_DELTAS, PILOT_SMALL_DELTAS


ROOT = Path(__file__).resolve().parents[2]


def _diagnostic_config(n_ensembles: int) -> SimConfig:
    return SimConfig(
        n_walkers=2,
        n_ensembles=n_ensembles,
        small_deltas=PILOT_SMALL_DELTAS,
        big_deltas=PILOT_BIG_DELTAS,
        b_values=list(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2),
    )


def test_between_ensemble_variance_is_sample_variance_and_subset_is_indexed() -> None:
    cfg = _diagnostic_config(n_ensembles=3)
    columns = sig.build_columns(cfg)
    n_columns = columns.n_pairs * columns.n_b
    ensemble_real = np.stack([
        np.full((columns.n_pairs, columns.n_b), value)
        for value in (0.2, 0.4, 0.8)
    ])
    result = {
        "ensemble_S": ensemble_real,
        "ensemble_S_imag": np.zeros_like(ensemble_real),
        "S_imag": np.zeros((columns.n_pairs, columns.n_b)),
    }

    signal_imag, variance, subset, imaginary_check = _v5_diagnostics_from_result(
        result, columns, cfg,
    )

    expected_sample_variance = np.var([0.2, 0.4, 0.8], ddof=1)
    assert signal_imag.dtype == np.float32
    assert variance.dtype == np.float32
    assert subset.dtype == np.float32
    assert signal_imag.shape == (n_columns,)
    assert variance.shape == (n_columns,)
    assert np.allclose(variance, expected_sample_variance)
    assert subset.shape == (3, ENSEMBLE_MEAN_SUBSET_N_COLUMNS)
    assert np.allclose(
        subset,
        ensemble_real.reshape(3, -1)[:, ensemble_mean_subset_column_indices(columns)],
    )
    assert imaginary_check["max_abs_standardized_deviation"] == 0.0


def test_merged_reduction_keeps_ensemble_order_for_crn_partners() -> None:
    def reduced(cos_sum: np.ndarray, sin_sum: np.ndarray) -> ReducedResult:
        return ReducedResult(
            cos_sum=cos_sum,
            sin_sum=sin_sum,
            n_walkers=2,
            n_escaped=0,
            occupancy_counts=np.zeros(2, dtype=np.int64),
            ensemble_cos_means=[cos_sum / 6.0],
            ensemble_sin_means=[sin_sum / 6.0],
        )

    first = reduced(np.asarray([6.0, 3.0]), np.asarray([0.0, 1.5]))
    second = reduced(np.asarray([3.0, 0.0]), np.asarray([1.5, 3.0]))
    merged = _merge_results([first, second])
    cfg = SimConfig(small_deltas=[1.0], big_deltas=[1.0], b_values=[0.0, 500.0])
    payload = sig._assemble(merged, sig.build_columns(cfg))

    assert np.array_equal(payload["ensemble_S"].reshape(2, -1), [[1.0, 0.5], [0.5, 0.0]])
    assert np.array_equal(payload["ensemble_S_imag"].reshape(2, -1), [[0.0, 0.25], [0.25, 0.5]])


def test_declared_replicate_subset_resolves_to_exactly_190_canonical_entries() -> None:
    canonical, _ = make_remediation_log_grid().triplets_and_weights()
    selected, provenance = load_remediation_entry_subset(
        "data/madi_v5_replicate_entry_subset.json", canonical,
    )
    assert len(selected) == 190
    assert sum(rho > 0.0 for _, rho, _ in selected) == 189
    assert provenance["cellular_entries"] == 189
    assert provenance["total_entries"] == 190


def test_shard_merger_rejects_mixed_build_seeds_for_v5_crn_contract(tmp_path) -> None:
    cfg = _diagnostic_config(n_ensembles=2)
    columns = sig.build_columns(cfg)
    n_columns = columns.n_pairs * columns.n_b
    subset_columns = ensemble_mean_subset_column_indices(columns)

    def write_shard(path: Path, value: float, build_seed: int) -> None:
        vector = np.full(n_columns, value, dtype=np.float64)
        ensemble_means = np.stack((vector - 0.1, vector + 0.1)).astype(np.float32)
        _save_library([
            LibraryEntry(
                kio=value, rho=1.0e5, V=5.0,
                vector=vector,
                signal_imag=np.zeros(n_columns, dtype=np.float32),
                signal_variance=np.full(n_columns, 0.02, dtype=np.float32),
                ensemble_means_subset=ensemble_means[:, subset_columns],
            ),
        ], str(path), cfg=cfg, columns=columns, build_seed=build_seed)

    first = tmp_path / "one.shard000.npz"
    second = tmp_path / "one.shard001.npz"
    write_shard(first, 1.0, build_seed=101)
    write_shard(second, 2.0, build_seed=202)
    merged_path = tmp_path / "merged.npz"
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.merge_shards", str(first), str(second),
         "--out", str(merged_path)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 1
    assert "metadata/schema/uncertainty mismatch" in completed.stdout

    write_shard(second, 2.0, build_seed=101)
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.merge_shards", str(first), str(second),
         "--out", str(merged_path)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    with np.load(merged_path, allow_pickle=False) as data:
        assert str(data["library_schema"]) == "madi-library-v5"
        assert data["signal_imag"].shape == (2, n_columns)
        assert data["signal_variance"].shape == (2, n_columns)
        assert data["ensemble_means_subset"].shape == (2, 2, ENSEMBLE_MEAN_SUBSET_N_COLUMNS)

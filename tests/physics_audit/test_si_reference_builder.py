"""Tier-A integrity tests for the SI §S.II governing-geometry builder."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_si_geometry_reference import (
    N_ALPHA,
    _write_shard,
    build_shard,
    merge,
    vi_from_x,
    x_from_vi,
)


pytestmark = pytest.mark.tier_a


def test_si_reference_shard_merger_uses_untrimmed_sufficient_statistics(tmp_path) -> None:
    """A small fixture verifies the exact production artifact schema.

    The deliberately tiny count is accepted only by the test configuration;
    the runtime production loader separately requires five million cells.
    """
    alpha = np.linspace(0.0, x_from_vi(0.40), N_ALPHA)
    sums, sums_sq, count = build_shard(8, shard_id=0, n_shards=1, seed=20_260_805, alphas=alpha)
    shard = tmp_path / "reference.shard000.npz"
    args = SimpleNamespace(n_single_cells=8, shard_id=0, n_shards=1, seed=20_260_805)
    _write_shard(shard, sums, sums_sq, count, alpha, args)
    output = tmp_path / "reference.npz"
    merge([shard], output)

    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        order = np.argsort(vi_from_x(alpha))
        assert np.array_equal(data["vi"], vi_from_x(alpha)[order])
        assert np.array_equal(data["alpha_x"], alpha[order])
        assert np.allclose(data["mean_A_over_V_norm"], (sums / count)[order])
        assert metadata["mean_estimator"] == "untrimmed_arithmetic_mean_A_over_V"
        assert metadata["contraction_rule"] == "all_shifted_voronoi_facets_S2"
        assert metadata["kappa"] == pytest.approx(0.90)
        assert metadata["n_alpha"] == 26
        assert metadata["n_single_cells"] == 8


def test_si_reference_merge_rejects_a_missing_shard_id(tmp_path) -> None:
    """P0-B: count agreement cannot hide a missing/renumbered shard."""
    alpha = np.linspace(0.0, x_from_vi(0.40), N_ALPHA)
    sums, sums_sq, count = build_shard(8, shard_id=0, n_shards=2, seed=20_260_806, alphas=alpha)
    shard = tmp_path / "reference.shard000.npz"
    args = SimpleNamespace(n_single_cells=8, shard_id=0, n_shards=2, seed=20_260_806)
    _write_shard(shard, sums, sums_sq, count, alpha, args)
    with pytest.raises(RuntimeError, match="incomplete reference cloud"):
        merge([shard], tmp_path / "incomplete.npz")

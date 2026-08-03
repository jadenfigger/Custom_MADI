"""Direct structural audit of ``madi_dense_universal.npz``.

Run the fast metadata checks with:

    PYTHONPATH=. pytest -q tests/physics_audit/test_production_library.py

Add ``MADI_AUDIT_SLOW=1`` to scan every one of the 1.265 billion stored
signals and to compare all shard parameter keys.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from madi.config import valid_delta_pairs
from madi.library import _grid_columns, load_library_meta

from ._artifact import (
    metadata,
    require_production_library,
    scan_signal_quality,
    shard_paths,
    triplet_keys,
    vectors_memmap,
)


def _have_production_library() -> bool:
    return require_production_library().is_file()


def _require_slow() -> None:
    if os.environ.get("MADI_AUDIT_SLOW") != "1":
        pytest.skip("set MADI_AUDIT_SLOW=1 to scan the 10 GB production artifact")


def test_production_schema_and_universal_axes() -> None:
    path = require_production_library()
    meta = metadata(path)
    vec = vectors_memmap(path)

    assert set(meta) == {
        "kios", "rhos", "Vs", "pair_deltas", "pair_Deltas", "b_values",
        "n_b", "h_ms",
    }
    assert vec.dtype == np.float64
    assert vec.shape == (40_645, 31_125)
    assert int(meta["n_b"]) == 25
    assert float(meta["h_ms"]) == 1.0
    assert np.array_equal(meta["b_values"], np.arange(0.0, 12_000.0 + 500.0, 500.0))

    pairs = list(zip(meta["pair_deltas"].tolist(), meta["pair_Deltas"].tolist()))
    expected_pairs = valid_delta_pairs(
        list(range(1, 31)), list(range(1, 51)) + list(range(55, 81, 5))
    )
    assert pairs == expected_pairs
    assert len(pairs) == 1_245
    assert vec.shape[1] == len(pairs) * int(meta["n_b"])


def test_reader_and_writer_use_the_same_pair_major_b_major_axis_order() -> None:
    path = require_production_library()
    meta = metadata(path)
    reader_meta = load_library_meta(str(path))
    assert reader_meta["format"] == "v2"
    assert reader_meta["delta_pairs"] == list(
        zip(meta["pair_deltas"].tolist(), meta["pair_Deltas"].tolist())
    )
    assert reader_meta["b_values"] == meta["b_values"].tolist()

    # (delta,Delta)=(20,50) is pair 923 and b=1000..2500 are b indices 2..5.
    # The bundled slim reference file independently records these source columns.
    cols = _grid_columns(
        [(20.0, 50.0, b) for b in (1_000.0, 1_500.0, 2_000.0, 2_500.0)],
        reader_meta["delta_pairs"],
        reader_meta["b_values"],
        reader_meta["n_b"],
    )
    assert cols.tolist() == [23_077, 23_078, 23_079, 23_080]

    slim = np.load(
        path.with_name("madi_dense_universal_delta20_D50_b1000-2500.npz"),
        allow_pickle=False,
    )
    try:
        vec = vectors_memmap(path)
        assert np.array_equal(vec[:100, cols], slim["vectors"][:100])
    finally:
        slim.close()


def test_current_parameter_topology_is_recorded_as_noncanonical_dense_mask() -> None:
    """This is a detection test, not an endorsement of the topology.

    It protects the audit conclusion: the current artifact is a masked dense
    rho x V grid, not the paper's 20 discrete v_i hyperbolae.
    """
    meta = metadata()
    kios, rhos, volumes = meta["kios"], meta["rhos"], meta["Vs"]
    vi = rhos * volumes * 1e-6
    pairs = np.column_stack((rhos, volumes))

    assert len(kios) == 40_645
    assert np.unique(kios).size == 55
    assert np.array_equal(np.unique(kios), np.r_[np.arange(1.0, 51.0), [60., 70., 80., 90., 100.]])
    assert 0.0 not in set(kios.tolist())
    assert np.unique(rhos).size == 100
    assert np.unique(volumes).size == 99
    assert np.unique(pairs, axis=0).shape[0] == 739
    assert np.unique(np.round(vi, 12)).size == 739
    assert np.isclose(vi.min(), 0.40078971533516986)
    assert np.isclose(vi.max(), 0.9488174676053464)
    assert np.all((vi >= 0.4) & (vi <= 0.95))


def test_current_grid_spacing_is_explicit() -> None:
    meta = metadata()
    kios = np.unique(meta["kios"])
    volumes = np.unique(meta["Vs"])
    rhos = np.unique(meta["rhos"])
    assert np.allclose(np.diff(rhos), np.diff(rhos)[0])
    assert np.allclose(np.diff(volumes), np.diff(volumes)[0])
    assert np.isclose(np.diff(volumes)[0], 0.0898989898989899)
    assert np.isclose(np.diff(rhos)[0], 29_292.929292929292)
    assert np.array_equal(np.diff(kios[:50]), np.ones(49))
    assert np.array_equal(np.diff(kios[50:]), np.full(4, 10.0))


@pytest.mark.slow
def test_all_shards_cover_the_production_triplet_set_without_duplicates() -> None:
    _require_slow()
    meta = metadata()
    production_keys = triplet_keys(meta["kios"], meta["rhos"], meta["Vs"])
    paths = shard_paths()
    assert len(paths) == 128
    assert [p.name for p in paths] == [f"madi_dense.shard{i:03d}.npz" for i in range(128)]

    all_keys: set[tuple[float, float, float]] = set()
    for shard in paths:
        with np.load(shard, allow_pickle=False) as data:
            # Do not index ``vectors`` here: that would materialise every
            # 80-MB shard merely to inspect its shape.
            assert "vectors" in data.files
            assert int(data["n_b"]) == 25
            assert np.array_equal(data["pair_deltas"], meta["pair_deltas"])
            assert np.array_equal(data["pair_Deltas"], meta["pair_Deltas"])
            assert np.array_equal(data["b_values"], meta["b_values"])
            keys = triplet_keys(data["kios"], data["rhos"], data["Vs"])
            assert len(keys) == len(data["kios"])
            assert not (all_keys & keys)
            all_keys |= keys
    assert all_keys == production_keys


@pytest.mark.slow
def test_production_vectors_are_monotone_down_to_the_stated_trust_floor() -> None:
    _require_slow()
    stats = scan_signal_quality()
    assert stats["nonfinite"] == 0, stats
    assert stats["b0_not_one"] == 0, stats
    assert stats["increases_at_or_above_floor"] == 0, stats


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="current artifact contains low-signal negative Monte-Carlo samples")
def test_production_vectors_are_nonnegative_everywhere() -> None:
    _require_slow()
    stats = scan_signal_quality()
    assert stats["negative"] == 0, stats

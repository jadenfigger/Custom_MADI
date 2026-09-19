"""Tests for the manifold explorer's column loader and slice arithmetic.

Runnable two ways:

    python -m pytest tests/manifold_explorer -q
    python -m tools.manifold_explorer.selftest

so there are no fixtures and no pytest-only constructs in the test bodies.
Tests that need the 15 GB production artifact are marked ``slow`` and skip
themselves when it is not present.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from madi.library import LibraryEntry, match_voxels_batch_fits0
from tools.manifold_explorer import slice as slicing
from tools.manifold_explorer.build_column_cache import DEFAULT_LIBRARY, build
from tools.manifold_explorer.columns import (
    CachedColumnReader,
    NpzColumnReader,
    cache_is_valid,
    cache_paths,
    load_labels,
    memmap_member,
    npy_member_layout,
    open_reader,
)

# --------------------------------------------------------------------------
# A tiny stand-in library with the production schema's members
# --------------------------------------------------------------------------

TINY_PAIRS = [(4.0, 20.0), (4.0, 50.0), (20.0, 50.0)]
TINY_B = np.arange(0.0, 5 * 500.0, 500.0)


def make_tiny_library(path: Path, n_cellular: int = 40, seed: int = 0) -> np.ndarray:
    """Write a small v5-shaped library and return its full vectors array."""
    rng = np.random.default_rng(seed)
    n_entries = n_cellular + 1                      # row 0 is free water
    n_columns = len(TINY_PAIRS) * len(TINY_B)

    # Plausible decaying curves; column 0 of each pair is b=0, hence exactly 1.
    decay = rng.uniform(1e-4, 8e-4, size=(n_entries, 1))
    vectors = np.exp(-decay * np.tile(TINY_B, len(TINY_PAIRS))[None, :])
    vectors[:, ::len(TINY_B)] = 1.0
    vectors = np.ascontiguousarray(vectors, dtype=np.float64)

    rhos = np.concatenate([[0.0], rng.uniform(1e4, 1e7, n_cellular)])
    Vs = np.concatenate([[0.0], rng.uniform(0.05, 90.0, n_cellular)])
    vis = np.clip(rhos * Vs * 1e-6, 0.0, 0.99)
    vis[0] = 0.0
    kios = np.concatenate([[np.nan], rng.uniform(0.0, 130.0, n_cellular)])
    free = np.zeros(n_entries, dtype=bool)
    free[0] = True

    np.savez(
        path,
        library_schema=np.array("madi-library-v5"),
        kios=kios, rhos=rhos, Vs=Vs, vis=vis, vectors=vectors,
        nominal_kios=kios, nominal_rhos=rhos, nominal_Vs=Vs,
        weights=np.ones(n_entries), is_free_water=free,
        signal_variance=np.asarray(
            rng.uniform(1e-8, 1e-6, size=(n_entries, n_columns)), dtype=np.float32),
        pair_deltas=np.array([d for d, _ in TINY_PAIRS]),
        pair_Deltas=np.array([D for _, D in TINY_PAIRS]),
        b_values=TINY_B, n_b=np.array(len(TINY_B)), h_ms=np.array(1.0),
    )
    return vectors


# --------------------------------------------------------------------------
# Column loader equivalence
# --------------------------------------------------------------------------

def test_npz_reader_matches_full_array():
    """The .npz reader returns exactly what indexing the full array returns."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        vectors = make_tiny_library(path)
        with zipfile.ZipFile(path) as archive:
            assert archive.getinfo("vectors.npy").compress_type == zipfile.ZIP_STORED

        reader = NpzColumnReader(path)
        rng = np.random.default_rng(1)
        for _ in range(5):
            columns = np.sort(rng.choice(vectors.shape[1], 4, replace=False))
            assert np.array_equal(reader.read(columns), vectors[:, columns])
        reader.close()


def test_cache_reader_matches_full_array():
    """The column-major cache returns exactly the same values, bit for bit."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        cache_dir = Path(workspace) / "cache"
        vectors = make_tiny_library(path)
        build(path, cache_dir, band_bytes=4096)

        paths = cache_paths(path, cache_dir)
        reader = CachedColumnReader(paths["signal"], paths["variance"])
        rng = np.random.default_rng(2)
        for _ in range(5):
            columns = np.sort(rng.choice(vectors.shape[1], 4, replace=False))
            assert np.array_equal(reader.read(columns), vectors[:, columns])

        # And the variance cache matches its float32 source exactly too.
        with np.load(path) as original:
            variance = original["signal_variance"]
        columns = np.array([0, 3, 11])
        assert np.array_equal(reader.read_variance(columns),
                              variance[:, columns].astype(np.float64))
        reader.close()

        # open_reader should now prefer the cache.
        assert open_reader(path, cache_dir).kind == "cache"
        assert open_reader(path, cache_dir, prefer_cache=False).kind == "npz"


def test_cache_band_width_does_not_change_values():
    """Transposing in one band or in many gives identical output."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        vectors = make_tiny_library(path)
        wide, narrow = Path(workspace) / "wide", Path(workspace) / "narrow"
        build(path, wide, band_bytes=10 ** 9, with_variance=False)
        build(path, narrow, band_bytes=1, with_variance=False)
        a = np.load(cache_paths(path, wide)["signal"])
        b = np.load(cache_paths(path, narrow)["signal"])
        assert np.array_equal(a, b)
        assert np.array_equal(a.T, vectors)


@pytest.mark.slow
def test_production_columns_match_stored_rows():
    """Spot-check the real artifact: cached column j, row i == stored row i, col j.

    Reads whole rows from the original row-major member (cheap: 249 KB each)
    rather than the whole 4.7 GB matrix, then compares those cells against the
    cache and against the .npz column reader.
    """
    if not DEFAULT_LIBRARY.exists():
        pytest.skip(f"production library not present at {DEFAULT_LIBRARY}")
    cached, why = cache_is_valid(DEFAULT_LIBRARY)
    labels = load_labels(DEFAULT_LIBRARY)
    stored = memmap_member(DEFAULT_LIBRARY, "vectors")

    rng = np.random.default_rng(3)
    rows = np.sort(rng.choice(labels.n_entries, 12, replace=False))
    columns = np.sort(rng.choice(labels.n_columns, 6, replace=False))
    truth = np.stack([np.asarray(stored[int(row)])[columns] for row in rows])

    npz_reader = NpzColumnReader(DEFAULT_LIBRARY)
    assert np.array_equal(npz_reader.read(columns)[rows], truth)
    npz_reader.close()

    if cached:
        reader = open_reader(DEFAULT_LIBRARY)
        assert reader.kind == "cache"
        assert np.array_equal(reader.read(columns)[rows], truth)

        # Stronger: one whole (delta, Delta) pair, all 18,820 rows, cache
        # against the original file's own bytes.
        pair_columns = labels.columns_for_pair(20.0, 50.0)
        direct = NpzColumnReader(DEFAULT_LIBRARY)
        assert np.array_equal(reader.read(pair_columns), direct.read(pair_columns))
        direct.close()
        reader.close()
    else:
        print(f"cache not checked: {why}")


# --------------------------------------------------------------------------
# Slice arithmetic
# --------------------------------------------------------------------------

def _tiny_setup(workspace: Path):
    path = Path(workspace) / "tiny.npz"
    vectors = make_tiny_library(path)
    labels = load_labels(path)
    eligible = np.flatnonzero(slicing.candidate_mask(labels, 0.0, 1.0))
    return path, vectors, labels, eligible


def test_small_sigma_keeps_only_the_reference_entry():
    """Shrink the noise and the slice collapses onto the point it is centred on."""
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(4.0, 20.0)
        block = vectors[eligible][:, columns]
        centre = 7
        reference = block[centre]

        tight = slicing.slice_manifold(block, reference, labels, eligible,
                                       threshold=len(columns), sigma_measurement=1e-12)
        assert tight.n_survivors == 1
        assert tight.survivor_rows[0] == eligible[centre]

        loose = slicing.slice_manifold(block, reference, labels, eligible,
                                       threshold=len(columns), sigma_measurement=1e6)
        assert loose.n_survivors == len(eligible)


def test_identical_predictions_all_survive_a_tiny_sigma():
    """With duplicate rows, a tiny sigma keeps exactly the duplicates."""
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(4.0, 20.0)
        block = vectors[eligible][:, columns].copy()
        block[3] = block[9]                      # make two entries indistinguishable
        result = slicing.slice_manifold(block, block[9], labels, eligible,
                                        threshold=len(columns), sigma_measurement=1e-12)
        assert sorted(result.survivor_rows.tolist()) == sorted(
            [int(eligible[3]), int(eligible[9])])


def test_threshold_is_the_bayes_weight_level_set():
    """chi2 <= T is exactly 'relative bayes weight >= exp(-T/2)'."""
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(20.0, 50.0)
        block = vectors[eligible][:, columns]
        sigma, threshold = 0.02, 3.0
        result = slicing.slice_manifold(block, block[5], labels, eligible,
                                        threshold=threshold, sigma_measurement=sigma)
        weight = np.exp(-np.sum((block - block[5]) ** 2, axis=1) / (2 * sigma ** 2))
        assert np.array_equal(result.survivors, weight >= np.exp(-threshold / 2))


def test_free_s0_chi2_agrees_with_the_projects_free_s0_matcher():
    """Our free-S0 residual reproduces madi.library.match_voxels_batch_fits0.

    That matcher reports, for its winning candidate, the raw residual
    ``||m||^2 - (m.s)^2/(s.s)``.  Ours is that quantity divided by the fitted
    amplitude squared and by sigma^2, so scaling back must reproduce it, and
    the argmin must pick the same entry.
    """
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(4.0, 50.0)
        triples = [(4.0, 50.0, float(b)) for b in labels.b_values]
        entries = [
            LibraryEntry(kio=float(labels.kios[row]), rho=float(labels.rhos[row]),
                         V=float(labels.Vs[row]), vector=vectors[row],
                         vi=float(labels.vis[row]),
                         is_free_water=bool(labels.is_free_water[row]))
            for row in eligible
        ]
        measured = 0.75 * vectors[eligible[6]][columns]      # amplitude 0.75
        block = vectors[eligible][:, columns]

        ours = slicing.chi2_free_s0(block, measured, 1.0)
        # Re-derive the matcher's raw residual from ours and compare.
        scaled = block
        amplitude = (scaled @ measured) / np.einsum("ij,ij->i", scaled, scaled)
        raw = ours * amplitude ** 2

        *_, residual, s0 = match_voxels_batch_fits0(
            measured[None, :], entries, labels.delta_pairs_list(),
            list(labels.b_values), labels.n_b, triples,
            vi_min=0.0, vi_max=1.0, include_free_water=True, use_gpu=False,
        )
        assert int(np.argmin(ours)) == int(np.argmin(raw))
        assert np.isclose(raw.min(), float(residual[0]), rtol=1e-9, atol=1e-18)
        assert np.isclose(amplitude[int(np.argmin(ours))], float(s0[0]), rtol=1e-9)


def test_free_s0_is_blind_to_a_global_amplitude():
    """Scaling the reference must not move the free-S0 slice at all."""
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(20.0, 50.0)
        block = vectors[eligible][:, columns]
        a = slicing.chi2_free_s0(block, block[4], 0.02)
        b = slicing.chi2_free_s0(block, 3.3 * block[4], 0.02)
        assert np.allclose(a, b, rtol=1e-9)


def test_candidate_mask_uses_the_fitters_filter():
    """Free water is out by default, in when asked, and the v_i band bites."""
    with tempfile.TemporaryDirectory() as workspace:
        _, _, labels, _ = _tiny_setup(Path(workspace))
        without = slicing.candidate_mask(labels, 0.0, 1.0)
        with_free = slicing.candidate_mask(labels, 0.0, 1.0, include_free_water=True)
        assert not without[0] and with_free[0]

        banded = slicing.candidate_mask(labels, 0.40, 0.99)
        expected = (labels.vis >= 0.40) & (labels.vis <= 0.99) & ~labels.is_free_water
        assert np.array_equal(banded, expected)


def test_monte_carlo_variance_widens_the_slice():
    """Adding the library's own MC error in quadrature can only keep more."""
    with tempfile.TemporaryDirectory() as workspace:
        path, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(4.0, 20.0)
        block = vectors[eligible][:, columns]
        variance = NpzColumnReader(path).read_variance(columns)[eligible]
        plain = slicing.slice_manifold(block, block[2], labels, eligible,
                                       threshold=len(columns), sigma_measurement=1e-4)
        widened = slicing.slice_manifold(block, block[2], labels, eligible,
                                         threshold=len(columns), sigma_measurement=1e-4,
                                         variance_block=variance)
        assert widened.n_survivors >= plain.n_survivors


def test_widths_shrink_monotonically_in_survivor_count():
    """Each extra measured column can only remove survivors, never add them."""
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = np.concatenate([labels.columns_for_pair(4.0, 20.0),
                                  labels.columns_for_pair(20.0, 50.0)])
        block = vectors[eligible][:, columns]
        curve = slicing.widths_versus_measured_count(
            block, block[5], labels, eligible, sigma_measurement=0.01,
            reduced_threshold=0.0,          # exact agreement: strictly nested sets
        )
        counts = [step["n_survivors"] for step in curve]
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] >= 1

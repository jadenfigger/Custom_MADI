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
from madi import fisher_crlb as fc
from tools.manifold_explorer import axes as axis_groups
from tools.manifold_explorer import fisher as fisher_tools
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


# --------------------------------------------------------------------------
# Grouped display axes
# --------------------------------------------------------------------------

def test_single_b_group_reproduces_the_single_column():
    """A group of one b-value collapses to exactly that column's values."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        vectors = make_tiny_library(path)
        labels = load_labels(path)
        column = labels.column_index(4.0, 20.0, 1500.0)
        group = axis_groups.pair_group(labels, 4.0, 20.0, [1500.0])

        assert group.columns.tolist() == [column]
        collapsed, valid = axis_groups.collapse(vectors[:, group.columns], group,
                                                "mean")
        assert np.array_equal(collapsed, vectors[:, column])
        assert valid.all()


def test_group_mean_matches_a_direct_numpy_mean():
    """Mean collapse is a plain mean over the group's columns, bit for bit."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        vectors = make_tiny_library(path)
        labels = load_labels(path)
        chosen = [500.0, 1000.0, 1500.0, 2000.0]
        group = axis_groups.pair_group(labels, 4.0, 50.0, chosen)
        block = NpzColumnReader(path).read(group.columns)

        collapsed, _ = axis_groups.collapse(block, group, "mean")
        assert np.array_equal(collapsed, vectors[:, group.columns].mean(axis=1))


def test_adc_matches_polyfit_entry_by_entry():
    """The closed-form slope equals numpy's least-squares fit of ln S on b."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        vectors = make_tiny_library(path)
        labels = load_labels(path)
        group = axis_groups.pair_group(labels, 20.0, 50.0,
                                       [0.0, 500.0, 1000.0, 1500.0, 2000.0])
        block = vectors[:, group.columns]
        values, valid = axis_groups.collapse_adc(block, group.b_values)
        assert valid.all()
        for row in (0, 5, 17, 33):
            slope = np.polyfit(np.asarray(group.b_values), np.log(block[row]), 1)[0]
            expected = -slope / axis_groups.B_S_MM2_TO_MS_UM2
            assert np.isclose(values[row], expected, rtol=1e-10)


def test_adc_recovers_a_known_gaussian_diffusivity():
    """A pure exp(-b D) curve must give back D in um^2/ms.

    The library's free-water atom is exactly this: a Gaussian compartment at
    D0 = 3 um^2/ms.  Here it is constructed analytically so the test does not
    need the 15 GB artifact.
    """
    b = np.array([0.0, 500.0, 1000.0, 1500.0])          # s/mm^2
    for D0 in (3.0, 1.7, 0.8):                          # um^2/ms
        signal = np.exp(-b * D0 * axis_groups.B_S_MM2_TO_MS_UM2)[None, :]
        values, valid = axis_groups.collapse_adc(signal, b)
        assert valid[0]
        assert np.isclose(values[0], D0, rtol=1e-10)


def test_adc_excludes_nonpositive_signal_without_clipping():
    """S <= 0 has no logarithm; the entry is dropped, never clipped."""
    b = np.array([0.0, 500.0, 1000.0])
    block = np.array([[1.0, 0.5, 0.25],
                      [1.0, 0.5, 0.0],        # exactly zero
                      [1.0, 0.5, -1e-4]])     # negative Monte-Carlo mean
    values, valid = axis_groups.collapse_adc(block, b)
    assert valid.tolist() == [True, False, False]
    assert np.isfinite(values[0]) and np.isnan(values[1]) and np.isnan(values[2])


def test_adc_is_refused_where_it_would_be_meaningless():
    """One b-value gives no slope; an 'any columns' group mixes diffusion times."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        make_tiny_library(path)
        labels = load_labels(path)

        single = axis_groups.pair_group(labels, 4.0, 20.0, [1000.0])
        assert not single.supports_adc
        assert "single b-value" in single.adc_refusal()

        mixed = axis_groups.any_group([0, 7, 12])
        assert not mixed.supports_adc
        assert "diffusion times" in mixed.adc_refusal()

        with pytest.raises(ValueError):
            axis_groups.collapse(np.ones((3, 1)), single, "adc")

        good = axis_groups.pair_group(labels, 4.0, 20.0, [500.0, 1000.0])
        assert good.supports_adc and good.adc_refusal() == ""


def test_group_labels_name_the_acquisition():
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        make_tiny_library(path)
        labels = load_labels(path)
        group = axis_groups.pair_group(labels, 4.0, 20.0,
                                       [500.0, 1000.0, 1500.0, 2000.0])
        assert group.label("mean") == "mean S/S0, d=4 D=20, b=500-2000"
        assert group.label("adc") == "ADC (um^2/ms), d=4 D=20, b=500-2000"


def test_reference_collapse_needs_every_column_of_the_group():
    """A pasted reference that misses a group's columns yields no marker."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        vectors = make_tiny_library(path)
        labels = load_labels(path)
        group = axis_groups.pair_group(labels, 4.0, 20.0, [500.0, 1000.0])

        covered = {int(c): float(vectors[3, c]) for c in group.columns}
        assert np.isclose(axis_groups.collapse_reference(covered, group, "mean"),
                          vectors[3, group.columns].mean())

        partial = {int(group.columns[0]): 0.5}
        assert axis_groups.collapse_reference(partial, group, "mean") is None


# --------------------------------------------------------------------------
# Reference-relative hues
# --------------------------------------------------------------------------

def test_rmse_and_chi_are_zero_at_the_reference_entry():
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(20.0, 50.0)
        block = vectors[eligible][:, columns]
        centre = 6
        rmse = slicing.rmse_to_reference(block, block[centre])
        chi2 = slicing.chi2_fixed_s0(block, block[centre], 0.02)
        assert rmse[centre] == 0.0
        assert chi2[centre] == 0.0
        assert np.all(rmse >= 0.0)


def test_rmse_is_the_root_mean_square_residual():
    block = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 5.0]])
    reference = np.array([1.0, 2.0, 3.0])
    expected = np.sqrt(np.array([0.0, 4.0 / 3.0]))
    assert np.allclose(slicing.rmse_to_reference(block, reference), expected)


def test_chi_threshold_selects_exactly_the_survivors():
    """chi <= sqrt(threshold) is the same set the slice keeps, in both S0 modes."""
    with tempfile.TemporaryDirectory() as workspace:
        _, vectors, labels, eligible = _tiny_setup(Path(workspace))
        columns = labels.columns_for_pair(4.0, 20.0)
        block = vectors[eligible][:, columns]
        variance = NpzColumnReader(
            Path(workspace) / "tiny.npz").read_variance(columns)[eligible]
        for mode in ("fixed", "free"):
            result = slicing.slice_manifold(
                block, block[4], labels, eligible, threshold=2.5,
                sigma_measurement=0.01, variance_block=variance, s0_mode=mode)
            chi = np.sqrt(np.maximum(result.chi2, 0.0))
            assert np.array_equal(chi <= np.sqrt(result.threshold),
                                  result.survivors)


# --------------------------------------------------------------------------
# Log-axis zeros
# --------------------------------------------------------------------------

def test_log_width_ignores_nonpositive_values_instead_of_going_nan():
    """The free-water atom (rho = 0) must not blank a whole width curve."""
    with_zero = slicing._spread(np.array([0.0, 2.0, 8.0]), log=True)
    assert with_zero["n_nonpositive"] == 1
    assert np.isclose(with_zero["log_width"], 4.0)

    all_positive = slicing._spread(np.array([2.0, 8.0]), log=True)
    assert all_positive["n_nonpositive"] == 0
    assert np.isclose(all_positive["log_width"], 4.0)


def test_width_curve_survives_the_free_water_atom():
    """With free water included the curve still has finite widths at every step."""
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "tiny.npz"
        vectors = make_tiny_library(path)
        labels = load_labels(path)
        eligible = np.flatnonzero(
            slicing.candidate_mask(labels, 0.0, 1.0, include_free_water=True))
        assert labels.is_free_water[eligible].any()
        columns = labels.columns_for_pair(4.0, 20.0)
        block = vectors[eligible][:, columns]
        curve = slicing.widths_versus_measured_count(
            block, block[5], labels, eligible, sigma_measurement=1e6)
        # Every entry survives at this sigma, free water included.
        assert curve[0]["n_nonpositive_rho"] >= 1
        assert np.isfinite(curve[0]["rho_log_width"])

# --------------------------------------------------------------------------
# Fisher information, eigen-spectrum and CRLB
# --------------------------------------------------------------------------

# A synthetic library laid out on the REAL canonical grid, so the stencil code
# walks the same node indices it will walk in production, but whose signal is a
# closed-form function of the parameters.  Central differences of a function
# linear in (log rho, log V, k_io) are exact, which turns "does the Jacobian
# agree with the derivative" into an exact equality rather than a tolerance.

# One coefficient row per column, chosen to have rank 3 so the Fisher matrix
# is non-singular: a model where every column responds the same way to the
# parameters has a rank-1 Jacobian and no confidence ellipse at all.
LINEAR_JACOBIAN = np.array([
    [0.31, -0.17, 0.0060],
    [0.12, 0.44, -0.0020],
    [-0.08, 0.05, 0.0110],
    [0.22, -0.31, 0.0040],
    [0.05, 0.09, -0.0070],
])
CURVATURE_WEIGHT = np.array([1.0, -0.6, 0.3, 0.8, -0.2])


def _canonical_synthetic(jacobian=LINEAR_JACOBIAN, quadratic: float = 0.0):
    """Rows on the canonical grid carrying `A theta` (+ optional curvature)."""
    rho_axis, volume_axis, kio_axis, retained = fc.canonical_grid()
    nodes = [(ir, iv, ik) for (ir, iv) in sorted(retained)
             for ik in range(len(kio_axis))]
    theta = np.array([[np.log(rho_axis[ir]), np.log(volume_axis[iv]),
                       kio_axis[ik]] for ir, iv, ik in nodes])
    signal = theta @ np.asarray(jacobian).T
    if quadratic:
        signal = signal + quadratic * ((theta[:, 2] ** 2)[:, None]
                                       * CURVATURE_WEIGHT[None, :])

    class Labels:
        """Only the fields build_node_grid reads."""
        nominal_rhos = np.array([rho_axis[ir] for ir, _, _ in nodes])
        nominal_Vs = np.array([volume_axis[iv] for _, iv, _ in nodes])
        kios = np.array([kio_axis[ik] for _, _, ik in nodes])
        is_free_water = np.zeros(len(nodes), dtype=bool)

    grid = fisher_tools.build_node_grid(Labels())
    return grid, signal, theta, nodes


def test_node_grid_round_trips_and_matches_the_canonical_axes():
    """Every row maps to a node and back, and k_io ascends as the axis does."""
    grid, _, _, nodes = _canonical_synthetic()
    rhos, volumes, kios, _ = fc.canonical_grid()
    assert len(grid.lookup) == len(nodes)
    for node, row in list(grid.lookup.items())[::997]:
        assert grid.node_of_row[row] == node
        assert grid.row(node) == row
    # One (rho, V) group carries the whole canonical k_io axis, in order.
    some = next(iter(grid.lookup))[:2]
    axis = np.array([grid.kios[ik] for ik in range(len(kios))])
    assert np.array_equal(axis, kios)
    assert grid.row((some[0], some[1], 0)) is not None


def test_central_differences_are_exact_for_a_linear_model():
    """If S is linear in (log rho, log V, k_io), J must be its coefficients."""
    grid, signal, _, _ = _canonical_synthetic()
    interior = [node for node in grid.lookup
                if all(fisher_tools.stencil(grid, node, axis) is not None
                       for axis in fisher_tools.AXES)]
    assert interior, "the canonical grid should have interior nodes"
    expected = LINEAR_JACOBIAN
    for node in interior[::1500]:
        J, missing = fisher_tools.jacobian(grid, signal, node)
        assert missing == []
        assert np.allclose(J, expected, rtol=1e-9, atol=1e-12)


def test_stencil_is_refused_rather_than_one_sided_at_the_edges():
    """A missing neighbour reports the axis; it never falls back to one-sided."""
    grid, signal, _, _ = _canonical_synthetic()
    edges = [node for node in grid.lookup
             if any(fisher_tools.stencil(grid, node, axis) is None
                    for axis in fisher_tools.AXES)]
    assert edges
    J, missing = fisher_tools.jacobian(grid, signal, edges[0])
    assert J is None and missing
    assert set(missing) <= set(fisher_tools.AXES)
    # k_io = 0 is the first node on its axis, so that axis must be missing.
    zero_kio = [node for node in grid.lookup if node[2] == 0][0]
    assert "k_io" in fisher_tools.jacobian(grid, signal, zero_kio)[1]


def test_richardson_beats_a_single_stencil_on_a_curved_model():
    """With real curvature, (4J1-J2)/3 is closer to the truth than J1 alone."""
    curvature = 3e-4
    grid, signal, theta, nodes = _canonical_synthetic(quadratic=curvature)
    node = next(node for node in grid.lookup
                if all(fisher_tools.stencil(grid, node, axis, w) is not None
                       for axis in fisher_tools.AXES for w in (1, 2)))
    row = grid.row(node)
    kio = grid.kios[node[2]]
    truth = LINEAR_JACOBIAN.copy()
    truth[:, 2] += CURVATURE_WEIGHT * 2.0 * curvature * kio   # d/dk of c3 k + q k^2

    J1, _ = fisher_tools.jacobian(grid, signal, node, 1)
    J2, _ = fisher_tools.jacobian(grid, signal, node, 2)
    combined = fisher_tools.richardson(J1, J2)
    assert np.abs(combined - truth).max() <= np.abs(J1 - truth).max() + 1e-12
    assert np.isfinite(fisher_tools.truncation_estimate(J1, J2))


def test_chi2_is_the_quadratic_form_of_the_fisher_matrix():
    """The claim the whole feature rests on, checked where it is exact.

    For a model linear in theta the second-order expansion is not an
    approximation: the measured chi2 between a node and its neighbour equals
    `dtheta^T F dtheta` exactly.
    """
    grid, signal, theta, nodes = _canonical_synthetic()
    node = next(node for node in grid.lookup
                if all(fisher_tools.stencil(grid, node, axis) is not None
                       for axis in fisher_tools.AXES))
    row = grid.row(node)
    J, _ = fisher_tools.jacobian(grid, signal, node)
    sigma = 0.05
    report = fisher_tools.fisher_report(J, signal[row], sigma, 20.0)

    index = {grid.row(n): n for n in grid.lookup}
    for axis, step in (("rho", (1, 0, 0)), ("V", (0, 1, 0)), ("k_io", (0, 0, 1))):
        neighbour = (node[0] + step[0], node[1] + step[1], node[2] + step[2])
        other = grid.row(neighbour)
        if other is None:
            continue
        rhos, volumes, kios, _ = fc.canonical_grid()
        delta = np.array([np.log(rhos[neighbour[0]]) - np.log(rhos[node[0]]),
                          np.log(volumes[neighbour[1]]) - np.log(volumes[node[1]]),
                          kios[neighbour[2]] - kios[node[2]]])
        measured = float(np.sum(((signal[other] - signal[row]) / sigma) ** 2))
        quadratic = float(delta @ report["F"] @ delta)
        assert np.isclose(measured, quadratic, rtol=1e-8)


def test_fisher_report_uses_the_project_primitives():
    """F is exactly fc.fisher_matrix's output, and CRLB its inverse diagonal."""
    grid, signal, _, _ = _canonical_synthetic()
    node = next(node for node in grid.lookup
                if all(fisher_tools.stencil(grid, node, axis) is not None
                       for axis in fisher_tools.AXES))
    J, _ = fisher_tools.jacobian(grid, signal, node)
    sigma = 0.02
    report = fisher_tools.fisher_report(J, signal[grid.row(node)], sigma, 20.0)
    assert np.allclose(report["F"], fc.fisher_matrix(J, sigma))
    if report["positive_definite"]:
        expected = np.sqrt(np.diag(np.linalg.inv(report["F"])))
        assert np.allclose(report["crlb"], expected, rtol=1e-8)


def test_confidence_ellipse_lies_on_the_threshold_level_set():
    """Every point returned satisfies dtheta^T S dtheta = threshold."""
    grid, signal, _, _ = _canonical_synthetic()
    node = next(node for node in grid.lookup
                if all(fisher_tools.stencil(grid, node, axis) is not None
                       for axis in fisher_tools.AXES))
    J, _ = fisher_tools.jacobian(grid, signal, node)
    report = fisher_tools.fisher_report(J, signal[grid.row(node)], 0.02, 20.0)
    threshold = 5.99
    for mode, precision in (("profiled", report["profiled_block"]),
                            ("conditional", np.asarray(report["F"])[:2, :2])):
        points = fisher_tools.confidence_ellipse(report, threshold, mode)
        if points is None:
            continue
        values = np.einsum("ni,ij,nj->n", points, precision, points)
        assert np.allclose(values, threshold, rtol=1e-8)


def test_knowing_k_io_can_only_shrink_the_ellipse():
    """The conditional ellipse is contained in the profiled (marginal) one."""
    grid, signal, _, _ = _canonical_synthetic()
    node = next(node for node in grid.lookup
                if all(fisher_tools.stencil(grid, node, axis) is not None
                       for axis in fisher_tools.AXES))
    J, _ = fisher_tools.jacobian(grid, signal, node)
    report = fisher_tools.fisher_report(J, signal[grid.row(node)], 0.02, 20.0)
    profiled = fisher_tools.confidence_ellipse(report, 5.99, "profiled")
    conditional = fisher_tools.confidence_ellipse(report, 5.99, "conditional")
    if profiled is None or conditional is None:
        pytest.skip("this synthetic node has no positive-definite 2x2 block")
    marginal_precision = np.asarray(report["profiled_block"])
    # Every conditional point is inside the profiled level set.
    inside = np.einsum("ni,ij,nj->n", conditional, marginal_precision, conditional)
    assert np.all(inside <= 5.99 * (1.0 + 1e-8))


def test_batch_quantities_match_the_single_node_report():
    grid, signal, _, _ = _canonical_synthetic()
    rows, J = fisher_tools.jacobian_batch(grid, signal)
    assert len(rows) == len(J)
    kios = np.zeros(signal.shape[0])
    for node, row in grid.lookup.items():
        kios[row] = grid.kios[node[2]]
    batch = fisher_tools.batch_quantities(J, rows, 0.02, kios)
    for position in (0, len(rows) // 2, len(rows) - 1):
        row = int(rows[position])
        report = fisher_tools.fisher_report(J[position], signal[row], 0.02,
                                            max(kios[row], 1.0))
        if not report["positive_definite"]:
            continue
        assert np.isclose(batch["crlb_log_rho"][position], report["crlb"][0],
                          rtol=1e-8)
        assert np.isclose(batch["crlb_log_vi"][position], report["crlb_log_vi"],
                          rtol=1e-8)


@pytest.mark.slow
def test_production_stencil_coverage_matches_the_phase3_node_count():
    """233 (rho, V) nodes carry a central stencil -- the Phase-3 table's row count.

    `docs/provenance/figures/fisher_phase3/table3_1_...csv` has one row per
    (rho, V) node that had a profiled block, which is the same set of nodes a
    complete central stencil exists at.  Agreeing with it is independent
    evidence that this module differences the same neighbours Phase 1 did.
    """
    if not DEFAULT_LIBRARY.exists():
        pytest.skip(f"production library not present at {DEFAULT_LIBRARY}")
    labels = load_labels(DEFAULT_LIBRARY)
    grid = fisher_tools.build_node_grid(labels)
    coverage = fisher_tools.coverage(grid, 1)
    assert coverage["total"] == 18819
    assert coverage["complete"] == 11417

    n_kio_interior = len(grid.kios) - 2
    assert coverage["complete"] % n_kio_interior == 0
    nodes_with_stencils = coverage["complete"] // n_kio_interior
    assert nodes_with_stencils == 233

    table = Path("docs/provenance/figures/fisher_phase3/"
                 "table3_1_rho_V_medians_full_stored_domain.csv")
    if table.exists():
        rows = [line for line in table.read_text().splitlines() if line.strip()]
        assert len(rows) - 1 == nodes_with_stencils


@pytest.mark.slow
def test_crn_subset_matches_the_stored_diagnostic_columns():
    """(20, 50) is one of the 8 stored pairs; (20, 80) is not."""
    if not DEFAULT_LIBRARY.exists():
        pytest.skip(f"production library not present at {DEFAULT_LIBRARY}")
    labels = load_labels(DEFAULT_LIBRARY)
    covered = labels.columns_for_pair(20.0, 50.0, b_max=6000.0)
    uncovered = labels.columns_for_pair(20.0, 80.0, b_max=6000.0)
    assert fisher_tools.build_crn_subset(DEFAULT_LIBRARY, labels,
                                         covered).n_covered == len(covered)
    assert fisher_tools.build_crn_subset(DEFAULT_LIBRARY, labels,
                                         uncovered).n_covered == 0

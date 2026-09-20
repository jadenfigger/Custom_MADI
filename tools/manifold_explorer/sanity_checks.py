"""Sanity checks and timings for the explorer, run against the real library.

    python -m tools.manifold_explorer.sanity_checks

Prints, and does not assert, so the degeneracy result is whatever it is:

1. Column-loader equivalence -- cached columns against the original file.
2. Noise limits -- a tiny sigma should keep only the reference entry, a huge
   sigma should keep everything.
3. The rho-V degeneracy -- slice around a mid-grid entry at one Delta, then
   add a second Delta, and report how the (rho, V) spread changes.
4. Load time and peak memory for a representative 50-column request.
5. The Fisher matrix at the same node: CRLB, spectrum, the constant-v_i angle,
   and how the predicted width compares with the measured slice width.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import fisher as fisher_tools
from . import slice as slicing
from .build_column_cache import DEFAULT_LIBRARY
from .columns import NpzColumnReader, cache_is_valid, load_labels, open_reader


def _rss_gb() -> float:
    try:
        import psutil
    except ImportError:
        return float("nan")
    return psutil.Process().memory_info().rss / 1e9


def _heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    labels = load_labels(args.library)
    valid, note = cache_is_valid(args.library, args.cache_dir)
    reader = open_reader(args.library, args.cache_dir)
    print(f"library : {args.library}")
    print(f"reader  : {reader.kind} ({note})")
    print(f"matrix  : {labels.n_entries} entries x {labels.n_columns} columns")
    print(f"RSS after opening: {_rss_gb():.3f} GB")

    # -- 1. equivalence ----------------------------------------------------
    _heading("1. Column loader vs the original file")
    for delta, Delta in [(20.0, 50.0), (4.0, 20.0)]:
        columns = labels.columns_for_pair(delta, Delta)
        started = time.time()
        cached = reader.read(columns)
        cache_seconds = time.time() - started
        direct_reader = NpzColumnReader(args.library)
        started = time.time()
        direct = direct_reader.read(columns)
        npz_seconds = time.time() - started
        direct_reader.close()
        print(f"  (delta={delta:g}, Delta={Delta:g}): {len(columns)} columns x "
              f"{cached.shape[0]} rows  identical={np.array_equal(cached, direct)}  "
              f"cache {cache_seconds * 1000:.1f} ms vs npz {npz_seconds:.2f} s")

    rng = np.random.default_rng(7)
    scattered = np.sort(rng.choice(labels.n_columns, 50, replace=False))
    started = time.time()
    block = reader.read(scattered)
    print(f"  50 scattered columns from the cache: {time.time() - started:.3f} s, "
          f"block {block.nbytes / 1e6:.1f} MB, RSS {_rss_gb():.3f} GB")

    # -- 2. noise limits ---------------------------------------------------
    _heading("2. Noise limits around a clicked entry")
    eligible = np.flatnonzero(slicing.candidate_mask(labels, 0.40, 0.99))
    columns = labels.columns_for_pair(20.0, 50.0, b_max=6000.0)
    measured = reader.read(columns)[eligible]
    centre_row = _mid_grid_row(labels, eligible)
    centre = int(np.flatnonzero(eligible == centre_row)[0])
    reference = measured[centre]
    print(f"  reference row {centre_row}: rho={labels.nominal_rhos[centre_row]:.4g}, "
          f"V={labels.nominal_Vs[centre_row]:.4g}, k_io={labels.kios[centre_row]:.4g}")
    for sigma in (1e-12, 1e-6, 1e-4, 0.005, 0.02, 1e6):
        result = slicing.slice_manifold(measured, reference, labels, eligible,
                                        threshold=len(columns),
                                        sigma_measurement=sigma)
        print(f"  sigma={sigma:<8g} survivors={result.n_survivors:6d}"
              f"  of {len(eligible)}")

    # -- 3. the rho-V degeneracy ------------------------------------------
    _heading("3. rho-V spread at one Delta, then at two")
    single = labels.columns_for_pair(20.0, 50.0, b_max=6000.0)
    second = labels.columns_for_pair(20.0, 20.0, b_max=6000.0)
    both = np.concatenate([single, second])
    for name, chosen in [("Delta=50 only", single),
                         ("Delta=20 only", second),
                         ("Delta=50 + Delta=20", both)]:
        measured = reader.read(chosen)[eligible]
        variance = reader.read_variance(chosen)[eligible]
        reference = measured[centre]
        result = slicing.slice_manifold(
            measured, reference, labels, eligible, threshold=1.0 * len(chosen),
            sigma_measurement=0.01, variance_block=variance)
        rho_stats = result.parameter_stats["rho"]
        V_stats = result.parameter_stats["V"]
        kio_stats = result.parameter_stats["k_io"]
        print(f"  {name:22s} n_cols={len(chosen):3d} survivors={result.n_survivors:5d}"
              f"  rho max/min={rho_stats['log_width']:7.3f}"
              f"  V max/min={V_stats['log_width']:7.3f}"
              f"  k_io std={kio_stats['std']:6.2f}")
        rows = result.survivor_rows
        if rows.size > 2:
            print(f"    {_correlation_note(labels, rows)}")

    # -- 4. representative request ----------------------------------------
    _heading("4. Representative request: 50 measured + 3 display columns")
    measured_columns = np.concatenate([
        labels.columns_for_pair(20.0, 50.0),
        labels.columns_for_pair(20.0, 20.0),
    ])
    display_columns = labels.column_indices([(20.0, 50.0, 1000.0),
                                             (20.0, 50.0, 3000.0),
                                             (20.0, 20.0, 3000.0)])
    before = _rss_gb()
    started = time.time()
    fresh = open_reader(args.library, args.cache_dir)
    measured = fresh.read(measured_columns)[eligible]
    variance = fresh.read_variance(measured_columns)[eligible]
    display = fresh.read(display_columns)[eligible]
    load_seconds = time.time() - started

    started = time.time()
    result = slicing.slice_manifold(
        measured, measured[centre], labels, eligible, threshold=len(measured_columns),
        sigma_measurement=0.02, variance_block=variance,
        display_values=display,
        display_names=[labels.column_label(int(c)) for c in display_columns])
    slice_seconds = time.time() - started
    print(f"  {len(measured_columns)} measured + {len(display_columns)} display columns")
    print(f"  load  : {load_seconds:.3f} s")
    print(f"  slice : {slice_seconds * 1000:.1f} ms -> {result.n_survivors} survivors")
    print(f"  arrays: {(measured.nbytes + variance.nbytes + display.nbytes) / 1e6:.1f} MB")
    print(f"  RSS   : {before:.3f} GB before -> {_rss_gb():.3f} GB after")

    _fisher_section(args, labels, reader, eligible, centre_row)


def _fisher_section(args, labels, reader, eligible, centre_row):
    """Print the Fisher/CRLB quantities beside the measured slice width."""
    _heading("5. Fisher information, spectrum and CRLB")
    started = time.time()
    grid = fisher_tools.build_node_grid(labels)
    coverage = fisher_tools.coverage(grid, 1)
    print(f"  node grid + coverage in {time.time() - started:.2f}s: "
          f"{coverage['complete']}/{coverage['total']} entries "
          f"({coverage['fraction']:.1%}) have a complete k=1 central stencil")
    print(f"  missing per axis: {coverage['missing_per_axis']}")

    node = grid.node_of_row.get(int(centre_row))
    if node is None:
        print("  the reference row is not a cellular grid node")
        return

    for name, chosen in [("Delta=50 only", labels.columns_for_pair(20.0, 50.0, b_max=6000.0)),
                         ("Delta=50 + Delta=20",
                          np.concatenate([labels.columns_for_pair(20.0, 50.0, b_max=6000.0),
                                          labels.columns_for_pair(20.0, 20.0, b_max=6000.0)]))]:
        block = reader.read(chosen)
        J, missing = fisher_tools.jacobian(grid, block, node)
        if J is None:
            print(f"  {name}: no central stencil on {missing}")
            continue
        report = fisher_tools.fisher_report(J, block[centre_row], 0.01,
                                            labels.kios[centre_row])
        threshold = 1.0 * len(chosen)
        measured = slicing.slice_manifold(
            block[eligible], block[centre_row], labels, eligible,
            threshold=threshold, sigma_measurement=0.01)
        predicted = float(np.exp(2.0 * np.sqrt(threshold) * report["crlb"][0]))
        print(f"  {name:22s} n_cols={len(chosen):3d}")
        print(f"     CRLB log rho={report['crlb'][0]:.4g} log V={report['crlb'][1]:.4g} "
              f"k_io={report['crlb'][2]:.4g}   log v_i={report['crlb_log_vi']:.4g}")
        print(f"     eig(D F D)={np.array2string(report['eigenvalues'], precision=4)} "
              f"kappa={report['condition_number']:.4g} pd={report['positive_definite']}")
        print(f"     sloppy angle from constant-v_i: {report['sloppy_angle_deg']:.2f} deg "
              f"(3-D), {report['profiled_angle_deg']:.2f} deg (k_io profiled)")
        print(f"     rho width: Fisher predicts {predicted:.4g}, slice measures "
              f"{measured.parameter_stats['rho']['log_width']:.4g}")


def _mid_grid_row(labels, eligible: np.ndarray) -> int:
    """An entry near the middle of the retained (rho, V) band, k_io mid-range."""
    log_rho = np.log(labels.nominal_rhos[eligible])
    log_V = np.log(labels.nominal_Vs[eligible])
    kio = labels.kios[eligible]
    target = np.array([np.median(log_rho), np.median(log_V), 20.0])
    scale = np.array([1.0, 1.0, 0.05])
    distance = np.sum(((np.column_stack([log_rho, log_V, kio]) - target) * scale) ** 2,
                      axis=1)
    return int(eligible[int(np.argmin(distance))])


def _correlation_note(labels, rows: np.ndarray) -> str:
    """Do the survivors lie along a curve in (log rho, log V)?

    Reports the correlation of log rho with log V and the slope of the fitted
    line.  A slope near -1 is the constant rho*V (constant v_i) hyperbola that
    docs/fisher_phase3.md identifies as the degeneracy direction.
    """
    x = np.log(labels.nominal_rhos[rows])
    y = np.log(labels.nominal_Vs[rows])
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return "survivors sit at a single rho or V value"
    correlation = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    vi = labels.vis[rows]
    return (f"log rho vs log V: r={correlation:+.3f}, slope={slope:+.3f} "
            f"(-1 = constant v_i); v_i range {vi.min():.3f}..{vi.max():.3f}")


if __name__ == "__main__":
    main()

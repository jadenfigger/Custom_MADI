"""Tier-A regression checks for universal-library identifiability columns."""

from __future__ import annotations

import numpy as np

from madi.identifiability import analyze_library, resolve_acquisition_columns
from madi.library import LibraryEntry
from scripts.analyze_identifiability import build_fit_triples


def test_identifiability_resolves_delta_delta_b_columns_and_runs_unchanged_fisher_core() -> None:
    lib_delta_pairs = [(5.0, 15.0), (7.0, 25.0)]
    b_values = [0.0, 500.0]
    col_idx = resolve_acquisition_columns(
        [(7.0, 25.0, 500.0), (5.0, 15.0, 0.0)],
        lib_delta_pairs, b_values, n_b=2,
    )
    assert np.array_equal(col_idx, [3, 0])
    assert build_fit_triples("current", lib_delta_pairs, b_values) == [
        (5.0, 15.0, 0.0), (5.0, 15.0, 500.0),
        (7.0, 25.0, 0.0), (7.0, 25.0, 500.0),
    ]
    assert build_fit_triples(
        "custom", lib_delta_pairs, b_values, triples_str="5/15:0,7/25:500",
    ) == [(5.0, 15.0, 0.0), (7.0, 25.0, 500.0)]

    library = []
    for kio in (1.0, 2.0):
        for rho in (1.0e5, 2.0e5):
            for volume in (1.0, 2.0):
                base = kio + rho * 1.0e-5 + volume
                library.append(LibraryEntry(
                    kio, rho, volume,
                    np.asarray([base, 2.0 * base, 3.0 * base, 4.0 * base]),
                ))

    result = analyze_library(
        library, lib_delta_pairs, b_values, 2,
        [(5.0, 15.0, 0.0), (7.0, 25.0, 500.0)], sigma_m=0.02,
    )
    assert result.summary["fit_triples"] == [[5.0, 15.0, 0.0], [7.0, 25.0, 500.0]]
    assert result.summary["n_entries_analyzed"] == len(library)

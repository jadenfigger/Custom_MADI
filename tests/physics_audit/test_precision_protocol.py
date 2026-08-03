"""The MADI II Table-2 precision protocol, retained as an explicit audit gate.

The bundled slim file makes this runnable without loading the 10 GB full
matrix.  It preserves all current candidates but only four (delta, Delta, b)
columns, so its result is *deliberately marked non-comparable* to the paper.
The expected failure protects against accidentally presenting this dense-grid,
four-shell result as a reproduction of MADI II.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from madi.library import load_library, match_voxels_batch

from ._artifact import require_production_library


N_REALIZATIONS = 50_000
B_VALUES = [1_000.0, 1_500.0, 2_000.0, 2_500.0]
TRIPLES = [(20.0, 50.0, b) for b in B_VALUES]


def _require_slow() -> Path:
    if os.environ.get("MADI_AUDIT_SLOW") != "1":
        pytest.skip("set MADI_AUDIT_SLOW=1 for the 50,000-realization precision gate")
    path = require_production_library().with_name(
        "madi_dense_universal_delta20_D50_b1000-2500.npz"
    )
    if not path.is_file():
        pytest.skip(f"missing bundled slim library: {path}")
    return path


def _nearest_entry_index(library, target: np.ndarray) -> int:
    parameters = np.asarray([[entry.kio, entry.rho, entry.V] for entry in library])
    # The scale just selects a sensible nearest available grid point; it does
    # not change the reference matcher or residual.
    scale = np.asarray([1.0, 1e5, 0.1])
    return int(np.argmin(np.sum(((parameters - target) / scale) ** 2, axis=1)))


def _run_gaussian_reference_match(library, target, seed: int) -> dict[str, np.ndarray]:
    idx = _nearest_entry_index(library, np.asarray(target, dtype=float))
    truth = library[idx].vector.astype(float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(0, N_REALIZATIONS, 1_000):
        # Reference mode: normalized Gaussian S/N=50, no Rician correction,
        # no fitted S0, and linear-space exhaustive matching.
        measured = truth + rng.normal(0.0, 1.0 / 50.0, (1_000, len(truth)))
        out.append(
            match_voxels_batch(
                measured,
                library,
                [(20.0, 50.0)],
                B_VALUES,
                len(B_VALUES),
                TRIPLES,
                vi_min=0.0,
                vi_max=0.95,
                use_gpu=False,
            )
        )
    return {
        "input_nearest": np.asarray([library[idx].kio, library[idx].rho, library[idx].V]),
        "kio": np.concatenate([x[0] for x in out]),
        "rho": np.concatenate([x[1] for x in out]),
        "V": np.concatenate([x[2] for x in out]),
    }


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the current artifact has four high-b shells and a masked dense grid, "
        "not the MADI-II library/protocol required to reproduce Table 2"
    ),
)
def test_table2_precision_medians_match_the_published_reference() -> None:
    path = _require_slow()
    library = load_library(str(path))
    cortex = _run_gaussian_reference_match(library, [6.6, 1.3e5, 6.0], 20260802)
    wm = _run_gaussian_reference_match(library, [22.0, 6.9e5, 0.9], 20260803)

    # MADI II Table 2 returned medians: (kio, rho, V).
    reported_cortex = np.asarray([11.0, 1.6e5, 4.7])
    reported_wm = np.asarray([22.0, 5.2e5, 1.2])
    observed_cortex = np.asarray([np.median(cortex[k]) for k in ("kio", "rho", "V")])
    observed_wm = np.asarray([np.median(wm[k]) for k in ("kio", "rho", "V")])
    assert np.allclose(observed_cortex, reported_cortex, rtol=0.10, atol=0.01)
    assert np.allclose(observed_wm, reported_wm, rtol=0.10, atol=0.01)

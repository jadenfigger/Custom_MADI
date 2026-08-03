"""Rerunnable diagnostics for the co-located universal-library glioma maps.

The map folders themselves lack MAP metadata.  Their 123,601 nonzero voxels
match the sibling universal-library Bayesian metadata, so these tests report
the maps as a useful provisional diagnostic, not a fully provenance-closed
paper reproduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = ROOT / "data" / "outputs" / "madi_output_glioma_v3.0"


def _load_triplet(subdir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nib = pytest.importorskip("nibabel")
    root = MAP_ROOT / subdir
    if not root.is_dir():
        pytest.skip(f"missing map directory: {root}")
    rho = np.asanyarray(nib.load(root / "rho_map.nii.gz").dataobj)
    volume = np.asanyarray(nib.load(root / "V_map.nii.gz").dataobj)
    kio = np.asanyarray(nib.load(root / "kio_map.nii.gz").dataobj)
    mask = np.isfinite(rho) & np.isfinite(volume) & np.isfinite(kio) & (rho > 0) & (volume > 0) & (kio > 0)
    return rho[mask], volume[mask], kio[mask]


def test_universal_map_values_obey_its_stored_grid_bounds() -> None:
    rho, volume, kio = _load_triplet("map")
    vi = rho * volume * 1e-6
    assert len(vi) == 123_601
    assert np.all((vi >= 0.4) & (vi <= 0.95))
    assert np.all((rho >= 100_000.0) & (rho <= 3_000_000.0))
    assert np.all((volume >= 0.1898989) & (volume <= 9.0))
    assert np.all((kio >= 1.0) & (kio <= 100.0))


@pytest.mark.xfail(
    strict=True,
    reason="co-located universal MAP has median v_i~0.512 and ~49% below 0.5, not the published brain range",
)
def test_universal_map_has_published_brain_like_vi_distribution() -> None:
    rho, volume, _ = _load_triplet("map")
    vi = rho * volume * 1e-6
    assert 0.80 <= np.median(vi) <= 0.90
    assert np.mean((vi < 0.5) | (vi > 0.95)) <= 0.10

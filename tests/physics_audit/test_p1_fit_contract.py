"""Tier-A acceptance tests for P1 fit transparency and derived endpoints."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from madi.biomarkers import (
    derive_voxelwise_biomarkers,
    summarise_voxelwise_biomarkers,
)
from madi.config import SimConfig
from madi.library import LibraryEntry, _save_library, resolve_grid_columns
from scripts.fit_data import _direction_contract, b_tensor_diagnostic


pytestmark = pytest.mark.tier_a


ROOT = Path(__file__).resolve().parents[2]


def _write_tiny_fit_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    nib = pytest.importorskip("nibabel")
    library_path = tmp_path / "tiny_library.npz"
    dwi_path = tmp_path / "dwi.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    bval_path = tmp_path / "dwi.bval"
    bvec_path = tmp_path / "dwi.bvec"

    cfg = SimConfig(
        small_deltas=[20.0], big_deltas=[50.0], b_values=[0.0, 1000.0],
        T_max_ms=80.0,
    )
    entries = [
        LibraryEntry(10.0, 500_000.0, 1.0, np.array([1.0, 0.50]), vi=0.50),
        LibraryEntry(20.0, 750_000.0, 1.0, np.array([1.0, 0.60]), vi=0.75),
    ]
    _save_library(entries, str(library_path), cfg=cfg)

    dwi = np.zeros((2, 2, 1, 2), dtype=np.float32)
    dwi[..., 0] = 100.0
    dwi[..., 1] = 50.0
    nib.save(nib.Nifti1Image(dwi, np.eye(4)), dwi_path)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), np.eye(4)), mask_path)
    bval_path.write_text("0 1012\n", encoding="utf-8")
    bvec_path.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
    return library_path, dwi_path, mask_path, bval_path, bvec_path


def test_voxelwise_derivation_rejects_roi_median_products() -> None:
    biomarkers = derive_voxelwise_biomarkers(
        np.array([2.0, 10.0]), np.array([1.0e6, 0.5e6]), np.array([1.0, 2.0])
    )
    # Median of voxelwise products is not product of median parameter maps.
    summary = summarise_voxelwise_biomarkers(biomarkers, statistic="median")
    assert summary["kioV_pL_s_cell"] == 11.0
    with pytest.raises(TypeError, match="voxel-wise array"):
        derive_voxelwise_biomarkers(6.0, 7.5e5, 1.5)
    with pytest.raises(TypeError, match="VoxelwiseBiomarkers"):
        summarise_voxelwise_biomarkers({"kio": 6.0})


def test_snap_resolver_reports_and_enforces_declared_tolerances() -> None:
    cols, events = resolve_grid_columns(
        [(20.5, 50.0, 1012.0)], [(20.0, 50.0)], [1000.0], n_b=1,
    )
    assert cols.tolist() == [0]
    assert {event["axis"] for event in events} == {"b", "timing"}
    with pytest.raises(ValueError, match="30"):
        resolve_grid_columns([(20.0, 50.0, 1031.0)], [(20.0, 50.0)], [1000.0], 1)
    with pytest.raises(ValueError, match="1.5"):
        resolve_grid_columns([(21.6, 50.0, 1000.0)], [(20.0, 50.0)], [1000.0], 1)


def test_direction_contract_refuses_ambiguous_data_and_records_b_tensor() -> None:
    # Three collinear directions unambiguously identify a single-direction run.
    bvecs = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    report = b_tensor_diagnostic(bvecs, np.arange(3))
    report.update(delta_ms=20.0, Delta_ms=50.0, b_s_mm2=1000.0)
    assert report["n_unique_directions"] == 1
    contract = _direction_contract(None, [report])
    assert contract["declared_scheme"] == "single_direction"
    assert contract["single_direction_madi_iii_bias_percent"]["vi_percent"] == -8.4

    # Two non-collinear directions cannot be inferred into any permitted
    # direction convention, so an explicit declaration is mandatory.
    ambiguous = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    ambiguous_report = b_tensor_diagnostic(ambiguous, np.arange(2))
    ambiguous_report.update(delta_ms=20.0, Delta_ms=50.0, b_s_mm2=1000.0)
    with pytest.raises(ValueError, match="required"):
        _direction_contract(None, [ambiguous_report])

    orthogonal = np.eye(3)
    report = b_tensor_diagnostic(orthogonal, np.arange(3))
    report.update(delta_ms=20.0, Delta_ms=50.0, b_s_mm2=1000.0)
    contract = _direction_contract(None, [report])
    assert contract["declared_scheme"] == "orthogonal_3"
    assert contract["inferred_from_bvecs"] is True


def test_map_fit_writes_contract_sidecars_and_voxelwise_maps(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    library, dwi, mask, bval, bvec = _write_tiny_fit_inputs(tmp_path)
    out = tmp_path / "maps"
    command = [
        sys.executable, "-m", "scripts.fit_data",
        "--fit",
        "--library", str(library),
        "--input", f"20,50:{dwi}:{bval}:{bvec}",
        "--small-delta", "20",
        "--mask", str(mask),
        "--direction-scheme", "single_direction",
        "--out", str(out),
        "--run-name", "contract",
        "--device", "cpu",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "SNAP[b]" in completed.stdout

    json_path = out / "contract.json"
    log_path = out / "contract.log"
    assert json_path.is_file()
    assert log_path.is_file()
    assert not (out / "fit_metadata.json").exists()
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["snap_summary"]["n_snapped_b_shells"] == 1
    assert record["fit_configuration"]["direction_scheme"]["declared_scheme"] == "single_direction"
    assert record["fit_configuration"]["direction_scheme"]["single_direction_madi_iii_bias_percent"]["ADC_percent"] == 18.1
    assert record["fit_configuration"]["linear_or_log_space"] == "linear"
    assert record["fit_configuration"]["S0_mode"] == "fixed"
    assert record["fit_configuration"]["rician_correction"] is False
    assert set(record["results_summary"]["derived_maps_written"]) == {
        "vi_map.nii.gz", "kioV_map.nii.gz", "kioVrho_map.nii.gz",
        "water_efflux_nmol_s_cell_map.nii.gz",
    }
    assert "SNAP[b]" in log_path.read_text(encoding="utf-8")
    assert "SINGLE-DIRECTION INPUT" in log_path.read_text(encoding="utf-8")

    kio = np.asanyarray(nib.load(out / "kio_map.nii.gz").dataobj)
    rho = np.asanyarray(nib.load(out / "rho_map.nii.gz").dataobj)
    volume = np.asanyarray(nib.load(out / "V_map.nii.gz").dataobj)
    vi = np.asanyarray(nib.load(out / "vi_map.nii.gz").dataobj)
    kioV = np.asanyarray(nib.load(out / "kioV_map.nii.gz").dataobj)
    kioVrho = np.asanyarray(nib.load(out / "kioVrho_map.nii.gz").dataobj)
    water = np.asanyarray(nib.load(out / "water_efflux_nmol_s_cell_map.nii.gz").dataobj)
    inside = np.ones(kio.shape, dtype=bool)
    assert np.allclose(vi[inside], rho[inside] * volume[inside] * 1.0e-6)
    assert np.allclose(kioV[inside], kio[inside] * volume[inside])
    assert np.allclose(kioVrho[inside], kio[inside] * volume[inside] * rho[inside])
    assert np.allclose(water[inside], 0.042 * kio[inside] * volume[inside])

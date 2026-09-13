"""Tests for the Phase-4 runner: its per-column Fisher sums, joins and guards.

`scripts/run_fisher_phase4.py` is orchestration, but three of its pieces are the
kind of code this project has been burned by before: a streaming sum that must
equal the audited Fisher primitives, a join from fitted voxels to Fisher nodes,
and the guards that stop a fit of different voxels from being compared.  Each is
pinned here on synthetic inputs.
"""
from __future__ import annotations

import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from madi.fisher_crlb import (amplitude_marginal_fisher, canonical_grid, derivative_variance,
                              fisher_matrix, pack_fisher, unpack_fisher)
from scripts.run_fisher_phase4 import (acquisition_fisher, at_rows, condition_contrast, fisher_layer,
                                       load_arm, node_rows, ridge_displacement)


def test_acquisition_fisher_weights_each_column_by_its_own_averaging() -> None:
    """Per-shell averaging must enter as a per-column weight, exactly.

    A measured acquisition averages a different number of directions per shell,
    which Phase 3's one-noise-level-per-pair `Domain` cannot express.  The runner
    therefore calls Phase 2's `pair_contributions` once per column; the sum must
    equal `fisher_matrix` with `sigma_c = sigma_norm / sqrt(n_c)`, and the
    amplitude block must equal `amplitude_marginal_fisher`'s.
    """
    rng = np.random.default_rng(90)
    n_entries, n_b, n_pairs, n_ensembles = 12, 4, 2, 8
    b_values = np.array([0.0, 500.0, 1000.0, 1500.0])
    vectors = rng.uniform(0.05, 1.0, size=(n_pairs * n_b, n_entries))
    variance = rng.uniform(1e-7, 1e-6, size=(n_pairs * n_b, n_entries))
    table = {
        "nodes": np.zeros((2, 3), dtype=int), "centre": np.asarray([0, 1]),
        "minus_rho": np.asarray([2, 3]), "plus_rho": np.asarray([4, 5]),
        "minus_V": np.asarray([6, 7]), "plus_V": np.asarray([8, 9]),
        "minus_k_io": np.asarray([10, 11]), "plus_k_io": np.asarray([0, 1]),
        "step_rho": np.asarray([0.11, 0.12]), "step_V": np.asarray([0.15, 0.16]),
        "step_k_io": np.asarray([2.0, 2.0]),
    }
    pair, sigma_norm = 1, 0.04
    shells = {500.0: 6, 1000.0: 18, 1500.0: 30}
    tissue, amplitude, _ = acquisition_fisher(table, vectors, variance, n_ensembles, pair=pair, n_b=n_b,
                                              b_values=b_values, shells=shells, sigma_norm=sigma_norm)
    axes = (("rho", "step_rho"), ("V", "step_V"), ("k_io", "step_k_io"))
    for node in range(2):
        J, var, sigma, signal = [], [], [], []
        for b, averages in shells.items():
            column = pair * n_b + int(np.flatnonzero(b_values == b)[0])
            J.append([(vectors[column, table[f"plus_{a}"][node]] - vectors[column, table[f"minus_{a}"][node]])
                      / table[s][node] for a, s in axes])
            var.append([float(derivative_variance(np.asarray([variance[column, table[f"minus_{a}"][node]]]),
                                                  np.asarray([variance[column, table[f"plus_{a}"][node]]]),
                                                  None, None, denominator=table[s][node],
                                                  n_ensembles=n_ensembles)[0]) for a, s in axes])
            sigma.append(sigma_norm / np.sqrt(averages))
            signal.append(vectors[column, table["centre"][node]])
        J, var, sigma, signal = map(np.asarray, (J, var, sigma, signal))
        assert np.allclose(unpack_fisher(tissue)[node], fisher_matrix(J, sigma, var), rtol=1e-12, atol=0)
        reference = amplitude_marginal_fisher(J, signal, sigma, variance=var)
        assert np.allclose(amplitude[node, :3], reference["F_theta_s0"], rtol=1e-12)
        assert np.isclose(amplitude[node, 3], reference["F_s0_s0"], rtol=1e-12)


def test_fisher_layer_regimes_are_ordered_and_collapse_to_the_tissue_matrix() -> None:
    """Known amplitude is the tissue matrix itself; marginalizing can only widen every bound."""
    rng = np.random.default_rng(91)
    J = rng.normal(size=(5, 10, 3))
    signal = rng.uniform(0.2, 1.0, size=(5, 10))
    tissue = pack_fisher(np.einsum("nci,ncj->nij", J, J))
    amplitude = np.concatenate([np.einsum("nci,nc->ni", J, signal), (signal ** 2).sum(axis=1)[:, None]], axis=1)
    layer = fisher_layer(tissue, amplitude, np.full(5, 5.0), {"known_amplitude": np.inf, "unknown_amplitude": 0.0})
    known, unknown = layer["known_amplitude"], layer["unknown_amplitude"]
    assert np.array_equal(known["packed"], tissue)
    both = known["positive"] & unknown["positive"]
    assert both.any()
    assert np.all(unknown["crlb"][both] >= known["crlb"][both] * (1 - 1e-12))
    assert np.all(unknown["log_vi_bound"][both] >= known["log_vi_bound"][both] * (1 - 1e-12))


def test_node_rows_joins_exact_labels_and_withholds_nodes_without_a_fisher_matrix() -> None:
    """A voxel reads its own node's Fisher quantities, or none — never a neighbour's."""
    rhos, volumes, kios, _ = canonical_grid()
    table = {"nodes": np.array([[10, 20, 5], [11, 20, 5]])}
    rho = np.array([rhos[10], rhos[11], rhos[12], 0.0])
    volume = np.array([volumes[20], volumes[20], volumes[20], 1.0])
    kio = np.array([kios[5], kios[5], kios[5], kios[5]])
    joined = node_rows(table, rho, volume, kio)
    assert joined["rows"].tolist() == [0, 1, -1, -1]
    assert joined["has_node"].tolist() == [True, True, True, False]
    assert joined["has_fisher"].tolist() == [True, True, False, False]
    assert np.allclose(joined["log_rho_offset"][:3], 0.0) and np.isnan(joined["log_rho_offset"][3])
    values = at_rows(np.array([[1.0, 2.0], [3.0, 4.0]]), joined["rows"])
    assert values[:2].tolist() == [[1.0, 2.0], [3.0, 4.0]] and np.isnan(values[2:]).all()


def _write_arm(root, name, mask, *, sha, status="completed", volume=None):
    directory = root / name
    directory.mkdir(parents=True)
    n = int(mask.sum())
    maps = {"kio_map": np.full(n, 10.0), "rho_map": np.full(n, 1e5),
            "V_map": np.arange(1, n + 1, dtype=float) if volume is None else volume,
            "residual_map": np.full(n, 1e-3)}
    if volume is not None:
        maps["rho_map"] = np.where(volume == 0, 0.0, 1e5)
    for stem, flat in maps.items():
        vol = np.zeros(mask.shape, dtype=np.float32)
        vol[mask] = flat
        nib.save(nib.Nifti1Image(vol, np.eye(4)), directory / f"{stem}.nii.gz")
    sidecar = {"status": status, "input_data_provenance": {"mask": {"sha256": sha}},
               "fit_configuration": {"method": "map", "S0_mode": "fixed", "trust_floor_mask": None,
                                     "fit_triples_delta_Delta_b": [[20.0, 50.0, 500.0]]}}
    (directory / f"{name}.json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_load_arm_refuses_a_fit_of_different_voxels_and_marks_unfitted_voxels(tmp_path) -> None:
    """The guard that the `madi_output_glioma_v4.0/map` provenance error calls for.

    That output is an edema sub-187 fit filed under a glioma directory, and its
    zero voxels look exactly like "not fitted".  An arm fitted with any mask other
    than the declared one must be refused by hash, an unfinished run refused by
    status, and a `rho = V = 0` voxel carried as unfitted rather than as a volume.
    """
    mask = np.zeros((4, 4, 2), dtype=bool)
    mask[1:3, 1:3, :] = True
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), mask_path)
    sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()

    volume = np.array([5.0, 0.0, 25.0, 30.0, 1.0, 2.0, 3.0, 4.0])
    _write_arm(tmp_path, "baseline_map", mask, sha=sha, volume=volume)
    arm = load_arm(tmp_path, "baseline_map", mask, sha)
    assert np.allclose(arm["V"], volume)
    assert arm["fitted"].tolist() == [True, False, True, True, True, True, True, True]
    assert arm["not_fitted_count"] == 1

    with pytest.raises(RuntimeError, match="refusing to compare fits of different voxels"):
        load_arm(tmp_path, "baseline_map", mask, "0" * 64)
    _write_arm(tmp_path, "still_running", mask, sha=sha, status="running")
    with pytest.raises(RuntimeError, match="not 'completed'"):
        load_arm(tmp_path, "still_running", mask, sha)
    assert load_arm(tmp_path, "absent_arm", mask, sha) is None


def test_condition_contrast_counts_resolved_and_new_blow_ups_on_shared_voxels() -> None:
    """A shrinkage must be fewer blow-ups, not a reshuffling of which voxels blow up."""
    baseline = {"V": np.array([25.0, 25.0, 5.0, 5.0, 30.0]), "fitted": np.array([True, True, True, True, False])}
    varied = {"V": np.array([25.0, 5.0, 25.0, 5.0, 5.0]), "fitted": np.ones(5, dtype=bool)}
    out = condition_contrast(baseline, varied, "synthetic")
    at_20 = next(row for row in out["by_cutoff"] if row["cutoff_pL"] == 20.0)
    assert out["voxels_fitted_in_both"] == 4
    assert (at_20["stayed_blown_up"], at_20["resolved"], at_20["newly_blown_up"]) == (1, 1, 1)
    assert at_20["fraction_before"] == 0.5 and at_20["fraction_after"] == 0.5
    assert at_20["relative_change"] == 0.0
    assert not out["shrinks_at_every_cutoff"] and not out["grows_at_every_cutoff"]


def test_ridge_displacement_reports_moves_along_the_hyperbola() -> None:
    from_arm = {"rho": np.array([1e5, 1e5, 1e5]), "V": np.array([5.0, 5.0, 5.0]), "fitted": np.ones(3, dtype=bool)}
    to_arm = {"rho": np.array([2e5, 1e5, 1e5]), "V": np.array([2.5, 5.0, 10.0]), "fitted": np.ones(3, dtype=bool)}
    out = ridge_displacement(from_arm, to_arm, "synthetic")
    assert out["voxels_moved"] == 2
    assert np.isclose(out["angle_to_hyperbola_deg"]["min"], 0.0, atol=1e-6)
    assert np.isclose(out["angle_to_hyperbola_deg"]["max"], 45.0)
    assert out["fraction_within_10_deg"] == 0.5
    assert np.isclose(out["abs_d_log_vi"]["min"], 0.0, atol=1e-12)
    assert np.isclose(out["abs_d_log_vi"]["max"], np.log(2.0))

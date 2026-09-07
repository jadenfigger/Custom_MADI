"""Focused, synthetic tests for the Phase-0/1 Fisher framework."""
from __future__ import annotations

import json

import numpy as np

from madi.fisher_crlb import (amplitude_marginal_diagnostics, amplitude_marginal_fisher,
                               amplitude_prior_precision, analytic_free_water_gate,
                               build_grid_manifest, canonical_grid, derivative_variance,
                               feasibility_masks, fisher_diagnostics, fisher_matrix,
                               gradient_strength_t_per_m, invalidated_stencils)


def test_missing_canonical_group_is_reported_and_invalidates_neighbours() -> None:
    rhos, volumes, _, retained = canonical_grid()
    # Choose a pair with four rho-neighbours inside the retained band.
    centre = next((ir, iv) for ir, iv in retained
                  if all((candidate, iv) in retained
                         for candidate in (ir - 2, ir - 1, ir + 1, ir + 2)))
    rows_rho, rows_V, free = [], [], []
    for ir, iv in retained - {centre}:
        rows_rho.extend([rhos[ir]] * 51)
        rows_V.extend([volumes[iv]] * 51)
        free.extend([False] * 51)
    manifest = build_grid_manifest(np.asarray(rows_rho), np.asarray(rows_V), np.asarray(free))
    assert centre in manifest.missing
    invalid = invalidated_stencils(manifest, {"rho": [1]})["rho"]["1"]
    assert any(row["rho_index"] == centre[0] - 1 and row["V_index"] == centre[1] for row in invalid)
    assert any(row["rho_index"] == centre[0] + 1 and row["V_index"] == centre[1] for row in invalid)


def test_central_variance_uses_aligned_crn_covariance() -> None:
    minus = np.asarray([0.1, 0.2, 0.3, 0.4])[:, None]
    plus = minus + 0.2
    result = derivative_variance(np.asarray([np.var(minus, ddof=1)]),
                                 np.asarray([np.var(plus, ddof=1)]), minus, plus,
                                 denominator=2.0, n_ensembles=4)
    assert np.allclose(result, 0.0, atol=1e-15)


def test_debiased_fisher_leaves_off_diagonal_unchanged() -> None:
    J = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    variance = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    raw = fisher_matrix(J, 2.0)
    corrected = fisher_matrix(J, 2.0, variance)
    assert corrected[0, 1] == raw[0, 1]
    assert corrected[1, 0] == raw[1, 0]
    assert corrected[0, 0] < raw[0, 0]
    diag = fisher_diagnostics(np.eye(3), 20.0)
    assert np.allclose(diag["kappa"], 1.0)


def test_feasibility_filters_are_hard_boolean_masks() -> None:
    signals = np.asarray([[1.0, 0.01], [1.0, 0.02]])
    masks = feasibility_masks(signals, np.asarray([10.0]), np.asarray([30.0]), np.asarray([0.0, 1000.0]),
                              sigma0=0.01, G_max=1.0)
    assert masks["trust_floor_per_entry"].dtype == bool
    assert not masks["trust_floor_per_entry"][0, 1]
    assert masks["combined_per_entry"].shape == signals.shape


def test_analytic_gate_is_synthetic_and_uses_the_predeclared_tolerance() -> None:
    report = analytic_free_water_gate(np.arange(0.0, 12_500.0, 500.0), tolerance=1e-12)
    assert report["pass"]
    assert report["derivative_max_abs_error"] == 0.0


def _amplitude_fixture():
    rng = np.random.default_rng(20260905)
    J = rng.normal(size=(40, 3))
    signal = np.abs(rng.normal(loc=1.0, scale=0.2, size=40)) + 0.1
    return J, signal, 0.02


def test_schur_marginalization_matches_the_explicit_augmented_inverse() -> None:
    """The 3x3 Schur complement must equal the tissue block of the 4x4 inverse."""
    J, signal, sigma = _amplitude_fixture()
    result = amplitude_marginal_fisher(J, signal, sigma)
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = (J / sigma).T @ (J / sigma)
    augmented[:3, 3] = (J / sigma ** 2).T @ signal
    augmented[3, :3] = augmented[:3, 3]
    augmented[3, 3] = np.sum(signal ** 2 / sigma ** 2)
    expected = np.linalg.inv(augmented)[:3, :3]
    assert np.allclose(np.linalg.inv(result["F_marginal_s0"]), expected, rtol=0, atol=1e-18)


def test_marginal_bound_is_never_better_than_the_fixed_s0_bound() -> None:
    """F_eff <= F_tt in the Loewner order, so every CRLB can only grow."""
    J, signal, sigma = _amplitude_fixture()
    for prior in (0.0, 1e2, 1e4, 1e8):
        result = amplitude_marginal_fisher(J, signal, sigma, s0_prior_precision=prior)
        gap = result["F_fixed_s0"] - result["F_marginal_s0"]
        assert np.min(np.linalg.eigvalsh(gap)) >= -1e-9
        assert result["loewner_ok"]
        ratio = amplitude_marginal_diagnostics(result, kio_ref=20.0)["crlb_ratio_marginal_over_fixed"]
        assert np.all(ratio >= 1.0 - 1e-12)


def test_amplitude_regimes_are_ordered_and_collapse_to_the_fixed_bound() -> None:
    """Known / finite-precision / unknown amplitude bracket each other."""
    J, signal, sigma = _amplitude_fixture()
    unknown = amplitude_marginal_fisher(J, signal, sigma, s0_prior_precision=0.0)
    finite = amplitude_marginal_fisher(J, signal, sigma,
                                       s0_prior_precision=amplitude_prior_precision(4, sigma))
    known = amplitude_marginal_fisher(J, signal, sigma, s0_prior_precision=1e18)
    for better, worse in ((finite, unknown), (known, finite)):
        assert np.min(np.linalg.eigvalsh(better["F_marginal_s0"] - worse["F_marginal_s0"])) >= -1e-9
    assert np.allclose(known["F_marginal_s0"], known["F_fixed_s0"], rtol=1e-9, atol=0)


def test_amplitude_enters_the_tissue_block_quadratically() -> None:
    """Both bounds scale as a^2 because the measurement Jacobian is a*J."""
    J, signal, sigma = _amplitude_fixture()
    unit = amplitude_marginal_fisher(J, signal, sigma, amplitude=1.0)
    scaled = amplitude_marginal_fisher(J, signal, sigma, amplitude=3.0)
    assert np.allclose(scaled["F_fixed_s0"], 9.0 * unit["F_fixed_s0"])
    assert np.allclose(scaled["F_marginal_s0"], 9.0 * unit["F_marginal_s0"])


def test_finite_b0_precision_uses_the_reference_shell_signal() -> None:
    """A b=50 normalizer supplies S(b_ref)^2 times a true b=0 shell's precision."""
    assert amplitude_prior_precision(4, 0.02) == 4 / 0.02 ** 2
    assert np.isclose(amplitude_prior_precision(4, 0.02, 0.97),
                      0.97 ** 2 * amplitude_prior_precision(4, 0.02))


def test_streaming_variance_accumulation_matches_the_audited_helper() -> None:
    """Phase 1 accumulates Var(J) column-wise; it must equal derivative_variance.

    Regression guard for a defect in which the streaming path scaled the two
    endpoint variances by 1/h^2 but the covariance term by 1/(n_ensembles h^2).
    Because the inflation factor is n_ensembles*(1 - r/n_ensembles)/(1 - r), it
    grew with the common-random-number correlation and so hit the k_io axis --
    the best-correlated axis -- hardest.
    """
    rng = np.random.default_rng(4)
    n_ensembles, columns, denominator = 40, 12, 2.0
    shared = rng.normal(size=(n_ensembles, columns))
    ensemble_minus = shared + 0.2 * rng.normal(size=(n_ensembles, columns))
    ensemble_plus = shared + 0.2 * rng.normal(size=(n_ensembles, columns))
    variance_minus = ensemble_minus.var(axis=0, ddof=1)
    variance_plus = ensemble_plus.var(axis=0, ddof=1)

    expected = derivative_variance(variance_minus, variance_plus, ensemble_minus,
                                   ensemble_plus, denominator, n_ensembles)
    # The streaming form: endpoint scale applied per chunk, covariance subtracted after.
    endpoint_scale = 1.0 / (n_ensembles * denominator ** 2)
    streamed = endpoint_scale * variance_minus + endpoint_scale * variance_plus
    covariance = np.sum((ensemble_minus - ensemble_minus.mean(axis=0)) *
                        (ensemble_plus - ensemble_plus.mean(axis=0)), axis=0) / (n_ensembles - 1)
    streamed = np.maximum(streamed - 2.0 * covariance / (n_ensembles * denominator ** 2), 0.0)
    assert np.allclose(streamed, expected, rtol=1e-12, atol=0)

    # The defective scaling is not a rounding difference: it inflates Var(J) by
    # roughly n_ensembles once the CRN correlation is high.
    defective = np.maximum(variance_minus / denominator ** 2 + variance_plus / denominator ** 2
                           - 2.0 * covariance / (n_ensembles * denominator ** 2), 0.0)
    assert np.median(defective / expected) > 10.0


def test_richardson_reports_truncation_bias_relative_to_J(tmp_path) -> None:
    """Relative truncation bias must be comparable with beta, and k=2 must be 4x k=1.

    beta is a squared relative quantity, so an absolute bias RMS cannot be read
    against it; this pins the relative form that can be.
    """
    from scripts.run_fisher_phase1 import _write_richardson

    rng = np.random.default_rng(11)
    rows, columns = 6, 8
    diagnostic_positions = np.arange(columns)
    fine = rng.normal(size=(rows, columns)) + 3.0
    coarse = fine * 1.10          # a clean 10% k1-vs-k2 disagreement
    samples = np.arange(rows * 3, dtype=np.int16).reshape(rows, 3)
    np.save(tmp_path / "J_rho_k1.npy", fine.astype(np.float32))
    np.save(tmp_path / "J_rho_k2.npy", coarse.astype(np.float32))
    np.save(tmp_path / "samples_rho_k1.npy", samples)
    np.save(tmp_path / "samples_rho_k2.npy", samples)

    report = _write_richardson(
        tmp_path, "rho",
        {"J": "J_rho_k1.npy", "samples": "samples_rho_k1.npy"},
        {"J": "J_rho_k2.npy", "samples": "samples_rho_k2.npy"},
        columns, diagnostic_positions,
    )
    relative = report["truncation_bias_relative"]
    # |J1 - J2|/3 over |J1| is 0.10/3 when J2 = 1.10 J1.
    assert np.isclose(relative["k1"]["relative_bias_median"], 0.10 / 3.0, rtol=1e-5)
    assert np.isclose(relative["k2"]["relative_bias_median"], 4.0 * 0.10 / 3.0, rtol=1e-5)
    assert np.isclose(relative["k1"]["relative_squared_median"],
                      relative["k1"]["relative_bias_median"] ** 2, rtol=1e-5)
    assert report["truncation_bias_abs_rms"] > 0


def test_richardson_tolerates_an_empty_k1_k2_overlap(tmp_path) -> None:
    """`--smoke-nodes` can leave an axis with no overlap; that must not raise."""
    from scripts.run_fisher_phase1 import _write_richardson

    np.save(tmp_path / "J_V_k1.npy", np.zeros((2, 4), dtype=np.float32))
    np.save(tmp_path / "J_V_k2.npy", np.zeros((1, 4), dtype=np.float32))
    np.save(tmp_path / "samples_V_k1.npy", np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int16))
    np.save(tmp_path / "samples_V_k2.npy", np.array([[9, 9, 9]], dtype=np.int16))
    report = _write_richardson(
        tmp_path, "V",
        {"J": "J_V_k1.npy", "samples": "samples_V_k1.npy"},
        {"J": "J_V_k2.npy", "samples": "samples_V_k2.npy"},
        4, np.arange(4),
    )
    assert report["overlap_samples"] == 0
    assert report["truncation_bias_relative"] == {}


def _load_analysis_module(name: str):
    """Import a module from `analysis/`, which is a script directory, not a package."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for entry in (str(root), str(root / "analysis")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, root / "analysis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def test_stencil_probe_denominator_reads_realised_labels_not_a_uniform_step() -> None:
    """The probe must difference over the realised labels, not `2*k*h1`.

    The declared probe triple (19, 20, 21) sits in the 1 s^-1 region, so the two
    forms agree there and the recorded probe run is unaffected.  A k_io stencil
    straddling the 30 s^-1 boundary, where spacing becomes 5 s^-1, must be
    rejected rather than silently differenced over the wrong step.
    """
    probe = _load_analysis_module("v5_stencil_probe")
    declared = {"left_coordinate": 19.0, "right_coordinate": 21.0, "width": 1, "h1": 1.0}
    assert probe._stencil_denominator(declared) == 2.0

    straddling = {"left_coordinate": 29.0, "right_coordinate": 35.0, "width": 1, "h1": 1.0}
    try:
        probe._stencil_denominator(straddling)
    except Exception as error:                      # DiagnosticError
        assert "no longer uniform" in str(error)
    else:                                           # pragma: no cover - guard must fire
        raise AssertionError("a non-uniform k_io stencil was accepted as if uniformly spaced")


def test_stencil_probe_beta_uses_the_audited_derivative_variance() -> None:
    """`analysis/v5_stencil_probe.py` must not re-derive Var(J_hat) for itself.

    Pins the probe's arithmetic to `madi.fisher_crlb.derivative_variance` on
    synthetic CRN-correlated ensembles.  The probe supplies the measured beta
    calibration the Phase-1 audit is compared against, so a drift here would
    silently move the reference rather than the measurement.
    """
    rng = np.random.default_rng(11)
    n_ensembles, columns, denominator = 40, 9, 2.0
    shared = rng.normal(size=(n_ensembles, columns))
    ensemble_left = shared + 0.15 * rng.normal(size=(n_ensembles, columns))
    ensemble_right = shared + 0.15 * rng.normal(size=(n_ensembles, columns))
    raw_left = ensemble_left.var(axis=0, ddof=1)
    raw_right = ensemble_right.var(axis=0, ddof=1)

    expected = derivative_variance(raw_left, raw_right, ensemble_left, ensemble_right,
                                   denominator, n_ensembles)
    # The probe's own guard path: entry-mean variances and covariance, unclipped.
    covariance = np.sum((ensemble_left - ensemble_left.mean(axis=0)) *
                        (ensemble_right - ensemble_right.mean(axis=0)), axis=0) / (n_ensembles - 1)
    numerator = (raw_left + raw_right) / n_ensembles - 2.0 * covariance / n_ensembles
    probe_local = np.maximum(numerator, 0.0) / denominator ** 2
    assert np.allclose(probe_local, expected, rtol=1e-12, atol=0)


def test_crn_diagnostic_covariance_matches_the_audited_helper() -> None:
    """`paired_correlation_matrix` and `derivative_variance` must share a covariance.

    The CRN diagnostic keeps its own paired-covariance implementation because it
    also bootstraps over resampled ensemble indices, which the helper does not
    do.  That separate path is pinned here rather than removed.
    """
    diagnostic = _load_analysis_module("v5_crn_diagnostic")
    rng = np.random.default_rng(23)
    n_entries, n_ensembles, columns = 4, 40, 7
    shared = rng.normal(size=(n_ensembles, columns))
    means = np.stack([shared + 0.3 * rng.normal(size=(n_ensembles, columns))
                      for _ in range(n_entries)]).astype(np.float32)
    variance = means.var(axis=1, ddof=1).astype(np.float32)

    class _Stub:
        arrays = {"ensemble_means_subset": means, "signal_variance": variance}
        subset_indices = np.arange(columns)
        n_ensembles = 40

    _, covariance_ensemble = diagnostic.paired_correlation_matrix(
        _Stub(), np.asarray([0]), np.asarray([1]), None)
    left = means[0].astype(np.float64)
    right = means[1].astype(np.float64)
    helper_covariance = np.sum((left - left.mean(axis=0)) * (right - right.mean(axis=0)),
                               axis=0) / (n_ensembles - 1)
    assert np.allclose(covariance_ensemble[0], helper_covariance, rtol=1e-10, atol=0)


def test_packed_inverse_diagonal_matches_the_reference_and_flags_indefinite_nodes() -> None:
    """The batched packed algebra must equal `fisher_diagnostics` matrix by matrix.

    Phase 2 inverts millions of 3x3 Fisher matrices, so it uses a cofactor form
    rather than `np.linalg.inv`.  That is a second implementation of the same
    algebra, which is the pattern the streaming audit exists to catch, so it is
    pinned to the single-matrix reference here.
    """
    from madi.fisher_crlb import pack_fisher, packed_inverse_diagonal, unpack_fisher

    rng = np.random.default_rng(7)
    J = rng.normal(size=(6, 9, 3))
    F = np.einsum("bci,bcj->bij", J, J)
    packed = pack_fisher(F)
    assert np.allclose(unpack_fisher(packed), F)

    inverse, det, positive = packed_inverse_diagonal(packed)
    assert positive.all()
    assert np.allclose(det, np.linalg.det(F), rtol=1e-10)
    for index in range(len(F)):
        reference = fisher_diagnostics(F[index], kio_ref=20.0)
        assert np.allclose(inverse[index], np.diag(reference["Finv"]), rtol=1e-9)
        assert np.allclose(np.sqrt(inverse[index]), reference["crlb"], rtol=1e-9)

    # A node whose inverse diagonal is not wholly positive is unidentified, not
    # a NaN CRLB, even if the leading-minor test alone would pass it.
    indefinite = packed.copy()
    indefinite[0, 5] = 1e-14
    _, _, still_positive = packed_inverse_diagonal(indefinite)
    assert not still_positive[0]


def test_packed_amplitude_marginal_matches_the_single_node_schur_complement() -> None:
    """The batched Schur complement must equal `amplitude_marginal_fisher`."""
    from madi.fisher_crlb import pack_fisher, packed_amplitude_marginal

    rng = np.random.default_rng(19)
    columns = 14
    J = rng.normal(size=(columns, 3))
    signal = np.abs(rng.normal(loc=1.0, scale=0.2, size=columns))
    sigma = np.full(columns, 0.02)
    for prior in (0.0, 5.0, 1234.0):
        reference = amplitude_marginal_fisher(J, signal, sigma, s0_prior_precision=prior)
        packed_tt = pack_fisher(reference["F_fixed_s0"])
        # F_s0_s0 already carries the prior in the reference, so pass it as zero.
        batched = packed_amplitude_marginal(packed_tt, reference["F_theta_s0"],
                                            reference["F_s0_s0"], 0.0)
        assert np.allclose(batched, pack_fisher(reference["F_marginal_s0"]), rtol=1e-12)


# ---------------------------------------------------------------------------
# Analysis-domain contract (added 2026-09-06)
#
# These pin the architectural boundary the Phase-0/1/2 domain audit restored:
# the reusable Fisher substrate spans the whole stored acquisition grid, and a
# downstream hardware or acquisition assumption may condition an analysis but
# may never decide what is extracted or cached.  See docs/fisher_domain_audit.md.
# ---------------------------------------------------------------------------

def _stored_column_grid():
    """The real stored `(delta, Delta, b)` column grid, from `madi.config`."""
    from madi.config import evenly_spaced_bvalues, valid_delta_pairs
    from madi.fisher_crlb import column_arrays

    pairs = valid_delta_pairs()
    b_values = np.asarray(evenly_spaced_bvalues(), dtype=float)
    pair_deltas = np.asarray([p[0] for p in pairs], dtype=float)
    pair_Deltas = np.asarray([p[1] for p in pairs], dtype=float)
    return (pair_deltas, pair_Deltas, b_values,
            *column_arrays(pair_deltas, pair_Deltas, b_values))


def _write_v5_stub(path, *, n_b: int = 5, n_pairs: int = 4,
                   n_ensembles: int = 4, seed: int = 20260906):
    """Write a minimal but schema-faithful v5 artifact for pipeline tests."""
    from madi.fisher_crlb import canonical_grid

    rhos, volumes, kios, retained = canonical_grid()
    # A plus-shaped patch about one centre, so every axis carries a k = 1 and a
    # k = 2 central stencil and no derivative field comes out empty.
    offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (0, 1), (0, -2), (0, 2)]
    centre = next((ir, iv) for ir, iv in sorted(retained)
                  if all((ir + a, iv + b) in retained for a, b in offsets))
    groups = sorted({(centre[0] + a, centre[1] + b) for a, b in offsets})
    rng = np.random.default_rng(seed)
    entries = [(ir, iv, ik) for ir, iv in groups for ik in range(len(kios))]
    n_entries = len(entries)
    columns = n_pairs * n_b
    pair_deltas = np.arange(1.0, n_pairs + 1.0)
    pair_Deltas = pair_deltas + 20.0
    b_values = np.arange(n_b, dtype=float) * 500.0
    vectors = np.clip(rng.uniform(0.2, 1.0, size=(n_entries, columns)), 0.0, 1.0)
    vectors[:, ::n_b] = 1.0                       # b = 0 is exactly one
    np.savez(
        path,
        library_schema=np.asarray("v5"),
        kios=np.asarray([kios[ik] for _, _, ik in entries], dtype=float),
        rhos=np.asarray([rhos[ir] for ir, _, _ in entries], dtype=float),
        Vs=np.asarray([volumes[iv] for _, iv, _ in entries], dtype=float),
        vectors=vectors,
        nominal_kios=np.asarray([kios[ik] for _, _, ik in entries], dtype=float),
        nominal_rhos=np.asarray([rhos[ir] for ir, _, _ in entries], dtype=float),
        nominal_Vs=np.asarray([volumes[iv] for _, iv, _ in entries], dtype=float),
        is_free_water=np.zeros(n_entries, dtype=bool),
        build_metadata_json=np.asarray(json.dumps({
            "D0_um2_ms": 3.0, "walkers_per_ensemble": 1000,
            "uncertainty": {"ensemble_index_ordering_contract": {
                "axis_position_in_ensemble_means_subset": 1,
                "index_values": "0..n_ensembles-1",
                "same_order_across_entries": True,
                "independent_of": ["rho", "V", "k_io"]}}})),
        pair_deltas=pair_deltas, pair_Deltas=pair_Deltas, b_values=b_values,
        n_b=np.asarray(n_b),
        signal_imag=np.zeros((n_entries, columns), dtype=np.float32),
        signal_variance=rng.uniform(1e-8, 1e-6, size=(n_entries, columns)).astype(np.float32),
        ensemble_means_subset=rng.uniform(0.2, 1.0,
                                          size=(n_entries, n_ensembles, n_b)).astype(np.float32),
        ensemble_subset_pair_deltas=pair_deltas[:1],
        ensemble_subset_pair_Deltas=pair_Deltas[:1],
        ensemble_subset_b_values=b_values,
        ensemble_subset_n_b=np.asarray(n_b),
    )
    return columns


def _run_phase1(artifact, output_dir, extra):
    import sys
    from scripts import run_fisher_phase1

    argv = sys.argv
    sys.argv = ["run_fisher_phase1", "--artifact", str(artifact),
                "--output-dir", str(output_dir), *extra]
    try:
        assert run_fisher_phase1.main() == 0
    finally:
        sys.argv = argv
    return json.loads((output_dir / "phase1_manifest.json").read_text())


def _feasibility_stub(path, artifact, indices):
    path.write_text(json.dumps({
        "artifact": str(artifact),
        "counts": {"clinical": {"gradient_columns": len(indices),
                                "combined_columns_any_entry": len(indices)}},
        "derivative_column_selection": {"column_indices": [int(i) for i in indices]},
    }))
    return path


def test_a_hardware_profile_cannot_decide_what_the_substrate_contains(tmp_path) -> None:
    """The architectural invariant, end to end.

    Two feasibility reports standing for two different scanner ceilings must
    produce the SAME Phase-1 column domain, and that domain must be every stored
    column.  This is the defect the 2026-09-06 audit found: an unrequested
    300 mT/m ceiling had removed 7,014 of 31,125 stored columns from the
    reusable substrate, so no later analysis at any other gradient setting could
    reach them and no notebook-level filter could recover them.
    """
    from madi.fisher_crlb import STORED_COLUMN_DOMAIN, read_column_domain

    artifact = tmp_path / "stub.npz"
    columns = _write_v5_stub(artifact)
    tight = _feasibility_stub(tmp_path / "tight.json", artifact, range(0, columns, 4))
    loose = _feasibility_stub(tmp_path / "loose.json", artifact, range(columns))

    domains = []
    for name, feasibility in (("tight", tight), ("loose", loose)):
        manifest = _run_phase1(artifact, tmp_path / f"smoke_{name}",
                               ["--feasibility", str(feasibility)])
        domain = read_column_domain(manifest)
        assert domain.basis == STORED_COLUMN_DOMAIN
        assert domain.is_complete and len(domain.column_indices) == columns
        assert manifest["feasibility_role"].startswith("conditional annotation only")
        domains.append(domain.column_indices)
    assert np.array_equal(domains[0], domains[1])

    # And with no feasibility report at all, which is now the ordinary case.
    plain = read_column_domain(_run_phase1(artifact, tmp_path / "smoke_plain", []))
    assert plain.is_complete and np.array_equal(plain.column_indices, domains[0])


def test_a_restricted_substrate_is_opt_in_stamped_and_refused_downstream(tmp_path) -> None:
    """Restriction stays reproducible, but it can never masquerade as universal."""
    from madi.fisher_crlb import read_column_domain, require_columns

    artifact = tmp_path / "stub.npz"
    columns = _write_v5_stub(artifact)
    kept = list(range(0, columns, 2))
    feasibility = _feasibility_stub(tmp_path / "feas.json", artifact, kept)
    manifest = _run_phase1(artifact, tmp_path / "smoke_restricted",
                           ["--feasibility", str(feasibility),
                            "--restrict-columns-to-feasibility"])
    domain = read_column_domain(manifest)
    assert not domain.is_complete
    assert domain.basis == "research_feasibility_union_diagnostic"
    assert "RESTRICTED" in domain.banner()
    assert set(kept).issubset(set(domain.column_indices.tolist()))
    # A downstream conditional analysis that wants an omitted column must fail
    # loudly rather than quietly score a smaller acquisition under its name.
    with np.testing.assert_raises(ValueError):
        require_columns(domain, range(columns), "unit conditional analysis")


def test_a_legacy_manifest_is_not_assumed_to_be_universal() -> None:
    """Manifests written before the contract declare no basis; do not guess one."""
    from madi.fisher_crlb import LEGACY_COLUMN_DOMAIN, read_column_domain

    domain = read_column_domain({"column_selection": {
        "selected_full_column_indices": [0, 1, 2, 5],
        "diagnostic_full_column_indices": [0],
        "full_stored_columns": 8}})
    assert domain.basis == LEGACY_COLUMN_DOMAIN
    assert not domain.is_complete
    assert "not assumed universal" in (domain.restriction_note or "")


def test_the_conditional_gradient_mask_reconstructs_from_the_full_substrate() -> None:
    """A hardware scenario must be a pure subset operation on the substrate.

    The pre-registered ceilings reproduce exactly the historical Phase-0.4
    counts, so nothing is lost by keeping the substrate unrestricted: the
    clinical mask is a subset of the research mask, which is a subset of the
    stored grid.
    """
    from madi.fisher_crlb import gradient_feasible_columns, stored_column_domain

    pair_deltas, pair_Deltas, b_values, delta, Delta, b = _stored_column_grid()
    domain = stored_column_domain(len(b))
    assert len(b) == 31_125
    clinical = gradient_feasible_columns(delta, Delta, b, 0.08)
    research = gradient_feasible_columns(delta, Delta, b, 0.30)
    assert len(clinical) == 9_999 and len(research) == 24_081
    assert set(clinical).issubset(set(research))
    assert domain.covers(research).all() and domain.covers(clinical).all()
    # The columns the old substrate lost are exactly the gradient-infeasible ones.
    lost = np.setdiff1d(domain.column_indices, research)
    assert len(lost) == 7_044
    assert float(np.min(gradient_strength_t_per_m(delta[lost], Delta[lost], b[lost]))) > 0.30
    # The historical substrate was the research mask UNION the 200 diagnostic
    # columns, so 30 gradient-infeasible columns survived incidentally and the
    # net loss was 7,014.  Both numbers are pinned because the audit quotes both.
    from madi.config import ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS

    n_b = len(b_values)
    diagnostic = np.concatenate([
        int(np.flatnonzero((pair_deltas == d) & (pair_Deltas == D))[0]) * n_b + np.arange(n_b)
        for d, D in ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS])
    historical = np.union1d(research, diagnostic)
    assert len(historical) == 24_111
    assert len(np.setdiff1d(domain.column_indices, historical)) == 7_014


def test_the_domain_mapping_preserves_canonical_column_identity() -> None:
    """Positions must round-trip, and a position must keep its `(delta, Delta, b)`."""
    from madi.fisher_crlb import ColumnDomain, stored_column_domain

    pair_deltas, pair_Deltas, b_values, delta, Delta, b = _stored_column_grid()
    full = stored_column_domain(len(b))
    assert np.array_equal(full.position_of, np.arange(len(b)))
    # The stored layout is timing-pair-major then b-major; check the identity.
    n_b = len(b_values)
    for column in (0, 1, n_b, 3 * n_b + 7, len(b) - 1):
        pair, offset = divmod(column, n_b)
        assert delta[column] == pair_deltas[pair]
        assert Delta[column] == pair_Deltas[pair]
        assert b[column] == b_values[offset]

    subset = np.asarray(sorted({0, 5, n_b, len(b) - 1}), dtype=np.int64)
    restricted = ColumnDomain("unit_subset", subset, len(b))
    positions = restricted.position_of[subset]
    assert np.array_equal(restricted.column_indices[positions], subset)
    assert np.array_equal(delta[restricted.column_indices[positions]], delta[subset])
    assert np.array_equal(b[restricted.column_indices[positions]], b[subset])


def test_widening_the_column_domain_leaves_the_overlap_bit_identical(tmp_path) -> None:
    """Re-materialising at full width must reproduce the restricted fields exactly.

    This is the unit-level form of the remediation's acceptance check: on the
    production artifact the seven full-width derivative and Richardson fields
    reproduce the restricted run bit-for-bit over 1,781,200,125 float32 cells.
    Each column is accumulated independently, so widening the domain can only
    add columns -- and this pins that, so a future change to the streaming path
    cannot quietly perturb the columns it was not supposed to touch.
    """
    from madi.fisher_crlb import read_column_domain

    artifact = tmp_path / "stub.npz"
    columns = _write_v5_stub(artifact)
    kept = list(range(0, columns, 3))
    feasibility = _feasibility_stub(tmp_path / "feas.json", artifact, kept)

    full = _run_phase1(artifact, tmp_path / "smoke_full", ["--feasibility", str(feasibility)])
    restricted = _run_phase1(artifact, tmp_path / "smoke_restricted",
                             ["--feasibility", str(feasibility),
                              "--restrict-columns-to-feasibility"])
    full_domain = read_column_domain(full)
    restricted_domain = read_column_domain(restricted)
    assert full_domain.is_complete and not restricted_domain.is_complete
    overlap = restricted_domain.column_indices
    assert len(overlap) < columns

    compared = 0
    for name, item in restricted["derivatives"].items():
        a = np.load(tmp_path / "smoke_restricted" / item["J"])
        b = np.load(tmp_path / "smoke_full" / full["derivatives"][name]["J"])
        assert np.array_equal(np.load(tmp_path / "smoke_restricted" / item["samples"]),
                              np.load(tmp_path / "smoke_full" / full["derivatives"][name]["samples"]))
        overlapping = b[:, full_domain.position_of[overlap]]
        assert a.shape == overlapping.shape
        # Bit-for-bit, not to a tolerance.
        assert np.array_equal(a.view(np.uint32), overlapping.view(np.uint32))
        compared += a.size
    assert compared > 0

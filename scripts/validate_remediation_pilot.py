#!/usr/bin/env python3
"""Tier A validator for the restricted P0 remediation pilot.

This performs the pre-written pilot acceptance checklist without rebuilding
or fitting anything.  It is intended to run locally after the four Sol shard
files have been returned.  It validates the exact pilot geometry grid encoded
in ``scripts/build_lib.sbatch`` and writes a JSON report; its exit status is
the result.

Example
-------
PYTHONPATH=. python -m scripts.validate_remediation_pilot \
  --shards libraries/madi_v5_remediation_pilot.shard*.npz \
  --output logs/remediation_pilot_validation.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t

from madi.config import (
    ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2,
    ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS,
    valid_delta_pairs,
)
from madi.library import (
    LibraryEntry,
    MADI_LIBRARY_SCHEMA_V5,
    _entry_key,
    ensemble_mean_subset_column_indices,
    load_library,
    load_library_meta,
    make_remediation_log_grid,
)
from madi.signal import ColumnGrid


PILOT_N_SHARDS = 4
PILOT_KIOS = np.asarray([0.0, 20.0, 130.0], dtype=float)
PILOT_N_RHO = 8
PILOT_N_V = 12
PILOT_RHO_MIN = 1.0e4
PILOT_RHO_MAX = 1.0e7
PILOT_V_MIN = 0.01
PILOT_V_MAX = 200.0
PILOT_VI_MIN = 0.40
PILOT_VI_MAX = 0.99
PILOT_N_ENSEMBLES = 8
PILOT_SMALL_DELTAS = [5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
PILOT_BIG_DELTAS = [15.0, 25.0, 30.0, 36.0, 40.0, 50.0, 60.0, 80.0]
PILOT_PAIRS = valid_delta_pairs(PILOT_SMALL_DELTAS, PILOT_BIG_DELTAS)
PILOT_B_VALUES = list(np.asarray(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2, dtype=float))
PILOT_IMAGINARY_FAMILYWISE_ALPHA = 0.01


def _columns_from_meta(meta: dict) -> ColumnGrid:
    """Minimal ColumnGrid used only to map the declared subset to vectors."""
    empty_int = np.empty(0, dtype=np.int32)
    return ColumnGrid(
        delta_pairs=list(meta["delta_pairs"]),
        b_values=np.asarray(meta["b_values"], dtype=float),
        j_delta=empty_int,
        j_Delta=empty_int,
        j_sum=empty_int,
        phase_coef=np.empty(0, dtype=np.float64),
        n_pairs=len(meta["delta_pairs"]),
        n_b=int(meta["n_b"]),
    )


def _pilot_triplets_and_weights() -> tuple[list[tuple[float, float, float]], dict]:
    grid = make_remediation_log_grid(
        n_rho=PILOT_N_RHO,
        n_V=PILOT_N_V,
        kios=PILOT_KIOS,
        rho_min=PILOT_RHO_MIN,
        rho_max=PILOT_RHO_MAX,
        V_min=PILOT_V_MIN,
        V_max=PILOT_V_MAX,
        vi_min=PILOT_VI_MIN,
        vi_max=PILOT_VI_MAX,
    )
    return grid.triplets_and_weights()


def _expected_shard_keys(shard_id: int) -> set[tuple]:
    triplets, _ = _pilot_triplets_and_weights()
    # Match scripts.fit_data exactly: groups are round-robined after sorting
    # by the same rho*V cost proxy used by the builder.  Lexicographic sorting
    # would validate the right 25 coordinates against the wrong shard IDs.
    pairs = sorted(
        {(rho, volume) for _, rho, volume in triplets if rho > 0.0},
        key=lambda pair: pair[0] * pair[1],
    )
    selected: set[tuple] = set()
    for kio, rho, volume in triplets:
        if rho == 0.0 and volume == 0.0:
            if shard_id == 0:
                selected.add(_entry_key(kio, rho, volume))
        elif pairs.index((rho, volume)) % PILOT_N_SHARDS == shard_id:
            selected.add(_entry_key(kio, rho, volume))
    return selected


def _entry_nominal_key(entry: LibraryEntry) -> tuple:
    return _entry_key(
        entry.kio_nominal if entry.kio_nominal is not None else entry.kio,
        entry.rho_nominal if entry.rho_nominal is not None else entry.rho,
        entry.V_nominal if entry.V_nominal is not None else entry.V,
    )


def _expand_paths(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        expanded = sorted(glob.glob(value))
        paths.extend(Path(item) for item in (expanded or [value]))
    return paths


def _metadata_errors(meta: dict, path: Path) -> list[str]:
    errors: list[str] = []
    build = meta.get("build_metadata") or {}
    if meta.get("format") != "v2":
        errors.append(f"{path}: expected v2 library format, got {meta.get('format')!r}")
    if meta.get("library_schema") != MADI_LIBRARY_SCHEMA_V5:
        errors.append(
            f"{path}: expected {MADI_LIBRARY_SCHEMA_V5!r}, got "
            f"{meta.get('library_schema')!r}"
        )
    if build.get("schema") != MADI_LIBRARY_SCHEMA_V5:
        errors.append(f"{path}: build metadata does not record the v5 schema")
    if meta.get("has_weights") is not True:
        errors.append(f"{path}: pilot must declare complete per-entry quadrature weights")
    if meta.get("delta_pairs") != PILOT_PAIRS:
        errors.append(f"{path}: unexpected (delta,Delta) axes {meta.get('delta_pairs')!r}")
    if meta.get("b_values") != PILOT_B_VALUES or meta.get("n_b") != len(PILOT_B_VALUES):
        errors.append(f"{path}: unexpected b axis or n_b")
    if build.get("phase_model") != "finite_lobe":
        errors.append(f"{path}: pilot must use finite_lobe phase model")
    if not np.isclose(float(build.get("kappa", np.nan)), 0.90, rtol=0.0, atol=1e-12):
        errors.append(f"{path}: pilot must use kappa=0.90")
    if build.get("boundary_mode") != "si_fatal_escape":
        errors.append(f"{path}: pilot must use SI fatal-escape boundary mode")
    if float(build.get("T_max_ms", np.nan)) != 128.0:
        errors.append(f"{path}: pilot must use T_max_ms=128")
    if (int(build.get("walkers_per_ensemble", -1)) != 512
            or int(build.get("ensembles_per_entry", -1)) != PILOT_N_ENSEMBLES):
        errors.append(
            f"{path}: pilot must use 512 walkers and {PILOT_N_ENSEMBLES} ensembles"
        )
    if "full-facet" not in str(build.get("classifier", "")) or "Eq. S2" not in str(build.get("classifier", "")):
        errors.append(f"{path}: missing full-facet SI Eq. S2 classifier provenance")
    reference = build.get("geometry_reference") or {}
    if reference.get("allow_uncertified") is not False:
        errors.append(f"{path}: uncertified geometry reference was allowed")
    if reference.get("mean_estimator") != "untrimmed_arithmetic_mean_A_over_V":
        errors.append(f"{path}: wrong Eq. 5 A/V estimator provenance")
    grid = build.get("grid") or {}
    transform = grid.get("coordinate_transform") or {}
    if transform != {"rho": "uniform_log", "V": "uniform_log", "kio": "piecewise_uniform_linear"}:
        errors.append(f"{path}: missing or incompatible weighted log-grid provenance")
    uncertainty = build.get("uncertainty") or {}
    variance = uncertainty.get("between_ensemble_variance") or {}
    subset = uncertainty.get("ensemble_means_subset") or {}
    contract = uncertainty.get("ensemble_index_ordering_contract") or {}
    if (variance.get("array") != "signal_variance"
            or variance.get("definition", "").find("sample variance") < 0
            or int(variance.get("n_ensembles", -1)) != PILOT_N_ENSEMBLES):
        errors.append(f"{path}: missing v5 between-ensemble-variance contract")
    declared_pairs = [list(pair) for pair in ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS]
    declared_b = list(np.asarray(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2, dtype=float))
    if (subset.get("array") != "ensemble_means_subset"
            or subset.get("declared_delta_Delta_pairs_ms") != declared_pairs
            or subset.get("declared_b_values_s_mm2") != declared_b
            or subset.get("column_order") != "pair-major, then b-major within each pair"):
        errors.append(f"{path}: missing or incompatible declared ensemble-mean subset metadata")
    if (contract.get("axis_position_in_ensemble_means_subset") != 1
            or contract.get("index_values") != "0..n_ensembles-1"
            or contract.get("same_order_across_entries") is not True
            or contract.get("independent_of") != ["rho", "V", "k_io"]
            or contract.get("walk_seed_formula")
            != "(build_seed + 104729 * ensemble_index) mod 2^31"):
        errors.append(f"{path}: missing v5 ensemble-index CRN ordering contract")
    return errors


def _v5_diagnostic_errors(path: Path, columns, n_entries: int) -> tuple[list[str], dict]:
    """Validate the new column-level v5 payload directly from the NPZ file."""
    errors: list[str] = []
    n_columns = int(columns.n_pairs * columns.n_b)
    subset_columns = ensemble_mean_subset_column_indices(columns)
    expected_subset_shape = (
        n_entries,
        PILOT_N_ENSEMBLES,
        len(ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS)
        * len(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2),
    )
    diagnostics = {
        "expected_signal_shape": [n_entries, n_columns],
        "expected_subset_shape": list(expected_subset_shape),
        "max_subset_mean_abs_error": None,
    }
    required = {
        "signal_imag",
        "signal_variance",
        "ensemble_means_subset",
        "ensemble_subset_pair_deltas",
        "ensemble_subset_pair_Deltas",
        "ensemble_subset_b_values",
        "ensemble_subset_n_b",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            return [f"{path}: missing v5 arrays {missing}"], diagnostics

        signal_imag = data["signal_imag"]
        signal_variance = data["signal_variance"]
        ensemble_subset = data["ensemble_means_subset"]
        vectors = data["vectors"]
        for name, value, expected_shape in (
            ("signal_imag", signal_imag, (n_entries, n_columns)),
            ("signal_variance", signal_variance, (n_entries, n_columns)),
            ("ensemble_means_subset", ensemble_subset, expected_subset_shape),
        ):
            if value.dtype != np.dtype(np.float32):
                errors.append(f"{path}: {name} dtype {value.dtype} is not float32")
            if value.shape != expected_shape:
                errors.append(
                    f"{path}: {name} shape {value.shape} != expected {expected_shape}"
                )
            if not np.all(np.isfinite(value)):
                errors.append(f"{path}: {name} contains non-finite values")
        if np.any(signal_variance < 0.0):
            errors.append(f"{path}: signal_variance contains negative values")

        actual_pairs = list(zip(
            np.asarray(data["ensemble_subset_pair_deltas"], dtype=float).tolist(),
            np.asarray(data["ensemble_subset_pair_Deltas"], dtype=float).tolist(),
        ))
        actual_b = list(np.asarray(data["ensemble_subset_b_values"], dtype=float))
        actual_n_b = int(data["ensemble_subset_n_b"])
        if actual_pairs != list(ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS):
            errors.append(f"{path}: ensemble subset timing pairs differ from the declared eight pairs")
        if (actual_b != list(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2)
                or actual_n_b != len(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2)):
            errors.append(f"{path}: ensemble subset b-values differ from the declared 25 values")

        if (vectors.shape != (n_entries, n_columns)
                or ensemble_subset.shape != expected_subset_shape):
            errors.append(f"{path}: cannot cross-check subset means because signal shapes are invalid")
        else:
            subset_mean = np.mean(ensemble_subset, axis=1, dtype=np.float64)
            main_subset = np.asarray(vectors[:, subset_columns], dtype=np.float64)
            difference = np.abs(subset_mean - main_subset)
            max_error = float(np.max(difference)) if difference.size else 0.0
            diagnostics["max_subset_mean_abs_error"] = max_error
            if not np.allclose(subset_mean, main_subset, rtol=2.0e-6, atol=2.0e-6):
                errors.append(
                    f"{path}: means over ensemble_means_subset do not reproduce "
                    "the main stored signal within float32 tolerance"
                )
    return errors, diagnostics


def validate(paths: list[Path]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    expected_ids = set(range(PILOT_N_SHARDS))
    shard_ids: dict[int, Path] = {}
    all_entries: list[LibraryEntry] = []
    expected_counts: dict[int, int] = {}
    per_shard: list[dict] = []
    imaginary_standardized_deviations: list[float] = []

    for path in paths:
        if not path.is_file():
            errors.append(f"missing shard file {path}")
            continue
        match = re.search(r"\.shard(\d+)\.npz$", path.name)
        if match is None:
            errors.append(f"cannot infer shard id from {path}")
            continue
        shard_id = int(match.group(1))
        if shard_id in shard_ids:
            errors.append(f"duplicate shard id {shard_id}: {path} and {shard_ids[shard_id]}")
            continue
        shard_ids[shard_id] = path

    if set(shard_ids) != expected_ids:
        errors.append(f"shard coverage {sorted(shard_ids)} != expected {sorted(expected_ids)}")

    for shard_id, path in sorted(shard_ids.items()):
        meta = load_library_meta(str(path))
        shard_errors = _metadata_errors(meta, path)
        errors.extend(shard_errors)
        entries = load_library(str(path))
        try:
            diagnostic_errors, diagnostic_report = _v5_diagnostic_errors(
                path, _columns_from_meta(meta), len(entries),
            )
        except (KeyError, TypeError, ValueError) as exc:
            diagnostic_errors = [f"{path}: cannot validate v5 diagnostics: {exc}"]
            diagnostic_report = {"max_subset_mean_abs_error": None}
        shard_errors.extend(diagnostic_errors)
        errors.extend(diagnostic_errors)
        geometry_tolerance = float(
            (meta.get("build_metadata") or {}).get("geometry_vi_tolerance", np.nan)
        )
        if not np.isfinite(geometry_tolerance) or geometry_tolerance <= 0.0:
            message = f"{path}: missing positive realised-v_i tolerance in build metadata"
            shard_errors.append(message)
            errors.append(message)
        else:
            for entry in entries:
                if entry.is_free_water:
                    continue
                target_vi = float(entry.rho_nominal * entry.V_nominal * 1e-6)
                if abs(float(entry.vi) - target_vi) > geometry_tolerance:
                    message = (
                        f"{path}: realised v_i={entry.vi:.8g} is outside "
                        f"{geometry_tolerance:.8g} of target {target_vi:.8g} "
                        f"for {_entry_nominal_key(entry)}"
                    )
                    shard_errors.append(message)
                    errors.append(message)
                realised = (entry.metadata.get("realised_geometry") or {}).get("vi")
                if realised is None or not np.isclose(
                    float(realised), float(entry.vi), rtol=0.0, atol=1e-12,
                ):
                    message = f"{path}: entry label and realised-geometry v_i disagree"
                    shard_errors.append(message)
                    errors.append(message)
        actual_keys = {_entry_nominal_key(entry) for entry in entries}
        expected_keys = _expected_shard_keys(shard_id)
        expected_counts[shard_id] = len(expected_keys)
        if len(actual_keys) != len(entries):
            errors.append(f"{path}: duplicate nominal entry keys inside shard")
        if actual_keys != expected_keys:
            errors.append(f"{path}: nominal coordinates do not match the pilot shard assignment")
        all_entries.extend(entries)
        per_shard.append({
            "path": str(path), "shard_id": shard_id, "entries": len(entries),
            "expected_entries": len(expected_keys), "errors": shard_errors,
            "v5_diagnostics": diagnostic_report,
        })

    all_keys = [_entry_nominal_key(entry) for entry in all_entries]
    if len(set(all_keys)) != len(all_keys):
        errors.append("duplicate nominal entries across pilot shards")
    expected_all = set().union(*(_expected_shard_keys(i) for i in range(PILOT_N_SHARDS)))
    if set(all_keys) != expected_all:
        errors.append("combined nominal pilot coverage does not equal the 25-entry specification")

    free = [entry for entry in all_entries if entry.is_free_water]
    cellular = [entry for entry in all_entries if not entry.is_free_water]
    if len(all_entries) != 25:
        errors.append(f"expected 25 total pilot entries, found {len(all_entries)}")
    if len(free) != 1:
        errors.append(f"expected one free-water atom, found {len(free)}")
    if sum(entry.kio_nominal is not None and np.isclose(entry.kio_nominal, 0.0) for entry in cellular) != 8:
        errors.append("expected one cellular k_io=0 entry for each retained (rho,V) pair")

    negative_signal_count = 0
    nonfinite_signal_count = 0
    min_signal = float("inf")
    for entry in all_entries:
        vector = np.asarray(entry.vector, dtype=np.float64)
        if vector.size != len(PILOT_PAIRS) * len(PILOT_B_VALUES):
            errors.append(
                f"entry {_entry_nominal_key(entry)} has vector length {vector.size}, "
                f"expected {len(PILOT_PAIRS) * len(PILOT_B_VALUES)}"
            )
        elif not np.array_equal(vector.reshape(len(PILOT_PAIRS), len(PILOT_B_VALUES))[:, 0], np.ones(len(PILOT_PAIRS))):
            errors.append(f"entry {_entry_nominal_key(entry)} does not have exact S(b=0)=1")
        nonfinite_signal_count += int(np.count_nonzero(~np.isfinite(vector)))
        negative_signal_count += int(np.count_nonzero(vector < 0.0))
        if vector.size:
            min_signal = min(min_signal, float(np.min(vector)))
        if entry.weight is None or not np.isfinite(entry.weight) or entry.weight <= 0.0:
            errors.append(f"entry {_entry_nominal_key(entry)} has no positive finite quadrature weight")
        imaginary_check = entry.metadata.get("imaginary_signal_check") or {}
        standardized = imaginary_check.get("max_abs_standardized_deviation")
        max_column = imaginary_check.get("max_column") or {}
        imag_mean = imaginary_check.get("mean_imaginary_signal")
        imag_se = imaginary_check.get("standard_error")
        try:
            pair_index = list(PILOT_PAIRS).index((
                float(max_column["delta_ms"]), float(max_column["Delta_ms"]),
            ))
            b_index = PILOT_B_VALUES.index(float(max_column["b_s_mm2"]))
            stored_imaginary = np.asarray(entry.signal_imag, dtype=np.float64).reshape(
                len(PILOT_PAIRS), len(PILOT_B_VALUES),
            )[pair_index, b_index]
        except (KeyError, TypeError, ValueError, AttributeError):
            stored_imaginary = np.nan
        if (imaginary_check.get("n_ensembles") != PILOT_N_ENSEMBLES
                or imaginary_check.get("degrees_of_freedom") != PILOT_N_ENSEMBLES - 1
                or imaginary_check.get("zero_standard_error_nonzero_count") != 0
                or standardized is None or not np.isfinite(float(standardized))
                or imag_mean is None or imag_se is None
                or not (np.isfinite(float(imag_mean)) and np.isfinite(float(imag_se)))
                or not np.isfinite(stored_imaginary)):
            errors.append(
                f"entry {_entry_nominal_key(entry)} lacks a finite v5 imaginary-signal "
                "standardization record"
            )
        else:
            if float(imag_se) == 0.0:
                expected_t = 0.0 if float(imag_mean) == 0.0 else float("inf")
            else:
                expected_t = abs(float(imag_mean) / float(imag_se))
            if not np.isclose(float(standardized), expected_t, rtol=2.0e-6, atol=2.0e-6):
                errors.append(
                    f"entry {_entry_nominal_key(entry)} has an inconsistent imaginary "
                    "standardization record"
                )
            if not np.isclose(stored_imaginary, float(imag_mean), rtol=2.0e-6, atol=2.0e-6):
                errors.append(
                    f"entry {_entry_nominal_key(entry)} has an imaginary standardization "
                    "record that does not match signal_imag"
                )
            imaginary_standardized_deviations.append(float(standardized))
        if entry.is_free_water:
            if not (np.isnan(entry.kio) and entry.rho == 0.0 and entry.V == 0.0):
                errors.append("free-water atom has invalid physical labels")
            continue
        vi = entry.realised_vi
        if not (0.0 < vi < 1.0):
            errors.append(f"entry {_entry_nominal_key(entry)} has invalid realised v_i={vi}")
        if not np.isclose(entry.rho * entry.V * 1e-6, vi, rtol=0.0, atol=1e-12):
            errors.append(f"entry {_entry_nominal_key(entry)} violates rho*V*1e-6=v_i")
        if entry.pp is None or not (0.0 <= entry.pp <= 1.0):
            errors.append(f"entry {_entry_nominal_key(entry)} has invalid p_p={entry.pp}")
        if (entry.kio_nominal is None or entry.kio_analytic_eq5 is None
                or not np.isclose(entry.kio_analytic_eq5, entry.kio_nominal, rtol=0.0, atol=1e-9)):
            errors.append(f"entry {_entry_nominal_key(entry)} does not retain its Eq. 5 k_io label")
        boundary = entry.metadata.get("boundary") or {}
        if int(boundary.get("n_escaped", -1)) != 0:
            errors.append(f"entry {_entry_nominal_key(entry)} recorded an escape")
        geometry = entry.metadata.get("realised_geometry") or {}
        if not geometry:
            errors.append(f"entry {_entry_nominal_key(entry)} lacks realised-geometry metadata")
        per_ensemble = entry.metadata.get("per_ensemble_geometry") or []
        reference_metadata = (per_ensemble[0].get("reference_metadata") if per_ensemble else None)
        if not isinstance(reference_metadata, dict):
            errors.append(f"entry {_entry_nominal_key(entry)} lacks per-ensemble reference metadata")
        elif (int(reference_metadata.get("n_single_cells", 0)) < 5_000_000
              or reference_metadata.get("contraction_rule") != "all_shifted_voronoi_facets_S2"
              or not np.isclose(float(reference_metadata.get("kappa", np.nan)), 0.90, rtol=0.0, atol=1e-12)):
            errors.append(f"entry {_entry_nominal_key(entry)} has uncertified or non-full-facet A/V provenance")

    if nonfinite_signal_count:
        errors.append(f"pilot contains {nonfinite_signal_count} non-finite signal values")
    if negative_signal_count:
        warnings.append(
            f"pilot contains {negative_signal_count} negative low-signal samples (minimum {min_signal:g}); "
            "report them to the P2 trust-floor check rather than treating a low-MC pilot as production evidence"
        )

    imaginary_max_abs_standardized_deviation = None
    imaginary_threshold = None
    if len(imaginary_standardized_deviations) != len(all_entries):
        errors.append("not every pilot entry has a usable imaginary-signal standardization")
    elif all_entries:
        imaginary_max_abs_standardized_deviation = float(max(imaginary_standardized_deviations))
        n_imaginary_tests = len(all_entries) * len(PILOT_PAIRS) * len(PILOT_B_VALUES)
        # Per-column statistics are Student-t values based on E independent
        # ensemble means.  Bonferroni is deliberately conservative here: the
        # pilot asks whether a gross symmetry violation is visible, not for a
        # finely powered discovery claim across highly correlated columns.
        imaginary_threshold = float(student_t.ppf(
            1.0 - PILOT_IMAGINARY_FAMILYWISE_ALPHA / (2.0 * n_imaginary_tests),
            df=PILOT_N_ENSEMBLES - 1,
        ))
        if imaginary_max_abs_standardized_deviation > imaginary_threshold:
            errors.append(
                "stored imaginary signal is inconsistent with zero at the "
                "pilot ensemble sample size after the declared family-wise check: "
                f"max |t|={imaginary_max_abs_standardized_deviation:.6g} > "
                f"{imaginary_threshold:.6g}"
            )

    min_pairwise_l2 = float("inf")
    exact_duplicate_vectors = 0
    if len(all_entries) > 1:
        vectors = np.stack([np.asarray(entry.vector, dtype=np.float64) for entry in all_entries])
        distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        min_pairwise_l2 = float(np.min(distances))
        exact_duplicate_vectors = int(np.count_nonzero(np.triu(distances == 0.0, k=1)))
        if exact_duplicate_vectors:
            errors.append(f"found {exact_duplicate_vectors} exact vector ties across distinct pilot labels")

    return {
        "schema": "madi-remediation-pilot-validation-v2",
        "tier": "A",
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "expected_shard_counts": {str(k): v for k, v in expected_counts.items()},
        "per_shard": per_shard,
        "total_entries": len(all_entries),
        "free_water_entries": len(free),
        "cellular_entries": len(cellular),
        "negative_signal_count": negative_signal_count,
        "nonfinite_signal_count": nonfinite_signal_count,
        "minimum_signal": None if not np.isfinite(min_signal) else min_signal,
        "minimum_pairwise_l2": None if not np.isfinite(min_pairwise_l2) else min_pairwise_l2,
        "exact_duplicate_vectors": exact_duplicate_vectors,
        "imaginary_max_abs_standardized_deviation": imaginary_max_abs_standardized_deviation,
        "imaginary_familywise_threshold": imaginary_threshold,
        "imaginary_familywise_alpha": PILOT_IMAGINARY_FAMILYWISE_ALPHA,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True, help="four pilot shard paths or glob patterns")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(_expand_paths(args.shards))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

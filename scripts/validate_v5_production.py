#!/usr/bin/env python3
"""Read-only acceptance validator for the merged v5 production library.

This is deliberately an artifact validator, not a simulation or fitter.  It
checks the production grid coverage, v5 schema/diagnostic payload, imaginary
symmetry metadata, and production-quality signal summaries from the final NPZ.

Example
-------
python -m scripts.validate_v5_production \
  --artifact libraries/madi_dense_universal_remediated.npz \
  --output logs/v5_production_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from madi.config import (
    ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2,
    ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS,
    PRODUCTION_AXIS_WALKS_PER_ENTRY,
    PRODUCTION_CLASSIFIER_CACHE_CANDIDATE_CAPACITY,
    PRODUCTION_CLASSIFIER_CACHE_DELTA_MAX_UM,
    PRODUCTION_CLASSIFIER_CACHE_MIN_SAFE_RADIUS_UM,
    PRODUCTION_CLASSIFIER_MODE,
    PRODUCTION_ENSEMBLES_PER_ENTRY,
    PRODUCTION_WALKERS_PER_ENSEMBLE,
    evenly_spaced_bvalues,
    valid_delta_pairs,
)
from madi.library import (
    MADI_LIBRARY_SCHEMA_V5,
    _entry_key,
    ensemble_mean_subset_column_indices,
    make_remediation_log_grid,
)
from madi.signal import ColumnGrid


EXPECTED_SHARDS = 369
EXPECTED_ENTRIES = 18_820
EXPECTED_CELLULAR = 18_819
IMAGINARY_FAMILYWISE_ALPHA = 0.01


def _columns_from_arrays(data: np.lib.npyio.NpzFile) -> ColumnGrid:
    pairs = list(zip(
        np.asarray(data["pair_deltas"], dtype=float).tolist(),
        np.asarray(data["pair_Deltas"], dtype=float).tolist(),
    ))
    b_values = np.asarray(data["b_values"], dtype=float)
    empty_i = np.empty(0, dtype=np.int32)
    return ColumnGrid(
        delta_pairs=pairs,
        b_values=b_values,
        j_delta=empty_i,
        j_Delta=empty_i,
        j_sum=empty_i,
        phase_coef=np.empty(0, dtype=np.float64),
        n_pairs=len(pairs),
        n_b=int(data["n_b"]),
    )


def _expected_keys() -> set[tuple]:
    triplets, _ = make_remediation_log_grid().triplets_and_weights()
    return {_entry_key(kio, rho, volume) for kio, rho, volume in triplets}


def _reported_key(kio: float, rho: float, volume: float, free: bool) -> tuple:
    if free:
        return _entry_key(0.0, 0.0, 0.0)
    return _entry_key(kio, rho, volume)


def _required_array_errors(data: np.lib.npyio.NpzFile) -> list[str]:
    required = {
        "library_schema", "kios", "rhos", "Vs", "vis", "vectors",
        "nominal_kios", "nominal_rhos", "nominal_Vs", "weights", "pps",
        "kio_analytic_eq5", "is_free_water", "entry_metadata_json",
        "build_metadata_json", "pair_deltas", "pair_Deltas", "b_values", "n_b",
        "signal_imag", "signal_variance", "ensemble_means_subset",
        "ensemble_subset_pair_deltas", "ensemble_subset_pair_Deltas",
        "ensemble_subset_b_values", "ensemble_subset_n_b",
    }
    missing = sorted(required.difference(data.files))
    return [f"missing required arrays: {missing}"] if missing else []


def validate(artifact: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if not artifact.is_file():
        return {"pass": False, "errors": [f"artifact does not exist: {artifact}"], "warnings": []}

    with np.load(artifact, allow_pickle=False) as data:
        errors.extend(_required_array_errors(data))
        if errors:
            return {"pass": False, "errors": errors, "warnings": warnings}

        if str(data["library_schema"]) != MADI_LIBRARY_SCHEMA_V5:
            errors.append(f"library_schema is {str(data['library_schema'])!r}, not {MADI_LIBRARY_SCHEMA_V5!r}")
        try:
            build = json.loads(str(data["build_metadata_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            build = {}
            errors.append(f"cannot parse build_metadata_json: {exc}")

        columns = _columns_from_arrays(data)
        expected_pairs = valid_delta_pairs()
        expected_b = np.asarray(evenly_spaced_bvalues(), dtype=float)
        n_entries = int(np.asarray(data["kios"]).shape[0])
        n_columns = int(columns.n_pairs * columns.n_b)
        if columns.delta_pairs != expected_pairs:
            errors.append("stored (delta, Delta) pairs differ from the declared production grid")
        if not np.array_equal(columns.b_values, expected_b) or columns.n_b != len(expected_b):
            errors.append("stored b grid differs from the declared production grid")
        if n_entries != EXPECTED_ENTRIES:
            errors.append(f"entry count is {n_entries}, expected {EXPECTED_ENTRIES}")
        if n_columns != 31_125:
            errors.append(f"column count is {n_columns}, expected 31125")

        metadata_checks = {
            "schema": MADI_LIBRARY_SCHEMA_V5,
            "walkers_per_ensemble": PRODUCTION_WALKERS_PER_ENSEMBLE,
            "ensembles_per_entry": PRODUCTION_ENSEMBLES_PER_ENTRY,
            "axis_walks_per_entry": PRODUCTION_AXIS_WALKS_PER_ENTRY,
            "phase_model": "finite_lobe",
            "boundary_mode": "si_fatal_escape",
        }
        for key, expected in metadata_checks.items():
            if build.get(key) != expected:
                errors.append(f"build metadata {key}={build.get(key)!r}, expected {expected!r}")
        if not np.isclose(float(build.get("kappa", np.nan)), 0.90, rtol=0.0, atol=1e-12):
            errors.append("build metadata kappa is not 0.90")
        cache = build.get("classifier_cache") or {}
        if (cache.get("mode") != PRODUCTION_CLASSIFIER_MODE
                or cache.get("delta_max_um") != PRODUCTION_CLASSIFIER_CACHE_DELTA_MAX_UM
                or cache.get("min_safe_radius_um") != PRODUCTION_CLASSIFIER_CACHE_MIN_SAFE_RADIUS_UM
                or cache.get("candidate_capacity") != PRODUCTION_CLASSIFIER_CACHE_CANDIDATE_CAPACITY):
            errors.append("build metadata does not record the validated exact-cache configuration")
        reference = build.get("geometry_reference") or {}
        if (reference.get("allow_uncertified") is not False
                or int(reference.get("required_single_cells", 0)) < 5_000_000
                or reference.get("mean_estimator") != "untrimmed_arithmetic_mean_A_over_V"):
            errors.append("build metadata does not record the certified SI geometry reference contract")
        uncertainty = build.get("uncertainty") or {}
        contract = uncertainty.get("ensemble_index_ordering_contract") or {}
        if (contract.get("axis_position_in_ensemble_means_subset") != 1
                or contract.get("index_values") != "0..n_ensembles-1"
                or contract.get("same_order_across_entries") is not True
                or contract.get("independent_of") != ["rho", "V", "k_io"]
                or contract.get("walk_seed_formula") != "(build_seed + 104729 * ensemble_index) mod 2^31"):
            errors.append("build metadata lacks the v5 ensemble-index CRN contract")

        vectors = data["vectors"]
        imag = data["signal_imag"]
        variance = data["signal_variance"]
        subset = data["ensemble_means_subset"]
        expected_subset_shape = (
            n_entries, PRODUCTION_ENSEMBLES_PER_ENTRY,
            len(ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS) * len(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2),
        )
        for name, value, shape, dtype in (
            ("vectors", vectors, (n_entries, n_columns), np.float64),
            ("signal_imag", imag, (n_entries, n_columns), np.float32),
            ("signal_variance", variance, (n_entries, n_columns), np.float32),
            ("ensemble_means_subset", subset, expected_subset_shape, np.float32),
        ):
            if value.shape != shape:
                errors.append(f"{name} shape {value.shape} != {shape}")
            if value.dtype != np.dtype(dtype):
                errors.append(f"{name} dtype {value.dtype} != {np.dtype(dtype)}")
            if not np.all(np.isfinite(value)):
                errors.append(f"{name} contains non-finite values")
        if np.any(variance < 0.0):
            errors.append("signal_variance contains negative values")

        declared_pairs = list(ENSEMBLE_MEAN_SUBSET_DELTA_PAIRS_MS)
        actual_pairs = list(zip(
            np.asarray(data["ensemble_subset_pair_deltas"], dtype=float).tolist(),
            np.asarray(data["ensemble_subset_pair_Deltas"], dtype=float).tolist(),
        ))
        if actual_pairs != declared_pairs:
            errors.append("declared ensemble subset timing pairs changed")
        if (not np.array_equal(np.asarray(data["ensemble_subset_b_values"], dtype=float),
                               ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2)
                or int(data["ensemble_subset_n_b"]) != len(ENSEMBLE_MEAN_SUBSET_B_VALUES_S_MM2)):
            errors.append("declared ensemble subset b values changed")

        subset_columns = ensemble_mean_subset_column_indices(columns)
        subset_mean = np.mean(subset, axis=1, dtype=np.float64)
        subset_mean_error = float(np.max(np.abs(subset_mean - vectors[:, subset_columns])))
        if not np.allclose(subset_mean, vectors[:, subset_columns], rtol=2e-6, atol=2e-6):
            errors.append("ensemble subset means do not reproduce the main signal")
        subset_variance = np.var(subset, axis=1, ddof=1, dtype=np.float64)
        subset_variance_error = float(np.max(np.abs(subset_variance - variance[:, subset_columns])))
        if not np.allclose(subset_variance, variance[:, subset_columns], rtol=2e-6, atol=2e-6):
            errors.append("ensemble subset sample variance does not reproduce signal_variance")

        b0_columns = np.arange(0, n_columns, columns.n_b)
        if not np.array_equal(vectors[:, b0_columns], np.ones((n_entries, len(b0_columns)))):
            errors.append("S(b=0) is not exactly one in every stored timing pair")
        negative_signal_count = int(np.count_nonzero(vectors < 0.0))
        minimum_signal = float(np.min(vectors))
        if negative_signal_count:
            warnings.append(
                f"{negative_signal_count} negative signal samples; minimum {minimum_signal:g}. "
                "This is a production-quality indicator, not a schema failure."
            )

        se_positive_b = np.sqrt(variance.reshape(n_entries, columns.n_pairs, columns.n_b)[:, :, 1:] / 40.0)
        se_summary = {
            "median": float(np.median(se_positive_b)),
            "q25": float(np.quantile(se_positive_b, 0.25)),
            "q75": float(np.quantile(se_positive_b, 0.75)),
        }

        free = np.asarray(data["is_free_water"], dtype=bool)
        nominal_kios = np.asarray(data["nominal_kios"], dtype=float)
        nominal_rhos = np.asarray(data["nominal_rhos"], dtype=float)
        nominal_vs = np.asarray(data["nominal_Vs"], dtype=float)
        actual_keys = {
            _reported_key(k, rho, volume, is_free)
            for k, rho, volume, is_free in zip(nominal_kios, nominal_rhos, nominal_vs, free)
        }
        if len(actual_keys) != n_entries:
            errors.append("duplicate nominal entry keys in merged artifact")
        if actual_keys != _expected_keys():
            errors.append("merged artifact coordinates do not cover the canonical production grid exactly")
        if int(np.count_nonzero(free)) != 1 or int(np.count_nonzero(~free)) != EXPECTED_CELLULAR:
            errors.append("free-water/cellular entry counts are not 1/18819")
        pps = np.asarray(data["pps"], dtype=float)
        if np.any((pps[~free] < 0.0) | (pps[~free] > 1.0) | ~np.isfinite(pps[~free])):
            errors.append("one or more cellular p_p values lie outside [0, 1]")

        imaginary_t: list[float] = []
        entry_metadata = np.asarray(data["entry_metadata_json"])
        for index, raw in enumerate(entry_metadata):
            try:
                entry = json.loads(str(raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                errors.append(f"entry {index} has invalid metadata JSON")
                continue
            if not free[index]:
                if int((entry.get("boundary") or {}).get("n_escaped", -1)) != 0:
                    errors.append(f"entry {index} recorded a fatal escape")
                check = entry.get("imaginary_signal_check") or {}
                value = check.get("max_abs_standardized_deviation")
                if (check.get("n_ensembles") != PRODUCTION_ENSEMBLES_PER_ENTRY
                        or check.get("degrees_of_freedom") != PRODUCTION_ENSEMBLES_PER_ENTRY - 1
                        or check.get("zero_standard_error_nonzero_count") != 0
                        or value is None or not np.isfinite(float(value))):
                    errors.append(f"entry {index} lacks a valid imaginary-symmetry record")
                else:
                    imaginary_t.append(float(value))
                per_ensemble = entry.get("per_ensemble_geometry") or []
                ref_meta = per_ensemble[0].get("reference_metadata") if per_ensemble else None
                if not isinstance(ref_meta, dict) or (
                    int(ref_meta.get("n_single_cells", 0)) < 5_000_000
                    or ref_meta.get("contraction_rule") != "all_shifted_voronoi_facets_S2"
                    or not np.isclose(float(ref_meta.get("kappa", np.nan)), 0.90, rtol=0.0, atol=1e-12)
                ):
                    errors.append(f"entry {index} lacks certified full-facet geometry provenance")
        expected_t_count = int(np.count_nonzero(~free))
        if len(imaginary_t) != expected_t_count:
            errors.append("imaginary-symmetry metadata is missing for one or more cellular entries")
            imag_max = None
            imag_threshold = None
        else:
            imag_max = float(max(imaginary_t))
            imag_threshold = float(student_t.ppf(
                1.0 - IMAGINARY_FAMILYWISE_ALPHA / (2.0 * expected_t_count * n_columns),
                df=PRODUCTION_ENSEMBLES_PER_ENTRY - 1,
            ))
            if imag_max > imag_threshold:
                errors.append(
                    f"imaginary symmetry failed: max |t|={imag_max:.6g} > threshold {imag_threshold:.6g}"
                )

    return {
        "schema": "madi-v5-production-validation-v1",
        "pass": not errors,
        "artifact": str(artifact),
        "expected_shards": EXPECTED_SHARDS,
        "entries": n_entries,
        "cellular_entries": int(np.count_nonzero(~free)),
        "free_water_entries": int(np.count_nonzero(free)),
        "columns": n_columns,
        "errors": errors,
        "warnings": warnings,
        "subset_mean_max_abs_error": subset_mean_error,
        "subset_variance_max_abs_error": subset_variance_error,
        "imaginary_max_abs_standardized_statistic": imag_max,
        "imaginary_familywise_threshold": imag_threshold,
        "negative_signal_count": negative_signal_count,
        "minimum_signal": minimum_signal,
        "positive_b_signal_se": se_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

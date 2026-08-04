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
  --shards libraries/madi_remediation_pilot.shard*.npz \
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

from madi.library import LibraryEntry, _entry_key, load_library, load_library_meta, make_remediation_log_grid


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
PILOT_PAIRS = [(7.0, 25.0), (7.0, 50.0), (20.0, 25.0), (20.0, 50.0)]
PILOT_B_VALUES = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0]


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
    if int(build.get("walkers_per_ensemble", -1)) != 512 or int(build.get("ensembles_per_entry", -1)) != 1:
        errors.append(f"{path}: pilot must use 512 walkers and one ensemble")
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
    return errors


def validate(paths: list[Path]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    expected_ids = set(range(PILOT_N_SHARDS))
    shard_ids: dict[int, Path] = {}
    all_entries: list[LibraryEntry] = []
    expected_counts: dict[int, int] = {}
    per_shard: list[dict] = []

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
            errors.append(f"entry {_entry_nominal_key(entry)} has vector length {vector.size}, expected 24")
        elif not np.array_equal(vector.reshape(len(PILOT_PAIRS), len(PILOT_B_VALUES))[:, 0], np.ones(len(PILOT_PAIRS))):
            errors.append(f"entry {_entry_nominal_key(entry)} does not have exact S(b=0)=1")
        nonfinite_signal_count += int(np.count_nonzero(~np.isfinite(vector)))
        negative_signal_count += int(np.count_nonzero(vector < 0.0))
        if vector.size:
            min_signal = min(min_signal, float(np.min(vector)))
        if entry.weight is None or not np.isfinite(entry.weight) or entry.weight <= 0.0:
            errors.append(f"entry {_entry_nominal_key(entry)} has no positive finite quadrature weight")
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
        "schema": "madi-remediation-pilot-validation-v1",
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

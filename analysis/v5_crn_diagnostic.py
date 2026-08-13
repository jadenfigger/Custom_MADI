#!/usr/bin/env python3
"""Pre-launch CRN / Monte-Carlo diagnostic for a v5 pilot artifact.

This is deliberately a standalone artifact reader.  It imports only the
standard library, NumPy, and Matplotlib; in particular, it does not import
``madi`` or execute any simulator, fitter, or library-build code.

It accepts one or more ``.npz`` shards (or a directory containing them),
validates the v5 ensemble-index contract before any correlation is computed,
and writes a machine-readable summary, CSV tables, and four figures.

Example
-------
python analysis/v5_crn_diagnostic.py \
  libraries/madi_v5_remediation_pilot.shard*.npz \
  --output-dir docs/figures/v5_crn_diagnostic

The code intentionally uses raw *realised* ``kios``, ``rhos``, and ``Vs``
labels as grouping keys.  It does not use ``nominal_*`` labels and does not
round values to manufacture an axis-neighbour relation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_SCHEMA = "madi-library-v5"
EXPECTED_WALK_SEED_FORMULA = "(build_seed + 104729 * ensemble_index) mod 2^31"
EXPECTED_GEOMETRY_SEED_FORMULA = "(build_seed + 97003 * ensemble_index) mod 2^31"
EXPECTED_CONTRACT_INDEPENDENT_OF = ["rho", "V", "k_io"]
EXPECTED_SUBSET_COLUMNS = 200
PILOT_N_ENSEMBLES = 8
PILOT_N_WALKERS = 512
PRODUCTION_N_ENSEMBLES = 40
PRODUCTION_N_WALKERS = 100_000
LOG_GRID_RATIO = 1.11588399
LOG_GRID_STEP = math.log(LOG_GRID_RATIO)
FLOAT32_VARIANCE_RTOL = 5.0e-5
FLOAT32_VARIANCE_ATOL = 2.0e-7
BOOTSTRAP_REPLICATES = 5_000


class DiagnosticError(RuntimeError):
    """A validation failure that makes the requested analysis unsafe."""


@dataclass(frozen=True)
class Artifact:
    paths: tuple[Path, ...]
    arrays: dict[str, np.ndarray]
    build_metadata: dict[str, Any]
    full_pairs: tuple[tuple[float, float], ...]
    b_values: np.ndarray
    subset_pairs: tuple[tuple[float, float], ...]
    subset_b_values: np.ndarray
    subset_indices: np.ndarray
    n_ensembles: int
    n_walkers: int
    variance_check: dict[str, Any]
    mean_check: dict[str, Any]


@dataclass(frozen=True)
class NeighborPair:
    axis: str
    left_index: int
    right_index: int
    left_value: float
    right_value: float
    context: tuple[float, float]


def _scalar(value: np.ndarray) -> Any:
    """Turn a zero-dimensional NPZ member into a normal Python scalar."""
    if np.asarray(value).shape != ():
        raise DiagnosticError(f"expected scalar metadata member, got shape {np.asarray(value).shape}")
    return np.asarray(value).item()


def _json_scalar(value: np.ndarray, path: Path, key: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(_scalar(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{path}: {key} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise DiagnosticError(f"{path}: {key} must contain a JSON object")
    return parsed


def _as_pair_tuple(first: np.ndarray, second: np.ndarray) -> tuple[tuple[float, float], ...]:
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise DiagnosticError("timing-pair metadata has incompatible shapes")
    return tuple((float(a), float(b)) for a, b in zip(first, second))


def _equal_metadata(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _require_crn_contract(metadata: dict[str, Any], path: Path) -> None:
    """Validate the artifact's explicit ensemble-index pairing contract.

    The task requires an immediate abort if this provenance is missing.  A
    matching seed formula alone is not enough: the metadata must say that the
    index ordering itself is shared across entries and independent of all
    three library coordinates.
    """
    uncertainty = metadata.get("uncertainty")
    if not isinstance(uncertainty, dict):
        raise DiagnosticError(
            f"ABORT: {path} has no v5 uncertainty metadata; cannot establish "
            "the CRN ensemble-index contract."
        )
    contract = uncertainty.get("ensemble_index_ordering_contract")
    if not isinstance(contract, dict):
        raise DiagnosticError(
            f"ABORT: {path} does not record ensemble_index_ordering_contract; "
            "correlating ensemble slots would be invalid."
        )
    expected = {
        "axis_position_in_ensemble_means_subset": 1,
        "index_values": "0..n_ensembles-1",
        "same_order_across_entries": True,
        "independent_of": EXPECTED_CONTRACT_INDEPENDENT_OF,
        "walk_seed_formula": EXPECTED_WALK_SEED_FORMULA,
        "geometry_seed_formula": EXPECTED_GEOMETRY_SEED_FORMULA,
    }
    failed = {
        key: {"expected": wanted, "observed": contract.get(key)}
        for key, wanted in expected.items()
        if contract.get(key) != wanted
    }
    if failed:
        raise DiagnosticError(
            f"ABORT: {path} has an incomplete or incompatible CRN "
            f"ensemble-index contract: {json.dumps(failed, sort_keys=True)}"
        )


def _validate_shard_layout(data: np.lib.npyio.NpzFile, path: Path) -> tuple[int, int, int]:
    required = {
        "library_schema",
        "build_metadata_json",
        "kios",
        "rhos",
        "Vs",
        "is_free_water",
        "nominal_kios",
        "nominal_rhos",
        "nominal_Vs",
        "vectors",
        "signal_imag",
        "signal_variance",
        "ensemble_means_subset",
        "pair_deltas",
        "pair_Deltas",
        "b_values",
        "n_b",
        "ensemble_subset_pair_deltas",
        "ensemble_subset_pair_Deltas",
        "ensemble_subset_b_values",
        "ensemble_subset_n_b",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise DiagnosticError(f"{path}: missing required v5 members {missing}")
    if str(_scalar(data["library_schema"])) != EXPECTED_SCHEMA:
        raise DiagnosticError(
            f"{path}: expected library schema {EXPECTED_SCHEMA!r}, got "
            f"{str(_scalar(data['library_schema']))!r}"
        )

    n_entries = len(data["kios"])
    if not all(
        len(data[name]) == n_entries
        for name in (
            "rhos", "Vs", "is_free_water", "nominal_kios", "nominal_rhos", "nominal_Vs",
        )
    ):
        raise DiagnosticError(f"{path}: parameter arrays disagree on entry count")
    n_b = int(_scalar(data["n_b"]))
    n_columns = len(data["pair_deltas"]) * n_b
    n_subset = len(data["ensemble_subset_pair_deltas"]) * int(
        _scalar(data["ensemble_subset_n_b"])
    )

    for name, expected_dtype, expected_shape in (
        ("vectors", np.dtype(np.float64), (n_entries, n_columns)),
        ("signal_imag", np.dtype(np.float32), (n_entries, n_columns)),
        ("signal_variance", np.dtype(np.float32), (n_entries, n_columns)),
    ):
        array = data[name]
        if array.dtype != expected_dtype or array.shape != expected_shape:
            raise DiagnosticError(
                f"{path}: {name} has dtype/shape {array.dtype}/{array.shape}; "
                f"expected {expected_dtype}/{expected_shape}"
            )
    subset = data["ensemble_means_subset"]
    if subset.dtype != np.dtype(np.float32) or subset.ndim != 3:
        raise DiagnosticError(
            f"{path}: ensemble_means_subset must be rank-3 float32, got "
            f"{subset.dtype}/{subset.shape}"
        )
    if subset.shape[0] != n_entries or subset.shape[2] != n_subset:
        raise DiagnosticError(
            f"{path}: ensemble_means_subset shape {subset.shape} does not match "
            f"entry/subset dimensions ({n_entries}, *, {n_subset})"
        )
    if n_subset != EXPECTED_SUBSET_COLUMNS:
        raise DiagnosticError(
            f"{path}: declared diagnostic subset contains {n_subset} columns, "
            f"not the required {EXPECTED_SUBSET_COLUMNS}"
        )
    for name in ("vectors", "signal_variance", "ensemble_means_subset"):
        if not np.all(np.isfinite(data[name])):
            raise DiagnosticError(f"{path}: {name} contains a non-finite value")
    if np.any(data["signal_variance"] < 0):
        raise DiagnosticError(f"{path}: signal_variance contains a negative value")
    return n_entries, n_columns, int(subset.shape[1])


def _expand_paths(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            matches = sorted(candidate.glob("madi_v5_remediation_pilot.shard*.npz"))
            if not matches:
                matches = sorted(candidate.glob("*.npz"))
            paths.extend(matches)
            continue
        matches = sorted(Path(item) for item in glob.glob(value))
        paths.extend(matches if matches else [candidate])
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise DiagnosticError("no artifact paths were supplied")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise DiagnosticError(f"artifact path(s) not found: {missing}")
    return unique


def _subset_indices(
    full_pairs: tuple[tuple[float, float], ...],
    b_values: np.ndarray,
    subset_pairs: tuple[tuple[float, float], ...],
    subset_b_values: np.ndarray,
) -> np.ndarray:
    """Map the declared pair-major/b-major subset into the main matrix."""
    indices: list[int] = []
    n_b = len(b_values)
    for pair in subset_pairs:
        matching_pairs = [i for i, candidate in enumerate(full_pairs) if candidate == pair]
        if len(matching_pairs) != 1:
            raise DiagnosticError(
                f"declared subset pair {pair!r} maps to {len(matching_pairs)} main-grid pairs"
            )
        pair_index = matching_pairs[0]
        for b in subset_b_values:
            matching_b = np.flatnonzero(b_values == b)
            if len(matching_b) != 1:
                raise DiagnosticError(
                    f"declared subset b={float(b):g} maps to {len(matching_b)} main-grid columns"
                )
            indices.append(pair_index * n_b + int(matching_b[0]))
    return np.asarray(indices, dtype=np.int64)


def load_artifact(paths: list[Path], expected_shards: int | None) -> Artifact:
    if expected_shards is not None and len(paths) != expected_shards:
        raise DiagnosticError(
            f"expected {expected_shards} shard files, received {len(paths)}; "
            "refusing to diagnose an incomplete pilot artifact"
        )

    dynamic_names = (
        "kios",
        "rhos",
        "Vs",
        "is_free_water",
        "nominal_kios",
        "nominal_rhos",
        "nominal_Vs",
        "vectors",
        "signal_imag",
        "signal_variance",
        "ensemble_means_subset",
        "entry_metadata_json",
    )
    combined: dict[str, list[np.ndarray]] = defaultdict(list)
    static_reference: dict[str, np.ndarray] = {}
    metadata_reference: dict[str, Any] | None = None
    n_ensembles_reference: int | None = None
    n_columns_reference: int | None = None
    entry_metadata_missing = False

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            _, n_columns, n_ensembles = _validate_shard_layout(data, path)
            metadata = _json_scalar(data["build_metadata_json"], path, "build_metadata_json")
            _require_crn_contract(metadata, path)
            if metadata.get("schema") != EXPECTED_SCHEMA:
                raise DiagnosticError(f"{path}: build metadata does not record {EXPECTED_SCHEMA!r}")
            if int(metadata.get("ensembles_per_entry", -1)) != n_ensembles:
                raise DiagnosticError(
                    f"{path}: metadata n_ensembles does not match ensemble_means_subset"
                )
            if int(metadata.get("walkers_per_ensemble", -1)) <= 0:
                raise DiagnosticError(f"{path}: metadata has no positive walkers_per_ensemble")
            if metadata_reference is None:
                metadata_reference = metadata
                n_ensembles_reference = n_ensembles
                n_columns_reference = n_columns
            else:
                if not _equal_metadata(metadata_reference, metadata):
                    raise DiagnosticError(f"{path}: build metadata differs from the other shard(s)")
                if n_ensembles != n_ensembles_reference or n_columns != n_columns_reference:
                    raise DiagnosticError(f"{path}: shape contract differs from the other shard(s)")

            for name in dynamic_names:
                if name not in data.files:
                    if name == "entry_metadata_json":
                        # Geometry reuse is only an auxiliary provenance check.
                        entry_metadata_missing = True
                        continue
                    raise DiagnosticError(f"{path}: missing required dynamic member {name}")
                combined[name].append(np.asarray(data[name]).copy())
            for name in (
                "pair_deltas",
                "pair_Deltas",
                "b_values",
                "n_b",
                "ensemble_subset_pair_deltas",
                "ensemble_subset_pair_Deltas",
                "ensemble_subset_b_values",
                "ensemble_subset_n_b",
            ):
                value = np.asarray(data[name]).copy()
                if name not in static_reference:
                    static_reference[name] = value
                elif not np.array_equal(static_reference[name], value):
                    raise DiagnosticError(f"{path}: static grid member {name} differs across shards")

    assert metadata_reference is not None
    assert n_ensembles_reference is not None
    if entry_metadata_missing:
        # Do not retain a partial entry-metadata array whose indices no longer
        # align with the concatenated signal arrays.
        combined.pop("entry_metadata_json", None)
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in combined.items()}
    if len(arrays["kios"]) == 0:
        raise DiagnosticError("artifact contains no entries")
    if len(arrays["ensemble_means_subset"]) != len(arrays["kios"]):
        raise DiagnosticError("combined diagnostic arrays have an inconsistent entry axis")

    full_pairs = _as_pair_tuple(static_reference["pair_deltas"], static_reference["pair_Deltas"])
    b_values = np.asarray(static_reference["b_values"], dtype=np.float64)
    subset_pairs = _as_pair_tuple(
        static_reference["ensemble_subset_pair_deltas"],
        static_reference["ensemble_subset_pair_Deltas"],
    )
    subset_b_values = np.asarray(static_reference["ensemble_subset_b_values"], dtype=np.float64)
    subset_n_b = int(_scalar(static_reference["ensemble_subset_n_b"]))
    if subset_n_b != len(subset_b_values):
        raise DiagnosticError("declared subset n_b does not match declared subset b-values")
    if len(subset_pairs) * subset_n_b != EXPECTED_SUBSET_COLUMNS:
        raise DiagnosticError("declared subset axes do not form 200 pair-major/b-major columns")

    uncertainty = metadata_reference["uncertainty"]
    subset_metadata = uncertainty.get("ensemble_means_subset", {})
    expected_pairs = [[float(delta), float(Delta)] for delta, Delta in subset_pairs]
    if (
        subset_metadata.get("array") != "ensemble_means_subset"
        or subset_metadata.get("declared_delta_Delta_pairs_ms") != expected_pairs
        or subset_metadata.get("declared_b_values_s_mm2") != subset_b_values.tolist()
        or subset_metadata.get("column_order") != "pair-major, then b-major within each pair"
    ):
        raise DiagnosticError(
            "declared ensemble subset metadata does not match the stored subset axes/order"
        )

    indices = _subset_indices(full_pairs, b_values, subset_pairs, subset_b_values)
    if indices.shape != (EXPECTED_SUBSET_COLUMNS,):
        raise DiagnosticError("could not construct the expected 200 main-grid subset indices")

    means = arrays["ensemble_means_subset"].astype(np.float64)
    stored_variance = arrays["signal_variance"][:, indices].astype(np.float64)
    recomputed_variance = np.var(means, axis=1, ddof=1)
    variance_difference = np.abs(recomputed_variance - stored_variance)
    variance_ok = np.isclose(
        recomputed_variance,
        stored_variance,
        rtol=FLOAT32_VARIANCE_RTOL,
        atol=FLOAT32_VARIANCE_ATOL,
    )
    if not np.all(variance_ok):
        worst = np.unravel_index(int(np.argmax(variance_difference)), variance_difference.shape)
        raise DiagnosticError(
            "ABORT: sample variance reconstructed from ensemble_means_subset does not "
            "match signal_variance within float32 tolerance. "
            f"Worst entry/subset column={worst}, stored={stored_variance[worst]:.9g}, "
            f"recomputed={recomputed_variance[worst]:.9g}, "
            f"abs_error={variance_difference[worst]:.3g}."
        )
    nonzero_variance = stored_variance > 0.0
    relative_difference = np.divide(
        variance_difference,
        stored_variance,
        out=np.zeros_like(variance_difference),
        where=nonzero_variance,
    )
    variance_check = {
        "rtol": FLOAT32_VARIANCE_RTOL,
        "atol": FLOAT32_VARIANCE_ATOL,
        "max_abs_error": float(np.max(variance_difference)),
        "max_relative_error_positive_variance": float(np.max(relative_difference)),
        "zero_variance_values": int(np.count_nonzero(~nonzero_variance)),
        "values_checked": int(variance_difference.size),
    }

    subset_mean = np.mean(means, axis=1)
    vector_subset = arrays["vectors"][:, indices].astype(np.float64)
    mean_difference = np.abs(subset_mean - vector_subset)
    if np.max(mean_difference) > 5.0e-6:
        raise DiagnosticError(
            "ABORT: declared subset mean does not reconstruct the matching main signal columns; "
            f"max absolute error is {float(np.max(mean_difference)):.3g}."
        )
    mean_check = {
        "max_abs_error": float(np.max(mean_difference)),
        "values_checked": int(mean_difference.size),
    }

    free = np.asarray(arrays["is_free_water"], dtype=bool)
    seen: set[tuple[float, float, float]] = set()
    for i in np.flatnonzero(~free):
        key = (float(arrays["kios"][i]), float(arrays["rhos"][i]), float(arrays["Vs"][i]))
        if key in seen:
            raise DiagnosticError(f"duplicate realised cellular label {key!r}")
        seen.add(key)

    return Artifact(
        paths=tuple(paths),
        arrays=arrays,
        build_metadata=metadata_reference,
        full_pairs=full_pairs,
        b_values=b_values,
        subset_pairs=subset_pairs,
        subset_b_values=subset_b_values,
        subset_indices=indices,
        n_ensembles=n_ensembles_reference,
        n_walkers=int(metadata_reference["walkers_per_ensemble"]),
        variance_check=variance_check,
        mean_check=mean_check,
    )


def _axis_spec(axis: str) -> tuple[str, tuple[str, str]]:
    if axis == "k_io":
        return "kios", ("rhos", "Vs")
    if axis == "rho":
        return "rhos", ("kios", "Vs")
    if axis == "V":
        return "Vs", ("kios", "rhos")
    raise ValueError(axis)


def build_adjacencies(artifact: Artifact) -> tuple[dict[str, list[NeighborPair]], dict[str, list[dict[str, Any]]]]:
    """Find immediate neighbours with raw realised labels only."""
    arrays = artifact.arrays
    cellular = ~np.asarray(arrays["is_free_water"], dtype=bool)
    neighbors: dict[str, list[NeighborPair]] = {}
    group_records: dict[str, list[dict[str, Any]]] = {}
    for axis in ("k_io", "rho", "V"):
        variable_name, fixed_names = _axis_spec(axis)
        groups: dict[tuple[float, float], list[tuple[float, int]]] = defaultdict(list)
        for index in np.flatnonzero(cellular):
            context = tuple(float(arrays[name][index]) for name in fixed_names)
            groups[context].append((float(arrays[variable_name][index]), int(index)))
        axis_groups: list[dict[str, Any]] = []
        axis_neighbors: list[NeighborPair] = []
        for context, members in sorted(groups.items()):
            members.sort(key=lambda item: item[0])
            values = [value for value, _ in members]
            if len(set(values)) != len(values):
                raise DiagnosticError(
                    f"{axis}: repeated realised coordinate within fixed-label group {context!r}"
                )
            record = {
                "context": context,
                "members": members,
                "n_entries": len(members),
            }
            axis_groups.append(record)
            for left, right in zip(members[:-1], members[1:]):
                axis_neighbors.append(
                    NeighborPair(
                        axis=axis,
                        left_index=left[1],
                        right_index=right[1],
                        left_value=left[0],
                        right_value=right[0],
                        context=context,
                    )
                )
        neighbors[axis] = axis_neighbors
        group_records[axis] = axis_groups
    return neighbors, group_records


def geometry_reuse_summary(artifact: Artifact, kio_groups: list[dict[str, Any]]) -> dict[str, Any]:
    """An auxiliary check that the k_io sweeps retain matching geometry records."""
    entry_json = artifact.arrays.get("entry_metadata_json")
    groups = [group for group in kio_groups if group["n_entries"] > 1]
    if entry_json is None:
        return {"available": False, "reason": "entry_metadata_json is absent"}
    identical = 0
    missing = 0
    mismatched = 0
    for group in groups:
        records: list[str] = []
        for _, index in group["members"]:
            try:
                metadata = json.loads(str(entry_json[index]))
                geometry = metadata["per_ensemble_geometry"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                missing += 1
                records = []
                break
            records.append(json.dumps(geometry, sort_keys=True, separators=(",", ":")))
        if records:
            if len(set(records)) == 1:
                identical += 1
            else:
                mismatched += 1
    return {
        "available": True,
        "groups_with_multiple_kio": len(groups),
        "groups_with_identical_per_ensemble_geometry": identical,
        "groups_with_mismatched_per_ensemble_geometry": mismatched,
        "groups_missing_geometry_metadata": missing,
    }


def paired_correlation_matrix(
    artifact: Artifact,
    left_indices: np.ndarray | list[int],
    right_indices: np.ndarray | list[int],
    ensemble_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired-ensemble correlations/covariances for entry-pair rows.

    With the default ``ensemble_indices=None``, the denominator deliberately
    uses the stored ``signal_variance``.  That is the schema cross-check used
    by the CRN diagnostic.  A caller performing an ensemble-index bootstrap
    may supply resampled indices; in that case both variances are recomputed
    from the resample so the correlation remains a valid sample statistic.
    """
    means = artifact.arrays["ensemble_means_subset"].astype(np.float64, copy=False)
    left = np.asarray(left_indices, dtype=np.int64)
    right = np.asarray(right_indices, dtype=np.int64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or left.size == 0:
        raise DiagnosticError("paired correlation requires equal non-empty one-dimensional entry-index arrays")
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= len(means)) or np.any(right >= len(means)):
        raise DiagnosticError("paired correlation received an out-of-range entry index")
    x = means[left]
    y = means[right]
    if ensemble_indices is not None:
        indices = np.asarray(ensemble_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size < 2:
            raise DiagnosticError("correlation bootstrap requires at least two ensemble indices")
        if np.any(indices < 0) or np.any(indices >= artifact.n_ensembles):
            raise DiagnosticError("correlation bootstrap supplied an out-of-range ensemble index")
        x = x[:, indices, :]
        y = y[:, indices, :]
        variance_left = np.var(x, axis=1, ddof=1)
        variance_right = np.var(y, axis=1, ddof=1)
        denominator_count = float(indices.size - 1)
    else:
        variance = artifact.arrays["signal_variance"][:, artifact.subset_indices].astype(
            np.float64, copy=False
        )
        variance_left = variance[left]
        variance_right = variance[right]
        denominator_count = float(artifact.n_ensembles - 1)
    covariance = np.sum(
        (x - np.mean(x, axis=1, keepdims=True))
        * (y - np.mean(y, axis=1, keepdims=True)),
        axis=1,
    ) / denominator_count
    denominator = np.sqrt(variance_left * variance_right)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0.0,
    )
    finite = np.isfinite(correlation)
    if np.any(np.abs(correlation[finite]) > 1.001):
        raise DiagnosticError(
            "computed sample correlation exceeds [-1, 1] beyond float32 round-off; "
            "do not trust this variance/subset interpretation"
        )
    return correlation, covariance


def pair_correlation(
    artifact: Artifact,
    left: int,
    right: int,
    ensemble_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired-ensemble correlation and covariance for one entry pair."""
    correlation, covariance = paired_correlation_matrix(
        artifact,
        np.asarray([left]),
        np.asarray([right]),
        ensemble_indices=ensemble_indices,
    )
    return correlation[0], covariance[0]


# Kept as a private compatibility alias for the pilot diagnostic's existing
# internal callers.  New companion analyses should use ``pair_correlation``.
def _pair_correlation(artifact: Artifact, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    return pair_correlation(artifact, left, right)


def _summary_stats(values: np.ndarray) -> dict[str, Any] | None:
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    return {
        "n": int(flat.size),
        "min": float(np.min(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q25": float(np.quantile(flat, 0.25)),
        "median": float(np.median(flat)),
        "q75": float(np.quantile(flat, 0.75)),
        "q95": float(np.quantile(flat, 0.95)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


def _group_block_bootstrap(pair_values: np.ndarray, pairs: list[NeighborPair]) -> dict[str, Any] | None:
    """Resample fixed-coordinate groups, not individual correlated columns."""
    blocks: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for values, pair in zip(pair_values, pairs):
        blocks[pair.context].append(values)
    block_values = [np.vstack(rows) for rows in blocks.values()]
    if len(block_values) < 2:
        return None
    rng = np.random.default_rng(20260812)
    medians = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPLICATES):
        selected = [block_values[index] for index in rng.integers(0, len(block_values), len(block_values))]
        medians[repetition] = np.nanmedian(np.concatenate(selected, axis=0))
    return {
        "method": "fixed-coordinate-group block bootstrap",
        "n_blocks": len(block_values),
        "n_replicates": BOOTSTRAP_REPLICATES,
        "median": float(np.median(medians)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
        "bootstrap_standard_error": float(np.std(medians, ddof=1)),
    }


def compute_correlations(
    artifact: Artifact, neighbors: dict[str, list[NeighborPair]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for axis in ("k_io", "V", "rho"):
        pairs = neighbors[axis]
        values: list[np.ndarray] = []
        covariances: list[np.ndarray] = []
        for pair in pairs:
            correlation, covariance = _pair_correlation(artifact, pair.left_index, pair.right_index)
            values.append(correlation)
            covariances.append(covariance)
        matrix = (
            np.vstack(values)
            if values
            else np.empty((0, EXPECTED_SUBSET_COLUMNS), dtype=np.float64)
        )
        invalid = int(np.count_nonzero(~np.isfinite(matrix)))
        aggregate = _summary_stats(matrix)
        status = "available" if len(pairs) >= 2 else "insufficient_adjacent_pairs"
        reason = None
        if not pairs:
            reason = "no neighbouring entries vary along exactly this realised-label axis"
        elif len(pairs) == 1:
            reason = "only one neighbouring pair is available; no axis-level conclusion is reported"
        pair_summaries = []
        transition_rows: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
        for pair, row in zip(pairs, matrix):
            summary = _summary_stats(row)
            transition_rows[(pair.left_value, pair.right_value)].append(row)
            pair_summaries.append(
                {
                    "left_index": pair.left_index,
                    "right_index": pair.right_index,
                    "left_value": pair.left_value,
                    "right_value": pair.right_value,
                    "context": list(pair.context),
                    "summary": summary,
                }
            )
        transition_summaries = [
            {
                "left_value": left_value,
                "right_value": right_value,
                "n_neighbor_pairs": len(rows),
                "summary": _summary_stats(np.vstack(rows)),
            }
            for (left_value, right_value), rows in sorted(transition_rows.items())
        ]
        output[axis] = {
            "pairs": pairs,
            "values": matrix,
            "covariances": covariances,
            "status": status,
            "reason": reason,
            "aggregate": aggregate if status == "available" else None,
            "descriptive_single_pair_aggregate": aggregate if status != "available" else None,
            "undefined_values": invalid,
            "pair_summaries": pair_summaries,
            "transition_summaries": transition_summaries,
            "bootstrap": _group_block_bootstrap(matrix, pairs) if status == "available" else None,
        }
    return output


def _column_labels(artifact: Artifact) -> list[dict[str, float]]:
    return [
        {"delta_ms": pair[0], "Delta_ms": pair[1], "b_s_mm2": float(b)}
        for pair in artifact.subset_pairs
        for b in artifact.subset_b_values
    ]


def compute_se_projection(artifact: Artifact) -> dict[str, Any]:
    """Decompose observed pilot SE into nominal-independent plus residual.

    The nominal 1/sqrt(3EW) term is an upper-bound style walker floor, so a
    low-variance column (notably b=0) can fall below it.  Assigning that
    column a negative residual would be nonsensical.  We therefore cap the
    pilot independent component at its observed variance and call only the
    positive excess over the nominal term "residual".
    """
    variance = artifact.arrays["signal_variance"][:, artifact.subset_indices].astype(
        np.float64, copy=False
    )
    observed_variance = variance / float(artifact.n_ensembles)
    observed_se = np.sqrt(observed_variance)
    nominal_se = 1.0 / math.sqrt(3.0 * artifact.n_ensembles * artifact.n_walkers)
    nominal_variance = nominal_se**2
    walker_component_pilot = np.minimum(observed_variance, nominal_variance)
    residual_component_pilot = np.maximum(observed_variance - nominal_variance, 0.0)

    configurations = (
        ("20 x 200,000", 20, 200_000),
        ("40 x 100,000 (current)", 40, 100_000),
        ("60 x 66,667", 60, 66_667),
        ("80 x 50,000", 80, 50_000),
        ("100 x 40,000", 100, 40_000),
    )
    projection: dict[str, np.ndarray] = {}
    configuration_summaries: list[dict[str, Any]] = []
    cellular = ~np.asarray(artifact.arrays["is_free_water"], dtype=bool)
    b_nonzero = np.tile(artifact.subset_b_values > 0.0, len(artifact.subset_pairs))
    for label, n_ensembles, n_walkers in configurations:
        walker_component = walker_component_pilot * (
            artifact.n_ensembles * artifact.n_walkers / float(n_ensembles * n_walkers)
        )
        residual_component = residual_component_pilot * (
            artifact.n_ensembles / float(n_ensembles)
        )
        projected_se = np.sqrt(walker_component + residual_component)
        projection[label] = projected_se
        summary = _summary_stats(projected_se[cellular][:, b_nonzero])
        assert summary is not None
        configuration_summaries.append(
            {
                "label": label,
                "n_ensembles": n_ensembles,
                "n_walkers": n_walkers,
                "axis_walks": int(3 * n_ensembles * n_walkers),
                "summary_b_gt_0_cellular": summary,
            }
        )

    residual_fraction = np.divide(
        residual_component_pilot,
        observed_variance,
        out=np.zeros_like(observed_variance),
        where=observed_variance > 0.0,
    )
    return {
        "observed_se": observed_se,
        "nominal_se": nominal_se,
        "nominal_variance": nominal_variance,
        "se_ratio": observed_se / nominal_se,
        "walker_component_pilot": walker_component_pilot,
        "residual_component_pilot": residual_component_pilot,
        "residual_fraction": residual_fraction,
        "projection": projection,
        "configuration_summaries": configuration_summaries,
        "configurations": configurations,
        "b_nonzero_mask": b_nonzero,
    }


def _central_stencils(
    axis: str,
    groups: list[dict[str, Any]],
    width: int,
    relative_tolerance: float,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return only symmetric, declared-log-grid central stencils.

    The supplied pilot has a piecewise-linear k_io axis that includes zero.
    It cannot be placed onto the supplied log-grid stencil, so this function
    deliberately declines to invent a k_io beta for it.
    """
    if axis == "k_io":
        return [], (
            "the pilot k_io axis is {0, 20, 130} s^-1 (includes zero and is not a "
            "symmetric log grid), so the declared log-grid central-stencil formula does not apply"
        )
    target_h = width * LOG_GRID_STEP
    candidates = 0
    stencils: list[dict[str, Any]] = []
    for group in groups:
        members = group["members"]
        for center_position in range(width, len(members) - width):
            candidates += 1
            left_value, left_index = members[center_position - width]
            center_value, center_index = members[center_position]
            right_value, right_index = members[center_position + width]
            if min(left_value, center_value, right_value) <= 0.0:
                continue
            h_minus = math.log(center_value / left_value)
            h_plus = math.log(right_value / center_value)
            if not (
                math.isclose(h_minus, target_h, rel_tol=relative_tolerance, abs_tol=1e-12)
                and math.isclose(h_plus, target_h, rel_tol=relative_tolerance, abs_tol=1e-12)
            ):
                continue
            stencils.append(
                {
                    "left_index": left_index,
                    "center_index": center_index,
                    "right_index": right_index,
                    "context": group["context"],
                    "h_minus": h_minus,
                    "h_plus": h_plus,
                }
            )
    if stencils:
        return stencils, None
    if candidates == 0:
        return [], f"no realised-label group contains the required {2 * width + 1} nodes"
    return [], (
        f"available groups do not satisfy symmetric ±{width} declared-log-grid "
        f"spacing ({target_h:.6g})"
    )


def compute_beta(
    artifact: Artifact,
    groups: dict[str, list[dict[str, Any]]],
    projection: dict[str, Any],
    stencil_relative_tolerance: float,
) -> dict[str, Any]:
    """Project beta for future qualifying artifacts; report NA for absent stencils.

    Endpoint covariance is projected as r_pilot * SE_plus * SE_minus.  This
    holds the observed total CRN correlation fixed because a single pilot does
    not identify separate walker and residual covariance components.
    """
    vectors = artifact.arrays["vectors"][:, artifact.subset_indices].astype(np.float64, copy=False)
    results: dict[str, Any] = {}
    for config_label, projected_se in projection["projection"].items():
        config_result: dict[str, Any] = {}
        for axis in ("k_io", "V", "rho"):
            axis_result: dict[str, Any] = {}
            for width in range(1, 5):
                stencils, reason = _central_stencils(
                    axis, groups[axis], width, stencil_relative_tolerance
                )
                if not stencils:
                    axis_result[str(width)] = {
                        "status": "not_estimable",
                        "reason": reason,
                        "n_stencils": 0,
                        "summary": None,
                    }
                    continue
                beta_rows: list[np.ndarray] = []
                for stencil in stencils:
                    left = stencil["left_index"]
                    right = stencil["right_index"]
                    corr, _ = _pair_correlation(artifact, left, right)
                    denominator = 2.0 * width * LOG_GRID_STEP
                    derivative = (vectors[right] - vectors[left]) / denominator
                    covariance = corr * projected_se[left] * projected_se[right]
                    variance_derivative = (
                        projected_se[left] ** 2
                        + projected_se[right] ** 2
                        - 2.0 * covariance
                    ) / denominator**2
                    variance_derivative = np.maximum(variance_derivative, 0.0)
                    beta = np.divide(
                        variance_derivative,
                        derivative**2,
                        out=np.full_like(variance_derivative, np.nan),
                        where=np.isfinite(corr) & (derivative != 0.0),
                    )
                    beta_rows.append(beta)
                beta_matrix = np.vstack(beta_rows)
                axis_result[str(width)] = {
                    "status": "available",
                    "reason": None,
                    "n_stencils": len(stencils),
                    "summary": _summary_stats(beta_matrix),
                }
            config_result[axis] = axis_result
        results[config_label] = config_result
    return results


def _display_context(axis: str, context: tuple[float, float]) -> str:
    _, fixed_names = _axis_spec(axis)
    return ", ".join(f"{name}={value:.8g}" for name, value in zip(fixed_names, context))


def _write_correlations_csv(output_dir: Path, artifact: Artifact, correlations: dict[str, Any]) -> None:
    labels = _column_labels(artifact)
    with (output_dir / "crn_correlations_by_pair_and_column.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "axis",
                "left_entry",
                "right_entry",
                "left_value",
                "right_value",
                "fixed_coordinate_1",
                "fixed_coordinate_2",
                "delta_ms",
                "Delta_ms",
                "b_s_mm2",
                "correlation",
            ),
        )
        writer.writeheader()
        for axis, result in correlations.items():
            for pair, values in zip(result["pairs"], result["values"]):
                for column, correlation in enumerate(values):
                    label = labels[column]
                    writer.writerow(
                        {
                            "axis": axis,
                            "left_entry": pair.left_index,
                            "right_entry": pair.right_index,
                            "left_value": pair.left_value,
                            "right_value": pair.right_value,
                            "fixed_coordinate_1": pair.context[0],
                            "fixed_coordinate_2": pair.context[1],
                            **label,
                            "correlation": "" if not np.isfinite(correlation) else correlation,
                        }
                    )


def _write_reallocation_csv(output_dir: Path, artifact: Artifact, projection: dict[str, Any]) -> None:
    labels = _column_labels(artifact)
    cellular = ~np.asarray(artifact.arrays["is_free_water"], dtype=bool)
    with (output_dir / "projected_reallocation_by_column.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "configuration",
                "n_ensembles",
                "n_walkers",
                "axis_walks",
                "delta_ms",
                "Delta_ms",
                "b_s_mm2",
                "n_cellular_entries",
                "se_min",
                "se_q05",
                "se_q25",
                "se_median",
                "se_q75",
                "se_q95",
                "se_max",
                "se_mean",
            ),
        )
        writer.writeheader()
        configs = {label: (n_ensembles, n_walkers) for label, n_ensembles, n_walkers in projection["configurations"]}
        for config_label, projected_se in projection["projection"].items():
            n_ensembles, n_walkers = configs[config_label]
            for column, label in enumerate(labels):
                summary = _summary_stats(projected_se[cellular, column])
                assert summary is not None
                writer.writerow(
                    {
                        "configuration": config_label,
                        "n_ensembles": n_ensembles,
                        "n_walkers": n_walkers,
                        "axis_walks": 3 * n_ensembles * n_walkers,
                        **label,
                        "n_cellular_entries": int(np.count_nonzero(cellular)),
                        **{f"se_{key}": value for key, value in summary.items() if key != "n"},
                    }
                )


def _write_beta_csv(output_dir: Path, beta: dict[str, Any]) -> None:
    with (output_dir / "projected_beta_by_stencil.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "configuration",
                "axis",
                "stencil_width_k",
                "status",
                "reason",
                "n_stencils",
                "n_values",
                "beta_q05",
                "beta_q25",
                "beta_median",
                "beta_q75",
                "beta_q95",
            ),
        )
        writer.writeheader()
        for configuration, axes in beta.items():
            for axis, widths in axes.items():
                for width, result in widths.items():
                    summary = result["summary"] or {}
                    writer.writerow(
                        {
                            "configuration": configuration,
                            "axis": axis,
                            "stencil_width_k": width,
                            "status": result["status"],
                            "reason": result["reason"] or "",
                            "n_stencils": result["n_stencils"],
                            "n_values": summary.get("n", ""),
                            "beta_q05": summary.get("q05", ""),
                            "beta_q25": summary.get("q25", ""),
                            "beta_median": summary.get("median", ""),
                            "beta_q75": summary.get("q75", ""),
                            "beta_q95": summary.get("q95", ""),
                        }
                    )


def _plot_correlations(output_dir: Path, correlations: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True, constrained_layout=True)
    for axis_name, axis in zip(("k_io", "V", "rho"), axes):
        result = correlations[axis_name]
        values = result["values"]
        finite = values[np.isfinite(values)]
        axis.set_title(axis_name)
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel("ensemble-index correlation r")
        axis.axvline(0.0, color="0.5", lw=0.8)
        if finite.size:
            axis.hist(finite, bins=np.linspace(-1.0, 1.0, 21), color="#3b82b6", edgecolor="white")
            summary = _summary_stats(finite)
            assert summary is not None
            axis.axvline(summary["median"], color="#b22222", lw=1.5, label="median")
            text = (
                f"pairs={len(result['pairs'])}; valid r={finite.size}\n"
                f"median={summary['median']:.3f}\n"
                f"IQR={summary['q25']:.3f} to {summary['q75']:.3f}"
            )
            if result["status"] != "available":
                text += "\nINSUFFICIENT for axis conclusion"
            axis.text(
                0.03,
                0.97,
                text,
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No realised-label\nneighbor pairs",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        if axis is axes[0]:
            axis.set_ylabel("pair-column count")
    fig.suptitle("CRN correlations across the recorded ensemble index (8 ensembles)", fontsize=12)
    fig.savefig(output_dir / "crn_correlation_histograms.png", dpi=180)
    plt.close(fig)


def _plot_se_ratio(output_dir: Path, artifact: Artifact, projection: dict[str, Any]) -> None:
    cellular = ~np.asarray(artifact.arrays["is_free_water"], dtype=bool)
    ratio = projection["se_ratio"][cellular]
    b_values = artifact.subset_b_values
    n_b = len(b_values)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for pair_index, pair in enumerate(artifact.subset_pairs):
        values = ratio[:, pair_index * n_b : (pair_index + 1) * n_b]
        left.plot(
            b_values,
            np.median(values, axis=0),
            marker="o",
            ms=2.5,
            lw=1.1,
            color=cmap(pair_index),
            label=f"({pair[0]:g}, {pair[1]:g}) ms",
        )
    left.axhline(1.0, color="black", ls="--", lw=1.0, label="nominal independent level")
    left.set_title("Median ratio by timing pair")
    left.set_xlabel("b (s/mm²)")
    left.set_ylabel(r"observed SE / $1/\sqrt{3 E N_w}$")
    left.legend(fontsize=7.6, ncol=2, loc="upper right")
    left.grid(alpha=0.2)

    medians = []
    lowers = []
    uppers = []
    for b_index in range(n_b):
        values = ratio[:, b_index::n_b].ravel()
        medians.append(np.median(values))
        lowers.append(np.quantile(values, 0.25))
        uppers.append(np.quantile(values, 0.75))
    right.fill_between(b_values, lowers, uppers, color="#3b82b6", alpha=0.25, label="IQR")
    right.plot(b_values, medians, color="#1f4e79", marker="o", ms=3, label="median")
    right.axhline(1.0, color="black", ls="--", lw=1.0, label="nominal independent level")
    right.set_title("All timing pairs and cellular entries")
    right.set_xlabel("b (s/mm²)")
    right.set_ylabel(r"observed SE / $1/\sqrt{3 E N_w}$")
    right.legend(fontsize=8)
    right.grid(alpha=0.2)
    fig.suptitle(
        "Observed per-column SE relative to the nominal independent walker level\n"
        "b=0 is exactly deterministic and therefore has ratio 0",
        fontsize=11.5,
    )
    fig.savefig(output_dir / "observed_vs_nominal_se_ratio.png", dpi=180)
    plt.close(fig)


def _plot_beta(output_dir: Path, beta: dict[str, Any]) -> None:
    production_label = "40 x 100,000 (current)"
    result = beta[production_label]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for axis_name, axis in zip(("k_io", "V", "rho"), axes):
        widths = result[axis_name]
        available = [(int(width), value["summary"]) for width, value in widths.items() if value["summary"]]
        axis.set_title(axis_name)
        axis.set_xlabel("stencil width k")
        axis.set_xticks([1, 2, 3, 4])
        if available:
            x = np.asarray([item[0] for item in available])
            med = np.asarray([item[1]["median"] for item in available])
            low = np.asarray([item[1]["q25"] for item in available])
            high = np.asarray([item[1]["q75"] for item in available])
            axis.errorbar(x, med, yerr=np.vstack((med - low, high - med)), marker="o", capsize=3)
            axis.set_yscale("log")
            axis.set_ylabel(r"$\beta = \mathrm{Var}(\hat J) / J^2$")
            axis.grid(alpha=0.2)
        else:
            if axis_name == "k_io":
                message = (
                    "No compatible central stencil\n\n"
                    "0, 20, 130 s⁻¹ is asymmetric\n"
                    "and includes k_io = 0; it cannot\n"
                    "use the declared log-grid formula.\n\n"
                    "β(k=1…4) = N/A"
                )
            else:
                message = (
                    "No realised-label central stencil\n\n"
                    "Every fixed-coordinate group\n"
                    "contains one entry only.\n\n"
                    "β(k=1…4) = N/A"
                )
            axis.text(
                0.5,
                0.52,
                message,
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
                wrap=True,
            )
            axis.set_ylim(0, 1)
            axis.set_yticks([])
    fig.suptitle("Projected derivative-noise ratio beta at 40 ensembles × 100,000 walkers", fontsize=11.5)
    fig.savefig(output_dir / "projected_beta_vs_stencil.png", dpi=180)
    plt.close(fig)


def _plot_reallocation_table(output_dir: Path, projection: dict[str, Any], beta: dict[str, Any]) -> None:
    rows = []
    for result in projection["configuration_summaries"]:
        config = result["label"]
        summary = result["summary_b_gt_0_cellular"]
        rho_beta = beta[config]["rho"]["1"]["summary"]
        rows.append(
            [
                config,
                f"{result['axis_walks']:,}",
                f"{summary['median']:.3e}",
                f"{summary['q05']:.3e}–{summary['q95']:.3e}",
                "N/A" if rho_beta is None else f"{rho_beta['median']:.3g}",
            ]
        )
    fig, axis = plt.subplots(figsize=(12.2, 3.9), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=(
            "configuration\n(ensembles × walkers)",
            "axis-walks",
            "median projected SE\n(b>0)",
            "P05–P95 projected SE\n(b>0)",
            "rho beta, k=1",
        ),
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.26, 0.15, 0.20, 0.25, 0.14),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.55)
    for column in range(5):
        table[(0, column)].set_facecolor("#dbeafe")
    axis.set_title(
        "Fixed-cost reallocation projection; detailed per-column values are in projected_reallocation_by_column.csv\n"
        "rho beta is N/A when the artifact has no valid rho central stencil",
        fontsize=10.5,
        pad=12,
    )
    fig.savefig(output_dir / "ensemble_walker_reallocation_table.png", dpi=180)
    plt.close(fig)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_summary(
    artifact: Artifact,
    neighbors: dict[str, list[NeighborPair]],
    groups: dict[str, list[dict[str, Any]]],
    geometry: dict[str, Any],
    correlations: dict[str, Any],
    projection: dict[str, Any],
    beta: dict[str, Any],
) -> dict[str, Any]:
    cellular = ~np.asarray(artifact.arrays["is_free_water"], dtype=bool)
    adjacency: dict[str, Any] = {}
    for axis in ("k_io", "rho", "V"):
        size_counts = Counter(group["n_entries"] for group in groups[axis])
        adjacency[axis] = {
            "n_fixed_coordinate_groups": len(groups[axis]),
            "group_size_counts": dict(sorted(size_counts.items())),
            "n_immediate_neighbor_pairs": len(neighbors[axis]),
            "pairs": [
                {
                    "left_entry": pair.left_index,
                    "right_entry": pair.right_index,
                    "left_value": pair.left_value,
                    "right_value": pair.right_value,
                    "fixed_context": list(pair.context),
                }
                for pair in neighbors[axis]
            ],
        }
    b_nonzero = projection["b_nonzero_mask"]
    ratio_values = projection["se_ratio"][cellular][:, b_nonzero]
    ratio_summary = _summary_stats(ratio_values)
    residual_summary = _summary_stats(projection["residual_fraction"][cellular][:, b_nonzero])
    return _json_ready(
        {
            "artifact_paths": artifact.paths,
            "artifact": {
                "schema": EXPECTED_SCHEMA,
                "n_entries": int(len(artifact.arrays["kios"])),
                "n_cellular_entries": int(np.count_nonzero(cellular)),
                "n_free_water_entries": int(np.count_nonzero(~cellular)),
                "n_ensembles": artifact.n_ensembles,
                "n_walkers": artifact.n_walkers,
                "build_seed": artifact.build_metadata.get("build_seed"),
                "subset_pairs": artifact.subset_pairs,
                "subset_b_values": artifact.subset_b_values,
            },
            "contract": artifact.build_metadata["uncertainty"]["ensemble_index_ordering_contract"],
            "subset_variance_crosscheck": artifact.variance_check,
            "subset_mean_crosscheck": artifact.mean_check,
            "adjacency": adjacency,
            "kio_geometry_reuse_metadata": geometry,
            "correlations": {
                axis: {
                    "status": result["status"],
                    "reason": result["reason"],
                    "aggregate": result["aggregate"],
                    "descriptive_single_pair_aggregate": result["descriptive_single_pair_aggregate"],
                    "undefined_values": result["undefined_values"],
                    "pair_summaries": result["pair_summaries"],
                    "transition_summaries": result["transition_summaries"],
                    "bootstrap": result["bootstrap"],
                }
                for axis, result in correlations.items()
            },
            "se_decomposition": {
                "pilot_nominal_walker_se": projection["nominal_se"],
                "observed_to_nominal_ratio_b_gt_0_cellular": ratio_summary,
                "fraction_b_gt_0_cellular_above_nominal": float(np.mean(ratio_values > 1.0)),
                "fraction_b_gt_0_cellular_below_nominal": float(np.mean(ratio_values < 1.0)),
                "residual_fraction_b_gt_0_cellular": residual_summary,
                "method": (
                    "independent pilot variance=min(observed variance, nominal independent variance); "
                    "residual=max(observed variance - nominal independent variance, 0)"
                ),
                "scaling": {
                    "independent": "variance scales as 1/(3 * n_ensembles * n_walkers)",
                    "residual": "variance scales as 1/n_ensembles",
                },
                "configuration_summaries": projection["configuration_summaries"],
            },
            "beta": beta,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="+",
        help="One or more NPZ shard paths, glob patterns, or a directory containing the shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to receive figures, CSV tables, and v5_crn_diagnostic_summary.json.",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=4,
        help="Require this many shard files (default: 4); pass 0 to disable the count check.",
    )
    parser.add_argument(
        "--stencil-relative-tolerance",
        type=float,
        default=0.03,
        help="Relative tolerance for verifying the declared log-grid central stencil (default: 0.03).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_shards < 0:
        raise DiagnosticError("--expected-shards must be non-negative")
    if args.stencil_relative_tolerance < 0:
        raise DiagnosticError("--stencil-relative-tolerance must be non-negative")
    paths = _expand_paths(args.artifact)
    artifact = load_artifact(paths, None if args.expected_shards == 0 else args.expected_shards)
    neighbors, groups = build_adjacencies(artifact)
    geometry = geometry_reuse_summary(artifact, groups["k_io"])
    correlations = compute_correlations(artifact, neighbors)
    projection = compute_se_projection(artifact)
    beta = compute_beta(artifact, groups, projection, args.stencil_relative_tolerance)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_correlations_csv(output_dir, artifact, correlations)
    _write_reallocation_csv(output_dir, artifact, projection)
    _write_beta_csv(output_dir, beta)
    _plot_correlations(output_dir, correlations)
    _plot_se_ratio(output_dir, artifact, projection)
    _plot_beta(output_dir, beta)
    _plot_reallocation_table(output_dir, projection, beta)
    summary = build_summary(artifact, neighbors, groups, geometry, correlations, projection, beta)
    with (output_dir / "v5_crn_diagnostic_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Validated {len(artifact.paths)} shard(s), {len(artifact.arrays['kios'])} entries.")
    print(
        "Adjacency (immediate realised-label pairs): "
        + ", ".join(f"{axis}={len(neighbors[axis])}" for axis in ("k_io", "rho", "V"))
    )
    print(f"Wrote diagnostic outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

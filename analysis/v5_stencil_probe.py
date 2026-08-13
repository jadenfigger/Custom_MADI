#!/usr/bin/env python3
"""Analyze the declared v5 production-grid stencil/CRN probe.

This companion extends :mod:`v5_crn_diagnostic`; it does not duplicate its
v5 schema, ensemble-index, variance, or subset-reconstruction checks.  The
probe deliberately uses its JSON-declared *canonical grid indices* to define
rho/V adjacency.  Stored ``rhos`` and ``Vs`` are finite-geometry realized
summaries, so literal equality of those labels would not identify a one-axis
production-grid derivative.

Example
-------
python analysis/v5_stencil_probe.py \
  libraries/madi_v5_stencil_probe.shard000.npz \
  libraries/madi_v5_stencil_probe.shard001.npz \
  libraries/madi_v5_stencil_probe.shard002.npz \
  libraries/madi_v5_stencil_probe.shard003.npz \
  --output-dir docs/figures/v5_stencil_probe \
  --report docs/v5_stencil_probe.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.validate_v5_stencil_probe import ProbeDefinition, load_probe_definition  # noqa: E402
from v5_crn_diagnostic import (  # noqa: E402
    Artifact,
    DiagnosticError,
    NeighborPair,
    _column_labels,
    _expand_paths,
    _summary_stats,
    compute_correlations,
    geometry_reuse_summary,
    load_artifact,
    paired_correlation_matrix,
)


AXES = ("rho", "V", "k_io")
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
PILOT_PROJECTION_MEDIAN_SE = 2.019e-4
PILOT_KIO_MEDIAN_CORRELATION = 0.313
PILOT_KIO_IQR = (-0.002, 0.571)
PRODUCTION_N_ENSEMBLES = 40
PRODUCTION_N_WALKERS = 100_000
PRODUCTION_AXIS_WALKS = 3 * PRODUCTION_N_ENSEMBLES * PRODUCTION_N_WALKERS


def _entry_key(kio: float, rho: float, volume: float) -> tuple[float, float, float]:
    """Match the v5 requested-coordinate key precision without rounding data."""
    return (round(float(kio), 4), round(float(rho), 1), round(float(volume), 6))


def _expected_triplets(definition: ProbeDefinition) -> dict[tuple[float, float, float], tuple[int, int, float]]:
    expected: dict[tuple[float, float, float], tuple[int, int, float]] = {}
    for i, j in definition.pairs:
        for kio in definition.kio_values:
            key = _entry_key(kio, float(definition.rhos[i]), float(definition.Vs[j]))
            if key in expected:
                raise DiagnosticError(f"declared probe has a duplicate canonical triplet {key!r}")
            expected[key] = (i, j, float(kio))
    return expected


def validate_probe_artifact(
    artifact: Artifact,
    definition: ProbeDefinition,
) -> tuple[dict[tuple[int, int, float], int], dict[str, Any]]:
    """Map stored entries to the declared canonical cross and abort on drift."""
    arrays = artifact.arrays
    if artifact.n_ensembles != PRODUCTION_N_ENSEMBLES:
        raise DiagnosticError(
            f"ABORT: stencil probe has {artifact.n_ensembles} ensembles, expected the production value "
            f"{PRODUCTION_N_ENSEMBLES}."
        )
    if artifact.n_walkers != PRODUCTION_N_WALKERS:
        raise DiagnosticError(
            f"ABORT: stencil probe has {artifact.n_walkers} walkers/ensemble, expected the production "
            f"value {PRODUCTION_N_WALKERS}."
        )
    metadata = artifact.build_metadata
    expected_metadata = {
        "build_seed": 20260803,
        "T_max_ms": 128.0,
        "ts_ms": 0.001,
        "kappa": 0.90,
        "phase_model": "finite_lobe",
        "boundary_mode": "si_fatal_escape",
    }
    failed_metadata = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    classifier = str(metadata.get("classifier", ""))
    geometry_reference = metadata.get("geometry_reference")
    if "full-facet" not in classifier or "SI Eq. S2" not in classifier:
        failed_metadata["classifier"] = {"expected": "full-facet SI Eq. S2", "observed": classifier}
    if not isinstance(geometry_reference, dict) or (
        geometry_reference.get("required_single_cells") != 5_000_000
        or geometry_reference.get("required_alpha_values") != 26
    ):
        failed_metadata["geometry_reference"] = {
            "expected": {"required_single_cells": 5_000_000, "required_alpha_values": 26},
            "observed": geometry_reference,
        }
    if failed_metadata:
        raise DiagnosticError(
            "ABORT: stencil probe does not record the required production configuration: "
            f"{json.dumps(failed_metadata, sort_keys=True)}"
        )
    if len(artifact.full_pairs) != 1245 or not np.array_equal(
        artifact.b_values, np.arange(0.0, 12_000.0 + 500.0, 500.0)
    ):
        raise DiagnosticError(
            "ABORT: stencil probe does not contain the production full timing/b storage grid "
            "(1245 timing pairs and b=0..12000 in 500 s/mm^2 steps)."
        )
    expected = _expected_triplets(definition)
    free = np.asarray(arrays["is_free_water"], dtype=bool)
    if np.any(free):
        raise DiagnosticError("ABORT: stencil probe unexpectedly contains a free-water entry")
    if len(free) != len(expected):
        raise DiagnosticError(
            f"ABORT: stencil probe has {len(free)} entries, expected exactly {len(expected)} declared entries"
        )

    entry_indices: dict[tuple[int, int, float], int] = {}
    seen: set[tuple[float, float, float]] = set()
    realized_rho_ratio: list[float] = []
    realized_V_ratio: list[float] = []
    for index in range(len(free)):
        key = _entry_key(
            float(arrays["nominal_kios"][index]),
            float(arrays["nominal_rhos"][index]),
            float(arrays["nominal_Vs"][index]),
        )
        if key not in expected:
            raise DiagnosticError(
                "ABORT: artifact nominal coordinate is outside the declared canonical probe cross: "
                f"entry={index}, coordinate={key!r}"
            )
        if key in seen:
            raise DiagnosticError(f"ABORT: duplicate nominal probe coordinate in artifact: {key!r}")
        seen.add(key)
        i, j, kio = expected[key]
        expected_rho = float(definition.rhos[i])
        expected_V = float(definition.Vs[j])
        if not (
            math.isclose(float(arrays["nominal_rhos"][index]), expected_rho, rel_tol=2e-12, abs_tol=2e-12)
            and math.isclose(float(arrays["nominal_Vs"][index]), expected_V, rel_tol=2e-12, abs_tol=2e-12)
            and math.isclose(float(arrays["nominal_kios"][index]), kio, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise DiagnosticError(
                f"ABORT: artifact nominal values do not reproduce declared index ({i}, {j}, {kio:g})"
            )
        if not math.isclose(float(arrays["kios"][index]), kio, rel_tol=0.0, abs_tol=1e-12):
            raise DiagnosticError(
                f"ABORT: stored k_io value at entry {index} differs from its requested production node"
            )
        entry_indices[(i, j, kio)] = index
        realized_rho_ratio.append(float(arrays["rhos"][index]) / expected_rho)
        realized_V_ratio.append(float(arrays["Vs"][index]) / expected_V)

    missing = sorted(set(expected).difference(seen))
    if missing:
        raise DiagnosticError(f"ABORT: artifact is missing declared canonical probe entries: {missing!r}")

    group_records = []
    for i, j in definition.pairs:
        members = [
            (kio, entry_indices[(i, j, kio)])
            for kio in definition.kio_values
        ]
        group_records.append({
            "context": (float(definition.rhos[i]), float(definition.Vs[j])),
            "members": members,
            "n_entries": len(members),
        })
    geometry = geometry_reuse_summary(artifact, group_records)
    if (
        not geometry.get("available")
        or geometry.get("groups_with_multiple_kio") != len(definition.pairs)
        or geometry.get("groups_with_identical_per_ensemble_geometry") != len(definition.pairs)
        or geometry.get("groups_with_mismatched_per_ensemble_geometry", 0) != 0
        or geometry.get("groups_missing_geometry_metadata", 0) != 0
    ):
        raise DiagnosticError(
            "ABORT: per_ensemble_geometry is not identical and available across every fixed nominal "
            f"(rho, V) k_io group: {json.dumps(geometry, sort_keys=True)}"
        )

    realized_summary = {
        "rho_realized_over_nominal": _summary_stats(np.asarray(realized_rho_ratio)),
        "V_realized_over_nominal": _summary_stats(np.asarray(realized_V_ratio)),
        "note": (
            "rhos and Vs are retained finite-geometry realized summaries. Canonical nominal grid indices, "
            "validated above, define the one-axis derivative topology."
        ),
    }
    return entry_indices, {"geometry_reuse": geometry, "realized_labels": realized_summary}


def declared_topology(
    definition: ProbeDefinition,
    entry_indices: dict[tuple[int, int, float], int],
) -> tuple[dict[str, list[NeighborPair]], dict[str, Any]]:
    """Construct immediate neighbours and central stencils from declared indices."""
    selection = definition.declaration["selection"]
    i0, j0 = definition.center
    rho_line = [int(value) for value in selection["rho_line_indices"]]
    V_line = [int(value) for value in selection["V_line_indices"]]
    neighbors: dict[str, list[NeighborPair]] = {axis: [] for axis in AXES}
    stencils: dict[str, dict[int, list[dict[str, Any]]]] = {
        "rho": defaultdict(list), "V": defaultdict(list), "k_io": defaultdict(list),
    }

    for kio in definition.kio_values:
        for left_i, right_i in zip(rho_line[:-1], rho_line[1:]):
            neighbors["rho"].append(NeighborPair(
                axis="rho",
                left_index=entry_indices[(left_i, j0, kio)],
                right_index=entry_indices[(right_i, j0, kio)],
                left_value=float(definition.rhos[left_i]),
                right_value=float(definition.rhos[right_i]),
                context=(float(kio), float(definition.Vs[j0])),
            ))
        for width in selection["supported_half_widths"]["rho"]:
            width = int(width)
            stencils["rho"][width].append({
                "left_index": entry_indices[(i0 - width, j0, kio)],
                "center_index": entry_indices[(i0, j0, kio)],
                "right_index": entry_indices[(i0 + width, j0, kio)],
                "context": (float(kio), float(definition.Vs[j0])),
                "width": width,
                "h1": definition.rho_step,
            })

        for left_j, right_j in zip(V_line[:-1], V_line[1:]):
            neighbors["V"].append(NeighborPair(
                axis="V",
                left_index=entry_indices[(i0, left_j, kio)],
                right_index=entry_indices[(i0, right_j, kio)],
                left_value=float(definition.Vs[left_j]),
                right_value=float(definition.Vs[right_j]),
                context=(float(kio), float(definition.rhos[i0])),
            ))
        for width in selection["supported_half_widths"]["V"]:
            width = int(width)
            stencils["V"][width].append({
                "left_index": entry_indices[(i0, j0 - width, kio)],
                "center_index": entry_indices[(i0, j0, kio)],
                "right_index": entry_indices[(i0, j0 + width, kio)],
                "context": (float(kio), float(definition.rhos[i0])),
                "width": width,
                "h1": definition.V_step,
            })

    for i, j in definition.pairs:
        for left_kio, right_kio in zip(definition.kio_values[:-1], definition.kio_values[1:]):
            neighbors["k_io"].append(NeighborPair(
                axis="k_io",
                left_index=entry_indices[(i, j, left_kio)],
                right_index=entry_indices[(i, j, right_kio)],
                left_value=float(left_kio),
                right_value=float(right_kio),
                context=(float(definition.rhos[i]), float(definition.Vs[j])),
            ))
        stencils["k_io"][1].append({
            "left_index": entry_indices[(i, j, definition.kio_values[0])],
            "center_index": entry_indices[(i, j, definition.kio_values[1])],
            "right_index": entry_indices[(i, j, definition.kio_values[2])],
            "context": (float(definition.rhos[i]), float(definition.Vs[j])),
            "width": 1,
            "h1": 1.0,
        })
    return neighbors, {axis: dict(widths) for axis, widths in stencils.items()}


def paired_ensemble_bootstrap(
    artifact: Artifact,
    pairs: list[NeighborPair],
    repetitions: int,
    seed: int,
) -> dict[str, Any] | None:
    """Bootstrap the aggregate median by resampling aligned ensemble slots.

    Each resample is shared across all pairs/columns, preserving the CRN
    partner relation.  This quantifies finite-ensemble sampling uncertainty;
    it is not a claim that the 200 timing/b columns are independent tissues.
    """
    if not pairs:
        return None
    left = np.asarray([pair.left_index for pair in pairs], dtype=np.int64)
    right = np.asarray([pair.right_index for pair in pairs], dtype=np.int64)
    rng = np.random.default_rng(seed)
    medians = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.integers(0, artifact.n_ensembles, size=artifact.n_ensembles)
        values, _ = paired_correlation_matrix(artifact, left, right, sampled)
        medians[repetition] = float(np.nanmedian(values))
    return {
        "method": "paired ensemble-index bootstrap",
        "n_ensembles": artifact.n_ensembles,
        "n_replicates": repetitions,
        "median": float(np.median(medians)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
        "bootstrap_standard_error": float(np.std(medians, ddof=1)),
        "scope": (
            "conditional on this declared cross; resamples aligned ensemble positions and does not treat "
            "diagnostic columns as independent tissue realizations"
        ),
    }


def compute_probe_correlations(
    artifact: Artifact,
    neighbors: dict[str, list[NeighborPair]],
    repetitions: int,
) -> dict[str, dict[str, Any]]:
    """Use the pilot diagnostic's correlation computation plus probe bootstrap."""
    results = compute_correlations(artifact, neighbors)
    for position, axis in enumerate(AXES):
        results[axis]["fixed_coordinate_block_bootstrap"] = results[axis]["bootstrap"]
        results[axis]["bootstrap"] = paired_ensemble_bootstrap(
            artifact, results[axis]["pairs"], repetitions, seed=20260812 + position,
        )
    return results


def _covariance_of_entry_means(artifact: Artifact, left: int, right: int) -> np.ndarray:
    """Covariance of the two stored entry *means*, retaining CRN alignment."""
    _, covariance_ensemble = paired_correlation_matrix(
        artifact, np.asarray([left]), np.asarray([right]), None,
    )
    return covariance_ensemble[0] / float(artifact.n_ensembles)


def compute_beta(
    artifact: Artifact,
    stencils: dict[str, dict[int, list[dict[str, Any]]]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Compute unprojected production-configuration beta on all declared stencils."""
    vectors = artifact.arrays["vectors"][:, artifact.subset_indices].astype(np.float64, copy=False)
    variance = artifact.arrays["signal_variance"][:, artifact.subset_indices].astype(
        np.float64, copy=False
    ) / float(artifact.n_ensembles)
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for axis in AXES:
        axis_output: dict[int, dict[str, Any]] = {}
        for width, records in sorted(stencils[axis].items()):
            beta_rows: list[np.ndarray] = []
            derivative_rows: list[np.ndarray] = []
            derivative_variance_rows: list[np.ndarray] = []
            covariance_rows: list[np.ndarray] = []
            for record in records:
                left = int(record["left_index"])
                right = int(record["right_index"])
                denominator = 2.0 * int(width) * float(record["h1"])
                derivative = (vectors[right] - vectors[left]) / denominator
                covariance = _covariance_of_entry_means(artifact, left, right)
                numerator = variance[left] + variance[right] - 2.0 * covariance
                tolerance = 1.0e-12 * np.maximum(
                    np.maximum(variance[left], variance[right]), np.abs(2.0 * covariance)
                )
                materially_negative = numerator < -tolerance
                if np.any(materially_negative):
                    worst = int(np.argmin(numerator))
                    raise DiagnosticError(
                        "ABORT: central-difference variance is materially negative; paired covariance or "
                        f"subset indexing is inconsistent (axis={axis}, k={width}, column={worst})."
                    )
                derivative_variance = np.maximum(numerator, 0.0) / denominator**2
                beta = np.full_like(derivative, np.nan)
                nonzero_derivative = derivative != 0.0
                beta[nonzero_derivative] = (
                    derivative_variance[nonzero_derivative] / derivative[nonzero_derivative]**2
                )
                beta[(~nonzero_derivative) & (derivative_variance > 0.0)] = np.inf
                beta_rows.append(beta)
                derivative_rows.append(derivative)
                derivative_variance_rows.append(derivative_variance)
                covariance_rows.append(covariance)
            beta_matrix = np.vstack(beta_rows)
            derivative_matrix = np.vstack(derivative_rows)
            derivative_variance_matrix = np.vstack(derivative_variance_rows)
            axis_output[int(width)] = {
                "axis": axis,
                "width": int(width),
                "records": records,
                "beta": beta_matrix,
                "derivative": derivative_matrix,
                "derivative_variance": derivative_variance_matrix,
                "covariance_of_entry_means": np.vstack(covariance_rows),
                "summary_finite": _summary_stats(beta_matrix),
                "n_total": int(beta_matrix.size),
                "n_finite": int(np.count_nonzero(np.isfinite(beta_matrix))),
                "n_infinite": int(np.count_nonzero(np.isinf(beta_matrix))),
                "n_undefined_zero_over_zero": int(np.count_nonzero(np.isnan(beta_matrix))),
            }
        output[axis] = axis_output
    return output


def observed_standard_errors(artifact: Artifact) -> dict[str, Any]:
    """Report directly observed entry-mean SEs, without a signal-free floor."""
    variance = artifact.arrays["signal_variance"][:, artifact.subset_indices].astype(
        np.float64, copy=False
    )
    observed_se = np.sqrt(variance / float(artifact.n_ensembles))
    labels = _column_labels(artifact)
    by_column = []
    for column, label in enumerate(labels):
        summary = _summary_stats(observed_se[:, column])
        assert summary is not None
        by_column.append({**label, "summary": summary})
    b_nonzero = np.tile(artifact.subset_b_values > 0.0, len(artifact.subset_pairs))
    summary_b_positive = _summary_stats(observed_se[:, b_nonzero])
    assert summary_b_positive is not None
    return {
        "observed_se": observed_se,
        "by_column": by_column,
        "summary_all_columns": _summary_stats(observed_se),
        "summary_b_gt_0": summary_b_positive,
        "pilot_projection_median_b_gt_0": PILOT_PROJECTION_MEDIAN_SE,
        "ratio_to_pilot_projection_median": (
            summary_b_positive["median"] / PILOT_PROJECTION_MEDIAN_SE
        ),
        "interpretation": (
            "This is the directly observed per-entry mean SE. No nominal 1/sqrt(3ENw) "
            "reference is used as a primary decomposition because its unit-variance cos(phi) "
            "assumption is invalid at low b. The stored v5 arrays cannot separately identify "
            "geometry-realization noise and within-walker three-axis correlation."
        ),
    }


FIXED_12M_CONFIGURATIONS = (
    ("20 x 200,000", 20, 200_000),
    ("40 x 100,000 (current)", 40, 100_000),
    ("60 x 66,667", 60, 66_667),
    ("80 x 50,000", 80, 50_000),
    ("100 x 40,000", 100, 40_000),
)

# These values were requested in the task text, but are a 36-million-axis-walk
# sensitivity budget (3 x the production budget), not a fixed 12-million
# reallocation.  They are retained as a separately labelled table rather than
# silently calling them fixed-cost alternatives.
REQUESTED_36M_SENSITIVITY = (
    ("20 x 600,000", 20, 600_000),
    ("40 x 300,000", 40, 300_000),
    ("60 x 200,000", 60, 200_000),
    ("80 x 150,000", 80, 150_000),
    ("100 x 120,000", 100, 120_000),
)


def _scaled_beta_summary(beta: np.ndarray, factor: float) -> dict[str, Any]:
    scaled = beta * factor
    return {
        "summary_finite": _summary_stats(scaled),
        "n_total": int(scaled.size),
        "n_finite": int(np.count_nonzero(np.isfinite(scaled))),
        "n_infinite": int(np.count_nonzero(np.isinf(scaled))),
        "n_undefined_zero_over_zero": int(np.count_nonzero(np.isnan(scaled))),
    }


def reallocation_beta_bounds(
    beta_results: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Bound rho beta under unidentifiable walker/correlated components.

    One 40 x 100,000 artifact cannot estimate how much endpoint-difference
    variance is independent-walker noise versus an ensemble-correlated term.
    At a new (E, Nw), those limiting terms scale respectively as
    ``(40*100000)/(E*Nw)`` and ``40/E``.  The returned interval spans those
    two extrema; it is intentionally a planning envelope, not a fitted split.
    """
    if 1 not in beta_results.get("rho", {}):
        return {"status": "not_estimable", "reason": "rho k=1 stencil is absent"}
    baseline = beta_results["rho"][1]["beta"]

    def rows_for(configurations: tuple[tuple[str, int, int], ...]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label, n_ensembles, n_walkers in configurations:
            independent_factor = (
                PRODUCTION_N_ENSEMBLES * PRODUCTION_N_WALKERS
                / float(n_ensembles * n_walkers)
            )
            correlated_factor = PRODUCTION_N_ENSEMBLES / float(n_ensembles)
            lower_factor = min(independent_factor, correlated_factor)
            upper_factor = max(independent_factor, correlated_factor)
            rows.append({
                "label": label,
                "n_ensembles": n_ensembles,
                "n_walkers": n_walkers,
                "axis_walks": int(3 * n_ensembles * n_walkers),
                "independent_walker_factor": independent_factor,
                "ensemble_correlated_factor": correlated_factor,
                "lower_factor": lower_factor,
                "upper_factor": upper_factor,
                "beta_lower": _scaled_beta_summary(baseline, lower_factor),
                "beta_upper": _scaled_beta_summary(baseline, upper_factor),
            })
        return rows

    return {
        "status": "available",
        "assumption": (
            "Endpoint-difference variance is an unknown non-negative mixture of an independent-walker "
            "component and an ensemble-correlated component; derivative magnitude and component CRN "
            "correlations are held fixed."
        ),
        "fixed_12m_axis_walks": rows_for(FIXED_12M_CONFIGURATIONS),
        "requested_36m_axis_walk_sensitivity": rows_for(REQUESTED_36M_SENSITIVITY),
        "budget_note": (
            "The requested 20x600,000 through 100x120,000 rows each cost 36,000,000 axis-walks, "
            "not 12,000,000; they are reported only as a three-times-budget sensitivity sweep."
        ),
    }


def kio_path_divergence_result(
    artifact: Artifact,
    correlations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare 19 -> 20 and 20 -> 21 to N_div=T*Delta k_io."""
    t_seconds = float(artifact.build_metadata["T_max_ms"]) / 1000.0
    transition_results = []
    for transition in correlations["k_io"]["transition_summaries"]:
        left = float(transition["left_value"])
        right = float(transition["right_value"])
        transition_results.append({
            "transition_s_inv": [left, right],
            "delta_kio_s_inv": right - left,
            "N_div": t_seconds * (right - left),
            "exp_minus_N_div": math.exp(-t_seconds * (right - left)),
            "n_neighbor_pairs": int(transition["n_neighbor_pairs"]),
            "correlation_summary": transition["summary"],
        })
    aggregate = correlations["k_io"].get("aggregate")
    bootstrap = correlations["k_io"].get("bootstrap")
    support_status = "not_evaluable"
    if bootstrap is not None:
        if bootstrap["ci95_low"] > PILOT_KIO_IQR[1]:
            support_status = "supported"
        elif bootstrap["ci95_high"] <= PILOT_KIO_MEDIAN_CORRELATION:
            support_status = "not_supported"
        else:
            support_status = "inconclusive"
    return {
        "walk_duration_seconds": t_seconds,
        "transitions": transition_results,
        "aggregate": aggregate,
        "aggregate_bootstrap": bootstrap,
        "pilot_reference": {
            "median": PILOT_KIO_MEDIAN_CORRELATION,
            "iqr": list(PILOT_KIO_IQR),
            "pilot_transitions_s_inv": [[0.0, 20.0], [20.0, 130.0]],
        },
        "support_for_path_divergence_prediction": support_status,
        "support_rule": (
            "The production-step result is called support only when its paired-bootstrap 95% lower bound "
            "exceeds the pilot IQR upper endpoint 0.571; it is called not supported when its upper bound is "
            "at or below the pilot median 0.313. Intermediate results are inconclusive. This is not a test that signal-level "
            "r must equal exp(-N_div); exp(-N_div) is a trajectory-survival heuristic."
        ),
    }


def _format_stat(summary: dict[str, Any] | None, key: str = "median", digits: int = 3) -> str:
    if summary is None or key not in summary:
        return "N/A"
    return f"{float(summary[key]):.{digits}g}"


def _format_interval(summary: dict[str, Any] | None, digits: int = 3) -> str:
    if summary is None:
        return "N/A"
    return f"{float(summary['q05']):.{digits}g}–{float(summary['q95']):.{digits}g}"


def _write_correlations_csv(
    output_dir: Path,
    artifact: Artifact,
    correlations: dict[str, dict[str, Any]],
) -> None:
    labels = _column_labels(artifact)
    with (output_dir / "crn_correlations_by_pair_and_column.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "axis", "left_entry", "right_entry", "left_value", "right_value",
            "fixed_coordinate_1", "fixed_coordinate_2", "delta_ms", "Delta_ms",
            "b_s_mm2", "correlation",
        ))
        writer.writeheader()
        for axis in AXES:
            for pair, values in zip(correlations[axis]["pairs"], correlations[axis]["values"]):
                for column, value in enumerate(values):
                    writer.writerow({
                        "axis": axis,
                        "left_entry": pair.left_index,
                        "right_entry": pair.right_index,
                        "left_value": pair.left_value,
                        "right_value": pair.right_value,
                        "fixed_coordinate_1": pair.context[0],
                        "fixed_coordinate_2": pair.context[1],
                        **labels[column],
                        "correlation": "" if not np.isfinite(value) else value,
                    })


def _write_observed_se_csv(output_dir: Path, observed: dict[str, Any]) -> None:
    with (output_dir / "observed_se_by_column.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "delta_ms", "Delta_ms", "b_s_mm2", "n", "min", "q05", "q25", "median",
            "q75", "q95", "max", "mean",
        ))
        writer.writeheader()
        for item in observed["by_column"]:
            writer.writerow({**{key: item[key] for key in ("delta_ms", "Delta_ms", "b_s_mm2")}, **item["summary"]})


def _beta_extremes(
    artifact: Artifact,
    beta_results: dict[str, dict[int, dict[str, Any]]],
    limit: int = 5,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    labels = _column_labels(artifact)
    output: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for axis in AXES:
        axis_output: dict[int, list[dict[str, Any]]] = {}
        for width, result in beta_results[axis].items():
            beta = result["beta"]
            derivative = result["derivative"]
            derivative_variance = result["derivative_variance"]
            finite_locations = np.argwhere(np.isfinite(beta))
            abs_derivatives = np.abs(derivative[np.isfinite(derivative)])
            rows: list[dict[str, Any]] = []
            if finite_locations.size:
                ordering = sorted(
                    finite_locations.tolist(), key=lambda pair: float(beta[pair[0], pair[1]]), reverse=True,
                )[:limit]
                for stencil_row, column in ordering:
                    absolute_derivative = abs(float(derivative[stencil_row, column]))
                    percentile = float(np.mean(abs_derivatives <= absolute_derivative)) if abs_derivatives.size else math.nan
                    rows.append({
                        "stencil_row": int(stencil_row),
                        "column": int(column),
                        "context": list(result["records"][stencil_row]["context"]),
                        **labels[column],
                        "beta": float(beta[stencil_row, column]),
                        "derivative": float(derivative[stencil_row, column]),
                        "derivative_variance": float(derivative_variance[stencil_row, column]),
                        "absolute_derivative_percentile": percentile,
                        "in_lowest_derivative_quartile": bool(percentile <= 0.25),
                    })
            axis_output[int(width)] = rows
        output[axis] = axis_output
    return output


def _write_beta_csv(
    output_dir: Path,
    artifact: Artifact,
    beta_results: dict[str, dict[int, dict[str, Any]]],
) -> None:
    labels = _column_labels(artifact)
    with (output_dir / "beta_by_stencil_and_column.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "axis", "stencil_width_k", "stencil_row", "left_entry", "center_entry", "right_entry",
            "context_1", "context_2", "delta_ms", "Delta_ms", "b_s_mm2", "derivative",
            "variance_derivative", "covariance_of_entry_means", "beta",
        ))
        writer.writeheader()
        for axis in AXES:
            for width, result in beta_results[axis].items():
                for row, record in enumerate(result["records"]):
                    for column, label in enumerate(labels):
                        beta = result["beta"][row, column]
                        writer.writerow({
                            "axis": axis,
                            "stencil_width_k": width,
                            "stencil_row": row,
                            "left_entry": record["left_index"],
                            "center_entry": record["center_index"],
                            "right_entry": record["right_index"],
                            "context_1": record["context"][0],
                            "context_2": record["context"][1],
                            **label,
                            "derivative": result["derivative"][row, column],
                            "variance_derivative": result["derivative_variance"][row, column],
                            "covariance_of_entry_means": result["covariance_of_entry_means"][row, column],
                            "beta": "" if np.isnan(beta) else beta,
                        })


def _write_reallocation_csv(output_dir: Path, reallocation: dict[str, Any]) -> None:
    with (output_dir / "rho_beta_reallocation_bounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sweep", "configuration", "n_ensembles", "n_walkers", "axis_walks",
            "independent_walker_factor", "ensemble_correlated_factor", "lower_factor", "upper_factor",
            "lower_median_finite_beta", "upper_median_finite_beta",
            "lower_q05_finite_beta", "lower_q95_finite_beta",
            "upper_q05_finite_beta", "upper_q95_finite_beta",
            "n_finite", "n_infinite", "n_undefined_zero_over_zero",
        ))
        writer.writeheader()
        if reallocation.get("status") != "available":
            return
        for sweep_name, rows in (
            ("fixed_12m_axis_walks", reallocation["fixed_12m_axis_walks"]),
            ("requested_36m_axis_walk_sensitivity", reallocation["requested_36m_axis_walk_sensitivity"]),
        ):
            for row in rows:
                lower = row["beta_lower"]
                upper = row["beta_upper"]
                lower_summary = lower["summary_finite"]
                upper_summary = upper["summary_finite"]
                writer.writerow({
                    "sweep": sweep_name,
                    "configuration": row["label"],
                    "n_ensembles": row["n_ensembles"],
                    "n_walkers": row["n_walkers"],
                    "axis_walks": row["axis_walks"],
                    "independent_walker_factor": row["independent_walker_factor"],
                    "ensemble_correlated_factor": row["ensemble_correlated_factor"],
                    "lower_factor": row["lower_factor"],
                    "upper_factor": row["upper_factor"],
                    "lower_median_finite_beta": "" if lower_summary is None else lower_summary["median"],
                    "upper_median_finite_beta": "" if upper_summary is None else upper_summary["median"],
                    "lower_q05_finite_beta": "" if lower_summary is None else lower_summary["q05"],
                    "lower_q95_finite_beta": "" if lower_summary is None else lower_summary["q95"],
                    "upper_q05_finite_beta": "" if upper_summary is None else upper_summary["q05"],
                    "upper_q95_finite_beta": "" if upper_summary is None else upper_summary["q95"],
                    "n_finite": lower["n_finite"],
                    "n_infinite": lower["n_infinite"],
                    "n_undefined_zero_over_zero": lower["n_undefined_zero_over_zero"],
                })


def _plot_correlations(output_dir: Path, correlations: dict[str, dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    bins = np.linspace(-1.0, 1.0, 41)
    for axis_plot, axis in zip(axes, AXES):
        values = correlations[axis]["values"].ravel()
        values = values[np.isfinite(values)]
        axis_plot.hist(values, bins=bins, color="#3d7ea6", edgecolor="white")
        summary = correlations[axis]["aggregate"]
        if summary is not None:
            axis_plot.axvline(summary["median"], color="#b34335", linewidth=2, label="median")
            axis_plot.legend(frameon=False, fontsize=8)
        axis_plot.set_title(f"{axis}: {len(correlations[axis]['pairs'])} immediate pairs")
        axis_plot.set_xlabel("paired ensemble correlation r")
        axis_plot.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("diagnostic pair-column count")
    figure.suptitle("Production-grid CRN correlations by declared axis", y=1.02)
    figure.tight_layout()
    path = output_dir / "crn_correlation_histograms.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_observed_se(output_dir: Path, artifact: Artifact, observed: dict[str, Any]) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    n_b = len(artifact.subset_b_values)
    for pair_index, (delta, Delta) in enumerate(artifact.subset_pairs):
        values = observed["observed_se"][:, pair_index * n_b:(pair_index + 1) * n_b]
        median = np.median(values, axis=0)
        nonzero_b = artifact.subset_b_values > 0.0
        axis.semilogy(
            artifact.subset_b_values[nonzero_b], median[nonzero_b], marker="o", markersize=2.5,
            linewidth=1.2, label=f"δ={delta:g}, Δ={Delta:g} ms",
        )
    axis.axhline(
        PILOT_PROJECTION_MEDIAN_SE, color="#555555", linestyle="--", linewidth=1.2,
        label="pilot projected median (b>0)",
    )
    axis.set_xlabel("b (s/mm²)")
    axis.set_ylabel("observed SE of entry mean (log scale)")
    axis.set_title("Directly observed production-probe SE by timing pair")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    path = output_dir / "observed_se_by_b_and_timing.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_beta(output_dir: Path, beta_results: dict[str, dict[int, dict[str, Any]]]) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = {"rho": "#b34335", "V": "#3d7ea6", "k_io": "#4c956c"}
    for axis_name in AXES:
        widths: list[int] = []
        medians: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for width, result in sorted(beta_results[axis_name].items()):
            summary = result["summary_finite"]
            if summary is None or summary["median"] <= 0.0:
                continue
            widths.append(width)
            medians.append(summary["median"])
            lower.append(summary["median"] - summary["q25"])
            upper.append(summary["q75"] - summary["median"])
        if widths:
            axis.errorbar(
                widths, medians, yerr=np.asarray([lower, upper]), marker="o", capsize=3,
                color=colors[axis_name], label=axis_name,
            )
    axis.set_yscale("log")
    axis.set_xlabel("central-stencil half-width k")
    axis.set_ylabel("finite beta: median and IQR")
    axis.set_title("Derivative noise-to-signal ratio by declared stencil")
    axis.set_xticks([1, 2, 3, 4])
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    figure.tight_layout()
    path = output_dir / "beta_vs_stencil_width.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_reallocation_table(output_dir: Path, reallocation: dict[str, Any]) -> Path:
    figure, axis = plt.subplots(figsize=(10, 3.4))
    axis.axis("off")
    rows = reallocation.get("fixed_12m_axis_walks", [])
    cell_text = []
    for row in rows:
        lower = row["beta_lower"]["summary_finite"]
        upper = row["beta_upper"]["summary_finite"]
        cell_text.append([
            row["label"],
            f"{row['axis_walks']:,}",
            f"{row['lower_factor']:.3g}–{row['upper_factor']:.3g}",
            f"{_format_stat(lower)}–{_format_stat(upper)}",
            f"[{_format_interval(lower)}] to [{_format_interval(upper)}]",
        ])
    table = axis.table(
        cellText=cell_text,
        colLabels=(
            "configuration", "axis-walks", "variance-scale envelope",
            "finite rho beta median envelope", "finite rho beta P05–P95 envelope",
        ),
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    axis.set_title("Fixed 12-million-axis-walk rho-beta bounds (not a fitted decomposition)", pad=12)
    figure.tight_layout()
    path = output_dir / "ensemble_walker_reallocation_table.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _correlation_summary_for_json(correlations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        axis: {
            "status": result["status"],
            "reason": result["reason"],
            "n_immediate_pairs": len(result["pairs"]),
            "aggregate": result["aggregate"],
            "undefined_values": result["undefined_values"],
            "transition_summaries": result["transition_summaries"],
            "bootstrap": result["bootstrap"],
            "fixed_coordinate_block_bootstrap": result["fixed_coordinate_block_bootstrap"],
        }
        for axis, result in correlations.items()
    }


def _beta_summary_for_json(beta_results: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    return {
        axis: {
            str(width): {
                key: result[key]
                for key in (
                    "axis", "width", "summary_finite", "n_total", "n_finite", "n_infinite",
                    "n_undefined_zero_over_zero",
                )
            }
            for width, result in widths.items()
        }
        for axis, widths in beta_results.items()
    }


def _relative_link(target: Path, report_path: Path) -> str:
    return os.path.relpath(target, start=report_path.parent).replace(os.sep, "/")


def _verdict(kio_result: dict[str, Any]) -> tuple[str, str]:
    status = kio_result["support_for_path_divergence_prediction"]
    if status == "supported":
        return (
            "GO — retain 40 ensembles × 100,000 walkers",
            "The production-step k_io result is decisively above the pilot correlation range, while the "
            "single-configuration probe does not identify evidence strong enough to justify an allocation change.",
        )
    if status == "not_supported":
        return (
            "INVESTIGATE — retain the current configuration but do not launch production yet",
            "The 1 s^-1 k_io result does not rise above the pilot weak-correlation result, contradicting the "
            "path-divergence prediction and requiring a seed/dynamics investigation before production spend.",
        )
    return (
        "INVESTIGATE — retain the current configuration pending review",
        "The production-step k_io result is not decisively separated from the pilot correlation range. A "
        "reallocation is not supported by one configuration with an unidentifiable noise decomposition.",
    )


def _beta_cell(result: dict[str, Any] | None) -> str:
    if result is None:
        return "N/A"
    summary = result["summary_finite"]
    if summary is None:
        return f"no finite values ({result['n_infinite']} ∞, {result['n_undefined_zero_over_zero']} undefined)"
    return (
        f"{summary['median']:.3g} [{summary['q25']:.3g}, {summary['q75']:.3g}] "
        f"(finite {result['n_finite']}/{result['n_total']}; ∞ {result['n_infinite']}; "
        f"undefined {result['n_undefined_zero_over_zero']})"
    )


def _write_report(
    report_path: Path,
    output_dir: Path,
    artifact: Artifact,
    definition: ProbeDefinition,
    topology_provenance: dict[str, Any],
    correlations: dict[str, dict[str, Any]],
    kio_result: dict[str, Any],
    observed: dict[str, Any],
    beta_results: dict[str, dict[int, dict[str, Any]]],
    beta_extremes: dict[str, dict[int, list[dict[str, Any]]]],
    reallocation: dict[str, Any],
    figures: dict[str, Path],
) -> None:
    verdict, verdict_reason = _verdict(kio_result)
    i0, j0 = definition.center
    rho_log10_step = definition.rho_step / math.log(10.0)
    V_log10_step = definition.V_step / math.log(10.0)
    lines: list[str] = []
    lines.extend([
        "# v5 production-grid stencil probe",
        "",
        "## Verdict",
        "",
        f"**{verdict}.** {verdict_reason}",
        "",
        "This report was generated from the restricted production-Monte-Carlo cross. It neither launches nor modifies the production build, W4, simulator, geometry code, library schema, or fitters.",
        "",
        "## Probe geometry and contract checks",
        "",
        f"The declaration contains {len(definition.pairs)} retained `(rho, V)` pairs crossed with `{list(definition.kio_values)}` s^-1: exactly {definition.expected_cellular_entries} cellular entries and no free-water atom.",
        "",
        f"Its centre is canonical indices `({i0}, {j0})`, `rho={definition.rhos[i0]:.9g}` cells/uL, `V={definition.Vs[j0]:.9g}` pL, and `v_i={definition.rhos[i0] * definition.Vs[j0] * 1e-6:.9g}`. The rho line supports `k=1,2,3,4`; the V line supports `k=1,2`; and `k_io={19,20,21}` supports its linear `k=1` central difference.",
        "",
        f"The canonical code-derived spacings are `ln(rho[i+1]/rho[i])={definition.rho_step:.12g}` (`{rho_log10_step:.12g}` decades) and `ln(V[j+1]/V[j])={definition.V_step:.12g}` (`{V_log10_step:.12g}` decades). `madi/config.py` supplies the timing/physics defaults, while the rho/V production generator is `madi.library.make_remediation_log_grid()`. These values differ slightly from the prompt's quoted decade values `0.047648` and `0.068259`; the canonical 64-node `geomspace` formulas were used rather than silently rounding either value.",
        "",
        f"All v5 checks passed before correlations were calculated: metadata records the ensemble-index CRN contract, the 40 x 100,000 production settings and full storage grid, variance reconstruction had maximum absolute error `{artifact.variance_check['max_abs_error']:.3g}`, and the subset mean reconstructed the main signal with maximum absolute error `{artifact.mean_check['max_abs_error']:.3g}`.",
        "",
        f"All {topology_provenance['geometry_reuse']['groups_with_multiple_kio']} fixed-nominal `(rho,V)` k_io groups have identical `per_ensemble_geometry` metadata. Stored `rhos`/`Vs` are retained as realized finite-geometry provenance, not used as literal adjacency keys: their realized/nominal ratio summaries are rho median `{topology_provenance['realized_labels']['rho_realized_over_nominal']['median']:.6g}` and V median `{topology_provenance['realized_labels']['V_realized_over_nominal']['median']:.6g}`.",
        "",
        "## CRN correlations at production spacing",
        "",
        "Every listed pair preserves the shared ensemble index. The bootstrap resamples those aligned indices across all pair-column values, so it describes finite-ensemble uncertainty conditional on this one declared cross; it does not treat the 200 columns as 200 independent tissue realizations.",
        "",
        "| Axis | Immediate pairs | Median r | IQR | Paired-ensemble bootstrap 95% CI | Undefined 0/0 values |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for axis in AXES:
        result = correlations[axis]
        summary = result["aggregate"]
        bootstrap = result["bootstrap"]
        iqr = "N/A" if summary is None else f"{summary['q25']:.3f} to {summary['q75']:.3f}"
        ci = "N/A" if bootstrap is None else f"{bootstrap['ci95_low']:.3f} to {bootstrap['ci95_high']:.3f}"
        median = "N/A" if summary is None else f"{summary['median']:.3f}"
        lines.append(
            f"| `{axis}` | {len(result['pairs'])} | {median} | {iqr} | {ci} | {result['undefined_values']} |"
        )
    lines.extend(["", f"![Per-axis CRN correlation histograms.]({_relative_link(figures['correlations'], report_path)})", ""])
    lines.extend([
        "For the 1 s^-1 `k_io` transitions, the path-divergence heuristic gives `N_div=T Δk_io=0.128` and trajectory survival `exp(-N_div)=0.880`. The result is evaluated against the pilot aggregate median 0.313 and pilot IQR upper endpoint 0.571; it is not a claim that the signal correlation itself must equal 0.880.",
        "",
        "| Transition (s^-1) | Neighbor pairs | N_div | exp(-N_div) | Median r | IQR |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for transition in kio_result["transitions"]:
        summary = transition["correlation_summary"]
        lines.append(
            f"| {transition['transition_s_inv'][0]:g} → {transition['transition_s_inv'][1]:g} | "
            f"{transition['n_neighbor_pairs']} | {transition['N_div']:.3f} | {transition['exp_minus_N_div']:.3f} | "
            f"{_format_stat(summary, 'median')} | {_format_stat(summary, 'q25')} to {_format_stat(summary, 'q75')} |"
        )
    lines.extend([
        "",
        f"The path-divergence prediction status is **{kio_result['support_for_path_divergence_prediction']}** under the predeclared bootstrap rule: {kio_result['support_rule']}",
        "",
        "## Directly observed noise",
        "",
        "The primary noise result is the directly observed per-entry SE `sqrt(signal_variance / 40)`, shown by timing and b below. A signal-independent `1/sqrt(3 E Nw)` line is intentionally not used as a decomposition reference: it assumes `Var(cos phi)=1`, which fails at low b.",
        "",
        f"Across all cellular entry-columns with `b>0`, observed SE has median `{observed['summary_b_gt_0']['median']:.3g}`, IQR `{observed['summary_b_gt_0']['q25']:.3g}` to `{observed['summary_b_gt_0']['q75']:.3g}`, and is `{observed['ratio_to_pilot_projection_median']:.3g}×` the pilot's projected median `{PILOT_PROJECTION_MEDIAN_SE:.3g}`. This comparison is descriptive, not a clean geometry/walker split.",
        "",
        f"![Observed production-probe SE by b and timing.]({_relative_link(figures['observed_se'], report_path)})",
        "",
        "The v5 stored arrays do not contain per-walker second moments or independently re-used geometries, so they cannot separate geometry-realization noise from within-walker three-axis correlation. The post-production W4 independent-seed replicate is the experiment that would distinguish those causes; it is not launched or assumed here.",
        "",
        "## Derivative magnitudes and beta",
        "",
        "For each declared central stencil, `J_hat=(S(+k)-S(-k))/(2 k h1)` and `Var(J_hat)=[Var(S+)+Var(S-)-2 Cov(S+,S-)]/(2 k h1)^2`, with endpoint covariance calculated from matched ensemble indices and divided by 40 for covariance of entry means. `h1` is the exact canonical natural-log step for rho/V and 1 s^-1 for k_io. `beta=Var(J_hat)/J_hat^2` is retained as undefined for deterministic `0/0` columns and infinite if a zero derivative retains variance; no columns are filtered away.",
        "",
        "| Axis | k=1 | k=2 | k=3 | k=4 |",
        "|---|---|---|---|---|",
    ])
    for axis in AXES:
        lines.append(
            "| `" + axis + "` | " + " | ".join(
                _beta_cell(beta_results[axis].get(width)) for width in range(1, 5)
            ) + " |"
        )
    lines.extend([
        "",
        "Fisher information is quadratic in the derivative: `E[J_hat^2]=J^2+Var(J_hat)`. Thus leaving this uncorrected inflates a Fisher diagonal by `1+beta` and deflates the corresponding CRLB by `sqrt(1+beta)`.",
        "",
        f"![Finite beta distributions versus stencil width.]({_relative_link(figures['beta'], report_path)})",
        "",
        "The largest finite beta values and their absolute-derivative percentile are listed below. A low derivative percentile means the column is weakly informative for that axis under comparable noise; this report cannot know a downstream Fisher weighting or exclusion policy, so it does not claim which columns a fitter will ultimately use.",
        "",
        "| Axis / k | δ, Δ, b | beta | |J| percentile | Lowest derivative quartile? |",
        "|---|---|---:|---:|---|",
    ])
    for axis in AXES:
        for width, rows in sorted(beta_extremes[axis].items()):
            for row in rows[:3]:
                lines.append(
                    f"| `{axis}` / {width} | ({row['delta_ms']:g}, {row['Delta_ms']:g}, {row['b_s_mm2']:g}) | "
                    f"{row['beta']:.3g} | {row['absolute_derivative_percentile']:.3f} | "
                    f"{'yes' if row['in_lowest_derivative_quartile'] else 'no'} |"
                )
    lines.extend([
        "",
        "## Fixed-cost ensemble/walker reallocation",
        "",
        "A single 40 × 100,000 configuration cannot identify the independent-walker versus ensemble-correlated share of the paired endpoint-difference variance. At a new configuration, those extrema scale as `(40×100,000)/(E×Nw)` and `40/E`, respectively. The table therefore gives a range, not a fitted point prediction. At fixed total cost, the independent extreme is invariant and the correlated extreme improves monotonically with more ensembles; no defensible interior optimum can be identified from this artifact alone.",
        "",
        "| Configuration | Axis-walks | Variance-scale envelope | Finite rho beta median envelope | Finite rho beta P05–P95 envelope |",
        "|---|---:|---:|---:|---:|",
    ])
    if reallocation.get("status") == "available":
        for row in reallocation["fixed_12m_axis_walks"]:
            lower = row["beta_lower"]["summary_finite"]
            upper = row["beta_upper"]["summary_finite"]
            lines.append(
                f"| {row['label']} | {row['axis_walks']:,} | {row['lower_factor']:.3g}–{row['upper_factor']:.3g} | "
                f"{_format_stat(lower)}–{_format_stat(upper)} | "
                f"[{_format_interval(lower)}] to [{_format_interval(upper)}] |"
            )
    lines.extend([
        "",
        f"![Fixed-cost rho-beta reallocation bounds.]({_relative_link(figures['reallocation'], report_path)})",
        "",
        "The literal 20 × 600,000 through 100 × 120,000 request is not a fixed 12-million-axis-walk allocation: every row costs 36 million axis-walks. It is retained in `rho_beta_reallocation_bounds.csv` as a separately labelled three-times-budget sensitivity sweep, not used to recommend a production configuration.",
        "",
        "No `GO-WITH-CHANGE` recommendation is made from a one-configuration envelope. A configuration change would require a new validation cycle and an identifiable component decomposition; absent that evidence, the current 40 × 100,000 specification is retained.",
        "",
        "## Limitations",
        "",
        "This is one deliberately chosen cross, not a measurement over the full masked grid. The CRN bootstrap is conditional on its 40 ensembles and correlated diagnostic columns. Beta is strongly structured and can diverge wherever the finite-difference derivative approaches zero. Reallocation bounds assume stationary derivative magnitude and component correlations while changing E/Nw; they are planning bounds, not a precision forecast. W4 remains out of scope and is required to separate geometry-realization noise from within-walker axis correlation.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python analysis/v5_stencil_probe.py \\",
        "  libraries/madi_v5_stencil_probe.shard*.npz \\",
        "  --declaration data/madi_v5_stencil_probe_entry_subset.json \\",
        f"  --expected-shards {len(artifact.paths)} \\",
        "  --output-dir docs/figures/v5_stencil_probe \\",
        "  --report docs/v5_stencil_probe.md",
        "```",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", help="four completed probe shards, glob, or containing directory")
    parser.add_argument(
        "--declaration", default="data/madi_v5_stencil_probe_entry_subset.json",
        help="declared production-grid stencil cross",
    )
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--output-dir", default="docs/figures/v5_stencil_probe")
    parser.add_argument("--report", default="docs/v5_stencil_probe.md")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 200:
        parser.error("--bootstrap-replicates must be at least 200")
    try:
        definition = load_probe_definition(args.declaration)
        artifact = load_artifact(_expand_paths(args.artifact), args.expected_shards)
        entry_indices, topology_provenance = validate_probe_artifact(artifact, definition)
        neighbors, stencils = declared_topology(definition, entry_indices)
        correlations = compute_probe_correlations(artifact, neighbors, args.bootstrap_replicates)
        kio_result = kio_path_divergence_result(artifact, correlations)
        observed = observed_standard_errors(artifact)
        beta_results = compute_beta(artifact, stencils)
        beta_extremes = _beta_extremes(artifact, beta_results)
        reallocation = reallocation_beta_bounds(beta_results)
    except (DiagnosticError, ValueError) as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    _write_correlations_csv(output_dir, artifact, correlations)
    _write_observed_se_csv(output_dir, observed)
    _write_beta_csv(output_dir, artifact, beta_results)
    _write_reallocation_csv(output_dir, reallocation)
    figures = {
        "correlations": _plot_correlations(output_dir, correlations),
        "observed_se": _plot_observed_se(output_dir, artifact, observed),
        "beta": _plot_beta(output_dir, beta_results),
        "reallocation": _plot_reallocation_table(output_dir, reallocation),
    }
    _write_report(
        report_path, output_dir, artifact, definition, topology_provenance, correlations, kio_result,
        observed, beta_results, beta_extremes, reallocation, figures,
    )
    summary = {
        "artifact_paths": [str(path) for path in artifact.paths],
        "declaration": str(definition.path),
        "probe": {
            "center_indices": list(definition.center),
            "cellular_pairs": len(definition.pairs),
            "cellular_entries": definition.expected_cellular_entries,
            "kio_values_s_inv": list(definition.kio_values),
            "rho_log_step_natural": definition.rho_step,
            "V_log_step_natural": definition.V_step,
        },
        "contract_checks": {
            "variance": artifact.variance_check,
            "subset_mean": artifact.mean_check,
            **topology_provenance,
        },
        "correlations": _correlation_summary_for_json(correlations),
        "kio_path_divergence": kio_result,
        "observed_standard_errors": {
            key: value for key, value in observed.items() if key != "observed_se" and key != "by_column"
        },
        "beta": _beta_summary_for_json(beta_results),
        "largest_finite_beta": beta_extremes,
        "reallocation": reallocation,
        "figures": {name: str(path) for name, path in figures.items()},
        "report": str(report_path),
    }
    with (output_dir / "v5_stencil_probe_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    print(json.dumps({
        "status": "complete",
        "report": str(report_path),
        "output_dir": str(output_dir),
        "verdict": _verdict(kio_result)[0],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

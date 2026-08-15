#!/usr/bin/env python3
"""Summarize the v5 stencil-probe beta gate and runtime benchmark JSON files.

This script is deliberately read-only.  It turns the retained stencil-probe
per-column table and the diagnostic-only runtime benchmark records into a
machine-readable summary for ``docs/v5_runtime_regression.md`` once the Sol
benchmark has returned.  More than one result directory is accepted so a
successful C1/C3 task can be retained when a different array task needs a
diagnostic-only rerun.  It does not create or inspect a library artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BETA_CSV = Path("docs/figures/v5_stencil_probe/beta_by_stencil_and_column.csv")
TOP_FRACTION = 0.05


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "median": None, "q25": None, "q75": None, "p05": None, "p95": None, "max": None}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def beta_summary(path: Path) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[tuple[float, float]]] = {}
    counts: dict[tuple[str, int], dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["axis"]), int(row["stencil_width_k"]))
            count = counts.setdefault(key, {"total": 0, "undefined": 0, "infinite": 0, "finite": 0})
            count["total"] += 1
            try:
                derivative = float(row["derivative"])
                beta = float(row["beta"])
            except (TypeError, ValueError):
                count["undefined"] += 1
                continue
            if not math.isfinite(beta):
                count["infinite"] += 1
                continue
            if not math.isfinite(derivative):
                count["undefined"] += 1
                continue
            count["finite"] += 1
            groups.setdefault(key, []).append((abs(derivative), beta))

    output: dict[str, Any] = {}
    load_bearing_maxima: list[float] = []
    for (axis, width), rows in sorted(groups.items()):
        rows.sort(key=lambda item: item[0], reverse=True)
        n_top = max(1, math.ceil(len(rows) * TOP_FRACTION))
        all_beta = np.asarray([item[1] for item in rows], dtype=np.float64)
        top_beta = np.asarray([item[1] for item in rows[:n_top]], dtype=np.float64)
        item = {
            "axis": axis,
            "half_width_k": width,
            "all_finite_beta": _quantiles(all_beta),
            "top_abs_derivative_fraction": TOP_FRACTION,
            "top_abs_derivative_beta": _quantiles(top_beta),
            "counts": counts[(axis, width)],
            "conservative_doubled_top_abs_derivative_beta_max": float(2.0 * np.max(top_beta)),
        }
        load_bearing_maxima.append(item["conservative_doubled_top_abs_derivative_beta_max"])
        output[f"{axis}_k{width}"] = item

    # The task's decision rule is intentionally conservative: it assumes all
    # paired derivative variance scales as 1/Nw even though the probe cannot
    # separate geometry/axis-correlated variance from independent walker noise.
    return {
        "source": str(path),
        "top_abs_derivative_fraction": TOP_FRACTION,
        "per_axis_stencil": output,
        "walker_halving_assessment": {
            "assumption": "all derivative variance doubles when walkers per ensemble halve",
            "maximum_doubled_beta_in_top_5pct_abs_derivative_columns": float(max(load_bearing_maxima)),
            "go_threshold": 0.01,
            "go": bool(max(load_bearing_maxima) <= 0.01),
            "interpretation": (
                "GO WITH CHANGED-CONFIGURATION VALIDATION when <=0.01; this is a Fisher/CRLB "
                "allocation decision, not permission to change production without the validation run."
            ),
        },
    }


def _load_benchmarks(directories: list[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    origins: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("v5_runtime_benchmark_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != "madi-v5-runtime-benchmark-v1":
                raise ValueError(f"unexpected benchmark schema in {path}: {data.get('schema')!r}")
            case = data.get("case")
            if not isinstance(case, str):
                raise ValueError(f"benchmark {path} has no case name")
            if case in output:
                raise ValueError(
                    f"duplicate benchmark case {case!r}: {origins[case]} and {path}; "
                    "choose one deliberately rather than silently mixing runs"
                )
            output[case] = data
            origins[case] = path
    return output


def runtime_summary(directories: list[Path]) -> dict[str, Any]:
    benchmarks = _load_benchmarks(directories)
    required = {"center-c1", "center-c3", "rho-low", "rho-high"}
    missing = sorted(required.difference(benchmarks))
    if missing:
        return {
            "status": "incomplete",
            "directories": [str(path) for path in directories],
            "missing_cases": missing,
            "available_cases": sorted(benchmarks),
        }

    c1 = benchmarks["center-c1"]
    c3 = benchmarks["center-c3"]
    if c1.get("gpu") != c3.get("gpu"):
        raise RuntimeError(
            "C1 and C3 used different GPU names; do not apply the geometry/walker decomposition across them"
        )
    C1 = float(c1["group_timing"]["C_seconds_excluding_jit"])
    C3 = float(c3["group_timing"]["C_seconds_excluding_jit"])
    W = (C3 - C1) / 2.0
    G = (3.0 * C1 - C3) / 2.0
    if W <= 0.0 or G < 0.0:
        raise RuntimeError(
            f"invalid C1/C3 decomposition: C1={C1:.3f}, C3={C3:.3f}, G={G:.3f}, W={W:.3f}"
        )

    endpoints: dict[str, Any] = {}
    for case in ("rho-low", "rho-high"):
        benchmark = benchmarks[case]
        timing = benchmark["group_timing"]
        g_one = float(timing["geometry_seconds"])
        w_one = float(timing["walk_and_reduction_seconds"])
        projected_51 = 40.0 * (g_one + 51.0 * w_one)
        phase = benchmark.get("phase_profile")
        endpoints[case] = {
            "coordinate": benchmark["coordinate"],
            "one_ensemble_geometry_seconds": g_one,
            "one_ensemble_one_kio_walk_and_reduction_seconds": w_one,
            "projected_40_ensemble_51_kio_group_seconds": projected_51,
            "projected_40_ensemble_51_kio_group_hours": projected_51 / 3600.0,
            "phase_profile": phase,
        }

    return {
        "status": "complete",
        "directories": [str(path) for path in directories],
        "center_gpu": c1["gpu"],
        "C1_seconds": C1,
        "C3_seconds": C3,
        "G_seconds": G,
        "W_seconds_per_kio_family": W,
        "geometry_fraction_C1": G / C1,
        "geometry_fraction_C3": G / C3,
        "center_projected_51_kio_group_seconds": G + 51.0 * W,
        "center_projected_51_kio_group_hours": (G + 51.0 * W) / 3600.0,
        "endpoint_extrapolations": endpoints,
        "caveat": (
            "Each endpoint uses one independently built ensemble.  Its 40-ensemble projection assumes "
            "geometry and walker costs are stationary across ensemble index; it is a planning estimate, not a replacement for a full endpoint group."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta-csv", type=Path, default=DEFAULT_BETA_CSV)
    parser.add_argument(
        "--benchmark-dir", type=Path, nargs="+", required=True,
        help="one or more benchmark output directories; each case may appear exactly once",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.beta_csv.is_file():
        raise FileNotFoundError(args.beta_csv)
    for directory in args.benchmark_dir:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    payload = {
        "schema": "madi-v5-runtime-regression-summary-v1",
        "beta": beta_summary(args.beta_csv),
        "runtime": runtime_summary(args.benchmark_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

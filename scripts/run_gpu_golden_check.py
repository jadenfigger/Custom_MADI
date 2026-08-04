#!/usr/bin/env python3
"""Run the CUDA half of the committed CPU/GPU full-facet golden check.

The exit status is the result: zero means the CUDA kernel reproduced the
committed CPU reference within the stated float64 tolerance.  Nonzero writes
``gpu_golden_diff.json`` explaining the mismatch, for a single unattended Sol
job rather than an interactive GPU debugging session.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from madi.config import SimConfig
from madi.ensemble import Ensemble, GeometryStats, PopulationCertificate
from madi.walker_gpu import HAS_CUDA, WalkRandomStream, run_walk_Y


DEFAULT_GOLDEN = Path("tests/physics_audit/data/cpu_gpu_golden_v1.npz")
RTOL = 5e-11
ATOL = 5e-12


def _max_abs_rel(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    delta = np.abs(actual - expected)
    denom = np.maximum(np.abs(expected), ATOL)
    return {
        "shape": list(actual.shape),
        "max_abs": float(np.max(delta)) if delta.size else 0.0,
        "max_rel": float(np.max(delta / denom)) if delta.size else 0.0,
        "allclose": bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)),
    }


def _read_hash(path: Path) -> str | None:
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if not hash_path.exists():
        return None
    return hash_path.read_text(encoding="utf-8").split()[0]


def _ensemble_from_golden(data, cfg: SimConfig, case: dict) -> Ensemble:
    geometry_values = {item.name: case["geometry"][item.name] for item in fields(GeometryStats)}
    geometry_values["population"] = PopulationCertificate(**geometry_values["population"])
    geometry = GeometryStats(**geometry_values)
    return Ensemble(
        seeds=np.asarray(data["seeds"], dtype=np.float64),
        annulus=np.asarray(data["annulus"], dtype=np.float64),
        rho=float(case["rho_realised_per_uL"]),
        V=float(case["V_realised_pL"]),
        vi=float(case["vi_realised"]),
        alpha_star=float(case["alpha_star_um"]),
        L=float(cfg.L),
        source_lo=float(case["source_lo_um"]),
        source_hi=float(case["source_hi_um"]),
        mean_AV=float(case["mean_A_over_V_um_inv"]),
        geometry=geometry,
        rho_requested=float(case["rho_requested_per_uL"]),
        V_requested=float(case["V_requested_pL"]),
        kd_node_seed=np.asarray(data["kd_node_seed"], dtype=np.int32),
        kd_node_axis=np.asarray(data["kd_node_axis"], dtype=np.int8),
        kd_node_left=np.asarray(data["kd_node_left"], dtype=np.int32),
        kd_node_right=np.asarray(data["kd_node_right"], dtype=np.int32),
        kd_node_parent=np.asarray(data["kd_node_parent"], dtype=np.int32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not HAS_CUDA:
        print("FAIL: CUDA is unavailable; this must run in the Sol GPU job.", file=sys.stderr)
        return 2
    if not args.golden.exists():
        print(f"FAIL: golden file is missing: {args.golden}", file=sys.stderr)
        return 2
    expected_hash = _read_hash(args.golden)
    actual_hash = hashlib.sha256(args.golden.read_bytes()).hexdigest()
    if expected_hash is None or actual_hash != expected_hash:
        print("FAIL: committed CPU golden hash is missing or does not match.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.golden, allow_pickle=False)
    cfg = SimConfig(**json.loads(str(data["config_json"])))
    cfg.assert_grid_alignment()
    case = json.loads(str(data["case_json"]))
    ensemble = _ensemble_from_golden(data, cfg, case)
    stream = WalkRandomStream(
        initial_positions=np.asarray(data["initial_positions"], dtype=np.float64),
        increments=np.asarray(data["increments"], dtype=np.float64),
        acceptance_uniforms=np.asarray(data["acceptance_uniforms"], dtype=np.float64),
    )
    Y, n_escaped, telemetry = run_walk_Y(
        ensemble, 0.0, cfg, pp=float(case["pp"]), seed=0, verbose=False,
        return_telemetry=True, classifier="exact", random_stream=stream,
        use_gpu=True,
    )
    comparisons = {
        "Y": _max_abs_rel(Y, data["Y_cpu"]),
        "occupancy_fraction": _max_abs_rel(
            telemetry["occupancy_fraction"], data["occupancy_fraction_cpu"]
        ),
        "escaped_equal": bool(int(n_escaped) == int(data["escaped_cpu"])),
    }
    passed = bool(comparisons["escaped_equal"] and all(
        item["allclose"] for key, item in comparisons.items() if key != "escaped_equal"
    ))
    report = {
        "schema": "madi-cpu-gpu-golden-report-v3-full-facet",
        "golden": str(args.golden),
        "golden_sha256": actual_hash,
        "rtol": RTOL,
        "atol": ATOL,
        "cuda_available": bool(HAS_CUDA),
        "pass": passed,
        "n_escaped_gpu": int(n_escaped),
        "comparisons": comparisons,
        "case": case,
    }
    report_path = args.output_dir / "gpu_golden_diff.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "gpu_golden_result.npz",
        Y_gpu=Y,
        escaped_gpu=np.array(n_escaped, dtype=np.int64),
        occupancy_fraction_gpu=np.asarray(telemetry["occupancy_fraction"], dtype=np.float64),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

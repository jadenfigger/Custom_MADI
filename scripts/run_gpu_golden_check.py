#!/usr/bin/env python3
"""Run the CUDA half of the committed CPU/GPU MADI golden-file check.

Exit status is the result: zero means the CUDA walk reproduced the committed
CPU reference within the stated float64 tolerance; nonzero means the written
``gpu_golden_diff.json`` explains the mismatch.  This is designed for a
single unattended Sol sbatch invocation, not an interactive debugging loop.
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
from madi.ensemble import Ensemble, GeometryStats
from madi.walker_gpu import HAS_CUDA, WalkRandomStream, run_walk_Y


DEFAULT_GOLDEN = Path("tests/physics_audit/data/cpu_gpu_golden_v1.npz")
RTOL = 5e-11
ATOL = 5e-12


def _max_abs_rel(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    delta = np.abs(a - e)
    denom = np.maximum(np.abs(e), ATOL)
    return {
        "shape": list(a.shape),
        "max_abs": float(np.max(delta)) if delta.size else 0.0,
        "max_rel": float(np.max(delta / denom)) if delta.size else 0.0,
        "allclose": bool(np.allclose(a, e, rtol=RTOL, atol=ATOL, equal_nan=True)),
    }


def _read_hash(path: Path) -> str | None:
    sha_path = path.with_suffix(path.suffix + ".sha256")
    if not sha_path.exists():
        return None
    return sha_path.read_text(encoding="utf-8").split()[0]


def _ensemble_from_golden(data, cfg: SimConfig, case: dict) -> Ensemble:
    geometry_keys = {item.name for item in fields(GeometryStats)}
    geometry = GeometryStats(**{key: case["geometry"][key] for key in geometry_keys})
    return Ensemble(
        seeds=np.asarray(data["seeds"], dtype=np.float64),
        annulus=np.asarray(data["annulus"], dtype=np.float64),
        grid_candidates=np.asarray(data["grid_candidates"], dtype=np.int32),
        rho=float(case["rho_realised_per_uL"]),
        V=float(case["V_realised_pL"]),
        vi=float(case["vi_realised"]),
        alpha_star=float(case["alpha_star_um"]),
        L=float(cfg.L),
        mean_AV=float(case["mean_A_over_V_um_inv"]),
        grid_spacing=float(cfg.grid_spacing),
        classifier_candidates=int(cfg.classifier_candidates),
        geometry=geometry,
        rho_requested=float(case["rho_requested_per_uL"]),
        V_requested=float(case["V_requested_pL"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not HAS_CUDA:
        print("FAIL: CUDA is unavailable; this must run in the Sol GPU job.", file=sys.stderr)
        return 2
    golden = args.golden
    if not golden.exists():
        print(f"FAIL: golden file is missing: {golden}", file=sys.stderr)
        return 2
    expected_hash = _read_hash(golden)
    actual_hash = hashlib.sha256(golden.read_bytes()).hexdigest()
    if expected_hash is None or actual_hash != expected_hash:
        print("FAIL: committed CPU golden hash is missing or does not match.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(golden, allow_pickle=False)
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
        return_telemetry=True, classifier="cache", random_stream=stream,
        use_gpu=True,
    )
    comparisons = {
        "Y": _max_abs_rel(Y, data["Y_cpu"]),
        "metrics": _max_abs_rel(telemetry["metrics"], data["metrics_cpu"]),
        "occupancy_fraction": _max_abs_rel(
            telemetry["occupancy_fraction"], data["occupancy_fraction_cpu"]
        ),
        "start_survivor_fraction": _max_abs_rel(
            telemetry["start_survivor_fraction"], data["start_survivor_fraction_cpu"]
        ),
        "escaped_equal": bool(int(n_escaped) == int(data["escaped_cpu"])),
    }
    passed = bool(comparisons["escaped_equal"] and all(
        item["allclose"] for name, item in comparisons.items() if name != "escaped_equal"
    ))
    report = {
        "schema": "madi-cpu-gpu-golden-report-v1",
        "golden": str(golden),
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
        metrics_gpu=np.asarray(telemetry["metrics"], dtype=np.float64),
        occupancy_fraction_gpu=np.asarray(telemetry["occupancy_fraction"], dtype=np.float64),
        start_survivor_fraction_gpu=np.asarray(telemetry["start_survivor_fraction"], dtype=np.float64),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

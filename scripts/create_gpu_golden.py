#!/usr/bin/env python3
"""Create the fixed CPU reference artifact for the Tier-C GPU golden test.

The artifact contains the *input random stream* as well as the expected CPU
trajectory telemetry.  CUDA is therefore checked against exactly the same
positions, Gaussian proposals and acceptance uniforms—not against a merely
similar stochastic run.

This script deliberately requires ``--overwrite`` to refresh the committed
reference.  Updating a golden is a physics-affecting change and should be
visible in review.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from madi.config import SimConfig
from madi.ensemble import create_ensemble
from madi.walker_gpu import make_walk_random_stream, run_walk_Y


DEFAULT_OUTPUT = Path("tests/physics_audit/data/cpu_gpu_golden_v1.npz")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def _case_config() -> SimConfig:
    """Small but nontrivial periodic/high-packing crossing case."""
    return SimConfig(
        L=48.0,
        grid_spacing=0.75,
        classifier_candidates=8,
        geometry_calibration_points=20_000,
        geometry_validation_points=20_000,
        geometry_sample_cells=20,
        geometry_vi_tolerance=0.005,
        n_walkers=64,
        n_ensembles=1,
        T_max_ms=4.0,
        small_deltas=[1.0],
        big_deltas=[2.0, 3.0],
        b_values=[0.0, 500.0, 1_000.0, 2_000.0],
        exchange_calibration_walkers=64,
        exchange_calibration_ms=4.0,
    )


def _config_metadata(cfg: SimConfig) -> dict:
    # ``asdict`` is stable and includes every physics-relevant default in
    # this deliberately fixed test configuration.
    return asdict(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output: Path = args.output
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace the CPU golden")
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = _case_config()
    # vi = 1e6 * 0.9 * 1e-6 = 0.9: deliberately exercises thin annuli.
    rho, V, pp = 1.0e6, 0.9, 0.02
    geometry_seed, stream_seed = 20_260_811, 20_260_812
    ensemble = create_ensemble(rho, V, cfg, seed=geometry_seed, verbose=False)
    stream = make_walk_random_stream(cfg, stream_seed)
    Y, n_escaped, telemetry = run_walk_Y(
        ensemble, 0.0, cfg, pp=pp, seed=0, verbose=False,
        return_telemetry=True, classifier="cache", random_stream=stream,
        use_gpu=False,
    )
    if n_escaped:
        raise RuntimeError("periodic CPU golden unexpectedly recorded an escape")

    case = {
        "schema": "madi-cpu-gpu-golden-v1",
        "rho_requested_per_uL": rho,
        "V_requested_pL": V,
        "pp": pp,
        "geometry_seed": geometry_seed,
        "stream_seed": stream_seed,
        "n_escaped": int(n_escaped),
        "geometry": asdict(ensemble.geometry),
        "rho_realised_per_uL": ensemble.rho,
        "V_realised_pL": ensemble.V,
        "vi_realised": ensemble.vi,
        "alpha_star_um": ensemble.alpha_star,
        "mean_A_over_V_um_inv": ensemble.mean_AV,
    }
    np.savez_compressed(
        output,
        schema=np.array(case["schema"]),
        config_json=np.array(json.dumps(_config_metadata(cfg), sort_keys=True, default=_json_default)),
        case_json=np.array(json.dumps(case, sort_keys=True, default=_json_default)),
        seeds=ensemble.seeds,
        annulus=ensemble.annulus,
        grid_candidates=ensemble.grid_candidates,
        initial_positions=stream.initial_positions,
        increments=stream.increments,
        acceptance_uniforms=stream.acceptance_uniforms,
        Y_cpu=Y,
        escaped_cpu=np.array(n_escaped, dtype=np.int64),
        metrics_cpu=np.asarray(telemetry["metrics"], dtype=np.float64),
        occupancy_fraction_cpu=np.asarray(telemetry["occupancy_fraction"], dtype=np.float64),
        start_survivor_fraction_cpu=np.asarray(telemetry["start_survivor_fraction"], dtype=np.float64),
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"CPU golden: {output}")
    print(f"SHA-256:    {digest}")
    print(f"Hash file:  {sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

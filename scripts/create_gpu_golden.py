#!/usr/bin/env python3
"""Create the fixed CPU reference artifact for the Tier-C GPU golden test.

The artifact contains one exact full-facet SI §S.II geometry, a fixed full random-input
stream, and the CPU output.  The Sol job runs the CUDA transliteration on
those same inputs and exits nonzero if the two implementations diverge.

This is deliberately a *test-only* small geometry reference.  It exercises
the classifier and crossing implementation, not the production 5e6-cell
``<A/V>`` calibration table.  A production library build remains blocked
until that certified table exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from madi.config import SimConfig
from madi.ensemble import alpha_star_from_vi, create_ensemble
from madi.walker_gpu import make_walk_random_stream, run_walk_Y


DEFAULT_OUTPUT = Path("tests/physics_audit/data/cpu_gpu_golden_v1.npz")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def _write_test_reference(path: Path) -> None:
    """Write a clearly non-certified SI-schema fixture for this golden case."""
    vi = np.linspace(0.40, 1.00, 26, dtype=np.float64)
    # rho=1 cell/um^3 makes alpha_star numerically equal to SI x.
    alpha_x = np.asarray([
        0.0 if value >= 1.0 else alpha_star_from_vi(value, 1.0e9)
        for value in vi
    ])
    # The magnitude is immaterial to the golden comparison; it only selects
    # a nonzero p_p for the exercised trajectory.  Its metadata prevents the
    # fixture from ever passing as a production calibration table.
    mean = 4.0 - 2.0 * vi
    metadata = {
        "schema": "madi-si-test-reference-v1",
        "source": "test fixture; not a production SI reference cloud",
        "rho_reference": 1.0,
        "kappa": 0.90,
        "n_single_cells": 16,
        "n_alpha": 26,
        "alpha_spacing": "linear",
        "mean_estimator": "untrimmed_arithmetic_mean_A_over_V",
        "contraction_rule": "all_shifted_voronoi_facets_S2",
    }
    np.savez(
        path,
        vi=vi,
        alpha_x=alpha_x,
        mean_A_over_V_norm=mean,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def _case_config(reference: Path) -> SimConfig:
    """Small exact-classifier case with a source-to-boundary safety margin."""
    return SimConfig(
        # Explicit L is permitted only in a Tier-A/Tier-C diagnostic.  The
        # production builder leaves it None and uses SI Eq. S8 per entry.
        L=160.0,
        geometry_reference_path=str(reference),
        allow_uncertified_geometry_reference=True,
        geometry_validation_points=10_000,
        geometry_vi_tolerance=0.03,
        n_walkers=64,
        n_ensembles=1,
        T_max_ms=4.0,
        small_deltas=[1.0],
        big_deltas=[2.0, 3.0],
        b_values=[0.0, 500.0, 1_000.0, 2_000.0],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output: Path = args.output
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace the CPU golden")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="madi_gpu_golden_") as temporary:
        reference = Path(temporary) / "test_geometry_reference.npz"
        _write_test_reference(reference)
        cfg = _case_config(reference)
        # v_i=0.90 deliberately exercises thin annuli, full-facet
        # classification, and two-membrane endpoint transitions.  It is not
        # a production calibration case.
        rho, V, pp = 1.0e6, 0.9, 0.02
        geometry_seed, stream_seed = 20_260_811, 20_260_812
        ensemble = create_ensemble(rho, V, cfg, seed=geometry_seed, verbose=False)
        stream = make_walk_random_stream(cfg, stream_seed, ensemble)
        Y, n_escaped, telemetry = run_walk_Y(
            ensemble, 0.0, cfg, pp=pp, seed=0, verbose=False,
            return_telemetry=True, classifier="exact", random_stream=stream,
            use_gpu=False,
        )
    if n_escaped:
        raise RuntimeError("SI source-domain CPU golden recorded an escape")

    case = {
        "schema": "madi-cpu-gpu-golden-v3-full-facet",
        "rho_requested_per_uL": rho,
        "V_requested_pL": V,
        "pp": pp,
        "geometry_seed": geometry_seed,
        "stream_seed": stream_seed,
        "n_escaped": int(n_escaped),
        "geometry": ensemble.geometry.to_dict(),
        "rho_realised_per_uL": ensemble.rho,
        "V_realised_pL": ensemble.V,
        "vi_realised": ensemble.vi,
        "alpha_star_um": ensemble.alpha_star,
        "mean_A_over_V_um_inv": ensemble.mean_AV,
        "source_lo_um": ensemble.source_lo,
        "source_hi_um": ensemble.source_hi,
        "reference_fixture": "non-certified test fixture; production table not used",
    }
    np.savez_compressed(
        output,
        schema=np.array(case["schema"]),
        config_json=np.array(json.dumps(asdict(cfg), sort_keys=True, default=_json_default)),
        case_json=np.array(json.dumps(case, sort_keys=True, default=_json_default)),
        seeds=ensemble.seeds,
        annulus=ensemble.annulus,
        kd_node_seed=ensemble.kd_node_seed,
        kd_node_axis=ensemble.kd_node_axis,
        kd_node_left=ensemble.kd_node_left,
        kd_node_right=ensemble.kd_node_right,
        kd_node_parent=ensemble.kd_node_parent,
        initial_positions=stream.initial_positions,
        increments=stream.increments,
        acceptance_uniforms=stream.acceptance_uniforms,
        Y_cpu=Y,
        escaped_cpu=np.array(n_escaped, dtype=np.int64),
        occupancy_fraction_cpu=np.asarray(telemetry["occupancy_fraction"], dtype=np.float64),
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

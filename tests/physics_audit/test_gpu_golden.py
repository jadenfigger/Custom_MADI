"""Tier-A integrity check for the committed CPU reference golden file."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from madi.config import SimConfig
from madi.ensemble import Ensemble, GeometryStats
from madi.walker_gpu import WalkRandomStream, run_walk_Y


pytestmark = pytest.mark.tier_a

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/physics_audit/data/cpu_gpu_golden_v1.npz"


def _ensemble(data, cfg: SimConfig, case: dict) -> Ensemble:
    geometry_keys = {field.name for field in fields(GeometryStats)}
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


def test_cpu_golden_hash_and_deterministic_reference_replay() -> None:
    """The committed reference is intact and CPU replay remains bitwise stable."""
    expected = GOLDEN.with_suffix(GOLDEN.suffix + ".sha256").read_text().split()[0]
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == expected
    data = np.load(GOLDEN, allow_pickle=False)
    cfg = SimConfig(**json.loads(str(data["config_json"])))
    case = json.loads(str(data["case_json"]))
    stream = WalkRandomStream(
        initial_positions=np.asarray(data["initial_positions"], dtype=np.float64),
        increments=np.asarray(data["increments"], dtype=np.float64),
        acceptance_uniforms=np.asarray(data["acceptance_uniforms"], dtype=np.float64),
    )
    Y, escaped, telemetry = run_walk_Y(
        _ensemble(data, cfg, case), 0.0, cfg, pp=float(case["pp"]),
        seed=0, verbose=False, return_telemetry=True, classifier="cache",
        random_stream=stream, use_gpu=False,
    )
    assert escaped == int(data["escaped_cpu"])
    assert np.array_equal(Y, data["Y_cpu"])
    assert np.array_equal(telemetry["metrics"], data["metrics_cpu"])
    assert np.array_equal(telemetry["occupancy_fraction"], data["occupancy_fraction_cpu"])
    assert np.array_equal(
        telemetry["start_survivor_fraction"], data["start_survivor_fraction_cpu"]
    )

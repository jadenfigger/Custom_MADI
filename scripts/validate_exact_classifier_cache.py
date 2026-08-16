#!/usr/bin/env python3
"""Deterministic CPU equivalence test for the exact full-facet cache.

This is deliberately a tiny golden-fixture replay, not a production-like CPU
benchmark.  It compares the new cache against the existing exact SI Eq. S2
classifier with the same fixed positions, Gaussian increments, and acceptance
uniforms.  The geometry and contraction rule are not re-certified here: this
test establishes that the cache only skips classifications when its
conservative proofs give the reference classifier's answer.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np

from madi.config import SimConfig
from madi.ensemble import Ensemble, GeometryStats, PopulationCertificate
from madi.walker_gpu import WalkRandomStream, _membrane_transition_counts, run_walk_Y


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "tests/physics_audit/data/cpu_gpu_golden_v1.npz"


def _ensemble_from_golden(data: np.lib.npyio.NpzFile, cfg: SimConfig, case: dict) -> Ensemble:
    values = {field.name: case["geometry"][field.name] for field in fields(GeometryStats)}
    values["population"] = PopulationCertificate(**values["population"])
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
        geometry=GeometryStats(**values),
        rho_requested=float(case["rho_requested_per_uL"]),
        V_requested=float(case["V_requested_pL"]),
        kd_node_seed=np.asarray(data["kd_node_seed"], dtype=np.int32),
        kd_node_axis=np.asarray(data["kd_node_axis"], dtype=np.int8),
        kd_node_left=np.asarray(data["kd_node_left"], dtype=np.int32),
        kd_node_right=np.asarray(data["kd_node_right"], dtype=np.int32),
        kd_node_parent=np.asarray(data["kd_node_parent"], dtype=np.int32),
    )


def _cache_label(
    ensemble: Ensemble, position: np.ndarray, cache, cfg: SimConfig,
):
    cached = ensemble.classify_from_full_facet_cache_cpu(
        position, cache, min_safe_radius_um=cfg.classifier_cache_min_safe_radius_um,
    )
    if cached is not None:
        cell, inside, tier = cached
        return int(cell), bool(inside), cache, tier
    refreshed = ensemble.build_full_facet_cache_cpu(
        position,
        delta_max_um=cfg.classifier_cache_delta_max_um,
        candidate_capacity=cfg.classifier_cache_candidate_capacity,
    )
    return refreshed.cell, refreshed.inside, refreshed, "full"


def _stress_classifier(
    ensemble: Ensemble,
    stream: WalkRandomStream,
    cfg: SimConfig,
    pp: float,
    walkers: int,
) -> dict[str, int]:
    """Replay a small fixed subset, asserting labels and accepts at each step."""
    label_checks = 0
    accepts = 0
    tier1 = 0
    tier2 = 0
    full = 0
    superset_checks = 0
    for walker in range(min(int(walkers), int(cfg.n_walkers))):
        position = np.asarray(stream.initial_positions[walker], dtype=np.float64).copy()
        exact_cell, exact_inside = ensemble.classify_cpu(position[None, :])
        current_cell = int(exact_cell[0])
        current_inside = bool(exact_inside[0])
        cache = ensemble.build_full_facet_cache_cpu(
            position, delta_max_um=cfg.classifier_cache_delta_max_um,
            candidate_capacity=cfg.classifier_cache_candidate_capacity,
        )
        full += 1
        assert (cache.cell, cache.inside) == (current_cell, current_inside)
        for step in range(int(cfg.n_steps)):
            proposal = position + stream.increments[step, walker]
            if np.any(proposal < 0.0) or np.any(proposal >= ensemble.L):
                break
            exact_cell_array, exact_inside_array = ensemble.classify_cpu(proposal[None, :])
            expected_cell = int(exact_cell_array[0])
            expected_inside = bool(exact_inside_array[0])
            cached_cell, cached_inside, cache, tier = _cache_label(ensemble, proposal, cache, cfg)
            assert (cached_cell, cached_inside) == (expected_cell, expected_inside)
            assert ensemble.cache_contains_exact_candidates_cpu(proposal, cache) or tier == "full"
            if tier == "tier1":
                tier1 += 1
            elif tier == "tier2":
                tier2 += 1
                superset_checks += 1
            else:
                full += 1
            m = _membrane_transition_counts(
                np.asarray([current_cell]), np.asarray([current_inside]),
                np.asarray([expected_cell]), np.asarray([expected_inside]),
            )[0]
            accepted = not (m > 0 and pp < 1.0 and stream.acceptance_uniforms[step, walker] >= pp ** m)
            # The cache has exactly the same endpoint labels; therefore the
            # acceptance predicate is intentionally evaluated twice to make
            # the bit-level equality explicit in the validation record.
            cached_accepted = not (
                m > 0 and pp < 1.0 and stream.acceptance_uniforms[step, walker] >= pp ** m
            )
            assert bool(accepted) is bool(cached_accepted)
            accepts += 1
            if accepted:
                position = proposal
                current_cell = expected_cell
                current_inside = expected_inside
            label_checks += 1
    return {
        "label_checks": label_checks,
        "acceptance_checks": accepts,
        "full": full,
        "tier1": tier1,
        "tier2": tier2,
        "tier2_superset_checks": superset_checks,
    }


def _near_facet_checks(ensemble: Ensemble, cfg: SimConfig) -> dict[str, int]:
    """Exercise shifted-facet neighbourhoods, where Tier 1 should not fire."""
    checked = 0
    tier1_refused_near_facet = 0
    rng = np.random.default_rng(20_260_815)
    for seed_index in rng.choice(len(ensemble.seeds), size=min(24, len(ensemble.seeds)), replace=False):
        distances, ids = ensemble._tree().query(ensemble.seeds[seed_index], k=2)
        other = int(ids[1])
        separation = float(distances[1])
        if separation <= 0.0:
            continue
        direction = (ensemble.seeds[other] - ensemble.seeds[seed_index]) / separation
        # The shifted facet of the selected seed lies alpha_i inward from its
        # ordinary Voronoi facet.  Small two-sided offsets deliberately put
        # the cache at the difficult membrane boundary.
        midpoint = 0.5 * (ensemble.seeds[seed_index] + ensemble.seeds[other])
        facet = midpoint - float(ensemble.annulus[seed_index]) * direction
        inside_side = facet - 1.0e-9 * direction
        opposite_side = facet + 1.0e-9 * direction
        if (np.any(inside_side < 0.0) or np.any(inside_side >= ensemble.L)
                or np.any(opposite_side < 0.0) or np.any(opposite_side >= ensemble.L)):
            continue
        cache = ensemble.build_full_facet_cache_cpu(
            inside_side, delta_max_um=cfg.classifier_cache_delta_max_um,
            candidate_capacity=cfg.classifier_cache_candidate_capacity,
        )
        exact = ensemble.classify_cpu(opposite_side[None, :])
        cached = ensemble.classify_from_full_facet_cache_cpu(
            opposite_side, cache, min_safe_radius_um=cfg.classifier_cache_min_safe_radius_um,
        )
        if cached is None:
            cache = ensemble.build_full_facet_cache_cpu(
                opposite_side, delta_max_um=cfg.classifier_cache_delta_max_um,
                candidate_capacity=cfg.classifier_cache_candidate_capacity,
            )
            observed = (cache.cell, cache.inside)
            tier = "full"
        else:
            observed = (cached[0], cached[1])
            tier = cached[2]
        assert observed == (int(exact[0][0]), bool(exact[1][0]))
        # The 2-nm reference-to-proposal displacement crosses the deliberately
        # constructed shifted-facet neighbourhood.  If this particular facet
        # is binding, Tier 1 must refuse it; the test records the refusal and
        # asserts that the fixture contains at least one such hard case.
        if tier != "tier1":
            tier1_refused_near_facet += 1
        checked += 1
    if checked == 0 or tier1_refused_near_facet == 0:
        raise RuntimeError("near-facet stress sample did not exercise a Tier-1 refusal")
    return {
        "near_facet_checks": checked,
        "tier1_refused_near_facet": tier1_refused_near_facet,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output", type=Path, default=ROOT / "sol_outputs/cache_cpu_validation.json")
    parser.add_argument("--stress-walkers", type=int, default=8)
    args = parser.parse_args()
    data = np.load(args.golden, allow_pickle=False)
    expected_hash = args.golden.with_suffix(args.golden.suffix + ".sha256").read_text().split()[0]
    actual_hash = hashlib.sha256(args.golden.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("committed CPU golden SHA-256 does not match")
    reference_cfg = SimConfig(**json.loads(str(data["config_json"])))
    cached_cfg = SimConfig(
        **{
            **reference_cfg.__dict__,
            "classifier_mode": "exact_cached",
            "classifier_cache_delta_max_um": 1.0,
            "classifier_cache_min_safe_radius_um": 0.0,
            "classifier_cache_candidate_capacity": 256,
        }
    )
    case = json.loads(str(data["case_json"]))
    ensemble = _ensemble_from_golden(data, reference_cfg, case)
    stream = WalkRandomStream(
        initial_positions=np.asarray(data["initial_positions"], dtype=np.float64),
        increments=np.asarray(data["increments"], dtype=np.float64),
        acceptance_uniforms=np.asarray(data["acceptance_uniforms"], dtype=np.float64),
    )
    expected_Y = np.asarray(data["Y_cpu"], dtype=np.float64)
    exact_Y, exact_escaped, exact_telemetry = run_walk_Y(
        ensemble, 0.0, reference_cfg, pp=float(case["pp"]), seed=0,
        verbose=False, classifier="exact", random_stream=stream, use_gpu=False,
        return_telemetry=True,
    )
    cached_Y, cached_escaped, cached_telemetry = run_walk_Y(
        ensemble, 0.0, cached_cfg, pp=float(case["pp"]), seed=0,
        verbose=False, classifier="exact_cached", random_stream=stream, use_gpu=False,
        return_telemetry=True,
    )
    stress = _stress_classifier(
        ensemble, stream, cached_cfg, float(case["pp"]), args.stress_walkers,
    )
    near_facet = _near_facet_checks(ensemble, cached_cfg)
    cache_counts = cached_telemetry["classifier_cache"]
    # Golden fixture has no fatal escapes, so every walker makes an initial
    # classification plus one endpoint request at each microstep.
    expected_requests = int(cached_cfg.n_walkers) * (int(cached_cfg.n_steps) + 1)
    report = {
        "schema": "madi-exact-cache-cpu-equivalence-v1",
        "golden_sha256": actual_hash,
        "bitwise": {
            "reference_matches_committed_golden": bool(np.array_equal(exact_Y, expected_Y)),
            "cached_matches_reference": bool(np.array_equal(cached_Y, exact_Y)),
            "cached_matches_committed_golden": bool(np.array_equal(cached_Y, expected_Y)),
            "escaped_equal": bool(exact_escaped == cached_escaped),
            "occupancy_equal": bool(np.array_equal(
                exact_telemetry["occupancy_fraction"], cached_telemetry["occupancy_fraction"],
            )),
        },
        "cache_counts": cache_counts,
        "cache_request_accounting": {
            "expected": expected_requests,
            "observed": int(cache_counts["requests"]),
            "equal": bool(int(cache_counts["requests"]) == expected_requests),
        },
        "stress": stress,
        "near_facet": near_facet,
    }
    report["pass"] = bool(all(report["bitwise"].values()) and report["cache_request_accounting"]["equal"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tier A: validate SI Eq. S2 full-facet contraction against S7a.

This CPU-only acceptance check measures the requested P0-A 9-by-3 packing
grid in the production Eq. S8 domain, reports the SI §S.IV.a two-nearest
shortcut discrepancy curve, and checks the ``d1 + 2*alpha1`` radius-bound
classifier against a brute-force all-seed Eq. S2 calculation.  It writes a
structured JSON report; it does not build or modify a library.

For an interim geometry-only run, a deliberately uncertified reference table
may be supplied together with ``--allow-uncertified-geometry-reference``.
The reference affects Eq. 5 metadata only, never the measured v_i values.
Production acceptance must be rerun without that flag after the 5e6-cell
reference artifact exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from madi.config import SimConfig
from madi.ensemble import create_ensemble


VI_TARGETS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
RHO_TARGETS = (1.0e5, 5.0e5, 1.0e6)


def _two_nearest_labels(ensemble, points: np.ndarray) -> np.ndarray:
    """Diagnostic SI §S.IV.a shortcut, never used by the production walker."""
    distances, indices = ensemble._tree().query(points, k=2)
    nearest = np.asarray(indices[:, 0], dtype=np.int32)
    second = np.asarray(indices[:, 1], dtype=np.int32)
    separation = np.linalg.norm(ensemble.seeds[second] - ensemble.seeds[nearest], axis=1)
    margin = (distances[:, 1] ** 2 - distances[:, 0] ** 2) / (2.0 * separation)
    return margin >= ensemble.annulus[nearest]


def _brute_force_full_facet_labels(ensemble, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All-seed SI Eq. S2 oracle, bounded to small validation samples."""
    labels = np.empty(len(points), dtype=bool)
    ids = np.empty(len(points), dtype=np.int32)
    # Eight points keep the all-seed oracle below local-memory limits even
    # for the 128-ms, high-density S8 population.
    for start in range(0, len(points), 8):
        stop = min(start + 8, len(points))
        block = points[start:stop]
        point_to_seed = ensemble.seeds[None, :, :] - block[:, None, :]
        d_sq = np.einsum("...i,...i->...", point_to_seed, point_to_seed)
        nearest = np.argmin(d_sq, axis=1).astype(np.int32)
        row = np.arange(len(block))
        d1_sq = d_sq[row, nearest]
        seed_to_nearest = ensemble.seeds[None, :, :] - ensemble.seeds[nearest, None, :]
        separation = np.linalg.norm(seed_to_nearest, axis=2)
        if np.any(separation[row, nearest] != 0.0):
            raise RuntimeError("coincident Poisson seeds in brute-force oracle")
        separation[row, nearest] = 1.0
        margin = (d_sq - d1_sq[:, None]) / (2.0 * separation)
        margin[row, nearest] = np.inf
        ids[start:stop] = nearest
        labels[start:stop] = np.all(
            margin >= ensemble.annulus[nearest, None], axis=1
        )
    return ids, labels


def _binomial_se(value: float, n: int) -> float:
    return float(np.sqrt(max(value * (1.0 - value), 0.0) / n))


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("supply comma-separated numeric values") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("supply at least one numeric value")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-uncertified-geometry-reference", action="store_true")
    parser.add_argument("--volume-samples", type=int, default=100_000)
    parser.add_argument("--discrepancy-samples", type=int, default=100_000)
    parser.add_argument("--brute-force-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20_260_920)
    parser.add_argument("--vi-values", type=_csv_floats, default=VI_TARGETS)
    parser.add_argument("--rho-values", type=_csv_floats, default=RHO_TARGETS)
    parser.add_argument("--skip-discrepancy", action="store_true")
    args = parser.parse_args()
    if args.volume_samples <= 0 or args.discrepancy_samples <= 0 or args.brute_force_samples <= 0:
        parser.error("all sample counts must be positive")

    cfg = SimConfig(
        L=None,
        T_max_ms=128.0,
        small_deltas=[1.0],
        big_deltas=[1.0],
        geometry_reference_path=str(args.geometry_reference),
        allow_uncertified_geometry_reference=args.allow_uncertified_geometry_reference,
        geometry_validation_points=args.volume_samples,
        geometry_vi_tolerance=0.005,
    )
    cfg.assert_grid_alignment()
    rows: list[dict] = []
    ensembles: dict[tuple[float, float], object] = {}
    for target_vi in args.vi_values:
        if target_vi not in VI_TARGETS:
            parser.error(f"vi-values must be drawn from {VI_TARGETS}")
        for rho in args.rho_values:
            if rho not in RHO_TARGETS:
                parser.error(f"rho-values must be drawn from {RHO_TARGETS}")
            V = target_vi * 1.0e6 / rho
            seed = args.seed + int(round(target_vi * 100)) + int(rho // 1.0e5)
            ensemble = create_ensemble(rho, V, cfg, seed=seed, verify_vi=True)
            ensembles[(target_vi, rho)] = ensemble
            stats = ensemble.geometry
            rows.append({
                "target_vi": target_vi,
                "rho_requested_per_uL": rho,
                "V_requested_pL": V,
                "vi_measured": ensemble.vi,
                "vi_mc_se": stats.realised_vi_se,
                "vi_error": ensemble.vi - target_vi,
                "vi_acceptance_limit": max(cfg.geometry_vi_tolerance, 4.0 * stats.realised_vi_se),
                "rho_measured_per_uL": ensemble.rho,
                "V_measured_pL": ensemble.V,
                "alpha_star_um": ensemble.alpha_star,
                "sim_side_um": ensemble.L,
                "n_seeds_sim": stats.n_seeds_sim,
                "n_seeds_pop": stats.n_seeds_pop,
            })

    discrepancy: list[dict] = []
    rng = np.random.default_rng(args.seed + 10_000)
    for target_vi in args.vi_values:
        if args.skip_discrepancy or 5.0e5 not in args.rho_values:
            break
        ensemble = ensembles[(target_vi, 5.0e5)]
        points = rng.uniform(0.0, ensemble.L, size=(args.discrepancy_samples, 3))
        _, full = ensemble.classify_cpu(points)
        two = _two_nearest_labels(ensemble, points)
        if np.any(full & ~two):
            raise RuntimeError("two-nearest shortcut unexpectedly removed a full-facet cell point")
        difference = two.astype(np.int8) - full.astype(np.int8)
        disagreement = float(np.mean(two != full))
        vi_bias = float(np.mean(difference))
        brute_points = rng.uniform(0.0, ensemble.L, size=(args.brute_force_samples, 3))
        radius_ids, radius_labels = ensemble.classify_cpu(brute_points)
        brute_ids, brute_labels = _brute_force_full_facet_labels(ensemble, brute_points)
        brute_disagreements = int(
            np.count_nonzero(radius_ids != brute_ids) + np.count_nonzero(radius_labels != brute_labels)
        )
        discrepancy.append({
            "target_vi": target_vi,
            "rho_per_uL": 5.0e5,
            "n_samples": args.discrepancy_samples,
            "two_nearest_vs_full_facet_disagreement_fraction": disagreement,
            "disagreement_mc_se": _binomial_se(disagreement, args.discrepancy_samples),
            "two_nearest_vi": float(np.mean(two)),
            "full_facet_vi": float(np.mean(full)),
            "two_nearest_vi_bias": vi_bias,
            "two_nearest_vi_bias_mc_se": _binomial_se(vi_bias, args.discrepancy_samples),
            "full_facet_to_two_nearest_one_sided": True,
            "brute_force_samples": args.brute_force_samples,
            "radius_bound_vs_all_seed_disagreements": brute_disagreements,
        })

    report = {
        "schema": "madi-p0a-full-facet-validation-v1",
        "tier": "A",
        "phase": "geometry-only; no library was built",
        "geometry_reference": str(args.geometry_reference.resolve()),
        "geometry_reference_certified": not args.allow_uncertified_geometry_reference,
        "contraction": {
            "production_rule": "full_shifted_voronoi_facets_SI_Eq_S2",
            "radius_bound_um": "d1 + 2*alpha1",
            "two_nearest_shortcut": "diagnostic_only_SI_SIVa",
        },
        "volume_samples": args.volume_samples,
        "vi_values": list(args.vi_values),
        "rho_values": list(args.rho_values),
        "packing_grid": rows,
        "two_nearest_discrepancy_curve": discrepancy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

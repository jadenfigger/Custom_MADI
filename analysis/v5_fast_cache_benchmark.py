#!/usr/bin/env python3
"""GPU-only equivalence and speed benchmark for the exact classifier cache.

This companion reuses the production-grid coordinate resolver and the normal
walk-plus-reduction path from ``v5_runtime_benchmark``.  It does not build a
library.  For one fixed geometry it runs the reference full classifier and
one or more conservative-cache configurations from the identical walk seed,
requires bit-identical reduced output, and records speed plus Tier-1/Tier-2
hit rates split by intracellular/extracellular endpoint class.

``--mode sweep`` is a small parameter-selection measurement.  ``--mode
speed`` is the full 50,000-walker gate at the selected cache settings.  Both
must run on Sol with CUDA; this program refuses a CPU host.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from analysis.v5_runtime_benchmark import (
    BUILD_SEED,
    _canonical_coordinate,
    _device_name,
    _warm_cuda,
)
from madi.config import PRODUCTION_WALKERS_PER_ENSEMBLE, SimConfig
from madi.library import make_remediation_log_grid
from madi.signal import build_columns
from madi.walker_gpu import (
    HAS_CUDA,
    _build_ensembles_for_entry,
    _checked_pp,
    _walk_and_reduce_one_ensemble,
    _walk_seed,
    kio_to_pp,
)


KIO_S_INV = 20.0
SWEEP_WALKERS = 5_000
DEFAULT_SWEEP_OPTIONS = (
    "0.5:0.0:256",
    "1.0:0.0:256",
    "2.0:0.0:256",
    "1.0:0.25:256",
    "1.0:0.50:256",
)


def _parse_option(value: str) -> tuple[float, float, int]:
    try:
        delta, safe, capacity = value.split(":")
        parsed = float(delta), float(safe), int(capacity)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "cache option must be delta_max_um:min_safe_radius_um:candidate_capacity"
        ) from exc
    if parsed[0] <= 0.0 or parsed[1] < 0.0 or parsed[2] < 2:
        raise argparse.ArgumentTypeError("cache option values must be positive (safe radius may be zero)")
    return parsed


def _cache_rates(stats: dict[str, int]) -> dict[str, float | int]:
    full = int(stats["full"])
    tier1 = int(stats["tier1"])
    tier2 = int(stats["tier2"])
    inside_requests = int(stats["full_inside"] + stats["tier1_inside"] + stats["tier2_inside"])
    outside_requests = int(stats["full_outside"] + stats["tier2_outside"])
    return {
        "requests": int(stats["requests"]),
        "full_requests": full,
        "tier1_hits": tier1,
        "tier2_hits": tier2,
        "all_fast_path_hits": tier1 + tier2,
        "all_fast_path_hit_rate": (tier1 + tier2) / int(stats["requests"]),
        "intracellular_requests": inside_requests,
        "intracellular_fast_path_hits": int(stats["tier1_inside"] + stats["tier2_inside"]),
        "intracellular_fast_path_hit_rate": (
            (stats["tier1_inside"] + stats["tier2_inside"]) / inside_requests
            if inside_requests else 0.0
        ),
        "extracellular_requests": outside_requests,
        "extracellular_fast_path_hits": int(stats["tier2_outside"]),
        "extracellular_fast_path_hit_rate": (
            stats["tier2_outside"] / outside_requests if outside_requests else 0.0
        ),
        "candidate_overflow": int(stats["candidate_overflow"]),
    }


def _run_one(ensemble: Any, cfg: SimConfig, columns: Any, pp: float, seed: int) -> tuple[Any, float]:
    start = time.perf_counter()
    result = _walk_and_reduce_one_ensemble(
        ensemble, pp, cfg,
        columns.j_delta, columns.j_Delta, columns.j_sum, columns.phase_coef,
        seed,
    )
    seconds = time.perf_counter() - start
    if result.n_escaped:
        raise RuntimeError(f"fatal SI escape in cache benchmark: {result.n_escaped}")
    if len(result.classifier_cache_stats) != 1:
        raise RuntimeError("benchmark expected exactly one cache-stat record")
    return result, seconds


def _result_equal(reference: Any, candidate: Any) -> dict[str, bool]:
    return {
        "cos_sum_bitwise_equal": bool(np.array_equal(reference.cos_sum, candidate.cos_sum)),
        "sin_sum_bitwise_equal": bool(np.array_equal(reference.sin_sum, candidate.sin_sum)),
        "occupancy_counts_bitwise_equal": bool(
            np.array_equal(reference.occupancy_counts, candidate.occupancy_counts)
        ),
        "n_escaped_equal": bool(reference.n_escaped == candidate.n_escaped),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", choices=("low", "center", "high"), required=True)
    parser.add_argument("--mode", choices=("sweep", "speed"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry-reference", type=Path,
        default=Path("data/geometry_reference_si_kappa_0p9.npz"),
    )
    parser.add_argument("--seed", type=int, default=BUILD_SEED)
    parser.add_argument("--walkers", type=int, default=None)
    parser.add_argument(
        "--cache-option", action="append", type=_parse_option, default=None,
        help="repeatable delta_max_um:min_safe_radius_um:candidate_capacity",
    )
    args = parser.parse_args(argv)

    if not HAS_CUDA:
        raise RuntimeError("CUDA is unavailable; run this benchmark on Sol")
    if not args.geometry_reference.is_file():
        raise FileNotFoundError(f"certified geometry reference missing: {args.geometry_reference}")
    walkers = int(args.walkers if args.walkers is not None else (
        SWEEP_WALKERS if args.mode == "sweep" else PRODUCTION_WALKERS_PER_ENSEMBLE
    ))
    if walkers <= 0:
        raise ValueError("walkers must be positive")
    if args.mode == "speed" and walkers != PRODUCTION_WALKERS_PER_ENSEMBLE:
        raise ValueError(
            f"speed mode must use {PRODUCTION_WALKERS_PER_ENSEMBLE:,} walkers; got {walkers:,}"
        )
    options = args.cache_option
    if options is None:
        options = [_parse_option(value) for value in (
            DEFAULT_SWEEP_OPTIONS if args.mode == "sweep" else ("1.0:0.0:256",)
        )]
    if args.mode == "speed" and len(options) != 1:
        raise ValueError("speed mode accepts exactly one selected cache option")

    coordinate = _canonical_coordinate(args.density)
    cfg_exact = SimConfig(
        n_walkers=walkers,
        n_ensembles=1,
        geometry_reference_path=str(args.geometry_reference),
        classifier_mode="exact",
    )
    cfg_exact.assert_grid_alignment()
    columns = build_columns(cfg_exact)
    if columns.n_pairs * columns.n_b != 31_125:
        raise RuntimeError("cache benchmark must use the production storage grid")
    grid = make_remediation_log_grid()
    if not np.isclose(float(coordinate["rho_per_uL"]), float(grid.rhos[int(coordinate["rho_index"])])):
        raise RuntimeError("coordinate no longer resolves to canonical production rho node")

    geometry_start = time.perf_counter()
    ensembles = _build_ensembles_for_entry(
        float(coordinate["rho_per_uL"]), float(coordinate["V_pL"]),
        cfg_exact, int(args.seed), verbose=False,
    )
    geometry_seconds = time.perf_counter() - geometry_start
    ensemble = ensembles[0]
    pp = _checked_pp(kio_to_pp(KIO_S_INV, ensemble.mean_AV, cfg_exact))
    walk_seed = _walk_seed(int(args.seed), 0)

    # CUDA compilation is deliberately outside timing.  It is a one-time job
    # cost, not a library-entry cost.
    exact_warmup_seconds = _warm_cuda(ensemble, pp, cfg_exact, columns, walk_seed)
    reference, reference_seconds = _run_one(ensemble, cfg_exact, columns, pp, walk_seed)

    runs: list[dict[str, Any]] = []
    for delta_max_um, min_safe_radius_um, capacity in options:
        cfg_cache = replace(
            cfg_exact,
            classifier_mode="exact_cached",
            classifier_cache_delta_max_um=delta_max_um,
            classifier_cache_min_safe_radius_um=min_safe_radius_um,
            classifier_cache_candidate_capacity=capacity,
        )
        cache_warmup_seconds = _warm_cuda(ensemble, pp, cfg_cache, columns, walk_seed)
        candidate, candidate_seconds = _run_one(ensemble, cfg_cache, columns, pp, walk_seed)
        equality = _result_equal(reference, candidate)
        if not all(equality.values()):
            raise RuntimeError(f"cached classifier is not bit-identical to exact reference: {equality}")
        stats = candidate.classifier_cache_stats[0]
        runs.append({
            "cache": {
                "delta_max_um": delta_max_um,
                "min_safe_radius_um": min_safe_radius_um,
                "candidate_capacity": capacity,
            },
            "cuda_jit_warmup_seconds_excluded_from_cost": cache_warmup_seconds,
            "walk_and_reduction_seconds": candidate_seconds,
            "speedup_vs_exact": reference_seconds / candidate_seconds,
            "bitwise_equivalence": equality,
            "cache_counts": stats,
            "cache_rates": _cache_rates(stats),
        })

    payload: dict[str, Any] = {
        "schema": "madi-v5-exact-cache-gpu-benchmark-v1",
        "mode": args.mode,
        "gpu": _device_name(),
        "build_seed": int(args.seed),
        "coordinate": coordinate,
        "production_grid": {"rho_nodes": len(grid.rhos), "V_nodes": len(grid.Vs)},
        "walkers": walkers,
        "n_ensembles": 1,
        "kio_s_inv": KIO_S_INV,
        "pp": pp,
        "geometry_seconds_excluded_from_walk_comparison": geometry_seconds,
        "exact": {
            "cuda_jit_warmup_seconds_excluded_from_cost": exact_warmup_seconds,
            "walk_and_reduction_seconds": reference_seconds,
            "cache_counts": reference.classifier_cache_stats[0],
        },
        "cached_runs": runs,
        "notes": [
            "Geometry is built once and excluded from the exact-versus-cached walk comparison.",
            "Every cached run uses the same geometry and walk seed as the exact reference.",
            "A cache result is accepted only when cos/sin reductions, occupancy, and escape status are bit-identical.",
            "Tier-1 has no extracellular hits by construction; extracellular acceleration is Tier 2.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"v5_exact_cache_{args.mode}_{args.density}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the MADI I SI §S.II / §S.IV single-cell geometry reference.

The production ``<A/V>`` table is intentionally a separate, auditable
artifact.  It is generated from 5,000,000 independent rho=1 Poisson-Voronoi
cells, κ=0.9, and 26 linearly spaced alpha-star values.  Each cell contributes
its *untrimmed* A_i/V_i at every alpha value; this script never samples a
multi-cell Ω_sim or uses a percentile trim.

The contraction is the full SI Eq. S2 conjunction: every Voronoi neighbour
contributes its shifted half-space before the convex hull is measured.  It is
therefore the same geometric object as the production walker classifier, not
the two-nearest S13--S14 shortcut in SI §S.IV.a.

The default range covers the remediation library's v_i=[0.40,0.99].  At
rho=1, alpha_star equals the dimensionless SI coordinate x=rho^(1/3)alpha*.
Use shards for the full calculation, then merge their sufficient statistics:

  python -m scripts.build_si_geometry_reference --shard-id 0 --n-shards 128 \\
      --output libraries/si_reference.shard000.npz
  python -m scripts.build_si_geometry_reference --merge-shards libraries/si_reference.shard*.npz \\
      --output data/geometry_reference_si_kappa_0p9.npz

This is CPU-only by design.  It is a once-per-geometry-definition stochastic
calculation, not part of a GPU library build.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi


KAPPA = 0.90
N_ALPHA = 26
N_SI_CELLS = 5_000_000
S7_COEFF = (-0.570444, 10.6047, -5.81457, 1.0)


def vi_from_x(x: np.ndarray | float) -> np.ndarray | float:
    a, b, c, d = S7_COEFF
    return ((a * x + b) * x + c) * x + d


def x_from_vi(vi: float) -> float:
    roots = np.roots([S7_COEFF[0], S7_COEFF[1], S7_COEFF[2], S7_COEFF[3] - vi])
    turn = min(
        root.real for root in np.roots([3 * S7_COEFF[0], 2 * S7_COEFF[1], S7_COEFF[2]])
        if abs(root.imag) < 1e-12 and root.real > 0.0
    )
    candidates = [root.real for root in roots if abs(root.imag) < 1e-12 and 0.0 <= root.real <= turn]
    if len(candidates) != 1:
        raise RuntimeError(f"no unique S7a physical root for v_i={vi}")
    return float(candidates[0])


def _central_cell_avs(rng: np.random.Generator, alphas: np.ndarray) -> np.ndarray:
    """One independently generated contracted rho=1 Poisson-Voronoi cell.

    The seed of interest is at the origin.  The enclosing Poisson cloud is
    expanded until its Voronoi region is bounded well inside the cloud.  The
    latter check prevents a finite sampling boundary from contributing a face
    to the reported convex hull.
    """
    extent = 2.5
    points = np.zeros((1, 3), dtype=np.float64)

    def add_poisson_shell(previous_extent: float, next_extent: float) -> np.ndarray:
        """Draw an independent homogeneous-Poisson cube shell exactly.

        Retaining the inner points while adding this shell matters: rejecting
        a complete finite cloud whenever its central cell is large would
        condition away precisely the large cells that contribute strongly to
        the untrimmed SI average.  Incremental shells construct one Poisson
        realization until the central cell is demonstrably insulated from
        every omitted seed.
        """
        shell_volume = (2.0 * next_extent) ** 3 - (2.0 * previous_extent) ** 3
        n = int(rng.poisson(shell_volume))
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)
        accepted: list[np.ndarray] = []
        remaining = n
        while remaining:
            proposal = rng.uniform(-next_extent, next_extent, size=(max(remaining * 2, 32), 3))
            shell = proposal[np.any(np.abs(proposal) > previous_extent, axis=1)]
            take = min(remaining, len(shell))
            if take:
                accepted.append(shell[:take])
                remaining -= take
        return np.vstack(accepted)

    for attempt in range(8):
        if attempt == 0:
            n = int(rng.poisson((2.0 * extent) ** 3))
            points = np.vstack((points, rng.uniform(-extent, extent, size=(n, 3))))
        else:
            next_extent = extent * 1.75
            points = np.vstack((points, add_poisson_shell(extent, next_extent)))
            extent = next_extent
        try:
            vor = Voronoi(points)
        except Exception:
            # Keep ``extent`` synchronized with the populated cube; the
            # next iteration appends the immediately adjacent Poisson shell.
            continue
        region = vor.regions[vor.point_region[0]]
        if not region or -1 in region:
            continue
        vertices = vor.vertices[np.asarray(region, dtype=int)]
        # Every omitted seed has norm >= extent.  If the whole present cell
        # lies inside the sphere of radius extent/2, no such seed can create
        # a nearer bisector.  The Euclidean, rather than coordinate-wise,
        # condition makes the finite-cloud certificate genuinely sufficient.
        if np.max(np.linalg.norm(vertices, axis=1)) >= 0.5 * extent:
            continue
        # A Voronoi ridge is exactly a bounding facet of the uncontracted
        # central cell.  Retaining every such neighbour below implements the
        # conjunction over all shifted facets in SI Eq. S2; selecting only
        # the second-nearest seed would be the inconsistent §S.IV.a shortcut.
        neighbours = []
        for i, j in vor.ridge_points:
            if i == 0:
                neighbours.append(int(j))
            elif j == 0:
                neighbours.append(int(i))
        if len(neighbours) < 4:
            extent *= 1.75
            continue
        neighbour_points = points[np.asarray(sorted(set(neighbours)), dtype=int)]
        d = np.linalg.norm(neighbour_points, axis=1)
        d_nn = float(np.min(d))
        values = np.empty(len(alphas), dtype=np.float64)
        for ai, alpha_star in enumerate(alphas):
            alpha_i = min(float(alpha_star), KAPPA * d_nn / 2.0)
            halfspaces = []
            for point, distance in zip(neighbour_points, d):
                normal = point / distance
                # n.x <= d/2-alpha_i; origin is strictly interior whenever
                # alpha_i < d_nn/2, guaranteed by kappa<1.
                halfspaces.append(np.r_[normal, -(distance / 2.0 - alpha_i)])
            try:
                intersections = HalfspaceIntersection(np.asarray(halfspaces), np.zeros(3), qhull_options="QJ").intersections
                hull = ConvexHull(intersections, qhull_options="QJ")
            except Exception:
                break
            if hull.volume <= 0.0 or hull.area <= 0.0:
                break
            values[ai] = hull.area / hull.volume
        else:
            return values
    raise RuntimeError("could not obtain a bounded single Poisson-Voronoi cell")


def build_shard(n_cells: int, shard_id: int, n_shards: int, seed: int, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if not (0 <= shard_id < n_shards):
        raise ValueError("shard-id must lie in [0, n-shards)")
    sums = np.zeros(len(alphas), dtype=np.float64)
    sums_sq = np.zeros(len(alphas), dtype=np.float64)
    count = 0
    for cell_index in range(shard_id, n_cells, n_shards):
        # Per-cell SeedSequence preserves independence and makes shards
        # reproducible regardless of array scheduling or chunk size.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(cell_index)]))
        value = _central_cell_avs(rng, alphas)
        sums += value
        sums_sq += value * value
        count += 1
        if count % 1000 == 0:
            print(f"  shard {shard_id}: {count:,} cells", flush=True)
    return sums, sums_sq, count


def _write_shard(path: Path, sums: np.ndarray, sums_sq: np.ndarray, count: int, alphas: np.ndarray, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "madi-si-single-cell-reference-shard-v1",
        "source": "MADI I Supporting Information §§S.II,S.IV",
        "rho_reference": 1.0,
        "kappa": KAPPA,
        "n_alpha": int(len(alphas)),
        "alpha_spacing": "linear",
        "mean_estimator": "untrimmed_arithmetic_mean_A_over_V",
        "contraction_rule": "all_shifted_voronoi_facets_S2",
        "n_single_cells_requested": int(args.n_single_cells),
        "shard_id": int(args.shard_id),
        "n_shards": int(args.n_shards),
        "seed": int(args.seed),
    }
    np.savez(path, alpha_x=alphas, sum_A_over_V=sums, sumsq_A_over_V=sums_sq,
             n_single_cells=np.array(count, dtype=np.int64), metadata_json=np.array(json.dumps(metadata, sort_keys=True)))


def merge(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("supply at least one shard to merge")
    alpha = None
    sums = sums_sq = None
    count = 0
    metadata = None
    seen_shards: set[int] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            item_alpha = np.asarray(data["alpha_x"], dtype=np.float64)
            item_sum = np.asarray(data["sum_A_over_V"], dtype=np.float64)
            item_sumsq = np.asarray(data["sumsq_A_over_V"], dtype=np.float64)
            item_count = int(data["n_single_cells"])
            item_metadata = json.loads(str(data["metadata_json"]))
        if alpha is None:
            alpha, sums, sums_sq, metadata = item_alpha, item_sum.copy(), item_sumsq.copy(), item_metadata
        else:
            if not np.array_equal(alpha, item_alpha):
                raise RuntimeError(f"incompatible alpha grid in {path}")
            for key in ("rho_reference", "kappa", "n_alpha", "alpha_spacing", "mean_estimator", "contraction_rule", "n_single_cells_requested", "n_shards", "seed"):
                if item_metadata.get(key) != metadata.get(key):
                    raise RuntimeError(f"incompatible {key} in {path}")
            sums += item_sum
            sums_sq += item_sumsq
        shard_id = int(item_metadata["shard_id"])
        if shard_id in seen_shards:
            raise RuntimeError(f"duplicate shard id {shard_id}")
        seen_shards.add(shard_id)
        count += item_count
    assert alpha is not None and sums is not None and sums_sq is not None and metadata is not None
    expected = int(metadata["n_single_cells_requested"])
    expected_shards = int(metadata["n_shards"])
    expected_ids = set(range(expected_shards))
    if count != expected or seen_shards != expected_ids:
        raise RuntimeError(
            f"incomplete reference cloud: {count:,}/{expected:,} cells; "
            f"shards={sorted(seen_shards)} expected={sorted(expected_ids)}"
        )
    vi = np.asarray(vi_from_x(alpha), dtype=np.float64)
    # Runtime interpolation requires ascending v_i while alpha decreases.
    order = np.argsort(vi)
    mean = sums / count
    variance = np.maximum(sums_sq / count - mean * mean, 0.0)
    output_metadata = {
        "schema": "madi-si-single-cell-reference-v1",
        "source": "MADI I Supporting Information §§S.II,S.IV",
        "rho_reference": 1.0,
        "kappa": KAPPA,
        "n_single_cells": count,
        "n_alpha": int(len(alpha)),
        "alpha_spacing": "linear",
        "mean_estimator": "untrimmed_arithmetic_mean_A_over_V",
        "contraction_rule": "all_shifted_voronoi_facets_S2",
        "alpha_range_x": [float(np.min(alpha)), float(np.max(alpha))],
        "vi_range": [float(np.min(vi)), float(np.max(vi))],
        "seed": int(metadata["seed"]),
        "shards": sorted(seen_shards),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        vi=vi[order], alpha_x=alpha[order], mean_A_over_V_norm=mean[order],
        se_A_over_V_norm=np.sqrt(variance[order] / count),
        metadata_json=np.array(json.dumps(output_metadata, sort_keys=True)),
    )
    print(f"Wrote certified SI geometry reference: {output} ({count:,} cells)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-single-cells", type=int, default=N_SI_CELLS)
    ap.add_argument("--n-alpha", type=int, default=N_ALPHA)
    ap.add_argument("--vi-min", type=float, default=0.40)
    ap.add_argument("--vi-max", type=float, default=1.00)
    ap.add_argument("--seed", type=int, default=20_260_803)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge-shards", nargs="*", default=None)
    args = ap.parse_args()
    output = Path(args.output)
    if args.merge_shards is not None:
        merge([Path(item) for item in args.merge_shards], output)
        return
    if args.n_alpha != N_ALPHA:
        raise ValueError(f"SI production reference requires exactly {N_ALPHA} alpha values")
    if not (0.0 < args.vi_min < args.vi_max <= 1.0):
        raise ValueError("require 0 < vi-min < vi-max <= 1")
    # The ρ=1 reference uses x=alpha*.  x=0 is a valid uncontracted cell.
    alpha_max = x_from_vi(args.vi_min)
    alphas = np.linspace(0.0, alpha_max, args.n_alpha, dtype=np.float64)
    sums, sums_sq, count = build_shard(
        args.n_single_cells, args.shard_id, args.n_shards, args.seed, alphas
    )
    _write_shard(output, sums, sums_sq, count, alphas, args)
    print(f"Wrote shard {args.shard_id}/{args.n_shards}: {count:,} cells -> {output}")


if __name__ == "__main__":
    main()

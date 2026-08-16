"""Exact SI random walkers for the MADI finite-domain geometry.

CPU and CUDA implement the same physical transition at every 1-µs step:

1. query the exact nearest seed at the actual endpoint position, then query
   every potentially binding shifted Voronoi facet within ``d_1+2*alpha_1``;
2. classify it with the full conjunction in SI Eq. S2;
3. infer ``m`` solely from the old/new endpoint labels (SI §S.IV.b.1);
4. accept a crossing iff ``u < p_p**m``; otherwise revert exactly; and
5. abort the entire simulation if any walker exits Ω_sim (SI §S.III).

The CUDA path traverses a balanced exact KD tree constructed from the same
seed array used by SciPy's CPU KD tree.  The former voxel-centre candidate
cache and all residence-time/direct-k_io calibration machinery are absent by
design: SI §S.IV defines ensemble-average k_io through Eq. 5 and explicitly
distinguishes it from inverse intracellular lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import PRODUCTION_AXIS_WALKS_PER_ENTRY, SimConfig
from .ensemble import Ensemble, create_dummy_ensemble, create_ensemble

try:
    from numba import cuda
    import numba
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_normal_float64,
        xoroshiro128p_uniform_float64,
    )
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False


# Retain the historical name for compatibility with the warning below, but
# bind it to the approved production allocation rather than a stale build.
PAPER_WALKS_PER_ENTRY = PRODUCTION_AXIS_WALKS_PER_ENTRY
_REDUCE_THREADS = 256


def pp_to_kio_eq5(pp: float, mean_AV: float, cfg: SimConfig) -> float:
    """MADI I Eq. 5, in s⁻¹, using the SI governing-process ``<A/V>``."""
    return float(pp) * math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * float(mean_AV) * 1000.0


def kio_to_pp(kio: float, mean_AV: float, cfg: SimConfig) -> float:
    """Analytic inverse of Eq. 5; raise rather than silently clamp p_p."""
    if kio < 0.0:
        raise ValueError(f"k_io must be non-negative, got {kio}")
    if mean_AV <= 0.0:
        raise ValueError("governing-process mean_A_over_V must be positive")
    pp = (float(kio) / 1000.0) / (math.sqrt(cfg.D0 / (6.0 * cfg.ts)) * mean_AV)
    return _checked_pp(pp)


def pp_to_kio(pp: float, mean_AV: float, cfg: SimConfig) -> float:
    """Backward-compatible name for the Eq. 5 mapping."""
    return pp_to_kio_eq5(pp, mean_AV, cfg)


def _checked_pp(pp: float) -> float:
    pp = float(pp)
    if not np.isfinite(pp) or pp < 0.0 or pp > 1.0:
        raise ValueError(f"p_p must be finite and in [0, 1], got {pp!r}")
    return pp


def _ensemble_seed(base_seed: int, ensemble_index: int) -> int:
    return (int(base_seed) + ensemble_index * 97_003) & 0x7FFFFFFF


def _walk_seed(base_seed: int, ensemble_index: int) -> int:
    """CRN seed shared across neighbouring rho/V/k_io entries."""
    return (int(base_seed) + ensemble_index * 104_729) & 0x7FFFFFFF


@dataclass(frozen=True)
class WalkRandomStream:
    """Pre-generated CPU/GPU input stream for the Sol golden-file harness."""

    initial_positions: np.ndarray      # (N,3), inside Ω_src
    increments: np.ndarray             # (n_steps,N,3), Gaussian [µm]
    acceptance_uniforms: np.ndarray    # (n_steps,N), U[0,1)

    def validate(self, cfg: SimConfig, ensemble: Optional[Ensemble] = None) -> None:
        n = int(cfg.n_walkers)
        if np.asarray(self.initial_positions).shape != (n, 3):
            raise ValueError("golden initial_positions must have shape (n_walkers, 3)")
        if np.asarray(self.increments).shape != (int(cfg.n_steps), n, 3):
            raise ValueError("golden increments must have shape (n_steps, n_walkers, 3)")
        if np.asarray(self.acceptance_uniforms).shape != (int(cfg.n_steps), n):
            raise ValueError("golden acceptance_uniforms must have shape (n_steps, n_walkers)")
        if not (np.all(np.isfinite(self.initial_positions))
                and np.all(np.isfinite(self.increments))
                and np.all(np.isfinite(self.acceptance_uniforms))):
            raise ValueError("golden stream contains non-finite values")
        if np.any(self.acceptance_uniforms < 0.0) or np.any(self.acceptance_uniforms >= 1.0):
            raise ValueError("golden acceptance uniforms lie outside [0,1)")
        if ensemble is not None and (
            np.any(self.initial_positions < ensemble.source_lo)
            or np.any(self.initial_positions >= ensemble.source_hi)
        ):
            raise ValueError("golden initial positions must lie in Ω_src")


def make_walk_random_stream(cfg: SimConfig, seed: int, ensemble: Optional[Ensemble] = None) -> WalkRandomStream:
    """Create a deterministic source-cube stream for CPU/GPU equivalence."""
    if ensemble is None:
        ensemble = create_dummy_ensemble(cfg)
    rng = np.random.default_rng(int(seed))
    n = int(cfg.n_walkers)
    stream = WalkRandomStream(
        initial_positions=rng.uniform(ensemble.source_lo, ensemble.source_hi, size=(n, 3)).astype(np.float64),
        increments=rng.normal(0.0, cfg.sigma, size=(int(cfg.n_steps), n, 3)).astype(np.float64),
        acceptance_uniforms=rng.uniform(0.0, 1.0, size=(int(cfg.n_steps), n)).astype(np.float64),
    )
    stream.validate(cfg, ensemble)
    return stream


@dataclass
class ReducedResult:
    cos_sum: np.ndarray
    sin_sum: np.ndarray
    n_walkers: int
    n_escaped: int
    occupancy_counts: np.ndarray
    pp_values: list[float] = field(default_factory=list)
    analytic_kio_eq5_values: list[float] = field(default_factory=list)
    geometry_stats: list[dict] = field(default_factory=list)
    # One real/imaginary mean vector per independently constructed ensemble,
    # in ensemble-index order.  These are kept only while an entry is being
    # assembled into the v5 library diagnostics; the production artifact does
    # not store them for every column.
    ensemble_cos_means: list[np.ndarray] = field(default_factory=list)
    ensemble_sin_means: list[np.ndarray] = field(default_factory=list)
    classifier_cache_stats: list[dict[str, int]] = field(default_factory=list)

    @property
    def n_eff(self) -> int:
        """Three Cartesian axes are isotropically harvested per walker."""
        return 3 * self.n_walkers

    @property
    def analytic_kio_eq5(self) -> float:
        return float(np.mean(self.analytic_kio_eq5_values)) if self.analytic_kio_eq5_values else 0.0

    @property
    def mean_pp(self) -> float:
        return float(np.mean(self.pp_values)) if self.pp_values else 0.0

    @property
    def occupancy_fraction(self) -> np.ndarray:
        if self.n_walkers <= 0:
            return np.zeros_like(self.occupancy_counts, dtype=np.float64)
        return self.occupancy_counts.astype(np.float64) / self.n_walkers


@dataclass
class ClassifierCacheStats:
    """Per-walk accounting for the exact classifier cache.

    The counters describe *endpoint classification requests*, including the
    initial endpoint.  They are diagnostic only: neither the physics nor the
    accept/reject stream depends on them.
    """

    full: int = 0
    tier1: int = 0
    tier2: int = 0
    candidate_overflow: int = 0
    full_inside: int = 0
    full_outside: int = 0
    tier1_inside: int = 0
    tier2_inside: int = 0
    tier2_outside: int = 0

    @property
    def requests(self) -> int:
        return self.full + self.tier1 + self.tier2

    def as_dict(self) -> dict[str, int]:
        return {
            "full": int(self.full),
            "tier1": int(self.tier1),
            "tier2": int(self.tier2),
            "candidate_overflow": int(self.candidate_overflow),
            "full_inside": int(self.full_inside),
            "full_outside": int(self.full_outside),
            "tier1_inside": int(self.tier1_inside),
            "tier2_inside": int(self.tier2_inside),
            "tier2_outside": int(self.tier2_outside),
            "requests": int(self.requests),
        }


def _membrane_transition_counts(
    old_cell: np.ndarray,
    old_inside: np.ndarray,
    new_cell: np.ndarray,
    new_inside: np.ndarray,
) -> np.ndarray:
    """SI §S.IV.b.1 endpoint-only membrane count (0, 1, or 2)."""
    m = np.zeros(len(old_cell), dtype=np.int8)
    m[old_inside & ~new_inside] = 1
    m[~old_inside & new_inside] = 1
    m[old_inside & new_inside & (old_cell != new_cell)] = 2
    return m


if HAS_CUDA:

    @cuda.jit(device=True)
    def _nearest_one_kdtree(
        px, py, pz, seeds, node_seed, node_axis, node_left, node_right, node_parent
    ):
        """Exact nearest-neighbour KD query without a fixed traversal stack.

        ``node_parent`` makes this a stackless depth-first traversal.  This
        avoids substituting a hardware-sized neighbour/cache approximation
        for the geometry, while retaining standard exact KD pruning.
        """
        node = 0
        previous = -1  # Root's parent is -1, so this is the initial entry.
        best_seed = -1
        best_dsq = math.inf
        while node >= 0:
            sid = node_seed[node]
            axis = node_axis[node]
            coordinate = px
            if axis == 1:
                coordinate = py
            elif axis == 2:
                coordinate = pz
            delta = coordinate - seeds[sid, axis]
            near = node_left[node] if delta <= 0.0 else node_right[node]
            far = node_right[node] if delta <= 0.0 else node_left[node]
            if previous == node_parent[node]:
                dx = px - seeds[sid, 0]
                dy = py - seeds[sid, 1]
                dz = pz - seeds[sid, 2]
                dsq = dx * dx + dy * dy + dz * dz
                if dsq < best_dsq:
                    best_dsq = dsq
                    best_seed = sid
                if near >= 0:
                    previous = node
                    node = near
                elif far >= 0 and delta * delta < best_dsq:
                    previous = node
                    node = far
                else:
                    previous = node
                    node = node_parent[node]
            elif previous == near and far >= 0 and delta * delta < best_dsq:
                previous = node
                node = far
            else:
                previous = node
                node = node_parent[node]
        return best_seed, best_dsq

    @cuda.jit(device=True)
    def _classify_full_facet(
        px, py, pz, seeds, annulus, node_seed, node_axis, node_left, node_right, node_parent
    ):
        """Exact SI Eq. S2 contracted-cell test using the adaptive radius bound.

        Let ``m_j=(d_j²-d_1²)/(2|c_1-c_j|)``.  A facet can bind only when
        ``m_j < alpha_1``.  Since ``|c_1-c_j| <= d_1+d_j``, rearrangement
        gives ``d_j < d_1+2*alpha_1``.  The stackless KD traversal below
        visits every seed in precisely that ball, rather than an arbitrary
        number of nearest neighbours.
        """
        s1, d1sq = _nearest_one_kdtree(
            px, py, pz, seeds, node_seed, node_axis, node_left, node_right, node_parent
        )
        if s1 < 0:
            return 0, False, 0
        alpha1 = annulus[s1]
        radius = math.sqrt(d1sq) + 2.0 * alpha1
        radius_sq = radius * radius
        node = 0
        previous = -1
        while node >= 0:
            sid = node_seed[node]
            axis = node_axis[node]
            coordinate = px
            if axis == 1:
                coordinate = py
            elif axis == 2:
                coordinate = pz
            delta = coordinate - seeds[sid, axis]
            near = node_left[node] if delta <= 0.0 else node_right[node]
            far = node_right[node] if delta <= 0.0 else node_left[node]
            if previous == node_parent[node]:
                dx = px - seeds[sid, 0]
                dy = py - seeds[sid, 1]
                dz = pz - seeds[sid, 2]
                djsq = dx * dx + dy * dy + dz * dz
                if sid != s1 and djsq <= radius_sq:
                    sx = seeds[sid, 0] - seeds[s1, 0]
                    sy = seeds[sid, 1] - seeds[s1, 1]
                    sz = seeds[sid, 2] - seeds[s1, 2]
                    separation = math.sqrt(sx * sx + sy * sy + sz * sz)
                    if separation <= 0.0:
                        return 0, False, 0
                    margin = (djsq - d1sq) / (2.0 * separation)
                    if margin < alpha1:
                        return s1, False, 1
                if near >= 0:
                    previous = node
                    node = near
                elif far >= 0 and delta * delta <= radius_sq:
                    previous = node
                    node = far
                else:
                    previous = node
                    node = node_parent[node]
            elif previous == near and far >= 0 and delta * delta <= radius_sq:
                previous = node
                node = far
            else:
                previous = node
                node = node_parent[node]
        return s1, True, 1

    @cuda.jit(device=True)
    def _refresh_full_facet_cache(
        px, py, pz, tid, seeds, annulus, max_annulus,
        node_seed, node_axis, node_left, node_right, node_parent,
        delta_max, cache_ref, cache_cell, cache_inside, cache_rsafe,
        cache_count, cache_valid, cache_ids, cache_stats,
    ):
        """Perform one unchanged exact classification and refresh both tiers.

        The exact SI Eq. S2 decision still uses ``d1 + 2*alpha1``.  The
        larger traversal is only to construct two *proof caches*: the Tier-1
        shifted-facet margin and a Tier-2 candidate superset.  A bounded
        candidate buffer is never truncated; an overflow disables Tier 2 and
        forces the next endpoint through this routine again.
        """
        s1, d1sq = _nearest_one_kdtree(
            px, py, pz, seeds, node_seed, node_axis, node_left, node_right, node_parent
        )
        if s1 < 0:
            return 0, False, 0
        d1 = math.sqrt(d1sq)
        alpha1 = annulus[s1]
        exact_radius = d1 + 2.0 * alpha1
        # r_safe is capped by alpha1.  A facet whose shifted margin is below
        # that cap must lie in this widened exact-radius search.
        safe_radius = d1 + 4.0 * alpha1
        candidate_radius = d1 + 2.0 * max_annulus + 2.0 * delta_max
        max_radius = safe_radius
        if candidate_radius > max_radius:
            max_radius = candidate_radius
        exact_radius_sq = exact_radius * exact_radius
        safe_radius_sq = safe_radius * safe_radius
        candidate_radius_sq = candidate_radius * candidate_radius
        max_radius_sq = max_radius * max_radius

        inside = 1
        safe_margin = alpha1
        candidate_total = 0
        overflow = 0
        node = 0
        previous = -1
        while node >= 0:
            sid = node_seed[node]
            axis = node_axis[node]
            coordinate = px
            if axis == 1:
                coordinate = py
            elif axis == 2:
                coordinate = pz
            split_delta = coordinate - seeds[sid, axis]
            near = node_left[node] if split_delta <= 0.0 else node_right[node]
            far = node_right[node] if split_delta <= 0.0 else node_left[node]
            if previous == node_parent[node]:
                dx = px - seeds[sid, 0]
                dy = py - seeds[sid, 1]
                dz = pz - seeds[sid, 2]
                djsq = dx * dx + dy * dy + dz * dz
                if djsq <= candidate_radius_sq:
                    if candidate_total < cache_ids.shape[1]:
                        cache_ids[tid, candidate_total] = sid
                    else:
                        overflow = 1
                    candidate_total += 1
                if sid != s1 and djsq <= safe_radius_sq:
                    sx = seeds[sid, 0] - seeds[s1, 0]
                    sy = seeds[sid, 1] - seeds[s1, 1]
                    sz = seeds[sid, 2] - seeds[s1, 2]
                    separation = math.sqrt(sx * sx + sy * sy + sz * sz)
                    if separation <= 0.0:
                        return 0, False, 0
                    margin = (djsq - d1sq) / (2.0 * separation)
                    if djsq <= exact_radius_sq and margin < alpha1:
                        inside = 0
                    shifted_margin = margin - alpha1
                    if shifted_margin < safe_margin:
                        safe_margin = shifted_margin
                if near >= 0:
                    previous = node
                    node = near
                elif far >= 0 and split_delta * split_delta <= max_radius_sq:
                    previous = node
                    node = far
                else:
                    previous = node
                    node = node_parent[node]
            elif previous == near and far >= 0 and split_delta * split_delta <= max_radius_sq:
                previous = node
                node = far
            else:
                previous = node
                node = node_parent[node]

        cache_ref[tid, 0] = px
        cache_ref[tid, 1] = py
        cache_ref[tid, 2] = pz
        cache_cell[tid] = s1
        cache_inside[tid] = inside
        if inside == 1 and safe_margin > 0.0:
            cache_rsafe[tid] = safe_margin if safe_margin < alpha1 else alpha1
        else:
            cache_rsafe[tid] = 0.0
        if overflow == 1:
            cache_count[tid] = 0
            cache_valid[tid] = 0
            cache_stats[tid, 3] += 1
        else:
            cache_count[tid] = candidate_total
            cache_valid[tid] = 1
        cache_stats[tid, 0] += 1
        if inside == 1:
            cache_stats[tid, 4] += 1
        else:
            cache_stats[tid, 5] += 1
        return s1, inside == 1, 1

    @cuda.jit(device=True)
    def _classify_from_full_facet_cache(
        px, py, pz, tid, seeds, annulus, delta_max, min_safe_radius,
        cache_ref, cache_cell, cache_inside, cache_rsafe, cache_count,
        cache_valid, cache_ids, cache_stats,
    ):
        """Use a proof cache, returning ``ok=0`` when an exact refresh is due."""
        dx_ref = px - cache_ref[tid, 0]
        dy_ref = py - cache_ref[tid, 1]
        dz_ref = pz - cache_ref[tid, 2]
        displacement_sq = dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref
        r_safe = cache_rsafe[tid]
        if (cache_inside[tid] == 1 and r_safe >= min_safe_radius
                and displacement_sq < r_safe * r_safe):
            cache_stats[tid, 1] += 1
            cache_stats[tid, 6] += 1
            return cache_cell[tid], True, 1

        if cache_valid[tid] == 0 or displacement_sq >= delta_max * delta_max:
            return 0, False, 0
        count = cache_count[tid]
        if count <= 0:
            return 0, False, 0

        s1 = cache_ids[tid, 0]
        dx = px - seeds[s1, 0]
        dy = py - seeds[s1, 1]
        dz = pz - seeds[s1, 2]
        d1sq = dx * dx + dy * dy + dz * dz
        for index in range(1, count):
            sid = cache_ids[tid, index]
            dx = px - seeds[sid, 0]
            dy = py - seeds[sid, 1]
            dz = pz - seeds[sid, 2]
            djsq = dx * dx + dy * dy + dz * dz
            if djsq < d1sq:
                s1 = sid
                d1sq = djsq
        alpha1 = annulus[s1]
        radius = math.sqrt(d1sq) + 2.0 * alpha1
        radius_sq = radius * radius
        inside = 1
        for index in range(count):
            sid = cache_ids[tid, index]
            if sid != s1:
                dx = px - seeds[sid, 0]
                dy = py - seeds[sid, 1]
                dz = pz - seeds[sid, 2]
                djsq = dx * dx + dy * dy + dz * dz
                if djsq <= radius_sq:
                    sx = seeds[sid, 0] - seeds[s1, 0]
                    sy = seeds[sid, 1] - seeds[s1, 1]
                    sz = seeds[sid, 2] - seeds[s1, 2]
                    separation = math.sqrt(sx * sx + sy * sy + sz * sz)
                    if separation <= 0.0:
                        # Treat this as a cache miss so the unchanged exact
                        # classifier owns the fatal-geometry decision.
                        return 0, False, 0
                    margin = (djsq - d1sq) / (2.0 * separation)
                    if margin < alpha1:
                        inside = 0
        cache_stats[tid, 2] += 1
        if inside == 1:
            cache_stats[tid, 7] += 1
        else:
            cache_stats[tid, 8] += 1
        return s1, inside == 1, 1

    @cuda.jit
    def _walk_kernel(
        seeds, annulus, node_seed, node_axis, node_left, node_right, node_parent,
        L, source_lo, source_hi, sigma, ts, n_steps, pp, has_cells,
        half_L, steps_per_h, rng_states, use_golden_stream,
        golden_initial_positions, golden_increments, golden_acceptance_uniforms,
        use_classifier_cache, max_annulus, cache_delta_max, cache_min_safe_radius,
        cache_ref, cache_cell, cache_inside, cache_rsafe, cache_count, cache_valid,
        cache_ids, cache_stats,
        Y_out, inside_trace, escaped_out, classifier_error_out,
    ):
        tid = cuda.grid(1)
        if tid >= Y_out.shape[0]:
            return
        if use_golden_stream == 1:
            px = golden_initial_positions[tid, 0]
            py = golden_initial_positions[tid, 1]
            pz = golden_initial_positions[tid, 2]
        else:
            px = source_lo + xoroshiro128p_uniform_float64(rng_states, tid) * (source_hi - source_lo)
            py = source_lo + xoroshiro128p_uniform_float64(rng_states, tid) * (source_hi - source_lo)
            pz = source_lo + xoroshiro128p_uniform_float64(rng_states, tid) * (source_hi - source_lo)
        if has_cells == 1:
            if use_classifier_cache == 1:
                cur_cell, cur_inside, ok = _refresh_full_facet_cache(
                    px, py, pz, tid, seeds, annulus, max_annulus,
                    node_seed, node_axis, node_left, node_right, node_parent,
                    cache_delta_max, cache_ref, cache_cell, cache_inside, cache_rsafe,
                    cache_count, cache_valid, cache_ids, cache_stats,
                )
            else:
                cur_cell, cur_inside, ok = _classify_full_facet(
                    px, py, pz, seeds, annulus, node_seed, node_axis, node_left, node_right, node_parent
                )
            if ok == 0:
                classifier_error_out[tid] = 1
                escaped_out[tid] = 1
                return
        else:
            cur_cell = 0
            cur_inside = False

        xprev = px - half_L
        yprev = py - half_L
        zprev = pz - half_L
        Yx = 0.0
        Yy = 0.0
        Yz = 0.0
        Y_out[tid, 0, 0] = 0.0
        Y_out[tid, 0, 1] = 0.0
        Y_out[tid, 0, 2] = 0.0
        inside_trace[tid, 0] = 1 if cur_inside else 0
        escaped = 0

        for step in range(n_steps):
            if escaped == 1:
                continue
            if use_golden_stream == 1:
                dx = golden_increments[step, tid, 0]
                dy = golden_increments[step, tid, 1]
                dz = golden_increments[step, tid, 2]
                u = golden_acceptance_uniforms[step, tid]
            else:
                dx = xoroshiro128p_normal_float64(rng_states, tid) * sigma
                dy = xoroshiro128p_normal_float64(rng_states, tid) * sigma
                dz = xoroshiro128p_normal_float64(rng_states, tid) * sigma
                # Consume one uniform per step in both CPU and CUDA paths so
                # common random numbers remain aligned across entries.
                u = xoroshiro128p_uniform_float64(rng_states, tid)
            nx = px + dx
            ny = py + dy
            nz = pz + dz
            if nx < 0.0 or nx >= L or ny < 0.0 or ny >= L or nz < 0.0 or nz >= L:
                escaped = 1
                continue
            if has_cells == 1:
                if use_classifier_cache == 1:
                    new_cell, new_inside, ok = _classify_from_full_facet_cache(
                        nx, ny, nz, tid, seeds, annulus, cache_delta_max,
                        cache_min_safe_radius, cache_ref, cache_cell, cache_inside,
                        cache_rsafe, cache_count, cache_valid, cache_ids, cache_stats,
                    )
                    if ok == 0:
                        new_cell, new_inside, ok = _refresh_full_facet_cache(
                            nx, ny, nz, tid, seeds, annulus, max_annulus,
                            node_seed, node_axis, node_left, node_right, node_parent,
                            cache_delta_max, cache_ref, cache_cell, cache_inside, cache_rsafe,
                            cache_count, cache_valid, cache_ids, cache_stats,
                        )
                else:
                    new_cell, new_inside, ok = _classify_full_facet(
                        nx, ny, nz, seeds, annulus, node_seed, node_axis, node_left, node_right, node_parent
                    )
                if ok == 0:
                    classifier_error_out[tid] = 1
                    escaped = 1
                    continue
            else:
                new_cell = 0
                new_inside = False
            m = 0
            if cur_inside and not new_inside:
                m = 1
            elif (not cur_inside) and new_inside:
                m = 1
            elif cur_inside and new_inside and cur_cell != new_cell:
                m = 2
            if m > 0 and pp < 1.0:
                threshold = pp if m == 1 else pp * pp
                if u >= threshold:
                    nx = px
                    ny = py
                    nz = pz
                    new_cell = cur_cell
                    new_inside = cur_inside
            px = nx
            py = ny
            pz = nz
            cur_cell = new_cell
            cur_inside = new_inside
            xnow = px - half_L
            ynow = py - half_L
            znow = pz - half_L
            Yx += 0.5 * (xprev + xnow) * ts
            Yy += 0.5 * (yprev + ynow) * ts
            Yz += 0.5 * (zprev + znow) * ts
            xprev = xnow
            yprev = ynow
            zprev = znow
            step_idx = step + 1
            if step_idx % steps_per_h == 0:
                j = step_idx // steps_per_h
                Y_out[tid, j, 0] = Yx
                Y_out[tid, j, 1] = Yy
                Y_out[tid, j, 2] = Yz
                inside_trace[tid, j] = 1 if cur_inside else 0
        escaped_out[tid] = escaped

    @cuda.jit
    def _reduce_kernel(Y, j_delta, j_Delta, j_sum, phase_coef, cos_sum_out, sin_sum_out):
        col = cuda.blockIdx.x
        if col >= phase_coef.shape[0]:
            return
        tid = cuda.threadIdx.x
        block_size = cuda.blockDim.x
        jd = j_delta[col]
        jD = j_Delta[col]
        js = j_sum[col]
        pc = phase_coef[col]
        n_total = Y.shape[0] * 3
        local_cos = 0.0
        local_sin = 0.0
        i = tid
        while i < n_total:
            w = i // 3
            axis = i % 3
            dM = Y[w, jd, axis] + Y[w, jD, axis] - Y[w, js, axis]
            phase = pc * dM
            local_cos += math.cos(phase)
            local_sin += math.sin(phase)
            i += block_size
        shared_cos = cuda.shared.array(_REDUCE_THREADS, dtype=numba.float64)
        shared_sin = cuda.shared.array(_REDUCE_THREADS, dtype=numba.float64)
        shared_cos[tid] = local_cos
        shared_sin[tid] = local_sin
        cuda.syncthreads()
        stride = block_size // 2
        while stride > 0:
            if tid < stride:
                shared_cos[tid] += shared_cos[tid + stride]
                shared_sin[tid] += shared_sin[tid + stride]
            cuda.syncthreads()
            stride //= 2
        if tid == 0:
            cos_sum_out[col] = shared_cos[0]
            sin_sum_out[col] = shared_sin[0]


def _walk_cpu(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    seed: int = 0,
    random_stream: Optional[WalkRandomStream] = None,
    snapshot_steps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], ClassifierCacheStats]:
    """CPU transliteration of the exact SI endpoint transition."""
    pp = _checked_pp(pp)
    if random_stream is not None:
        random_stream.validate(cfg, ens)
    rng = np.random.default_rng(seed) if random_stream is None else None
    n = int(cfg.n_walkers)
    if random_stream is None:
        positions = rng.uniform(ens.source_lo, ens.source_hi, size=(n, 3))
    else:
        positions = np.asarray(random_stream.initial_positions, dtype=np.float64).copy()
    initial_positions = positions.copy()
    cache_stats = ClassifierCacheStats()
    use_cache = cfg.classifier_mode == "exact_cached" and not ens.is_free_water
    caches = None
    if ens.is_free_water:
        cur_cell = np.zeros(n, dtype=np.int32)
        cur_inside = np.zeros(n, dtype=bool)
    elif use_cache:
        caches = []
        cur_cell = np.empty(n, dtype=np.int32)
        cur_inside = np.empty(n, dtype=bool)
        for index in range(n):
            cache = ens.build_full_facet_cache_cpu(
                positions[index],
                delta_max_um=cfg.classifier_cache_delta_max_um,
                candidate_capacity=cfg.classifier_cache_candidate_capacity,
            )
            caches.append(cache)
            cur_cell[index] = cache.cell
            cur_inside[index] = cache.inside
            cache_stats.full += 1
            cache_stats.candidate_overflow += int(cache.candidate_overflow)
            if cache.inside:
                cache_stats.full_inside += 1
            else:
                cache_stats.full_outside += 1
    else:
        cur_cell, cur_inside = ens.classify_cpu(positions)

    steps = np.empty(0, dtype=np.int32) if snapshot_steps is None else np.asarray(snapshot_steps, dtype=np.int32)
    if steps.ndim != 1 or (len(steps) and np.any(np.diff(steps) <= 0)):
        raise ValueError("snapshot_steps must be a strictly increasing 1-D array")
    if len(steps) and (np.any(steps < 0) or np.any(steps > cfg.n_steps)):
        raise ValueError("snapshot_steps lies outside the random-walk duration")
    snapshots = np.zeros((n, len(steps), 3), dtype=np.float64) if len(steps) else None
    snapshot_cursor = 0
    if len(steps) and steps[0] == 0:
        snapshot_cursor = 1

    Y = np.zeros((n, cfg.n_grid, 3), dtype=np.float64)
    inside_trace = np.zeros((n, cfg.n_grid), dtype=np.int8)
    inside_trace[:, 0] = cur_inside.astype(np.int8)
    escaped = np.zeros(n, dtype=bool)
    previous = positions - ens.L / 2.0
    Y_running = np.zeros((n, 3), dtype=np.float64)

    for step in range(cfg.n_steps):
        active = ~escaped
        if not np.any(active):
            break
        if random_stream is None:
            increments = rng.normal(0.0, cfg.sigma, size=(n, 3))
            acceptance = rng.uniform(0.0, 1.0, size=n)
        else:
            increments = np.asarray(random_stream.increments[step], dtype=np.float64)
            acceptance = np.asarray(random_stream.acceptance_uniforms[step], dtype=np.float64)
        proposal = positions + increments
        out = ((proposal < 0.0).any(axis=1) | (proposal >= ens.L).any(axis=1)) & active
        escaped[out] = True
        proposal[out] = positions[out]
        active = ~escaped
        if ens.is_free_water:
            new_cell = np.zeros(n, dtype=np.int32)
            new_inside = np.zeros(n, dtype=bool)
        elif use_cache:
            assert caches is not None
            new_cell = cur_cell.copy()
            new_inside = cur_inside.copy()
            for index in np.flatnonzero(active):
                cached = ens.classify_from_full_facet_cache_cpu(
                    proposal[index], caches[index],
                    min_safe_radius_um=cfg.classifier_cache_min_safe_radius_um,
                )
                if cached is None:
                    cache = ens.build_full_facet_cache_cpu(
                        proposal[index],
                        delta_max_um=cfg.classifier_cache_delta_max_um,
                        candidate_capacity=cfg.classifier_cache_candidate_capacity,
                    )
                    caches[index] = cache
                    new_cell[index] = cache.cell
                    new_inside[index] = cache.inside
                    cache_stats.full += 1
                    cache_stats.candidate_overflow += int(cache.candidate_overflow)
                    if cache.inside:
                        cache_stats.full_inside += 1
                    else:
                        cache_stats.full_outside += 1
                else:
                    new_cell[index], new_inside[index], tier = cached
                    if tier == "tier1":
                        cache_stats.tier1 += 1
                        cache_stats.tier1_inside += 1
                    else:
                        cache_stats.tier2 += 1
                        if new_inside[index]:
                            cache_stats.tier2_inside += 1
                        else:
                            cache_stats.tier2_outside += 1
        else:
            new_cell, new_inside = ens.classify_cpu(proposal)
        m = _membrane_transition_counts(cur_cell, cur_inside, new_cell, new_inside)
        crossing = active & (m > 0)
        if np.any(crossing) and pp < 1.0:
            rejected = np.flatnonzero(crossing)[acceptance[crossing] >= pp ** m[crossing]]
            proposal[rejected] = positions[rejected]
            new_cell[rejected] = cur_cell[rejected]
            new_inside[rejected] = cur_inside[rejected]
        positions[active] = proposal[active]
        cur_cell[active] = new_cell[active]
        cur_inside[active] = new_inside[active]
        current = positions - ens.L / 2.0
        Y_running[active] += 0.5 * (previous[active] + current[active]) * cfg.ts
        previous[active] = current[active]
        step_idx = step + 1
        while snapshot_cursor < len(steps) and steps[snapshot_cursor] == step_idx:
            snapshots[:, snapshot_cursor, :] = positions - initial_positions
            snapshot_cursor += 1
        if step_idx % cfg.steps_per_h == 0:
            j = step_idx // cfg.steps_per_h
            Y[:, j, :] = Y_running
            inside_trace[:, j] = cur_inside.astype(np.int8)
    if snapshot_cursor != len(steps):
        raise RuntimeError("walk ended before a requested narrow-pulse snapshot")
    return Y, escaped.astype(np.int8), inside_trace, snapshots, cache_stats


def _run_walk_one_ensemble(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    seed: int,
    verbose: bool,
    random_stream: Optional[WalkRandomStream] = None,
    use_gpu: Optional[bool] = None,
    snapshot_steps: Optional[np.ndarray] = None,
) -> Tuple[object, np.ndarray, np.ndarray, Optional[np.ndarray], bool, ClassifierCacheStats]:
    """Run one walk; narrow snapshots deliberately select the CPU diagnostic."""
    _ = verbose
    pp = _checked_pp(pp)
    if random_stream is not None:
        random_stream.validate(cfg, ens)
    if use_gpu is True and not HAS_CUDA:
        raise RuntimeError("GPU golden run requested but CUDA is unavailable")
    has_snapshots = snapshot_steps is not None and len(snapshot_steps) > 0
    if use_gpu is True and has_snapshots:
        raise RuntimeError("narrow-pulse endpoint snapshots are CPU-only diagnostics")
    run_on_gpu = HAS_CUDA if use_gpu is None else bool(use_gpu)
    if not run_on_gpu or has_snapshots:
        Y, escaped, inside, snapshots, cache_stats = _walk_cpu(
            ens, pp, cfg, seed, random_stream=random_stream, snapshot_steps=snapshot_steps
        )
        return Y, escaped, inside, snapshots, False, cache_stats

    n = int(cfg.n_walkers)
    d_seeds = cuda.to_device(ens.seeds)
    d_annulus = cuda.to_device(ens.annulus)
    d_node_seed = cuda.to_device(ens.kd_node_seed)
    d_node_axis = cuda.to_device(ens.kd_node_axis)
    d_node_left = cuda.to_device(ens.kd_node_left)
    d_node_right = cuda.to_device(ens.kd_node_right)
    d_node_parent = cuda.to_device(ens.kd_node_parent)
    d_Y = cuda.device_array((n, cfg.n_grid, 3), dtype=np.float64)
    d_inside = cuda.device_array((n, cfg.n_grid), dtype=np.int8)
    d_escape = cuda.device_array(n, dtype=np.int8)
    # The kernel raises this one-way status flag only on a true classifier
    # failure.  It must begin at zero; an uninitialised device allocation can
    # otherwise turn a valid full-facet walk into a spurious fatal error.
    d_classifier_error = cuda.to_device(np.zeros(n, dtype=np.int8))
    use_classifier_cache = np.int32(
        1 if (cfg.classifier_mode == "exact_cached" and not ens.is_free_water) else 0
    )
    if use_classifier_cache:
        d_cache_ref = cuda.device_array((n, 3), dtype=np.float64)
        d_cache_cell = cuda.device_array(n, dtype=np.int32)
        d_cache_inside = cuda.device_array(n, dtype=np.int8)
        d_cache_rsafe = cuda.device_array(n, dtype=np.float64)
        d_cache_count = cuda.device_array(n, dtype=np.int32)
        d_cache_valid = cuda.device_array(n, dtype=np.int8)
        d_cache_ids = cuda.device_array(
            (n, int(cfg.classifier_cache_candidate_capacity)), dtype=np.int32
        )
        d_cache_stats = cuda.to_device(np.zeros((n, 9), dtype=np.int64))
    else:
        # The exact-reference path does not read these arrays.  Small valid
        # allocations keep the single CUDA kernel signature simple and avoid
        # a second source of truth for the walk transition.
        d_cache_ref = cuda.device_array((1, 3), dtype=np.float64)
        d_cache_cell = cuda.device_array(1, dtype=np.int32)
        d_cache_inside = cuda.device_array(1, dtype=np.int8)
        d_cache_rsafe = cuda.device_array(1, dtype=np.float64)
        d_cache_count = cuda.device_array(1, dtype=np.int32)
        d_cache_valid = cuda.device_array(1, dtype=np.int8)
        d_cache_ids = cuda.device_array((1, 2), dtype=np.int32)
        d_cache_stats = cuda.to_device(np.zeros((1, 9), dtype=np.int64))
    states = create_xoroshiro128p_states(n, seed=seed)
    if random_stream is None:
        d_initial = cuda.to_device(np.zeros((1, 3), dtype=np.float64))
        d_increments = cuda.to_device(np.zeros((1, 1, 3), dtype=np.float64))
        d_uniforms = cuda.to_device(np.zeros((1, 1), dtype=np.float64))
        use_golden = np.int32(0)
    else:
        d_initial = cuda.to_device(np.ascontiguousarray(random_stream.initial_positions, dtype=np.float64))
        d_increments = cuda.to_device(np.ascontiguousarray(random_stream.increments, dtype=np.float64))
        d_uniforms = cuda.to_device(np.ascontiguousarray(random_stream.acceptance_uniforms, dtype=np.float64))
        use_golden = np.int32(1)
    threads = 128
    blocks = (n + threads - 1) // threads
    _walk_kernel[blocks, threads](
        d_seeds, d_annulus, d_node_seed, d_node_axis, d_node_left, d_node_right, d_node_parent,
        np.float64(ens.L), np.float64(ens.source_lo), np.float64(ens.source_hi),
        np.float64(cfg.sigma), np.float64(cfg.ts), np.int32(cfg.n_steps), np.float64(pp),
        np.int32(0 if ens.is_free_water else 1), np.float64(ens.L / 2.0),
        np.int32(cfg.steps_per_h), states, use_golden, d_initial, d_increments, d_uniforms,
        use_classifier_cache, np.float64(np.max(ens.annulus)),
        np.float64(cfg.classifier_cache_delta_max_um),
        np.float64(cfg.classifier_cache_min_safe_radius_um),
        d_cache_ref, d_cache_cell, d_cache_inside, d_cache_rsafe, d_cache_count,
        d_cache_valid, d_cache_ids, d_cache_stats,
        d_Y, d_inside, d_escape, d_classifier_error,
    )
    cuda.synchronize()
    classifier_error = d_classifier_error.copy_to_host()
    if np.any(classifier_error):
        raise RuntimeError("CUDA full-facet KD-tree classification failed")
    cache_counts = d_cache_stats.copy_to_host().sum(axis=0)
    return (
        d_Y, d_escape.copy_to_host(), d_inside.copy_to_host(), None, True,
        ClassifierCacheStats(
            full=int(cache_counts[0]), tier1=int(cache_counts[1]),
            tier2=int(cache_counts[2]), candidate_overflow=int(cache_counts[3]),
            full_inside=int(cache_counts[4]), full_outside=int(cache_counts[5]),
            tier1_inside=int(cache_counts[6]), tier2_inside=int(cache_counts[7]),
            tier2_outside=int(cache_counts[8]),
        ),
    )


def _reduce_cpu(
    Y: np.ndarray,
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cos_sum = np.empty(len(phase_coef), dtype=np.float64)
    sin_sum = np.empty(len(phase_coef), dtype=np.float64)
    for c in range(len(phase_coef)):
        dM = Y[:, j_delta[c], :] + Y[:, j_Delta[c], :] - Y[:, j_sum[c], :]
        phase = phase_coef[c] * dM
        cos_sum[c] = np.cos(phase).sum()
        sin_sum[c] = np.sin(phase).sum()
    return cos_sum, sin_sum


def _reduce_narrow_cpu(
    snapshots: np.ndarray,
    snapshot_index: np.ndarray,
    phase_coef: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cos_sum = np.empty(len(phase_coef), dtype=np.float64)
    sin_sum = np.empty(len(phase_coef), dtype=np.float64)
    for c in range(len(phase_coef)):
        phase = phase_coef[c] * snapshots[:, snapshot_index[c], :]
        cos_sum[c] = np.cos(phase).sum()
        sin_sum[c] = np.sin(phase).sum()
    return cos_sum, sin_sum


def _with_n_walkers(cfg: SimConfig, n_walkers: int) -> SimConfig:
    return replace(cfg, n_walkers=int(n_walkers))


def _walk_and_reduce_one_ensemble(
    ens: Ensemble,
    pp: float,
    cfg: SimConfig,
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
    walk_seed: int,
    *,
    narrow_snapshot_steps: Optional[np.ndarray] = None,
    narrow_snapshot_index: Optional[np.ndarray] = None,
    narrow_phase_coef: Optional[np.ndarray] = None,
) -> ReducedResult:
    narrow_args = (narrow_snapshot_steps, narrow_snapshot_index, narrow_phase_coef)
    if any(x is None for x in narrow_args) and not all(x is None for x in narrow_args):
        raise ValueError("narrow-pulse reduction requires all endpoint arrays together")
    narrow = all(x is not None for x in narrow_args)
    if narrow and cfg.phase_model != "narrow_pulse":
        raise ValueError("narrow-pulse arrays require cfg.phase_model='narrow_pulse'")
    n_cols = len(phase_coef)
    cos_total = np.zeros(n_cols, dtype=np.float64)
    sin_total = np.zeros(n_cols, dtype=np.float64)
    occupancy = np.zeros(cfg.n_grid, dtype=np.int64)
    total = 0
    n_escape = 0
    chunk = int(cfg.walker_chunk or cfg.n_walkers)
    for offset in range(0, int(cfg.n_walkers), chunk):
        n_this = min(chunk, int(cfg.n_walkers) - offset)
        sub_cfg = cfg if n_this == cfg.n_walkers else _with_n_walkers(cfg, n_this)
        Y, escaped, inside, snapshots, device, cache_stats = _run_walk_one_ensemble(
            ens, pp, sub_cfg, walk_seed + offset, verbose=False,
            snapshot_steps=narrow_snapshot_steps if narrow else None,
        )
        escaped_here = int(np.sum(escaped))
        if escaped_here:
            raise RuntimeError(
                "SI §S.III fatal boundary condition: "
                f"{escaped_here}/{len(escaped)} walkers exited Ω_sim."
            )
        if device:
            if narrow:
                raise RuntimeError("narrow-pulse diagnostic unexpectedly selected CUDA")
            d_jd = cuda.to_device(np.asarray(j_delta, dtype=np.int32))
            d_jD = cuda.to_device(np.asarray(j_Delta, dtype=np.int32))
            d_js = cuda.to_device(np.asarray(j_sum, dtype=np.int32))
            d_pc = cuda.to_device(np.asarray(phase_coef, dtype=np.float64))
            d_cos = cuda.device_array(n_cols, dtype=np.float64)
            d_sin = cuda.device_array(n_cols, dtype=np.float64)
            _reduce_kernel[n_cols, _REDUCE_THREADS](Y, d_jd, d_jD, d_js, d_pc, d_cos, d_sin)
            cuda.synchronize()
            cos_total += d_cos.copy_to_host()
            sin_total += d_sin.copy_to_host()
        elif narrow:
            if snapshots is None:
                raise RuntimeError("narrow-pulse CPU walk returned no endpoint snapshots")
            c, s = _reduce_narrow_cpu(snapshots, narrow_snapshot_index, narrow_phase_coef)
            cos_total += c
            sin_total += s
        else:
            c, s = _reduce_cpu(Y, j_delta, j_Delta, j_sum, phase_coef)
            cos_total += c
            sin_total += s
        total += len(escaped)
        n_escape += escaped_here
        occupancy += np.sum(inside, axis=0, dtype=np.int64)
    return ReducedResult(
        cos_sum=cos_total, sin_sum=sin_total, n_walkers=total, n_escaped=n_escape,
        occupancy_counts=occupancy, pp_values=[float(pp)],
        analytic_kio_eq5_values=[pp_to_kio_eq5(pp, ens.mean_AV, cfg)] if not ens.is_free_water else [0.0],
        geometry_stats=[ens.geometry.to_dict()],
        ensemble_cos_means=[cos_total / float(3 * total)],
        ensemble_sin_means=[sin_total / float(3 * total)],
        classifier_cache_stats=[cache_stats.as_dict()],
    )


def _merge_results(results: Sequence[ReducedResult]) -> ReducedResult:
    if not results:
        raise ValueError("cannot merge no results")
    merged = ReducedResult(
        cos_sum=np.zeros_like(results[0].cos_sum), sin_sum=np.zeros_like(results[0].sin_sum),
        n_walkers=0, n_escaped=0,
        occupancy_counts=np.zeros_like(results[0].occupancy_counts),
    )
    for result in results:
        merged.cos_sum += result.cos_sum
        merged.sin_sum += result.sin_sum
        merged.n_walkers += result.n_walkers
        merged.n_escaped += result.n_escaped
        merged.occupancy_counts += result.occupancy_counts
        merged.pp_values.extend(result.pp_values)
        merged.analytic_kio_eq5_values.extend(result.analytic_kio_eq5_values)
        merged.geometry_stats.extend(result.geometry_stats)
        # ``results`` is ordered by ensemble_index in
        # run_simulation_multi_kio_reduced.  Do not sort, aggregate, or
        # otherwise reorder these arrays: v5 uses this order as the
        # common-random-number partner identity across library entries.
        merged.ensemble_cos_means.extend(result.ensemble_cos_means)
        merged.ensemble_sin_means.extend(result.ensemble_sin_means)
        merged.classifier_cache_stats.extend(result.classifier_cache_stats)
    return merged


def run_walk_Y(
    ens: Ensemble,
    kio: float,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
    *,
    pp: Optional[float] = None,
    return_telemetry: bool = False,
    classifier: str = "exact",
    random_stream: Optional[WalkRandomStream] = None,
    use_gpu: Optional[bool] = None,
):
    """Validation API for an exact walk and its microstep Y(t) integral.

    ``exact_cached`` is an equivalence test mode: its only shortcuts are the
    conservative cache proofs, and every other endpoint calls the reference
    full-facet classifier.
    """
    if cfg is None:
        cfg = SimConfig()
    if classifier not in {"exact", "exact_cached"}:
        raise ValueError("classifier must be 'exact' or 'exact_cached'")
    if classifier != cfg.classifier_mode:
        cfg = replace(cfg, classifier_mode=classifier)
    actual_pp = _checked_pp(pp if pp is not None else kio_to_pp(kio, ens.mean_AV, cfg))
    Y, escaped, inside, _, device, cache_stats = _run_walk_one_ensemble(
        ens, actual_pp, cfg, seed, verbose, random_stream=random_stream, use_gpu=use_gpu
    )
    Y_host = Y.copy_to_host() if device else Y
    n_escape = int(np.sum(escaped))
    if n_escape:
        raise RuntimeError(f"SI §S.III fatal boundary condition: {n_escape} walkers escaped Ω_sim")
    if return_telemetry:
        return Y_host, n_escape, {
            "occupancy_counts": np.sum(inside, axis=0),
            "occupancy_fraction": np.mean(inside, axis=0),
            "pp": actual_pp,
            "classifier_cache": cache_stats.as_dict(),
        }
    return Y_host, n_escape


def _build_ensembles_for_entry(rho: float, V: float, cfg: SimConfig, seed: int, verbose: bool) -> list[Ensemble]:
    if rho <= 0.0 or V <= 0.0:
        return [create_dummy_ensemble(cfg) for _ in range(int(cfg.n_ensembles))]
    return [
        create_ensemble(rho, V, cfg, seed=_ensemble_seed(seed, i), verbose=verbose)
        for i in range(int(cfg.n_ensembles))
    ]


def run_simulation_reduced(
    rho: float,
    V: float,
    kio: float,
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
    *,
    narrow_snapshot_steps: Optional[np.ndarray] = None,
    narrow_snapshot_index: Optional[np.ndarray] = None,
    narrow_phase_coef: Optional[np.ndarray] = None,
) -> ReducedResult:
    """Simulate one Eq.-5-labelled entry."""
    outputs = run_simulation_multi_kio_reduced(
        rho, V, [kio], j_delta, j_Delta, j_sum, phase_coef, cfg=cfg, seed=seed,
        verbose=verbose, narrow_snapshot_steps=narrow_snapshot_steps,
        narrow_snapshot_index=narrow_snapshot_index, narrow_phase_coef=narrow_phase_coef,
    )
    return outputs[float(kio)]


def run_simulation_multi_kio_reduced(
    rho: float,
    V: float,
    kios: Sequence[float],
    j_delta: np.ndarray,
    j_Delta: np.ndarray,
    j_sum: np.ndarray,
    phase_coef: np.ndarray,
    cfg: Optional[SimConfig] = None,
    seed: int = 0,
    verbose: bool = True,
    *,
    narrow_snapshot_steps: Optional[np.ndarray] = None,
    narrow_snapshot_index: Optional[np.ndarray] = None,
    narrow_phase_coef: Optional[np.ndarray] = None,
) -> dict[float, ReducedResult]:
    """Reuse SI geometry across a k_io sweep, with Eq. 5 p_p mapping."""
    if cfg is None:
        cfg = SimConfig()
    cfg.assert_grid_alignment()
    narrow_args = (narrow_snapshot_steps, narrow_snapshot_index, narrow_phase_coef)
    if any(x is None for x in narrow_args) and not all(x is None for x in narrow_args):
        raise ValueError("narrow-pulse reduction requires all endpoint arrays together")
    narrow = all(x is not None for x in narrow_args)
    if cfg.phase_model == "narrow_pulse":
        if not narrow:
            raise ValueError("narrow-pulse cfg requires endpoint displacement arrays")
        sys.stderr.write(
            "WARNING: APPROXIMATE narrow-pulse diagnostic requested; CPU-only and prohibited for production builds.\n"
        )
    elif narrow:
        raise ValueError("endpoint displacement arrays require phase_model='narrow_pulse'")
    kios = [float(k) for k in kios]
    if any(k < 0.0 for k in kios):
        raise ValueError("k_io grid values must be non-negative")
    total_walks = 3 * int(cfg.n_ensembles) * int(cfg.n_walkers)
    if total_walks < PAPER_WALKS_PER_ENTRY // 10:
        sys.stderr.write(
            f"    WARNING: only {total_walks:,} axis-walks per entry; production reference is ~{PAPER_WALKS_PER_ENTRY:,}.\n"
        )
    ensembles = _build_ensembles_for_entry(rho, V, cfg, seed, verbose)
    outputs: dict[float, list[ReducedResult]] = {k: [] for k in kios}
    for ensemble_index, ens in enumerate(ensembles):
        walk_seed = _walk_seed(seed, ensemble_index)
        for kio in kios:
            if ens.is_free_water:
                pp = 0.0
            elif kio == 0.0:
                pp = 0.0
            else:
                pp = _checked_pp(kio_to_pp(kio, ens.mean_AV, cfg))
            result = _walk_and_reduce_one_ensemble(
                ens, pp, cfg, j_delta, j_Delta, j_sum, phase_coef, walk_seed,
                narrow_snapshot_steps=narrow_snapshot_steps,
                narrow_snapshot_index=narrow_snapshot_index,
                narrow_phase_coef=narrow_phase_coef,
            )
            outputs[kio].append(result)
            if verbose:
                print(
                    f"    ensemble {ensemble_index + 1}/{len(ensembles)}, k_io={kio:g}: "
                    f"p_p={pp:.7g}, Eq.5={result.analytic_kio_eq5:.5g} s^-1"
                )
    return {k: _merge_results(value) for k, value in outputs.items()}

"""Periodic contracted-Poisson-Voronoi ensembles.

This module deliberately makes geometry *measured state*, not requested
state.  A library entry is allowed to carry a ``rho``, ``V`` or ``v_i`` label
only after the realised Poisson tessellation has been calibrated and measured.

The production geometry is periodic.  Primary seeds live in ``[0, L)^3`` and
the Voronoi distance uses the minimum-image convention.  This removes the
old finite-box escape-selection bias while retaining an exactly stationary,
uniform-in-volume initial distribution.  Phase accumulation uses unwrapped
walker coordinates in :mod:`madi.walker_gpu`.

The CUDA kernel cannot call SciPy's periodic KD-tree.  Both CPU and CUDA
therefore use the same candidate-cache algorithm: retain ``K`` primary seeds
at each cache-voxel centre, re-rank all ``K`` at the actual walker position,
and classify from the closest two.  ``classify_exact_cpu`` exists only for
Tier-A validation of that algorithm; it is not a second production engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

from .config import SimConfig


def _rho_um3(rho_per_uL: float) -> float:
    """cells/µL -> cells/µm³."""
    return float(rho_per_uL) / 1e9


def _V_um3(V_pL: float) -> float:
    """pL -> µm³."""
    return float(V_pL) * 1e3


def _rho_per_uL(rho_um3: float) -> float:
    return float(rho_um3) * 1e9


def _V_pL(V_um3: float) -> float:
    return float(V_um3) / 1e3


def _minimum_image(delta: np.ndarray, L: float) -> np.ndarray:
    """Return the nearest periodic image displacement in ``[-L/2, L/2)``."""
    return delta - L * np.floor(delta / L + 0.5)


def _nearest_image(seed: np.ndarray, point: np.ndarray, L: float) -> np.ndarray:
    """Image of ``seed`` nearest to ``point`` (vectorised in the last axis)."""
    return seed + L * np.floor((point - seed) / L + 0.5)


@dataclass(frozen=True)
class GeometryStats:
    """Direct measurements associated with one realised Poisson ensemble."""

    vi: float
    vi_se: float
    rho_per_uL: float
    mean_volume_um3: float
    mean_volume_pL: float
    sampled_mean_volume_um3: float
    sampled_mean_volume_se_um3: float
    mean_area_um2: float
    mean_area_se_um2: float
    mean_A_over_V_um_inv: float
    mean_A_over_V_se_um_inv: float
    alpha_star_um: float
    annulus_mean_um: float
    annulus_std_um: float
    annulus_min_um: float
    annulus_max_um: float
    annulus_q05_um: float
    annulus_q50_um: float
    annulus_q95_um: float
    n_primary_cells: int
    n_geometry_cells: int
    n_vi_validation_points: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class Ensemble:
    """One periodic contracted-Voronoi realisation.

    ``rho`` and ``V`` are realised labels, not the requested coordinates.
    ``rho_requested`` and ``V_requested`` remain in provenance so a failed
    geometry calibration cannot silently masquerade as a valid grid point.
    """

    seeds: np.ndarray                 # (n_primary, 3), in [0,L), float64
    annulus: np.ndarray               # (n_primary,), µm
    grid_candidates: np.ndarray       # (G,G,G,K), primary indices or -1
    rho: float                        # realised cells/µL
    V: float                          # realised mean cell volume, pL
    vi: float                         # direct-MC realised volume fraction
    alpha_star: float
    L: float
    mean_AV: float                    # untrimmed sample mean A_i/V_i, µm^-1
    grid_spacing: float
    classifier_candidates: int
    geometry: GeometryStats
    rho_requested: float
    V_requested: float
    periodic_tree: Optional[Any] = None
    is_free_water: bool = False

    @property
    def grid_s1(self) -> np.ndarray:
        """Compatibility view of the first cached candidate."""
        return self.grid_candidates[..., 0]

    @property
    def grid_s2(self) -> np.ndarray:
        """Compatibility view of the second cached candidate."""
        return self.grid_candidates[..., 1]

    def classify_cpu(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Production candidate-cache classifier used by the CPU walk.

        It intentionally mirrors the CUDA classifier: candidates are obtained
        from the voxel containing the *wrapped* position and are re-ranked at
        the actual position using periodic images of their primary seeds.
        """
        pts = np.asarray(positions, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.is_free_water:
            return np.zeros(len(pts), dtype=np.int32), np.zeros(len(pts), dtype=bool)

        wrapped = np.mod(pts, self.L)
        gi = np.floor(wrapped / self.grid_spacing).astype(np.int64)
        G = self.grid_candidates.shape[0]
        gi = np.clip(gi, 0, G - 1)
        candidates = self.grid_candidates[gi[:, 0], gi[:, 1], gi[:, 2]]
        return _classify_candidate_rows(
            wrapped, candidates, self.seeds, self.annulus, self.L
        )

    def classify_exact_cpu(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Exact periodic KD-tree classification for validation only."""
        pts = np.asarray(positions, dtype=np.float64)
        if self.is_free_water:
            return np.zeros(len(pts), dtype=np.int32), np.zeros(len(pts), dtype=bool)
        if self.periodic_tree is None:
            self.periodic_tree = cKDTree(self.seeds, boxsize=self.L)
        wrapped = np.mod(pts, self.L)
        _, idx = self.periodic_tree.query(wrapped, k=2)
        if idx.ndim == 1:
            idx = idx[:, None]
        s1 = idx[:, 0].astype(np.int32)
        s2 = idx[:, 1].astype(np.int32)
        im1 = _nearest_image(self.seeds[s1], wrapped, self.L)
        im2 = _nearest_image(self.seeds[s2], wrapped, self.L)
        diff = im2 - im1
        norm = np.maximum(np.linalg.norm(diff, axis=1), 1e-30)
        midpoint = 0.5 * (im1 + im2)
        signed = np.einsum("ij,ij->i", wrapped - midpoint, diff / norm[:, None])
        inside = np.abs(signed) >= self.annulus[s1]
        return s1, inside


def _classify_candidate_rows(
    points: np.ndarray,
    candidates: np.ndarray,
    seeds: np.ndarray,
    annulus: np.ndarray,
    L: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy reference for the candidate-cache classifier.

    This is the source-level algorithm translated line-for-line into the
    CUDA kernel.  It is deliberately separate from ``classify_exact_cpu`` so
    Tier-A tests can bound cache error against the exact KD-tree.
    """
    if candidates.ndim != 2 or candidates.shape[1] < 2:
        raise ValueError("candidate cache must provide at least two seeds")
    n, k = candidates.shape
    valid = candidates >= 0
    safe = np.where(valid, candidates, 0)
    seed_rows = seeds[safe]                             # (N,K,3)
    images = _nearest_image(seed_rows, points[:, None, :], L)
    d2 = np.sum((points[:, None, :] - images) ** 2, axis=2)
    d2[~valid] = np.inf
    order = np.argsort(d2, axis=1, kind="stable")
    first = order[:, 0]
    second = order[:, 1]
    rows = np.arange(n)
    s1 = safe[rows, first].astype(np.int32)
    s2 = safe[rows, second].astype(np.int32)
    im1 = images[rows, first]
    im2 = images[rows, second]
    diff = im2 - im1
    norm = np.maximum(np.linalg.norm(diff, axis=1), 1e-30)
    midpoint = 0.5 * (im1 + im2)
    signed = np.einsum("ij,ij->i", points - midpoint, diff / norm[:, None])
    return s1, np.abs(signed) >= annulus[s1]


def _nearest_face_data(
    tree: cKDTree, seeds: np.ndarray, points: np.ndarray, L: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest primary label and its distance to the nearest face."""
    wrapped = np.mod(np.asarray(points, dtype=np.float64), L)
    _, idx = tree.query(wrapped, k=2)
    if idx.ndim == 1:
        idx = idx[:, None]
    s1 = idx[:, 0].astype(np.int32)
    s2 = idx[:, 1].astype(np.int32)
    im1 = _nearest_image(seeds[s1], wrapped, L)
    im2 = _nearest_image(seeds[s2], wrapped, L)
    diff = im2 - im1
    norm = np.maximum(np.linalg.norm(diff, axis=1), 1e-30)
    midpoint = 0.5 * (im1 + im2)
    signed = np.einsum("ij,ij->i", wrapped - midpoint, diff / norm[:, None])
    return s1, np.abs(signed)


def _measure_vi(
    seeds: np.ndarray,
    annulus: np.ndarray,
    tree: cKDTree,
    L: float,
    n: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Direct uniform-volume measurement of periodic intracellular fraction."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, L, size=(int(n), 3))
    s1, face_dist = _nearest_face_data(tree, seeds, pts, L)
    return float(np.mean(face_dist >= annulus[s1]))


def _solve_alpha_for_vi(
    target_vi: float,
    face_labels: np.ndarray,
    face_distances: np.ndarray,
    annulus_caps: np.ndarray,
    cfg: SimConfig,
) -> float:
    """Calibrate α* on this *specific* Poisson realisation.

    The random probe points and their exact nearest faces are fixed before
    bisection.  Consequently the objective is monotone, reproducible and
    cheap: each iteration is only a vector comparison, not another KD query.
    """
    if not (0.0 < target_vi < 1.0):
        raise ValueError(f"target v_i must lie in (0, 1), got {target_vi}")
    caps_at_points = annulus_caps[face_labels]

    def volume_fraction(alpha: float) -> float:
        return float(np.mean(face_distances >= np.minimum(alpha, caps_at_points)))

    lo = 0.0
    hi = float(np.max(annulus_caps))
    vi_lo = volume_fraction(lo)
    vi_hi = volume_fraction(hi)
    if target_vi > vi_lo + cfg.geometry_vi_tolerance:
        raise RuntimeError(
            f"target v_i={target_vi:.5f} exceeds alpha=0 realised value {vi_lo:.5f}"
        )
    if target_vi < vi_hi - cfg.geometry_vi_tolerance:
        raise RuntimeError(
            "target v_i cannot be realised by this contracted tessellation: "
            f"minimum sampled v_i={vi_hi:.5f}, requested={target_vi:.5f}."
        )

    for _ in range(int(cfg.geometry_alpha_iterations)):
        mid = 0.5 * (lo + hi)
        if volume_fraction(mid) > target_vi:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _periodic_nearest_neighbour_distances(tree: cKDTree, seeds: np.ndarray) -> np.ndarray:
    """Nearest different-primary distance for every primary seed."""
    _, idx = tree.query(seeds, k=2)
    if idx.ndim == 1:
        raise RuntimeError("periodic tessellation needs at least two seeds")
    second = idx[:, 1]
    return np.linalg.norm(_minimum_image(seeds[second] - seeds, tree.boxsize[0]), axis=1)


def _cell_polyhedron(
    seed_index: int,
    seeds: np.ndarray,
    annulus: np.ndarray,
    tree: cKDTree,
    L: float,
    neighbour_count: int = 64,
) -> Optional[Tuple[float, float]]:
    """Exact volume and area of one contracted periodic Voronoi cell.

    A bounded local neighbour set plus the 26 periodic images of the cell
    itself supplies all halfspaces.  The function returns ``None`` on a
    numerical Qhull failure rather than substituting a trimmed/analytic
    proxy; callers enforce a high successful-cell fraction.
    """
    n_cells = len(seeds)
    if n_cells < 2:
        return None
    si = seeds[seed_index]
    k = min(max(neighbour_count, 2), n_cells)
    _, idx = tree.query(si, k=k)
    idx = np.atleast_1d(idx)
    halfspaces: list[np.ndarray] = []

    def add_plane(neighbour: np.ndarray) -> bool:
        dvec = neighbour - si
        dmag = float(np.linalg.norm(dvec))
        if dmag < 1e-12:
            return True
        offset = dmag / 2.0 - float(annulus[seed_index])
        if offset <= 1e-9:
            return False
        normal = dvec / dmag
        halfspaces.append(np.r_[normal, -(offset + float(normal @ si))])
        return True

    for j in idx:
        j = int(j)
        if j == seed_index:
            continue
        dvec = _minimum_image(seeds[j] - si, L)
        if not add_plane(si + dvec):
            return None

    # A primary seed's own periodic images can bound a sparse low-density
    # realisation.  They are harmless redundant halfspaces in dense cases.
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        if not add_plane(si + L * np.asarray(offset, dtype=np.float64)):
            return None

    if len(halfspaces) < 4:
        return None
    try:
        hs = np.asarray(halfspaces, dtype=np.float64)
        hsi = HalfspaceIntersection(hs, si, qhull_options="QJ")
        vertices = hsi.intersections
        if vertices.shape[0] < 4 or not np.all(np.isfinite(vertices)):
            return None
        hull = ConvexHull(vertices, qhull_options="QJ")
        if hull.volume <= 0.0 or hull.area <= 0.0:
            return None
        return float(hull.volume), float(hull.area)
    except Exception:
        return None


def _sample_geometry_stats(
    seeds: np.ndarray,
    annulus: np.ndarray,
    tree: cKDTree,
    L: float,
    vi: float,
    vi_se: float,
    alpha_star: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> GeometryStats:
    n_cells = len(seeds)
    n_sample = min(int(cfg.geometry_sample_cells), n_cells)
    sample_idx = rng.choice(n_cells, size=n_sample, replace=False)
    volumes: list[float] = []
    areas: list[float] = []
    for i in sample_idx:
        item = _cell_polyhedron(int(i), seeds, annulus, tree, L)
        if item is not None:
            vol, area = item
            volumes.append(vol)
            areas.append(area)

    # A numerical failure is never silently hidden by a partial/trimmed mean.
    # A sparse but valid Poisson realisation can legitimately contain fewer
    # than eight primary cells (for example, a local Tier-A diagnostic).  Do
    # not make that configuration mathematically impossible by demanding a
    # fixed absolute sample count greater than ``n_sample``; production
    # builds use hundreds or more cells and still require at least 90% of the
    # selected cells to yield an exact polyhedron.
    required = min(n_sample, max(2, int(np.ceil(0.90 * n_sample))))
    if len(volumes) < required:
        raise RuntimeError(
            "Could not measure enough exact contracted-cell polyhedra: "
            f"{len(volumes)}/{n_sample} succeeded (need {required})."
        )
    vol = np.asarray(volumes, dtype=np.float64)
    area = np.asarray(areas, dtype=np.float64)
    av = area / vol
    n_ok = len(vol)
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    # The mean volume over *all* primary cells is total intracellular volume
    # divided by the realised primary count.  The direct v_i measurement is
    # therefore the authoritative, untrimmed V label; sampled exact cells are
    # retained independently as a geometric cross-check.
    rho_real = _rho_per_uL(n_cells / L**3)
    mean_volume = vi * L**3 / n_cells
    qs = np.quantile(annulus, [0.05, 0.50, 0.95])
    return GeometryStats(
        vi=float(vi),
        vi_se=float(vi_se),
        rho_per_uL=float(rho_real),
        mean_volume_um3=float(mean_volume),
        mean_volume_pL=_V_pL(mean_volume),
        sampled_mean_volume_um3=float(np.mean(vol)),
        sampled_mean_volume_se_um3=sem(vol),
        mean_area_um2=float(np.mean(area)),
        mean_area_se_um2=sem(area),
        mean_A_over_V_um_inv=float(np.mean(av)),
        mean_A_over_V_se_um_inv=sem(av),
        alpha_star_um=float(alpha_star),
        annulus_mean_um=float(np.mean(annulus)),
        annulus_std_um=float(np.std(annulus, ddof=1)) if len(annulus) > 1 else 0.0,
        annulus_min_um=float(np.min(annulus)),
        annulus_max_um=float(np.max(annulus)),
        annulus_q05_um=float(qs[0]),
        annulus_q50_um=float(qs[1]),
        annulus_q95_um=float(qs[2]),
        n_primary_cells=int(n_cells),
        n_geometry_cells=int(n_ok),
        n_vi_validation_points=int(cfg.geometry_validation_points),
    )


def _build_candidate_cache(
    tree: cKDTree,
    L: float,
    cfg: SimConfig,
) -> np.ndarray:
    """Build periodic K-candidate cache, padded with -1 where necessary."""
    G = cfg.grid_size
    K = int(cfg.classifier_candidates)
    if K < 2:
        raise ValueError("classifier_candidates must be at least two")
    out = np.full((G, G, G, K), -1, dtype=np.int32)
    coords = (np.arange(G, dtype=np.float64) + 0.5) * cfg.grid_spacing
    coords = np.minimum(coords, np.nextafter(L, 0.0))
    yy, zz = np.meshgrid(coords, coords, indexing="ij")
    yz = np.column_stack((yy.ravel(), zz.ravel()))
    query_k = min(K, tree.n)
    for ix, x in enumerate(coords):
        slab = np.column_stack((np.full(G * G, x), yz))
        _, idx = tree.query(slab, k=query_k)
        idx = np.asarray(idx)
        if query_k == 1:
            idx = idx[:, None]
        out[ix, :, :, :query_k] = idx.reshape(G, G, query_k).astype(np.int32)
    return out


def create_ensemble(
    rho: float,
    V: float,
    cfg: SimConfig | None = None,
    seed: int | None = None,
    verbose: bool = False,
    verify_vi: bool = True,
) -> Ensemble:
    """Create and directly measure a periodic contracted-Poisson ensemble.

    ``rho`` and ``V`` are requested grid coordinates.  They set
    ``v_i,target = rho * V * 1e-6``.  The returned object's public labels
    are the realised density, directly measured volume fraction and the
    implied realised mean cell volume.  A geometry that misses the target by
    more than ``geometry_vi_tolerance`` raises instead of clamping.
    """
    if cfg is None:
        cfg = SimConfig()
    cfg.assert_grid_alignment()
    if cfg.boundary_mode != "periodic":
        raise ValueError(
            "create_ensemble is a periodic production constructor. "
            "absorbing_legacy is available only as a walker diagnostic."
        )
    rho_um3 = _rho_um3(rho)
    V_um3 = _V_um3(V)
    target_vi = rho_um3 * V_um3
    if not (0.0 < target_vi < 1.0):
        raise ValueError(f"rho={rho:g}, V={V:g} give invalid v_i={target_vi:g}")
    if target_vi > 0.99 + 1e-12:
        raise ValueError(f"v_i={target_vi:.6f} exceeds the supported 0.99 ceiling")
    if rho_um3 <= 0.0:
        raise ValueError("rho must be positive for a cellular ensemble")

    rng = np.random.default_rng(seed)
    L = float(cfg.L)
    n_expected = rho_um3 * L**3
    n_cells = int(rng.poisson(n_expected))
    if n_cells < 2:
        raise RuntimeError(
            f"Poisson realisation has {n_cells} cells (expected {n_expected:.3g}); "
            "increase L or rho rather than altering the requested density."
        )
    seeds = rng.uniform(0.0, L, size=(n_cells, 3)).astype(np.float64)
    tree = cKDTree(seeds, boxsize=L)
    nn_dist = _periodic_nearest_neighbour_distances(tree, seeds)
    annulus_caps = cfg.kappa * nn_dist / 2.0

    # Calibrate α* using exact KD-tree labels on fixed probe points.
    cal_points = rng.uniform(0.0, L, size=(int(cfg.geometry_calibration_points), 3))
    cal_label, cal_face = _nearest_face_data(tree, seeds, cal_points, L)
    alpha_star = _solve_alpha_for_vi(target_vi, cal_label, cal_face, annulus_caps, cfg)
    annulus = np.minimum(alpha_star, annulus_caps).astype(np.float64)

    # Independent direct-volume validation avoids declaring success merely
    # because the same probe sample was used by the bisection.
    val_points = rng.uniform(0.0, L, size=(int(cfg.geometry_validation_points), 3))
    val_label, val_face = _nearest_face_data(tree, seeds, val_points, L)
    realised_vi = float(np.mean(val_face >= annulus[val_label]))
    vi_se = float(np.sqrt(max(realised_vi * (1.0 - realised_vi), 0.0) /
                          max(len(val_points), 1)))
    allowed = max(float(cfg.geometry_vi_tolerance), 4.0 * vi_se)
    if abs(realised_vi - target_vi) > allowed:
        raise RuntimeError(
            "Realised v_i failed geometry acceptance: "
            f"target={target_vi:.6f}, measured={realised_vi:.6f}, "
            f"allowed={allowed:.6f}."
        )

    stats = _sample_geometry_stats(
        seeds, annulus, tree, L, realised_vi, vi_se, alpha_star, cfg, rng
    )
    cache = _build_candidate_cache(tree, L, cfg)
    ens = Ensemble(
        seeds=seeds,
        annulus=annulus,
        grid_candidates=cache,
        rho=stats.rho_per_uL,
        V=stats.mean_volume_pL,
        vi=stats.vi,
        alpha_star=alpha_star,
        L=L,
        mean_AV=stats.mean_A_over_V_um_inv,
        grid_spacing=float(cfg.grid_spacing),
        classifier_candidates=int(cfg.classifier_candidates),
        geometry=stats,
        rho_requested=float(rho),
        V_requested=float(V),
        periodic_tree=tree,
    )
    if verbose:
        print(
            "    [geometry] "
            f"cells={n_cells}, rho={ens.rho:.1f}/uL, "
            f"target v_i={target_vi:.5f}, realised={ens.vi:.5f}±{vi_se:.5f}, "
            f"V={ens.V:.5f} pL, alpha*={alpha_star:.5f} um, "
            f"<A/V>={ens.mean_AV:.5f} um^-1"
        )
    return ens


def create_dummy_ensemble(cfg: SimConfig) -> Ensemble:
    """Pure-water sentinel.  It has no classifier or cellular geometry."""
    stats = GeometryStats(
        vi=0.0, vi_se=0.0, rho_per_uL=0.0,
        mean_volume_um3=0.0, mean_volume_pL=0.0,
        sampled_mean_volume_um3=0.0, sampled_mean_volume_se_um3=0.0,
        mean_area_um2=0.0, mean_area_se_um2=0.0,
        mean_A_over_V_um_inv=0.0, mean_A_over_V_se_um_inv=0.0,
        alpha_star_um=0.0,
        annulus_mean_um=0.0, annulus_std_um=0.0,
        annulus_min_um=0.0, annulus_max_um=0.0,
        annulus_q05_um=0.0, annulus_q50_um=0.0, annulus_q95_um=0.0,
        n_primary_cells=0, n_geometry_cells=0, n_vi_validation_points=0,
    )
    return Ensemble(
        seeds=np.zeros((2, 3), dtype=np.float64),
        annulus=np.zeros(2, dtype=np.float64),
        grid_candidates=np.zeros((1, 1, 1, 2), dtype=np.int32),
        rho=0.0, V=0.0, vi=0.0, alpha_star=0.0, L=float(cfg.L),
        mean_AV=0.0, grid_spacing=float(cfg.grid_spacing),
        classifier_candidates=2, geometry=stats,
        rho_requested=0.0, V_requested=0.0, is_free_water=True,
    )


def estimate_vi(ens: Ensemble, n: int = 200_000, seed: int = 42) -> float:
    """Independent exact-KD volume-fraction estimate for a realised ensemble."""
    if ens.is_free_water:
        return 0.0
    if ens.periodic_tree is None:
        ens.periodic_tree = cKDTree(ens.seeds, boxsize=ens.L)
    return _measure_vi(
        ens.seeds, ens.annulus, ens.periodic_tree, ens.L,
        n=n, rng=np.random.default_rng(seed),
    )


# ---------------------------------------------------------------------------
# Compatibility helpers kept only for external audit notebooks.  There is no
# longer a v_i lookup table or a clamp path in the production constructor.
# ---------------------------------------------------------------------------

def _lookup_table_path(cfg: SimConfig) -> str:
    """Deprecated compatibility name; no production code reads this path."""
    return ""


def compute_mean_AV_exact(
    seeds: np.ndarray, annulus: np.ndarray, L: float, verbose: bool = False
) -> Optional[float]:
    """Untrimmed exact A/V mean over all cells in a small periodic ensemble.

    This retained utility is intentionally conservative and should be used
    for diagnostics, not large production ensembles.  The production path
    samples cells and records its sample count and standard error.
    """
    tree = cKDTree(np.asarray(seeds), boxsize=float(L))
    vals: list[float] = []
    for i in range(len(seeds)):
        item = _cell_polyhedron(i, seeds, annulus, tree, float(L))
        if item is not None:
            v, a = item
            vals.append(a / v)
    if verbose:
        print(f"[exact A/V] {len(vals)}/{len(seeds)} periodic cells succeeded")
    return float(np.mean(vals)) if vals else None

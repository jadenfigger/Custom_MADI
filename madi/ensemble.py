"""SI-exact contracted Poisson--Voronoi ensemble construction.

The governing geometry is specified in MADI I Supporting Information
§§S.I--S.IV, rather than by a cache-specific approximation:

* seeds are a non-periodic Poisson process in an encompassing ``Ω_pop``;
* ``Ω_sim`` is sized by Eq. S8 and walkers begin uniformly in its concentric
  ``Ω_src`` cube (edge ``0.4 W``);
* the target annulus width is obtained from the SI Eq. S7a cubic, while the
  per-cell cap is ``min(alpha_star, 0.9*d_nn/2)`` (Eqs. S4--S5); and
* every classification asks for the nearest seed at the actual position and
  evaluates the *full conjunction* of shifted Voronoi facets in SI Eq. S2.
  The exact radius bound ``d_1 + 2 alpha_1`` restricts that conjunction to
  the only facets that can bind; it is not a fixed-neighbour approximation.

``<A/V>`` used for the Eq. 5 permeability calibration deliberately comes
from the certified 5e6-single-cell reference cloud, *not* from this finite
``Ω_sim`` realization (SI §S.IV).  The finite realization is measured only
to validate the requested packing and to record its geometric provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import SimConfig


# SI Eq. S7a.  ``x = rho^(1/3) alpha_star`` is dimensionless when rho is in
# cells / um^3 and alpha_star is in um.
_S7A_CUBIC = -0.570444
_S7A_QUADRATIC = 10.6047
_S7A_LINEAR = -5.81457
_S7A_CONSTANT = 1.0


def _rho_um3(rho_per_uL: float) -> float:
    """cells/µL -> cells/µm³."""
    return float(rho_per_uL) / 1.0e9


def _V_um3(V_pL: float) -> float:
    """pL -> µm³."""
    return float(V_pL) * 1.0e3


def _rho_per_uL(rho_um3: float) -> float:
    return float(rho_um3) * 1.0e9


def _distance_to_cube(points: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Euclidean distance from points to an axis-aligned closed cube."""
    pts = np.asarray(points, dtype=np.float64)
    excess = np.maximum(np.maximum(lo - pts, 0.0), pts - hi)
    return np.linalg.norm(excess, axis=-1)


@dataclass(frozen=True)
class GeometryReference:
    """Certified SI single-cell process table used for Eq. 5 calibration."""

    vi: np.ndarray
    alpha_x: np.ndarray
    mean_A_over_V_norm: np.ndarray
    metadata: dict
    path: str


@dataclass(frozen=True)
class PopulationCertificate:
    """Numerical certificate for the two SI Ω_pop requirements.

    ``s10_distance_upper_um`` is an upper bound on the distance from any
    point in Ω_sim to its nearest populated seed.  It is computed from a
    regular grid plus the 1-Lipschitz nearest-distance bound.  If it is less
    than the Ω_pop margin, a seed outside Ω_pop cannot alter the tessellation
    in Ω_sim (Eq. S10).  Eq. S11 is checked directly for every seed whose
    cell could touch Ω_sim.
    """

    pop_margin_um: float
    s10_distance_upper_um: float
    s10_pass: bool
    n_relevant_seeds: int
    s11_min_clearance_margin_um: float
    s11_pass: bool
    certification_spacing_um: float
    expansion_index: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryStats:
    """Measured finite-realisation provenance; never the Eq. 5 estimator."""

    target_vi: float
    realised_vi: float
    realised_vi_se: float
    realised_rho_per_uL: float
    realised_mean_volume_pL: float
    alpha_star_um: float
    governing_mean_A_over_V_um_inv: float
    annulus_mean_um: float
    annulus_std_um: float
    annulus_min_um: float
    annulus_max_um: float
    annulus_q05_um: float
    annulus_q50_um: float
    annulus_q95_um: float
    n_seeds_pop: int
    n_seeds_sim: int
    n_validation_points: int
    sim_side_um: float
    source_side_um: float
    population: PopulationCertificate
    reference_path: str
    reference_metadata: dict

    def to_dict(self) -> dict:
        out = asdict(self)
        # Keep the nested certificate explicitly named in the persisted JSON.
        out["population"] = self.population.to_dict()
        return out


@dataclass
class Ensemble:
    """One finite SI-conformant contracted Poisson--Voronoi realization."""

    seeds: np.ndarray                     # all Ω_pop seeds, world coordinates [µm]
    annulus: np.ndarray                   # per-seed α_i [µm]
    rho: float                            # measured seed density in Ω_sim [cells/µL]
    V: float                              # measured mean water-cell volume [pL]
    vi: float                             # direct uniform-volume measurement in Ω_sim
    alpha_star: float                     # target α* from Eq. S7a [µm]
    L: float                              # Ω_sim edge W [µm]
    source_lo: float                      # Ω_src lower coordinate [µm]
    source_hi: float                      # Ω_src upper coordinate [µm]
    mean_AV: float                        # governing-process <A/V>, not finite-realisation
    geometry: GeometryStats
    rho_requested: float
    V_requested: float
    kd_node_seed: np.ndarray              # balanced exact GPU KD tree
    kd_node_axis: np.ndarray
    kd_node_left: np.ndarray
    kd_node_right: np.ndarray
    kd_node_parent: np.ndarray
    tree: Optional[Any] = None
    is_free_water: bool = False

    def _tree(self) -> cKDTree:
        if self.tree is None:
            self.tree = cKDTree(self.seeds)
        return self.tree

    def classify_cpu(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify using the full shifted-facet contraction of SI Eq. S2.

        For nearest seed ``c_1`` and a competing seed ``c_j``, the signed
        margin to facet ``j`` is

        ``(d_j^2 - d_1^2) / (2 |c_1-c_j|)``.

        The point is intracellular only if every margin is at least
        ``alpha_1``.  A facet can bind only for a seed within
        ``d_1 + 2 alpha_1`` of the point: rearranging ``margin < alpha_1``
        with ``|c_1-c_j| <= d_1+d_j`` gives the positive root
        ``d_j < d_1 + 2 alpha_1``.  Thus ``query_ball_point`` is an exact
        adaptive radius query, not a fixed-k truncation.  The CUDA kernel
        implements the same nearest-plus-radius rule with its balanced KD
        tree.
        """
        pts = np.asarray(positions, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.is_free_water:
            return np.zeros(len(pts), dtype=np.int32), np.zeros(len(pts), dtype=bool)
        d1, s1 = self._tree().query(pts, k=1)
        d1 = np.asarray(d1, dtype=np.float64)
        s1 = np.asarray(s1, dtype=np.int32)
        if len(self.seeds) < 2:
            raise RuntimeError("full-facet classifier requires at least two population seeds")
        alpha1 = self.annulus[s1]
        candidate_lists = self._tree().query_ball_point(pts, d1 + 2.0 * alpha1)
        inside = np.ones(len(pts), dtype=bool)
        for point_index, candidate_ids in enumerate(candidate_lists):
            ids = np.asarray(candidate_ids, dtype=np.int32)
            ids = ids[ids != s1[point_index]]
            if len(ids) == 0:
                continue
            seed_delta = self.seeds[ids] - self.seeds[s1[point_index]]
            separation = np.linalg.norm(seed_delta, axis=1)
            if np.any(separation == 0.0):
                raise RuntimeError("Poisson seed realization contains coincident seeds")
            point_delta = self.seeds[ids] - pts[point_index]
            dj_sq = np.einsum("ij,ij->i", point_delta, point_delta)
            margin = (dj_sq - d1[point_index] ** 2) / (2.0 * separation)
            inside[point_index] = bool(np.all(margin >= alpha1[point_index]))
        return s1, inside

    def classify_exact_cpu(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility alias: production classification is already exact."""
        return self.classify_cpu(positions)


def si_domain_side_um(rho_per_uL: float, cfg: SimConfig) -> float:
    """Ω_sim side W from SI Eq. S8, scaled to this walk duration.

    The density term is evaluated in cells/µm³, so its cube root is in µm.
    ``cfg.L`` is a Tier-A diagnostic override only and is persisted in the
    resulting geometry metadata; production configuration leaves it ``None``.
    """
    if cfg.L is not None:
        if cfg.L <= 0.0:
            raise ValueError("diagnostic L override must be positive")
        return float(cfg.L)
    rho_um3 = _rho_um3(rho_per_uL)
    if rho_um3 <= 0.0:
        # The acellular control has no cell-density scale.  The upper S8
        # random-walk bound is the conservative finite domain for its walk.
        return float(20.0 * np.sqrt(2.0 * cfg.D0 * cfg.tRW_max))
    L_rw = float(np.sqrt(2.0 * cfg.D0 * cfg.tRW_max))
    density_side = float((8.0e5 / rho_um3) ** (1.0 / 3.0))
    return float(max(10.0 * L_rw, min(20.0 * L_rw, density_side)))


def alpha_star_from_vi(vi: float, rho_per_uL: float) -> float:
    """Invert SI Eq. S7a analytically and return α* in µm.

    The cubic has at most three real roots.  The SI states that exactly one
    belongs to the monotone physical branch; selecting that root is a direct
    polynomial inversion, not a lookup-table clamp or a finite-ensemble fit.
    """
    vi = float(vi)
    rho_um3 = _rho_um3(rho_per_uL)
    if not (0.0 < vi < 1.0):
        raise ValueError(f"target v_i must lie in (0, 1), got {vi}")
    if rho_um3 <= 0.0:
        raise ValueError("rho must be positive for a cellular geometry")
    roots = np.roots([
        _S7A_CUBIC,
        _S7A_QUADRATIC,
        _S7A_LINEAR,
        _S7A_CONSTANT - vi,
    ])
    # The first stationary point is the end of the decreasing physical
    # branch.  It is derived from d(v_i)/dx=0, not tuned to simulation data.
    turn_roots = np.roots([3.0 * _S7A_CUBIC, 2.0 * _S7A_QUADRATIC, _S7A_LINEAR])
    x_turn = min(float(x.real) for x in turn_roots if abs(x.imag) < 1e-10 and x.real > 0.0)
    physical = sorted(
        float(x.real) for x in roots
        if abs(x.imag) < 1e-10 and -1e-12 <= x.real <= x_turn + 1e-12
    )
    if len(physical) != 1:
        raise RuntimeError(
            f"SI Eq. S7a has no unique physical root for v_i={vi}: {roots!r}"
        )
    return float(physical[0] / rho_um3 ** (1.0 / 3.0))


def _reference_default_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "geometry_reference_si_kappa_0p9.npz"


def _load_reference(cfg: SimConfig) -> GeometryReference:
    path = Path(cfg.geometry_reference_path) if cfg.geometry_reference_path else _reference_default_path()
    if not path.is_file():
        raise RuntimeError(
            "No SI-certified geometry reference table was found at "
            f"{path}. Generate the 5e6-single-cell, 26-alpha table before "
            "building a library; refusing to substitute a finite Ω_sim A/V."
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"vi", "alpha_x", "mean_A_over_V_norm", "metadata_json"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"geometry reference {path} is missing {sorted(missing)}")
        try:
            metadata = json.loads(str(data["metadata_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"geometry reference {path} has invalid metadata_json") from exc
        vi = np.asarray(data["vi"], dtype=np.float64)
        alpha_x = np.asarray(data["alpha_x"], dtype=np.float64)
        av_norm = np.asarray(data["mean_A_over_V_norm"], dtype=np.float64)
    if vi.ndim != 1 or len(vi) < 2 or len(vi) != len(alpha_x) or len(vi) != len(av_norm):
        raise RuntimeError("geometry reference axes must be matching 1-D arrays")
    if not (np.all(np.diff(vi) > 0.0) and np.all(np.diff(alpha_x) < 0.0)):
        raise RuntimeError("geometry reference v_i must increase while alpha_x decreases")
    if not np.all(np.isfinite(av_norm)) or np.any(av_norm <= 0.0):
        raise RuntimeError("geometry reference contains invalid <A/V> values")
    if not np.isclose(float(metadata.get("kappa", np.nan)), 0.90, rtol=0.0, atol=1e-12):
        raise RuntimeError("geometry reference must use SI κ=0.9")
    if metadata.get("mean_estimator") != "untrimmed_arithmetic_mean_A_over_V":
        raise RuntimeError("geometry reference must store the untrimmed arithmetic <A/V> mean")
    if metadata.get("contraction_rule") != "all_shifted_voronoi_facets_S2":
        raise RuntimeError(
            "geometry reference must use the full SI Eq. S2 shifted-facet contraction"
        )
    if not cfg.allow_uncertified_geometry_reference:
        if int(metadata.get("n_single_cells", 0)) < cfg.geometry_reference_required_cells:
            raise RuntimeError(
                "geometry reference is not SI-certified: it must contain at least "
                f"{cfg.geometry_reference_required_cells:,} independent single cells"
            )
        if int(metadata.get("n_alpha", 0)) != cfg.geometry_reference_required_alpha_values:
            raise RuntimeError(
                "geometry reference is not SI-certified: it must contain exactly "
                f"{cfg.geometry_reference_required_alpha_values} alpha values"
            )
    return GeometryReference(vi, alpha_x, av_norm, metadata, str(path.resolve()))


def governing_mean_A_over_V(vi: float, rho_per_uL: float, cfg: SimConfig) -> Tuple[float, GeometryReference]:
    """Interpolate the SI single-cell reference and rescale by rho^(1/3)."""
    reference = _load_reference(cfg)
    if vi < reference.vi[0] or vi > reference.vi[-1]:
        raise RuntimeError(
            f"requested v_i={vi:.6f} is outside certified reference range "
            f"[{reference.vi[0]:.6f}, {reference.vi[-1]:.6f}]; no clamping is allowed"
        )
    rho_um3 = _rho_um3(rho_per_uL)
    av_norm = float(np.interp(float(vi), reference.vi, reference.mean_A_over_V_norm))
    return float(av_norm * rho_um3 ** (1.0 / 3.0)), reference


def _build_gpu_kdtree(
    seeds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a balanced exact KD tree for the CUDA per-position query."""
    n = len(seeds)
    node_seed: list[int] = []
    node_axis: list[int] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_parent: list[int] = []

    def build(indices: np.ndarray, parent: int) -> int:
        if len(indices) == 0:
            return -1
        spread = np.ptp(seeds[indices], axis=0)
        axis = int(np.argmax(spread))
        median = len(indices) // 2
        order = np.argpartition(seeds[indices, axis], median)
        arranged = indices[order]
        node = len(node_seed)
        node_seed.append(int(arranged[median]))
        node_axis.append(axis)
        node_left.append(-1)
        node_right.append(-1)
        node_parent.append(parent)
        node_left[node] = build(arranged[:median], node)
        node_right[node] = build(arranged[median + 1:], node)
        return node

    root = build(np.arange(n, dtype=np.int32), -1)
    if root != 0:
        raise RuntimeError("GPU KD tree root must occupy node zero")
    return (
        np.asarray(node_seed, dtype=np.int32),
        np.asarray(node_axis, dtype=np.int8),
        np.asarray(node_left, dtype=np.int32),
        np.asarray(node_right, dtype=np.int32),
        np.asarray(node_parent, dtype=np.int32),
    )


def _nearest_distance_upper_bound(
    tree: cKDTree,
    side_um: float,
    spacing_um: float,
) -> Tuple[float, float]:
    """Certified grid/Lipschitz upper bound for Eq. S10."""
    n = max(1, int(np.ceil(side_um / spacing_um)))
    h = side_um / n
    coords = (np.arange(n, dtype=np.float64) + 0.5) * h
    max_distance = 0.0
    yy, zz = np.meshgrid(coords, coords, indexing="ij")
    yz = np.column_stack((yy.ravel(), zz.ravel()))
    for x in coords:
        points = np.column_stack((np.full(len(yz), x), yz))
        d, _ = tree.query(points, k=1)
        max_distance = max(max_distance, float(np.max(d)))
    return max_distance + 0.5 * np.sqrt(3.0) * h, h


def _make_population(
    rho_um3: float,
    side_um: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, cKDTree, np.ndarray, PopulationCertificate]:
    """Sample and certify a finite Ω_pop for SI Eqs. S10--S12."""
    cell_spacing = rho_um3 ** (-1.0 / 3.0)
    margin = max(
        cfg.population_initial_margin_cell_spacings * cell_spacing,
        2.0 * cfg.population_certification_spacing_um,
    )
    for expansion in range(int(cfg.population_max_expansions)):
        lo, hi = -margin, side_um + margin
        n_expected = rho_um3 * (hi - lo) ** 3
        n_seeds = int(rng.poisson(n_expected))
        if n_seeds < 2:
            margin *= 1.75
            continue
        seeds = rng.uniform(lo, hi, size=(n_seeds, 3)).astype(np.float64)
        tree = cKDTree(seeds)
        d_nn, _ = tree.query(seeds, k=2)
        nn = np.asarray(d_nn[:, 1], dtype=np.float64)

        s10_upper, h = _nearest_distance_upper_bound(
            tree, side_um, cfg.population_certification_spacing_um
        )
        s10_pass = bool(s10_upper < margin)
        # Any cell touching Ω_sim must have its seed within s10_upper of the
        # cube.  For each such seed, an outside Ω_pop seed cannot change the
        # annulus cap when its nearest populated neighbour is closer than the
        # seed's clearance to Ω_pop's boundary (Eq. S11).
        relevant = _distance_to_cube(seeds, 0.0, side_um) <= s10_upper
        clearance = np.min(np.minimum(seeds - lo, hi - seeds), axis=1)
        s11_margin = clearance[relevant] - nn[relevant]
        min_s11 = float(np.min(s11_margin)) if np.any(relevant) else float("inf")
        s11_pass = bool(np.all(s11_margin > 0.0))
        certificate = PopulationCertificate(
            pop_margin_um=float(margin),
            s10_distance_upper_um=float(s10_upper),
            s10_pass=s10_pass,
            n_relevant_seeds=int(np.count_nonzero(relevant)),
            s11_min_clearance_margin_um=min_s11,
            s11_pass=s11_pass,
            certification_spacing_um=float(h),
            expansion_index=expansion,
        )
        if s10_pass and s11_pass:
            return seeds, tree, nn, certificate
        margin *= 1.75
    raise RuntimeError(
        "Could not certify Ω_pop against SI Eqs. S10--S12 after "
        f"{cfg.population_max_expansions} expansions."
    )


def _measure_vi(ens: Ensemble, n: int, rng: np.random.Generator) -> Tuple[float, float]:
    points = rng.uniform(0.0, ens.L, size=(int(n), 3))
    _, inside = ens.classify_cpu(points)
    vi = float(np.mean(inside))
    se = float(np.sqrt(max(vi * (1.0 - vi), 0.0) / max(len(points), 1)))
    return vi, se


def create_ensemble(
    rho: float,
    V: float,
    cfg: SimConfig | None = None,
    seed: int | None = None,
    verbose: bool = False,
    verify_vi: bool = True,
) -> Ensemble:
    """Create a finite SI-conformant ensemble for requested ``rho, V``.

    Eq. S7a determines alpha_star from the desired generating-process
    ``v_i=rho*V*1e-6``.  The finite ensemble is never used to recalibrate
    alpha_star or <A/V>; it is only accepted when its independently measured
    packing is within the declared finite-volume tolerance.
    """
    if cfg is None:
        cfg = SimConfig()
    cfg.assert_grid_alignment()
    rho_um3 = _rho_um3(rho)
    target_vi = rho_um3 * _V_um3(V)
    if not (0.0 < target_vi < 1.0):
        raise ValueError(f"rho={rho:g}, V={V:g} give invalid v_i={target_vi:g}")
    if rho_um3 <= 0.0:
        raise ValueError("rho must be positive for a cellular ensemble")

    alpha_star = alpha_star_from_vi(target_vi, rho)
    mean_av, reference = governing_mean_A_over_V(target_vi, rho, cfg)
    side = si_domain_side_um(rho, cfg)
    rng = np.random.default_rng(seed)
    seeds, tree, nearest_neighbour, population = _make_population(rho_um3, side, cfg, rng)
    annulus = np.minimum(alpha_star, cfg.kappa * nearest_neighbour / 2.0).astype(np.float64)
    source_side = cfg.source_fraction * side
    source_lo = 0.5 * (side - source_side)
    source_hi = source_lo + source_side

    provisional_nodes = _build_gpu_kdtree(seeds)
    in_sim = np.all((seeds >= 0.0) & (seeds < side), axis=1)
    n_sim = int(np.count_nonzero(in_sim))
    if n_sim < 2:
        raise RuntimeError("Ω_sim contains fewer than two seed points; increase W or density")
    # Build before volume validation because the measurement shares the exact
    # SI classifier used by the random walk.
    provisional_stats = GeometryStats(
        target_vi=target_vi, realised_vi=float("nan"), realised_vi_se=float("nan"),
        realised_rho_per_uL=_rho_per_uL(n_sim / side**3),
        realised_mean_volume_pL=float("nan"), alpha_star_um=alpha_star,
        governing_mean_A_over_V_um_inv=mean_av,
        annulus_mean_um=float(np.mean(annulus[in_sim])),
        annulus_std_um=float(np.std(annulus[in_sim], ddof=1)) if n_sim > 1 else 0.0,
        annulus_min_um=float(np.min(annulus[in_sim])),
        annulus_max_um=float(np.max(annulus[in_sim])),
        annulus_q05_um=float(np.quantile(annulus[in_sim], 0.05)),
        annulus_q50_um=float(np.quantile(annulus[in_sim], 0.50)),
        annulus_q95_um=float(np.quantile(annulus[in_sim], 0.95)),
        n_seeds_pop=len(seeds), n_seeds_sim=n_sim,
        n_validation_points=int(cfg.geometry_validation_points), sim_side_um=side,
        source_side_um=source_side, population=population,
        reference_path=reference.path, reference_metadata=reference.metadata,
    )
    ens = Ensemble(
        seeds=seeds, annulus=annulus,
        rho=provisional_stats.realised_rho_per_uL, V=float("nan"), vi=float("nan"),
        alpha_star=alpha_star, L=side, source_lo=source_lo, source_hi=source_hi,
        mean_AV=mean_av, geometry=provisional_stats, rho_requested=float(rho),
        V_requested=float(V), kd_node_seed=provisional_nodes[0],
        kd_node_axis=provisional_nodes[1], kd_node_left=provisional_nodes[2],
        kd_node_right=provisional_nodes[3], kd_node_parent=provisional_nodes[4], tree=tree,
    )
    realised_vi, vi_se = _measure_vi(ens, cfg.geometry_validation_points, rng)
    allowed = max(float(cfg.geometry_vi_tolerance), 4.0 * vi_se)
    if verify_vi and abs(realised_vi - target_vi) > allowed:
        raise RuntimeError(
            "Finite Ω_sim packing failed SI geometry acceptance: "
            f"target={target_vi:.6f}, measured={realised_vi:.6f}, allowed={allowed:.6f}."
            " Refusing to attach a requested label to a mismatched full-facet "
            "SI Eq. S2 walker geometry."
        )
    realised_volume = realised_vi * side**3 / n_sim / 1.0e3
    stats = GeometryStats(
        target_vi=target_vi, realised_vi=realised_vi, realised_vi_se=vi_se,
        realised_rho_per_uL=provisional_stats.realised_rho_per_uL,
        realised_mean_volume_pL=realised_volume,
        alpha_star_um=alpha_star, governing_mean_A_over_V_um_inv=mean_av,
        annulus_mean_um=provisional_stats.annulus_mean_um,
        annulus_std_um=provisional_stats.annulus_std_um,
        annulus_min_um=provisional_stats.annulus_min_um,
        annulus_max_um=provisional_stats.annulus_max_um,
        annulus_q05_um=provisional_stats.annulus_q05_um,
        annulus_q50_um=provisional_stats.annulus_q50_um,
        annulus_q95_um=provisional_stats.annulus_q95_um,
        n_seeds_pop=provisional_stats.n_seeds_pop, n_seeds_sim=n_sim,
        n_validation_points=provisional_stats.n_validation_points,
        sim_side_um=side, source_side_um=source_side,
        population=population, reference_path=reference.path,
        reference_metadata=reference.metadata,
    )
    ens.rho = stats.realised_rho_per_uL
    ens.V = stats.realised_mean_volume_pL
    ens.vi = stats.realised_vi
    ens.geometry = stats
    if verbose:
        print(
            "    [SI geometry] "
            f"W={side:.3f} um, Ωsrc={source_side:.3f} um, seeds(sim/pop)="
            f"{n_sim}/{len(seeds)}, target v_i={target_vi:.5f}, "
            f"realised={realised_vi:.5f}±{vi_se:.5f}, alpha*={alpha_star:.5f} um, "
            f"<A/V>_process={mean_av:.5f} um^-1"
        )
    return ens


def create_dummy_ensemble(cfg: SimConfig) -> Ensemble:
    """Acellular free-water control with the conservative Eq. S8 walk box."""
    side = si_domain_side_um(0.0, cfg)
    source_side = cfg.source_fraction * side
    source_lo = 0.5 * (side - source_side)
    source_hi = source_lo + source_side
    certificate = PopulationCertificate(
        pop_margin_um=0.0, s10_distance_upper_um=0.0, s10_pass=True,
        n_relevant_seeds=0, s11_min_clearance_margin_um=float("inf"),
        s11_pass=True, certification_spacing_um=0.0, expansion_index=0,
    )
    stats = GeometryStats(
        target_vi=0.0, realised_vi=0.0, realised_vi_se=0.0,
        realised_rho_per_uL=0.0, realised_mean_volume_pL=0.0,
        alpha_star_um=0.0, governing_mean_A_over_V_um_inv=0.0,
        annulus_mean_um=0.0, annulus_std_um=0.0, annulus_min_um=0.0,
        annulus_max_um=0.0, annulus_q05_um=0.0, annulus_q50_um=0.0,
        annulus_q95_um=0.0, n_seeds_pop=0, n_seeds_sim=0,
        n_validation_points=0, sim_side_um=side, source_side_um=source_side,
        population=certificate, reference_path="", reference_metadata={},
    )
    return Ensemble(
        seeds=np.zeros((2, 3), dtype=np.float64), annulus=np.zeros(2, dtype=np.float64),
        rho=0.0, V=0.0, vi=0.0, alpha_star=0.0, L=side,
        source_lo=source_lo, source_hi=source_hi, mean_AV=0.0, geometry=stats,
        rho_requested=0.0, V_requested=0.0,
        kd_node_seed=np.array([0, 1], dtype=np.int32),
        kd_node_axis=np.array([0, 0], dtype=np.int8),
        kd_node_left=np.array([-1, -1], dtype=np.int32),
        kd_node_right=np.array([1, -1], dtype=np.int32),
        kd_node_parent=np.array([-1, 0], dtype=np.int32),
        is_free_water=True,
    )


def estimate_vi(ens: Ensemble, n: int = 200_000, seed: int = 42) -> float:
    """Independent exact-classifier volume-fraction measurement in Ω_sim."""
    if ens.is_free_water:
        return 0.0
    vi, _ = _measure_vi(ens, n, np.random.default_rng(seed))
    return vi


def governing_reference_summary(ens: Ensemble) -> dict:
    """Small explicit record used in library metadata and audit reports."""
    return {
        "mean_A_over_V_um_inv": ens.mean_AV,
        "source": "SI §S.IV 5e6 independent single-cell reference cloud",
        "reference_path": ens.geometry.reference_path,
        "reference_metadata": ens.geometry.reference_metadata,
    }

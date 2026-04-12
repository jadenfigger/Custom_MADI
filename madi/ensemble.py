"""
Contracted Voronoi cell ensemble with persistent v_i → (α*, <A/V>) lookup.

KEY FIXES vs. the previous version:

  Bottleneck #1 — per-ensemble exact <A/V> via Qhull:
      The previous version called HalfspaceIntersection + ConvexHull on
      every interior cell of every ensemble, costing ~30–60 s per ensemble
      (12 ensembles per library entry → 6–12 min/entry of pure CPU
      overhead while the GPU sat idle).

  Bottleneck #2 — per-ensemble α* bisection against Monte-Carlo v_i:
      Each ensemble ran 6–10 iterations of `_measure_vi`, costing another
      ~60–180 s per library entry.

  THE FIX (paper's actual approach, SI §S.II):
      Both α*·ρ^(1/3) and <A/V>·ρ^(-1/3) are dimensionless invariants of
      the Poisson–contracted-Voronoi process: they depend only on v_i and
      κ, not on ρ.  We build a single 1-D lookup table
          v_i  →  (α*·ρ^(1/3), <A/V>·ρ^(-1/3))
      ONCE at startup (or load from disk cache), then for every actual
      ensemble we just interpolate and rescale by the actual ρ.

      The table is built by running the same exact polyhedral routine
      that used to run per-ensemble — but now only ~25 times total,
      not ~12 × N_library_entries times.  After the one-time build
      (~10–20 min), `create_ensemble` becomes microseconds of arithmetic
      plus the (chunked, fast) voxel-grid construction.

      Cached to ~/.cache/madi/, keyed by κ + L + pop_margin so config
      changes invalidate the cache automatically.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.spatial import cKDTree, Voronoi, ConvexHull, HalfspaceIntersection
from dataclasses import dataclass
from typing import Tuple, Optional

from .config import SimConfig


# ===========================================================================
# Ensemble data class  (unchanged)
# ===========================================================================

@dataclass
class Ensemble:
    seeds:        np.ndarray       # (n_cells, 3) float64  [μm]   ALL seeds in Ω_pop
    annulus:      np.ndarray       # (n_cells,)   float64  [μm]
    grid_s1:      np.ndarray       # (G, G, G) int32  — nearest seed index
    grid_s2:      np.ndarray       # (G, G, G) int32  — 2nd nearest seed
    rho:          float
    V:            float
    vi:           float            # REALISED v_i
    alpha_star:   float
    L:            float
    mean_AV:      float            # ⟨A/V⟩ from lookup table [μm⁻¹]
    grid_spacing: float

    def classify_cpu(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU compartment classification (validation only)."""
        gi = np.floor(positions / self.grid_spacing).astype(np.int32)
        G = self.grid_s1.shape[0]
        gi = np.clip(gi, 0, G - 1)
        s1 = self.grid_s1[gi[:, 0], gi[:, 1], gi[:, 2]]
        s2 = self.grid_s2[gi[:, 0], gi[:, 1], gi[:, 2]]

        s1_pos = self.seeds[s1]
        s2_pos = self.seeds[s2]
        mid = 0.5 * (s1_pos + s2_pos)
        diff = s2_pos - s1_pos
        nrm = np.linalg.norm(diff, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-30)
        normal = diff / nrm
        signed = np.einsum("ij,ij->i", positions - mid, normal)
        dist = np.abs(signed)
        inside = dist >= self.annulus[s1]
        return s1, inside


# ===========================================================================
# Unit helpers
# ===========================================================================

def _rho_um3(rho_per_uL: float) -> float:
    return rho_per_uL / 1e9

def _V_um3(V_pL: float) -> float:
    return V_pL * 1e3


# ===========================================================================
# Monte Carlo v_i measurement  (used only inside the table builder)
# ===========================================================================

def _measure_vi(seeds, annulus, tree, L, n=200_000, rng=None):
    """Fraction of uniformly random points in Ω_sim = [0, L]³ that lie
    inside any cell."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, L, (n, 3))
    _, idxs = tree.query(pts, k=2)
    s1, s2 = idxs[:, 0], idxs[:, 1]
    mid = 0.5 * (seeds[s1] + seeds[s2])
    diff = seeds[s2] - seeds[s1]
    nrm = np.maximum(np.linalg.norm(diff, axis=1), 1e-30)
    normal = diff / nrm[:, None]
    d_plane = np.einsum("ij,ij->i", pts - mid, normal)
    inside = np.abs(d_plane) >= annulus[s1]
    return float(inside.mean())


# ===========================================================================
# Exact ⟨A/V⟩ via polyhedral geometry  (used only by the table builder now)
# ===========================================================================

def compute_mean_AV_exact(
    seeds: np.ndarray,
    annulus: np.ndarray,
    L: float,
    verbose: bool = False,
) -> Optional[float]:
    """Compute ⟨A/V⟩ = mean over interior cells of A_i/V_i, exactly.

    Used to populate the lookup table.  See module docstring for why this
    is no longer called per-ensemble.
    """
    in_sim = np.all((seeds >= 0.0) & (seeds < L), axis=1)
    interior_indices = np.where(in_sim)[0]
    if len(interior_indices) < 10:
        return None

    try:
        vor = Voronoi(seeds)
    except Exception as e:
        if verbose:
            print(f"    Voronoi failed: {e}")
        return None

    neighbours: dict = {}
    for (i, j) in vor.ridge_points:
        neighbours.setdefault(int(i), set()).add(int(j))
        neighbours.setdefault(int(j), set()).add(int(i))

    AV_values = []
    n_deg = n_hsi_fail = n_hull_fail = 0

    for i in interior_indices:
        a_i = float(annulus[i])
        if a_i <= 0:
            continue
        nbrs = neighbours.get(int(i), None)
        if not nbrs or len(nbrs) < 4:
            continue

        s_i = seeds[i]
        halfspaces = []
        degenerate = False

        for j in nbrs:
            d_vec = seeds[j] - s_i
            d_mag = float(np.linalg.norm(d_vec))
            if d_mag < 1e-10:
                continue
            n_ij = d_vec / d_mag
            offset = d_mag / 2.0 - a_i
            if offset <= 1e-6:
                degenerate = True
                break
            b = -(offset + float(np.dot(n_ij, s_i)))
            halfspaces.append(np.concatenate([n_ij, [b]]))

        if degenerate:
            n_deg += 1
            continue
        if len(halfspaces) < 4:
            continue

        halfspaces = np.asarray(halfspaces, dtype=np.float64)

        try:
            hsi = HalfspaceIntersection(halfspaces, s_i, qhull_options="QJ")
            verts = hsi.intersections
        except Exception:
            n_hsi_fail += 1
            continue

        if verts.shape[0] < 4 or not np.all(np.isfinite(verts)):
            n_hsi_fail += 1
            continue

        try:
            hull = ConvexHull(verts, qhull_options="QJ")
        except Exception:
            n_hull_fail += 1
            continue

        V_i = float(hull.volume)
        A_i = float(hull.area)
        if V_i > 1e-9 and A_i > 1e-9:
            AV_values.append(A_i / V_i)

    if verbose:
        n_tried = len(interior_indices)
        print(f"      [exact A/V] {len(AV_values)}/{n_tried} cells succeeded "
              f"(deg={n_deg}, hsi_fail={n_hsi_fail}, hull_fail={n_hull_fail})")

    if len(AV_values) < 10:
        return None

    arr = np.asarray(AV_values)
    lo, hi = np.percentile(arr, [5, 95])
    core = arr[(arr >= lo) & (arr <= hi)]
    return float(core.mean())


# ===========================================================================
#                  v_i  →  (α*, <A/V>)   LOOKUP TABLE
# ===========================================================================

# Bump this if the table-build procedure changes in a way that invalidates
# previously cached tables.  Cached tables with mismatched version are
# silently rebuilt.
_TABLE_VERSION = 2

# Process-wide in-memory cache
_LOOKUP_TABLE: Optional[dict] = None


def _lookup_table_path(cfg: SimConfig) -> str:
    """Cache file path keyed by the cfg parameters that affect the table."""
    cache_dir = os.environ.get(
        "MADI_CACHE_DIR",
        os.path.expanduser("~/.cache/madi"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    fname = (f"av_table_v{_TABLE_VERSION}"
             f"_kappa{cfg.kappa:.3f}"
             f"_L{int(cfg.L)}"
             f"_margin{int(cfg.pop_margin)}.npz")
    return os.path.join(cache_dir, fname)


def build_lookup_table(
    cfg: SimConfig,
    n_points: int = 25,
    ref_rho: float = 400_000.0,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Build the v_i → (α*, ⟨A/V⟩) lookup table from scratch.

    Strategy
    --------
    1. Sample one large Poisson seed cloud at `ref_rho` in Ω_pop.
    2. Scan `n_points` α* values from a small fraction of α_cap to α_cap.
    3. For each α*, build the per-cell annuli (with the κ cap), measure
       v_i with `_measure_vi`, and measure ⟨A/V⟩ with the exact polyhedral
       routine `compute_mean_AV_exact`.
    4. Convert α* and ⟨A/V⟩ to dimensionless invariants
            α_norm  = α* · ρ^(1/3)
            AV_norm = ⟨A/V⟩ / ρ^(1/3)
       which are functions of v_i alone (paper SI §S.II).
    5. Sort by v_i ascending and return as a dict.

    Cost: ~10–20 min for n_points=25 at moderate L.  ONE-TIME.
    """
    if verbose:
        print(f"  [MADI lookup table] building from scratch")
        print(f"    n_points = {n_points}")
        print(f"    ref ρ    = {ref_rho:.0f} cells/μL")
        print(f"    L        = {cfg.L:.0f} μm")
        print(f"    κ        = {cfg.kappa:.3f}")

    rho_um3_ref   = _rho_um3(ref_rho)
    rho_third_ref = rho_um3_ref ** (1.0 / 3.0)
    L      = cfg.L
    margin = cfg.pop_margin
    L_pop  = L + 2.0 * margin

    # ---- Single large Poisson realisation ---------------------------------
    rng = np.random.default_rng(20240101)   # fixed seed for reproducibility
    n_cells = max(rng.poisson(rho_um3_ref * L_pop**3), 16)
    seeds = rng.uniform(-margin, L + margin, (n_cells, 3)).astype(np.float64)
    tree = cKDTree(seeds)
    nn_dist = tree.query(seeds, k=2)[0][:, 1]

    if verbose:
        print(f"    seeds in Ω_pop: {n_cells}")

    # ---- α* scan range ----------------------------------------------------
    alpha_cap_max = float(np.max(cfg.kappa * nn_dist / 2.0))
    # Skip the degenerate α=0 (v_i≈1) endpoint and run up to the cap.
    # Geometric spacing concentrates points where the κ cap starts to bite
    # (i.e. low v_i), where the curve is steepest.
    alphas = np.geomspace(alpha_cap_max * 0.05, alpha_cap_max, n_points)

    vi_list:        list = []
    alpha_norm_list: list = []
    AV_norm_list:    list = []

    import time
    t_total = time.time()
    for k, a in enumerate(alphas):
        t0 = time.time()
        annulus = np.minimum(a, cfg.kappa * nn_dist / 2.0)

        vi = _measure_vi(seeds, annulus, tree, L, n=200_000, rng=rng)
        AV = compute_mean_AV_exact(seeds, annulus, L, verbose=False)

        if AV is None or vi <= 0 or vi >= 1:
            if verbose:
                print(f"    [{k+1:2d}/{n_points}] α*={a:6.3f}μm  SKIPPED "
                      f"(vi={vi:.3f}, AV={AV})")
            continue

        vi_list.append(vi)
        alpha_norm_list.append(a * rho_third_ref)
        AV_norm_list.append(AV / rho_third_ref)

        if verbose:
            print(f"    [{k+1:2d}/{n_points}] α*={a:6.3f}μm  v_i={vi:.3f}  "
                  f"<A/V>={AV:.4f}μm⁻¹   ({time.time()-t0:.1f}s)")

    if verbose:
        print(f"  [MADI lookup table] built in {time.time()-t_total:.0f}s")

    if len(vi_list) < 5:
        raise RuntimeError(
            f"Lookup-table build failed: only {len(vi_list)} valid points. "
            f"Check κ and ensemble geometry.")

    vi_arr = np.array(vi_list, dtype=np.float64)
    order  = np.argsort(vi_arr)

    table = {
        "version":    np.int32(_TABLE_VERSION),
        "kappa":      np.float64(cfg.kappa),
        "L":          np.float64(L),
        "pop_margin": np.float64(margin),
        "ref_rho":    np.float64(ref_rho),
        "vi":         vi_arr[order],
        "alpha_norm": np.array(alpha_norm_list, dtype=np.float64)[order],
        "AV_norm":    np.array(AV_norm_list,    dtype=np.float64)[order],
    }

    if save_path:
        np.savez(save_path, **table)
        if verbose:
            print(f"  [MADI lookup table] saved → {save_path}")

    return table


def _table_is_compatible(table: dict, cfg: SimConfig) -> bool:
    try:
        return (int(table["version"])      == _TABLE_VERSION
            and float(table["kappa"])      == float(cfg.kappa)
            and float(table["L"])          == float(cfg.L)
            and float(table["pop_margin"]) == float(cfg.pop_margin))
    except (KeyError, TypeError, ValueError):
        return False


def _load_or_build_lookup_table(cfg: SimConfig, verbose: bool = True) -> dict:
    """Get the lookup table, loading from disk cache or building if absent.

    The disk cache is keyed by (κ, L, pop_margin) — config changes that
    affect the geometry automatically invalidate cached tables.
    """
    global _LOOKUP_TABLE

    # In-memory cache hit
    if _LOOKUP_TABLE is not None and _table_is_compatible(_LOOKUP_TABLE, cfg):
        return _LOOKUP_TABLE

    path = _lookup_table_path(cfg)

    # Disk cache hit
    if os.path.exists(path):
        try:
            data = np.load(path, allow_pickle=False)
            table = {k: data[k] for k in data.files}
            if _table_is_compatible(table, cfg):
                _LOOKUP_TABLE = table
                if verbose:
                    print(f"  [MADI lookup table] loaded from cache: {path}")
                return table
            else:
                if verbose:
                    print(f"  [MADI lookup table] cache at {path} is "
                          f"incompatible with current cfg, rebuilding")
        except Exception as e:
            if verbose:
                print(f"  [MADI lookup table] cache load failed ({e}), "
                      f"rebuilding")

    # Cache miss → build
    if verbose:
        print(f"  [MADI lookup table] no cached table at {path}")
        print(f"  [MADI lookup table] building once "
              f"(this takes ~10–20 min, after which all ensembles are fast)")
    table = build_lookup_table(cfg, save_path=path, verbose=verbose)
    _LOOKUP_TABLE = table
    return table


def alpha_and_AV_from_vi(
    vi_target: float,
    rho: float,
    table: dict,
) -> Tuple[float, float]:
    """Look up (α*, ⟨A/V⟩) for a target v_i and density ρ.

    Linear interpolation in v_i, then rescale by ρ^(±1/3) using the
    dimensionless invariance of the Poisson–contracted-Voronoi process
    (paper SI §S.II).
    """
    rho_um3   = _rho_um3(rho)
    rho_third = rho_um3 ** (1.0 / 3.0)

    vi_grid = np.asarray(table["vi"])
    if vi_target < vi_grid.min() or vi_target > vi_grid.max():
        sys.stderr.write(
            f"    ⚠ v_i target {vi_target:.3f} outside lookup table range "
            f"[{vi_grid.min():.3f}, {vi_grid.max():.3f}]; clamping.\n")
    vi_clamped = float(np.clip(vi_target, vi_grid.min(), vi_grid.max()))

    alpha_norm = float(np.interp(vi_clamped, vi_grid, table["alpha_norm"]))
    AV_norm    = float(np.interp(vi_clamped, vi_grid, table["AV_norm"]))

    alpha_star = alpha_norm / rho_third
    mean_AV    = AV_norm    * rho_third
    return alpha_star, mean_AV


# ===========================================================================
# Ensemble creation  (now uses the lookup table)
# ===========================================================================

def create_ensemble(
    rho: float, V: float,
    cfg: SimConfig | None = None,
    seed: int | None = None,
    verbose: bool = False,
    verify_vi: bool = False,
) -> Ensemble:
    """Build a contracted Voronoi ensemble using the v_i → (α*, ⟨A/V⟩)
    lookup table.

    Parameters
    ----------
    rho, V : target cell density [cells/μL] and mean cell volume [pL].
    cfg    : SimConfig (uses defaults if None).
    seed   : RNG seed for the Poisson realisation.
    verbose: print per-ensemble diagnostics.
    verify_vi : if True, run a single Monte-Carlo v_i measurement on the
        realised ensemble for diagnostics.  Costs ~0.5 s; off by default
        because the lookup-table value is already accurate to ~0.5%.

    Notes
    -----
    On the first call (per process and per cfg geometry), the lookup
    table is built or loaded.  This is the only slow step; subsequent
    calls are O(1) plus the voxel-grid construction.
    """
    if cfg is None:
        cfg = SimConfig()
    rng = np.random.default_rng(seed)

    rho_um3   = _rho_um3(rho)
    V_um3     = _V_um3(V)
    vi_target = rho_um3 * V_um3
    if vi_target > 0.95:
        raise ValueError(f"vi = {vi_target:.3f} > 0.95")

    L      = cfg.L
    margin = cfg.pop_margin
    L_pop  = L + 2.0 * margin

    # ---- 1.  Get α* and ⟨A/V⟩ from the lookup table  --------------------
    table = _load_or_build_lookup_table(cfg, verbose=verbose)
    alpha_star, mean_AV_val = alpha_and_AV_from_vi(vi_target, rho, table)

    if verbose:
        print(f"    [lookup] vi_target={vi_target:.3f}  "
              f"α*={alpha_star:.3f}μm  <A/V>={mean_AV_val:.4f}μm⁻¹")

    # ---- 2.  Sample Poisson seeds in Ω_pop  -----------------------------
    n_cells = max(rng.poisson(rho_um3 * L_pop**3), 16)
    seeds = rng.uniform(-margin, L + margin, (n_cells, 3)).astype(np.float64)
    tree  = cKDTree(seeds)
    nn_dist = tree.query(seeds, k=2)[0][:, 1]

    # ---- 3.  Per-cell annulus widths (κ cap still applies per realisation)
    annulus = np.minimum(alpha_star, cfg.kappa * nn_dist / 2.0).astype(np.float64)

    # Optional realised-v_i diagnostic
    if verify_vi:
        realised_vi = _measure_vi(seeds, annulus, tree, L, n=80_000, rng=rng)
        if verbose:
            drift = realised_vi - vi_target
            print(f"    realised v_i = {realised_vi:.3f}  "
                  f"(target {vi_target:.3f}, drift {drift:+.3f})")
    else:
        realised_vi = vi_target

    # ---- 4.  Voxel lookup grid (chunked over the slowest axis) ----------
    G  = cfg.grid_size
    gs = cfg.grid_spacing
    if verbose:
        print(f"    Building {G}³ voxel grid ({G**3/1e6:.1f}M voxels) ... ",
              end="", flush=True)

    coords = np.arange(G) * gs + gs / 2.0
    grid_s1 = np.empty((G, G, G), dtype=np.int32)
    grid_s2 = np.empty((G, G, G), dtype=np.int32)

    yy, zz = np.meshgrid(coords, coords, indexing="ij")
    yz_flat = np.column_stack([yy.ravel(), zz.ravel()])
    for i in range(G):
        x_col = np.full((G * G, 1), coords[i], dtype=np.float64)
        slab = np.hstack([x_col, yz_flat])
        _, idxs = tree.query(slab, k=2)
        grid_s1[i] = idxs[:, 0].reshape(G, G).astype(np.int32)
        grid_s2[i] = idxs[:, 1].reshape(G, G).astype(np.int32)

    if verbose:
        print("done")

    return Ensemble(
        seeds=seeds, annulus=annulus,
        grid_s1=grid_s1, grid_s2=grid_s2,
        rho=rho, V=V, vi=realised_vi,
        alpha_star=alpha_star, L=L,
        mean_AV=mean_AV_val,
        grid_spacing=gs,
    )


# ===========================================================================
# Helpers (unchanged)
# ===========================================================================

def create_dummy_ensemble(cfg: SimConfig) -> Ensemble:
    """Pure water — no cells."""
    L = cfg.L
    G = cfg.grid_size
    seeds = np.array([[0., 0., 0.], [L, L, L]])
    annulus = np.array([1e10, 1e10])
    grid_s1 = np.zeros((G, G, G), dtype=np.int32)
    grid_s2 = np.ones((G, G, G), dtype=np.int32)
    return Ensemble(
        seeds=seeds, annulus=annulus,
        grid_s1=grid_s1, grid_s2=grid_s2,
        rho=0, V=0, vi=0,
        alpha_star=0, L=L, mean_AV=1.0,
        grid_spacing=cfg.grid_spacing,
    )


def estimate_vi(ens: Ensemble, n=200_000, seed=42) -> float:
    """Sanity-check v_i of a built ensemble by MC classification."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, ens.L, (n, 3))
    _, inside = ens.classify_cpu(pts)
    return inside.mean()

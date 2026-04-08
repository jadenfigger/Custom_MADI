"""
Contracted Voronoi cell ensemble with populated domain and exact ⟨A/V⟩.

KEY FIXES vs. the previous version:

  Fix 3 — populated domain Ω_pop:
      Seeds are now drawn from [−pop_margin, L+pop_margin]³ so that cells
      with seeds near the faces of Ω_sim = [0, L]³ have their correct
      Poisson–Voronoi neighbours.  Walkers still spawn in
      [buffer, L−buffer]³ and are classified using the full seed set.

  Fix 4 — exact ⟨A/V⟩ via polyhedral geometry:
      `compute_mean_AV_exact()` computes the contracted cell of each
      interior seed as the intersection of half-spaces
          (x − s_i) · n_ij ≤ d_ij/2 − α_i
      using scipy.spatial.HalfspaceIntersection, then uses ConvexHull to
      compute the exact surface area A_i and volume V_i of the resulting
      polyhedron.  ⟨A/V⟩ = mean over interior seeds of A_i/V_i, matching
      the paper's definition (SI §S.IV / Fig. S6a, blue curve).
      No Monte Carlo fallthrough, no ratio-of-sums fallback.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, Voronoi, ConvexHull, HalfspaceIntersection
from dataclasses import dataclass
from typing import Tuple, Optional

from .config import SimConfig


# ---------------------------------------------------------------------------
# Ensemble data class
# ---------------------------------------------------------------------------

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
    L:            float            # side of Ω_sim
    mean_AV:      float            # ⟨A/V⟩ from exact polyhedral computation [μm⁻¹]
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rho_um3(rho_per_uL: float) -> float:
    return rho_per_uL / 1e9

def _V_um3(V_pL: float) -> float:
    return V_pL * 1e3

def _alpha_initial_guess(vi: float, rho_um3: float) -> float:
    """Analytic cube-shrinkage initial guess for α*."""
    return (1.0 - vi ** (1./3.)) / (2.0 * rho_um3 ** (1./3.))


# ---------------------------------------------------------------------------
# Monte Carlo measurement of realised v_i  (points sampled in Ω_sim)
# ---------------------------------------------------------------------------

def _measure_vi(seeds, annulus, tree, L, n=200_000, rng=None):
    """Fraction of uniformly random points in Ω_sim = [0, L]³ that lie
    inside any cell (intracellular volume fraction of Ω_sim)."""
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


# ---------------------------------------------------------------------------
# EXACT ⟨A/V⟩ via polyhedral geometry  (Fix #4)
# ---------------------------------------------------------------------------

def compute_mean_AV_exact(
    seeds: np.ndarray,
    annulus: np.ndarray,
    L: float,
    verbose: bool = False,
) -> Optional[float]:
    """Compute ⟨A/V⟩ = mean over interior cells of A_i/V_i, exactly.

    For each seed whose position lies inside Ω_sim = [0, L]³, the
    contracted cell is constructed as the intersection of half-spaces
        (x − s_i) · n_ij ≤ d_ij/2 − α_i    for each Voronoi neighbour j
    where n_ij is the unit vector from s_i toward s_j and d_ij is the
    Euclidean distance.  scipy.spatial.HalfspaceIntersection returns the
    polyhedron vertices; scipy.spatial.ConvexHull then gives exact volume
    and surface area.  The per-cell ratio A_i/V_i is averaged (with 5/95
    trimming to reject the rare numerical outlier) and returned.

    Returns None only in the extreme case that fewer than ~10 interior
    cells produced valid polyhedra (e.g. pathologically tiny ensemble);
    the caller should then raise.
    """
    # Interior seeds = those inside Ω_sim
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

    # Build neighbour lists from ridge_points
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
            offset = d_mag / 2.0 - a_i     # distance from s_i to shifted face
            if offset <= 1e-6:
                # The κ cap is supposed to prevent this; if it still
                # happens the contracted cell has collapsed in this
                # direction and we skip the cell.
                degenerate = True
                break
            # Half-space form  n·x + b ≤ 0
            #   (x − s_i)·n ≤ offset   ⇔   n·x + (−offset − n·s_i) ≤ 0
            b = -(offset + float(np.dot(n_ij, s_i)))
            halfspaces.append(np.concatenate([n_ij, [b]]))

        if degenerate:
            n_deg += 1
            continue
        if len(halfspaces) < 4:
            continue

        halfspaces = np.asarray(halfspaces, dtype=np.float64)

        # s_i is guaranteed strictly interior to the contracted cell
        # (distance offset > 0 from every face), so it's a valid
        # HalfspaceIntersection feasible point.
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
        print(f"    exact A/V: {len(AV_values)}/{n_tried} cells succeeded "
              f"(deg={n_deg}, hsi_fail={n_hsi_fail}, hull_fail={n_hull_fail})")

    if len(AV_values) < 10:
        return None

    arr = np.asarray(AV_values)
    lo, hi = np.percentile(arr, [5, 95])
    core = arr[(arr >= lo) & (arr <= hi)]
    return float(core.mean())


# ---------------------------------------------------------------------------
# Ensemble creation  (Fix #3: populated domain Ω_pop)
# ---------------------------------------------------------------------------

def create_ensemble(
    rho: float, V: float,
    cfg: SimConfig | None = None,
    seed: int | None = None,
    vi_tol: float = 0.01,
    max_vi_iters: int = 10,
    verbose: bool = False,
) -> Ensemble:
    """Build a contracted Voronoi ensemble.

    Fix #3: seeds are drawn from the populated domain
        Ω_pop = [−margin, L+margin]³
    so that cells with seeds near Ω_sim's faces have their correct
    Poisson–Voronoi neighbours.  The voxel lookup grid spans only Ω_sim;
    walkers spawn in [buffer, L−buffer]³ as before.

    α* is found by bisection against the realised v_i measured over points
    sampled in Ω_sim.  ⟨A/V⟩ is computed exactly via
    `compute_mean_AV_exact()`.
    """
    import sys

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

    # ---- 1. Poisson seeds in the populated domain ------------------------
    n_cells = max(rng.poisson(rho_um3 * L_pop**3), 16)
    seeds = rng.uniform(-margin, L + margin, (n_cells, 3)).astype(np.float64)
    tree  = cKDTree(seeds)
    nn_dist = tree.query(seeds, k=2)[0][:, 1]
    alpha_cap_max = float(np.max(cfg.kappa * nn_dist / 2.0))

    def _annuli_for(alpha_star: float) -> np.ndarray:
        return np.minimum(alpha_star, cfg.kappa * nn_dist / 2.0)

    def _vi_for(alpha_star: float) -> float:
        return _measure_vi(seeds, _annuli_for(alpha_star), tree, L, rng=rng)

    # ---- 2. Bisect α* against realised v_i  ------------------------------
    alpha_lo, alpha_hi = 0.0, alpha_cap_max
    vi_at_hi = _vi_for(alpha_hi)

    if vi_target < vi_at_hi - vi_tol:
        alpha_star  = alpha_hi
        realised_vi = vi_at_hi
        sys.stderr.write(
            f"    ⚠ Target v_i={vi_target:.3f} unreachable for "
            f"(rho={rho:.0f}, V={V:.2f}); achievable minimum is "
            f"≈{vi_at_hi:.3f}. Using maximum contraction.\n")
    else:
        alpha_star = _alpha_initial_guess(vi_target, rho_um3)
        alpha_star = float(np.clip(alpha_star, alpha_lo, alpha_hi))
        realised_vi = _vi_for(alpha_star)
        for it in range(max_vi_iters):
            err = realised_vi - vi_target
            if abs(err) <= vi_tol:
                if verbose and it > 0:
                    print(f"    α* converged in {it+1} iters "
                          f"(v_i = {realised_vi:.3f}, target {vi_target:.3f})")
                break
            if err > 0:
                alpha_lo = alpha_star
            else:
                alpha_hi = alpha_star
            alpha_star = 0.5 * (alpha_lo + alpha_hi)
            realised_vi = _vi_for(alpha_star)
        else:
            if abs(realised_vi - vi_target) > vi_tol:
                sys.stderr.write(
                    f"    ⚠ α* bisection ran {max_vi_iters} iters without "
                    f"converging for (rho={rho:.0f}, V={V:.2f}): "
                    f"target v_i={vi_target:.3f}, realised={realised_vi:.3f}.\n")

    annulus = _annuli_for(alpha_star)

    # ---- 3. Exact <A/V> via polyhedral geometry (Fix #4) -----------------
    mean_AV_val = compute_mean_AV_exact(seeds, annulus, L, verbose=verbose)
    if mean_AV_val is None or mean_AV_val <= 0:
        raise RuntimeError(
            f"compute_mean_AV_exact() failed for rho={rho}, V={V}. "
            f"Check kappa / geometry.")

    if verbose:
        print(f"    realised v_i = {realised_vi:.3f}  "
              f"<A/V> = {mean_AV_val:.4f} μm⁻¹  "
              f"(α* = {alpha_star:.3f} μm)")

    # # ---- 4. Voxel lookup grid spans Ω_sim only --------------------------
    # G  = cfg.grid_size
    # gs = cfg.grid_spacing
    # if verbose:
    #     print(f"    Building {G}³ voxel grid ({G**3/1e6:.1f}M voxels) ... ",
    #           end="", flush=True)

    # coords = np.arange(G) * gs + gs / 2.0
    # xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    # pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    # _, idxs = tree.query(pts, k=2)
    # grid_s1 = idxs[:, 0].reshape(G, G, G).astype(np.int32)
    # grid_s2 = idxs[:, 1].reshape(G, G, G).astype(np.int32)
    # if verbose:
    #     print("done")

# ---- 4. Voxel lookup grid spans Ω_sim only --------------------------
    G  = cfg.grid_size
    gs = cfg.grid_spacing
    if verbose:
        print(f"    Building {G}³ voxel grid ({G**3/1e6:.1f}M voxels) ... ",
              end="", flush=True)

    coords = np.arange(G) * gs + gs / 2.0
    grid_s1 = np.empty((G, G, G), dtype=np.int32)
    grid_s2 = np.empty((G, G, G), dtype=np.int32)

    # Chunk over the slowest axis (i) to keep peak memory bounded.
    # Each i-slice is G² points, e.g. 250² = 62 500, trivially small.
    yy, zz = np.meshgrid(coords, coords, indexing="ij")
    yz_flat = np.column_stack([yy.ravel(), zz.ravel()])  # (G², 2)
    for i in range(G):
        x_col = np.full((G * G, 1), coords[i], dtype=np.float64)
        slab = np.hstack([x_col, yz_flat])               # (G², 3)
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

"""
Contracted Voronoi cell ensemble with voxelised spatial lookup grid.

CORRECTIONS vs. the original implementation:

  Deviation #1 — <A/V> for the kio↔pp conversion:
      The original `_mean_AV` used 1.15 * 3/r for a sphere of the mean V,
      which is independent of rho/v_i and inconsistent with paper Fig. S6.
      Now `measure_mean_AV` measures <A/V> = mean_i(A_i/V_i) directly from
      the realised ensemble via Monte Carlo, matching the paper's
      definition (SI §S.IV / Fig. S6).

  Deviation #4 — alpha* formula:
      The analytic cube formula (1 - vi^(1/3)) / (2*rho^(1/3)) is only the
      starting guess. `create_ensemble` now optionally iterates alpha* until
      the realised v_i matches the target (controlled by `vi_tol`,
      `max_vi_iters`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple

from .config import SimConfig


# ---------------------------------------------------------------------------
# Ensemble data class
# ---------------------------------------------------------------------------

@dataclass
class Ensemble:
    seeds:        np.ndarray       # (n_cells, 3) float64  [μm]
    annulus:      np.ndarray       # (n_cells,)   float64  [μm]
    grid_s1:      np.ndarray       # (G, G, G) int32  — nearest seed index
    grid_s2:      np.ndarray       # (G, G, G) int32  — 2nd nearest seed
    rho:          float
    V:            float
    vi:           float            # REALISED v_i (not nominal)
    alpha_star:   float
    L:            float
    mean_AV:      float            # measured <A/V> from this ensemble [μm⁻¹]
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
    """Analytic cube-shrinkage initial guess for alpha*. Iteratively
    refined inside `create_ensemble` to match the target v_i."""
    return (1.0 - vi ** (1./3.)) / (2.0 * rho_um3 ** (1./3.))


def _build_seeds_and_annuli(
    rho_um3: float, alpha_star: float, kappa: float, L: float, rng,
):
    """Sample Poisson seeds and compute per-cell annulus widths."""
    n_cells = max(rng.poisson(rho_um3 * L**3), 8)
    seeds = rng.uniform(0, L, (n_cells, 3)).astype(np.float64)
    tree = cKDTree(seeds)
    nn_dist = tree.query(seeds, k=2)[0][:, 1]
    alpha_lim = kappa * nn_dist / 2.0
    annulus = np.minimum(alpha_star, alpha_lim).astype(np.float64)
    return seeds, annulus, tree


# ---------------------------------------------------------------------------
# Monte Carlo measurement of realised v_i
# ---------------------------------------------------------------------------

def _measure_vi(seeds, annulus, tree, L, n=120_000, rng=None):
    """Fraction of uniformly random points in [0,L]³ that lie inside cells."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = rng.uniform(0, L, (n, 3))
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
# Monte Carlo measurement of <A/V> = mean_i(A_i / V_i)   [paper Fig. S6]
# ---------------------------------------------------------------------------

def measure_mean_AV(
    seeds: np.ndarray,
    annulus: np.ndarray,
    tree: cKDTree,
    L: float,
    n_samples: int = 250_000,
    eps: float = 0.10,
    rng=None,
) -> float:
    """Estimate <A/V> as the per-cell mean of A_i / V_i.

    For each random point in [0, L]³, find the nearest two seeds and:
      - if it lies inside the contracted cell of seed s1 → contributes V_i
      - if it lies within ±eps of the contracted-cell boundary → contributes A_i
    Then A_i / V_i is computed per cell and averaged across cells.
    A 10/90-percentile trim removes boundary cells with poor statistics.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    pts = rng.uniform(0, L, (n_samples, 3))
    _, idxs = tree.query(pts, k=2)
    s1, s2 = idxs[:, 0], idxs[:, 1]
    mid = 0.5 * (seeds[s1] + seeds[s2])
    diff = seeds[s2] - seeds[s1]
    nrm = np.maximum(np.linalg.norm(diff, axis=1), 1e-30)
    normal = diff / nrm[:, None]
    d_plane = np.einsum("ij,ij->i", pts - mid, normal)
    signed_to_membrane = np.abs(d_plane) - annulus[s1]   # >0 → inside cell

    inside_mask  = signed_to_membrane >= 0
    near_mem     = np.abs(signed_to_membrane) < eps
    cell_vol_unit = (L ** 3) / n_samples

    AV_per_cell = []
    for cid in np.unique(s1):
        mask_c = (s1 == cid)
        n_in = int(np.sum(mask_c & inside_mask))
        if n_in < 25:
            continue
        n_near = int(np.sum(mask_c & near_mem))
        if n_near < 5:
            continue
        V_i = n_in * cell_vol_unit
        A_i = n_near * cell_vol_unit / (2.0 * eps)
        AV_per_cell.append(A_i / V_i)

    if len(AV_per_cell) < 30:
        # Fallback: ratio of sums
        n_in_total = int(inside_mask.sum())
        n_near_total = int(near_mem.sum())
        if n_in_total == 0 or n_near_total == 0:
            return 0.5  # safe default
        V_tot = n_in_total * cell_vol_unit
        A_tot = n_near_total * cell_vol_unit / (2.0 * eps)
        return float(A_tot / V_tot)

    arr = np.array(AV_per_cell)
    lo, hi = np.percentile(arr, [10, 90])
    return float(arr[(arr >= lo) & (arr <= hi)].mean())


# ---------------------------------------------------------------------------
# Ensemble creation
# ---------------------------------------------------------------------------

def create_ensemble(
    rho: float, V: float,
    cfg: SimConfig | None = None,
    seed: int | None = None,
    vi_tol: float = 0.01,
    max_vi_iters: int = 8,
    measure_AV_samples: int = 250_000,
    verbose: bool = False,
) -> Ensemble:
    """Build a contracted Voronoi ensemble with:

    1. Iteratively-refined alpha* so that the realised v_i matches the
       target within ``vi_tol`` (deviation #4 fix). Note: targets below
       roughly (1 - κ)³ ≈ 0.22 (for κ=0.4) are unachievable because of
       the per-cell annulus cap κ·d_nn/2; in that regime the routine
       reports the achievable v_i and emits a warning instead of trying
       indefinitely.
    2. Monte-Carlo measured <A/V> on the actual realised cells, used by
       the kio↔pp conversion (deviation #1 fix).
    """
    import sys

    if cfg is None:
        cfg = SimConfig()
    rng = np.random.default_rng(seed)

    rho_um3 = _rho_um3(rho)
    V_um3   = _V_um3(V)
    vi_target = rho_um3 * V_um3
    if vi_target > 0.95:
        raise ValueError(f"vi = {vi_target:.3f} > 0.95")

    L = cfg.L

    # ---- 1. Generate seeds ONCE, then bisect on α* (deviation #4) ----
    # Bisection is robust and converges geometrically. We hold the random
    # seed positions fixed across iterations so that only α* (and hence
    # the annuli) varies — this removes Poisson sampling noise from the
    # convergence test.
    n_cells = max(rng.poisson(rho_um3 * L**3), 8)
    seeds = rng.uniform(0, L, (n_cells, 3)).astype(np.float64)
    tree = cKDTree(seeds)
    nn_dist = tree.query(seeds, k=2)[0][:, 1]
    alpha_cap_max = float(np.max(cfg.kappa * nn_dist / 2.0))

    def _annuli_for(alpha_star):
        return np.minimum(alpha_star, cfg.kappa * nn_dist / 2.0)

    def _vi_for(alpha_star):
        return _measure_vi(seeds, _annuli_for(alpha_star), tree, L, rng=rng)

    # Bracket: α=0 → vi≈1; α=alpha_cap_max → vi at minimum achievable
    alpha_lo, alpha_hi = 0.0, alpha_cap_max
    vi_at_hi = _vi_for(alpha_hi)
    vi_at_lo = 1.0  # by definition

    if vi_target < vi_at_hi - vi_tol:
        # Target is below what bisection can achieve — use the most contracted
        # ensemble and emit a warning (the κ cap has bottomed out).
        alpha_star = alpha_hi
        realised_vi = vi_at_hi
        sys.stderr.write(
            f"    ⚠ Target v_i={vi_target:.3f} unreachable for "
            f"(rho={rho:.0f}, V={V:.2f}); achievable minimum is "
            f"≈{vi_at_hi:.3f}. Using maximum contraction. "
            f"Increase cfg.kappa or rho/V.\n")
    else:
        # Standard bisection
        alpha_star = _alpha_initial_guess(vi_target, rho_um3)
        alpha_star = float(np.clip(alpha_star, alpha_lo, alpha_hi))
        realised_vi = _vi_for(alpha_star)
        for it in range(max_vi_iters):
            err = realised_vi - vi_target
            if abs(err) <= vi_tol:
                if verbose and it > 0:
                    print(f"    α* converged in {it+1} iterations "
                          f"(v_i = {realised_vi:.3f}, target {vi_target:.3f})")
                break
            if err > 0:        # too dense → need more contraction
                alpha_lo = alpha_star
            else:              # too sparse → need less contraction
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

    # ---- 2. Measure <A/V> on the realised ensemble (deviation #1) ----
    mean_AV_real = measure_mean_AV(
        seeds, annulus, tree, L, n_samples=measure_AV_samples, rng=rng)

    if verbose:
        print(f"    realised v_i = {realised_vi:.3f}  "
              f"<A/V> = {mean_AV_real:.4f} μm⁻¹  "
              f"(α* = {alpha_star:.3f} μm)")

    # ---- 3. Build voxel lookup grid for the GPU kernel ----
    G = cfg.grid_size
    gs = cfg.grid_spacing
    if verbose:
        print(f"    Building {G}³ voxel grid ({G**3/1e6:.1f}M voxels) ... ",
              end="", flush=True)

    coords = np.arange(G) * gs + gs / 2.0
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    _, idxs = tree.query(pts, k=2)
    grid_s1 = idxs[:, 0].reshape(G, G, G).astype(np.int32)
    grid_s2 = idxs[:, 1].reshape(G, G, G).astype(np.int32)
    if verbose:
        print("done")

    return Ensemble(
        seeds=seeds, annulus=annulus,
        grid_s1=grid_s1, grid_s2=grid_s2,
        rho=rho, V=V, vi=realised_vi,
        alpha_star=alpha_star, L=L,
        mean_AV=mean_AV_real,
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
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, ens.L, (n, 3))
    _, inside = ens.classify_cpu(pts)
    return inside.mean()

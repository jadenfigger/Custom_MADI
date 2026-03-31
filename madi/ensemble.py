"""
Contracted Voronoi cell ensemble with voxelised spatial lookup grid.

The grid stores (nearest_seed, second_nearest_seed) indices at each
voxel for O(1) compartment classification on the GPU.
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
    vi:           float
    alpha_star:   float
    L:            float
    mean_AV:      float
    grid_spacing: float

    def classify_cpu(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU compartment classification (for validation)."""
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
# Creation functions
# ---------------------------------------------------------------------------

def _rho_um3(rho_per_uL: float) -> float:
    return rho_per_uL / 1e9

def _V_um3(V_pL: float) -> float:
    return V_pL * 1e3

def _alpha_from_vi(vi: float, rho_um3: float) -> float:
    return (1.0 - vi ** (1./3.)) / (2.0 * rho_um3 ** (1./3.))

def _mean_AV(V_um3: float) -> float:
    r = (3.0 * V_um3 / (4.0 * np.pi)) ** (1./3.)
    return 1.15 * 3.0 / r


def create_ensemble(
    rho: float, V: float,
    cfg: SimConfig | None = None,
    seed: int | None = None,
) -> Ensemble:
    """Build ensemble with pre-computed voxel lookup grid."""
    if cfg is None:
        cfg = SimConfig()
    rng = np.random.default_rng(seed)

    ru3 = _rho_um3(rho)
    vu3 = _V_um3(V)
    vi = ru3 * vu3
    if vi > 0.95:
        raise ValueError(f"vi = {vi:.3f} > 0.95")

    L = cfg.L
    n_cells = max(rng.poisson(int(ru3 * L**3)), 2)
    seeds = rng.uniform(0, L, (n_cells, 3)).astype(np.float64)

    tree = cKDTree(seeds)

    alpha_star = _alpha_from_vi(vi, ru3)
    dd, _ = tree.query(seeds, k=2)
    nn_dist = dd[:, 1]
    alpha_lim = cfg.kappa * nn_dist / 2.0
    annulus = np.minimum(alpha_star, alpha_lim).astype(np.float64)

    # Build voxelised grid
    G = cfg.grid_size
    gs = cfg.grid_spacing
    print(f"    Building {G}³ voxel grid ({G**3/1e6:.1f}M voxels) ... ", end="", flush=True)

    coords = np.arange(G) * gs + gs / 2.0
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    dists, idxs = tree.query(pts, k=2)
    grid_s1 = idxs[:, 0].reshape(G, G, G).astype(np.int32)
    grid_s2 = idxs[:, 1].reshape(G, G, G).astype(np.int32)
    print("done")

    return Ensemble(
        seeds=seeds, annulus=annulus,
        grid_s1=grid_s1, grid_s2=grid_s2,
        rho=rho, V=V, vi=vi,
        alpha_star=alpha_star, L=L,
        mean_AV=_mean_AV(vu3),
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

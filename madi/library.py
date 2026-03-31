"""
MADI library: build a lookup table of simulated signals indexed by
(k_io, ρ, V), then match experimental voxel data to estimate parameters.
"""

from __future__ import annotations

import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Optional

from .config     import SimConfig, BVALS_UNIQUE
from .walker_gpu import run_simulation
from .signal     import compute_signals, signals_to_flat


# ---------------------------------------------------------------------------
# Library entry
# ---------------------------------------------------------------------------

@dataclass
class LibraryEntry:
    kio:    float
    rho:    float
    V:      float
    vector: np.ndarray     # flattened signal (n_deltas × n_b,)


# ---------------------------------------------------------------------------
# Default parameter grids (mouse brain in-vivo)
# ---------------------------------------------------------------------------

DEFAULT_KIOS = [2, 5, 8, 12, 18, 25, 35, 50, 75, 100]           # s⁻¹
DEFAULT_RHOS = [100_000, 200_000, 400_000, 600_000, 800_000]     # cells/μL
DEFAULT_VS   = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]                   # pL


def build_library(
    kios: list | None = None,
    rhos: list | None = None,
    Vs:   list | None = None,
    cfg:  SimConfig | None = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> list[LibraryEntry]:
    """Build a MADI simulation library.

    Parameters
    ----------
    kios, rhos, Vs : lists of parameter values to sample.
    cfg : SimConfig
    save_path : str, optional
        If given, saves the library as a .npz file.

    Returns
    -------
    list of LibraryEntry
    """
    if kios is None: kios = DEFAULT_KIOS
    if rhos is None: rhos = DEFAULT_RHOS
    if Vs is None:   Vs = DEFAULT_VS
    if cfg is None:  cfg = SimConfig()

    total = len(kios) * len(rhos) * len(Vs)
    library = []
    idx = 0

    t0 = time.time()
    for kio in kios:
        for rho in rhos:
            for V in Vs:
                idx += 1
                vi = (rho / 1e9) * (V * 1e3)
                if vi > 0.95:
                    if verbose:
                        print(f"  [{idx}/{total}] SKIP  kio={kio}, rho={rho}, V={V} (vi={vi:.2f})")
                    continue

                if verbose:
                    print(f"  [{idx}/{total}] kio={kio}, rho={rho/1e3:.0f}k, V={V:.1f}  ...", end="", flush=True)

                tt = time.time()
                wr = run_simulation(rho, V, kio, cfg, seed=idx * 100, verbose=False)
                res = compute_signals(wr, cfg)
                vec = signals_to_flat(res)
                library.append(LibraryEntry(kio=kio, rho=rho, V=V, vector=vec))

                if verbose:
                    print(f"  {time.time()-tt:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nLibrary built: {len(library)} entries in {elapsed:.0f}s")

    if save_path:
        _save_library(library, save_path)
        if verbose:
            print(f"Saved to {save_path}")

    return library


def _save_library(lib: list[LibraryEntry], path: str):
    """Save library to .npz."""
    kios = np.array([e.kio for e in lib])
    rhos = np.array([e.rho for e in lib])
    Vs   = np.array([e.V for e in lib])
    vecs = np.array([e.vector for e in lib])
    np.savez(path, kios=kios, rhos=rhos, Vs=Vs, vectors=vecs)


def load_library(path: str) -> list[LibraryEntry]:
    """Load library from .npz."""
    data = np.load(path)
    lib = []
    for i in range(len(data['kios'])):
        lib.append(LibraryEntry(
            kio=float(data['kios'][i]),
            rho=float(data['rhos'][i]),
            V=float(data['Vs'][i]),
            vector=data['vectors'][i],
        ))
    return lib


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_voxel(
    measured: np.ndarray,
    library:  list[LibraryEntry],
) -> tuple[LibraryEntry, float]:
    """Find the library entry closest to a measured signal vector.

    Parameters
    ----------
    measured : ndarray (n_deltas × n_b,)
        Normalised signal from one voxel (same shape as library vectors).
    library : list of LibraryEntry

    Returns
    -------
    best_entry : LibraryEntry
    residual : float (sum of squared differences)
    """
    best = None
    best_res = np.inf
    for entry in library:
        res = np.sum((measured - entry.vector) ** 2)
        if res < best_res:
            best_res = res
            best = entry
    return best, best_res


def match_voxels_batch(
    measured_batch: np.ndarray,
    library: list[LibraryEntry],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised matching for many voxels.

    Parameters
    ----------
    measured_batch : ndarray (n_voxels, n_features)
    library : list of LibraryEntry

    Returns
    -------
    kio_map, rho_map, V_map, residual_map : each (n_voxels,)
    """
    lib_mat = np.array([e.vector for e in library])      # (n_lib, n_feat)
    kios = np.array([e.kio for e in library])
    rhos = np.array([e.rho for e in library])
    Vs   = np.array([e.V for e in library])

    # Compute all pairwise squared distances
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    m2 = np.sum(measured_batch ** 2, axis=1, keepdims=True)
    l2 = np.sum(lib_mat ** 2, axis=1, keepdims=True).T
    dists = m2 + l2 - 2.0 * measured_batch @ lib_mat.T   # (n_vox, n_lib)

    best_idx = np.argmin(dists, axis=1)
    kio_map = kios[best_idx]
    rho_map = rhos[best_idx]
    V_map   = Vs[best_idx]
    res_map = dists[np.arange(len(best_idx)), best_idx]

    return kio_map, rho_map, V_map, res_map

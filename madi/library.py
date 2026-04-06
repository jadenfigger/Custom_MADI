"""
MADI library: build a lookup table of simulated signals indexed by
(k_io, ρ, V), then match experimental voxel data to estimate parameters.

Supports:
  - Full preset grids (kio × rho × V cross-product)
  - Explicit sub-grids (only specified kio × rho × V crossed)
  - Individual (kio, rho, V) triplets
  - Appending to existing libraries (skips duplicates)
  - Fitting with any subset of Δ values
"""

from __future__ import annotations

import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .config     import SimConfig, BVALS_UNIQUE, DELTAS_BIG
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
# Default parameter grids
# ---------------------------------------------------------------------------

DEFAULT_KIOS = [2, 5, 8, 12, 18, 25, 35, 50, 75, 100]
DEFAULT_RHOS = [100_000, 200_000, 400_000, 600_000, 800_000]
DEFAULT_VS   = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry_key(kio: float, rho: float, V: float) -> tuple:
    return (round(kio, 4), round(rho, 1), round(V, 6))

def _existing_keys(library: list[LibraryEntry]) -> set:
    return {_entry_key(e.kio, e.rho, e.V) for e in library}

def _filter_valid(triplets, vi_max=0.95):
    return [(k, r, v) for k, r, v in triplets
            if (r / 1e9) * (v * 1e3) <= vi_max]


# ---------------------------------------------------------------------------
# Core builder: works on an explicit list of (kio, rho, V) triplets
# ---------------------------------------------------------------------------

def build_library_from_triplets(
    triplets: List[Tuple[float, float, float]],
    cfg: SimConfig | None = None,
    save_path: Optional[str] = None,
    existing_library: Optional[list[LibraryEntry]] = None,
    verbose: bool = True,
) -> list[LibraryEntry]:
    """Build/extend library from an explicit list of (kio, rho, V) triplets.

    Skips any triplet already present in existing_library.
    """
    if cfg is None:
        cfg = SimConfig()

    if existing_library is not None:
        library = list(existing_library)
        done = _existing_keys(library)
        if verbose:
            print(f"  Loaded {len(library)} existing entries")
    else:
        library = []
        done = set()

    valid = _filter_valid(triplets)
    new_triplets = [t for t in valid if _entry_key(*t) not in done]

    if verbose:
        print(f"  Requested: {len(triplets)} triplets")
        print(f"  Skipped: {len(triplets)-len(valid)} (vi>0.95) "
              f"+ {len(valid)-len(new_triplets)} (already exist)")
        print(f"  New to compute: {len(new_triplets)}")

    if not new_triplets:
        if verbose:
            print("  Nothing new to compute!")
        if save_path:
            _save_library(library, save_path)
        return library

    t0 = time.time()
    for idx, (kio, rho, V) in enumerate(new_triplets):
        if verbose:
            print(f"  [{idx+1}/{len(new_triplets)}] kio={kio}, "
                  f"rho={rho/1e3:.0f}k, V={V:.2f}  ...", end="", flush=True)

        tt = time.time()
        seed = int(abs(hash((kio, rho, V)))) % (2**31)
        wr = run_simulation(rho, V, kio, cfg, seed=seed, verbose=False)
        res = compute_signals(wr, cfg)
        vec = signals_to_flat(res)
        library.append(LibraryEntry(kio=kio, rho=rho, V=V, vector=vec))

        if verbose:
            print(f"  {time.time()-tt:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nLibrary: {len(library)} entries total "
              f"({len(new_triplets)} new in {elapsed:.0f}s)")

    if save_path:
        _save_library(library, save_path)
        if verbose:
            print(f"Saved to {save_path}")

    return library


# ---------------------------------------------------------------------------
# Convenience: build from full cross-product grid
# ---------------------------------------------------------------------------

def build_library(
    kios=None, rhos=None, Vs=None,
    cfg=None, save_path=None, existing_library=None, verbose=True,
) -> list[LibraryEntry]:
    """Build/extend library from kio × rho × V grid (full cross-product)."""
    if kios is None: kios = DEFAULT_KIOS
    if rhos is None: rhos = DEFAULT_RHOS
    if Vs is None:   Vs = DEFAULT_VS

    triplets = [(k, r, v) for k in kios for r in rhos for v in Vs]
    if verbose:
        print(f"  Grid: {len(kios)} kio × {len(rhos)} rho × {len(Vs)} V "
              f"= {len(triplets)} combos")

    return build_library_from_triplets(
        triplets, cfg=cfg, save_path=save_path,
        existing_library=existing_library, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _save_library(lib: list[LibraryEntry], path: str):
    kios = np.array([e.kio for e in lib])
    rhos = np.array([e.rho for e in lib])
    Vs   = np.array([e.V for e in lib])
    vecs = np.array([e.vector for e in lib])
    deltas = np.array(DELTAS_BIG)
    n_b = len(BVALS_UNIQUE)
    np.savez(path, kios=kios, rhos=rhos, Vs=Vs, vectors=vecs,
             deltas=deltas, n_b=np.array(n_b))


def load_library(path: str) -> list[LibraryEntry]:
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


def load_library_meta(path: str) -> dict:
    """Load metadata (deltas, n_b) from library file."""
    data = np.load(path)
    meta = {}
    meta['deltas'] = list(data['deltas']) if 'deltas' in data else list(DELTAS_BIG)
    meta['n_b'] = int(data['n_b']) if 'n_b' in data else len(BVALS_UNIQUE)
    return meta


def library_summary(lib: list[LibraryEntry]):
    if not lib:
        print("  (empty library)")
        return
    kios = sorted(set(e.kio for e in lib))
    rhos = sorted(set(e.rho for e in lib))
    Vs   = sorted(set(e.V for e in lib))
    vis  = sorted(set(round((e.rho/1e9)*(e.V*1e3), 4) for e in lib))
    print(f"  Entries: {len(lib)}")
    print(f"  kio  ({len(kios)}): {[f'{k:.1f}' for k in kios]}")
    print(f"  rho  ({len(rhos)}): {[f'{r/1e3:.0f}k' for r in rhos]}")
    print(f"  V    ({len(Vs)}):   {[f'{v:.2f}' for v in Vs]}")
    print(f"  vi range: [{min(vis):.3f}, {max(vis):.3f}]")
    n_b = lib[0].vector.size
    n_deltas = len(DELTAS_BIG)
    print(f"  Vector length: {n_b}  "
          f"({n_b // n_deltas} b-values × {n_deltas} Δ values)")


# ---------------------------------------------------------------------------
# Matching (with delta subsetting)
# ---------------------------------------------------------------------------

def _delta_indices(fit_deltas: list[float],
                   lib_deltas: list[float]) -> list[int]:
    """Map fitting Δ values to their indices in the library vector."""
    indices = []
    for d in fit_deltas:
        found = False
        for i, ld in enumerate(lib_deltas):
            if abs(d - ld) < 0.01:
                indices.append(i)
                found = True
                break
        if not found:
            raise ValueError(
                f"Δ = {d} ms not in library deltas {lib_deltas}. "
                f"Rebuild the library with this Δ.")
    return indices


def _subset_vectors(lib_mat: np.ndarray, delta_indices: list[int],
                    n_b: int) -> np.ndarray:
    """Extract columns for requested Δ indices from all library vectors."""
    cols = []
    for di in delta_indices:
        cols.append(lib_mat[:, di * n_b : (di + 1) * n_b])
    return np.hstack(cols)


def match_voxels_batch(
    measured_batch: np.ndarray,
    library: list[LibraryEntry],
    fit_deltas: list[float] | None = None,
    lib_deltas: list[float] | None = None,
    n_b: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised nearest-neighbour matching.

    Parameters
    ----------
    measured_batch : (n_voxels, n_fit_deltas * n_b)
    library : list of LibraryEntry
    fit_deltas : Δ values [ms] present in data.  None = use all.
    lib_deltas : all Δ values in library.  None = DELTAS_BIG.
    n_b : b-values per Δ (default 4).

    Returns
    -------
    kio_map, rho_map, V_map, residual_map : each (n_voxels,)
    """
    if lib_deltas is None:
        lib_deltas = list(DELTAS_BIG)

    full_mat = np.array([e.vector for e in library])  # (n_lib, full_len)

    if fit_deltas is not None:
        di_idx = _delta_indices(fit_deltas, lib_deltas)
        lib_mat = _subset_vectors(full_mat, di_idx, n_b)
    else:
        lib_mat = full_mat

    kios = np.array([e.kio for e in library])
    rhos = np.array([e.rho for e in library])
    Vs   = np.array([e.V for e in library])

    m2 = np.sum(measured_batch ** 2, axis=1, keepdims=True)
    l2 = np.sum(lib_mat ** 2, axis=1, keepdims=True).T
    dists = m2 + l2 - 2.0 * measured_batch @ lib_mat.T

    best_idx = np.argmin(dists, axis=1)
    return kios[best_idx], rhos[best_idx], Vs[best_idx], \
           dists[np.arange(len(best_idx)), best_idx]

"""
MADI library: build a lookup table of simulated signals indexed by
(k_io, ρ, V), then match experimental voxel data to estimate parameters.
 
NEW IN THIS VERSION
-------------------
match_voxels_batch_fits0()
    Same library/candidate filtering as match_voxels_batch, but operates
    on UN-NORMALIZED measured signals and treats S0 as a free linear
    parameter per voxel.  For each candidate library entry r (a vector
    of simulated S/S0 ratios), the L2-optimal S0 is
 
        S0* = (M . r) / (r . r)
 
    and the residual is
 
        ||M - S0* r||^2 = ||M||^2 - (M.r)^2 / (r.r)
 
    This is fully vectorizable and adds only one cheap inner-product
    per (voxel, entry) compared with the fixed-S0 matcher.
 
    The fitted S0 is returned alongside the parameter maps.
"""
 
from __future__ import annotations
 
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
 
from collections import defaultdict
 
from .config     import SimConfig, BVALS_UNIQUE, DELTAS_BIG
from .walker_gpu import run_simulation, run_simulation_multi_kio
from .signal     import compute_signals, signals_to_flat
 
 
# ---------------------------------------------------------------------------
# Library entry
# ---------------------------------------------------------------------------
 
@dataclass
class LibraryEntry:
    kio:    float
    rho:    float
    V:      float
    vector: np.ndarray
 
 
DEFAULT_KIOS = [2, 5, 8, 12, 18, 25, 35, 50, 75, 100]
DEFAULT_RHOS = [100_000, 200_000, 400_000, 600_000, 800_000]
DEFAULT_VS   = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
 
 
def _entry_key(kio, rho, V):
    return (round(kio, 4), round(rho, 1), round(V, 6))
 
def _existing_keys(library):
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

    # -----------------------------------------------------------------
    # Group triplets by (rho, V).  Ensemble geometry depends ONLY on
    # (rho, V); kio only affects the membrane-crossing probability in
    # the walk kernel.  So we build each ensemble once and sweep all
    # kios for that (rho, V) on it — ~N_kios× speedup on the CPU
    # scipy Voronoi / HalfspaceIntersection cost, which dominates at
    # high ρ.
    # -----------------------------------------------------------------
    groups: dict = defaultdict(list)
    for kio, rho, V in new_triplets:
        groups[(rho, V)].append(kio)

    if verbose:
        print(f"  Grouped into {len(groups)} unique (ρ,V) pairs "
              f"(ensembles reused across kio values)")

    # Process groups in order of increasing cost proxy so a shard makes
    # progress on cheap entries first — makes checkpointing useful.
    sorted_groups = sorted(groups.items(), key=lambda kv: kv[0][0] * kv[0][1])

    entry_idx = 0
    for (rho, V), kios_for_group in sorted_groups:
        kios_for_group = sorted(kios_for_group)

        if verbose:
            print(f"  [(ρ,V)=({rho/1e3:.0f}k, {V:.2f})] "
                  f"{len(kios_for_group)} kio values "
                  f"→ {[f'{k:g}' for k in kios_for_group]}",
                  flush=True)

        tt = time.time()
        geom_seed = int(abs(hash((round(rho, 1), round(V, 6))))) % (2**31)

        results = run_simulation_multi_kio(
            rho, V, kios_for_group, cfg,
            seed=geom_seed, verbose=False,
        )

        for kio in kios_for_group:
            res = compute_signals(results[kio], cfg)
            vec = signals_to_flat(res)
            library.append(LibraryEntry(kio=kio, rho=rho, V=V, vector=vec))
            entry_idx += 1

        dt = time.time() - tt
        if verbose:
            print(f"    → {len(kios_for_group)} entries in {dt:.1f}s "
                  f"({dt/len(kios_for_group):.1f}s/entry)  "
                  f"[{entry_idx}/{len(new_triplets)} done]",
                  flush=True)

        # Checkpoint after every (rho, V) group — cheap insurance
        # against SLURM preemption / walltime kills.
        if save_path:
            _save_library(library, save_path)

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
# Matching helpers
# ---------------------------------------------------------------------------
 
def _delta_indices(fit_deltas, lib_deltas):
    indices = []
    for d in fit_deltas:
        for i, ld in enumerate(lib_deltas):
            if abs(d - ld) < 0.01:
                indices.append(i); break
        else:
            raise ValueError(f"Δ = {d} ms not in library deltas {lib_deltas}.")
    return indices
 
 
def _subset_vectors(lib_mat, delta_indices, n_b):
    return np.hstack([lib_mat[:, di * n_b : (di + 1) * n_b] for di in delta_indices])
 
 
def _build_candidate_lib_matrix(library, fit_deltas, lib_deltas,
                                 n_b, vi_min, vi_max, rho_max):
    """Apply candidate filtering and produce the masked, Δ-subset library matrix.
 
    Returns
    -------
    lib_mat : (n_candidates, n_fit_deltas * n_b)
    kios_arr, rhos_arr, Vs_arr : (n_candidates,)
    """
    if lib_deltas is None:
        lib_deltas = list(DELTAS_BIG)
 
    vis  = np.array([(e.rho / 1e9) * (e.V * 1e3) for e in library])
    rhos = np.array([e.rho for e in library])
 
    mask = (vis >= vi_min) & (vis <= vi_max)
    if rho_max is not None:
        mask &= (rhos <= rho_max)
 
    n_candidates = int(mask.sum())
    if n_candidates == 0:
        raise ValueError(
            f"No library entries survive vi in [{vi_min}, {vi_max}] "
            f"and rho <= {rho_max}.")
    if n_candidates < 50:
        import warnings
        warnings.warn(f"Only {n_candidates} library entries pass the filter.")
 
    lib_entries = [e for e, m in zip(library, mask) if m]
    full_mat    = np.array([e.vector for e in lib_entries])
 
    if fit_deltas is not None:
        di_idx  = _delta_indices(fit_deltas, lib_deltas)
        lib_mat = _subset_vectors(full_mat, di_idx, n_b)
    else:
        lib_mat = full_mat
 
    kios_arr = np.array([e.kio for e in lib_entries])
    rhos_arr = np.array([e.rho for e in lib_entries])
    Vs_arr   = np.array([e.V   for e in lib_entries])
 
    return lib_mat, kios_arr, rhos_arr, Vs_arr
 

# def match_voxels_batch(
#     measured_batch: np.ndarray,
#     library: list[LibraryEntry],
#     fit_deltas: list[float] | None = None,
#     lib_deltas: list[float] | None = None,
#     n_b: int = 4,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Vectorised nearest-neighbour matching.

#     Parameters
#     ----------
#     measured_batch : (n_voxels, n_fit_deltas * n_b)
#     library : list of LibraryEntry
#     fit_deltas : Δ values [ms] present in data.  None = use all.
#     lib_deltas : all Δ values in library.  None = DELTAS_BIG.
#     n_b : b-values per Δ (default 4).

#     Returns
#     -------
#     kio_map, rho_map, V_map, residual_map : each (n_voxels,)
#     """
#     if lib_deltas is None:
#         lib_deltas = list(DELTAS_BIG)

#     full_mat = np.array([e.vector for e in library])  # (n_lib, full_len)

#     if fit_deltas is not None:
#         di_idx = _delta_indices(fit_deltas, lib_deltas)
#         lib_mat = _subset_vectors(full_mat, di_idx, n_b)
#     else:
#         lib_mat = full_mat

#     kios = np.array([e.kio for e in library])
#     rhos = np.array([e.rho for e in library])
#     Vs   = np.array([e.V for e in library])

#     m2 = np.sum(measured_batch ** 2, axis=1, keepdims=True)
#     l2 = np.sum(lib_mat ** 2, axis=1, keepdims=True).T
#     dists = m2 + l2 - 2.0 * measured_batch @ lib_mat.T

#     best_idx = np.argmin(dists, axis=1)
#     return kios[best_idx], rhos[best_idx], Vs[best_idx], \
#            dists[np.arange(len(best_idx)), best_idx]

# ---------------------------------------------------------------------------
# FIXED-S0 matcher (existing behavior, kept)
# ---------------------------------------------------------------------------
 
def match_voxels_batch(
    measured_batch,
    library,
    fit_deltas=None, lib_deltas=None, n_b=4,
    log_space=True, s_floor=1e-3,
    vi_min=0.5, vi_max=0.95, rho_max=None,
):
    """Log-space nearest-neighbour matching, S0 fixed (data already
    divided by measured b=0).
 
    Inputs are S/S0 ratios.
    """
    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, fit_deltas, lib_deltas, n_b, vi_min, vi_max, rho_max)
 
    if log_space:
        measured = np.log(np.clip(measured_batch, s_floor, 1.0))
        lib_m    = np.log(np.clip(lib_mat,         s_floor, 1.0))
    else:
        measured = measured_batch
        lib_m    = lib_mat
 
    m2 = np.sum(measured ** 2, axis=1, keepdims=True)
    l2 = np.sum(lib_m   ** 2, axis=1, keepdims=True).T
    dists = m2 + l2 - 2.0 * measured @ lib_m.T
 
    best_idx = np.argmin(dists, axis=1)
    return (kios_arr[best_idx], rhos_arr[best_idx], Vs_arr[best_idx],
            dists[np.arange(len(best_idx)), best_idx])
 
 
# ---------------------------------------------------------------------------
# FREE-S0 matcher (new)
# ---------------------------------------------------------------------------
 
def match_voxels_batch_fits0(
    raw_signal,
    library,
    fit_deltas=None, lib_deltas=None, n_b=4,
    vi_min=0.5, vi_max=0.95, rho_max=None,
):
    """Match un-normalized signals with S0 as a free per-voxel linear param.
 
    For each voxel m and each candidate library ratio vector r,
 
        S0*(m, r) = (m . r) / (r . r)
        residual  = ||m||^2  -  (m . r)^2 / (r . r)
 
    The matcher picks the (kio, rho, V) entry that minimises the residual,
    and reports the corresponding S0*.
 
    Notes
    -----
    - Uses LINEAR-space L2 (not log).  Log-space cannot accommodate a free
      multiplicative scale cleanly because log(S0*r) = log(S0) + log(r)
      makes S0 an additive offset, but then negative residuals from noise
      can blow up under clipping.  Linear L2 with analytic S0 is the
      cleanest and is widely used (Walsh et al., Jiang et al.).
    - Adds one degree of freedom (S0) per voxel, so residuals are always
      smaller than the fixed-S0 matcher even when the model is wrong.
      Compare maps from both modes; large parameter shifts indicate
      either S0 problems in the data OR that the model can't fit some
      voxels and was being held in check by the S0 constraint.
 
    Returns
    -------
    kio_map, rho_map, V_map, residual_map, s0_fit_map  (each shape (n_voxels,))
    """
    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, fit_deltas, lib_deltas, n_b, vi_min, vi_max, rho_max)
 
    M = raw_signal.astype(np.float64)            # (n_vox, n_feat)
    R = lib_mat.astype(np.float64)               # (n_lib, n_feat)
 
    # Per-library-entry  r.r   shape (n_lib,)
    rr = np.sum(R * R, axis=1)
    rr = np.maximum(rr, 1e-30)
 
    # Per-voxel  m.m  shape (n_vox,)
    mm = np.sum(M * M, axis=1)
 
    # Cross  M @ R.T  shape (n_vox, n_lib)
    MR = M @ R.T
 
    # S0 candidates per (voxel, library entry)   shape (n_vox, n_lib)
    S0_cand = MR / rr[None, :]
 
    # Residuals  shape (n_vox, n_lib)
    #   ||m||^2 - (m.r)^2 / (r.r)
    resid = mm[:, None] - (MR ** 2) / rr[None, :]
 
    # Forbid negative S0 (would correspond to flipping the signal)
    resid_masked = np.where(S0_cand > 0, resid, np.inf)
 
    best_idx = np.argmin(resid_masked, axis=1)
    rows = np.arange(len(best_idx))
 
    return (kios_arr[best_idx],
            rhos_arr[best_idx],
            Vs_arr[best_idx],
            resid_masked[rows, best_idx],
            S0_cand[rows, best_idx])
 
"""
MADI library: build a lookup table of simulated signals indexed by
(k_io, ρ, V), then match experimental voxel data to estimate parameters.

(δ, Δ, b)-UNIVERSAL LIBRARY
----------------------------
Each `LibraryEntry.vector` is now a flattened S[δ,Δ,b] block (row-major:
pair-major then b, matching `madi.signal.ColumnGrid`) instead of a
fixed-δ, multi-Δ vector. One MC walk per (ρ,V) ensemble (reused across the
kio sweep, per `build_lookup_table`'s ensemble-reuse — see
`walker_gpu.run_simulation_multi_kio_reduced`) fills the WHOLE (δ,Δ,b)
grid, so building the library no longer requires re-simulating per δ/Δ.

Matching is NEAREST-COLUMN, never interpolated: a measured (δ,Δ,b) that
doesn't exactly land on the library's stored grid is matched to its nearest
stored column and the resulting mismatch is accepted as error (by design —
see `_grid_columns`).

Legacy fixed-δ `.npz` libraries (built before this refactor) can still be
read — `load_library_meta` detects the old format and synthesizes an
equivalent `delta_pairs` list so downstream matching code is format-
agnostic.

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

from .config       import SimConfig, DELTA_SMALL, DELTAS_BIG, BVALS_UNIQUE
from . import signal as sig
from . import fitters_gpu


# ---------------------------------------------------------------------------
# Library entry
# ---------------------------------------------------------------------------

@dataclass
class LibraryEntry:
    kio:    float
    rho:    float
    V:      float
    vector: np.ndarray   # flat S[δ,Δ,b].ravel() — pair-major then b


DEFAULT_KIOS = [2, 5, 8, 12, 18, 25, 35, 50, 75, 100]
DEFAULT_RHOS = [100_000, 200_000, 400_000, 600_000, 800_000]
DEFAULT_VS   = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


def _entry_key(kio, rho, V):
    return (round(kio, 4), round(rho, 1), round(V, 6))

def _existing_keys(library):
    return {_entry_key(e.kio, e.rho, e.V) for e in library}

# Hard physical ceiling: create_ensemble() raises for vi > 0.95, so no build
# may request an entry above it regardless of the caller's vi_max.
VI_HARD_MAX = 0.95


def _filter_valid(triplets, vi_min=0.0, vi_max=VI_HARD_MAX):
    hi = min(vi_max, VI_HARD_MAX)
    return [(k, r, v) for k, r, v in triplets
            if vi_min <= (r / 1e9) * (v * 1e3) <= hi]


# ---------------------------------------------------------------------------
# Core builder: works on an explicit list of (kio, rho, V) triplets
# ---------------------------------------------------------------------------

def build_library_from_triplets(
    triplets: List[Tuple[float, float, float]],
    cfg: SimConfig | None = None,
    save_path: Optional[str] = None,
    existing_library: Optional[list[LibraryEntry]] = None,
    seed: int = 0,
    vi_min: float = 0.0,
    vi_max: float = VI_HARD_MAX,
    verbose: bool = True,
) -> list[LibraryEntry]:
    """Build/extend library from an explicit list of (kio, rho, V) triplets.

    Skips any triplet already present in existing_library.

    `seed` is a BUILD-LEVEL constant, deliberately the SAME across every
    (ρ,V) group — geometry/walker RNG seeds are derived from
    (seed, ensemble_index[, kio]) only, never from (ρ,V), so that
    neighbouring (ρ,V) grid points share correlated random-number streams
    (common random numbers). This keeps future Fisher/CRLB finite
    differences w.r.t. ρ,V low-noise without needing a library rebuild.
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

    valid = _filter_valid(triplets, vi_min=vi_min, vi_max=vi_max)
    new_triplets = [t for t in valid if _entry_key(*t) not in done]

    if verbose:
        eff_hi = min(vi_max, VI_HARD_MAX)
        print(f"  Requested: {len(triplets)} triplets")
        print(f"  vi filter: keeping {vi_min:.2f} <= vi <= {eff_hi:.2f}")
        print(f"  Skipped: {len(triplets)-len(valid)} (outside vi range) "
              f"+ {len(valid)-len(new_triplets)} (already exist)")
        print(f"  New to compute: {len(new_triplets)}")

    if not new_triplets:
        if verbose:
            print("  Nothing new to compute!")
        if save_path:
            _save_library(library, save_path, cfg=cfg)
        return library

    columns = sig.build_columns(cfg)
    if verbose:
        print(f"  (δ,Δ,b) grid: {columns.n_pairs} pairs × {columns.n_b} "
              f"b-values = {columns.n_pairs * columns.n_b} columns/entry")

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

        results = sig.compute_signals_multi_kio(
            rho, V, kios_for_group, cfg, columns=columns,
            seed=seed, verbose=False,
        )

        for kio in kios_for_group:
            vec = sig.signals_to_flat(results[kio])
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
            _save_library(library, save_path, cfg=cfg, columns=columns)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nLibrary: {len(library)} entries total "
              f"({len(new_triplets)} new in {elapsed:.0f}s)")

    if save_path:
        _save_library(library, save_path, cfg=cfg, columns=columns)
        if verbose:
            print(f"Saved to {save_path}")

    return library


# ---------------------------------------------------------------------------
# Convenience: build from full cross-product grid
# ---------------------------------------------------------------------------

def build_library(
    kios=None, rhos=None, Vs=None,
    cfg=None, save_path=None, existing_library=None, seed=0,
    vi_min=0.0, vi_max=VI_HARD_MAX, verbose=True,
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
        existing_library=existing_library, seed=seed,
        vi_min=vi_min, vi_max=vi_max, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _save_library(lib: list[LibraryEntry], path: str,
                  cfg: SimConfig | None = None,
                  columns: "sig.ColumnGrid | None" = None):
    """Persist library + (δ,Δ,b) grid metadata to npz.

    Stores:
        kios, rhos, Vs, vectors  — entry parameters and flat S vectors
        pair_deltas, pair_Deltas — (n_pairs,) δ, Δ for each stored pair
        b_values                 — (n_b,) b-values [s/mm²]
        n_b                      — number of b-values (for reshaping)
        h_ms                     — Y(t) storage stride the pairs were built on
    """
    if cfg is None:
        cfg = SimConfig()
    if columns is None:
        columns = sig.build_columns(cfg)

    kios = np.array([e.kio for e in lib])
    rhos = np.array([e.rho for e in lib])
    Vs   = np.array([e.V for e in lib])
    vecs = np.array([e.vector for e in lib])
    pair_deltas = np.array([d for d, D in columns.delta_pairs], dtype=float)
    pair_Deltas = np.array([D for d, D in columns.delta_pairs], dtype=float)
    np.savez(
        path,
        kios=kios, rhos=rhos, Vs=Vs, vectors=vecs,
        pair_deltas=pair_deltas, pair_Deltas=pair_Deltas,
        b_values=np.asarray(columns.b_values, dtype=float),
        n_b=np.array(columns.n_b),
        h_ms=np.array(float(cfg.h_ms)),
    )


def load_library(path: str) -> list[LibraryEntry]:
    # np.load() on a .npz returns a lazy NpzFile: EVERY `data[key]` subscript
    # re-opens the zip member and re-parses the array from scratch (no
    # caching). Pulling each array out ONCE here (rather than indexing
    # `data['vectors'][i]` inside the loop) avoids re-reading the full
    # (n_entries, n_features) vectors array once per entry -- an O(n^2)
    # blowup that made merging many shards balloon in memory/time.
    data = np.load(path)
    kios    = data['kios']
    rhos    = data['rhos']
    Vs      = data['Vs']
    vectors = data['vectors']
    lib = []
    for i in range(len(kios)):
        lib.append(LibraryEntry(
            kio=float(kios[i]),
            rho=float(rhos[i]),
            V=float(Vs[i]),
            vector=vectors[i],
        ))
    return lib


def load_library_meta(path: str) -> dict:
    """Load metadata from a library file — new (δ,Δ,b) format or legacy
    fixed-δ format, normalised to a common shape.

    Returns
    -------
    dict with keys:
        delta_pairs : list of (δ,Δ) [ms] — the pairs each column-group of
            n_b entries in the flat vector corresponds to
        b_values    : list of b-values [s/mm²]
        n_b         : number of b-values per pair
        format      : 'v2' (new, (δ,Δ,b)-universal) or 'legacy'
    """
    data = np.load(path)
    meta = {}

    if 'pair_deltas' in data.files and 'pair_Deltas' in data.files:
        meta['format'] = 'v2'
        pair_deltas = np.asarray(data['pair_deltas'], dtype=float)
        pair_Deltas = np.asarray(data['pair_Deltas'], dtype=float)
        meta['delta_pairs'] = list(zip(pair_deltas.tolist(), pair_Deltas.tolist()))
        meta['n_b'] = int(data['n_b']) if 'n_b' in data.files else None
        meta['b_values'] = (list(np.asarray(data['b_values'], dtype=float))
                             if 'b_values' in data.files else None)
        meta['h_ms'] = float(data['h_ms']) if 'h_ms' in data.files else None
        return meta

    # ---- Legacy fixed-δ format: synthesize an equivalent delta_pairs ----
    meta['format'] = 'legacy'
    legacy_deltas = list(data['deltas']) if 'deltas' in data.files else list(DELTAS_BIG)
    n_b = int(data['n_b']) if 'n_b' in data.files else len(BVALS_UNIQUE)
    small_delta = (float(data['small_delta']) if 'small_delta' in data.files
                   else DELTA_SMALL)
    if small_delta is None:
        small_delta = DELTA_SMALL
    b_values = (list(np.asarray(data['b_values'], dtype=float))
                if 'b_values' in data.files
                else (list(BVALS_UNIQUE.astype(float)) if n_b == len(BVALS_UNIQUE) else None))

    meta['delta_pairs'] = [(small_delta, D) for D in legacy_deltas]
    meta['n_b'] = n_b
    meta['b_values'] = b_values
    meta['small_delta'] = small_delta   # kept for legacy call sites
    meta['h_ms'] = None
    return meta


def library_summary(lib: list[LibraryEntry], meta: dict | None = None):
    """Print a summary of a library."""
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

    vec_len = lib[0].vector.size
    if meta is not None:
        n_b = meta['n_b']
        pairs = meta['delta_pairs']
        bvs = meta.get('b_values')
        print(f"  Format: {meta.get('format', '?')}")
        print(f"  Vector length: {vec_len}  ({len(pairs)} (δ,Δ) pairs × "
              f"{n_b} b-values)")
        d_range = sorted(set(d for d, D in pairs))
        D_range = sorted(set(D for d, D in pairs))
        print(f"  δ range [ms]: [{d_range[0]:g}, {d_range[-1]:g}]  "
              f"({len(d_range)} unique)")
        print(f"  Δ range [ms]: [{D_range[0]:g}, {D_range[-1]:g}]  "
              f"({len(D_range)} unique)")
        if bvs is not None:
            print(f"  b-values [s/mm²]: {[f'{b:g}' for b in bvs]}")
        else:
            print(f"  b-values [s/mm²]: (not stored)")
    else:
        print(f"  Vector length: {vec_len}")


# ---------------------------------------------------------------------------
# Matching helpers — NEAREST column, never interpolated (see module docstring)
# ---------------------------------------------------------------------------

def _nearest_pair_index(delta: float, Delta: float, lib_pairs: np.ndarray) -> int:
    d2 = (lib_pairs[:, 0] - delta) ** 2 + (lib_pairs[:, 1] - Delta) ** 2
    return int(np.argmin(d2))


def _grid_columns(fit_triples, lib_delta_pairs, lib_b_values, n_b,
                   b_tol=50.0, pair_warn_tol=1.5):
    """Column indices into a flat library vector for a list of (δ,Δ,b)
    triples.

    NOT interpolation: each triple is matched to its NEAREST stored (δ,Δ)
    pair and NEAREST b-value (b must be within ``b_tol``, matching the
    existing rounding-tolerance convention; (δ,Δ) has no hard tolerance —
    any mismatch is accepted and simply becomes matching error, per design).
    A warning is printed if the nearest (δ,Δ) pair is more than
    ``pair_warn_tol`` ms away in either coordinate, since that likely means
    the library's grid doesn't actually cover the requested protocol.

    Parameters
    ----------
    fit_triples : list of (δ_ms, Δ_ms, b_s_mm2)
    lib_delta_pairs : list of (δ,Δ) [ms]
    lib_b_values : list of float [s/mm²]
    n_b : int — b-values per pair in the flat vector

    Returns
    -------
    cols : (n_triples,) int array
    """
    if lib_b_values is None:
        raise ValueError(
            "Library has no stored b-values. Rebuild the library with the "
            "current _save_library, or pass lib_b_values explicitly.")

    lib_pairs_arr = np.asarray(lib_delta_pairs, dtype=float)
    lib_b_arr = np.asarray(lib_b_values, dtype=float)

    cols = np.empty(len(fit_triples), dtype=int)
    for k, (delta, Delta, b) in enumerate(fit_triples):
        pi = _nearest_pair_index(delta, Delta, lib_pairs_arr)
        nd, nD = lib_pairs_arr[pi]
        if abs(nd - delta) > pair_warn_tol or abs(nD - Delta) > pair_warn_tol:
            import warnings
            warnings.warn(
                f"(δ={delta:g}, Δ={Delta:g}) ms not on the library grid; "
                f"nearest stored pair is (δ={nd:g}, Δ={nD:g}) ms — "
                f"matching error will be introduced (no interpolation).")

        bi = int(np.argmin(np.abs(lib_b_arr - b)))
        if abs(lib_b_arr[bi] - b) > b_tol:
            raise ValueError(
                f"b = {b} s/mm² not within {b_tol:g} of any library "
                f"b-value {sorted(lib_b_values)}.")
        cols[k] = pi * n_b + bi
    return cols


def _build_candidate_lib_matrix(library, lib_delta_pairs, lib_b_values,
                                 n_b, vi_min, vi_max, rho_max,
                                 fit_triples):
    """Apply candidate filtering and produce the masked, subset library matrix.

    Returns
    -------
    lib_mat : (n_candidates, n_features)
    kios_arr, rhos_arr, Vs_arr : (n_candidates,)
    """
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

    col_idx = _grid_columns(fit_triples, lib_delta_pairs, lib_b_values, n_b)
    lib_mat = full_mat[:, col_idx]

    kios_arr = np.array([e.kio for e in lib_entries])
    rhos_arr = np.array([e.rho for e in lib_entries])
    Vs_arr   = np.array([e.V   for e in lib_entries])

    return lib_mat, kios_arr, rhos_arr, Vs_arr


# ---------------------------------------------------------------------------
# FIXED-S0 matcher (existing behavior, kept)
# ---------------------------------------------------------------------------

def match_voxels_batch(
    measured_batch,
    library,
    lib_delta_pairs, lib_b_values, n_b,
    fit_triples,
    log_space=False, s_floor=1e-3,
    vi_min=0.5, vi_max=0.95, rho_max=None,
    use_gpu=None,
):
    """Log-space nearest-neighbour matching, S0 fixed (data already
    divided by measured b=0).

    Inputs are S/S0 ratios. ``fit_triples`` is a list of (δ,Δ,b) tuples in
    the column order of ``measured_batch`` — see `_grid_columns` for the
    nearest-column (no-interpolation) matching semantics.

    ``use_gpu`` : None (default) = use CUDA if available, else CPU.
    ``True``/``False`` force a path (``True`` raises if CUDA is
    unavailable). The GPU path (``fitters_gpu.map_match_gpu``) is an exact
    reordering of this function's math — same output to float64 precision.
    """

    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, lib_delta_pairs, lib_b_values, n_b, vi_min, vi_max, rho_max,
        fit_triples)

    if log_space:
        measured = np.log(np.clip(measured_batch, s_floor, 1.0))
        lib_m    = np.log(np.clip(lib_mat,         s_floor, 1.0))
    else:
        measured = measured_batch
        lib_m    = lib_mat

    if use_gpu is None:
        use_gpu = fitters_gpu.HAS_CUDA
    if use_gpu:
        if not fitters_gpu.HAS_CUDA:
            raise RuntimeError("use_gpu=True but CUDA is not available.")
        return fitters_gpu.map_match_gpu(measured, lib_m, kios_arr, rhos_arr,
                                          Vs_arr)

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
    lib_delta_pairs, lib_b_values, n_b,
    fit_triples,
    vi_min=0.5, vi_max=0.95, rho_max=None,
    use_gpu=None,
):
    """Match un-normalized signals with S0 as a free per-voxel linear param.

    See ``match_voxels_batch`` for the ``fit_triples`` semantics and the
    ``use_gpu`` convention.

    For each voxel m and each candidate library ratio vector r,

        S0*(m, r) = (m . r) / (r . r)
        residual  = ||m||^2  -  (m . r)^2 / (r . r)

    Returns
    -------
    kio_map, rho_map, V_map, residual_map, s0_fit_map  (each shape (n_voxels,))
    """
    lib_mat, kios_arr, rhos_arr, Vs_arr = _build_candidate_lib_matrix(
        library, lib_delta_pairs, lib_b_values, n_b, vi_min, vi_max, rho_max,
        fit_triples)

    M = raw_signal.astype(np.float64)            # (n_vox, n_feat)
    R = lib_mat.astype(np.float64)               # (n_lib, n_feat)

    if use_gpu is None:
        use_gpu = fitters_gpu.HAS_CUDA
    if use_gpu:
        if not fitters_gpu.HAS_CUDA:
            raise RuntimeError("use_gpu=True but CUDA is not available.")
        return fitters_gpu.map_match_fits0_gpu(M, R, kios_arr, rhos_arr,
                                                Vs_arr)

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

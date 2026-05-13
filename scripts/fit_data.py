#!/usr/bin/env python3
"""
fit_data.py — Build MADI libraries & fit in-vivo DWI data
==========================================================

NEW IN THIS VERSION
-------------------
1. **Bvals/bvecs-driven shell detection**
       The hardcoded SHELLS table is gone.  Each input acquisition
       carries its own bvals (FSL format) and optional bvecs file.
       b=0 indices and shell groupings are detected from bvals.  This
       lets you fit datasets with arbitrary numbers of shells, arbitrary
       direction counts, and interleaved b=0 volumes — as long as the
       (Δ, b) pairs you have are a subset of what the library covers.

2. **(Δ, b) pair-level matching**
       Library entries store an n_deltas × n_b signal vector.  The
       matcher now subsets to whatever (Δ, b) pairs the data provides,
       not just whatever Δ values.  A single-shell dataset against a
       4-shell library will match on the column for that one b-value.

3. **Acquisition consistency checks**
       Pre-fit: small δ must match library, every input Δ must be in
       the library, every measured b-value must be in the library's
       b-values for that Δ.  Underdetermined fits (fewer measurements
       than free parameters) trigger a loud warning.

4. **Multi-b=0 averaging**
       Each input file may have many b=0 volumes scattered throughout.
       They are averaged within each Δ acquisition to produce that Δ's
       S0.  --avg-s0 still cross-averages S0 across Δ.

LEGACY FEATURES (unchanged)
---------------------------
1. **Rician noise-bias correction** (--rician-correct)
2. **S0 averaging across Delta scans** (--avg-s0)
3. **Optional S0 fitting in the matcher** (--fit-s0)

INPUT FORMAT
------------
  New (recommended):
    --input "Δ:dwi.nii.gz:bvals.bval[:bvecs.bvec]"
  Legacy (still works for old protocol):
    --input "Δ:dwi.nii.gz"   (uses LEGACY_SHELLS)

EXAMPLES
--------
  python fit_data.py --fit \\
      --input 25:dwi25.nii.gz:dwi25.bval:dwi25.bvec \\
      --mask mask.nii.gz \\
      --small-delta 6.0 \\
      --rician-correct
"""

import argparse, os, sys, time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from madi.config   import SimConfig, BVALS_S_MM2, BVALS_UNIQUE, DELTAS_BIG
from madi.library  import (build_library, build_library_from_triplets,
                            load_library, load_library_meta,
                            match_voxels_batch, match_voxels_batch_fits0,
                            library_summary)
from madi.signal   import signals_to_flat


# ===================================================================
# Acquisition protocol — LEGACY fallback only
# ===================================================================
# Used only when an --input has no bvals path attached (old workflow).
# New code paths use bvals files directly.

LEGACY_SHELLS = [
    (1000.0, slice(1, 25)),
    (2500.0, slice(25, 49)),
    (4000.0, slice(49, 73)),
    (6000.0, slice(73, 97)),
]
LEGACY_B0_INDEX = 0
LEGACY_N_VOLS   = 97


# Tolerance for clustering bvals into shells (s/mm²).  Anything below
# B0_THRESHOLD is treated as b=0; non-zero bvals within B_TOL of each
# other are grouped into the same shell.
B0_THRESHOLD = 50.0
B_TOL        = 50.0


# ===================================================================
# Presets  (unchanged - omitted for brevity, keep yours)
# ===================================================================

PRESETS = {
    "calibration": {
        "kios": [10, 35],
        "rhos": [200_000, 800_000],
        "Vs":   [1.0, 3.0],
        "cfg":  dict(n_walkers=100_000, n_ensembles=120, n_steps=50_000,
                     L=250.0, buffer=60.0, grid_spacing=1.0),
    },
    "small": {
        "kios": [5, 12, 25, 50],
        "rhos": [100_000, 200_000, 400_000, 800_000, 1_200_000],
        "Vs":   [0.5, 1.0, 2.0, 3.5],
        "cfg":  dict(n_walkers=5_000, n_ensembles=2, n_steps=50_000,
                     L=180.0, buffer=45.0, grid_spacing=1.2),
    },
    "default": {
        "kios": [2, 5, 8, 12, 18, 25, 35, 50, 75],
        "rhos": [100_000, 200_000, 300_000, 400_000, 600_000,
                 800_000, 1_000_000, 1_200_000, 1_500_000],
        "Vs":   [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        "cfg":  dict(n_walkers=100_000, n_ensembles=120, n_steps=50_000,
                     L=250.0, buffer=60.0, grid_spacing=1.0),
    },
    "dense": {
        "kios": [2, 4, 6, 8, 10, 12, 15, 18, 22, 25, 30, 35, 45, 60, 80, 100],
        "rhos": [100_000, 150_000, 200_000, 300_000, 400_000, 500_000,
                600_000, 800_000, 1_000_000, 1_200_000, 1_500_000,
                2_000_000, 3_000_000],
        "Vs":   [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5,
                3.0, 4.0, 5.0, 7.0, 9.0],
        "cfg":  dict(n_walkers=100_000, n_ensembles=40, n_steps=75_000,
                     L=250.0, buffer=60.0, grid_spacing=1.0),
    },
}


# ===================================================================
# Parsers
# ===================================================================

def parse_triplet(s: str):
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Triplet must be 'kio,rho,V', got '{s}'")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def parse_input(s: str):
    """Parse a single --input spec.

    Supported forms:
        Δ:dwi.nii.gz                          — legacy, uses LEGACY_SHELLS
        Δ:dwi.nii.gz:bvals.bval               — bvals-driven, no bvecs
        Δ:dwi.nii.gz:bvals.bval:bvecs.bvec    — bvals + bvecs

    Returns
    -------
    (delta_ms, dwi_path, bvals_path_or_None, bvecs_path_or_None)
    """
    parts = s.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Input must be 'Δ:dwi.nii.gz[:bvals[:bvecs]]', got '{s}'")
    try:
        delta = float(parts[0])
    except ValueError:
        raise ValueError(f"First field of --input must be Δ in ms; got '{parts[0]}'")
    dwi   = parts[1]
    bvals = parts[2] if len(parts) >= 3 and parts[2] else None
    bvecs = parts[3] if len(parts) >= 4 and parts[3] else None
    if len(parts) > 4:
        raise ValueError(f"Too many ':'-separated fields in --input '{s}'")
    return (delta, dwi, bvals, bvecs)


def parse_z_slice(s):
    """Parse a --z-slice spec into a Python slice object (or None).

    Examples
    --------
        '50'    → slice(50, 51)        — single Z-slice
        '40:60' → slice(40, 60)        — Z in [40, 60)
        ':60'   → slice(None, 60)      — Z in [0, 60)
        '40:'   → slice(40, None)      — Z in [40, end)
        None    → None                 — no restriction
    """
    if s is None:
        return None
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"--z-slice range must be 'a:b', got '{s}'")
        a = int(parts[0]) if parts[0] else None
        b = int(parts[1]) if parts[1] else None
        return slice(a, b)
    return slice(int(s), int(s) + 1)


# ===================================================================
# bvals / bvecs parsing
# ===================================================================

def parse_bvals(path: str, b0_thresh: float = B0_THRESHOLD,
                b_tol: float = B_TOL):
    """Read an FSL-format bvals file and group into b=0 + shells.

    Parameters
    ----------
    path : str
        Path to a whitespace-delimited bvals file (typically one row).
    b0_thresh : float
        Anything with b < this is treated as b=0 [s/mm²].
    b_tol : float
        Non-zero bvals within ±b_tol of each other are grouped into the
        same shell [s/mm²].

    Returns
    -------
    bvals : (n_vols,) float ndarray  — raw values from the file
    b0_idx : (n_b0,) int ndarray
    shells : list of (b_value, idx_array) sorted by ascending b
        b_value is the rounded representative for that shell [s/mm²].
    """
    raw = np.loadtxt(path).ravel().astype(float)
    if raw.size == 0:
        raise ValueError(f"bvals file is empty: {path}")

    b0_mask = raw < b0_thresh
    b0_idx  = np.where(b0_mask)[0]
    nz_idx  = np.where(~b0_mask)[0]

    if nz_idx.size == 0:
        raise ValueError(f"No non-zero b-values in {path}")

    nz_vals = raw[nz_idx]

    # Cluster by sorting and walking with b_tol gap.  Robust to scanner
    # rounding (e.g. 998, 1001 → one cluster at 1000).
    order = np.argsort(nz_vals)
    sorted_vals = nz_vals[order]
    sorted_idx  = nz_idx[order]

    shells = []
    cur_lo = sorted_vals[0]
    cur_idx = [sorted_idx[0]]
    cur_vals = [sorted_vals[0]]
    for v, i in zip(sorted_vals[1:], sorted_idx[1:]):
        if v - cur_lo <= b_tol:
            cur_idx.append(i)
            cur_vals.append(v)
        else:
            b_rep = round(float(np.mean(cur_vals)) / 50.0) * 50.0
            shells.append((b_rep, np.array(sorted(cur_idx), dtype=int)))
            cur_lo = v
            cur_idx = [i]
            cur_vals = [v]
    b_rep = round(float(np.mean(cur_vals)) / 50.0) * 50.0
    shells.append((b_rep, np.array(sorted(cur_idx), dtype=int)))

    return raw, b0_idx, shells


def parse_bvecs(path: str, n_vols_expected: int):
    """Read an FSL-format bvecs file (3 rows × N columns).

    Returns
    -------
    bvecs : (3, n_vols) ndarray, or None if path is None
    """
    if path is None:
        return None
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != 3:
        # Some pipelines store bvecs as N×3; transpose if needed
        if arr.shape[1] == 3:
            arr = arr.T
        else:
            raise ValueError(
                f"bvecs file {path} has shape {arr.shape}; expected (3, N) "
                f"or (N, 3).")
    if arr.shape[1] != n_vols_expected:
        raise ValueError(
            f"bvecs file {path}: {arr.shape[1]} columns, "
            f"DWI has {n_vols_expected} volumes.")
    return arr


def check_direction_uniformity(bvecs, idx, label=""):
    """Loose check that a set of directions is reasonably uniformly
    distributed on the sphere.  Prints a warning if highly biased.

    Heuristic: the mean of the unit vectors (after antipodal symmetry
    folding) should have small magnitude for an isotropic set.  We use
    the resultant length of v_k ⊗ v_k (rank-2 tensor) eigenvalue spread:
    isotropic → all eigenvalues ≈ 1/3.
    """
    if bvecs is None or len(idx) < 3:
        return
    v = bvecs[:, idx]                   # (3, n)
    norms = np.linalg.norm(v, axis=0)
    keep = norms > 1e-6
    if keep.sum() < 3:
        return
    v = v[:, keep] / norms[keep]
    T = (v @ v.T) / v.shape[1]          # (3,3) rank-2 tensor
    eigvals = np.sort(np.linalg.eigvalsh(T))[::-1]
    spread = eigvals[0] - eigvals[2]
    # spread = 0 → perfect isotropy; spread ≥ ~0.4 → strongly biased
    if spread > 0.4:
        print(f"    ⚠ {label}: directions look biased "
              f"(eigenvalue spread = {spread:.2f}, isotropic = 0). "
              f"Powder average may be skewed.")


# ===================================================================
# Noise estimation & Rician correction
# ===================================================================

def estimate_noise_sigma(data_4d, mask_brain, method="background"):
    """Estimate Rician noise sigma from a magnitude DWI volume.

    Parameters
    ----------
    data_4d : ndarray (X,Y,Z,N_vols)  full DWI volume series
    mask_brain : bool ndarray (X,Y,Z) brain mask (used to find air region)
    method : 'background' uses voxels outside the brain mask in the b=0
             image.  In a true-zero-signal region, magnitude follows a
             Rayleigh distribution with sigma = sqrt(mean(M^2) / 2).

    Returns
    -------
    sigma : float (single noise std; assumes spatially uniform noise)
    """
    # Use b=0 volume (volume 0) for noise estimation
    b0 = data_4d[..., 0]

    # Background = outside brain, with a small erosion margin to avoid
    # partial-volume edge effects.  Simple version: just outside mask.
    bg = ~mask_brain

    # Drop zero voxels (often the FOV padding) - they're not noise samples
    bg_vals = b0[bg]
    bg_vals = bg_vals[bg_vals > 0]

    if len(bg_vals) < 100:
        print("    WARNING: very few background voxels; sigma estimate "
              "may be unreliable")
        return None

    # Rayleigh: sigma = sqrt(<M^2>/2)
    sigma = float(np.sqrt(np.mean(bg_vals.astype(np.float64)**2) / 2.0))
    return sigma


def rician_correct_secondmoment(M, sigma):
    """Recover unbiased A from magnitude M using E[M^2] = A^2 + 2 sigma^2.

    Vectorized.  Clips negative values to 0 (occurs when M < sqrt(2)*sigma,
    i.e. essentially pure noise).

    Parameters
    ----------
    M : ndarray  magnitude signal (any shape)
    sigma : float  Rician noise std

    Returns
    -------
    A : ndarray  bias-corrected signal, same shape as M
    """
    A2 = M.astype(np.float64)**2 - 2.0 * sigma**2
    A2 = np.clip(A2, 0.0, None)
    return np.sqrt(A2)


# ===================================================================
# Data loading
# ===================================================================

def load_dwi_and_average(input_specs, mask_path,
                         lib_b_values,
                         rician_correct=False,
                         noise_sigma=None,
                         avg_s0=False,
                         return_raw=False,
                         z_slice=None,
                         b_tol=B_TOL):
    """Load DWI NIfTIs, derive (Δ, b) layout from each input's bvals,
    and assemble the measured matrix in a column order matching
    ``fit_pairs``.

    Parameters
    ----------
    input_specs : list of (delta_ms, dwi_path, bvals_path_or_None,
                          bvecs_path_or_None)
        For each input either bvals_path is given (preferred) or it is
        None (legacy path: assumes LEGACY_SHELLS protocol).
    mask_path : str or None
        Path to a brain mask NIfTI.  If None, fit every voxel of the
        full DWI volume — affine and shape are taken from the first
        input DWI.  Required when ``rician_correct`` is True and
        ``noise_sigma`` is None (auto-σ estimation needs background
        voxels, which need a mask to find).
    z_slice : slice or None
        Optional Python slice along the third spatial axis to restrict
        fitting to a Z-slice or range.  Combined (intersected) with
        ``mask_path`` if both are given.  Noise σ estimation continues
        to use the un-sliced mask so air voxels from outside the slice
        range still contribute.
    lib_b_values : list of float
        b-values present in the library [s/mm²].  Used to filter shells
        in the data: only shells whose b matches a library b-value
        (within b_tol) are retained.
    rician_correct : bool
    noise_sigma : float or None
    avg_s0 : bool
        If True, replace each Δ's S0 with the grand mean across Δs.
    return_raw : bool
    b_tol : float
        Tolerance for matching data b-values to library b-values.

    Returns
    -------
    measured : (n_voxels, n_features) ndarray  — S/S0 ratios
    fit_pairs : list of (Δ, b) tuples in the column order of ``measured``
    affine, mask_indices, shape
    extras (if return_raw) : dict with 'raw', 's0', 'sigma'
    """
    import nibabel as nib

    input_specs = sorted(input_specs, key=lambda x: x[0])
    n_deltas = len(input_specs)

    # ----------------------------------------------------------------
    # Mask (optional).  If absent, fit every voxel of the volume —
    # spatial reference is taken from the first DWI input.
    # ----------------------------------------------------------------
    if mask_path is not None:
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(bool)
        affine = mask_img.affine
        shape = mask.shape
        print(f"  Mask: {mask.sum()} voxels (from --mask)")
    else:
        first_img = nib.load(input_specs[0][1])
        affine = first_img.affine
        shape  = tuple(first_img.shape[:3])
        mask   = np.ones(shape, dtype=bool)
        print(f"  No --mask: starting from full volume "
              f"(shape {shape}, {int(np.prod(shape))} voxels). "
              f"Output maps will contain garbage in air regions; "
              f"mask the maps post-hoc if desired.")

    # Keep the un-sliced mask around for noise estimation (background
    # voxels should be drawn from the whole volume regardless of
    # whether we're only fitting one slice).
    mask_for_noise = mask.copy()

    # Optional z-slice restriction.
    if z_slice is not None:
        if len(shape) < 3:
            raise ValueError(f"z_slice given but volume has shape {shape} "
                             f"(need ≥3 spatial dims).")
        z_keep = np.zeros(shape, dtype=bool)
        z_keep[:, :, z_slice] = True
        before = int(mask.sum())
        mask = mask & z_keep
        after = int(mask.sum())
        # Pretty-print the slice
        a = z_slice.start if z_slice.start is not None else 0
        b = z_slice.stop  if z_slice.stop  is not None else shape[2]
        print(f"  --z-slice: restricting to Z in [{a}, {b}) "
              f"→ {after} voxels (was {before})")

    mask_idx = np.where(mask)
    n_vox = len(mask_idx[0])
    if n_vox == 0:
        raise ValueError("Mask (after --z-slice intersection) contains "
                         "zero voxels.")

    # ----------------------------------------------------------------
    # Pass 1: load DWI volumes and parse each input's shell layout
    # ----------------------------------------------------------------
    all_data         = []         # list of (X, Y, Z, N_vols) arrays
    all_b0_idx       = []         # list of int arrays
    all_shells_kept  = []         # list of [(b, idx)] retained against library

    for di, (delta_ms, dwi_path, bvals_path, bvecs_path) in enumerate(input_specs):
        print(f"  Δ={delta_ms:.1f}ms  ({os.path.basename(dwi_path)})", flush=True)
        img = nib.load(dwi_path)
        data = img.get_fdata().astype(np.float64)
        n_vols = data.shape[-1]
        print(f"    DWI shape: {data.shape}")

        if bvals_path is None:
            # Legacy path: rebuild shell layout from LEGACY_SHELLS
            if n_vols != LEGACY_N_VOLS:
                print(f"    ⚠ legacy mode expected {LEGACY_N_VOLS} volumes, "
                      f"got {n_vols}.  Pass a bvals file for non-default "
                      f"protocols.")
            b0_idx = np.array([LEGACY_B0_INDEX], dtype=int)
            shells = [(float(b), np.arange(sl.start, sl.stop, dtype=int))
                      for b, sl in LEGACY_SHELLS]
            print(f"    LEGACY mode: 1 b=0 vol, {len(shells)} shells")
        else:
            bvals, b0_idx, shells = parse_bvals(bvals_path)
            if bvals.size != n_vols:
                raise ValueError(
                    f"bvals length ({bvals.size}) ≠ DWI volumes ({n_vols}) "
                    f"for {dwi_path}")
            print(f"    bvals: {len(b0_idx)} b=0 vols, "
                  f"{len(shells)} non-zero shells "
                  f"({[f'b={int(b)}×{len(idx)}' for b, idx in shells]})")

            # Optional bvecs uniformity check
            bvecs = parse_bvecs(bvecs_path, n_vols_expected=n_vols)
            if bvecs is not None:
                for b, idx in shells:
                    check_direction_uniformity(bvecs, idx,
                                               label=f"Δ={delta_ms:g}, b={int(b)}")

        # Filter shells against library
        kept = []
        dropped = []
        for b, idx in shells:
            match = next((lb for lb in lib_b_values if abs(b - lb) <= b_tol),
                         None)
            if match is not None:
                kept.append((float(match), idx))   # use library's canonical b
            else:
                dropped.append(b)
        if dropped:
            print(f"    ⚠ dropping shells not in library: "
                  f"{[f'{b:g}' for b in dropped]}  "
                  f"(library has {[f'{b:g}' for b in lib_b_values]})")
        if not kept:
            raise ValueError(
                f"No shells in {dwi_path} match any library b-value "
                f"(library: {lib_b_values}).")
        kept.sort(key=lambda x: x[0])

        all_data.append(data)
        all_b0_idx.append(b0_idx)
        all_shells_kept.append(kept)

    # ----------------------------------------------------------------
    # Build the column ordering: (Δ, b) pairs sorted by Δ then by b
    # ----------------------------------------------------------------
    fit_pairs = []
    for (delta_ms, *_), kept in zip(input_specs, all_shells_kept):
        for b, _ in kept:
            fit_pairs.append((float(delta_ms), float(b)))
    n_features = len(fit_pairs)
    print(f"  → {n_features} (Δ,b) features per voxel")

    # ----------------------------------------------------------------
    # Noise sigma estimation (optional) — uses first scan's first b=0
    # ----------------------------------------------------------------
    sigma_used = None
    if rician_correct:
        if noise_sigma is not None:
            sigma_used = float(noise_sigma)
            print(f"  Rician correction ENABLED, user-provided sigma={sigma_used:.2f}")
        else:
            b0_for_sigma = all_data[0][..., all_b0_idx[0][0]]
            bg = ~mask_for_noise
            bg_vals = b0_for_sigma[bg]
            bg_vals = bg_vals[bg_vals > 0]
            if len(bg_vals) < 100:
                print("    ⚠ very few background voxels — disabling Rician correction.")
                rician_correct = False
            else:
                sigma_used = float(np.sqrt(np.mean(bg_vals.astype(np.float64) ** 2) / 2.0))
                b0_brain_med = float(np.median(b0_for_sigma[mask_for_noise]))
                print(f"  Rician correction ENABLED")
                print(f"    sigma (auto, background)  = {sigma_used:.2f}")
                print(f"    median brain b=0 signal   = {b0_brain_med:.1f}")
                print(f"    median brain b=0 SNR      = {b0_brain_med/sigma_used:.1f}")

    if rician_correct and sigma_used is not None:
        for di in range(n_deltas):
            all_data[di] = rician_correct_secondmoment(all_data[di], sigma_used)

    # ----------------------------------------------------------------
    # Per-Δ S0 = mean of all b=0 volumes within that input
    # ----------------------------------------------------------------
    s0_per_delta = []
    for di in range(n_deltas):
        b0_idx = all_b0_idx[di]
        # (X, Y, Z, n_b0)  →  per-voxel mean → flatten with mask
        b0_block = all_data[di][..., b0_idx]
        b0_mean_vol = np.mean(b0_block, axis=-1)
        s0_vox = b0_mean_vol[mask]
        s0_per_delta.append(s0_vox)
        print(f"  Δ={input_specs[di][0]:g}: S0 from {len(b0_idx)} b=0 vols, "
              f"median brain S0 = {np.median(s0_vox):.1f}")

    if avg_s0 and n_deltas > 1:
        s0_stack = np.stack(s0_per_delta, axis=0)             # (n_d, n_vox)
        s0_common = np.mean(s0_stack, axis=0)
        s0_cv = np.std(s0_stack, axis=0) / (s0_common + 1e-10)
        print(f"  --avg-s0: averaging S0 across {n_deltas} Δ scans")
        print(f"    median across-Δ S0 CV = {np.median(s0_cv)*100:.2f}%   "
              f"(low = scans well registered)")
        print(f"    95th-pct CV           = {np.percentile(s0_cv, 95)*100:.2f}%")
        if np.median(s0_cv) > 0.10:
            print(f"    ⚠ High S0 variability across Δ — check motion/drift "
                  f"before trusting the averaged S0.")
        s0_used = [s0_common.copy() for _ in range(n_deltas)]
    else:
        if avg_s0 and n_deltas == 1:
            print("  --avg-s0 is a no-op for single-Δ input "
                  "(b=0 vols within the input are already averaged).")
        s0_used = s0_per_delta

    # ----------------------------------------------------------------
    # Build measured matrix in fit_pairs order
    # ----------------------------------------------------------------
    measured   = np.zeros((n_vox, n_features), dtype=np.float64)
    raw_signal = np.zeros((n_vox, n_features), dtype=np.float64)

    col = 0
    for di, kept in enumerate(all_shells_kept):
        vox_data = all_data[di][mask, :]            # (n_vox, n_vols)
        S0 = s0_used[di].copy()
        S0[S0 < 1e-10] = 1e-10
        for b, idx in kept:
            shell_mean = np.mean(vox_data[:, idx], axis=1)
            measured[:, col]   = shell_mean / S0
            raw_signal[:, col] = shell_mean
            col += 1
    assert col == n_features

    if return_raw:
        s0_ref = np.mean(np.stack(s0_used, axis=0), axis=0)
        extras = dict(raw=raw_signal, s0=s0_ref, sigma=sigma_used)
        return measured, fit_pairs, affine, mask_idx, shape, extras

    return measured, fit_pairs, affine, mask_idx, shape


# ===================================================================
# Saving maps
# ===================================================================

def save_map(data_1d, mask_idx, shape, affine, path, dtype=np.float32):
    import nibabel as nib
    vol = np.zeros(shape, dtype=dtype)
    vol[mask_idx] = data_1d.astype(dtype)
    nib.save(nib.Nifti1Image(vol, affine), path)
    print(f"  Saved {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="MADI library builder & fitter (with Rician correction "
                    "and flexible S0 handling)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # -- Actions --
    ap.add_argument("--build-library", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--info", action="store_true")

    # -- Library file --
    ap.add_argument("--library", default="madi_library.npz")
    ap.add_argument("--append", action="store_true")

    # -- Preset grid --
    ap.add_argument("--lib-preset", default="default", choices=list(PRESETS.keys()))

    # -- Custom additions --
    ap.add_argument("--custom-kios", type=float, nargs="+")
    ap.add_argument("--custom-rhos", type=float, nargs="+")
    ap.add_argument("--custom-Vs",   type=float, nargs="+")

    # -- Explicit sub-grid --
    ap.add_argument("--explicit", action="store_true")
    ap.add_argument("--grid-kios", type=float, nargs="+")
    ap.add_argument("--grid-rhos", type=float, nargs="+")
    ap.add_argument("--grid-Vs",   type=float, nargs="+")

    # -- Exact triplets --
    ap.add_argument("--triplets", type=str, nargs="+")

    # -- Sharding --
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)

    # -- Fitting inputs --
    ap.add_argument("--input", type=str, nargs="+",
                    help="'delta:path' pairs (e.g. 15:dwi15.nii.gz)")
    ap.add_argument("--dwi15"); ap.add_argument("--dwi25")
    ap.add_argument("--dwi30"); ap.add_argument("--dwi40")
    ap.add_argument("--mask", default=None,
                    help="Optional brain mask NIfTI.  If omitted, the "
                         "fit is run over every voxel of the volume.  "
                         "Required when --rician-correct is used without "
                         "an explicit --noise-sigma.")
    ap.add_argument("--z-slice", default=None,
                    help="Restrict fitting to a single Z-slice or range. "
                         "Examples: '50' (just slice 50), '40:60' "
                         "(slices 40-59), ':60' (slices 0-59), '40:' "
                         "(slices 40 to end).  Intersects with --mask "
                         "if both are given.  Noise σ estimation still "
                         "uses the un-sliced volume.")
    ap.add_argument("--out", default="madi_output")

    # -- NEW: Rician + S0 options --
    ap.add_argument("--rician-correct", action="store_true",
                    help="Apply Rician noise-bias correction "
                         "(E[M^2] = A^2 + 2 sigma^2) to each volume "
                         "before normalization.")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="Rician noise std.  If omitted and --rician-correct "
                         "is set, sigma is estimated from background voxels "
                         "of the first scan's b=0 image.")
    ap.add_argument("--avg-s0", action="store_true",
                    help="Average the b=0 volumes across all Δ scans into a "
                         "single S0 image for normalization.  TE is constant "
                         "across Δ so this only loses information if the "
                         "scans are misregistered.")
    ap.add_argument("--fit-s0", action="store_true",
                    help="Treat S0 as a free per-voxel parameter in the "
                         "matcher (analytic L2-optimal projection per "
                         "library entry).  Diagnostic for S0 reliability.")
    ap.add_argument("--log_space", action="store_true",
                    help="Whether to preform fitting within log space or to use no transformations.")
    # -- NEW: acquisition metadata (must match library) --
    ap.add_argument("--small-delta", type=float, default=None,
                    help="δ (PFG duration) [ms] of the data being fit.  "
                         "Must match the library's small δ (within 0.05 ms).  "
                         "If omitted, the library's stored small δ is used "
                         "and a warning is printed.")
    # -- Matcher tuning (already exposed in library.py but worth making CLI) --
    ap.add_argument("--vi-min", type=float, default=0.5,
                    help="Lower bound on intracellular volume fraction "
                         "for library candidates (paper uses 0.5).")
    ap.add_argument("--vi-max", type=float, default=0.95)
    ap.add_argument("--rho-max", type=float, default=None,
                    help="Optional upper bound on library rho [cells/uL] "
                         "for matching (e.g. 1500000 for brain).")

    args = ap.parse_args()
    
    # # Manually overridden arguments matching the terminal command
    # args.fit = True
    # args.input = [
    #     "15:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_15ms/eddy_corrected.nii.gz",
    #     "25:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_25ms/eddy_corrected.nii.gz",
    #     "30:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_30ms/eddy_corrected.nii.gz",
    #     "40:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_40ms/eddy_corrected.nii.gz"
    # ]
    # args.mask = "/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_15ms/mask_cropped.nii.gz"
    # args.out = "out_baseline"
    # args.library = "data/libraries/madi_dense.npz"

    # args.fit_s0 = False
    # args.rician_correct = False
    # args.avg_s0 = False

    if not any([args.build_library, args.fit, args.info]):
        ap.print_help(); return

    # ================================================================
    #  INFO / BUILD branches unchanged - omitted for brevity
    # ================================================================
    if args.info:
        if not os.path.exists(args.library):
            print(f"Library not found: {args.library}"); return
        lib = load_library(args.library)
        meta = load_library_meta(args.library)
        print(f"\nLibrary: {args.library}")
        library_summary(lib, meta=meta)
        return

# ================================================================
    #  BUILD LIBRARY
    # ================================================================
    if args.build_library:
        print("=" * 60)
        print("Building MADI library")
        print("=" * 60)

        # Get simulation config from preset
        preset = PRESETS[args.lib_preset]
        cfg = SimConfig(**preset["cfg"])

        # Load existing if appending
        existing = None
        if args.append and os.path.exists(args.library):
            print(f"\n  Loading existing library: {args.library}")
            existing = load_library(args.library)
        elif args.append:
            print(f"\n  --append: {args.library} not found, starting fresh.")

        # ---- Mode: exact triplets ----
        if args.triplets:
            triplets = [parse_triplet(s) for s in args.triplets]
            print(f"\n  Mode: exact triplets ({len(triplets)})")
            for k, r, v in triplets:
                print(f"    kio={k}, rho={r/1e3:.0f}k, V={v:.2f}")

            build_library_from_triplets(
                triplets, cfg=cfg, save_path=args.library,
                existing_library=existing)
            return

        # ---- Mode: explicit sub-grid ----
        if args.explicit:
            gk = args.grid_kios or preset["kios"]
            gr = [int(r) for r in (args.grid_rhos or preset["rhos"])]
            gv = args.grid_Vs or preset["Vs"]
            print(f"\n  Mode: explicit sub-grid")
            print(f"  kio ({len(gk)}): {gk}")
            print(f"  rho ({len(gr)}): {[f'{r/1e3:.0f}k' for r in gr]}")
            print(f"  V   ({len(gv)}): {gv}")

            build_library(
                kios=gk, rhos=gr, Vs=gv, cfg=cfg,
                save_path=args.library, existing_library=existing)
            return

        # ---- Mode: preset grid + optional custom additions ----
        kios = list(preset["kios"])
        rhos = list(preset["rhos"])
        Vs   = list(preset["Vs"])

        if args.custom_kios:
            kios = sorted(set(kios + args.custom_kios))
        if args.custom_rhos:
            rhos = sorted(set(rhos + [int(r) for r in args.custom_rhos]))
        if args.custom_Vs:
            Vs = sorted(set(Vs + args.custom_Vs))

        print(f"\n  Mode: preset '{args.lib_preset}' + custom additions")
        print(f"  kio ({len(kios)}): {kios}")
        print(f"  rho ({len(rhos)}): {[f'{r/1e3:.0f}k' for r in rhos]}")
        print(f"  V   ({len(Vs)}):   {Vs}")

        # ---- Optional sharding for SLURM job arrays ----
        if args.shard_id is not None:
            if args.n_shards is None or args.n_shards < 1:
                print("ERROR: --shard-id requires --n-shards >= 1")
                return
            if not (0 <= args.shard_id < args.n_shards):
                print(f"ERROR: --shard-id must be in [0, {args.n_shards})")
                return

            # Build full triplet list, filter by vi, slice by (ρ,V) pair.
            all_triplets = [(k, r, v) for k in kios for r in rhos for v in Vs]
            valid = [(k, r, v) for k, r, v in all_triplets
                     if (r / 1e9) * (v * 1e3) <= 0.95]

            # Unique (ρ,V) pairs, sorted by cost proxy (ρ·V), round-robin
            # across shards so each shard gets a mix of cheap + expensive.
            pairs = sorted(set((r, v) for _, r, v in valid),
                           key=lambda p: p[0] * p[1])
            my_pairs = set(pairs[i] for i in range(len(pairs))
                           if i % args.n_shards == args.shard_id)
            shard_triplets = [(k, r, v) for (k, r, v) in valid
                              if (r, v) in my_pairs]

            print(f"\n  Sharding: shard {args.shard_id}/{args.n_shards}")
            print(f"    (ρ,V) pairs in shard : {len(my_pairs)}/{len(pairs)}")
            print(f"    triplets in shard    : {len(shard_triplets)}/{len(valid)}")

            # Tag the output file with shard id unless user explicitly
            # supplied a per-shard path already.
            save_path = args.library
            if "{shard" not in save_path and "shard" not in os.path.basename(save_path):
                root, ext = os.path.splitext(save_path)
                save_path = f"{root}.shard{args.shard_id:03d}{ext}"
            else:
                save_path = save_path.format(shard=args.shard_id,
                                             n_shards=args.n_shards)
            print(f"    output               : {save_path}")

            build_library_from_triplets(
                shard_triplets, cfg=cfg, save_path=save_path,
                existing_library=existing)
            return

        build_library(
            kios=kios, rhos=rhos, Vs=Vs, cfg=cfg,
            save_path=args.library, existing_library=existing)
        return

    # ================================================================
    #  FIT DATA
    # ================================================================
    if args.fit:
        # Parse inputs
        input_specs = []
        if args.input:
            for s in args.input:
                input_specs.append(parse_input(s))
        else:
            legacy_map = [(15.0, args.dwi15), (25.0, args.dwi25),
                          (30.0, args.dwi30), (40.0, args.dwi40)]
            for delta, path in legacy_map:
                if path is not None:
                    # Legacy CLI args have no bvals/bvecs — fall back to
                    # LEGACY_SHELLS by passing None.
                    input_specs.append((delta, path, None, None))

        if not input_specs:
            print("ERROR: No DWI inputs specified."); return
        if args.mask is None and args.rician_correct and args.noise_sigma is None:
            print("ERROR: --rician-correct without --mask requires "
                  "--noise-sigma <value>.  Auto-estimation of σ needs "
                  "background (air) voxels, which requires a brain mask "
                  "to identify."); return

        input_specs.sort(key=lambda x: x[0])
        fit_deltas = [d for d, _, _, _ in input_specs]

        print("=" * 60)
        print("MADI Fitting")
        print("=" * 60)
        print(f"  Δ values to fit:        {fit_deltas} ms")
        print(f"  Rician correction:      {args.rician_correct}")
        print(f"  S0 averaging across Δ:  {args.avg_s0}")
        print(f"  S0 fitted per voxel:    {args.fit_s0}")
        print(f"  vi range:               [{args.vi_min}, {args.vi_max}]")
        print(f"  rho_max:                {args.rho_max}")

        os.makedirs(args.out, exist_ok=True)

        print(f"\nLoading library: {args.library}")
        if not os.path.exists(args.library):
            print(f"ERROR: Library not found: {args.library}"); return
        lib = load_library(args.library)
        meta = load_library_meta(args.library)
        lib_deltas    = meta['deltas']
        lib_n_b       = meta['n_b']
        lib_small_d   = meta['small_delta']
        lib_b_values  = meta['b_values']

        if lib_b_values is None:
            print(f"ERROR: library has no stored b-values metadata and no "
                  f"safe default could be inferred.  Rebuild the library "
                  f"with the updated _save_library, or patch lib_b_values "
                  f"manually."); return

        print(f"  {len(lib)} entries")
        library_summary(lib, meta=meta)

        # ---- Acquisition consistency checks ----
        # 1. small δ
        if args.small_delta is not None:
            if lib_small_d is None:
                print(f"  ⚠ library was built before small δ was saved; "
                      f"trusting --small-delta={args.small_delta} ms.")
            elif abs(args.small_delta - lib_small_d) > 0.05:
                print(f"ERROR: --small-delta ({args.small_delta} ms) does not "
                      f"match library small δ ({lib_small_d} ms)."); return
            else:
                print(f"  ✓ small δ matches library: {args.small_delta} ms")
        else:
            if lib_small_d is None:
                print(f"  ⚠ no --small-delta supplied and library has none "
                      f"stored.  Assuming default {SimConfig().delta} ms.")
            else:
                print(f"  ⓘ no --small-delta supplied; assuming library value "
                      f"{lib_small_d} ms.")

        # 2. Δ values
        for d in fit_deltas:
            if not any(abs(d - ld) < 0.01 for ld in lib_deltas):
                print(f"ERROR: Δ = {d} ms not in library {list(lib_deltas)}"); return

        # Load data; produces fit_pairs ----
        print("\nLoading DWI data ...")
        z_slice_obj = parse_z_slice(args.z_slice)
        load_out = load_dwi_and_average(
            input_specs, args.mask,
            lib_b_values=lib_b_values,
            rician_correct=args.rician_correct,
            noise_sigma=args.noise_sigma,
            avg_s0=args.avg_s0,
            return_raw=args.fit_s0,
            z_slice=z_slice_obj,
        )

        if args.fit_s0:
            measured, fit_pairs, affine, mask_idx, shape, extras = load_out
        else:
            measured, fit_pairs, affine, mask_idx, shape = load_out

        n_features = len(fit_pairs)
        print(f"\n  Feature vector ({n_features} cols):")
        for d, b in fit_pairs:
            print(f"    Δ={d:g} ms,  b={b:g} s/mm²")

        # Underdetermination warning (3 free params: kio, ρ, V)
        if n_features < 3:
            print(f"\n  ⚠ Only {n_features} measurement(s) per voxel for "
                  f"3 free parameters (kio, ρ, V).  The fit is severely "
                  f"underdetermined; many library entries will produce "
                  f"essentially identical residuals.  Treat the maps as "
                  f"diagnostic only.")
        elif n_features < 6:
            print(f"\n  ⓘ {n_features} measurements per voxel for 3 free "
                  f"parameters — a workable but tight fit.")

        # ---- Match ----
        if args.fit_s0:
            raw_signal = extras['raw']
            print(f"\nMatching {raw_signal.shape[0]} voxels with S0 FITTED ...")
            t0 = time.time()
            kio_map, rho_map, V_map, res_map, s0_fit_map = match_voxels_batch_fits0(
                raw_signal, lib,
                fit_pairs=fit_pairs,
                lib_deltas=lib_deltas,
                lib_b_values=lib_b_values,
                n_b=lib_n_b,
                vi_min=args.vi_min,
                vi_max=args.vi_max,
                rho_max=args.rho_max,
            )
            print(f"  Done in {time.time()-t0:.1f}s")
            save_map(s0_fit_map, mask_idx, shape, affine,
                     os.path.join(args.out, "s0_fit_map.nii.gz"))
            s0_ratio = s0_fit_map / (extras['s0'] + 1e-10)
            save_map(s0_ratio, mask_idx, shape, affine,
                     os.path.join(args.out, "s0_fit_over_measured.nii.gz"))
            print(f"\n  Fitted-S0 / Measured-S0 ratio:")
            print(f"    median = {np.median(s0_ratio):.3f}")
            print(f"    5-95%  = [{np.percentile(s0_ratio, 5):.3f}, "
                  f"{np.percentile(s0_ratio, 95):.3f}]")
        else:
            print(f"\nMatching {measured.shape[0]} voxels ...")
            t0 = time.time()
            kio_map, rho_map, V_map, res_map = match_voxels_batch(
                measured, lib,
                fit_pairs=fit_pairs,
                lib_deltas=lib_deltas,
                lib_b_values=lib_b_values,
                n_b=lib_n_b,
                vi_min=args.vi_min,
                vi_max=args.vi_max,
                rho_max=args.rho_max,
                log_space=args.log_space,
            )
            print(f"  Done in {time.time()-t0:.1f}s")

        # Stats
        print(f"\n  kio:  median={np.median(kio_map):.1f}, "
              f"range=[{kio_map.min():.1f}, {kio_map.max():.1f}] s-1")
        print(f"  rho:  median={np.median(rho_map)/1e3:.0f}k, "
              f"range=[{rho_map.min()/1e3:.0f}k, "
              f"{rho_map.max()/1e3:.0f}k] cells/uL")
        print(f"  V:    median={np.median(V_map):.2f}, "
              f"range=[{V_map.min():.2f}, {V_map.max():.2f}] pL")

        # Boundary warnings
        for name, vals, grid in [
            ("rho", rho_map, sorted(set(e.rho for e in lib))),
            ("V",   V_map,   sorted(set(e.V for e in lib))),
            ("kio", kio_map, sorted(set(e.kio for e in lib))),
        ]:
            n = len(vals)
            at_max = np.sum(vals >= grid[-1] - 1e-10)
            at_min = np.sum(vals <= grid[0] + 1e-10)
            if at_max / n > 0.10:
                print(f"  ⚠ {at_max/n*100:.0f}% of voxels hit {name} "
                      f"UPPER bound ({grid[-1]}). Extend the grid.")
            if at_min / n > 0.10:
                print(f"  ⚠ {at_min/n*100:.0f}% of voxels hit {name} "
                      f"LOWER bound ({grid[0]}). Extend the grid.")

        print("\nSaving maps ...")
        save_map(kio_map, mask_idx, shape, affine,
                 os.path.join(args.out, "kio_map.nii.gz"))
        save_map(rho_map, mask_idx, shape, affine,
                 os.path.join(args.out, "rho_map.nii.gz"))
        save_map(V_map, mask_idx, shape, affine,
                 os.path.join(args.out, "V_map.nii.gz"))
        save_map(res_map, mask_idx, shape, affine,
                 os.path.join(args.out, "residual_map.nii.gz"))

        print("\nDone!")


if __name__ == "__main__":
    main()
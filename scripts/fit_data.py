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
from madi.fitters  import (bayes_fit, amico_fit, estimate_sigma_m,
                           DEFAULT_SIGMA_M, DEFAULT_LAMBDA1, DEFAULT_LAMBDA2)
from madi.signal   import signals_to_flat
from madi import fitters_gpu


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


# Anything below B0_THRESHOLD is treated as b=0 [s/mm²].
B0_THRESHOLD = 50.0

# Each non-b0 DWI volume's raw b-value is matched individually (not
# clustered with its neighbors first) against the library's b-value grid:
# if the nearest library b-value is within B_LIB_MATCH_TOL, that volume is
# snapped to it; otherwise the volume is discarded from the fit entirely.
B_LIB_MATCH_TOL = 30.0


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

def parse_bvals(path: str, lib_b_values, b0_thresh: float = B0_THRESHOLD,
                tol: float = B_LIB_MATCH_TOL):
    """Read an FSL-format bvals file and match each volume directly against
    the library's b-value grid.

    Each non-b0 volume's raw b-value is checked individually (not
    clustered with its neighbors first): if the nearest library b-value is
    within ``tol``, the volume is snapped to that library b-value;
    otherwise it is dropped from the fit entirely. This is deliberately
    library-driven rather than scanner-driven -- a chain-clustering
    approach that groups nearby raw b-values together first and then
    checks whether the cluster's mean happens to land near a library value
    can silently misclassify volumes near the boundary between shells, and
    can't express "this whole run of intermediate b-values (e.g. an
    IVIM/perfusion-range shell) isn't in the library at all, throw it
    away" as precisely as a per-volume check does.

    Parameters
    ----------
    path : str
        Path to a whitespace-delimited bvals file (typically one row).
    lib_b_values : list of float
        b-values present in the library [s/mm²].
    b0_thresh : float
        Anything with b < this is treated as b=0 [s/mm²].
    tol : float
        Max |raw b-value - nearest library b-value| to accept a volume
        (default 30 s/mm²). Volumes farther than this from every library
        b-value are discarded.

    Returns
    -------
    bvals : (n_vols,) float ndarray  — raw values from the file
    b0_idx : (n_b0,) int ndarray
    shells : list of (b_value, idx_array) sorted by ascending b
        b_value is the library's own canonical value (not a rounded
        scanner-side representative); only library b-values with at least
        one matching volume appear.
    n_dropped : int
        Number of non-b0 volumes discarded (too far from every library
        b-value).
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
    lib_arr = np.asarray(sorted(set(lib_b_values)), dtype=float)

    # Nearest library b-value (and its distance) for every volume.
    dists = np.abs(nz_vals[:, None] - lib_arr[None, :])
    nearest_j = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(nz_vals.size), nearest_j]
    nearest_b = lib_arr[nearest_j]

    keep = nearest_dist <= tol
    n_dropped = int((~keep).sum())

    shells = []
    for lb in lib_arr:
        idx = nz_idx[keep & (nearest_b == lb)]
        if idx.size > 0:
            shells.append((float(lb), np.sort(idx)))
    shells.sort(key=lambda x: x[0])

    return raw, b0_idx, shells, n_dropped


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

def estimate_noise_sigma(b0_image, mask_brain, dilate_iters=16):
    """Estimate Rician noise sigma from a b=0 magnitude image's background.

    Background is voxels OUTSIDE a DILATED copy of the brain mask, not
    simply outside the raw mask. Skull/scalp tissue sits immediately
    outside a typical brain mask and its magnitude values are not pure
    Rayleigh-distributed noise (bone/marrow signal, motion, EPI ghosting
    near the skull) -- sampling background right up against the mask
    boundary biases the estimate high. Dilating the mask first pushes the
    sampled background out past the skull into cleaner air.

    Parameters
    ----------
    b0_image : ndarray (X,Y,Z)  a single b=0 magnitude volume
    mask_brain : bool ndarray (X,Y,Z)  brain mask
    dilate_iters : int  binary-dilation iterations applied to mask_brain
        before excluding it from the background region (default 8;
        0 reproduces the old "just outside the raw mask" behavior).

    Returns
    -------
    sigma : float or None (None if too few background voxels)
    n_bg  : int  number of background voxels actually used
    """
    if dilate_iters > 0:
        from scipy.ndimage import binary_dilation
        bg_mask = ~binary_dilation(mask_brain, iterations=dilate_iters)
    else:
        bg_mask = ~mask_brain

    # Drop zero voxels (often the FOV padding) - they're not noise samples
    bg_vals = b0_image[bg_mask]
    bg_vals = bg_vals[bg_vals > 0]

    n_bg = len(bg_vals)
    if n_bg < 100:
        return None, n_bg

    # Rayleigh: sigma = sqrt(<M^2>/2)
    sigma = float(np.sqrt(np.mean(bg_vals.astype(np.float64)**2) / 2.0))
    return sigma, n_bg


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
                         noise_bg_dilate_iters=16,
                         avg_s0=False,
                         return_raw=False,
                         z_slice=None,
                         b_tol=B_LIB_MATCH_TOL):
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
    noise_bg_dilate_iters : int
        [auto sigma only, ignored if noise_sigma is given] binary-dilation
        iterations applied to the mask before its background (~mask) is
        sampled for noise estimation -- see estimate_noise_sigma().
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
    sigma_used : float
        The noise standard deviation used for fitting.
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
            raw_shells = [(float(b), np.arange(sl.start, sl.stop, dtype=int))
                          for b, sl in LEGACY_SHELLS]
            # LEGACY_SHELLS's hardcoded b-values aren't necessarily on the
            # library's grid either -- match them the same way (nearest
            # library b-value within b_tol, else drop the whole shell;
            # there's no raw per-volume bvals array in legacy mode to
            # round individually, only these fixed nominal shell values).
            kept = []
            dropped = []
            for b, idx in raw_shells:
                nearest = min(lib_b_values, key=lambda lb: abs(lb - b))
                if abs(nearest - b) <= b_tol:
                    kept.append((float(nearest), idx))
                else:
                    dropped.append(b)
            if dropped:
                print(f"    ⚠ dropping legacy shells not in library: "
                      f"{[f'{b:g}' for b in dropped]} "
                      f"(library has {sorted(lib_b_values)})")
            if not kept:
                raise ValueError(
                    f"No LEGACY_SHELLS b-value matched any library "
                    f"b-value within {b_tol:g} s/mm² "
                    f"(library: {sorted(lib_b_values)}).")
            print(f"    LEGACY mode: 1 b=0 vol, {len(kept)} shells "
                  f"matched to library")
        else:
            bvals, b0_idx, shells, n_dropped = parse_bvals(
                bvals_path, lib_b_values, tol=b_tol)
            if bvals.size != n_vols:
                raise ValueError(
                    f"bvals length ({bvals.size}) ≠ DWI volumes ({n_vols}) "
                    f"for {dwi_path}")
            print(f"    bvals: {len(b0_idx)} b=0 vols, "
                  f"{len(shells)} shells matched to library "
                  f"(±{b_tol:g} s/mm²) "
                  f"({[f'b={int(b)}×{len(idx)}' for b, idx in shells]})")
            if n_dropped:
                print(f"    ⚠ discarded {n_dropped} volume(s) whose "
                      f"b-value is more than {b_tol:g} s/mm² from every "
                      f"library b-value {sorted(lib_b_values)} "
                      f"(not used in the fit)")
            if not shells:
                raise ValueError(
                    f"No volumes in {dwi_path} matched any library "
                    f"b-value within {b_tol:g} s/mm² "
                    f"(library: {sorted(lib_b_values)}).")

            # Optional bvecs uniformity check
            bvecs = parse_bvecs(bvecs_path, n_vols_expected=n_vols)
            if bvecs is not None:
                for b, idx in shells:
                    check_direction_uniformity(bvecs, idx,
                                               label=f"Δ={delta_ms:g}, b={int(b)}")

            kept = shells

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
            sigma_used, n_bg = estimate_noise_sigma(
                b0_for_sigma, mask_for_noise,
                dilate_iters=noise_bg_dilate_iters)
            if sigma_used is None:
                print(f"    ⚠ very few background voxels ({n_bg}) outside "
                      f"the {noise_bg_dilate_iters}x-dilated mask — "
                      f"disabling Rician correction. Try a smaller "
                      f"--noise-bg-dilate-iters.")
                rician_correct = False
            else:
                b0_brain_med = float(np.median(b0_for_sigma[mask_for_noise]))
                print(f"  Rician correction ENABLED")
                print(f"    sigma (auto, background, "
                      f"{noise_bg_dilate_iters}x-dilated-mask excluded, "
                      f"n={n_bg}) = {sigma_used:.2f}")
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
        # Mean number of averaged directions per kept (Δ,b) shell — used to
        # propagate Rician σ through shell averaging when auto-estimating σ_m.
        n_dir = [len(idx) for kept in all_shells_kept for _, idx in kept]
        mean_n_dir = float(np.mean(n_dir)) if n_dir else 1.0
        extras = dict(raw=raw_signal, s0=s0_ref, sigma=sigma_used,
                      mean_n_dir=mean_n_dir, s0_median=float(np.median(s0_ref)))
        return measured, fit_pairs, affine, mask_idx, shape, extras, sigma_used

    return measured, fit_pairs, affine, mask_idx, shape, sigma_used


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
    ap.add_argument("--export-voxel", type=int, nargs=3, default=None,
                    metavar=("I", "J", "K"),
                    help="Export ONE voxel's measured decay (normalized + "
                         "raw, in fit_pairs column order) to an .npz for "
                         "analysis/view_error_landscape_3d.py, instead of "
                         "fitting the whole volume. Uses the same "
                         "--input/--mask/--rician-correct/--noise-sigma/"
                         "--avg-s0/--library/--small-delta flags as --fit, "
                         "so the exported curve is built by the exact same "
                         "code path (load_dwi_and_average) as a real fit -- "
                         "no separate shell-averaging logic to drift out of "
                         "sync. Voxel indices (I,J,K) are in the --mask/DWI "
                         "native voxel grid.")

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
                         "of the first scan's b=0 image, drawn from outside "
                         "a dilated copy of --mask (see "
                         "--noise-bg-dilate-iters) so skull/scalp tissue "
                         "immediately outside the brain mask doesn't bias "
                         "the estimate.")
    ap.add_argument("--noise-bg-dilate-iters", type=int, default=48,
                    help="[auto sigma only] Number of binary-dilation "
                         "iterations applied to --mask before excluding it "
                         "from the background region used to estimate "
                         "noise sigma (default 64). Larger values push the "
                         "background sample further from the brain, past "
                         "the skull/scalp, at the cost of fewer background "
                         "voxels; tune upward if the auto sigma still looks "
                         "too high, downward if too few background voxels "
                         "remain for a small FOV.")
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

    # -- NEW: fitting method selection --
    method_grp = ap.add_argument_group(
        "fitting method",
        "Choose how each voxel is fit against the library.  'map' is the "
        "original point-estimate matcher and is byte-for-byte unchanged; "
        "'bayes' and 'amico' additionally emit posterior mean/std maps.")
    method_grp.add_argument(
        "--method", choices=["map", "bayes", "amico"], default="map",
        help="map: nearest-library-entry MAP estimate (default, "
             "backwards-compatible).  bayes: Gaussian posterior mean/std "
             "over the whole library.  amico: elastic-net NNLS mixture.")
    method_grp.add_argument(
        "--sigma-m", type=float, default=None,
        help="[bayes only] Residual-noise std on the normalized S/S0 "
             "signal.  If omitted, auto-estimated from Rician σ when "
             "--rician-correct is on, else defaults to "
             f"{DEFAULT_SIGMA_M} (a placeholder — a warning is logged).")
    method_grp.add_argument(
        "--lambda1", type=float, default=DEFAULT_LAMBDA1,
        help=f"[amico only] L1 (sparsity) penalty (default {DEFAULT_LAMBDA1}).")
    method_grp.add_argument(
        "--lambda2", type=float, default=DEFAULT_LAMBDA2,
        help=f"[amico only] L2 (ridge) penalty (default {DEFAULT_LAMBDA2}).")
    method_grp.add_argument(
        "--device", choices=["auto", "cpu", "gpu"], default="auto",
        help="auto (default): use CUDA if available, else CPU.  gpu: force "
             "GPU (errors if CUDA is unavailable).  cpu: force CPU.  MAP and "
             "bayes GPU kernels are exact reorderings of the CPU math "
             "(same output to float64 precision); amico's GPU path is an "
             "approximate FISTA solve replacing the CPU's exact per-voxel "
             "NNLS, so amico GPU/CPU outputs will be close but not "
             "bit-identical (see docs/fitting_methods.md).")
    method_grp.add_argument(
        "--gpu-chunk-voxels", type=int,
        default=fitters_gpu.DEFAULT_GPU_CHUNK_VOXELS,
        help="[amico + --device gpu only] voxels per GPU batch for the "
             f"FISTA solve (default {fitters_gpu.DEFAULT_GPU_CHUNK_VOXELS}). "
             "MAP/bayes GPU kernels process all voxels in one launch and "
             "ignore this.")
    method_grp.add_argument(
        "--amico-gpu-iters", type=int,
        default=fitters_gpu.DEFAULT_AMICO_ITERS,
        help="[amico + --device gpu only] max FISTA iterations per voxel "
             f"(default {fitters_gpu.DEFAULT_AMICO_ITERS}).")
    method_grp.add_argument(
        "--amico-gpu-tol", type=float,
        default=fitters_gpu.DEFAULT_AMICO_TOL,
        help="[amico + --device gpu only] relative-objective-change "
             f"early-exit tolerance for the FISTA solve (default "
             f"{fitters_gpu.DEFAULT_AMICO_TOL}). Convergence speed depends "
             "strongly on --lambda2 (it also conditions the problem for "
             "this solver, since MADI library entries are highly "
             "correlated): near --lambda2 0, n_eff can look stuck high even "
             "though it is still slowly dropping — raise --amico-gpu-iters "
             "substantially (tens of thousands) in that regime, or use "
             "--device cpu for an exact answer.")
    # -- NEW: acquisition metadata (must match library) --
    ap.add_argument("--small-delta", type=float, default=None,
                    help="δ (PFG duration) [ms] of the data being fit.  "
                         "Must match the library's small δ (within 0.05 ms).  "
                         "If omitted, the library's stored small δ is used "
                         "and a warning is printed.")
    # -- Matcher tuning (already exposed in library.py but worth making CLI) --
    ap.add_argument("--vi-min", type=float, default=0.0,
                    help="Lower bound on intracellular volume fraction "
                         "for library candidates.")
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

    if not any([args.build_library, args.fit, args.info,
                args.export_voxel is not None]):
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
    #  EXPORT ONE VOXEL'S DECAY CURVE (for view_error_landscape_3d.py)
    # ================================================================
    if args.export_voxel is not None:
        i_vox, j_vox, k_vox = args.export_voxel

        input_specs = []
        if args.input:
            for s in args.input:
                input_specs.append(parse_input(s))
        else:
            legacy_map = [(15.0, args.dwi15), (25.0, args.dwi25),
                          (30.0, args.dwi30), (40.0, args.dwi40)]
            for delta, path in legacy_map:
                if path is not None:
                    input_specs.append((delta, path, None, None))
        if not input_specs:
            print("ERROR: No DWI inputs specified."); return
        input_specs.sort(key=lambda x: x[0])

        if not os.path.exists(args.library):
            print(f"ERROR: Library not found: {args.library}"); return
        meta = load_library_meta(args.library)
        lib_b_values = meta['b_values']
        if lib_b_values is None:
            print("ERROR: library has no stored b-values metadata."); return

        print("=" * 60)
        print(f"Exporting voxel ({i_vox}, {j_vox}, {k_vox})")
        print("=" * 60)
        measured, fit_pairs, affine, mask_idx, shape, extras, sigma_used = \
            load_dwi_and_average(
                input_specs, args.mask,
                lib_b_values=lib_b_values,
                rician_correct=args.rician_correct,
                noise_sigma=args.noise_sigma,
                noise_bg_dilate_iters=args.noise_bg_dilate_iters,
                avg_s0=args.avg_s0,
                return_raw=True,
            )

        hit = ((mask_idx[0] == i_vox) & (mask_idx[1] == j_vox) &
               (mask_idx[2] == k_vox))
        pos = np.where(hit)[0]
        if pos.size == 0:
            print(f"ERROR: voxel ({i_vox},{j_vox},{k_vox}) is not in "
                  f"--mask (or falls outside the volume, shape {shape}). "
                  f"Nothing exported.")
            return
        pos = int(pos[0])

        measured_vec = measured[pos]
        raw_vec = extras['raw'][pos]
        s0 = float(extras['s0'][pos])
        deltas = np.array([d for d, _ in fit_pairs], dtype=float)
        bvals = np.array([b for _, b in fit_pairs], dtype=float)

        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(
            args.out, f"voxel_{i_vox}_{j_vox}_{k_vox}.npz")
        np.savez(
            out_path,
            measured=measured_vec, raw=raw_vec,
            fit_deltas=deltas, fit_bvals=bvals,
            s0=s0, sigma=(sigma_used if sigma_used is not None else np.nan),
            mean_n_dir=extras['mean_n_dir'], s0_median=extras['s0_median'],
            ijk=np.array([i_vox, j_vox, k_vox], dtype=int),
            affine=affine, library=args.library,
            rician_correct=bool(args.rician_correct),
        )
        print(f"\n  S0 = {s0:.1f}   sigma = {sigma_used}")
        print(f"  measured (S/S0): {np.array2string(measured_vec, precision=4)}")
        print(f"  fit_pairs: {list(zip(deltas.tolist(), bvals.tolist()))}")
        print(f"\n  Saved {out_path}")
        print("  Open with: python analysis/view_error_landscape_3d.py "
              f"--library {args.library} --voxel-data {out_path}")
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

        # ---- Warn about method-specific flags that are ignored ----
        if args.method != "bayes" and args.sigma_m is not None:
            print(f"  ⚠ --sigma-m is only used by --method bayes; "
                  f"ignored for --method {args.method}.")
        if args.method != "amico":
            if args.lambda1 != DEFAULT_LAMBDA1:
                print(f"  ⚠ --lambda1 is only used by --method amico; "
                      f"ignored for --method {args.method}.")
            if args.lambda2 != DEFAULT_LAMBDA2:
                print(f"  ⚠ --lambda2 is only used by --method amico; "
                      f"ignored for --method {args.method}.")
        if args.method == "amico" and args.log_space:
            print(f"  ⚠ --log_space has no effect for --method amico "
                  f"(the regression is linear in signal); ignored.")

        # ---- Resolve --device -> use_gpu, threaded into every fit call ----
        if args.device == "auto":
            use_gpu = None  # let each fitter auto-detect (None = HAS_CUDA)
        elif args.device == "gpu":
            if not fitters_gpu.HAS_CUDA:
                print("ERROR: --device gpu requested but CUDA is not "
                      "available in this environment."); return
            use_gpu = True
        else:  # cpu
            use_gpu = False
        resolved_device = ("gpu" if (use_gpu or
                           (use_gpu is None and fitters_gpu.HAS_CUDA))
                           else "cpu")

        gpu_only_flags_touched = (
            args.gpu_chunk_voxels != fitters_gpu.DEFAULT_GPU_CHUNK_VOXELS or
            args.amico_gpu_iters != fitters_gpu.DEFAULT_AMICO_ITERS or
            args.amico_gpu_tol != fitters_gpu.DEFAULT_AMICO_TOL)
        if gpu_only_flags_touched and not (args.method == "amico" and resolved_device == "gpu"):
            print("  ⚠ --gpu-chunk-voxels/--amico-gpu-iters/--amico-gpu-tol "
                  "only affect --method amico with --device gpu (resolved: "
                  f"method={args.method}, device={resolved_device}); ignored.")

        print("=" * 60)
        print("MADI Fitting")
        print("=" * 60)
        print(f"  Method:                 {args.method}")
        print(f"  Device:                 {resolved_device} "
              f"(--device {args.device})")
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
        # bayes/amico always need the extras dict (raw signal for --fit-s0,
        # plus σ / S0 / n_dir for σ_m auto-estimation), so request it for any
        # non-map method regardless of --fit-s0.
        need_extras = args.fit_s0 or (args.method != "map")
        print("\nLoading DWI data ...")
        z_slice_obj = parse_z_slice(args.z_slice)
        load_out = load_dwi_and_average(
            input_specs, args.mask,
            lib_b_values=lib_b_values,
            rician_correct=args.rician_correct,
            noise_sigma=args.noise_sigma,
            noise_bg_dilate_iters=args.noise_bg_dilate_iters,
            avg_s0=args.avg_s0,
            return_raw=need_extras,
            z_slice=z_slice_obj,
        )


        if need_extras:
            measured, fit_pairs, affine, mask_idx, shape, extras, sigma_used = load_out
        else:
            measured, fit_pairs, affine, mask_idx, shape, sigma_used = load_out
            extras = None

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

        # ================================================================
        #  METHOD DISPATCH
        # ================================================================
        if args.method == "map":
            # ---- MAP: point-estimate matcher (UNCHANGED behaviour) ----
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
                    use_gpu=use_gpu,
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
                    use_gpu=use_gpu,
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
            return

        # ---- bayes / amico: distributional fitters ----
        raw_signal = extras['raw'] if args.fit_s0 else None
        method_meta = {}   # method-specific params recorded in JSON

        if args.method == "bayes":
            # Resolve σ_m: user > auto-from-Rician > placeholder default.
            if args.sigma_m is not None:
                sigma_m = float(args.sigma_m)
                sigma_src = "user"
                print(f"\n  σ_m = {sigma_m:.4g}  (user-specified)")
            else:
                est = estimate_sigma_m(extras.get('sigma'),
                                       extras.get('s0_median'),
                                       extras.get('mean_n_dir'))
                if args.rician_correct and est is not None and est > 0:
                    sigma_m = est
                    sigma_src = "auto-rician"
                    print(f"\n  σ_m = {sigma_m:.4g}  (auto from Rician σ="
                          f"{extras['sigma']:.2f}, S0_med="
                          f"{extras['s0_median']:.1f}, "
                          f"mean n_dir={extras['mean_n_dir']:.1f})")
                else:
                    sigma_m = DEFAULT_SIGMA_M
                    sigma_src = "default-placeholder"
                    print(f"\n  ⚠ σ_m = {sigma_m:.4g}  (PLACEHOLDER default — "
                          f"pass --sigma-m or --rician-correct for a "
                          f"data-driven value).")
            method_meta = dict(sigma_m=sigma_m, sigma_m_source=sigma_src)

            print(f"\nBayes posterior over {measured.shape[0]} voxels "
                  f"({'S0 FITTED' if args.fit_s0 else 'S0 fixed'}) ...")
            t0 = time.time()
            res = bayes_fit(
                measured, lib,
                sigma_m=sigma_m,
                fit_pairs=fit_pairs, lib_deltas=lib_deltas,
                lib_b_values=lib_b_values, n_b=lib_n_b,
                vi_min=args.vi_min, vi_max=args.vi_max, rho_max=args.rho_max,
                log_space=args.log_space,
                fit_s0=args.fit_s0, raw_signal=raw_signal,
                use_gpu=use_gpu,
            )
            print(f"  Done in {time.time()-t0:.1f}s")

        else:  # amico
            print(f"\n  AMICO elastic-net: λ1={args.lambda1:g} (L1), "
                  f"λ2={args.lambda2:g} (L2)")
            method_meta = dict(lambda1=float(args.lambda1),
                               lambda2=float(args.lambda2))
            if resolved_device == "gpu":
                method_meta.update(
                    gpu_chunk_voxels=args.gpu_chunk_voxels,
                    amico_gpu_iters=args.amico_gpu_iters,
                    amico_gpu_tol=args.amico_gpu_tol)
            print(f"\nAMICO NNLS over {measured.shape[0]} voxels "
                  f"({'S0 FITTED' if args.fit_s0 else 'S0 fixed'}) ...")
            t0 = time.time()
            res = amico_fit(
                measured, lib,
                lambda1=args.lambda1, lambda2=args.lambda2,
                fit_pairs=fit_pairs, lib_deltas=lib_deltas,
                lib_b_values=lib_b_values, n_b=lib_n_b,
                vi_min=args.vi_min, vi_max=args.vi_max, rho_max=args.rho_max,
                fit_s0=args.fit_s0, raw_signal=raw_signal,
                use_gpu=use_gpu,
                gpu_chunk_voxels=args.gpu_chunk_voxels,
                gpu_n_iters=args.amico_gpu_iters,
                gpu_tol=args.amico_gpu_tol,
            )
            print(f"  Done in {time.time()-t0:.1f}s")

        # ---- Stats (weighted means + posterior std) ----
        print(f"\n  kio_mean: median={np.median(res['kio_mean']):.1f}, "
              f"range=[{res['kio_mean'].min():.1f}, {res['kio_mean'].max():.1f}] s-1")
        print(f"  rho_mean: median={np.median(res['rho_mean'])/1e3:.0f}k cells/uL")
        print(f"  V_mean:   median={np.median(res['V_mean']):.2f} pL")
        print(f"  kio_std:  median={np.median(res['kio_std']):.2f}")
        print(f"  rho_std:  median={np.median(res['rho_std'])/1e3:.1f}k")
        print(f"  V_std:    median={np.median(res['V_std']):.3f}")
        print(f"  n_eff:    median={np.median(res['n_eff']):.2f} "f"(effective # of library atoms per voxel)")

        # ---- Save maps ----
        print("\nSaving maps ...")
        save_map(res['kio_mean'], mask_idx, shape, affine,
                 os.path.join(args.out, "kio_mean.nii.gz"))
        save_map(res['rho_mean'], mask_idx, shape, affine,
                 os.path.join(args.out, "rho_mean.nii.gz"))
        save_map(res['V_mean'], mask_idx, shape, affine,
                 os.path.join(args.out, "V_mean.nii.gz"))
        save_map(res['kio_std'], mask_idx, shape, affine,
                 os.path.join(args.out, "kio_std.nii.gz"))
        save_map(res['rho_std'], mask_idx, shape, affine,
                 os.path.join(args.out, "rho_std.nii.gz"))
        save_map(res['V_std'], mask_idx, shape, affine,
                 os.path.join(args.out, "V_std.nii.gz"))
        save_map(res['residual'], mask_idx, shape, affine,
                 os.path.join(args.out, "residual.nii.gz"))
        save_map(res['n_eff'], mask_idx, shape, affine, os.path.join(args.out, "n_eff.nii.gz"))
        if "s0_fit" in res:
            save_map(res['s0_fit'], mask_idx, shape, affine,
                     os.path.join(args.out, "s0_fit_map.nii.gz"))

        # ---- JSON run metadata ----
        import json
        print("Saving metadata ...")

        sigma = args.noise_sigma if args.noise_sigma is not None else sigma_used
        meta_out = dict(
            method=args.method,
            device=resolved_device,
            library=args.library,
            inputs=[list(s) for s in input_specs],
            fit_deltas=fit_deltas,
            fit_pairs=[[float(d), float(b)] for d, b in fit_pairs],
            n_features=n_features,
            rician_correct=bool(args.rician_correct),
            noise_sigma=sigma,
            noise_bg_dilate_iters=(args.noise_bg_dilate_iters
                                    if args.rician_correct and
                                    args.noise_sigma is None else None),
            avg_s0=bool(args.avg_s0),
            fit_s0=bool(args.fit_s0),
            log_space=bool(args.log_space),
            vi_min=args.vi_min, vi_max=args.vi_max, rho_max=args.rho_max,
            n_voxels=int(measured.shape[0]),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            **method_meta,
        )
        meta_path = os.path.join(args.out, "fit_metadata.json")
        with open(meta_path, "w") as fh:
            json.dump(meta_out, fh, indent=2)
        print(f"  Saved {meta_path}")

        print("\nDone!")


if __name__ == "__main__":
    main()
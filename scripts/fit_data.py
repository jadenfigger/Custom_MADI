#!/usr/bin/env python3
"""
fit_data.py — Build MADI libraries & fit in-vivo DWI data
==========================================================

NEW IN THIS VERSION
-------------------
1. **Rician noise-bias correction** (--rician-correct)
       Uses the second-moment identity  E[M^2] = A^2 + 2 sigma^2
       to recover the underlying true signal A from magnitude data M
       before normalization.  sigma is auto-estimated from a background
       (air) region of each volume, or supplied via --noise-sigma.

2. **S0 averaging across Delta scans** (--avg-s0)
       Because TE is fixed across all Delta acquisitions in this
       protocol (PPR confirms TE=54 ms regardless of big_delta), the
       four b=0 images should differ only by noise.  Averaging gives
       up to 2x SNR on S0, which propagates as reduced noise in EVERY
       normalized data point.  ONLY safe if the four scans are well
       co-registered or motion is negligible.

3. **Optional S0 fitting in the matcher** (--fit-s0)
       Instead of dividing by the measured b=0 and matching ratios,
       treat S0 as a free linear parameter per voxel and find the
       (kio, rho, V, S0) combo that best matches the *un-normalized*
       signal.  S0 is solved analytically per library entry as the
       L2-optimal projection.  Compare against the default (fixed S0)
       to diagnose S0-related biases.

LIBRARY BUILDING
----------------
[unchanged - see previous docstring]

FITTING
-------
  python fit_data.py --fit \\
      --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \\
      --mask mask.nii.gz \\
      --rician-correct --avg-s0

  Compare with S0 fitted as free parameter:
  python fit_data.py --fit --input ... --mask ... --rician-correct --fit-s0
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
# Acquisition protocol
# ===================================================================

SHELLS = [
    (1000, slice(1, 25)),
    (2500, slice(25, 49)),
    (4000, slice(49, 73)),
    (6000, slice(73, 97)),
]
N_SHELLS = len(SHELLS)


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
        "cfg":  dict(n_walkers=100_000, n_ensembles=40, n_steps=50_000,
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
    if ":" not in s:
        raise ValueError(f"Input must be 'delta:/path/to/file.nii.gz', got '{s}'")
    delta_str, path = s.split(":", 1)
    return (float(delta_str), path)


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
                         rician_correct=False,
                         noise_sigma=None,
                         avg_s0=False,
                         return_raw=False):
    """Load DWI NIfTIs for arbitrary delta values.

    Parameters
    ----------
    input_specs : list of (delta_ms, path) tuples
    mask_path : str
    rician_correct : if True, apply E[M^2]=A^2+2*sigma^2 correction
                     to every voxel of every volume before averaging
                     and normalization.
    noise_sigma : float or None.  If None and rician_correct, sigma
                  is estimated from background voxels of the FIRST
                  acquisition's b=0 volume.
    avg_s0 : if True, average the b=0 volumes across all Delta scans
             into a single S0 image and use it for normalizing every
             shell at every Delta.  Only safe if scans are co-registered.
    return_raw : if True, also return the un-normalized per-shell
                 means and the S0 used (for --fit-s0 mode).

    Returns
    -------
    measured : ndarray (n_voxels, n_deltas * n_shells)  S/S0
    fit_deltas : list of delta values [ms]
    affine, mask_indices, shape
    extras (if return_raw) : dict with 'raw' (n_voxels, n_deltas*n_shells),
                             's0' (n_voxels,), 'sigma' (float or None)
    """
    import nibabel as nib

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine
    shape = mask.shape
    mask_idx = np.where(mask)
    n_vox = len(mask_idx[0])

    print(f"  Mask voxels: {n_vox}")

    input_specs = sorted(input_specs, key=lambda x: x[0])
    fit_deltas = [d for d, _ in input_specs]
    n_deltas = len(input_specs)

    # ---- Pass 1: load all volumes and (optionally) estimate sigma ----
    all_data = []
    sigma_used = None

    for di, (delta_ms, dwi_path) in enumerate(input_specs):
        print(f"  Loading Δ={delta_ms:.0f}ms: {os.path.basename(dwi_path)} ...",
              end=" ", flush=True)
        img = nib.load(dwi_path)
        data = img.get_fdata().astype(np.float64)
        all_data.append(data)
        print(f"shape={data.shape}")

    # ---- Noise sigma estimation (uses first scan's b=0 + background) ----
    if rician_correct:
        if noise_sigma is not None:
            sigma_used = float(noise_sigma)
            print(f"  Rician correction ENABLED, user-provided sigma={sigma_used:.2f}")
        else:
            sigma_used = estimate_noise_sigma(all_data[0], mask)
            if sigma_used is None:
                print("  Could not estimate sigma; disabling Rician correction.")
                rician_correct = False
            else:
                # Report SNR of typical brain b=0 voxel for context
                b0_brain_median = float(np.median(all_data[0][..., 0][mask]))
                print(f"  Rician correction ENABLED")
                print(f"    sigma (auto, background)  = {sigma_used:.2f}")
                print(f"    median brain b=0 signal   = {b0_brain_median:.1f}")
                print(f"    median brain b=0 SNR      = {b0_brain_median/sigma_used:.1f}")

    # ---- Apply Rician correction in-place to all volumes ----
    if rician_correct:
        for di in range(n_deltas):
            all_data[di] = rician_correct_secondmoment(all_data[di], sigma_used)

    # ---- S0 handling: per-Delta or averaged across Deltas ----
    s0_per_delta = []  # list of (n_vox,) arrays
    for di in range(n_deltas):
        vox_data = all_data[di][mask, :]   # (n_vox, 97)
        s0_per_delta.append(vox_data[:, 0].copy())

    if avg_s0:
        # Stack and average; shape (n_deltas, n_vox)
        s0_stack = np.stack(s0_per_delta, axis=0)
        s0_common = np.mean(s0_stack, axis=0)
        # Per-voxel CV of the four S0s, useful diagnostic
        s0_cv = np.std(s0_stack, axis=0) / (s0_common + 1e-10)
        print(f"  S0 averaging ENABLED (TE constant across Δ confirmed by PPR)")
        print(f"    median S0 across-Δ CV = {np.median(s0_cv)*100:.2f}%   "
              f"(low = scans well registered / no drift)")
        print(f"    95th-pct CV           = {np.percentile(s0_cv, 95)*100:.2f}%")
        if np.median(s0_cv) > 0.10:
            print(f"    ⚠ High S0 variability across Δ — check for motion/drift "
                  f"before trusting averaged S0.")
        # Use the average for every Delta
        s0_used = [s0_common.copy() for _ in range(n_deltas)]
    else:
        s0_used = s0_per_delta

    # ---- Build normalized measured matrix and (optionally) raw matrix ----
    measured = np.zeros((n_vox, n_deltas * N_SHELLS))
    raw_signal = np.zeros((n_vox, n_deltas * N_SHELLS))

    for di, (delta_ms, _) in enumerate(input_specs):
        vox_data = all_data[di][mask, :]
        S0 = s0_used[di].copy()
        S0[S0 < 1e-10] = 1e-10

        for si, (b_val, vol_slice) in enumerate(SHELLS):
            shell_mean = np.mean(vox_data[:, vol_slice], axis=1)
            measured[:, di * N_SHELLS + si] = shell_mean / S0
            raw_signal[:, di * N_SHELLS + si] = shell_mean

    if return_raw:
        # For --fit-s0 mode we need a single per-voxel S0 reference
        # (only used as a starting estimate / scale).
        s0_ref = np.mean(np.stack(s0_used, axis=0), axis=0)
        extras = dict(raw=raw_signal, s0=s0_ref, sigma=sigma_used)
        return measured, fit_deltas, affine, mask_idx, shape, extras

    return measured, fit_deltas, affine, mask_idx, shape


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
    ap.add_argument("--mask")
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
        library_summary(lib)
        print(f"  Δ values: {meta['deltas']} ms")
        print(f"  b-values per Δ: {meta['n_b']}")
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
                    input_specs.append((delta, path))

        if not input_specs:
            print("ERROR: No DWI inputs specified."); return
        if args.mask is None:
            print("ERROR: --mask is required for --fit"); return

        input_specs.sort(key=lambda x: x[0])
        fit_deltas = [d for d, _ in input_specs]

        print("=" * 60)
        print("MADI Fitting")
        print("=" * 60)
        print(f"  Δ values to fit:        {fit_deltas} ms")
        print(f"  Signal vector length:   {len(fit_deltas) * N_SHELLS}")
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
        lib_deltas = meta['deltas']
        n_b = meta['n_b']
        print(f"  {len(lib)} entries, lib Δ = {lib_deltas} ms")
        library_summary(lib)

        for d in fit_deltas:
            if not any(abs(d - ld) < 0.01 for ld in lib_deltas):
                print(f"ERROR: Δ = {d} ms not in library ({lib_deltas})"); return

        # Load data — get raw signals too if we'll fit S0
        print("\nLoading DWI data ...")
        load_out = load_dwi_and_average(
            input_specs, args.mask,
            rician_correct=args.rician_correct,
            noise_sigma=args.noise_sigma,
            avg_s0=args.avg_s0,
            return_raw=args.fit_s0,
        )

        if args.fit_s0:
            measured, fit_deltas, affine, mask_idx, shape, extras = load_out
            raw_signal = extras['raw']
            print(f"\nMatching {raw_signal.shape[0]} voxels with S0 FITTED "
                  f"(using Δ = {fit_deltas} ms) ...")
            t0 = time.time()
            kio_map, rho_map, V_map, res_map, s0_fit_map = match_voxels_batch_fits0(
                raw_signal, lib,
                fit_deltas=fit_deltas,
                lib_deltas=lib_deltas,
                n_b=n_b,
                vi_min=args.vi_min,
                vi_max=args.vi_max,
                rho_max=args.rho_max,
            )
            print(f"  Done in {time.time()-t0:.1f}s")
            # Save the fitted S0 map for inspection
            save_map(s0_fit_map, mask_idx, shape, affine,
                     os.path.join(args.out, "s0_fit_map.nii.gz"))
            # Diagnostic: ratio of fitted S0 to measured b=0 mean
            s0_ratio = s0_fit_map / (extras['s0'] + 1e-10)
            save_map(s0_ratio, mask_idx, shape, affine,
                     os.path.join(args.out, "s0_fit_over_measured.nii.gz"))
            print(f"\n  Fitted-S0 / Measured-S0 ratio:")
            print(f"    median = {np.median(s0_ratio):.3f}")
            print(f"    5-95%  = [{np.percentile(s0_ratio, 5):.3f}, "
                  f"{np.percentile(s0_ratio, 95):.3f}]")
            print(f"    (deviation from 1.0 indicates S0 mismatch — "
                  f"voxels with ratio >> 1 may have b=0 underestimated, "
                  f"e.g. due to Rician bias on a low-SNR b=0)")
        else:
            measured, fit_deltas, affine, mask_idx, shape = load_out
            print(f"\nMatching {measured.shape[0]} voxels (using Δ = {fit_deltas} ms) ...")
            t0 = time.time()
            kio_map, rho_map, V_map, res_map = match_voxels_batch(
                measured, lib,
                fit_deltas=fit_deltas,
                lib_deltas=lib_deltas,
                n_b=n_b,
                vi_min=args.vi_min,
                vi_max=args.vi_max,
                rho_max=args.rho_max,
                log_space=args.log_space
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
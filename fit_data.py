#!/usr/bin/env python3
"""
fit_data.py — Fit MADI model to your in-vivo mouse DWI data
=============================================================

Workflow:
    1. Build (or load) a MADI simulation library
    2. Load your 4 preprocessed DWI volumes + mask
    3. Extract directional-average S(b)/S0 per voxel per delta
    4. Match each voxel to the closest library entry
    5. Save kio, rho, V parametric maps as NIfTI

Usage
-----
    # Step 1: build library (do this once — takes 30-60 min with GPU)
    python fit_data.py --build-library

    # Step 2: fit your data
    python fit_data.py --fit \
        --dwi15 /path/to/DWI_15ms/eddy_corrected.nii.gz \
        --dwi25 /path/to/DWI_25ms/eddy_corrected.nii.gz \
        --dwi30 /path/to/DWI_30ms/eddy_corrected.nii.gz \
        --dwi40 /path/to/DWI_40ms/eddy_corrected.nii.gz \
        --mask  /path/to/mask.nii.gz \
        --out   /path/to/madi_output/
"""

import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from madi.config   import SimConfig, BVALS_S_MM2, BVALS_UNIQUE, DELTAS_BIG
from madi.library  import build_library, load_library, match_voxels_batch
from madi.signal   import signals_to_flat


# ===================================================================
# Acquisition protocol (EDIT if yours differs)
# ===================================================================

# b-values per volume (97 volumes: 1 b0 + 4 shells x 24 dirs)
BVALS = BVALS_S_MM2.copy()

# Shell boundaries for directional averaging
SHELLS = [
    (1000, slice(1, 25)),      # volumes 1-24
    (2500, slice(25, 49)),     # volumes 25-48
    (4000, slice(49, 73)),     # volumes 49-72
    (6000, slice(73, 97)),     # volumes 73-96
]

DELTAS_MS = DELTAS_BIG   # [15, 25, 30, 40] ms


# ===================================================================
# Data loading
# ===================================================================

def load_dwi_and_average(dwi_paths, mask_path):
    """Load DWI NIfTIs, apply mask, compute directional-average S/S0.

    Returns
    -------
    measured : ndarray (n_voxels, n_deltas * n_shells)
        Flattened normalised signal vectors for masked voxels.
    affine : ndarray (4, 4)
    mask_indices : tuple of index arrays
    shape : tuple
    """
    import nibabel as nib

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine
    shape = mask.shape
    mask_idx = np.where(mask)
    n_vox = len(mask_idx[0])

    print(f"  Mask voxels: {n_vox}")

    n_deltas = len(dwi_paths)
    n_shells = len(SHELLS)
    measured = np.zeros((n_vox, n_deltas * n_shells))

    for di, dwi_path in enumerate(dwi_paths):
        print(f"  Loading {os.path.basename(dwi_path)} ...", end=" ", flush=True)
        img = nib.load(dwi_path)
        data = img.get_fdata()

        # Extract masked voxels: (n_vox, n_volumes)
        vox_data = data[mask_idx[0], mask_idx[1], mask_idx[2], :]

        # S0 = b=0 volume (first volume)
        S0 = vox_data[:, 0].copy()
        S0[S0 < 1e-10] = 1e-10    # avoid division by zero

        for si, (b_val, vol_slice) in enumerate(SHELLS):
            # Directional average across the 24 directions in this shell
            shell_data = vox_data[:, vol_slice]
            shell_mean = np.mean(shell_data, axis=1)
            S_norm = shell_mean / S0
            S_norm = np.clip(S_norm, 0.0, 1.0)
            measured[:, di * n_shells + si] = S_norm

        print("done")

    return measured, affine, mask_idx, shape


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
    ap = argparse.ArgumentParser(description="MADI fitting for in-vivo mouse DWI")
    ap.add_argument("--build-library", action="store_true",
                    help="Build simulation library (run once)")
    ap.add_argument("--fit", action="store_true",
                    help="Fit data using pre-built library")
    ap.add_argument("--library", default="madi_library.npz",
                    help="Path to library .npz file")
    ap.add_argument("--dwi15", help="Preprocessed DWI for Delta=15ms")
    ap.add_argument("--dwi25", help="Preprocessed DWI for Delta=25ms")
    ap.add_argument("--dwi30", help="Preprocessed DWI for Delta=30ms")
    ap.add_argument("--dwi40", help="Preprocessed DWI for Delta=40ms")
    ap.add_argument("--mask",  help="Brain mask NIfTI")
    ap.add_argument("--out",   default="madi_output", help="Output directory")
    # Library grid overrides
    ap.add_argument("--lib-preset", default="default",
                    choices=["small", "default", "dense"],
                    help="Library resolution")
    args = ap.parse_args()

    if not args.build_library and not args.fit:
        ap.print_help()
        return

    # ---- BUILD LIBRARY -----------------------------------------------
    if args.build_library:
        print("=" * 50)
        print("Building MADI library")
        print("=" * 50)

        if args.lib_preset == "small":
            kios = [5, 12, 25, 50]
            rhos = [20_000, 40_000, 80_000, 150_000, 300_000]
            Vs   = [1.0, 2.0, 3.5]
            cfg  = SimConfig(n_walkers=5_000, n_ensembles=2, n_steps=50_000,
                             L=180.0, buffer=45.0, grid_spacing=1.2)
        elif args.lib_preset == "default":
            kios = [2, 5, 8, 12, 18, 25, 35, 50, 75]
            rhos = [100_000, 200_000, 300_000, 400_000, 600_000, 800_000]
            Vs   = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
            cfg  = SimConfig(n_walkers=15_000, n_ensembles=5, n_steps=50_000,
                             L=250.0, buffer=60.0, grid_spacing=1.0)
        elif args.lib_preset == "dense":
            kios = [2, 4, 6, 8, 10, 12, 15, 18, 22, 25, 30, 35, 45, 60, 80, 100]
            rhos = [100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 600_000, 800_000]
            Vs   = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
            cfg  = SimConfig(n_walkers=25_000, n_ensembles=8, n_steps=50_000,
                             L=300.0, buffer=70.0, grid_spacing=0.8)

        build_library(kios, rhos, Vs, cfg, save_path=args.library)
        return

    # ---- FIT DATA ----------------------------------------------------
    if args.fit:
        for attr in ['dwi15', 'dwi25', 'dwi30', 'dwi40', 'mask']:
            if getattr(args, attr) is None:
                print(f"ERROR: --{attr} is required for --fit")
                return

        print("=" * 50)
        print("MADI Fitting")
        print("=" * 50)

        os.makedirs(args.out, exist_ok=True)

        # Load library
        print(f"\nLoading library: {args.library}")
        lib = load_library(args.library)
        print(f"  {len(lib)} entries")

        # Load and process data
        print("\nLoading DWI data ...")
        dwi_paths = [args.dwi15, args.dwi25, args.dwi30, args.dwi40]
        measured, affine, mask_idx, shape = load_dwi_and_average(dwi_paths, args.mask)

        # Match
        print(f"\nMatching {measured.shape[0]} voxels to library ...")
        t0 = time.time()
        kio_map, rho_map, V_map, res_map = match_voxels_batch(measured, lib)
        print(f"  Done in {time.time()-t0:.1f}s")

        # Stats
        print(f"\n  kio:  median={np.median(kio_map):.1f}, "
              f"range=[{kio_map.min():.1f}, {kio_map.max():.1f}] s-1")
        print(f"  rho:  median={np.median(rho_map)/1e3:.0f}k, "
              f"range=[{rho_map.min()/1e3:.0f}k, {rho_map.max()/1e3:.0f}k] cells/uL")
        print(f"  V:    median={np.median(V_map):.2f}, "
              f"range=[{V_map.min():.2f}, {V_map.max():.2f}] pL")

        # Save maps
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

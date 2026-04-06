#!/usr/bin/env python3
"""
fit_data.py — Build MADI libraries & fit in-vivo DWI data
==========================================================

LIBRARY BUILDING
----------------
Three modes for specifying which (kio, rho, V) to simulate:

  1. PRESET GRID: full cross-product of a named preset
     python fit_data.py --build-library --lib-preset default

  2. CUSTOM ADDITIONS: merge extra values into the preset grid
     python fit_data.py --build-library --append --custom-rhos 1500000 2000000

  3. EXPLICIT SUB-GRID: only cross the values you specify
     python fit_data.py --build-library --append --explicit \\
         --grid-kios 12 25 --grid-rhos 1500000 2000000 --grid-Vs 0.5 1.0 2.0

  4. EXACT TRIPLETS: individual (kio, rho, V) points
     python fit_data.py --build-library --append \\
         --triplets 12,1500000,0.5  25,2000000,1.0

  All modes support --append to skip already-computed entries.

FITTING
-------
Specify any number of DWI files with their Δ values:

  python fit_data.py --fit \\
      --input 15:/path/to/DWI_15ms.nii.gz \\
      --input 25:/path/to/DWI_25ms.nii.gz \\
      --mask /path/to/mask.nii.gz

  Or use all four:
  python fit_data.py --fit \\
      --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \\
      --mask mask.nii.gz
"""

import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from madi.config   import SimConfig, BVALS_S_MM2, BVALS_UNIQUE, DELTAS_BIG
from madi.library  import (build_library, build_library_from_triplets,
                            load_library, load_library_meta,
                            match_voxels_batch, library_summary)
from madi.signal   import signals_to_flat


# ===================================================================
# Acquisition protocol (EDIT if yours differs)
# ===================================================================

# Shell boundaries for directional averaging within each DWI volume
SHELLS = [
    (1000, slice(1, 25)),      # volumes 1-24
    (2500, slice(25, 49)),     # volumes 25-48
    (4000, slice(49, 73)),     # volumes 49-72
    (6000, slice(73, 97)),     # volumes 73-96
]
N_SHELLS = len(SHELLS)


# ===================================================================
# Presets  (see README for paper references)
# ===================================================================

PRESETS = {
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
        "cfg":  dict(n_walkers=15_000, n_ensembles=5, n_steps=50_000,
                     L=250.0, buffer=60.0, grid_spacing=1.0),
    },
    "dense": {
        "kios": [2, 4, 6, 8, 10, 12, 15, 18, 22, 25, 30, 35, 45, 60, 80, 100],
        "rhos": [100_000, 150_000, 200_000, 300_000, 400_000, 500_000,
                 600_000, 800_000, 1_000_000, 1_200_000, 1_500_000,
                 2_000_000, 3_000_000],
        "Vs":   [0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5,
                 3.0, 4.0, 5.0, 7.0, 9.0],
        "cfg":  dict(n_walkers=25_000, n_ensembles=8, n_steps=50_000,
                     L=300.0, buffer=70.0, grid_spacing=0.8),
    },
}


# ===================================================================
# Parse triplet strings  "kio,rho,V"
# ===================================================================

def parse_triplet(s: str):
    """Parse 'kio,rho,V' string → (float, float, float)."""
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Triplet must be 'kio,rho,V', got '{s}'")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


# ===================================================================
# Parse input specs  "delta:path"
# ===================================================================

def parse_input(s: str):
    """Parse 'delta:path' string → (float, str)."""
    if ":" not in s:
        raise ValueError(f"Input must be 'delta:/path/to/file.nii.gz', got '{s}'")
    delta_str, path = s.split(":", 1)
    return (float(delta_str), path)


# ===================================================================
# Data loading (flexible number of deltas)
# ===================================================================

def load_dwi_and_average(input_specs, mask_path):
    """Load DWI NIfTIs for arbitrary delta values.

    Parameters
    ----------
    input_specs : list of (delta_ms, path) tuples
    mask_path : str

    Returns
    -------
    measured : ndarray (n_voxels, n_deltas * n_shells)
    fit_deltas : list of delta values [ms] in the measured data
    affine, mask_indices, shape
    """
    import nibabel as nib

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine
    shape = mask.shape
    mask_idx = np.where(mask)
    n_vox = len(mask_idx[0])

    print(f"  Mask voxels: {n_vox}")

    # Sort by delta
    input_specs = sorted(input_specs, key=lambda x: x[0])
    fit_deltas = [d for d, _ in input_specs]
    n_deltas = len(input_specs)

    measured = np.zeros((n_vox, n_deltas * N_SHELLS))

    for di, (delta_ms, dwi_path) in enumerate(input_specs):
        print(f"  Loading Δ={delta_ms:.0f}ms: {os.path.basename(dwi_path)} ...",
              end=" ", flush=True)
        img = nib.load(dwi_path)
        data = img.get_fdata()

        vox_data = data[mask_idx[0], mask_idx[1], mask_idx[2], :]

        S0 = vox_data[:, 0].copy()
        S0[S0 < 1e-10] = 1e-10

        for si, (b_val, vol_slice) in enumerate(SHELLS):
            shell_mean = np.mean(vox_data[:, vol_slice], axis=1)
            S_norm = np.clip(shell_mean / S0, 0.0, 1.0)
            measured[:, di * N_SHELLS + si] = S_norm

        print("done")

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
        description="MADI library builder & fitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LIBRARY EXAMPLES:
  # Full preset grid
  python fit_data.py --build-library --lib-preset default

  # Extend with extra rho values (merged into preset grid)
  python fit_data.py --build-library --append --custom-rhos 1500000 2000000

  # Explicit sub-grid: ONLY these values crossed (not merged with preset)
  python fit_data.py --build-library --append --explicit \\
      --grid-kios 12 25 --grid-rhos 1500000 2000000 --grid-Vs 0.5 1.0 2.0

  # Exact triplets
  python fit_data.py --build-library --append \\
      --triplets 12,1500000,0.5  25,2000000,1.0

FITTING EXAMPLES:
  # All four deltas
  python fit_data.py --fit \\
      --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \\
      --mask mask.nii.gz

  # Only two deltas
  python fit_data.py --fit \\
      --input 15:dwi15.nii.gz 25:dwi25.nii.gz \\
      --mask mask.nii.gz

  # Inspect library
  python fit_data.py --info
        """)

    # -- Actions --
    ap.add_argument("--build-library", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--info", action="store_true")

    # -- Library file --
    ap.add_argument("--library", default="madi_library.npz")
    ap.add_argument("--append", action="store_true",
                    help="Add to existing library, skip duplicates")

    # -- Preset grid --
    ap.add_argument("--lib-preset", default="default",
                    choices=list(PRESETS.keys()))

    # -- Custom additions (merged into preset grid) --
    ap.add_argument("--custom-kios", type=float, nargs="+")
    ap.add_argument("--custom-rhos", type=float, nargs="+")
    ap.add_argument("--custom-Vs",   type=float, nargs="+")

    # -- Explicit sub-grid (NOT merged with preset) --
    ap.add_argument("--explicit", action="store_true",
                    help="Use ONLY --grid-kios/rhos/Vs (ignore preset)")
    ap.add_argument("--grid-kios", type=float, nargs="+",
                    help="kio values for explicit sub-grid")
    ap.add_argument("--grid-rhos", type=float, nargs="+",
                    help="rho values for explicit sub-grid")
    ap.add_argument("--grid-Vs",   type=float, nargs="+",
                    help="V values for explicit sub-grid")

    # -- Exact triplets --
    ap.add_argument("--triplets", type=str, nargs="+",
                    help="Exact 'kio,rho,V' triplets (e.g. 12,1500000,0.5)")

    # -- Fitting --
    ap.add_argument("--input", type=str, nargs="+",
                    help="'delta:path' pairs (e.g. 15:dwi15.nii.gz)")
    # Legacy flags (still supported)
    ap.add_argument("--dwi15"); ap.add_argument("--dwi25")
    ap.add_argument("--dwi30"); ap.add_argument("--dwi40")
    ap.add_argument("--mask")
    ap.add_argument("--out", default="madi_output")

    args = ap.parse_args()

    if not any([args.build_library, args.fit, args.info]):
        ap.print_help()
        return

    # ================================================================
    #  INFO
    # ================================================================
    if args.info:
        if not os.path.exists(args.library):
            print(f"Library not found: {args.library}")
            return
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

        build_library(
            kios=kios, rhos=rhos, Vs=Vs, cfg=cfg,
            save_path=args.library, existing_library=existing)
        return

    # ================================================================
    #  FIT DATA
    # ================================================================
    if args.fit:
        # --- Parse inputs ---
        input_specs = []

        if args.input:
            # New flexible format:  --input 15:path 25:path ...
            for s in args.input:
                input_specs.append(parse_input(s))
        else:
            # Legacy format:  --dwi15 path --dwi25 path ...
            legacy_map = [(15.0, args.dwi15), (25.0, args.dwi25),
                          (30.0, args.dwi30), (40.0, args.dwi40)]
            for delta, path in legacy_map:
                if path is not None:
                    input_specs.append((delta, path))

        if not input_specs:
            print("ERROR: No DWI inputs specified.")
            print("  Use --input delta:path  or  --dwi15/--dwi25/... flags")
            return

        if args.mask is None:
            print("ERROR: --mask is required for --fit")
            return

        input_specs.sort(key=lambda x: x[0])
        fit_deltas = [d for d, _ in input_specs]

        print("=" * 60)
        print("MADI Fitting")
        print("=" * 60)
        print(f"  Δ values to fit: {fit_deltas} ms")
        print(f"  Signal vector length: {len(fit_deltas) * N_SHELLS}")

        os.makedirs(args.out, exist_ok=True)

        # Load library
        print(f"\nLoading library: {args.library}")
        if not os.path.exists(args.library):
            print(f"ERROR: Library not found: {args.library}")
            return
        lib = load_library(args.library)
        meta = load_library_meta(args.library)
        lib_deltas = meta['deltas']
        n_b = meta['n_b']
        print(f"  {len(lib)} entries, lib Δ = {lib_deltas} ms")
        library_summary(lib)

        # Validate deltas
        for d in fit_deltas:
            if not any(abs(d - ld) < 0.01 for ld in lib_deltas):
                print(f"ERROR: Δ = {d} ms not in library ({lib_deltas})")
                return

        # Load data
        print("\nLoading DWI data ...")
        measured, fit_deltas, affine, mask_idx, shape = \
            load_dwi_and_average(input_specs, args.mask)

        # Match
        print(f"\nMatching {measured.shape[0]} voxels "
              f"(using Δ = {fit_deltas} ms) ...")
        t0 = time.time()
        kio_map, rho_map, V_map, res_map = match_voxels_batch(
            measured, lib,
            fit_deltas=fit_deltas,
            lib_deltas=lib_deltas,
            n_b=n_b,
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

        # Save
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

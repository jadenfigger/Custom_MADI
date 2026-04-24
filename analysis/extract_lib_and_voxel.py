#!/usr/bin/env python3
"""
extract_lib_and_voxel.py — Pull one library entry's 16-point signal vector
and one voxel's 16-point raw signal out of the MADI pipeline.

Place in: MADI/analyze/
Uses the same package layout and shell convention as fit_data.py.
"""

import os
import sys
import numpy as np
import nibabel as nib

# Make the `madi` package importable from this sibling folder (same
# trick fit_data.py uses).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from madi.library import load_library, load_library_meta


# ======================================================================
# CONFIG — edit these paths and targets
# ======================================================================

LIBRARY_PATH = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/processing/madi_gpu/custom_madi/data/libraries/madi_dense.npz"
MASK_PATH    = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_15ms/mask_cropped.nii.gz"

# (delta_ms, path) — any subset of {15, 25, 30, 40} that matches your library.
DWI_INPUTS = [
    (15.0, "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_15ms/eddy_corrected.nii.gz"),
    (25.0, "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_25ms/eddy_corrected.nii.gz"),
    (30.0, "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_30ms/eddy_corrected.nii.gz"),
    (40.0, "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_40ms/eddy_corrected.nii.gz"),
]

# Target library entry (must match grid values exactly).
TARGET_KIO = 45.0          # s^-1
TARGET_RHO = 3_000_000.0     # cells / uL
TARGET_V   = 0.20           # pL

# Target voxel, image-space (i, j, k) indices.
TARGET_VOXEL = (135, 51, 3)

# Optional: dump the three vectors to disk alongside this script.
SAVE_NPZ = False
OUT_NAME = "extracted_vectors.npz"


# ======================================================================
# Protocol (mirrors fit_data.py)
# ======================================================================

SHELLS = [
    (1000, slice(1, 25)),
    (2500, slice(25, 49)),
    (4000, slice(49, 73)),
    (6000, slice(73, 97)),
]
N_SHELLS = len(SHELLS)


# ======================================================================
# Helpers
# ======================================================================

def find_entry(lib, kio, rho, V, rtol=1e-4):
    """Return the library entry matching (kio, rho, V), or raise with
    the nearest triplet for guidance."""
    for e in lib:
        if (abs(e.kio - kio) <= rtol * max(abs(kio), 1.0) and
            abs(e.rho - rho) <= rtol * max(abs(rho), 1.0) and
            abs(e.V   - V)   <= rtol * max(abs(V),   1.0)):
            return e

    # No match — find nearest in normalized coords for a helpful error.
    def norm_dist(e):
        return ( ((e.kio - kio) / max(kio, 1.0))**2 +
                 ((e.rho - rho) / max(rho, 1.0))**2 +
                 ((e.V   - V)   / max(V,   1.0))**2 )
    best = min(lib, key=norm_dist)
    raise ValueError(
        f"No library entry for (kio={kio}, rho={rho}, V={V}). "
        f"Nearest available: (kio={best.kio}, rho={best.rho}, V={best.V})."
    )


def extract_voxel_signal(dwi_inputs, mask_path, voxel):
    """Return (raw, normalized, deltas, s0_vals) for a single voxel.

    raw        : (n_deltas * N_SHELLS,)  shell means, un-normalized
    normalized : (n_deltas * N_SHELLS,)  shell means / S0  (same layout as library)
    deltas     : sorted list of delta values [ms]
    s0_vals    : (n_deltas,) S0 values for each delta
    """
    mask = nib.load(mask_path).get_fdata().astype(bool)
    i, j, k = voxel
    if not mask[i, j, k]:
        print(f"  WARNING: voxel {voxel} is OUTSIDE the mask.")

    dwi_inputs = sorted(dwi_inputs, key=lambda t: t[0])
    deltas = [d for d, _ in dwi_inputs]
    n_d = len(deltas)

    raw  = np.zeros(n_d * N_SHELLS)
    norm = np.zeros(n_d * N_SHELLS)
    s0_vals = np.zeros(n_d)

    for di, (delta_ms, path) in enumerate(dwi_inputs):
        data = nib.load(path).get_fdata()
        vox  = data[i, j, k, :].astype(np.float64)   # (97,)
        s0   = vox[0] if vox[0] > 1e-10 else 1e-10
        s0_vals[di] = s0
        
        for si, (_b, vol_slice) in enumerate(SHELLS):
            m = float(np.mean(vox[vol_slice]))
            raw[ di * N_SHELLS + si] = m
            norm[di * N_SHELLS + si] = m / s0

    return raw, norm, deltas, s0_vals


def print_vec(label, vec, deltas, s0_vals=None):
    if s0_vals is not None:
        b_hdr = "b0       | " + " | ".join(f"b{b:<5}" for b, _ in SHELLS)
        print(f"\n  {label}")
        print(f"    Δ(ms) | {b_hdr}")
        print(f"    ------+----------+-" + "-+-".join("-" * 6 for _ in SHELLS))
    else:
        b_hdr = " | ".join(f"b{b:<5}" for b, _ in SHELLS)
        print(f"\n  {label}")
        print(f"    Δ(ms) | {b_hdr}")
        print(f"    ------+-" + "-+-".join("-" * 6 for _ in SHELLS))
        
    for di, d in enumerate(deltas):
        vals = vec[di * N_SHELLS : (di + 1) * N_SHELLS]
        if s0_vals is not None:
            print(f"    {d:>5.0f} | {s0_vals[di]:>8.2f} | " + " | ".join(f"{v:>6.4f}" for v in vals))
        else:
            print(f"    {d:>5.0f} | " + " | ".join(f"{v:>6.4f}" for v in vals))


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("MADI — Library entry & voxel signal extractor")
    print("=" * 60)

    # -- Library side --
    print(f"\nLoading library: {LIBRARY_PATH}")
    lib  = load_library(LIBRARY_PATH)
    meta = load_library_meta(LIBRARY_PATH)
    lib_deltas = meta['deltas']
    n_b        = meta['n_b']
    print(f"  {len(lib)} entries | lib Δ = {lib_deltas} ms | n_b = {n_b}")

    entry = find_entry(lib, TARGET_KIO, TARGET_RHO, TARGET_V)
    fit_deltas = sorted(d for d, _ in DWI_INPUTS)

    for d in fit_deltas:
        if not any(abs(d - ld) < 0.01 for ld in lib_deltas):
            raise ValueError(f"Δ = {d} ms not in library ({lib_deltas})")

    # entry.vector is the library's flat signal, laid out as
    # signals[delta_idx, b_idx].ravel() over lib_deltas. Subset the
    # delta rows to match fit_deltas (same logic as match_voxels_batch).
    di_idx = [next(i for i, ld in enumerate(lib_deltas) if abs(d - ld) < 0.01)
              for d in fit_deltas]
    lib_vec = entry.vector.reshape(len(lib_deltas), n_b)[di_idx, :].ravel()

    print(f"\nLibrary entry: kio={entry.kio}, rho={entry.rho}, V={entry.V}")
    print_vec("library S/S0 (16 pts)", lib_vec, fit_deltas)

    # -- Voxel side --
    print(f"\nExtracting voxel {TARGET_VOXEL} ...")
    raw_vec, norm_vec, deltas, s0_vals = extract_voxel_signal(
        DWI_INPUTS, MASK_PATH, TARGET_VOXEL)

    print_vec("voxel raw (shell means, un-normalized)", raw_vec, deltas, s0_vals)
    print_vec("voxel S/S0 (normalized, matches library)", norm_vec, deltas)

    # -- Save --
    if SAVE_NPZ:
        out_path = os.path.join(_HERE, OUT_NAME)
        np.savez(
            out_path,
            lib_vec=lib_vec,
            voxel_raw=raw_vec,
            voxel_norm=norm_vec,
            voxel_s0=s0_vals,
            fit_deltas=np.array(fit_deltas),
            shells=np.array([b for b, _ in SHELLS]),
            kio=entry.kio, rho=entry.rho, V=entry.V,
            voxel=np.array(TARGET_VOXEL),
        )
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
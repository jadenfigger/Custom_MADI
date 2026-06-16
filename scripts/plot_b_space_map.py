#!/usr/bin/env python3
"""
plot_b_space_map.py — replicate Springer 2022 (MADI II) Figure 3.

Output: a PNG with
  • light-gray library decay curves (every Nth entry) as background
  • a black dashed pure-water reference  (S = exp(−b·D₀),  D₀ = 3 μm²/ms)
  • per-ROI region-averaged experimental data points (markers)
  • per-ROI best-matched library curves (colored lines)
  • region names annotated at the right edge of each curve

Usage
-----
    python scripts/plot_b_space_map.py \
        --library  data/libraries/madi_dense_human.npz \
        --dwi      /path/to/dwi_madi.nii.gz \
        --bval     /path/to/dwi_madi.bval \
        --seg      /path/to/segmentation.nii.gz \
        --labels   /path/to/labels.csv \
        --out      figures/b_space_map.png

Labels file (auto-detects format)
---------------------------------
1) Simple CSV:
       label,name,color
       1,Cortical GM,#2ca02c
       2,WM,#1f77b4

2) ITK-SNAP "Save Label Descriptions" export:
       # ITK-SnAP Label Description File
       # IDX -R- -G- -B- -A-- VIS MSH LABEL
           1 255   0   0    1  1  1    "Cortical GM"
           2  31 119 180    1  1  1    "WM"
"""

import argparse
import os
import shlex
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Make `madi` importable when this script lives in scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from madi.library import load_library, load_library_meta


# --------------------------------------------------------------------------
# Labels file parsing  (supports CSV + ITK-SNAP)
# --------------------------------------------------------------------------
def parse_labels_file(path):
    """Return {label_id: (name, color_or_None)}."""
    labels = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # CSV path
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if parts[0].lower() in ("label", "label_id", "id"):
                    continue
                try:
                    lid = int(parts[0])
                except ValueError:
                    continue
                name = parts[1].strip("'\"")
                color = parts[2] if len(parts) >= 3 and parts[2] else None
                labels[lid] = (name, color)
                continue

            # ITK-SNAP path:  IDX R G B A VIS MSH "LABEL"
            try:
                parts = shlex.split(line)
            except ValueError:
                continue
            if len(parts) < 8:
                continue
            try:
                lid = int(parts[0])
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if lid == 0:
                continue  # ITK-SNAP "Clear Label"
            name = parts[7]
            color = f"#{r:02x}{g:02x}{b:02x}"
            labels[lid] = (name, color)
    return labels


# --------------------------------------------------------------------------
# Powder-averaging and S/S0 normalization
# --------------------------------------------------------------------------
def powder_average(data, bvals, lib_b, b_tol=50.0, eps=1e-8):
    """Geometric-mean powder average of S/S0 per library shell.

    Returns
    -------
    S0          : (X,Y,Z) array, mean of b=0 volumes
    S_norm_4d   : (n_b, X, Y, Z) array of S(b)/S0, powder-averaged
    shells_info : list of (b_target, n_volumes)
    """
    b0_mask = bvals < b_tol
    if not b0_mask.any():
        raise ValueError("No b=0 volumes found in the data.")
    S0 = data[..., b0_mask].mean(axis=-1)

    n_b = len(lib_b)
    out = np.zeros((n_b,) + data.shape[:3], dtype=np.float32)
    shells_info = []
    for i, b in enumerate(lib_b):
        idx = np.where(np.abs(bvals - b) < b_tol)[0]
        shells_info.append((float(b), len(idx)))
        if len(idx) == 0:
            continue
        S = data[..., idx]                                  # (X,Y,Z,n_dir)
        ratio = np.where(S0[..., None] > eps, S / S0[..., None], 0)
        log_ratio = np.where(ratio > eps,
                             np.log(np.clip(ratio, eps, None)), 0)
        out[i] = np.exp(log_ratio.mean(axis=-1))
    return S0, out, shells_info


# --------------------------------------------------------------------------
# Library matching
# --------------------------------------------------------------------------
def find_best_match(measured, lib, n_b, delta_idx):
    """Brute-force least-squares match. Returns (entry, residual)."""
    best_res = np.inf
    best = None
    for entry in lib:
        sig = entry.vector.reshape(-1, n_b)[delta_idx]
        res = float(np.sum((measured - sig) ** 2))
        if res < best_res:
            best_res = res
            best = entry
    return best, best_res


# --------------------------------------------------------------------------
# End-of-curve label spreader (avoid overlapping text)
# --------------------------------------------------------------------------
def spread_labels(items, log_y_min_gap=0.07, y_floor=0.011):
    """Adjust label y-positions so no two are closer than log_y_min_gap
    in log10 space. items: list of (x, y, name, color)."""
    if not items:
        return items
    items = sorted(items, key=lambda t: t[1], reverse=True)  # top → bottom
    out = []
    prev_log_y = None
    for x, y, name, color in items:
        ly = np.log10(max(y, y_floor))
        if prev_log_y is not None and (prev_log_y - ly) < log_y_min_gap:
            ly = prev_log_y - log_y_min_gap
        y_adj = max(10 ** ly, y_floor)
        out.append((x, y_adj, name, color))
        prev_log_y = ly
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
DEFAULT_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                  "#e377c2", "#8c564b", "#17becf", "#bcbd22", "#7f7f7f"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--dwi",     required=True,
                    help="DWI NIfTI (post-eddy, post-MADI-filter)")
    ap.add_argument("--bval",    required=True)
    ap.add_argument("--seg",     required=True,
                    help="Segmentation NIfTI (integer labels per voxel)")
    ap.add_argument("--labels",  required=True,
                    help="Labels file: CSV or ITK-SNAP description")
    ap.add_argument("--out",     required=True, help="Output PNG path")
    ap.add_argument("--delta",   type=float, default=None,
                    help="Δ in ms (default: first Δ in the library)")
    ap.add_argument("--n-gray",  type=int, default=200,
                    help="Library entries to draw as gray background")
    ap.add_argument("--min-voxels", type=int, default=10,
                    help="Skip ROIs with fewer voxels than this")
    ap.add_argument("--ylim-min",  type=float, default=0.01)
    ap.add_argument("--figsize",   nargs=2, type=float, default=[9.0, 7.0])
    ap.add_argument("--dpi",       type=int, default=200)
    ap.add_argument("--title",     default="")
    args = ap.parse_args()

    # ----- library -----
    print(f"Loading library: {args.library}")
    lib = load_library(args.library)
    meta = load_library_meta(args.library)
    lib_b = np.asarray(meta["b_values"], dtype=float)
    lib_deltas = list(meta["deltas"])
    n_b = int(meta["n_b"])
    print(f"  {len(lib)} entries; Δ={lib_deltas} ms; b={list(lib_b)} s/mm²")

    if args.delta is None:
        delta_idx, Delta_plot = 0, lib_deltas[0]
    else:
        diffs = [abs(args.delta - d) for d in lib_deltas]
        delta_idx = int(np.argmin(diffs))
        Delta_plot = lib_deltas[delta_idx]
        if min(diffs) > 0.5:
            print(f"  WARN: --delta {args.delta} ≠ library; using {Delta_plot}")

    # ----- data -----
    print("Loading DWI / bvals / seg ...")
    img = nib.load(args.dwi)
    data = img.get_fdata()
    bvals = np.loadtxt(args.bval).ravel()
    seg_raw = nib.load(args.seg).get_fdata().astype(int)
    if seg_raw.ndim == 4:
        # Seg was saved with the source DWI's time dimension. Collapse to 3D
        # by taking time 0; warn if any later slice differs from it.
        n_diff = sum(
            not np.array_equal(seg_raw[..., 0], seg_raw[..., t])
            for t in range(1, seg_raw.shape[-1])
        )
        if n_diff > 0:
            print(f"  WARN: seg is 4D and {n_diff}/{seg_raw.shape[-1] - 1} "
                  f"later time slices differ from time 0. Using time 0.")
        else:
            print(f"  Note: seg is 4D ({seg_raw.shape[-1]} identical "
                  f"time slices). Reducing to 3D.")
        seg = seg_raw[..., 0]
    else:
        seg = seg_raw
    if seg.shape != data.shape[:3]:
        raise ValueError(f"seg shape {seg.shape} != DWI {data.shape[:3]}")
    if data.shape[-1] != len(bvals):
        raise ValueError(f"DWI has {data.shape[-1]} vols, bvals has {len(bvals)}")

    S0, S_norm_4d, shells_info = powder_average(data, bvals, lib_b)
    for b, n in shells_info:
        print(f"  shell b={b:.0f}: {n} volumes")

    # ----- ROI loop -----
    labels_map = parse_labels_file(args.labels)
    print(f"\n{len(labels_map)} ROIs in labels file\n")

    region_results = []
    for ci, (lid, (name, color)) in enumerate(sorted(labels_map.items())):
        roi = (seg == lid) & (S0 > 0)
        n = int(roi.sum())
        if n < args.min_voxels:
            print(f"  skipping label {lid} '{name}' (only {n} voxels)")
            continue
        sig = np.array([S_norm_4d[i][roi].mean() for i in range(n_b)])
        entry, res = find_best_match(sig, lib, n_b, delta_idx)
        if color is None:
            color = DEFAULT_COLORS[ci % len(DEFAULT_COLORS)]
        region_results.append(dict(
            label=lid, name=name, color=color, n=n,
            measured=sig,
            matched=entry.vector.reshape(-1, n_b)[delta_idx].copy(),
            kio=entry.kio, rho=entry.rho, V=entry.V, residual=res,
        ))
        print(f"  label {lid:>3} '{name}' (n={n:>5}): "
              f"kio={entry.kio:>5g} s⁻¹, ρ={entry.rho/1e3:>6.0f}k, "
              f"V={entry.V:>5.2f} pL, res={res:.4f}")

    # ----- plot -----
    fig, ax = plt.subplots(figsize=args.figsize)

    b_plot = np.concatenate(([0.0], lib_b))

    # gray background
    n_gray = min(args.n_gray, len(lib))
    idx_gray = np.linspace(0, len(lib) - 1, n_gray, dtype=int)
    for i in idx_gray:
        sig_e = lib[i].vector.reshape(-1, n_b)[delta_idx]
        S_plot = np.concatenate(([1.0], sig_e))
        ax.semilogy(b_plot, np.clip(S_plot, args.ylim_min * 0.5, None),
                    color="0.78", lw=0.4, alpha=0.55, zorder=1)

    # pure-water reference.  b is in s/mm², D0 in μm²/ms.
    # Dimensionless exponent:  b[s/mm²] × D0[μm²/ms] / 1e3
    b_dense = np.linspace(0, lib_b.max(), 200)
    D0 = 3.0  # μm²/ms
    S_water = np.exp(-b_dense * D0 / 1e3)
    ax.semilogy(b_dense, np.clip(S_water, args.ylim_min * 0.5, None),
                "k--", lw=1.0, alpha=0.8, zorder=2,
                label=f"pure water  (D₀ = {D0} μm²/ms)")

    # per-region curves + markers
    label_items = []
    for r in region_results:
        S_match = np.concatenate(([1.0], r["matched"]))
        S_meas  = np.concatenate(([1.0], r["measured"]))
        ax.semilogy(b_plot, np.clip(S_match, args.ylim_min * 0.5, None),
                    color=r["color"], lw=2.0, alpha=0.95, zorder=3)
        ax.semilogy(b_plot, np.clip(S_meas, args.ylim_min * 0.5, None),
                    "o", color=r["color"], ms=6,
                    mec="black", mew=0.4, zorder=4)
        label_items.append((lib_b[-1], r["matched"][-1], r["name"], r["color"]))

    # spread + draw end-of-curve labels
    x_text = lib_b[-1] * 1.02
    for x, y, name, color in spread_labels(label_items,
                                            y_floor=args.ylim_min * 1.1):
        ax.text(x_text, y, name, color=color, fontsize=10,
                ha="left", va="center", weight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1.5))

    ax.set_xlabel("b  (s/mm²)", fontsize=12)
    ax.set_ylabel("S / S₀", fontsize=12)
    ax.set_xlim(0, lib_b[-1] * 1.30)
    ax.set_ylim(args.ylim_min, 1.1)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)

    title_parts = []
    if args.title:
        title_parts.append(args.title)
    title_parts.append(
        f"Δ = {Delta_plot:g} ms;  library: "
        f"{os.path.basename(args.library)} ({len(lib)} entries)")
    ax.set_title("\n".join(title_parts), fontsize=11)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_identifiability.py — Cramer-Rao / Fisher Information analysis of
MADI acquisition identifiability
====================================================================

Quantifies how well a given (Delta, b) acquisition can identify the MADI
parameters (kio, rho, V), using the library itself (finite differences
across neighbouring grid points -- see madi/identifiability.py for the
method and its caveats). The headline diagnostic is the rho-V correlation:
how degenerate rho and V are at the acquisition being analyzed.

See docs/identifiability.md for the full explanation, and read the module
docstring of madi/identifiability.py for the important caveat that CRLB
here is a RELATIVE tool for comparing acquisitions, not an absolute
variance prediction (the MADI matcher is discrete/biased, unlike the
continuous unbiased estimator CRLB assumes).

EXAMPLES
--------
Single acquisition (the library's own current acquisition):
    python scripts/analyze_identifiability.py \\
        --library data/libraries/madi_dense_human.npz \\
        --sigma-m 0.02 --acquisition current \\
        --out data/outputs/identifiability_current/

Compare current vs. a b<=1500 truncation:
    python scripts/analyze_identifiability.py \\
        --library data/libraries/madi_dense_human.npz \\
        --sigma-m 0.02 \\
        --compare current "subset_b:500,1000,1500" \\
        --out data/outputs/identifiability_compare/
"""

import argparse
import json
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from madi.config import D0_MOUSE
from madi.library import load_library, load_library_meta
from madi.identifiability import (
    compute_finite_diff_derivatives,
    analyze_library,
    gaussian_crlb_sanity_check,
    PARAM_NAMES,
)


# ---------------------------------------------------------------------------
# Acquisition-spec parsing
# ---------------------------------------------------------------------------

def _parse_b_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def build_fit_pairs(acq_type, lib_deltas, lib_b_values, b_subset=None, pairs_str=None):
    """Build the (Delta, b) column list for one acquisition.

    acq_type : 'current' | 'subset_b' | 'custom'
      current  -> every (Delta, b) the library has (all deltas x all b).
      subset_b -> every library Delta, but only b in b_subset.
      custom   -> explicit 'Delta:b,Delta:b,...' pairs from pairs_str.
    """
    if acq_type == "current":
        return [(d, b) for d in lib_deltas for b in lib_b_values]

    if acq_type == "subset_b":
        if not b_subset:
            raise ValueError("--acquisition subset_b requires --b-subset")
        return [(d, b) for d in lib_deltas for b in lib_b_values
                if any(abs(b - k) < 1e-6 for k in b_subset)]

    if acq_type == "custom":
        if not pairs_str:
            raise ValueError("--acquisition custom requires --pairs 'D:b,D:b,...'")
        pairs = []
        for tok in pairs_str.split(","):
            d_s, b_s = tok.split(":")
            pairs.append((float(d_s), float(b_s)))
        return pairs

    raise ValueError(f"Unknown acquisition type '{acq_type}'")


def parse_compare_spec(spec, lib_deltas, lib_b_values):
    """Parse one '--compare' token into (label, fit_pairs).

    Accepted forms:
        current
        subset_b:500,1000,1500
        custom:50:500,50:1000,50:1500
        mylabel=subset_b:500,1000,1500        (explicit label)
    """
    if "=" in spec:
        label, rest = spec.split("=", 1)
    else:
        label, rest = spec, spec

    if ":" not in rest:
        acq_type, params = rest, None
    else:
        acq_type, params = rest.split(":", 1)

    if acq_type == "current":
        fit_pairs = build_fit_pairs("current", lib_deltas, lib_b_values)
    elif acq_type == "subset_b":
        fit_pairs = build_fit_pairs("subset_b", lib_deltas, lib_b_values,
                                     b_subset=_parse_b_list(params))
    elif acq_type == "custom":
        fit_pairs = build_fit_pairs("custom", lib_deltas, lib_b_values,
                                     pairs_str=params)
    else:
        raise ValueError(f"Unrecognized --compare spec '{spec}'")

    return label, fit_pairs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_single_acquisition_plots(rows, out_dir, label):
    import matplotlib.pyplot as plt

    corr = np.array([r["rhoV_correlation"] for r in rows])
    V = np.array([r["V"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    crlb_rho = np.array([r["CRLB_rho"] for r in rows])
    crlb_V = np.array([r["CRLB_V"] for r in rows])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(corr, bins=40, range=(-1, 1), color="#4C72B0")
    ax.axvline(0.9, color="red", ls="--", lw=1, label="|corr|=0.9")
    ax.axvline(-0.9, color="red", ls="--", lw=1)
    ax.set_xlabel("rho-V correlation  corr(dS/drho, dS/dV)")
    ax.set_ylabel("# library entries")
    ax.set_title(f"rho-V degeneracy — {label}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "rhoV_correlation_hist.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, x, xlabel in zip(axes, (V, rho), ("V [pL]", "rho [cells/uL]")):
        order = np.argsort(x)
        ax.semilogy(x[order], np.clip(crlb_rho[order], 1e-30, None),
                    "o", ms=3, alpha=0.5, label="CRLB(rho)")
        ax.semilogy(x[order], np.clip(crlb_V[order], 1e-30, None),
                    "s", ms=3, alpha=0.5, label="CRLB(V)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CRLB (min variance, log scale)")
        ax.legend()
    fig.suptitle(f"CRLB(rho), CRLB(V) across parameter space — {label}")
    fig.savefig(os.path.join(out_dir, "crlb_vs_paramspace.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_comparison_plot(all_summaries, out_dir):
    import matplotlib.pyplot as plt

    labels = list(all_summaries.keys())
    med_rho = [all_summaries[l]["CRLB_rho"]["median"] for l in labels]
    med_V = [all_summaries[l]["CRLB_V"]["median"] for l in labels]
    deg_frac = [all_summaries[l]["degenerate_fraction"] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(4 * len(labels) * 0.8 + 3, 4.5))
    x = np.arange(len(labels))

    axes[0].bar(x, med_rho, color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median CRLB(rho)")

    axes[1].bar(x, med_V, color="#DD8452")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median CRLB(V)")

    axes[2].bar(x, deg_frac, color="#55A868")
    axes[2].set_ylabel("degenerate fraction  (|rho-V corr| > threshold)")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")

    fig.suptitle("Acquisition comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "acquisition_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional: map CRLBs onto fitted parameter maps (voxel space)
# ---------------------------------------------------------------------------

def map_crlb_to_voxels(rows, library, kio_map_path, rho_map_path, V_map_path,
                        mask_path, out_dir):
    import nibabel as nib
    from scipy.spatial import cKDTree

    kio_img = nib.load(kio_map_path)
    kio_vol = kio_img.get_fdata()
    rho_vol = nib.load(rho_map_path).get_fdata()
    V_vol = nib.load(V_map_path).get_fdata()
    affine = kio_img.affine

    if mask_path is not None:
        mask = nib.load(mask_path).get_fdata().astype(bool)
    else:
        mask = np.ones(kio_vol.shape, dtype=bool)

    # Standardize each axis by the library's own spread so that Euclidean
    # nearest-neighbour in (kio, rho, V) is meaningful despite wildly
    # different native scales/units.
    lib_kio = np.array([e.kio for e in library])
    lib_rho = np.array([e.rho for e in library])
    lib_V = np.array([e.V for e in library])

    scale_kio = np.std(lib_kio) or 1.0
    scale_rho = np.std(lib_rho) or 1.0
    scale_V = np.std(lib_V) or 1.0

    lib_pts = np.stack([lib_kio / scale_kio, lib_rho / scale_rho,
                         lib_V / scale_V], axis=1)
    tree = cKDTree(lib_pts)

    # Build a lookup array: library index -> row index in `rows` (rows may
    # be a strict subset of the library, since isolated grid-edge entries
    # with no derivative are skipped in analyze_library).
    idx_to_row = {}
    lib_key_to_idx = {(round(e.kio, 4), round(e.rho, 1), round(e.V, 6)): i
                       for i, e in enumerate(library)}
    for r in rows:
        key = (round(r["kio"], 4), round(r["rho"], 1), round(r["V"], 6))
        if key in lib_key_to_idx:
            idx_to_row[lib_key_to_idx[key]] = r

    vox_idx = np.where(mask)
    n_vox = len(vox_idx[0])
    q = np.stack([
        kio_vol[vox_idx] / scale_kio,
        rho_vol[vox_idx] / scale_rho,
        V_vol[vox_idx] / scale_V,
    ], axis=1)
    _, nn = tree.query(q)

    for field, fname in (("CRLB_kio", "CRLB_kio.nii.gz"),
                         ("CRLB_rho", "CRLB_rho.nii.gz"),
                         ("CRLB_V", "CRLB_V.nii.gz"),
                         ("rhoV_correlation", "rhoV_correlation.nii.gz")):
        out_vol = np.full(kio_vol.shape, np.nan, dtype=np.float32)
        vals = np.full(n_vox, np.nan, dtype=np.float32)
        for v in range(n_vox):
            row = idx_to_row.get(int(nn[v]))
            if row is not None:
                vals[v] = row[field]
        out_vol[vox_idx] = vals
        path = os.path.join(out_dir, fname)
        nib.save(nib.Nifti1Image(out_vol, affine), path)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fisher-Information / CRLB identifiability analysis "
                    "for a MADI library + acquisition.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--sigma-m", type=float, default=0.02,
                    help="Noise std on the normalized S/S0 signal "
                         "(default 0.02, matching the Bayesian fitter's "
                         "placeholder default). Ignored if "
                         "--sigma-m-pershell is given.")
    ap.add_argument("--sigma-m-pershell", type=str, default=None,
                    help="Comma list of per-(Delta,b)-column noise stds, "
                         "same length and order as the resolved acquisition "
                         "columns. Only usable for a single (non --compare) "
                         "acquisition.")

    ap.add_argument("--acquisition", choices=["current", "subset_b", "custom"],
                    default="current")
    ap.add_argument("--b-subset", type=str, default=None,
                    help="Comma list of b-values [s/mm^2] for "
                         "--acquisition subset_b (e.g. 500,1000,1500).")
    ap.add_argument("--pairs", type=str, default=None,
                    help="Explicit 'Delta:b,Delta:b,...' pairs for "
                         "--acquisition custom.")

    ap.add_argument("--compare", type=str, nargs="+", default=None,
                    help="Run several acquisitions and additionally produce "
                         "a comparison plot/summary. Each token is "
                         "'[label=]type[:params]', type in "
                         "{current, subset_b, custom}, e.g. "
                         "'current' 'trunc=subset_b:500,1000,1500'. "
                         "Overrides --acquisition/--b-subset/--pairs; uses "
                         "the scalar --sigma-m for every acquisition "
                         "(--sigma-m-pershell is not supported in this mode).")

    ap.add_argument("--degenerate-threshold", type=float, default=0.9,
                    help="|rho-V correlation| above this counts as "
                         "'degenerate' for the summary fraction (default 0.9).")

    ap.add_argument("--kio-map", type=str, default=None,
                    help="Optional fitted kio map (NIfTI) to project CRLBs "
                         "onto, via nearest-library-entry lookup.")
    ap.add_argument("--rho-map", type=str, default=None)
    ap.add_argument("--V-map", type=str, default=None)
    ap.add_argument("--seg", type=str, default=None,
                    help="Optional mask/segmentation NIfTI restricting which "
                         "voxels get a projected CRLB (default: all voxels).")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 70)
    print("MADI identifiability analysis (Fisher Information / CRLB)")
    print("=" * 70)
    print(f"  Library: {args.library}")
    if not os.path.exists(args.library):
        print(f"ERROR: library not found: {args.library}")
        return

    library = load_library(args.library)
    meta = load_library_meta(args.library)
    lib_deltas = meta["deltas"]
    lib_b_values = meta["b_values"]
    n_b = meta["n_b"]
    if lib_b_values is None:
        print("ERROR: library has no stored b-values metadata; rebuild it "
              "with the current madi/library.py before running this "
              "analysis.")
        return

    print(f"  {len(library)} entries, deltas={lib_deltas}, "
          f"b_values={lib_b_values}")

    # Finite-difference derivatives depend only on the library grid, not on
    # the acquisition -- compute once and reuse across every acquisition.
    print("\nComputing finite-difference derivatives across the "
          "(kio, rho, V) grid...")
    derivatives = compute_finite_diff_derivatives(library)
    print(f"  Derivatives computed for {len(derivatives)} entries "
          f"(some may be grid-edge/isolated -- see per-acquisition report).")

    # ------------------------------------------------------------------
    # Resolve which acquisition(s) to run
    # ------------------------------------------------------------------
    if args.compare:
        if args.sigma_m_pershell is not None:
            print("  Note: --sigma-m-pershell is ignored in --compare mode; "
                  f"using scalar --sigma-m={args.sigma_m} for all "
                  "acquisitions.")
        specs = [parse_compare_spec(s, lib_deltas, lib_b_values)
                 for s in args.compare]
    else:
        fit_pairs = build_fit_pairs(
            args.acquisition, lib_deltas, lib_b_values,
            b_subset=(_parse_b_list(args.b_subset) if args.b_subset else None),
            pairs_str=args.pairs,
        )
        specs = [(args.acquisition, fit_pairs)]

    # ------------------------------------------------------------------
    # Run each acquisition
    # ------------------------------------------------------------------
    all_summaries = {}
    all_rows = {}

    for label, fit_pairs in specs:
        print("\n" + "-" * 70)
        print(f"Acquisition: {label}  ({len(fit_pairs)} (Delta,b) columns)")
        print("-" * 70)
        for d, b in fit_pairs:
            print(f"    Delta={d:g} ms, b={b:g} s/mm^2")

        if args.sigma_m_pershell is not None and len(specs) == 1:
            sigma_m = np.array(_parse_b_list(args.sigma_m_pershell))
            if sigma_m.size != len(fit_pairs):
                print(f"ERROR: --sigma-m-pershell has {sigma_m.size} values "
                      f"but this acquisition has {len(fit_pairs)} columns.")
                return
            print(f"  sigma_m: per-shell {sigma_m.tolist()}")
        else:
            sigma_m = args.sigma_m
            print(f"  sigma_m: {sigma_m} (scalar, applied to every "
                  f"(Delta,b) column)")

        result = analyze_library(
            library, lib_deltas, lib_b_values, n_b, fit_pairs, sigma_m,
            degenerate_corr_threshold=args.degenerate_threshold,
            derivatives=derivatives,
        )

        acq_dir = os.path.join(args.out, label if len(specs) > 1 else ".")
        os.makedirs(acq_dir, exist_ok=True)

        # CSV table
        import csv
        csv_path = os.path.join(acq_dir, "identifiability_table.csv")
        fieldnames = list(result.rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result.rows)
        print(f"  Saved {csv_path}  ({len(result.rows)} rows)")

        json_path = os.path.join(acq_dir, "identifiability_summary.json")
        with open(json_path, "w") as f:
            json.dump(result.summary, f, indent=2)
        print(f"  Saved {json_path}")

        print(f"\n  Summary — {label}:")
        print(f"    entries analyzed:       {result.summary['n_entries_analyzed']}"
              f" / {result.summary['n_entries_total']} "
              f"({result.summary['n_skipped_isolated']} isolated/skipped)")
        print(f"    non-PSD FIMs (FD warn): {result.summary['n_nonpsd']}")
        print(f"    median CRLB(kio):       {result.summary['CRLB_kio']['median']:.4g}"
              f"  (IQR {result.summary['CRLB_kio']['q25']:.4g}"
              f"-{result.summary['CRLB_kio']['q75']:.4g})")
        print(f"    median CRLB(rho):       {result.summary['CRLB_rho']['median']:.4g}"
              f"  (IQR {result.summary['CRLB_rho']['q25']:.4g}"
              f"-{result.summary['CRLB_rho']['q75']:.4g})")
        print(f"    median CRLB(V):         {result.summary['CRLB_V']['median']:.4g}"
              f"  (IQR {result.summary['CRLB_V']['q25']:.4g}"
              f"-{result.summary['CRLB_V']['q75']:.4g})")
        print(f"    median trace(F^-1):     {result.summary['trace_Finv']['median']:.4g}")
        print(f"    median rho-V corr:      {result.summary['rhoV_correlation_median']:.3f}")
        print(f"    degenerate fraction     "
              f"(|corr|>{args.degenerate_threshold}): "
              f"{result.summary['degenerate_fraction']*100:.1f}%")
        sc = result.summary["derivative_stencil_counts"]
        print(f"    derivative stencils:    "
              f"kio central/edge={sc['kio_central']}/{sc['kio_edge']}, "
              f"rho central/edge={sc['rho_central']}/{sc['rho_edge']}, "
              f"V central/edge={sc['V_central']}/{sc['V_edge']}")

        make_single_acquisition_plots(result.rows, acq_dir, label)
        print(f"  Saved plots to {acq_dir}")

        all_summaries[label] = result.summary
        all_rows[label] = result.rows

    # ------------------------------------------------------------------
    # Cross-acquisition comparison
    # ------------------------------------------------------------------
    if len(specs) > 1:
        print("\n" + "=" * 70)
        print("Acquisition comparison")
        print("=" * 70)
        make_comparison_plot(all_summaries, args.out)
        with open(os.path.join(args.out, "comparison_summary.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)
        for label, s in all_summaries.items():
            print(f"  {label:20s}  median CRLB(rho)={s['CRLB_rho']['median']:.4g}"
                  f"  CRLB(V)={s['CRLB_V']['median']:.4g}"
                  f"  degenerate_frac={s['degenerate_fraction']*100:.1f}%")

        # ---- Monotonicity sanity check ----------------------------------
        # Fewer (Delta,b) columns should never IMPROVE (decrease) the
        # median CRLBs. Compare the acquisition with the most columns
        # against the one with the fewest as a simple check.
        by_ncols = sorted(all_summaries.items(),
                          key=lambda kv: len(kv[1]["fit_pairs"]))
        fewest_label, fewest_summary = by_ncols[0]
        most_label, most_summary = by_ncols[-1]
        if fewest_label != most_label:
            ok_rho = (fewest_summary["CRLB_rho"]["median"]
                      >= most_summary["CRLB_rho"]["median"])
            ok_V = (fewest_summary["CRLB_V"]["median"]
                    >= most_summary["CRLB_V"]["median"])
            status = "PASS" if (ok_rho and ok_V) else "FAIL"
            print(f"\n  Monotonicity check ({status}): fewer (Delta,b) "
                  f"columns should not lower the CRLBs.")
            print(f"    fewest columns = '{fewest_label}' "
                  f"({len(fewest_summary['fit_pairs'])} cols): "
                  f"CRLB(rho)={fewest_summary['CRLB_rho']['median']:.4g}, "
                  f"CRLB(V)={fewest_summary['CRLB_V']['median']:.4g}")
            print(f"    most columns   = '{most_label}' "
                  f"({len(most_summary['fit_pairs'])} cols): "
                  f"CRLB(rho)={most_summary['CRLB_rho']['median']:.4g}, "
                  f"CRLB(V)={most_summary['CRLB_V']['median']:.4g}")
            if status == "FAIL":
                print("    WARNING: truncating the acquisition lowered a "
                      "CRLB -- check finite-difference/edge effects before "
                      "trusting this comparison.")

    # ------------------------------------------------------------------
    # Gaussian mono-exponential sanity check (fully analytic, independent
    # of the MADI library — cross-checks the optimal-b ~ 1/ADC trend).
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Sanity check: Gaussian mono-exponential CRLB(ADC) vs b")
    print("=" * 70)
    ref_b_values = np.asarray(lib_b_values, dtype=float)
    ref_sigma = args.sigma_m
    adc = D0_MOUSE  # representative tissue ADC [um^2/ms]
    gc = gaussian_crlb_sanity_check(ref_b_values, adc, ref_sigma)
    print(f"  Reference ADC = {adc:g} um^2/ms, sigma_m = {ref_sigma:g}")
    for b, c in zip(gc["b_values_s_mm2"], gc["crlb_adc"]):
        print(f"    b={b:8.1f} s/mm^2   CRLB(ADC)={c:.4g}")
    print(f"  Best b on this grid:     {gc['best_b_on_grid_s_mm2']:.1f} s/mm^2")
    print(f"  Theoretical optimal b:   {gc['theoretical_optimal_b_s_mm2']:.1f} "
          f"s/mm^2 (= 1/ADC)")
    print("  (This is an independent analytic check, not derived from the "
          "MADI library -- it only validates that this module's CRLB "
          "machinery reproduces the textbook optimal-b~1/ADC trend.)")

    # ------------------------------------------------------------------
    # Optional: project onto fitted parameter maps
    # ------------------------------------------------------------------
    if args.kio_map and args.rho_map and args.V_map:
        print("\n" + "=" * 70)
        print("Projecting CRLBs onto fitted parameter maps")
        print("=" * 70)
        primary_label = specs[0][0]
        map_crlb_to_voxels(
            all_rows[primary_label], library,
            args.kio_map, args.rho_map, args.V_map, args.seg, args.out,
        )
    elif any([args.kio_map, args.rho_map, args.V_map]):
        print("\n  --kio-map/--rho-map/--V-map must ALL be given together "
              "to project CRLBs onto voxel space; skipping (some were "
              "omitted).")

    print("\nDone.")


if __name__ == "__main__":
    main()

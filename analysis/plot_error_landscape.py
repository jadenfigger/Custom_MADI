#!/usr/bin/env python3
"""
plot_error_landscape.py — Visualise the MADI fitting error landscape
====================================================================

For a manually specified normalized decay (a set of (Δ, b, S/S0) points),
this script evaluates the *standard* MADI matcher loss
(``match_voxels_batch``, fixed-S0 / no fit-S0) at every DISCRETE library
grid point, holding ONE of (kio, rho, V) fixed and sweeping the other two
between user-chosen (or library min/max) bounds.  It produces a 2-D
heatmap of the loss for each free-variable pair, both WITHOUT and WITH the
Rician-correction option enabled, plus their difference.

What "loss" means here
----------------------
It is byte-for-byte the quantity ``match_voxels_batch`` minimises:

    optional log-transform:   x -> log(clip(x, s_floor, 1.0))     (if LOG_SPACE)
    loss(entry) = || measured_subset - library_vector_subset ||^2   (squared L2)

evaluated per library entry, *before* the argmin.  Only entries with
vi = (rho/1e9)*(V*1e3) in [VI_MIN, VI_MAX] (and rho <= RHO_MAX) are
candidates during fitting; cells outside that window — or grid triplets
the library never simulated — are masked (shown blank / NaN), exactly as
the fitter would never see them.

Rician option
-------------
``--rician-correct`` in fit_data.py applies the second-moment correction
    A = sqrt( max(M^2 - 2*sigma^2, 0) )
to every raw magnitude volume (signal *and* b=0) before normalising.
On a NORMALIZED decay this is a function of the b=0 SNR alone:

    norm_corrected = sqrt(max(norm^2 - 2/SNR0^2, 0)) / sqrt(max(1 - 2/SNR0^2, 0))

so the Rician landscape is obtained by transforming the measured vector
with the single number SNR0 (= S0/sigma at b=0) and re-running the loss.
The library vectors are noiseless simulations and are NOT modified.

The library and output directory are the only command-line arguments;
everything describing *what* to plot lives in the CONFIG block below.

Usage
-----
    python plot_error_landscape.py \
        --library data/libraries/madi_dense.npz \
        --out     error_landscape_out

    python plot_error_landscape.py \
        --library /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/data/libraries/madi_dense_human.npz \
        --out     /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/figures/error_landscape_out
"""

import argparse
import os
import numpy as np


# ===================================================================
# CONFIG  —  edit this block to describe the landscape you want
# ===================================================================

# -- The normalized decay to fit ------------------------------------
# Each row: (Delta_ms, b_s_mm2, normalized_signal = S(b)/S0).
# These (Δ, b) pairs must all exist in the library being loaded.
# MEASURED = [
#     # (Δ [ms],  b [s/mm²],  S/S0)  — Grey Matter, read from b_space_map_125.png
#     (50.0,  500.0, 0.680),
#     (50.0, 1000.0, 0.460),
#     (50.0, 1500.0, 0.350),
#     (50.0, 2000.0, 0.260),
#     (50.0, 2500.0, 0.195),
# ]

# # Typical white-matter tract (Δ=50 ms)
# MEASURED = [
#     (50.0,  500.0, 0.640),
#     (50.0, 1000.0, 0.440),
#     (50.0, 1500.0, 0.320),
#     (50.0, 2000.0, 0.245),
#     (50.0, 2500.0, 0.200),
# ]
# Edema grey matter — extracellular water + retained cells (Δ=50 ms)
MEASURED = [
    (50.0,  500.0, 0.580),
    (50.0, 1000.0, 0.360),
    (50.0, 1500.0, 0.240),
    (50.0, 2000.0, 0.170),
    (50.0, 2500.0, 0.125),
]
# -- Rician option --------------------------------------------------
# b=0 SNR (= S0 / sigma).  Used ONLY for the Rician-on landscape.
# Typical brain b=0 SNR is ~20-40.  Must be > sqrt(2) (~1.41).
SNR0 = 30.0

# -- Which variable is fixed, and at what value ---------------------
# FIXED_VAR is one of: "kio", "rho", "V".
# FIXED_VALUE is snapped to the nearest available library grid value
# (a warning is printed if it is not an exact match).
FIXED_VAR   = "rho"
FIXED_VALUE = 1.2e6         # kio [s^-1] | rho [cells/uL] | V [pL]

# -- Sweep ranges for the two FREE variables ------------------------
# (low, high) inclusive, in native units; or None to use the library's
# full min/max for that variable.  The variable named in FIXED_VAR is
# ignored here.
KIO_RANGE = None           # e.g. (5, 50)        [s^-1]
RHO_RANGE = None           # e.g. (1e5, 1.2e6)   [cells/uL]
V_RANGE   = None           # e.g. (0.5, 5.0)     [pL]

# -- Matcher options (mirror match_voxels_batch defaults) -----------
VI_MIN    = 0.0            # intracellular-fraction window (paper: 0.5)
VI_MAX    = 10.0           # vi = rho*V*1e-6
RHO_MAX   = None           # optional upper rho bound [cells/uL] or None
LOG_SPACE = False          # log-space residual (match --log_space)
S_FLOOR   = 1e-3           # clip floor used in log-space

# -- Cosmetics ------------------------------------------------------
ANNOTATE  = "auto"         # "auto" -> annotate cells if grid <= 12x12; True/False to force
DPI       = 150


# ===================================================================
#  Library I/O  (mirrors madi.library.load_library / load_library_meta)
# ===================================================================

def load_library_npz(path):
    """Load entries and metadata from a MADI .npz library.

    Returns
    -------
    entries : dict of arrays {kio, rho, V, vectors}
        vectors has shape (n_entries, n_deltas*n_b), delta-major.
    meta : dict {deltas, n_b, small_delta, b_values}
    """
    data = np.load(path)
    entries = dict(
        kio=np.asarray(data["kios"], dtype=float),
        rho=np.asarray(data["rhos"], dtype=float),
        V=np.asarray(data["Vs"], dtype=float),
        vectors=np.asarray(data["vectors"], dtype=float),
    )

    meta = {}
    meta["deltas"] = list(np.asarray(data["deltas"], dtype=float)) \
        if "deltas" in data.files else None
    meta["n_b"] = int(data["n_b"]) if "n_b" in data.files else None
    meta["small_delta"] = float(data["small_delta"]) \
        if "small_delta" in data.files else None
    meta["b_values"] = list(np.asarray(data["b_values"], dtype=float)) \
        if "b_values" in data.files else None

    if meta["deltas"] is None or meta["n_b"] is None or meta["b_values"] is None:
        raise ValueError(
            f"Library {path} is missing deltas / n_b / b_values metadata. "
            f"Rebuild it with the current _save_library."
        )
    return entries, meta


# ===================================================================
#  Column subsetting  (mirrors madi.library._pair_indices)
# ===================================================================

def pair_indices(fit_pairs, lib_deltas, lib_b_values, n_b, b_tol=50.0):
    """Flat-vector column index for each (Δ, b) pair: col = di*n_b + bi."""
    cols = np.empty(len(fit_pairs), dtype=int)
    for k, (d, b) in enumerate(fit_pairs):
        di = next((i for i, ld in enumerate(lib_deltas) if abs(d - ld) < 0.01), None)
        if di is None:
            raise ValueError(f"Δ = {d} ms not in library deltas {list(lib_deltas)}.")
        bi = next((j for j, lb in enumerate(lib_b_values) if abs(b - lb) < b_tol), None)
        if bi is None:
            raise ValueError(
                f"b = {b} s/mm² not in library b-values {list(lib_b_values)} "
                f"(tol ±{b_tol})."
            )
        cols[k] = di * n_b + bi
    return cols


# ===================================================================
#  Loss + Rician  (mirror match_voxels_batch math exactly)
# ===================================================================

def rician_normalized(norm_vec, snr0):
    """Apply the second-moment Rician correction to a NORMALIZED vector.

    norm_corrected = sqrt(max(norm^2 - 2/snr0^2, 0)) / sqrt(max(1 - 2/snr0^2, 0))
    """
    norm_vec = np.asarray(norm_vec, dtype=float)
    nt = 2.0 / (snr0 ** 2)                      # = 2*(sigma/S0)^2
    denom2 = 1.0 - nt
    if denom2 <= 0:
        raise ValueError(
            f"SNR0={snr0} is too low: the b=0 signal sits below the Rician "
            f"noise floor (needs SNR0 > sqrt(2) ≈ 1.41)."
        )
    num = np.sqrt(np.clip(norm_vec ** 2 - nt, 0.0, None))
    return num / np.sqrt(denom2)


def loss_per_entry(measured_vec, lib_subset, log_space, s_floor):
    """Squared-L2 loss for each library row vs the measured vector.

    lib_subset : (n_entries, n_features)
    measured_vec : (n_features,)
    Returns (n_entries,) array of ||m - r||^2 (in log-space if requested).
    """
    if log_space:
        m = np.log(np.clip(measured_vec, s_floor, 1.0))
        r = np.log(np.clip(lib_subset, s_floor, 1.0))
    else:
        m = measured_vec
        r = lib_subset
    diff = r - m[None, :]
    return np.sum(diff * diff, axis=1)


# ===================================================================
#  Grid assembly
# ===================================================================

VAR_INFO = {
    "kio": dict(unit="s$^{-1}$", label="k$_{io}$"),
    "rho": dict(unit="cells/µL", label="ρ"),
    "V":   dict(unit="pL",       label="V"),
}


def _snap_fixed_value(entries, fixed_var, fixed_value):
    grid = np.array(sorted(set(entries[fixed_var])))
    j = int(np.argmin(np.abs(grid - fixed_value)))
    snapped = float(grid[j])
    if abs(snapped - fixed_value) > 1e-6 * max(1.0, abs(snapped)):
        print(f"  ⚠ {fixed_var}={fixed_value:g} is not an exact library grid "
              f"value; snapping to nearest = {snapped:g}.")
        print(f"    available {fixed_var}: {[f'{g:g}' for g in grid]}")
    else:
        print(f"  Fixed {fixed_var} = {snapped:g} (exact grid value).")
    return snapped


def _free_axis_values(entries, var, rng):
    vals = np.array(sorted(set(entries[var])))
    if rng is not None:
        lo, hi = rng
        vals = vals[(vals >= lo - 1e-9) & (vals <= hi + 1e-9)]
    if vals.size == 0:
        raise ValueError(f"No library {var} values in range {rng}.")
    return vals


def build_loss_grid(entries, meta, measured_vec, fit_pairs,
                    fixed_var, fixed_value, free_vars, free_ranges,
                    log_space, s_floor, vi_min, vi_max, rho_max,
                    snr0=None):
    """Assemble a 2-D loss grid over the two free variables.

    If snr0 is given, the measured vector is Rician-corrected first.

    Returns
    -------
    Z       : (n_y, n_x) loss grid (NaN where no candidate entry exists)
    x_vals  : free_vars[0] axis values   (columns)
    y_vals  : free_vars[1] axis values   (rows)
    fixed_value_snapped : float
    """
    cols = pair_indices(fit_pairs, meta["deltas"], meta["b_values"], meta["n_b"])
    lib_subset_full = entries["vectors"][:, cols]            # (n_entries, n_feat)

    m = measured_vec if snr0 is None else rician_normalized(measured_vec, snr0)

    fixed_value = _snap_fixed_value(entries, fixed_var, fixed_value)

    xvar, yvar = free_vars
    x_vals = _free_axis_values(entries, xvar, free_ranges[xvar])
    y_vals = _free_axis_values(entries, yvar, free_ranges[yvar])

    # Per-entry loss (independent of which slice the entry lands in)
    all_loss = loss_per_entry(m, lib_subset_full, log_space, s_floor)

    # vi / rho candidate window (entries outside are never fit candidates)
    vi = (entries["rho"] / 1e9) * (entries["V"] * 1e3)
    candidate = (vi >= vi_min) & (vi <= vi_max)
    if rho_max is not None:
        candidate &= entries["rho"] <= rho_max

    # Map (xvar,yvar) -> entry index for this fixed slice
    Z = np.full((y_vals.size, x_vals.size), np.nan, dtype=float)

    fixed_match = np.abs(entries[fixed_var] - fixed_value) <= 1e-6 * max(1.0, abs(fixed_value))
    sel = np.where(fixed_match & candidate)[0]
    xi = {round(v, 9): i for i, v in enumerate(x_vals)}
    yi = {round(v, 9): i for i, v in enumerate(y_vals)}
    for idx in sel:
        xk = round(float(entries[xvar][idx]), 9)
        yk = round(float(entries[yvar][idx]), 9)
        if xk in xi and yk in yi:
            Z[yi[yk], xi[xk]] = all_loss[idx]

    return Z, x_vals, y_vals, fixed_value


# ===================================================================
#  Plotting
# ===================================================================

def _fmt_ticklabels(var, vals):
    if var == "rho":
        return [f"{v/1e3:.0f}k" for v in vals]
    if var == "kio":
        return [f"{v:g}" for v in vals]
    return [f"{v:g}" for v in vals]


def _annotate_decision(Z):
    if ANNOTATE is True:
        return True
    if ANNOTATE is False:
        return False
    return Z.size <= 144  # "auto": annotate small grids only


def plot_landscapes(Z_no, Z_ri, x_vals, y_vals, free_vars,
                    fixed_var, fixed_value, out_dir, snr0, measured_desc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    xvar, yvar = free_vars
    xlab = f"{VAR_INFO[xvar]['label']}  [{VAR_INFO[xvar]['unit']}]"
    ylab = f"{VAR_INFO[yvar]['label']}  [{VAR_INFO[yvar]['unit']}]"
    xticklabels = _fmt_ticklabels(xvar, x_vals)
    yticklabels = _fmt_ticklabels(yvar, y_vals)

    # Shared colour scale across the two loss panels for fair comparison.
    both = np.concatenate([Z_no[np.isfinite(Z_no)], Z_ri[np.isfinite(Z_ri)]])
    if both.size == 0:
        raise ValueError("No valid (candidate) cells in this slice — nothing "
                         "to plot. Check FIXED_VALUE, ranges, and vi window.")
    vmin, vmax = float(both.min()), float(both.max())

    Zdiff = Z_ri - Z_no
    dmax = np.nanmax(np.abs(Zdiff)) if np.isfinite(Zdiff).any() else 1.0
    dmax = dmax if dmax > 0 else 1.0

    panels = [
        ("No Rician", Z_no, "viridis", dict(vmin=vmin, vmax=vmax)),
        (f"Rician (SNR₀={snr0:g})", Z_ri, "viridis", dict(vmin=vmin, vmax=vmax)),
        ("Rician − No Rician", Zdiff, "RdBu_r",
         dict(norm=TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax))),
    ]

    annotate = _annotate_decision(Z_no)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)

    for ax, (title, Z, cmap, norm_kw) in zip(axes, panels):
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                       interpolation="nearest", **norm_kw)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels(yticklabels, fontsize=8)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)

        # Mark the in-slice minimum loss for the two loss panels.
        if cmap == "viridis" and np.isfinite(Z).any():
            j, i = np.unravel_index(np.nanargmin(Z), Z.shape)
            ax.plot(i, j, marker="*", color="red", markersize=16,
                    markeredgecolor="white", markeredgewidth=0.8)

        if annotate:
            for jj in range(Z.shape[0]):
                for ii in range(Z.shape[1]):
                    v = Z[jj, ii]
                    if np.isfinite(v):
                        ax.text(ii, jj, f"{v:.2g}", ha="center", va="center",
                                fontsize=6,
                                color="white" if cmap == "viridis" else "black")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("loss  ‖m − r‖²" if cmap == "viridis" else "Δ loss",
                     fontsize=9)

    fx_unit = VAR_INFO[fixed_var]["unit"]
    fig.suptitle(
        f"MADI fitting error landscape   "
        f"({VAR_INFO[fixed_var]['label']} fixed = {fixed_value:g} {fx_unit})\n"
        f"{measured_desc}",
        fontsize=12,
    )

    combined = os.path.join(out_dir, "error_landscape_combined.png")
    fig.savefig(combined, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Also save the two loss panels individually for slides/reports.
    saved = [combined]
    for tag, Z, cmap, norm_kw, title in [
        ("no_rician", Z_no, "viridis", dict(vmin=vmin, vmax=vmax), "No Rician"),
        ("rician",    Z_ri, "viridis", dict(vmin=vmin, vmax=vmax),
         f"Rician (SNR₀={snr0:g})"),
    ]:
        f, a = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
        im = a.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                      interpolation="nearest", **norm_kw)
        a.set_xticks(range(len(x_vals)))
        a.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(len(y_vals)))
        a.set_yticklabels(yticklabels, fontsize=8)
        a.set_xlabel(xlab); a.set_ylabel(ylab)
        a.set_title(f"{title}   ({VAR_INFO[fixed_var]['label']}={fixed_value:g} "
                    f"{fx_unit})", fontsize=11)
        if np.isfinite(Z).any():
            j, i = np.unravel_index(np.nanargmin(Z), Z.shape)
            a.plot(i, j, marker="*", color="red", markersize=16,
                   markeredgecolor="white", markeredgewidth=0.8)
        if annotate:
            for jj in range(Z.shape[0]):
                for ii in range(Z.shape[1]):
                    v = Z[jj, ii]
                    if np.isfinite(v):
                        a.text(ii, jj, f"{v:.2g}", ha="center", va="center",
                               fontsize=6, color="white")
        cb = f.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        cb.set_label("loss  ‖m − r‖²", fontsize=9)
        p = os.path.join(out_dir, f"error_landscape_{tag}.png")
        f.savefig(p, dpi=DPI, bbox_inches="tight")
        plt.close(f)
        saved.append(p)

    return saved


# ===================================================================
#  Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Plot the MADI fitting error landscape over library "
                    "grid points (Rician on/off).")
    ap.add_argument("--library", required=True,
                    help="Path to the MADI .npz library.")
    ap.add_argument("--out", default="error_landscape_out",
                    help="Output directory for the plots.")
    args = ap.parse_args()

    if FIXED_VAR not in VAR_INFO:
        raise ValueError(f"FIXED_VAR must be one of {list(VAR_INFO)}, "
                         f"got '{FIXED_VAR}'.")
    free_vars = [v for v in ("kio", "rho", "V") if v != FIXED_VAR]  # ordered
    free_ranges = {"kio": KIO_RANGE, "rho": RHO_RANGE, "V": V_RANGE}

    os.makedirs(args.out, exist_ok=True)

    print("=" * 64)
    print("MADI error-landscape plotter")
    print("=" * 64)
    print(f"  Library:     {args.library}")
    print(f"  Output dir:  {args.out}")

    entries, meta = load_library_npz(args.library)
    print(f"  Entries:     {len(entries['kio'])}")
    print(f"  Δ [ms]:      {[f'{d:g}' for d in meta['deltas']]}")
    print(f"  b [s/mm²]:   {[f'{b:g}' for b in meta['b_values']]}  (n_b={meta['n_b']})")
    print(f"  small δ:     {meta['small_delta']} ms")

    fit_pairs = [(d, b) for (d, b, _) in MEASURED]
    measured_vec = np.array([s for (_, _, s) in MEASURED], dtype=float)
    measured_desc = "measured S/S0:  " + ",  ".join(
        f"Δ{d:g}/b{b:g}={s:g}" for (d, b, s) in MEASURED)
    print(f"  Features:    {len(fit_pairs)}  → {fit_pairs}")
    print(f"  {measured_desc}")
    print(f"  Fixed:       {FIXED_VAR} = {FIXED_VALUE:g}")
    print(f"  Free axes:   x={free_vars[0]}, y={free_vars[1]}")
    print(f"  vi window:   [{VI_MIN}, {VI_MAX}]   rho_max={RHO_MAX}   "
          f"log_space={LOG_SPACE}")
    print(f"  Rician SNR₀: {SNR0:g}")

    # ---- Build both landscapes ----
    Z_no, x_vals, y_vals, fixed_snapped = build_loss_grid(
        entries, meta, measured_vec, fit_pairs,
        FIXED_VAR, FIXED_VALUE, free_vars, free_ranges,
        LOG_SPACE, S_FLOOR, VI_MIN, VI_MAX, RHO_MAX, snr0=None)

    Z_ri, _, _, _ = build_loss_grid(
        entries, meta, measured_vec, fit_pairs,
        FIXED_VAR, FIXED_VALUE, free_vars, free_ranges,
        LOG_SPACE, S_FLOOR, VI_MIN, VI_MAX, RHO_MAX, snr0=SNR0)

    # ---- Report the best (argmin) point in this slice ----
    for tag, Z in [("No-Rician", Z_no), ("Rician", Z_ri)]:
        if np.isfinite(Z).any():
            j, i = np.unravel_index(np.nanargmin(Z), Z.shape)
            print(f"  [{tag}] in-slice min loss = {Z[j, i]:.4g} at "
                  f"{free_vars[0]}={x_vals[i]:g}, {free_vars[1]}={y_vals[j]:g}")
        else:
            print(f"  [{tag}] no candidate cells in slice.")

    n_valid = int(np.isfinite(Z_no).sum())
    print(f"  Valid candidate cells in slice: {n_valid} / {Z_no.size}")

    # ---- Plot ----
    saved = plot_landscapes(Z_no, Z_ri, x_vals, y_vals, free_vars,
                            FIXED_VAR, fixed_snapped, args.out, SNR0,
                            measured_desc)

    # ---- Dump the raw grids for further analysis ----
    npz_path = os.path.join(args.out, "error_landscape_grids.npz")
    np.savez(npz_path,
             Z_no_rician=Z_no, Z_rician=Z_ri,
             x_var=free_vars[0], y_var=free_vars[1],
             x_vals=x_vals, y_vals=y_vals,
             fixed_var=FIXED_VAR, fixed_value=fixed_snapped,
             snr0=SNR0, log_space=LOG_SPACE,
             vi_min=VI_MIN, vi_max=VI_MAX)
    print(f"\n  Saved grids: {npz_path}")
    print("  Saved plots:")
    for p in saved:
        print(f"    {p}")
    print("\nDone.")


if __name__ == "__main__":
    main()

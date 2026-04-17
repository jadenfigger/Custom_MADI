#!/usr/bin/env python3
"""
MADI Interactive Viewer
========================
A clean diagnostic tool for inspecting MADI fits voxel by voxel.

Click a voxel on the parametric map to see:
  • The top-10 library matches (ranked table)
  • Observed vs predicted signal across b-values, for each Δ
  • A live comparison between the top library match and the saved
    parameter maps on disk (flags disagreements)

Designed for single-output inspection. To compare different fitting
configurations (averageb0 vs Rician vs baseline, etc.), launch the
viewer separately for each output directory.


USAGE
-----
    python madi_viewer.py \\
        --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \\
        --mask mask.nii.gz \\
        --library madi_library_default.npz \\
        --madi-dir madi_output_v2

Optional:
    --slice N       Starting slice index (default: middle of volume)
    --axis {0,1,2}  Slicing axis: 0=sagittal, 1=coronal, 2=axial (default)
    --map NAME      Starting map: kio | rho | V | residual (default: kio)
    --margin MM     Padding around brain in auto-zoom (default: 1.5 mm)
    --log-space     Rank matches by log-space SSE (use this when the fit was
                    run with log-space). Default is linear-space SSE. Both
                    values are always shown in the match table either way;
                    this flag only controls which column drives the ranking.


CONTROLS
--------
    Click               Select voxel
    ← / →               Cycle Δ shown in signal plot
    ↑ / ↓               Cycle top-10 library matches
    Ctrl + Arrows       Nudge selected voxel by 1
    Ctrl+Shift+Arrows   Nudge selected voxel by 5
    PageUp / PageDown   Previous / next slice
    m                   Cycle parametric map (kio → ρ → V → residual)
    s                   Save screenshot to --madi-dir
    h                   Print controls to terminal
    q                   Quit


WHAT EACH PANEL SHOWS
---------------------
  [Left]   Parametric map with crosshair at the selected voxel.
           Auto-zoomed to brain extent. Colorbar below.

  [Top R]  Info strip with the selected voxel's disk-map values and
           the top-1 live-fit values, side by side. A ⚠ appears if
           they disagree (live #1 ≠ saved maps).

  [Mid R]  Signal plot. Black dots = observed; colored line = the
           currently-selected library match; faded lines = the other
           9 top matches. Arrow keys swap which Δ and which match.

  [Bot R]  Top-10 match table. Selected row is highlighted in color
           that matches the signal-plot curve.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from madi.library import load_library, load_library_meta


# ============================================================
# CONFIG
# ============================================================
SHELLS = [
    (1000, slice(1, 25)),
    (2500, slice(25, 49)),
    (4000, slice(49, 73)),
    (6000, slice(73, 97)),
]
BVALS_DISPLAY = np.array([1000, 2500, 4000, 6000])
N_SHELLS = len(SHELLS)
N_TOP = 10

# Qualitative palette for the 10 matches. First color = selected match by default.
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
          "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5"]
OBS_COLOR = "#111827"

MAP_CMAPS = {"kio": "inferno", "rho": "viridis", "V": "plasma", "residual": "magma"}
MAP_LABELS = {
    "kio":      r"$k_{io}$  (s$^{-1}$)",
    "rho":      r"$\rho$  (cells/μL)",
    "V":        r"$V$  (pL)",
    "residual": "SSE (from fit)",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "#cbd5e1",
    "axes.facecolor":    "#ffffff",
    "axes.grid":         True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.9,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cbd5e1",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.facecolor":  "white",
})


# ============================================================
# HELPERS
# ============================================================
def get_slice_2d(vol, axis, idx):
    s = [slice(None)] * vol.ndim
    s[axis] = idx
    return vol[tuple(s)]


def bbox_of(mask_2d_rot, margin_px_x, margin_px_y):
    rows_m, cols_m = np.where(mask_2d_rot)
    if len(rows_m) == 0:
        return None
    return (cols_m.min(), cols_m.max(), rows_m.min(), rows_m.max())


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="MADI Interactive Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--madi-dir", default="madi_output",
                    help="Directory containing *_map.nii.gz outputs")
    ap.add_argument("--library", default="madi_library.npz",
                    help="Path to MADI library .npz")
    ap.add_argument("--input", type=str, nargs="+", required=True,
                    help="'delta:path' pairs, e.g. 15:dwi15.nii.gz 25:dwi25.nii.gz")
    ap.add_argument("--mask", required=True, help="Brain mask NIfTI")
    ap.add_argument("--slice", type=int, default=None,
                    help="Starting slice (default: middle)")
    ap.add_argument("--axis", type=int, default=2,
                    help="0=sag, 1=cor, 2=axial (default)")
    ap.add_argument("--map", default="kio",
                    choices=["kio", "rho", "V", "residual"],
                    help="Starting parametric map")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="Auto-zoom margin around brain (mm)")
    ap.add_argument("--log-space", action="store_true",
                    help="Rank library matches by log-space SSE (matches the "
                         "fitting pipeline when it was run with log-space). "
                         "Default is linear-space SSE. Both values are always "
                         "shown in the match table regardless.")
    args = ap.parse_args()
    sort_mode = "log" if args.log_space else "lin"

    # ---------------- Parse DWI inputs ----------------
    dwi_inputs = []
    for s in args.input:
        d, p = s.split(":", 1)
        dwi_inputs.append((float(d), p))
    dwi_inputs.sort()
    fit_deltas = [d for d, _ in dwi_inputs]
    n_fit = len(fit_deltas)

    print("=" * 64)
    print("  MADI INTERACTIVE VIEWER")
    print("=" * 64)
    print(f"  Output dir : {args.madi_dir}")
    print(f"  Library    : {args.library}")
    print(f"  Δ values   : {fit_deltas} ms")
    print(f"  Sort by    : {'log-space' if args.log_space else 'linear-space'} SSE")

    # ---------------- Load parameter maps ----------------
    print("\n[1/4] Loading parameter maps ...")
    maps = {}
    zooms = None
    for name in ["kio", "rho", "V", "residual"]:
        path = os.path.join(args.madi_dir, f"{name}_map.nii.gz")
        if os.path.exists(path):
            img = nib.load(path)
            maps[name] = img.get_fdata()
            if zooms is None:
                zooms = np.array(img.header.get_zooms()[:3], dtype=float)
            print(f"      ✓ {name:<10} shape={maps[name].shape}")
        else:
            print(f"      ✗ {name:<10} NOT FOUND ({path})")
    if not maps:
        print("\nERROR: No parameter maps found in --madi-dir. Exiting.")
        return

    # Fall back starting map if the requested one isn't available.
    if args.map not in maps:
        args.map = next(iter(maps))
        print(f"      (starting map fell back to '{args.map}')")

    # ---------------- Load mask ----------------
    print("\n[2/4] Loading mask ...")
    mask_img = nib.load(args.mask)
    mask = mask_img.get_fdata().astype(bool)
    if zooms is None:
        zooms = np.array(mask_img.header.get_zooms()[:3], dtype=float)
    print(f"      Voxel size : {zooms[0]:.3f} × {zooms[1]:.3f} × {zooms[2]:.3f} mm")
    print(f"      Volume     : {mask.shape}, {int(mask.sum())} mask voxels total")

    # ---------------- Load library ----------------
    print("\n[3/4] Loading library ...")
    library = load_library(args.library)
    meta = load_library_meta(args.library)
    lib_deltas = meta["deltas"]
    n_b = meta["n_b"]
    lib_kios = np.array([e.kio for e in library])
    lib_rhos = np.array([e.rho for e in library])
    lib_Vs   = np.array([e.V   for e in library])
    lib_vecs = np.array([e.vector for e in library])
    print(f"      Entries    : {len(library)}")
    print(f"      Library Δ  : {list(lib_deltas)} ms")

    # Map fit deltas → library column indices
    fit_di = []
    for d in fit_deltas:
        for i, ld in enumerate(lib_deltas):
            if abs(d - ld) < 0.01:
                fit_di.append(i); break
        else:
            print(f"\nERROR: Δ={d}ms not in library. Available: {list(lib_deltas)}")
            return
    lib_sub = np.hstack([lib_vecs[:, di * n_b:(di + 1) * n_b] for di in fit_di])
    print(f"      Using lib columns {fit_di} → sub-vector length {lib_sub.shape[1]}")

    # ---------------- Load DWI signals ----------------
    print("\n[4/4] Loading DWI signals ...")
    ref_shape = mask.shape
    signal_vol = np.zeros((*ref_shape, n_fit * N_SHELLS), dtype=np.float32)

    for di, (delta_ms, path) in enumerate(dwi_inputs):
        print(f"      Δ={delta_ms:>3.0f}ms  {os.path.basename(path)}", end=" ... ",
              flush=True)
        data = nib.load(path).get_fdata()
        mi_idx = np.where(mask)
        n_loaded = 0
        for vi in range(len(mi_idx[0])):
            ix, iy, iz = mi_idx[0][vi], mi_idx[1][vi], mi_idx[2][vi]
            S0 = data[ix, iy, iz, 0]
            if S0 < 1e-10:
                continue
            for si, (_, vol_sl) in enumerate(SHELLS):
                sm = np.mean(data[ix, iy, iz, vol_sl])
                signal_vol[ix, iy, iz, di * N_SHELLS + si] = np.clip(sm / S0, 0, 1)
            n_loaded += 1
        print(f"{n_loaded} voxels")

    # ---------------- Geometry setup ----------------
    ax_dir = args.axis
    n_slices = ref_shape[ax_dir]
    sl_idx_init = args.slice if args.slice is not None else n_slices // 2

    axes_2d = [i for i in range(3) if i != ax_dir]
    zx = zooms[axes_2d[0]]
    zy = zooms[axes_2d[1]]

    # These depend on slice geometry, which is constant across slices.
    tmp_slice = get_slice_2d(mask, ax_dir, sl_idx_init)
    nx, ny = tmp_slice.shape
    W = nx * zx
    H = ny * zy
    extent = [0, W, 0, H]

    print(f"\n  Slice plane     : {nx} × {ny} voxels → {W:.1f} × {H:.1f} mm")
    print(f"  Aspect (zx/zy)  : {zx / zy:.3f}")

    # ----------------------------------------------------------------
    # Coordinate transforms (array ↔ physical display).
    # rot90(A) with origin="lower" means:
    #   col c  → voxel x-index c
    #   row r  → voxel y-index (ny - 1 - r)
    # ----------------------------------------------------------------
    def disp_to_vx(xphys, yphys):
        c = int(xphys / zx)
        r = int(yphys / zy)
        vx = int(np.clip(c, 0, nx - 1))
        vy = int(np.clip(ny - 1 - r, 0, ny - 1))
        return vx, vy

    def vx_to_disp(vx, vy):
        xp = (vx + 0.5) * zx
        yp = (ny - 1 - vy + 0.5) * zy
        return xp, yp

    # ---------------- Data access helpers ----------------
    def get_map_slice(name, sl):
        if name not in maps:
            return np.zeros((nx, ny))
        return get_slice_2d(maps[name], ax_dir, sl)

    def get_mask_slice(sl):
        return get_slice_2d(mask, ax_dir, sl)

    def get_signal_vec(vx, vy, sl):
        if ax_dir == 0:   return signal_vol[sl, vx, vy, :]
        elif ax_dir == 1: return signal_vol[vx, sl, vy, :]
        else:             return signal_vol[vx, vy, sl, :]

    def get_map_val(name, vx, vy, sl):
        if name not in maps:
            return 0.0
        if ax_dir == 0:   return maps[name][sl, vx, vy]
        elif ax_dir == 1: return maps[name][vx, sl, vy]
        else:             return maps[name][vx, vy, sl]

    # Sanity: coord roundtrip
    tx, ty = nx // 2, ny // 2
    bx, by = disp_to_vx(*vx_to_disp(tx, ty))
    assert (bx, by) == (tx, ty), "Coordinate transform failed"

    # ---------------- State ----------------
    class S:
        pass
    st = S()
    st.sl = sl_idx_init
    st.x = nx // 2
    st.y = ny // 2
    st.selected = False
    st.cur_map = args.map
    st.di_show = 0
    st.mi = 0
    st.matches = None
    st.measured = None
    st.log_y = False

    # ================================================================
    # FIGURE LAYOUT
    # ================================================================
    fig = plt.figure(figsize=(17, 9))
    fig.canvas.manager.set_window_title(f"MADI Viewer — {args.madi_dir}")

    # Header (thin bar at top, full width)
    header_ax = fig.add_axes([0.02, 0.955, 0.96, 0.035])
    header_ax.set_axis_off()
    header_text = header_ax.text(
        0.0, 0.5, "", transform=header_ax.transAxes,
        fontsize=10, fontweight="bold", color="#111827",
        family="monospace", va="center", ha="left")

    # Master grid: 2 cols (map | right panel), with right panel stacked 3-high.
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[1.0, 1.35],
        height_ratios=[0.18, 1.0, 0.7],
        left=0.04, right=0.985, top=0.94, bottom=0.055,
        wspace=0.14, hspace=0.28,
    )

    # LEFT: parametric map spans all rows
    ax_img = fig.add_subplot(gs[:, 0])

    # RIGHT column
    ax_info   = fig.add_subplot(gs[0, 1])   # live-fit vs disk-map summary
    ax_signal = fig.add_subplot(gs[1, 1])   # signal plot
    ax_table  = fig.add_subplot(gs[2, 1])   # match table

    ax_info.set_axis_off()
    ax_table.set_axis_off()

    # Colorbar sits just below the map axis
    cbar_ax = fig.add_axes([0.06, 0.035, 0.34, 0.013])

    crosshair_artist = [None]
    image_artist     = [None]
    footer_text_artist = [None]

    # Footer hint line
    footer_ax = fig.add_axes([0.02, 0.005, 0.96, 0.022])
    footer_ax.set_axis_off()
    footer_text_artist[0] = footer_ax.text(
        0.5, 0.5,
        "Click a voxel  •  ← / → cycle Δ  •  ↑ / ↓ cycle match  "
        "•  m change map  •  PgUp/PgDn slice  •  s save  •  h help  •  q quit",
        transform=footer_ax.transAxes, ha="center", va="center",
        fontsize=8, color="#6b7280")

    # ================================================================
    # DRAW: HEADER
    # ================================================================
    def update_header():
        if st.selected:
            tag = f"Voxel ({st.x}, {st.y})  |  Slice {st.sl}"
        else:
            tag = f"Slice {st.sl}"
        header_text.set_text(
            f"  MADI  •  {os.path.basename(args.madi_dir)}  "
            f"•  Map: {st.cur_map}   —   {tag}   "
            f"•  Δ: {fit_deltas} ms"
        )

    # ================================================================
    # DRAW: PARAMETRIC MAP
    # ================================================================
    def draw_map(keep_view=False):
        # Preserve current zoom if the user has panned/zoomed
        if keep_view:
            xlim = ax_img.get_xlim()
            ylim = ax_img.get_ylim()
        ax_img.clear()

        mask_sl = get_mask_slice(st.sl)

        # Black background
        ax_img.imshow(np.zeros((ny, nx)), cmap="gray", origin="lower",
                      extent=extent, aspect="equal", vmin=0, vmax=1)

        # Parametric overlay (masked)
        sl2d = get_map_slice(st.cur_map, st.sl)
        sl_rot = np.rot90(sl2d)
        overlay_mask = np.rot90(~mask_sl) | (sl_rot == 0)
        sl_masked = np.ma.masked_where(overlay_mask, sl_rot)
        cmap = MAP_CMAPS.get(st.cur_map, "viridis")
        image_artist[0] = ax_img.imshow(
            sl_masked, cmap=cmap, origin="lower",
            extent=extent, aspect="equal", interpolation="nearest")

        # Auto-zoom to brain (unless user already zoomed)
        if not keep_view:
            mask_rot = np.rot90(mask_sl)
            bb = bbox_of(mask_rot, 0, 0)
            if bb is not None:
                cmin, cmax, rmin, rmax = bb
                xmin_p = cmin * zx - args.margin
                xmax_p = (cmax + 1) * zx + args.margin
                ymin_p = rmin * zy - args.margin
                ymax_p = (rmax + 1) * zy + args.margin
                ax_img.set_xlim(xmin_p, xmax_p)
                ax_img.set_ylim(ymin_p, ymax_p)
        else:
            ax_img.set_xlim(xlim)
            ax_img.set_ylim(ylim)

        ax_img.set_title(f"{st.cur_map}   —   slice {st.sl}",
                         fontsize=11, pad=6, color="#111827")
        ax_img.set_xlabel("mm"); ax_img.set_ylabel("mm")
        ax_img.tick_params(labelsize=7, colors="#6b7280")

        # Colorbar
        cbar_ax.clear()
        valid = sl_masked.compressed()
        if valid.size:
            vmin, vmax = float(valid.min()), float(valid.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7, colors="#374151")
        cb.set_label(MAP_LABELS.get(st.cur_map, ""),
                     fontsize=8, color="#374151", labelpad=2)
        cb.outline.set_edgecolor("#cbd5e1")
        cb.outline.set_linewidth(0.6)

        # Crosshair
        crosshair_artist[0], = ax_img.plot([], [], "+", color="white",
                                           ms=16, mew=2.2,
                                           markeredgecolor="white",
                                           path_effects=None, zorder=10)
        if st.selected:
            px, py = vx_to_disp(st.x, st.y)
            crosshair_artist[0].set_data([px], [py])

    # ================================================================
    # DRAW: INFO STRIP (live #1 vs disk maps)
    # ================================================================
    def draw_info():
        ax_info.clear()
        ax_info.set_axis_off()

        if not st.selected:
            ax_info.text(0.5, 0.5,
                         "Click a voxel on the map to begin",
                         transform=ax_info.transAxes,
                         ha="center", va="center",
                         fontsize=11, color="#6b7280", style="italic")
            return

        # Pull values
        kv = get_map_val("kio", st.x, st.y, st.sl)
        rv = get_map_val("rho", st.x, st.y, st.sl)
        vv = get_map_val("V",   st.x, st.y, st.sl)
        sv = get_map_val("residual", st.x, st.y, st.sl)

        has_live = bool(st.matches)
        if has_live:
            m1 = st.matches[0]
            lk, lr, lv = m1["kio"], m1["rho"], m1["V"]
            ls_lin, ls_log = m1["sse_lin"], m1["sse_log"]
            # Agreement check
            disagree = (
                abs(lk - kv) > 0.5 or
                abs(lr - rv) / max(rv, 1) > 0.05 or
                abs(lv - vv) > 0.05
            )
        else:
            lk = lr = lv = None
            ls_lin = ls_log = None
            disagree = False

        # Draw two-column comparison as a mini-table
        def fmt(kio, rho, V):
            if kio is None:
                return "—", "—", "—"
            return (f"{kio:>6.1f}",
                    f"{rho/1e3:>6.0f} k",
                    f"{V:>6.2f}")

        def fmt_sse(val):
            return "   —   " if val is None else f"{val:>8.4f}"

        s_k, s_r, s_v = fmt(kv, rv, vv)
        l_k, l_r, l_v = fmt(lk, lr, lv)
        s_sse_lin = fmt_sse(sv)        # disk "residual" map (unknown space)
        l_sse_lin = fmt_sse(ls_lin)
        l_sse_log = fmt_sse(ls_log)

        flag = "  ⚠ disagree" if disagree else ""
        title_color = "#b45309" if disagree else "#065f46"

        header = f"Voxel ({st.x}, {st.y})  —  slice {st.sl}{flag}"
        ax_info.text(0.0, 0.95, header, transform=ax_info.transAxes,
                     fontsize=10, fontweight="bold", color=title_color,
                     va="top", ha="left")

        # Mark which SSE the viewer is using to rank
        sort_tag = "(log)" if sort_mode == "log" else "(lin)"
        table_txt = (
            f"               k_io         ρ            V         "
            f"SSE lin     SSE log\n"
            f"Saved map   {s_k}    {s_r}    {s_v}    {s_sse_lin}       —\n"
            f"Live #1     {l_k}    {l_r}    {l_v}    {l_sse_lin}    {l_sse_log}"
            f"   {sort_tag}"
        )
        ax_info.text(0.0, 0.62, table_txt, transform=ax_info.transAxes,
                     fontsize=9, family="monospace",
                     color="#111827", va="top", ha="left")

    # ================================================================
    # MATCHING
    # ================================================================
    def find_top(vx, vy, sl):
        measured = get_signal_vec(vx, vy, sl)
        if np.all(measured < 1e-10):
            return None, measured

        # Match the pipeline's matching parameters
        s_floor = 1e-3
        vi_min = 0.050
        vi_max = 0.95

        vis = (lib_rhos / 1e9) * (lib_Vs * 1e3)
        valid_mask = (vis >= vi_min) & (vis <= vi_max)

        # ---- Linear-space SSE ----
        dists_lin = np.full(len(lib_sub), np.inf)
        dists_lin[valid_mask] = np.sum(
            (lib_sub[valid_mask] - measured[np.newaxis, :]) ** 2, axis=1)

        # ---- Log-space SSE ----
        meas_log = np.log(np.clip(measured, s_floor, 1.0))
        lib_log  = np.log(np.clip(lib_sub,  s_floor, 1.0))
        dists_log = np.full(len(lib_sub), np.inf)
        dists_log[valid_mask] = np.sum(
            (lib_log[valid_mask] - meas_log[np.newaxis, :]) ** 2, axis=1)

        # ---- Pick which to sort by ----
        dists_primary = dists_log if sort_mode == "log" else dists_lin
        order = np.argsort(dists_primary)[:N_TOP]

        matches = []
        for rank, idx in enumerate(order):
            if dists_primary[idx] == np.inf:
                break
            matches.append({
                "rank":    rank + 1,
                "kio":     lib_kios[idx],
                "rho":     lib_rhos[idx],
                "V":       lib_Vs[idx],
                "sse_lin": dists_lin[idx],
                "sse_log": dists_log[idx],
                # Keep `residual` for back-compat with info strip; it holds
                # whichever SSE the user asked to sort by.
                "residual": dists_primary[idx],
                "pred": lib_sub[idx],
            })
        return (matches if matches else None), measured

    # ================================================================
    # DRAW: SIGNAL PLOT
    # ================================================================
    def draw_signal():
        ax_signal.clear()

        if not st.selected:
            ax_signal.set_title("Signal",  fontsize=10)
            ax_signal.set_xlabel("b-value  (s/mm²)")
            ax_signal.set_ylabel(r"$S(b)\,/\,S_0$")
            ax_signal.text(0.5, 0.5, "Select a voxel to view its signal",
                           transform=ax_signal.transAxes,
                           ha="center", va="center",
                           fontsize=11, color="#6b7280", style="italic")
            ax_signal.set_xlim(600, 6800)
            ax_signal.set_ylim(-0.03, 1.05)
            return

        if not st.matches:
            ax_signal.set_title(f"Voxel ({st.x}, {st.y}) — no matches",
                                fontsize=10, color="#b91c1c")
            ax_signal.set_xlabel("b-value  (s/mm²)")
            ax_signal.set_ylabel(r"$S(b)\,/\,S_0$")
            ax_signal.set_xlim(600, 6800)
            ax_signal.set_ylim(-0.03, 1.05)
            return
    

        di = st.di_show % n_fit
        delta_ms = fit_deltas[di]
        m = st.matches[st.mi]

        obs  = st.measured[di * n_b : (di + 1) * n_b]
        pred = m["pred"][di * n_b : (di + 1) * n_b]

        # Faint: other matches
        for j, om in enumerate(st.matches):
            if j == st.mi:
                continue
            op = om["pred"][di * n_b : (di + 1) * n_b]
            ax_signal.plot(BVALS_DISPLAY, op, "o-",
                           color=COLORS[j % len(COLORS)],
                           ms=3, lw=0.8, alpha=0.22, zorder=2)

        # Selected match (bold)
        c = COLORS[st.mi % len(COLORS)]
        ax_signal.plot(BVALS_DISPLAY, pred, "o-", color=c,
                       ms=9, lw=2.4, alpha=0.95, zorder=5,
                       label=(f"Match #{m['rank']}   "
                              f"$k_{{io}}$={m['kio']:.0f}  "
                              f"ρ={m['rho']/1e3:.0f}k  "
                              f"V={m['V']:.2f}"))

        # Observed
        ax_signal.scatter(BVALS_DISPLAY, obs, c=OBS_COLOR, s=70, zorder=6,
                          label="Observed",
                          edgecolors="white", linewidths=1.0)

        ax_signal.set_title(f"Δ = {delta_ms:.0f} ms    "
                            f"({di + 1}/{n_fit})",
                            fontsize=10, color="#111827", pad=4)
        ax_signal.set_xlabel("b-value  (s/mm²)")
        ax_signal.set_ylabel(r"$S(b)\,/\,S_0$")
        ax_signal.legend(loc="upper right", frameon=True, fontsize=8)
        ax_signal.set_xlim(600, 6800)
        ymax = max(1.05,
                   float(np.max(obs))  * 1.15 if obs.size  else 1.05,
                   float(np.max(pred)) * 1.15 if pred.size else 1.05)
        if st.log_y:
            ax_signal.set_yscale("log")
            ax_signal.set_ylim(1e-3, ymax)   # log-scale can't handle 0 or negative
        else:
            ax_signal.set_yscale("linear")
            ax_signal.set_ylim(-0.03, ymax)

    # ================================================================
    # DRAW: TABLE
    # ================================================================
    def draw_table():
        ax_table.clear()
        ax_table.set_axis_off()

        if not st.selected or not st.matches:
            return

        # Header label: mark which SSE column drove the ranking
        sse_lin_lbl = "SSE (lin)" + ("  ◀" if sort_mode == "lin" else "")
        sse_log_lbl = "SSE (log)" + ("  ◀" if sort_mode == "log" else "")
        cols = ["#", "k_io (s⁻¹)", "ρ (k/μL)", "V (pL)", "v_i",
                sse_lin_lbl, sse_log_lbl]
        rows = []
        for m in st.matches:
            vi = (m["rho"] / 1e9) * (m["V"] * 1e3)
            rows.append([
                f"{m['rank']}",
                f"{m['kio']:.1f}",
                f"{m['rho'] / 1e3:.0f}",
                f"{m['V']:.2f}",
                f"{vi:.3f}",
                f"{m['sse_lin']:.5f}",
                f"{m['sse_log']:.5f}",
            ])

        cell_colors = []
        for i in range(len(rows)):
            if i == st.mi:
                cell_colors.append([COLORS[i % len(COLORS)] + "33"] * len(cols))
            else:
                cell_colors.append(
                    ["#ffffff" if i % 2 == 0 else "#f9fafb"] * len(cols))

        tbl = ax_table.table(
            cellText=rows, colLabels=cols,
            cellLoc="center", loc="center",
            cellColours=cell_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.28)

        # Header styling
        sort_col_idx = 5 if sort_mode == "lin" else 6
        for j in range(len(cols)):
            cell = tbl[0, j]
            if j == sort_col_idx:
                cell.set_facecolor("#0f766e")  # teal — the active sort
            else:
                cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#374151")

        # Highlight selected row outline
        for j in range(len(cols)):
            cell = tbl[st.mi + 1, j]
            cell.set_edgecolor(COLORS[st.mi % len(COLORS)])
            cell.set_linewidth(1.6)
            cell.set_text_props(fontweight="bold")

        # Soft edges elsewhere
        for i in range(len(rows)):
            for j in range(len(cols)):
                if i != st.mi:
                    tbl[i + 1, j].set_edgecolor("#e5e7eb")

        sort_label = "log-space" if sort_mode == "log" else "linear-space"
        ax_table.set_title(
            f"Top {len(st.matches)} library matches   —   "
            f"ranked by {sort_label} SSE",
            fontsize=9, color="#374151", pad=2)

    # ================================================================
    # COORDINATION: full update after voxel/slice change
    # ================================================================
    def update_voxel(vx, vy):
        st.matches, st.measured = find_top(vx, vy, st.sl)
        st.mi = 0
        st.di_show = 0
        draw_info()
        draw_signal()
        draw_table()
        update_header()
        fig.canvas.draw_idle()

    def change_slice(new_sl):
        new_sl = int(np.clip(new_sl, 0, n_slices - 1))
        if new_sl == st.sl:
            return
        st.sl = new_sl
        # Slice changed — voxel indices in the plane are the same, but
        # the underlying data differs.
        draw_map(keep_view=False)
        if st.selected:
            update_voxel(st.x, st.y)
        else:
            update_header()
            fig.canvas.draw_idle()

    # ================================================================
    # EVENTS
    # ================================================================
    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None:
            return
        vx, vy = disp_to_vx(event.xdata, event.ydata)
        mask_sl = get_mask_slice(st.sl)
        in_mask = bool(mask_sl[vx, vy])
        if not in_mask:
            print(f"  (click ignored — voxel ({vx},{vy}) is outside mask)")
            return
        st.x, st.y = vx, vy
        st.selected = True
        px, py = vx_to_disp(vx, vy)
        crosshair_artist[0].set_data([px], [py])
        update_voxel(vx, vy)

        if st.matches:
            top = st.matches[0]
            kv = get_map_val("kio", vx, vy, st.sl)
            rv = get_map_val("rho", vx, vy, st.sl)
            vv = get_map_val("V",   vx, vy, st.sl)
            print(f"  Voxel ({vx},{vy}) slice {st.sl}:")
            print(f"    Disk : kio={kv:7.1f}  rho={rv/1e3:6.0f}k  V={vv:.2f}")
            print(f"    Live : kio={top['kio']:7.1f}  "
                  f"rho={top['rho']/1e3:6.0f}k  V={top['V']:.2f}  "
                  f"SSE(lin)={top['sse_lin']:.5f}  "
                  f"SSE(log)={top['sse_log']:.5f}")

    def on_key(event):
        key = event.key or ""

        if key == "l":
            st.log_y = not st.log_y
            draw_signal()
            fig.canvas.draw_idle()
            print(f"  Signal y-axis → {'log' if st.log_y else 'linear'}")
            return

        if key == "h":
            print_help()
            return

        if key == "q":
            plt.close(fig)
            return

        if key == "m":
            names = [k for k in ["kio", "rho", "V", "residual"] if k in maps]
            ci = names.index(st.cur_map) if st.cur_map in names else -1
            st.cur_map = names[(ci + 1) % len(names)]
            draw_map(keep_view=True)
            update_header()
            draw_info()
            fig.canvas.draw_idle()
            print(f"  Map → {st.cur_map}")
            return

        if key == "s":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"viewer_{st.cur_map}_sl{st.sl}_vx{st.x}_{st.y}_{ts}.png"
            p = os.path.join(args.madi_dir, fname)
            fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {p}")
            return

        if key == "pageup":
            change_slice(st.sl + 1); return
        if key == "pagedown":
            change_slice(st.sl - 1); return

        if not st.selected:
            return

        # Δ cycle
        if key == "right" and st.matches:
            st.di_show = (st.di_show + 1) % n_fit
            draw_signal(); fig.canvas.draw_idle(); return
        if key == "left" and st.matches:
            st.di_show = (st.di_show - 1) % n_fit
            draw_signal(); fig.canvas.draw_idle(); return

        # Match cycle
        if key == "up" and st.matches:
            st.mi = max(0, st.mi - 1)
            draw_signal(); draw_table(); fig.canvas.draw_idle(); return
        if key == "down" and st.matches:
            st.mi = min(len(st.matches) - 1, st.mi + 1)
            draw_signal(); draw_table(); fig.canvas.draw_idle(); return

        # Ctrl+arrow move
        if "ctrl" not in key:
            return
        step = 5 if "shift" in key else 1
        base = key.replace("shift+", "").replace("ctrl+", "")
        dx = dy = 0
        if   base == "right": dx =  step
        elif base == "left":  dx = -step
        elif base == "up":    dy =  step
        elif base == "down":  dy = -step
        else: return

        nvx = int(np.clip(st.x + dx, 0, nx - 1))
        nvy = int(np.clip(st.y + dy, 0, ny - 1))
        if nvx != st.x or nvy != st.y:
            st.x, st.y = nvx, nvy
            px, py = vx_to_disp(nvx, nvy)
            crosshair_artist[0].set_data([px], [py])
            update_voxel(nvx, nvy)

    def print_help():
        print("""
  ┌────────────────────────────────────────────────────────┐
  │            MADI VIEWER — CONTROLS                      │
  ├────────────────────────────────────────────────────────┤
  │  Click              Select voxel                       │
  │  ← / →              Cycle Δ in signal plot             │
  │  ↑ / ↓              Cycle top-10 library matches       │
  │  Ctrl + Arrows      Nudge selected voxel by 1          │
  │  Ctrl+Shift+Arrows  Nudge voxel by 5                   │
  │  PageUp / PageDown  Previous / next slice              |
  │  l                  Toggle signal y-axis log/linear    │
  │  m                  Cycle map (kio → ρ → V → residual) │
  │  s                  Save screenshot                    │
  │  h                  Print this help                    │
  │  q                  Quit                               │
  └────────────────────────────────────────────────────────┘
""")

    # ================================================================
    # LAUNCH
    # ================================================================
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # First draw
    draw_map(keep_view=False)
    draw_info()
    draw_signal()
    draw_table()
    update_header()

    # Console summary
    mask_sl_init = get_mask_slice(st.sl)
    n_m = int(mask_sl_init.sum())
    n_s = int(np.any(get_slice_2d(signal_vol, ax_dir, st.sl) > 0, axis=-1).sum())
    print(f"\n  Starting slice {st.sl}: {n_m} mask voxels, {n_s} with signal")
    print_help()
    plt.show()


if __name__ == "__main__":
    main()

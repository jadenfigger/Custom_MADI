#!/usr/bin/env python3
"""
MADI Interactive Viewer — multi-run comparison edition
=======================================================

Click voxels on parametric maps to inspect library matches, observed vs
predicted signals, and parameter rankings.

NEW IN THIS VERSION
-------------------
* Multiple output directories with --madi-dirs, cycle with 'c'
* Live Rician noise-bias correction toggle on observed signals ('r')
* Live S0-averaging toggle on observed signals ('a')
* All-Δ overlay mode in signal plot ('d') — see signal vs Δ at once
* S0-fit and S0-ratio maps shown when present (cycle with 'm')
* Per-Δ S0 diagnostic panel for the selected voxel
* Linear / log y-axis toggle ('o')
* Sigma auto-estimated from background of first volume; override
  with --noise-sigma

Usage:
    python madi_viewer.py \\
        --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \\
        --mask mask.nii.gz \\
        --library madi_library_default.npz \\
        --madi-dirs out_baseline out_rician_avgs0 out_fits0 \\
        --labels baseline rician+avgS0 fitS0 \\
        --slice 3
"""

import argparse, os, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
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

COLORS = ["#22c55e", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444",
          "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1"]
DELTA_COLORS = ["#1d4ed8", "#059669", "#d97706", "#dc2626"]  # for all-Δ overlay
OBS_COLOR = "#1f2937"

MAP_CMAPS  = {"kio": "inferno", "rho": "viridis", "V": "plasma",
              "residual": "magma",
              "s0_fit": "cividis", "s0_fit_over_measured": "RdBu_r"}
MAP_LABELS = {"kio": "k_io (s⁻¹)", "rho": "ρ (cells/μL)", "V": "V (pL)",
              "residual": "SSE", "s0_fit": "Fitted S₀",
              "s0_fit_over_measured": "S₀_fit / S₀_measured"}
MAP_ORDER = ["kio", "rho", "V", "residual", "s0_fit", "s0_fit_over_measured"]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.labelsize': 10, 'axes.linewidth': 0.8,
    'axes.edgecolor': '#d1d5db', 'axes.facecolor': '#fafafa',
    'axes.grid': True, 'grid.alpha': 0.3,
    'legend.fontsize': 8, 'legend.framealpha': 0.9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.facecolor': 'white',
})

# ============================================================
# HELPERS
# ============================================================

def get_slice_2d(vol, axis, idx):
    s = [slice(None)] * vol.ndim
    s[axis] = idx
    return vol[tuple(s)]


def rician_correct(M, sigma):
    """E[M^2] = A^2 + 2 sigma^2  =>  A = sqrt(max(M^2 - 2 sigma^2, 0))."""
    A2 = M.astype(np.float64)**2 - 2.0 * sigma**2
    return np.sqrt(np.clip(A2, 0.0, None))


def estimate_sigma_from_bg(b0_vol, brain_mask):
    """Rayleigh: sigma = sqrt(mean(M^2)/2) over true-zero-signal voxels."""
    bg = ~brain_mask
    bg_vals = b0_vol[bg]
    bg_vals = bg_vals[bg_vals > 0]
    if len(bg_vals) < 100:
        return None
    return float(np.sqrt(np.mean(bg_vals.astype(np.float64)**2) / 2.0))


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="MADI Interactive Viewer")

    # Back-compat: accept --madi-dir (single) OR --madi-dirs (multi)
    ap.add_argument("--madi-dir", default=None,
                    help="(legacy) single output directory")
    ap.add_argument("--madi-dirs", nargs="+", default=None,
                    help="One or more output directories to compare. "
                         "Cycle with 'c' in the viewer.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Friendly labels for each --madi-dirs entry")

    ap.add_argument("--library", default="madi_library.npz")
    ap.add_argument("--input", type=str, nargs="+", required=True,
                    help="'delta:path' pairs (e.g. 15:dwi15.nii.gz)")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--axis", type=int, default=2,
                    help="0=sag, 1=cor, 2=axial (default)")
    ap.add_argument("--map", default="kio",
                    choices=MAP_ORDER)
    ap.add_argument("--margin", type=float, default=1.5,
                    help="Auto-zoom margin around brain (mm)")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="Rician noise std.  If omitted, auto-estimated "
                         "from background of first scan's b=0 volume.")
    args = ap.parse_args()

    # ---- Resolve madi dirs ----
    if args.madi_dirs:
        madi_dirs = list(args.madi_dirs)
    elif args.madi_dir:
        madi_dirs = [args.madi_dir]
    else:
        madi_dirs = ["madi_output"]
    if args.labels:
        if len(args.labels) != len(madi_dirs):
            print("ERROR: --labels must match length of --madi-dirs"); return
        dir_labels = list(args.labels)
    else:
        dir_labels = [os.path.basename(d.rstrip("/")) for d in madi_dirs]

    # ---- Parse DWI inputs ----
    dwi_inputs = []
    for s in args.input:
        d, p = s.split(":", 1)
        dwi_inputs.append((float(d), p))
    dwi_inputs.sort()
    fit_deltas = [d for d, _ in dwi_inputs]
    n_fit = len(fit_deltas)

    print("=" * 60)
    print("MADI Interactive Viewer (multi-run)")
    print("=" * 60)
    print(f"  Output dirs: {dir_labels}")

    # ---- Load mask first (for sigma estimation) ----
    print("\nLoading mask ...")
    mask_img = nib.load(args.mask)
    mask = mask_img.get_fdata().astype(bool)
    zooms = np.array(mask_img.header.get_zooms()[:3], dtype=float)
    print(f"  Voxel sizes: {zooms[0]:.4f} × {zooms[1]:.4f} × {zooms[2]:.4f} mm")

    # ---- Load maps from EACH madi dir ----
    print("\nLoading parametric maps ...")
    # maps_per_run[run_idx][name] -> 3D ndarray
    maps_per_run = []
    for d, label in zip(madi_dirs, dir_labels):
        run_maps = {}
        for name in MAP_ORDER:
            path = os.path.join(d, f"{name}_map.nii.gz")
            if os.path.exists(path):
                run_maps[name] = nib.load(path).get_fdata()
        maps_per_run.append(run_maps)
        print(f"  [{label}]  found: {sorted(run_maps.keys())}")

    if not any(maps_per_run):
        print("ERROR: No maps found in any directory"); return

    # ---- Load library ----
    print("\nLoading library ...")
    library = load_library(args.library)
    meta = load_library_meta(args.library)
    lib_deltas = meta['deltas']
    n_b = meta['n_b']
    lib_kios = np.array([e.kio for e in library])
    lib_rhos = np.array([e.rho for e in library])
    lib_Vs   = np.array([e.V for e in library])
    lib_vecs = np.array([e.vector for e in library])
    print(f"  {len(library)} entries, lib Δ = {lib_deltas} ms")

    fit_di = []
    for d in fit_deltas:
        for i, ld in enumerate(lib_deltas):
            if abs(d - ld) < 0.01:
                fit_di.append(i); break
        else:
            print(f"ERROR: Δ={d}ms not in library"); return
    lib_sub = np.hstack([lib_vecs[:, di*n_b:(di+1)*n_b] for di in fit_di])
    print(f"  Fit Δ: {fit_deltas} ms (indices {fit_di})")

    # ---- Load raw DWI volumes (keep ALL volumes for live recomputation) ----
    print("\nLoading DWI volumes (full 4D for live toggles) ...")
    ref_shape = mask.shape

    # raw_per_delta[di] = full 4D volume (X,Y,Z,97)
    raw_per_delta = []
    for di, (delta_ms, path) in enumerate(dwi_inputs):
        print(f"  Δ={delta_ms:.0f}ms: {os.path.basename(path)} ... ",
              end="", flush=True)
        data = nib.load(path).get_fdata().astype(np.float32)
        raw_per_delta.append(data)
        print(f"shape={data.shape}")

    # ---- Sigma estimation ----
    if args.noise_sigma is not None:
        sigma = float(args.noise_sigma)
        print(f"\n  Rician sigma (user)       = {sigma:.2f}")
    else:
        sigma = estimate_sigma_from_bg(raw_per_delta[0], mask)
        if sigma is None:
            print("\n  WARNING: Could not auto-estimate sigma; "
                  "Rician toggle disabled.")
            sigma = 0.0
        else:
            b0_brain_med = float(np.median(raw_per_delta[0][..., 0][mask]))
            print(f"\n  Rician sigma (auto, bg)   = {sigma:.2f}")
            print(f"  median brain b=0 signal   = {b0_brain_med:.1f}")
            print(f"  median brain b=0 SNR      = {b0_brain_med/sigma:.1f}")

    # ---- Slice setup ----
    ax_dir = args.axis
    sl_idx = args.slice if args.slice is not None else ref_shape[ax_dir] // 2

    mask_sl = get_slice_2d(mask, ax_dir, sl_idx)
    nx, ny = mask_sl.shape
    print(f"\n  Slice {sl_idx} (axis {ax_dir}): {nx} × {ny} voxels")

    axes_2d = [i for i in range(3) if i != ax_dir]
    zx = zooms[axes_2d[0]]
    zy = zooms[axes_2d[1]]
    W = nx * zx
    H = ny * zy
    extent = [0, W, 0, H]
    print(f"  Physical size: {W:.1f} × {H:.1f} mm")

    # ---- Coordinate transforms ----
    def disp_to_vx(xphys, yphys):
        c = int(xphys / zx); r = int(yphys / zy)
        vx = np.clip(c, 0, nx - 1); vy = np.clip(ny - 1 - r, 0, ny - 1)
        return int(vx), int(vy)

    def vx_to_disp(vx, vy):
        return (vx + 0.5) * zx, (ny - 1 - vy + 0.5) * zy

    # ---- Map / signal access ----
    def get_map_slice(run_idx, name):
        if name not in maps_per_run[run_idx]:
            return None
        return get_slice_2d(maps_per_run[run_idx][name], ax_dir, sl_idx)

    def get_map_val(run_idx, name, vx, vy):
        if name not in maps_per_run[run_idx]:
            return None
        m = maps_per_run[run_idx][name]
        if ax_dir == 0:   return m[sl_idx, vx, vy]
        elif ax_dir == 1: return m[vx, sl_idx, vy]
        else:             return m[vx, vy, sl_idx]

    def get_voxel_4d(di, vx, vy):
        """Return all 97 raw volumes for one (vx,vy) at given Δ index."""
        if ax_dir == 0:   return raw_per_delta[di][sl_idx, vx, vy, :]
        elif ax_dir == 1: return raw_per_delta[di][vx, sl_idx, vy, :]
        else:             return raw_per_delta[di][vx, vy, sl_idx, :]

    def compute_observed(vx, vy, apply_rician, avg_s0):
        """Compute (n_fit * n_b,) S/S0 vector live, with optional
        Rician correction and S0 averaging.  Also returns the per-Δ
        S0 values used (for diagnostic display)."""
        # Per-Δ raw 97-vectors
        raw_vecs = [get_voxel_4d(di, vx, vy).astype(np.float64)
                    for di in range(n_fit)]
        if apply_rician and sigma > 0:
            raw_vecs = [rician_correct(v, sigma) for v in raw_vecs]

        s0_each = np.array([v[0] for v in raw_vecs])
        if avg_s0:
            s0_used = np.full(n_fit, np.mean(s0_each))
        else:
            s0_used = s0_each.copy()
        s0_used_safe = np.where(s0_used < 1e-10, 1e-10, s0_used)

        out = np.zeros(n_fit * N_SHELLS)
        for di in range(n_fit):
            for si, (b_val, vol_sl) in enumerate(SHELLS):
                shell_mean = float(np.mean(raw_vecs[di][vol_sl]))
                out[di * N_SHELLS + si] = np.clip(
                    shell_mean / s0_used_safe[di], 0, 2.0)
        return out, s0_each, s0_used

    # ---- Auto-zoom ----
    mask_rot = np.rot90(mask_sl)
    rows_m, cols_m = np.where(mask_rot)
    if len(rows_m) > 0:
        xmin_p = cols_m.min() * zx - args.margin
        xmax_p = (cols_m.max() + 1) * zx + args.margin
        ymin_p = rows_m.min() * zy - args.margin
        ymax_p = (rows_m.max() + 1) * zy + args.margin
    else:
        xmin_p, xmax_p = 0, W
        ymin_p, ymax_p = 0, H

    # Sanity check
    test_vx, test_vy = nx // 2, ny // 2
    test_xp, test_yp = vx_to_disp(test_vx, test_vy)
    back_vx, back_vy = disp_to_vx(test_xp, test_yp)
    assert back_vx == test_vx and back_vy == test_vy
    print(f"  Coordinate transform verified ✓")

    # ---- State ----
    class S:
        x = nx // 2
        y = ny // 2
        selected = False
        cur_map = args.map
        run_idx = 0          # which output dir is shown
        di_show = 0
        mi = 0
        matches = None
        # Live toggles
        rician_on = False
        avg_s0_on = False
        all_deltas = False   # overlay all Δ in signal plot
        log_y = True         # log-scale y axis
    st = S()

    # If chosen map isn't in run 0, fall back to first available
    while st.cur_map not in maps_per_run[st.run_idx]:
        idx = MAP_ORDER.index(st.cur_map)
        st.cur_map = MAP_ORDER[(idx + 1) % len(MAP_ORDER)]

    # ================================================================
    # FIGURE LAYOUT
    # ================================================================
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.0, 1.4, 0.6],
        height_ratios=[1.0, 1.0, 0.5],
        left=0.04, right=0.985, top=0.93, bottom=0.05,
        wspace=0.22, hspace=0.42,
    )
    ax_img    = fig.add_subplot(gs[0:2, 0])     # main map
    ax_signal = fig.add_subplot(gs[0,   1])     # observed vs predicted
    ax_table  = fig.add_subplot(gs[1,   1])     # top-N table
    ax_s0     = fig.add_subplot(gs[2,   0])     # per-Δ S0 bar
    ax_status = fig.add_subplot(gs[2,   1])     # toggles / status box
    ax_runs   = fig.add_subplot(gs[:,   2])     # cross-run param comparison

    cbar_ax = fig.add_axes([0.355, 0.42, 0.010, 0.30])
    crosshair_artist = [None]

    # ================================================================
    # DRAW MAIN MAP
    # ================================================================
    def draw_map():
        ax_img.clear()
        black_bg = np.zeros((ny, nx))
        ax_img.imshow(black_bg, cmap="gray", origin="lower",
                      extent=extent, aspect="equal", vmin=0, vmax=1)

        sl = get_map_slice(st.run_idx, st.cur_map)
        if sl is None:
            ax_img.set_title(f"[{dir_labels[st.run_idx]}]  "
                             f"map '{st.cur_map}' not available", fontsize=10)
            ax_img.set_xlim(xmin_p, xmax_p); ax_img.set_ylim(ymin_p, ymax_p)
            return

        sl_rot = np.rot90(sl)
        sl_masked = np.ma.masked_where(
            np.rot90(~mask_sl) | (sl_rot == 0), sl_rot)
        cmap = MAP_CMAPS.get(st.cur_map, "viridis")

        # For ratio map, center at 1.0
        if st.cur_map == "s0_fit_over_measured":
            v = sl_masked.compressed()
            if len(v):
                spread = max(abs(np.percentile(v, 5) - 1.0),
                             abs(np.percentile(v, 95) - 1.0), 0.1)
                vmin, vmax = 1.0 - spread, 1.0 + spread
            else:
                vmin, vmax = 0.5, 1.5
            im = ax_img.imshow(sl_masked, cmap=cmap, origin="lower",
                               extent=extent, aspect="equal",
                               vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            im = ax_img.imshow(sl_masked, cmap=cmap, origin="lower",
                               extent=extent, aspect="equal",
                               interpolation="nearest")

        ax_img.set_xlim(xmin_p, xmax_p)
        ax_img.set_ylim(ymin_p, ymax_p)

        ax_img.set_title(
            f"[{dir_labels[st.run_idx]}]   slice {sl_idx}   —   {st.cur_map}\n"
            f"[c] cycle run    [m] cycle map",
            fontsize=10, pad=6)
        ax_img.set_xlabel("mm"); ax_img.set_ylabel("mm")
        ax_img.tick_params(labelsize=7)

        cbar_ax.clear()
        try:
            vmin = float(np.nanmin(sl_masked))
            vmax = float(np.nanmax(sl_masked))
        except (ValueError, TypeError):
            vmin, vmax = 0, 1
        if vmin == vmax: vmax = vmin + 1
        if st.cur_map == "s0_fit_over_measured":
            v = sl_masked.compressed()
            if len(v):
                spread = max(abs(np.percentile(v, 5) - 1.0),
                             abs(np.percentile(v, 95) - 1.0), 0.1)
                vmin, vmax = 1.0 - spread, 1.0 + spread
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(MAP_LABELS.get(st.cur_map, ""), fontsize=8)

        crosshair_artist[0], = ax_img.plot([], [], 'w+', ms=14, mew=2.5)
        if st.selected:
            px, py = vx_to_disp(st.x, st.y)
            crosshair_artist[0].set_data([px], [py])

    # ================================================================
    # MATCHING (uses the LIVE-recomputed observed signals)
    # ================================================================
    def find_top(measured):
        if np.all(measured < 1e-10):
            return None
        s_floor = 1e-3
        vi_min = 0.5     # paper default; matches updated library.py
        vi_max = 0.95

        vis = (lib_rhos / 1e9) * (lib_Vs * 1e3)
        valid_mask = (vis >= vi_min) & (vis <= vi_max)

        meas_log = np.log(np.clip(measured, s_floor, 1.0))
        lib_log  = np.log(np.clip(lib_sub,   s_floor, 1.0))

        dists = np.full(len(lib_sub), np.inf)
        dists[valid_mask] = np.sum(
            (lib_log[valid_mask] - meas_log[np.newaxis, :]) ** 2, axis=1)

        order = np.argsort(dists)[:N_TOP]
        matches = []
        for rank, idx in enumerate(order):
            if dists[idx] == np.inf: break
            matches.append({
                'rank': rank + 1,
                'kio':  lib_kios[idx],
                'rho':  lib_rhos[idx],
                'V':    lib_Vs[idx],
                'residual': dists[idx],
                'pred': lib_sub[idx],
            })
        return matches if matches else None

    # ================================================================
    # DRAW SIGNAL PLOT
    # ================================================================
    def draw_signal(measured):
        ax_signal.clear()
        if not st.matches:
            ax_signal.set_title(f"Voxel ({st.x},{st.y}) — no matches",
                                fontsize=10)
            return

        m = st.matches[st.mi]

        if st.all_deltas:
            # ---- Overlay all Δ ----
            for di in range(n_fit):
                obs_d  = measured[di * n_b : (di + 1) * n_b]
                pred_d = m['pred'][di * n_b : (di + 1) * n_b]
                col = DELTA_COLORS[di % len(DELTA_COLORS)]
                ax_signal.plot(BVALS_DISPLAY, obs_d, "o", color=col,
                               ms=8, mec="white", mew=0.8,
                               label=f"obs Δ={fit_deltas[di]:.0f}")
                ax_signal.plot(BVALS_DISPLAY, pred_d, "--", color=col,
                               lw=1.6, alpha=0.9,
                               label=f"sim Δ={fit_deltas[di]:.0f}")
            title_extra = "all Δ overlaid"
        else:
            # ---- Single Δ with top-N library cloud ----
            di = st.di_show % n_fit
            obs  = measured[di * n_b : (di + 1) * n_b]
            pred = m['pred'][di * n_b : (di + 1) * n_b]

            for j, om in enumerate(st.matches):
                if j == st.mi: continue
                op = om['pred'][di * n_b : (di + 1) * n_b]
                ax_signal.plot(BVALS_DISPLAY, op, "o-",
                               color=COLORS[j % len(COLORS)],
                               ms=3, lw=0.8, alpha=0.20, zorder=2)

            ax_signal.scatter(BVALS_DISPLAY, obs, c=OBS_COLOR, s=80, zorder=6,
                              label="Observed", edgecolors="white",
                              linewidths=1.0)
            c = COLORS[st.mi % len(COLORS)]
            ax_signal.plot(BVALS_DISPLAY, pred, "o-", color=c, ms=10, lw=2.5,
                           alpha=0.9, zorder=5,
                           label=f"#{m['rank']}  kio={m['kio']:.0f}  "
                                 f"ρ={m['rho']/1e3:.0f}k  V={m['V']:.2f}")
            title_extra = f"Δ = {fit_deltas[di]:.0f} ms   [←/→]"

        # Live #1
        top = st.matches[0]
        toggles = []
        if st.rician_on: toggles.append("RICIAN")
        if st.avg_s0_on: toggles.append("AVG-S0")
        toggles_s = " | ".join(toggles) if toggles else "raw"

        live_line = (f"Live #1: kio={top['kio']:.1f}  "
                     f"ρ={top['rho']/1e3:.0f}k  V={top['V']:.2f}  "
                     f"SSE={top['residual']:.4f}")

        # Disk maps for current run
        kv = get_map_val(st.run_idx, "kio", st.x, st.y)
        rv = get_map_val(st.run_idx, "rho", st.x, st.y)
        vv = get_map_val(st.run_idx, "V",   st.x, st.y)
        if kv is not None and rv is not None and vv is not None:
            disk_line = (f"From [{dir_labels[st.run_idx]}]: "
                         f"kio={kv:.1f}  ρ={rv/1e3:.0f}k  V={vv:.2f}")
            disagree = (abs(top['kio'] - kv) > 0.5 or
                        abs(top['rho'] - rv) / max(rv, 1) > 0.05 or
                        abs(top['V']   - vv) > 0.05)
            if disagree:
                disk_line += "   ⚠ live differs from disk"
        else:
            disk_line = f"[{dir_labels[st.run_idx]}]: maps unavailable"

        ax_signal.set_title(
            f"Voxel ({st.x},{st.y})   {title_extra}   "
            f"toggles: {toggles_s}\n"
            f"{live_line}\n{disk_line}",
            fontsize=9)
        ax_signal.set_xlabel("b-value  (s/mm²)")
        ax_signal.set_ylabel("S(b) / S₀")
        ax_signal.set_xlim(600, 6800)

        if st.log_y:
            ax_signal.set_yscale("log")
            ax_signal.set_ylim(0.01, 1.5)
        else:
            ax_signal.set_yscale("linear")
            ymax = max(1.05, float(np.max(measured)) * 1.15)
            ax_signal.set_ylim(-0.03, ymax)

        ax_signal.legend(loc="lower left", frameon=True, fontsize=7, ncol=2)

    # ================================================================
    # DRAW TABLE
    # ================================================================
    def draw_table():
        ax_table.clear(); ax_table.set_axis_off()
        if not st.matches: return

        cols = ["#", "k_io (s⁻¹)", "ρ (k/μL)", "V (pL)", "v_i", "SSE"]
        rows = []
        for m in st.matches:
            vi = (m['rho'] / 1e9) * (m['V'] * 1e3)
            rows.append([
                f"{m['rank']}", f"{m['kio']:.1f}",
                f"{m['rho']/1e3:.0f}", f"{m['V']:.2f}",
                f"{vi:.3f}", f"{m['residual']:.5f}",
            ])

        cell_colors = []
        for i in range(len(rows)):
            if i == st.mi:
                cell_colors.append([COLORS[i % len(COLORS)] + "30"] * len(cols))
            else:
                cell_colors.append(
                    ["white" if i % 2 == 0 else "#f9fafb"] * len(cols))

        tbl = ax_table.table(cellText=rows, colLabels=cols,
                             cellLoc='center', loc='center',
                             cellColours=cell_colors)
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.30)

        for j in range(len(cols)):
            cell = tbl[0, j]
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_edgecolor('#374151')
        for j in range(len(cols)):
            cell = tbl[st.mi + 1, j]
            cell.set_edgecolor(COLORS[st.mi % len(COLORS)])
            cell.set_text_props(fontweight='bold')
        for i in range(len(rows)):
            for j in range(len(cols)):
                if i != st.mi:
                    tbl[i + 1, j].set_edgecolor('#e5e7eb')

        ax_table.set_title(
            f"Top {len(st.matches)} library matches   [↑/↓ select]",
            fontsize=9, color='#374151', pad=4)

    # ================================================================
    # DRAW PER-Δ S0 BAR
    # ================================================================
    def draw_s0(s0_each, s0_used):
        ax_s0.clear()
        if s0_each is None:
            ax_s0.set_title("Per-Δ S₀ (click voxel)", fontsize=9)
            ax_s0.set_axis_off(); return

        x = np.arange(n_fit)
        bar_colors = [DELTA_COLORS[i % len(DELTA_COLORS)] for i in range(n_fit)]
        ax_s0.bar(x, s0_each, color=bar_colors, alpha=0.7,
                  edgecolor='white', linewidth=1.0, label='per-Δ b=0')

        # Horizontal line for S0 actually used
        if st.avg_s0_on:
            ax_s0.axhline(s0_used[0], color='#dc2626', linestyle='--',
                          lw=1.8, label=f"avg S₀ = {s0_used[0]:.1f}")

        # Show Rician-corrected vs raw if relevant
        rician_label = "Rician-corrected" if st.rician_on else "raw magnitude"
        cv = float(np.std(s0_each) / max(np.mean(s0_each), 1e-10) * 100)

        ax_s0.set_xticks(x)
        ax_s0.set_xticklabels([f"Δ={d:.0f}" for d in fit_deltas], fontsize=8)
        ax_s0.set_ylabel("S₀ value", fontsize=8)
        ax_s0.set_title(f"Per-Δ S₀ ({rician_label})   "
                        f"CV = {cv:.1f}%   "
                        f"{'⚠ high' if cv > 5 else 'ok'}",
                        fontsize=9, color='#374151')
        ax_s0.legend(loc='lower right', fontsize=7)
        ax_s0.grid(axis='y', alpha=0.3)

    # ================================================================
    # DRAW STATUS / TOGGLES
    # ================================================================
    def draw_status():
        ax_status.clear(); ax_status.set_axis_off()

        lines = ["LIVE TOGGLES",
                 f"  [r] Rician correct : {'ON' if st.rician_on else 'off'}"
                 f"   (σ = {sigma:.2f})",
                 f"  [a] Average S₀     : {'ON' if st.avg_s0_on else 'off'}",
                 f"  [d] All-Δ overlay  : {'ON' if st.all_deltas else 'off'}",
                 f"  [o] Log y-axis     : {'ON' if st.log_y else 'off'}",
                 "",
                 "RUN COMPARISON",
                 f"  [c] Active run : {dir_labels[st.run_idx]}  "
                 f"({st.run_idx+1}/{len(madi_dirs)})",
                 f"  [m] Active map : {st.cur_map}",
                 "",
                 "VOXEL & SHORTCUTS",
                 f"  Selected voxel : ({st.x}, {st.y})",
                 "  [click] select  [↑/↓] match  [←/→] Δ",
                 "  [Ctrl+arrows] move 1 voxel",
                 "  [s] screenshot   [h] help",
                 ]
        ax_status.text(0.02, 0.98, "\n".join(lines),
                       transform=ax_status.transAxes,
                       fontsize=8.5, family='monospace',
                       verticalalignment='top',
                       bbox=dict(facecolor='#f3f4f6', edgecolor='#d1d5db',
                                 boxstyle='round,pad=0.5'))

    # ================================================================
    # DRAW CROSS-RUN PARAMETER COMPARISON
    # ================================================================
    def draw_runs():
        """Bar chart comparing kio/rho/V at the selected voxel
        across all loaded madi_dirs."""
        ax_runs.clear()
        if not st.selected or len(madi_dirs) < 1:
            ax_runs.set_title("Cross-run params\n(click a voxel)",
                              fontsize=9)
            ax_runs.set_axis_off()
            return

        # Collect values
        kio_vals, rho_vals, V_vals, res_vals = [], [], [], []
        for ri in range(len(madi_dirs)):
            kio_vals.append(get_map_val(ri, "kio", st.x, st.y) or np.nan)
            rho_vals.append(get_map_val(ri, "rho", st.x, st.y) or np.nan)
            V_vals.append(get_map_val(ri, "V",   st.x, st.y) or np.nan)
            res_vals.append(get_map_val(ri, "residual", st.x, st.y) or np.nan)

        # Normalize each parameter to its max for display on common axes
        kio_arr = np.array(kio_vals, dtype=float)
        rho_arr = np.array(rho_vals, dtype=float) / 1e3
        V_arr   = np.array(V_vals,   dtype=float)

        n_runs = len(madi_dirs)
        y_pos = np.arange(n_runs)

        # Build a small text table instead of overlapping bars
        ax_runs.set_axis_off()
        lines = [f"Voxel ({st.x},{st.y}) cross-run\n"]
        header = f"{'run':<14}{'kio':>6}{'ρ_k':>7}{'V':>6}{'SSE':>9}"
        lines.append(header)
        lines.append("─" * len(header))
        for ri, lab in enumerate(dir_labels):
            mark = "▶" if ri == st.run_idx else " "
            lines.append(
                f"{mark}{lab[:13]:<13}"
                f"{kio_arr[ri]:>6.1f}"
                f"{rho_arr[ri]:>7.0f}"
                f"{V_arr[ri]:>6.2f}"
                f"{res_vals[ri]:>9.4f}"
            )

        ax_runs.text(0.02, 0.98, "\n".join(lines),
                     transform=ax_runs.transAxes,
                     fontsize=8.5, family='monospace',
                     verticalalignment='top',
                     bbox=dict(facecolor='#fff7ed', edgecolor='#fed7aa',
                               boxstyle='round,pad=0.5'))
        ax_runs.set_title("Cross-run comparison", fontsize=9, pad=4)

    # ================================================================
    # UPDATE
    # ================================================================
    def update_voxel():
        measured, s0_each, s0_used = compute_observed(
            st.x, st.y, st.rician_on, st.avg_s0_on)
        if np.all(measured < 1e-10):
            st.matches = None
            ax_signal.clear()
            ax_signal.set_title(f"Voxel ({st.x},{st.y}) — no signal",
                                fontsize=10)
            ax_table.clear(); ax_table.set_axis_off()
            draw_s0(None, None)
        else:
            st.matches = find_top(measured)
            st.mi = 0
            draw_signal(measured)
            draw_table()
            draw_s0(s0_each, s0_used)
        draw_status()
        draw_runs()
        fig.canvas.draw_idle()

    def redraw_signal_only():
        """Re-derive observed signal (live toggles changed) and redraw."""
        if not st.selected: return
        measured, s0_each, s0_used = compute_observed(
            st.x, st.y, st.rician_on, st.avg_s0_on)
        if np.all(measured < 1e-10):
            return
        st.matches = find_top(measured)
        st.mi = 0
        draw_signal(measured)
        draw_table()
        draw_s0(s0_each, s0_used)
        draw_status()
        fig.canvas.draw_idle()

    # ================================================================
    # EVENTS
    # ================================================================
    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None: return
        vx, vy = disp_to_vx(event.xdata, event.ydata)
        in_mask = bool(mask_sl[vx, vy])
        print(f"  Click ({event.xdata:.1f}, {event.ydata:.1f}) mm "
              f"→ voxel ({vx},{vy})  in_mask={in_mask}")
        if not in_mask: return
        st.x, st.y = vx, vy
        st.selected = True
        px, py = vx_to_disp(vx, vy)
        if crosshair_artist[0] is not None:
            crosshair_artist[0].set_data([px], [py])
        update_voxel()

    def on_key(event):
        key = event.key or ''

        if key == 'h':
            print_help(); return

        if key == 'm':
            avail = [k for k in MAP_ORDER if k in maps_per_run[st.run_idx]]
            if not avail: return
            ci = avail.index(st.cur_map) if st.cur_map in avail else -1
            st.cur_map = avail[(ci + 1) % len(avail)]
            draw_map(); fig.canvas.draw_idle()
            print(f"  Map → {st.cur_map}"); return

        if key == 'c':
            st.run_idx = (st.run_idx + 1) % len(madi_dirs)
            # Make sure cur_map exists in new run
            if st.cur_map not in maps_per_run[st.run_idx]:
                avail = [k for k in MAP_ORDER if k in maps_per_run[st.run_idx]]
                if avail: st.cur_map = avail[0]
            draw_map()
            if st.selected:
                draw_runs()
                # Redraw signal too — disk-map readout depends on run
                measured, s0_each, s0_used = compute_observed(
                    st.x, st.y, st.rician_on, st.avg_s0_on)
                if not np.all(measured < 1e-10):
                    st.matches = find_top(measured)
                    st.mi = 0
                    draw_signal(measured)
                    draw_table()
            draw_status()
            fig.canvas.draw_idle()
            print(f"  Run → {dir_labels[st.run_idx]}"); return

        if key == 'r':
            if sigma <= 0:
                print("  Cannot toggle Rician: sigma not available."); return
            st.rician_on = not st.rician_on
            print(f"  Rician correction → {st.rician_on}")
            redraw_signal_only(); return

        if key == 'a':
            st.avg_s0_on = not st.avg_s0_on
            print(f"  Average S0 → {st.avg_s0_on}")
            redraw_signal_only(); return

        if key == 'd':
            st.all_deltas = not st.all_deltas
            print(f"  All-Δ overlay → {st.all_deltas}")
            if st.selected:
                measured, _, _ = compute_observed(
                    st.x, st.y, st.rician_on, st.avg_s0_on)
                draw_signal(measured)
                draw_status()
                fig.canvas.draw_idle()
            return

        if key == 'o':
            st.log_y = not st.log_y
            print(f"  Log y → {st.log_y}")
            if st.selected:
                measured, _, _ = compute_observed(
                    st.x, st.y, st.rician_on, st.avg_s0_on)
                draw_signal(measured)
                draw_status()
                fig.canvas.draw_idle()
            return

        if key == 's':
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = madi_dirs[st.run_idx]
            p = os.path.join(outdir,
                f"viewer_{dir_labels[st.run_idx]}_{st.x}_{st.y}_{ts}.png")
            fig.savefig(p, dpi=180, bbox_inches='tight')
            print(f"  Saved: {p}"); return

        if not st.selected: return

        # Signal navigation
        if key == 'right' and st.matches and not st.all_deltas:
            st.di_show = (st.di_show + 1) % n_fit
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); fig.canvas.draw_idle(); return
        if key == 'left' and st.matches and not st.all_deltas:
            st.di_show = (st.di_show - 1) % n_fit
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); fig.canvas.draw_idle(); return

        if key == 'up' and st.matches:
            st.mi = max(0, st.mi - 1)
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); draw_table(); fig.canvas.draw_idle(); return
        if key == 'down' and st.matches:
            st.mi = min(len(st.matches) - 1, st.mi + 1)
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); draw_table(); fig.canvas.draw_idle(); return

        # Ctrl+arrows move voxel
        if 'ctrl' not in key: return
        step = 5 if 'shift' in key else 1
        base = key.replace('shift+', '').replace('ctrl+', '')
        dx = dy = 0
        if   base == 'right': dx =  step
        elif base == 'left':  dx = -step
        elif base == 'up':    dy =  step
        elif base == 'down':  dy = -step
        else: return
        nvx = int(np.clip(st.x + dx, 0, nx - 1))
        nvy = int(np.clip(st.y + dy, 0, ny - 1))
        if nvx != st.x or nvy != st.y:
            st.x, st.y = nvx, nvy
            px, py = vx_to_disp(nvx, nvy)
            if crosshair_artist[0] is not None:
                crosshair_artist[0].set_data([px], [py])
            print(f"  Move → voxel ({nvx},{nvy})")
            update_voxel()

    def print_help():
        print("""
    ┌──────────────────────────────────────────────────────────┐
    │            MADI VIEWER — CONTROLS                        │
    ├──────────────────────────────────────────────────────────┤
    │  Click               Select voxel                        │
    │  ← / →               Cycle Δ in signal plot              │
    │  ↑ / ↓               Cycle top-N library matches         │
    │  Ctrl + Arrows       Move voxel ±1   (+Shift = ±5)       │
    │                                                          │
    │  m                   Cycle parametric map                │
    │  c                   Cycle output run (multi-dir mode)   │
    │                                                          │
    │  r                   Toggle live Rician correction       │
    │  a                   Toggle live S₀ averaging across Δ   │
    │  d                   Toggle all-Δ overlay in signal plot │
    │  o                   Toggle linear / log y-axis          │
    │                                                          │
    │  s                   Save screenshot                     │
    │  h                   Print this help                     │
    └──────────────────────────────────────────────────────────┘
        """)

    # ================================================================
    # INITIAL DRAW
    # ================================================================
    draw_map()
    ax_signal.set_title("Click a voxel to begin", fontsize=11)
    ax_signal.set_xlabel("b-value (s/mm²)"); ax_signal.set_ylabel("S(b) / S₀")
    ax_table.set_axis_off()
    draw_s0(None, None)
    draw_status()
    draw_runs()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print_help()
    print(f"Δ = {fit_deltas} ms, {len(library)} library entries")
    print(f"Auto-zoom: x=[{xmin_p:.1f}, {xmax_p:.1f}] "
          f"y=[{ymin_p:.1f}, {ymax_p:.1f}] mm")
    plt.show()


if __name__ == "__main__":
    main()

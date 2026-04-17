#!/usr/bin/env python3
"""
MADI Interactive Viewer  [v3 — diagnostic edition]
====================================================

Click voxels on parametric maps to inspect library matches, observed vs
predicted signals, residuals, and cross-run comparisons.

NEW IN v3
---------
* Redesigned 4-column layout with dedicated panels for every diagnostic
* Residual plot: per-Δ (obs−pred)/pred % error vs b-value
* Map histogram: brain-mask value distribution + voxel percentile marker
* Cross-run comparison redesigned as visual bar charts (4 panels: kio/ρ/V/SSE)
* Diff map mode [v]: show run_A − run_B signed difference map
* Slice navigation [z/x]: step through slices without restarting
* Confidence envelope: top-N match spread shaded in signal plot
* Full crosshair (H+V lines) instead of point marker
* Suptitle strip: always-visible run/map/slice/toggle state
* Voxel intracellular volume fraction (v_i) displayed prominently
* SNR estimate shown in S0 panel
* Voxel percentile rank in brain distribution
* [p] key: print detailed voxel report to terminal
* [v] key: toggle diff map mode
* [z/x] keys: previous/next slice

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
from matplotlib.gridspec import GridSpecFromSubplotSpec

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

# Color palettes
RUN_COLORS   = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
                "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"]
MATCH_COLORS = ["#22c55e", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444",
                "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1"]
DELTA_COLORS = ["#1d4ed8", "#059669", "#d97706", "#dc2626"]
OBS_COLOR    = "#1f2937"

MAP_CMAPS  = {"kio": "inferno", "rho": "viridis", "V": "plasma",
              "residual": "magma",
              "s0_fit": "cividis", "s0_fit_over_measured": "RdBu_r"}
MAP_LABELS = {"kio": "k_io (s⁻¹)", "rho": "ρ (cells/μL)", "V": "V (pL)",
              "residual": "SSE", "s0_fit": "Fitted S₀",
              "s0_fit_over_measured": "S₀_fit / S₀_measured"}
MAP_ORDER  = ["kio", "rho", "V", "residual", "s0_fit", "s0_fit_over_measured"]
DIFF_CMAP  = "RdBu_r"

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
    'axes.labelsize': 8.5, 'axes.linewidth': 0.8,
    'axes.edgecolor': '#d1d5db', 'axes.facecolor': '#fafafa',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'legend.fontsize': 7.5, 'legend.framealpha': 0.92,
    'legend.borderpad': 0.4, 'legend.handlelength': 1.4,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'figure.facecolor': '#f1f5f9',
})

# ============================================================
# HELPERS
# ============================================================

def get_slice_2d(vol, axis, idx):
    s = [slice(None)] * vol.ndim
    s[axis] = idx
    return vol[tuple(s)]


def rician_correct(M, sigma):
    """E[M²] = A² + 2σ²  =>  A = sqrt(max(M² − 2σ², 0))."""
    A2 = M.astype(np.float64) ** 2 - 2.0 * sigma ** 2
    return np.sqrt(np.clip(A2, 0.0, None))


def estimate_sigma_from_bg(b0_vol, brain_mask):
    """Rayleigh estimator from background voxels."""
    bg_vals = b0_vol[~brain_mask]
    bg_vals = bg_vals[bg_vals > 0]
    if len(bg_vals) < 100:
        return None
    return float(np.sqrt(np.mean(bg_vals.astype(np.float64) ** 2) / 2.0))


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="MADI Interactive Viewer v3")
    ap.add_argument("--madi-dir",    default=None,
                    help="(legacy) single output directory")
    ap.add_argument("--madi-dirs",   nargs="+", default=None,
                    help="Output directories to compare. Cycle with 'c'.")
    ap.add_argument("--labels",      nargs="+", default=None)
    ap.add_argument("--library",     default="madi_library.npz")
    ap.add_argument("--input",       type=str, nargs="+", required=True,
                    help="'delta:path' pairs (e.g. 15:dwi15.nii.gz)")
    ap.add_argument("--mask",        required=True)
    ap.add_argument("--slice",       type=int, default=None)
    ap.add_argument("--axis",        type=int, default=2,
                    help="0=sag, 1=cor, 2=axial (default)")
    ap.add_argument("--map",         default="kio", choices=MAP_ORDER)
    ap.add_argument("--margin",      type=float, default=1.5)
    ap.add_argument("--noise-sigma", type=float, default=None)
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
            print("ERROR: --labels length must match --madi-dirs"); return
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

    print("=" * 62)
    print("MADI Interactive Viewer v3  (diagnostic edition)")
    print("=" * 62)
    print(f"  Runs: {dir_labels}")

    # ---- Load mask ----
    print("\nLoading mask ...")
    mask_img = nib.load(args.mask)
    mask     = mask_img.get_fdata().astype(bool)
    zooms    = np.array(mask_img.header.get_zooms()[:3], dtype=float)
    print(f"  Voxel sizes: {zooms[0]:.4f} × {zooms[1]:.4f} × {zooms[2]:.4f} mm")

    # ---- Load parametric maps ----
    print("\nLoading parametric maps ...")
    maps_per_run = []
    for d, label in zip(madi_dirs, dir_labels):
        rm = {}
        for name in MAP_ORDER:
            path = os.path.join(d, f"{name}_map.nii.gz")
            if os.path.exists(path):
                rm[name] = nib.load(path).get_fdata()
        maps_per_run.append(rm)
        print(f"  [{label}]  found: {sorted(rm.keys())}")
    if not any(maps_per_run):
        print("ERROR: No maps found in any directory"); return

    # ---- Load library ----
    print("\nLoading library ...")
    library    = load_library(args.library)
    meta       = load_library_meta(args.library)
    lib_deltas = meta['deltas']
    n_b        = meta['n_b']           # should equal N_SHELLS
    lib_kios   = np.array([e.kio    for e in library])
    lib_rhos   = np.array([e.rho    for e in library])
    lib_Vs     = np.array([e.V      for e in library])
    lib_vecs   = np.array([e.vector for e in library])
    print(f"  {len(library)} entries, lib Δ = {lib_deltas} ms")

    fit_di = []
    for d in fit_deltas:
        for i, ld in enumerate(lib_deltas):
            if abs(d - ld) < 0.01:
                fit_di.append(i); break
        else:
            print(f"ERROR: Δ={d}ms not in library"); return
    lib_sub = np.hstack([lib_vecs[:, di * n_b:(di + 1) * n_b] for di in fit_di])
    print(f"  Fit Δ: {fit_deltas} ms  (lib indices {fit_di})")

    # ---- Load raw DWI volumes ----
    print("\nLoading DWI volumes (full 4D for live toggles) ...")
    raw_per_delta = []
    for di, (delta_ms, path) in enumerate(dwi_inputs):
        print(f"  Δ={delta_ms:.0f}ms: {os.path.basename(path)} ... ", end="", flush=True)
        data = nib.load(path).get_fdata().astype(np.float32)
        raw_per_delta.append(data)
        print(f"shape={data.shape}")

    # ---- Sigma estimation ----
    if args.noise_sigma is not None:
        sigma = float(args.noise_sigma)
        print(f"\n  σ (user-supplied) = {sigma:.2f}")
    else:
        sigma = estimate_sigma_from_bg(raw_per_delta[0], mask)
        if sigma is None:
            print("\n  WARNING: σ not estimable — Rician toggle disabled.")
            sigma = 0.0
        else:
            b0_med = float(np.median(raw_per_delta[0][..., 0][mask]))
            print(f"\n  σ (auto, bg Rayleigh) = {sigma:.2f}")
            print(f"  median brain b=0       = {b0_med:.1f}")
            print(f"  brain b=0 SNR          = {b0_med / sigma:.1f}")

    # ---- Spatial setup ----
    ax_dir   = args.axis
    n_slices = mask.shape[ax_dir]
    init_sl  = args.slice if args.slice is not None else n_slices // 2

    axes_2d = [i for i in range(3) if i != ax_dir]
    zx = zooms[axes_2d[0]]
    zy = zooms[axes_2d[1]]
    nx = mask.shape[axes_2d[0]]
    ny = mask.shape[axes_2d[1]]
    W, H  = nx * zx, ny * zy
    extent = [0, W, 0, H]
    print(f"\n  Slices: 0–{n_slices-1}  (starting at {init_sl})")
    print(f"  In-plane: {nx}×{ny} voxels, {W:.1f}×{H:.1f} mm")

    # ---- Coordinate transforms ----
    def disp_to_vx(xp, yp):
        vx = np.clip(int(xp / zx), 0, nx - 1)
        vy = np.clip(ny - 1 - int(yp / zy), 0, ny - 1)
        return int(vx), int(vy)

    def vx_to_disp(vx, vy):
        return (vx + 0.5) * zx, (ny - 1 - vy + 0.5) * zy

    # ---- Data accessors (use st.sl_idx so slice changes take effect) ----
    def get_mask_sl():
        return get_slice_2d(mask, ax_dir, st.sl_idx)

    def get_map_slice(run_idx, name):
        if name not in maps_per_run[run_idx]: return None
        return get_slice_2d(maps_per_run[run_idx][name], ax_dir, st.sl_idx)

    def get_map_val(run_idx, name, vx, vy):
        if name not in maps_per_run[run_idx]: return None
        m = maps_per_run[run_idx][name]
        if   ax_dir == 0: return float(m[st.sl_idx, vx, vy])
        elif ax_dir == 1: return float(m[vx, st.sl_idx, vy])
        else:             return float(m[vx, vy, st.sl_idx])

    def get_voxel_4d(di, vx, vy):
        if   ax_dir == 0: return raw_per_delta[di][st.sl_idx, vx, vy, :]
        elif ax_dir == 1: return raw_per_delta[di][vx, st.sl_idx, vy, :]
        else:             return raw_per_delta[di][vx, vy, st.sl_idx, :]

    def compute_zoom():
        msl  = get_mask_sl()
        mrot = np.rot90(msl)
        rows, cols = np.where(mrot)
        if len(rows):
            return (cols.min() * zx - args.margin,
                    (cols.max() + 1) * zx + args.margin,
                    rows.min() * zy - args.margin,
                    (rows.max() + 1) * zy + args.margin)
        return 0.0, W, 0.0, H

    def compute_observed(vx, vy, apply_rician, avg_s0):
        raw = [get_voxel_4d(di, vx, vy).astype(np.float64) for di in range(n_fit)]
        if apply_rician and sigma > 0:
            raw = [rician_correct(v, sigma) for v in raw]
        s0_each = np.array([v[0] for v in raw])
        s0_used = np.full(n_fit, np.mean(s0_each)) if avg_s0 else s0_each.copy()
        s0_safe = np.where(s0_used < 1e-10, 1e-10, s0_used)
        out = np.zeros(n_fit * N_SHELLS)
        for di in range(n_fit):
            for si, (_, vol_sl) in enumerate(SHELLS):
                out[di * N_SHELLS + si] = np.clip(
                    float(np.mean(raw[di][vol_sl])) / s0_safe[di], 0, 2.0)
        return out, s0_each, s0_used

    # ---- Matching ----
    def find_top(measured):
        if np.all(measured < 1e-10): return None
        s_floor = 1e-3
        vis   = (lib_rhos / 1e9) * (lib_Vs * 1e3)
        valid = (vis >= 0.5) & (vis <= 0.95)
        ml  = np.log(np.clip(measured, s_floor, 1.0))
        ll  = np.log(np.clip(lib_sub,  s_floor, 1.0))
        d   = np.full(len(lib_sub), np.inf)
        d[valid] = np.sum((ll[valid] - ml[np.newaxis, :]) ** 2, axis=1)
        order = np.argsort(d)[:N_TOP]
        matches = []
        for rank, idx in enumerate(order):
            if d[idx] == np.inf: break
            matches.append({'rank': rank + 1, 'kio': lib_kios[idx],
                            'rho': lib_rhos[idx], 'V': lib_Vs[idx],
                            'residual': d[idx], 'pred': lib_sub[idx]})
        return matches or None

    # ---- State ----
    class S:
        x = nx // 2;  y = ny // 2
        selected    = False
        cur_map     = args.map
        run_idx     = 0
        di_show     = 0
        mi          = 0
        matches     = None
        rician_on   = False
        avg_s0_on   = False
        all_deltas  = False
        log_y       = True
        diff_mode   = False
        ref_run     = 0
        sl_idx      = init_sl
    st = S()

    # Ensure initial map exists in run 0
    while st.cur_map not in maps_per_run[st.run_idx]:
        idx = MAP_ORDER.index(st.cur_map)
        st.cur_map = MAP_ORDER[(idx + 1) % len(MAP_ORDER)]

    # ================================================================
    # FIGURE LAYOUT
    # ================================================================
    fig = plt.figure(figsize=(26, 13))
    fig.patch.set_facecolor('#f1f5f9')

    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1.25, 1.65, 1.1, 1.0],
        height_ratios=[1.9, 1.3, 0.75],
        left=0.04, right=0.987, top=0.92, bottom=0.05,
        wspace=0.30, hspace=0.52,
    )

    ax_img    = fig.add_subplot(gs[0:2, 0])   # Parametric map (tall)
    ax_signal = fig.add_subplot(gs[0,   1])   # Signal vs b-value
    ax_resid  = fig.add_subplot(gs[1,   1])   # Percent residuals  [NEW]
    ax_table  = fig.add_subplot(gs[0:2, 2])   # Library match table (full-height)
    ax_s0     = fig.add_subplot(gs[2,   0])   # Per-Δ S0 bars
    ax_hist   = fig.add_subplot(gs[2,   1])   # Brain map histogram  [NEW]
    ax_status = fig.add_subplot(gs[2,   2])   # Status / toggle panel

    # Cross-run comparison: 4 mini bar charts stacked in the right column
    gs_runs = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 3], hspace=0.75)
    ax_rkio  = fig.add_subplot(gs_runs[0])
    ax_rrho  = fig.add_subplot(gs_runs[1])
    ax_rV    = fig.add_subplot(gs_runs[2])
    ax_rsse  = fig.add_subplot(gs_runs[3])

    # Colorbar as inset of the map panel (survives ax_img.clear())
    cbar_ax = ax_img.inset_axes([0.02, 0.03, 0.055, 0.32])

    # Persistent crosshair artists — updated in-place so we skip imshow redraws
    # _ch = [h_line, v_line, center_point]  reset to None when ax_img is cleared
    _ch = [None, None, None]

    # Pre-compute histogram data for every (run, map) combination once at load
    print("\nPre-caching histogram data ...")
    _hist_cache = {}
    for _ri in range(len(madi_dirs)):
        for _name in MAP_ORDER:
            if _name in maps_per_run[_ri]:
                _v = maps_per_run[_ri][_name][mask].ravel()
                _v = _v[(_v > 0) & np.isfinite(_v)]
                if len(_v):
                    _lo, _hi = np.percentile(_v, [0.5, 99.5])
                    _hist_cache[(_ri, _name)] = (_v, float(_lo), float(_hi))
    print(f"  Cached {len(_hist_cache)} map/run histograms.")

    # ================================================================
    # SUPTITLE  — persistent header strip
    # ================================================================
    _supt = [None]

    def draw_suptitle():
        if _supt[0] is not None:
            try:
                _supt[0].remove()
            except (ValueError, AttributeError):
                pass
            _supt[0] = None
        bits = []
        if st.rician_on:  bits.append("RICIAN")
        if st.avg_s0_on:  bits.append("AVG-S0")
        if st.all_deltas: bits.append("ALL-Δ")
        if st.log_y:      bits.append("LOG")
        if st.diff_mode:  bits.append(f"DIFF(vs {dir_labels[st.ref_run][:6]})")
        toggle_s = " | ".join(bits) if bits else "raw"
        vox_s    = f"   voxel ({st.x},{st.y})" if st.selected else ""
        _supt[0] = fig.suptitle(
            f"MADI Viewer v3   ·   [{dir_labels[st.run_idx]}]   "
            f"map: {st.cur_map}   slice: {st.sl_idx}/{n_slices-1}"
            f"   σ={sigma:.1f}   {toggle_s}{vox_s}",
            fontsize=10.5, fontweight='bold', color='#1e293b',
            x=0.5, y=0.974,
            bbox=dict(facecolor='#e2e8f0', edgecolor='#94a3b8',
                      boxstyle='round,pad=0.28', alpha=0.85))

    # ================================================================
    # DRAW MAP
    # ================================================================
    def draw_map():
        ax_img.clear()
        ax_img.set_facecolor('#0f172a')
        msl  = get_mask_sl()
        zoom = compute_zoom()

        if st.diff_mode and len(madi_dirs) >= 2:
            sl_a = get_map_slice(st.run_idx, st.cur_map)
            sl_b = get_map_slice(st.ref_run,  st.cur_map)
            if sl_a is not None and sl_b is not None:
                diff    = sl_a - sl_b
                drot    = np.rot90(diff)
                dmasked = np.ma.masked_where(np.rot90(~msl), drot)
                v       = dmasked.compressed()
                if len(v):
                    spread = max(float(np.abs(np.percentile(v, [3, 97])).max()), 1e-9)
                    vmin_cb, vmax_cb = -spread, spread
                else:
                    vmin_cb, vmax_cb = -1.0, 1.0
                ax_img.imshow(dmasked, cmap=DIFF_CMAP, origin="lower",
                              extent=extent, aspect="equal",
                              vmin=vmin_cb, vmax=vmax_cb, interpolation="nearest")
                cmap_used  = DIFF_CMAP
                title_mode = (f"DIFF  [{dir_labels[st.run_idx]}] − "
                              f"[{dir_labels[st.ref_run]}]")
                label_used = f"Δ {MAP_LABELS.get(st.cur_map, '')}"
            else:
                ax_img.set_title("Diff not available for this map/run", fontsize=8.5)
                ax_img.set_xlim(*zoom[:2]); ax_img.set_ylim(*zoom[2:])
                return
        else:
            sl = get_map_slice(st.run_idx, st.cur_map)
            if sl is None:
                ax_img.set_title(
                    f"'{st.cur_map}' not found in [{dir_labels[st.run_idx]}]",
                    fontsize=8.5, color='#f8fafc')
                ax_img.set_xlim(*zoom[:2]); ax_img.set_ylim(*zoom[2:])
                return
            slr     = np.rot90(sl)
            smasked = np.ma.masked_where(np.rot90(~msl) | (slr == 0), slr)
            cmap_used = MAP_CMAPS.get(st.cur_map, "viridis")
            v = smasked.compressed()
            if st.cur_map == "s0_fit_over_measured":
                if len(v):
                    spread = max(abs(np.percentile(v, 5) - 1.0),
                                 abs(np.percentile(v, 95) - 1.0), 0.1)
                    vmin_cb, vmax_cb = 1.0 - spread, 1.0 + spread
                else:
                    vmin_cb, vmax_cb = 0.5, 1.5
            else:
                vmin_cb = float(np.nanpercentile(v, 1))  if len(v) else 0.0
                vmax_cb = float(np.nanpercentile(v, 99)) if len(v) else 1.0
                if vmin_cb == vmax_cb: vmax_cb = vmin_cb + 1.0
            ax_img.imshow(smasked, cmap=cmap_used, origin="lower",
                          extent=extent, aspect="equal",
                          vmin=vmin_cb, vmax=vmax_cb, interpolation="nearest")
            title_mode = MAP_LABELS.get(st.cur_map, st.cur_map)
            label_used = MAP_LABELS.get(st.cur_map, "")

        ax_img.set_xlim(*zoom[:2]); ax_img.set_ylim(*zoom[2:])
        ax_img.set_title(
            f"{title_mode}\n[m] map · [c] run · [v] diff · [z/x] slice",
            fontsize=8, pad=5, color='#cbd5e1')
        ax_img.set_xlabel("mm", fontsize=7.5, color='#64748b')
        ax_img.set_ylabel("mm", fontsize=7.5, color='#64748b')
        ax_img.tick_params(labelsize=7, colors='#94a3b8')
        for sp in ax_img.spines.values():
            sp.set_edgecolor('#334155')

        # Colorbar (inset_axes child — survives ax_img.clear())
        cbar_ax.clear()
        norm = plt.Normalize(vmin=vmin_cb, vmax=vmax_cb)
        sm   = plt.cm.ScalarMappable(cmap=cmap_used, norm=norm)
        cb   = fig.colorbar(sm, cax=cbar_ax)
        cb.ax.tick_params(labelsize=6, colors='#cbd5e1')
        cb.set_label(label_used, fontsize=6.5, color='#cbd5e1')
        cb.outline.set_edgecolor('#475569')

        # Full crosshair lines — reset cache; update_crosshair() adds them back
        _ch[0] = _ch[1] = _ch[2] = None
        if st.selected:
            update_crosshair()

    # ================================================================
    # CROSSHAIR  — update artists in-place (no imshow redraw needed)
    # ================================================================
    def update_crosshair():
        """Move crosshair to current voxel.  Creates artists on first call
        after a map redraw; otherwise updates existing artists in-place."""
        if not st.selected:
            return
        px, py = vx_to_disp(st.x, st.y)
        if _ch[0] is None:
            _ch[0] = ax_img.axhline(py, color='#f8fafc', lw=0.7, alpha=0.6, zorder=10)
            _ch[1] = ax_img.axvline(px, color='#f8fafc', lw=0.7, alpha=0.6, zorder=10)
            _ch[2], = ax_img.plot(px, py, '+', color='#fbbf24', ms=14, mew=2.5, zorder=11)
        else:
            _ch[0].set_ydata([py, py])
            _ch[1].set_xdata([px, px])
            _ch[2].set_data([px], [py])

    # ================================================================
    # DRAW SIGNAL PLOT
    # ================================================================
    def draw_signal(measured):
        ax_signal.clear()
        if not st.matches:
            ax_signal.set_axis_off()
            ax_signal.text(0.5, 0.5, "No matches", ha='center', va='center',
                           transform=ax_signal.transAxes, fontsize=10,
                           color='#94a3b8')
            return

        m = st.matches[st.mi]

        if st.all_deltas:
            for di in range(n_fit):
                obs_d  = measured[di * N_SHELLS:(di + 1) * N_SHELLS]
                pred_d = m['pred'][di * N_SHELLS:(di + 1) * N_SHELLS]
                col = DELTA_COLORS[di % len(DELTA_COLORS)]
                ax_signal.plot(BVALS_DISPLAY, obs_d, "o", color=col,
                               ms=8, mec="white", mew=0.8,
                               label=f"obs Δ={fit_deltas[di]:.0f}")
                ax_signal.plot(BVALS_DISPLAY, pred_d, "--", color=col,
                               lw=1.6, alpha=0.9,
                               label=f"sim Δ={fit_deltas[di]:.0f}")
            title_sfx = "all Δ overlay"
        else:
            di   = st.di_show % n_fit
            obs  = measured[di * N_SHELLS:(di + 1) * N_SHELLS]
            pred = m['pred'][di * N_SHELLS:(di + 1) * N_SHELLS]

            # Top-N confidence envelope
            if len(st.matches) > 1:
                all_p = np.array([om['pred'][di * N_SHELLS:(di + 1) * N_SHELLS]
                                  for om in st.matches])
                ax_signal.fill_between(BVALS_DISPLAY, all_p.min(0), all_p.max(0),
                                       alpha=0.11, color='#94a3b8',
                                       label=f"top-{len(st.matches)} range")

            # Faint cloud of other matches
            for j, om in enumerate(st.matches):
                if j == st.mi: continue
                ax_signal.plot(BVALS_DISPLAY,
                               om['pred'][di * N_SHELLS:(di + 1) * N_SHELLS],
                               "o-", color=MATCH_COLORS[j % len(MATCH_COLORS)],
                               ms=3, lw=0.7, alpha=0.17, zorder=2)

            # Best/selected match
            c  = MATCH_COLORS[st.mi % len(MATCH_COLORS)]
            vi = (m['rho'] / 1e9) * (m['V'] * 1e3)
            ax_signal.plot(BVALS_DISPLAY, pred, "o-", color=c,
                           ms=9, lw=2.5, alpha=0.9, zorder=5,
                           label=(f"#{m['rank']}  kio={m['kio']:.1f}  "
                                  f"ρ={m['rho']/1e3:.0f}k  "
                                  f"V={m['V']:.2f}  v_i={vi:.3f}"))

            # Observed
            ax_signal.scatter(BVALS_DISPLAY, obs, c=OBS_COLOR, s=80, zorder=6,
                              label="Observed", edgecolors="white", linewidths=1.0)
            title_sfx = f"Δ={fit_deltas[di]:.0f} ms  [←/→]"

        # Disk-map annotation box
        kv = get_map_val(st.run_idx, "kio", st.x, st.y)
        rv = get_map_val(st.run_idx, "rho", st.x, st.y)
        vv = get_map_val(st.run_idx, "V",   st.x, st.y)
        if all(v is not None for v in [kv, rv, vv]):
            vi_d = (rv / 1e9) * (vv * 1e3)
            top  = st.matches[0]
            diff_flag = (abs(top['kio'] - kv) > 0.5 or
                         abs(top['rho'] - rv) / max(rv, 1) > 0.05 or
                         abs(top['V']   - vv) > 0.05)
            ann_color = '#dc2626' if diff_flag else '#16a34a'
            ann_text  = (f"disk [{dir_labels[st.run_idx]}]:  "
                         f"kio={kv:.1f}  ρ={rv/1e3:.0f}k  "
                         f"V={vv:.2f}  v_i={vi_d:.3f}"
                         + ("  ⚠" if diff_flag else ""))
            ax_signal.annotate(ann_text, xy=(0.02, 0.04),
                               xycoords='axes fraction', fontsize=7,
                               color=ann_color,
                               bbox=dict(boxstyle='round,pad=0.25',
                                         facecolor='white', edgecolor=ann_color,
                                         alpha=0.88))

        ax_signal.set_title(f"Signal   ({st.x},{st.y})   {title_sfx}",
                            fontsize=9)
        ax_signal.set_xlabel("b-value (s/mm²)", fontsize=8)
        ax_signal.set_ylabel("S(b) / S₀", fontsize=8)
        ax_signal.set_xlim(600, 6800)
        if st.log_y:
            ax_signal.set_yscale("log"); ax_signal.set_ylim(0.01, 1.5)
        else:
            ax_signal.set_yscale("linear")
            ax_signal.set_ylim(-0.03, max(1.05, float(np.max(measured)) * 1.15))
        ax_signal.legend(loc="lower left", frameon=True, fontsize=7, ncol=1)

    # ================================================================
    # DRAW RESIDUALS  [new panel]
    # ================================================================
    def draw_resid(measured):
        ax_resid.clear()
        if not st.matches:
            ax_resid.set_axis_off()
            ax_resid.text(0.5, 0.5, "Residuals", ha='center', va='center',
                          transform=ax_resid.transAxes, fontsize=9,
                          color='#94a3b8')
            return

        m = st.matches[st.mi]
        ax_resid.axhline(0, color='#94a3b8', lw=1.0, zorder=1)
        ax_resid.fill_between([600, 6800], [-10, -10], [10, 10],
                              alpha=0.07, color='#22c55e', zorder=0)

        for di in range(n_fit):
            obs_d  = measured[di * N_SHELLS:(di + 1) * N_SHELLS]
            pred_d = m['pred'][di * N_SHELLS:(di + 1) * N_SHELLS]
            pred_s = np.where(pred_d < 1e-8, 1e-8, pred_d)
            pct    = (obs_d - pred_d) / pred_s * 100.0
            col    = DELTA_COLORS[di % len(DELTA_COLORS)]
            ax_resid.plot(BVALS_DISPLAY, pct, "o-", color=col,
                          ms=6, lw=1.4, alpha=0.9,
                          label=f"Δ={fit_deltas[di]:.0f}")

        sse = m['residual']
        ax_resid.set_title(
            f"% Residual  (obs−pred)/pred×100   match #{m['rank']}  SSE={sse:.4f}",
            fontsize=8.5)
        ax_resid.set_xlabel("b-value (s/mm²)", fontsize=8)
        ax_resid.set_ylabel("Residual (%)", fontsize=8)
        ax_resid.set_xlim(600, 6800)
        yl = max(30.0, float(np.max(np.abs(ax_resid.get_ylim()))))
        ax_resid.set_ylim(-yl, yl)
        ax_resid.legend(loc='upper right', fontsize=7, frameon=True, ncol=n_fit)

    # ================================================================
    # DRAW TABLE
    # ================================================================
    def draw_table():
        ax_table.clear(); ax_table.set_axis_off()
        if not st.matches: return

        cols = ["#", "k_io", "ρ (k/μL)", "V (pL)", "v_i", "SSE"]
        rows = []
        for m in st.matches:
            vi = (m['rho'] / 1e9) * (m['V'] * 1e3)
            rows.append([f"{m['rank']}",
                         f"{m['kio']:.1f}",
                         f"{m['rho']/1e3:.0f}",
                         f"{m['V']:.2f}",
                         f"{vi:.3f}",
                         f"{m['residual']:.4f}"])

        sses    = [m['residual'] for m in st.matches]
        sse_min = min(sses); sse_max = max(sses)

        cell_colors = []
        for i, m in enumerate(st.matches):
            if i == st.mi:
                c = MATCH_COLORS[i % len(MATCH_COLORS)]
                cell_colors.append([c + "28"] * len(cols))
            else:
                base = "white" if i % 2 == 0 else "#f9fafb"
                # SSE cell gets a white→red gradient
                frac = ((m['residual'] - sse_min) / (sse_max - sse_min)
                        if sse_max > sse_min else 0.0)
                r_val = 1.0
                g_val = 1.0 - frac * 0.65
                b_val = 1.0 - frac * 0.65
                row_c = [base] * (len(cols) - 1) + [(r_val, g_val, b_val)]
                cell_colors.append(row_c)

        tbl = ax_table.table(cellText=rows, colLabels=cols,
                             cellLoc='center', loc='upper center',
                             cellColours=cell_colors)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.35)

        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#1e293b")
            tbl[0, j].set_edgecolor('#334155')
            tbl[0, j].set_text_props(color='white', fontweight='bold')

        for j in range(len(cols)):
            tbl[st.mi + 1, j].set_edgecolor(MATCH_COLORS[st.mi % len(MATCH_COLORS)])
            tbl[st.mi + 1, j].set_linewidth(2.0)
            tbl[st.mi + 1, j].set_text_props(fontweight='bold')

        for i in range(len(rows)):
            for j in range(len(cols)):
                if i != st.mi:
                    tbl[i + 1, j].set_edgecolor('#e5e7eb')

        ax_table.set_title(f"Top {len(st.matches)} library matches  [↑/↓ select]",
                           fontsize=9, color='#374151', pad=4)

    # ================================================================
    # DRAW S0 BARS
    # ================================================================
    def draw_s0(s0_each, s0_used):
        ax_s0.clear()
        if s0_each is None:
            ax_s0.set_axis_off()
            ax_s0.text(0.5, 0.5, "Per-Δ S₀\n(click voxel)",
                       ha='center', va='center',
                       transform=ax_s0.transAxes, fontsize=9, color='#94a3b8')
            return

        x    = np.arange(n_fit)
        cols = [DELTA_COLORS[i % len(DELTA_COLORS)] for i in range(n_fit)]
        bars = ax_s0.bar(x, s0_each, color=cols, alpha=0.75,
                         edgecolor='white', linewidth=1.0)
        for bar, val in zip(bars, s0_each):
            ax_s0.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() * 1.02, f"{val:.0f}",
                       ha='center', va='bottom', fontsize=7)

        if st.avg_s0_on:
            ax_s0.axhline(s0_used[0], color='#dc2626', ls='--', lw=1.8,
                          label=f"avg={s0_used[0]:.0f}")
            ax_s0.legend(loc='lower right', fontsize=7)

        cv      = float(np.std(s0_each) / max(np.mean(s0_each), 1e-10) * 100)
        cv_col  = '#dc2626' if cv > 5 else '#16a34a'
        snr_str = f"  SNR≈{np.mean(s0_each)/sigma:.1f}" if sigma > 0 else ""

        ax_s0.set_xticks(x)
        ax_s0.set_xticklabels([f"Δ={d:.0f}" for d in fit_deltas], fontsize=7.5)
        ax_s0.set_ylabel("S₀", fontsize=8)
        ax_s0.set_ylim(0, max(s0_each) * 1.22)
        ax_s0.set_title(
            f"Per-Δ S₀{'  [Rician]' if st.rician_on else ''}  "
            f"CV={cv:.1f}%{snr_str}",
            fontsize=8.5, color=cv_col)
        ax_s0.grid(axis='y', alpha=0.25)

    # ================================================================
    # DRAW HISTOGRAM  [new panel]
    # ================================================================
    def draw_hist():
        ax_hist.clear()
        key = (st.run_idx, st.cur_map)
        if key not in _hist_cache:
            ax_hist.set_axis_off()
            ax_hist.text(0.5, 0.5, "no map data", ha='center', va='center',
                         transform=ax_hist.transAxes, fontsize=9, color='#94a3b8')
            return

        vals, lo, hi = _hist_cache[key]
        ax_hist.hist(vals, bins=60, range=(lo, hi),
                     color='#6366f1', alpha=0.70, edgecolor='none')

        if st.selected:
            vv = get_map_val(st.run_idx, st.cur_map, st.x, st.y)
            if vv is not None and np.isfinite(vv):
                pct = float(np.mean(vals < vv) * 100)
                ax_hist.axvline(vv, color='#f59e0b', lw=2.0, zorder=5,
                                label=f"voxel={vv:.2f}  ({pct:.0f}th %ile)")
                ax_hist.legend(loc='upper right', fontsize=7.5, frameon=True)

        ax_hist.set_title(
            f"{MAP_LABELS.get(st.cur_map, st.cur_map)}  ·  "
            f"brain dist. [{dir_labels[st.run_idx]}]",
            fontsize=8.5)
        ax_hist.set_xlabel(MAP_LABELS.get(st.cur_map, ""), fontsize=7.5)
        ax_hist.set_ylabel("count", fontsize=7.5)
        ax_hist.set_xlim(lo, hi)
        ax_hist.grid(axis='y', alpha=0.2)

    # ================================================================
    # DRAW STATUS PANEL
    # ================================================================
    def draw_status():
        ax_status.clear(); ax_status.set_axis_off()
        ax_status.set_xlim(0, 1); ax_status.set_ylim(0, 1)

        y = 0.97

        def row(key, label, state, hint=""):
            nonlocal y
            on_col  = '#16a34a' if state else '#94a3b8'
            st_str  = 'ON ' if state else 'off'
            ax_status.text(0.04, y, f'[{key}]', fontsize=7.5, va='top',
                           family='monospace', color='#64748b',
                           transform=ax_status.transAxes)
            ax_status.text(0.22, y, label, fontsize=7.5, va='top',
                           family='monospace', color='#374151',
                           transform=ax_status.transAxes)
            ax_status.text(0.66, y, st_str, fontsize=7.5, va='top',
                           family='monospace', color=on_col, fontweight='bold',
                           transform=ax_status.transAxes)
            if hint:
                ax_status.text(0.82, y, hint, fontsize=6.5, va='top',
                               color='#94a3b8', transform=ax_status.transAxes)
            y -= 0.125

        ax_status.text(0.04, y, "TOGGLES", fontsize=8, fontweight='bold',
                       va='top', family='monospace', color='#1e293b',
                       transform=ax_status.transAxes)
        y -= 0.13

        row('r', 'Rician  ', st.rician_on,  f"σ={sigma:.1f}")
        row('a', 'Avg S₀  ', st.avg_s0_on)
        row('d', 'All-Δ   ', st.all_deltas)
        row('o', 'Log-Y   ', st.log_y)
        row('v', 'Diff    ', st.diff_mode,
            f"vs [{dir_labels[st.ref_run][:5]}]" if st.diff_mode else "")

        y -= 0.03
        ax_status.text(0.04, y, "VOXEL", fontsize=8, fontweight='bold',
                       va='top', family='monospace', color='#1e293b',
                       transform=ax_status.transAxes)
        y -= 0.13
        vox_s = f"({st.x},{st.y})" if st.selected else "none"
        ax_status.text(0.04, y, f"  {vox_s}", fontsize=7.5, va='top',
                       family='monospace', color='#374151',
                       transform=ax_status.transAxes)
        y -= 0.13
        ax_status.text(0.04, y, "  [s]save  [p]report  [h]help",
                       fontsize=6.8, va='top', family='monospace',
                       color='#64748b', transform=ax_status.transAxes)

        bg = '#f0fdf4' if not st.diff_mode else '#fff7ed'
        ax_status.add_patch(
            plt.Rectangle((0, 0), 1, 1, transform=ax_status.transAxes,
                           facecolor=bg, edgecolor='#cbd5e1', linewidth=1.0,
                           zorder=-1))

    # ================================================================
    # DRAW CROSS-RUN BAR CHARTS  [completely redesigned]
    # ================================================================
    def draw_runs():
        specs = [
            (ax_rkio, "kio",      "k_io  (s⁻¹)", 1.0,   "{:.1f}"),
            (ax_rrho, "rho",      "ρ  (k/μL)",   1e-3,  "{:.0f}"),
            (ax_rV,   "V",        "V  (pL)",      1.0,   "{:.2f}"),
            (ax_rsse, "residual", "SSE",           1.0,   "{:.4f}"),
        ]

        # Column header above the top bar chart
        ax_rkio.annotate(
            f"Cross-run  ({st.x},{st.y})" if st.selected else "Cross-run",
            xy=(0.5, 1.24), xycoords='axes fraction',
            ha='center', fontsize=9, fontweight='bold', color='#1e293b',
            annotation_clip=False)

        for ax, mapkey, title, scale, fmt in specs:
            ax.clear()

            if not st.selected:
                ax.set_axis_off()
                ax.text(0.5, 0.5, title, ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='#94a3b8')
                continue

            vals  = []
            for ri in range(len(madi_dirs)):
                v = get_map_val(ri, mapkey, st.x, st.y)
                vals.append((v * scale) if v is not None else np.nan)

            y_pos = np.arange(len(madi_dirs))
            for i, (v, lab) in enumerate(zip(vals, dir_labels)):
                if np.isnan(v): continue
                col   = RUN_COLORS[i % len(RUN_COLORS)]
                alpha = 1.0 if i == st.run_idx else 0.55
                ax.barh(y_pos[i], v, color=col, alpha=alpha,
                        edgecolor='white', linewidth=0.8, height=0.6)
                if i == st.run_idx:
                    ax.barh(y_pos[i], v, fill=False,
                            edgecolor='#f59e0b', linewidth=2.0, height=0.6)
                ax.text(v * 1.04 if v >= 0 else v * 0.96, y_pos[i],
                        fmt.format(v), va='center', fontsize=6.5,
                        color='#374151')

            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                xmax = max(valid) * 1.40
                ax.set_xlim(0, max(xmax, 1e-9))

            short = [lb[:10] for lb in dir_labels]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(short, fontsize=6.5)
            ax.set_title(title, fontsize=8, pad=2, color='#374151')
            ax.tick_params(axis='x', labelsize=6.5)
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('#fafafa')

    # ================================================================
    # UPDATE HELPERS
    # ================================================================
    def _clear_voxel_panels():
        for ax in [ax_signal, ax_resid]:
            ax.clear(); ax.set_axis_off()
        ax_table.clear(); ax_table.set_axis_off()

    def update_voxel():
        measured, s0_each, s0_used = compute_observed(
            st.x, st.y, st.rician_on, st.avg_s0_on)
        if np.all(measured < 1e-10):
            st.matches = None
            _clear_voxel_panels()
            ax_signal.text(0.5, 0.5, f"Voxel ({st.x},{st.y}) — no signal",
                           ha='center', va='center',
                           transform=ax_signal.transAxes, fontsize=10,
                           color='#94a3b8')
            draw_s0(None, None)
        else:
            st.matches = find_top(measured)
            st.mi = 0
            draw_signal(measured)
            draw_table()
            draw_s0(s0_each, s0_used)
            draw_resid(measured)
        draw_hist()
        draw_runs()
        draw_status()
        draw_suptitle()
        fig.canvas.draw_idle()

    def redraw_signal_only():
        if not st.selected: return
        measured, s0_each, s0_used = compute_observed(
            st.x, st.y, st.rician_on, st.avg_s0_on)
        if np.all(measured < 1e-10): return
        st.matches = find_top(measured)
        st.mi = 0
        draw_signal(measured)
        draw_table()
        draw_s0(s0_each, s0_used)
        draw_resid(measured)
        draw_status()
        draw_suptitle()
        fig.canvas.draw_idle()

    def change_slice(new_sl):
        if new_sl == st.sl_idx: return
        st.sl_idx  = new_sl
        st.selected = False
        st.matches  = None
        draw_map()
        _clear_voxel_panels()
        ax_signal.text(0.5, 0.5, f"Slice {st.sl_idx} — click a voxel",
                       ha='center', va='center',
                       transform=ax_signal.transAxes, fontsize=10,
                       color='#94a3b8')
        draw_s0(None, None)
        draw_hist()
        draw_runs()
        draw_status()
        draw_suptitle()
        fig.canvas.draw_idle()
        print(f"  Slice → {st.sl_idx}")

    # ================================================================
    # EVENTS
    # ================================================================
    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None: return
        vx, vy   = disp_to_vx(event.xdata, event.ydata)
        msl      = get_mask_sl()
        if not bool(msl[vx, vy]): return
        st.x, st.y = vx, vy
        st.selected = True
        print(f"  Click → voxel ({vx},{vy})")
        update_voxel()
        update_crosshair()   # just moves the crosshair lines — no imshow redraw
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key or ''

        if key == 'h':
            print_help(); return

        if key == 'm':
            avail = [k for k in MAP_ORDER if k in maps_per_run[st.run_idx]]
            if not avail: return
            ci = avail.index(st.cur_map) if st.cur_map in avail else -1
            st.cur_map = avail[(ci + 1) % len(avail)]
            draw_map(); draw_hist(); draw_suptitle()
            fig.canvas.draw_idle()
            print(f"  Map → {st.cur_map}"); return

        if key == 'c':
            st.run_idx = (st.run_idx + 1) % len(madi_dirs)
            if st.cur_map not in maps_per_run[st.run_idx]:
                avail = [k for k in MAP_ORDER if k in maps_per_run[st.run_idx]]
                if avail: st.cur_map = avail[0]
            draw_map()
            if st.selected:
                measured, s0_each, s0_used = compute_observed(
                    st.x, st.y, st.rician_on, st.avg_s0_on)
                if not np.all(measured < 1e-10):
                    st.matches = find_top(measured)
                    st.mi = 0
                    draw_signal(measured)
                    draw_table()
                    draw_resid(measured)
            draw_hist(); draw_runs(); draw_status(); draw_suptitle()
            fig.canvas.draw_idle()
            print(f"  Run → {dir_labels[st.run_idx]}"); return

        if key == 'v':
            if len(madi_dirs) < 2:
                print("  Diff mode requires ≥ 2 dirs."); return
            st.diff_mode = not st.diff_mode
            print(f"  Diff mode → {st.diff_mode}")
            draw_map(); draw_status(); draw_suptitle()
            fig.canvas.draw_idle(); return

        if key in ('z', 'minus', 'pagedown'):
            change_slice(max(0, st.sl_idx - 1)); return

        if key in ('x', 'equal', 'pageup'):
            change_slice(min(n_slices - 1, st.sl_idx + 1)); return

        if key == 'r':
            if sigma <= 0:
                print("  Rician toggle unavailable (σ not estimated)."); return
            st.rician_on = not st.rician_on
            print(f"  Rician → {st.rician_on}")
            redraw_signal_only(); return

        if key == 'a':
            st.avg_s0_on = not st.avg_s0_on
            print(f"  Avg S0 → {st.avg_s0_on}")
            redraw_signal_only(); return

        if key == 'd':
            st.all_deltas = not st.all_deltas
            print(f"  All-Δ → {st.all_deltas}")
            if st.selected:
                measured, _, _ = compute_observed(
                    st.x, st.y, st.rician_on, st.avg_s0_on)
                draw_signal(measured); draw_resid(measured)
            draw_status(); draw_suptitle()
            fig.canvas.draw_idle(); return

        if key == 'o':
            st.log_y = not st.log_y
            print(f"  Log-Y → {st.log_y}")
            if st.selected:
                measured, _, _ = compute_observed(
                    st.x, st.y, st.rician_on, st.avg_s0_on)
                draw_signal(measured)
            draw_status(); draw_suptitle()
            fig.canvas.draw_idle(); return

        if key == 's':
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = madi_dirs[st.run_idx]
            p   = os.path.join(out,
                  f"viewer_{dir_labels[st.run_idx]}_{st.x}_{st.y}_{ts}.png")
            fig.savefig(p, dpi=180, bbox_inches='tight')
            print(f"  Saved: {p}"); return

        if key == 'p':
            # Detailed voxel report to terminal
            if not st.selected:
                print("  No voxel selected."); return
            print(f"\n{'='*60}")
            print(f"  VOXEL REPORT  ({st.x},{st.y})  slice={st.sl_idx}")
            print(f"{'='*60}")
            for ri, lab in enumerate(dir_labels):
                kv = get_map_val(ri, "kio",      st.x, st.y)
                rv = get_map_val(ri, "rho",      st.x, st.y)
                vv = get_map_val(ri, "V",        st.x, st.y)
                sv = get_map_val(ri, "residual", st.x, st.y)
                if all(v is not None for v in [kv, rv, vv]):
                    vi_v = (rv / 1e9) * (vv * 1e3)
                    print(f"  [{lab}]  kio={kv:.2f}  ρ={rv/1e3:.1f}k  "
                          f"V={vv:.3f}  v_i={vi_v:.4f}  SSE={sv:.5f}")
                else:
                    print(f"  [{lab}]  (maps not available)")
            if st.matches:
                print(f"\n  Live top-{len(st.matches)} matches"
                      f"  (rician={st.rician_on}, avgS0={st.avg_s0_on}):")
                print(f"  {'#':>3}  {'kio':>6}  {'ρ(k)':>7}  "
                      f"{'V':>6}  {'v_i':>6}  {'SSE':>10}")
                for m in st.matches:
                    vi_v = (m['rho'] / 1e9) * (m['V'] * 1e3)
                    print(f"  {m['rank']:>3}  {m['kio']:>6.1f}  "
                          f"{m['rho']/1e3:>7.0f}  {m['V']:>6.2f}  "
                          f"{vi_v:>6.3f}  {m['residual']:>10.5f}")
            print(f"{'='*60}\n")
            return

        if not st.selected: return

        # ---- Signal navigation ----
        if key == 'right' and st.matches and not st.all_deltas:
            st.di_show = (st.di_show + 1) % n_fit
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); draw_resid(measured)
            fig.canvas.draw_idle(); return

        if key == 'left' and st.matches and not st.all_deltas:
            st.di_show = (st.di_show - 1) % n_fit
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); draw_resid(measured)
            fig.canvas.draw_idle(); return

        if key == 'up' and st.matches:
            st.mi = max(0, st.mi - 1)
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); draw_table(); draw_resid(measured)
            fig.canvas.draw_idle(); return

        if key == 'down' and st.matches:
            st.mi = min(len(st.matches) - 1, st.mi + 1)
            measured, _, _ = compute_observed(
                st.x, st.y, st.rician_on, st.avg_s0_on)
            draw_signal(measured); draw_table(); draw_resid(measured)
            fig.canvas.draw_idle(); return

        # ---- Ctrl+arrows: move voxel ----
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
            print(f"  Move → ({nvx},{nvy})")
            update_voxel()
            update_crosshair()   # just moves crosshair — no imshow redraw
            fig.canvas.draw_idle()

    def print_help():
        print("""
    ┌──────────────────────────────────────────────────────────────┐
    │              MADI VIEWER v3  —  CONTROLS                     │
    ├──────────────────────────────────────────────────────────────┤
    │  Click               Select voxel                            │
    │  ← / →               Cycle Δ in signal / residual plots      │
    │  ↑ / ↓               Cycle top-N library matches             │
    │  Ctrl + Arrows       Move voxel ±1   (+Shift = ±5)           │
    │                                                              │
    │  m                   Cycle parametric map                    │
    │  c                   Cycle output run (multi-dir mode)       │
    │  z / x               Previous / next slice                   │
    │                                                              │
    │  r                   Toggle Rician noise correction          │
    │  a                   Toggle S₀ averaging across Δ            │
    │  d                   Toggle all-Δ overlay in signal plot     │
    │  o                   Toggle linear / log y-axis              │
    │  v                   Toggle diff map (run A − run B)         │
    │                                                              │
    │  s                   Save screenshot (PNG)                   │
    │  p                   Print full voxel report to terminal     │
    │  h                   Print this help                         │
    └──────────────────────────────────────────────────────────────┘
        """)

    # ================================================================
    # INITIAL DRAW
    # ================================================================
    draw_map()
    ax_signal.set_axis_off()
    ax_signal.text(0.5, 0.5, "Click a voxel to begin",
                   ha='center', va='center', transform=ax_signal.transAxes,
                   fontsize=12, color='#94a3b8')
    ax_resid.set_axis_off()
    ax_resid.text(0.5, 0.5, "Residuals", ha='center', va='center',
                  transform=ax_resid.transAxes, fontsize=10, color='#94a3b8')
    ax_table.set_axis_off()
    draw_s0(None, None)
    draw_hist()
    draw_runs()
    draw_status()
    draw_suptitle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print_help()
    print(f"Δ = {fit_deltas} ms  ·  {len(library)} library entries")
    print(f"Slices: 0–{n_slices-1}   starting at: {st.sl_idx}")
    plt.show()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MADI Interactive Viewer
========================
Click voxels on parametric maps to inspect library matches,
observed vs predicted signals, and parameter rankings.

Automatically corrects aspect ratio from NIfTI voxel sizes and
auto-zooms to the brain region.

Usage:
    python madi_viewer.py \\
        --input 15:dwi15.nii.gz 25:dwi25.nii.gz 30:dwi30.nii.gz 40:dwi40.nii.gz \\
        --mask mask.nii.gz \\
        --library madi_library_default.npz \\
        --madi-dir madi_output_v2 \\
        --slice 3

Controls:
    Click           Select voxel
    ← / →           Cycle Δ in signal plot
    ↑ / ↓           Cycle top-10 matches
    Ctrl+Arrows     Move selected voxel
    m               Cycle map (kio/ρ/V/residual)
    s               Save screenshot
    h               Print help
"""

import argparse, os, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
OBS_COLOR = "#1f2937"

MAP_CMAPS  = {"kio": "inferno", "rho": "viridis", "V": "plasma", "residual": "magma"}
MAP_LABELS = {"kio": "k_io (s⁻¹)", "rho": "ρ (cells/μL)", "V": "V (pL)", "residual": "SSE"}

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

# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="MADI Interactive Viewer")
    ap.add_argument("--madi-dir", default="madi_output")
    ap.add_argument("--library", default="madi_library.npz")
    ap.add_argument("--input", type=str, nargs="+", required=True,
                    help="'delta:path' pairs (e.g. 15:dwi15.nii.gz)")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--axis", type=int, default=2,
                    help="0=sag, 1=cor, 2=axial (default)")
    ap.add_argument("--map", default="kio",
                    choices=["kio", "rho", "V", "residual"])
    ap.add_argument("--margin", type=float, default=1.5,
                    help="Auto-zoom margin around brain (mm)")
    args = ap.parse_args()

    # ---- Parse DWI inputs ----
    dwi_inputs = []
    for s in args.input:
        d, p = s.split(":", 1)
        dwi_inputs.append((float(d), p))
    dwi_inputs.sort()
    fit_deltas = [d for d, _ in dwi_inputs]
    n_fit = len(fit_deltas)

    print("=" * 60)
    print("MADI Interactive Viewer")
    print("=" * 60)

    # ---- Load maps ----
    print("\nLoading maps ...")
    maps = {}
    zooms = None
    for name in ["kio", "rho", "V", "residual"]:
        path = os.path.join(args.madi_dir, f"{name}_map.nii.gz")
        if os.path.exists(path):
            img = nib.load(path)
            maps[name] = img.get_fdata()
            if zooms is None:
                zooms = np.array(img.header.get_zooms()[:3], dtype=float)
            print(f"  {name}: shape={maps[name].shape}")
    if not maps:
        print("ERROR: No maps found"); return

    # ---- Load mask ----
    print("Loading mask ...")
    mask_img = nib.load(args.mask)
    mask = mask_img.get_fdata().astype(bool)
    if zooms is None:
        zooms = np.array(mask_img.header.get_zooms()[:3], dtype=float)
    print(f"  Voxel sizes: {zooms[0]:.4f} × {zooms[1]:.4f} × {zooms[2]:.4f} mm")

    # ---- Load library ----
    print("Loading library ...")
    library = load_library(args.library)
    meta = load_library_meta(args.library)
    lib_deltas = meta['deltas']
    n_b = meta['n_b']
    lib_kios = np.array([e.kio for e in library])
    lib_rhos = np.array([e.rho for e in library])
    lib_Vs   = np.array([e.V for e in library])
    lib_vecs = np.array([e.vector for e in library])
    print(f"  {len(library)} entries, lib Δ = {lib_deltas} ms")

    # Map fit deltas → library indices
    fit_di = []
    for d in fit_deltas:
        for i, ld in enumerate(lib_deltas):
            if abs(d - ld) < 0.01:
                fit_di.append(i); break
        else:
            print(f"ERROR: Δ={d}ms not in library"); return
    lib_sub = np.hstack([lib_vecs[:, di*n_b:(di+1)*n_b] for di in fit_di])
    print(f"  Fit Δ: {fit_deltas} ms (indices {fit_di})")
    print(f"  Library sub-vector length: {lib_sub.shape[1]}")

    # ---- Load DWI signals ----
    print("Loading DWI signals (this may take a moment) ...")
    ref_shape = mask.shape
    signal_vol = np.zeros((*ref_shape, n_fit * N_SHELLS), dtype=np.float32)

    for di, (delta_ms, path) in enumerate(dwi_inputs):
        print(f"  Δ={delta_ms:.0f}ms: {os.path.basename(path)} ... ", end="", flush=True)
        data = nib.load(path).get_fdata()
        mi = np.where(mask)
        n_loaded = 0
        for vi in range(len(mi[0])):
            ix, iy, iz = mi[0][vi], mi[1][vi], mi[2][vi]
            S0 = data[ix, iy, iz, 0]
            if S0 < 1e-10:
                continue
            for si, (bv, vol_sl) in enumerate(SHELLS):
                sm = np.mean(data[ix, iy, iz, vol_sl])
                signal_vol[ix, iy, iz, di * N_SHELLS + si] = np.clip(sm / S0, 0, 1)
            n_loaded += 1
        print(f"done ({n_loaded} voxels)")

    # ---- Slice setup ----
    ax_dir = args.axis
    sl_idx = args.slice if args.slice is not None else ref_shape[ax_dir] // 2

    mask_sl = get_slice_2d(mask, ax_dir, sl_idx)  # shape (nx, ny)
    nx, ny = mask_sl.shape
    print(f"\n  Slice {sl_idx} (axis {ax_dir}): {nx} × {ny} voxels")

    # Voxel sizes for the two displayed axes
    axes_2d = [i for i in range(3) if i != ax_dir]
    zx = zooms[axes_2d[0]]  # voxel size for first dim (→ display columns after rot90)
    zy = zooms[axes_2d[1]]  # voxel size for second dim (→ display rows after rot90)

    # Physical extent of the full image (after rot90)
    # rot90(A) has shape (ny, nx): ny rows, nx columns
    W = nx * zx  # physical width
    H = ny * zy  # physical height
    extent = [0, W, 0, H]

    print(f"  Physical size: {W:.1f} × {H:.1f} mm")
    print(f"  Aspect correction: zx={zx:.4f}, zy={zy:.4f} "
          f"(ratio {zx/zy:.2f})")

    # ----------------------------------------------------------------
    # COORDINATE TRANSFORMS
    #
    # Original array A has shape (nx, ny), indexed A[vx, vy].
    # Display uses rot90(A), shape (ny, nx), with origin="lower".
    #
    # rot90(A)[r, c] = A[c, ny - 1 - r]
    #
    # With origin="lower" and extent=[0, W, 0, H]:
    #   physical x → column c ≈ x / zx → vx = c
    #   physical y → row r ≈ y / zy → vy = ny - 1 - r
    # ----------------------------------------------------------------

    def disp_to_vx(xphys, yphys):
        """Physical display coords → voxel indices in original array."""
        c = int(xphys / zx)
        r = int(yphys / zy)
        vx = np.clip(c, 0, nx - 1)
        vy = np.clip(ny - 1 - r, 0, ny - 1)
        return int(vx), int(vy)

    def vx_to_disp(vx, vy):
        """Voxel indices → physical display coords (center of voxel)."""
        xp = (vx + 0.5) * zx
        yp = (ny - 1 - vy + 0.5) * zy
        return xp, yp

    # ---- Data access helpers ----

    def get_map_slice(name):
        if name not in maps:
            return np.zeros((nx, ny))
        return get_slice_2d(maps[name], ax_dir, sl_idx)

    def get_signal_vec(vx, vy):
        if ax_dir == 0:   return signal_vol[sl_idx, vx, vy, :]
        elif ax_dir == 1: return signal_vol[vx, sl_idx, vy, :]
        else:             return signal_vol[vx, vy, sl_idx, :]

    def get_map_val(name, vx, vy):
        if name not in maps:
            return 0.0
        if ax_dir == 0:   return maps[name][sl_idx, vx, vy]
        elif ax_dir == 1: return maps[name][vx, sl_idx, vy]
        else:             return maps[name][vx, vy, sl_idx]

    # ---- Auto-zoom: find bounding box of mask in physical coords ----
    mask_rot = np.rot90(mask_sl)  # shape (ny, nx)
    rows_m, cols_m = np.where(mask_rot)
    if len(rows_m) > 0:
        xmin_p = cols_m.min() * zx - args.margin
        xmax_p = (cols_m.max() + 1) * zx + args.margin
        ymin_p = rows_m.min() * zy - args.margin
        ymax_p = (rows_m.max() + 1) * zy + args.margin
    else:
        xmin_p, xmax_p = 0, W
        ymin_p, ymax_p = 0, H

    # Verify coordinate transform with a quick sanity check
    test_vx, test_vy = nx // 2, ny // 2
    test_xp, test_yp = vx_to_disp(test_vx, test_vy)
    back_vx, back_vy = disp_to_vx(test_xp, test_yp)
    assert back_vx == test_vx and back_vy == test_vy, \
        f"Coord roundtrip failed: ({test_vx},{test_vy}) → ({test_xp:.1f},{test_yp:.1f}) → ({back_vx},{back_vy})"
    print(f"  Coordinate transform verified ✓")

    # ---- State ----
    class S:
        x = nx // 2
        y = ny // 2
        selected = False
        cur_map = args.map
        di_show = 0    # which Δ to show
        mi = 0         # which match is selected
        matches = None
    st = S()

    # ================================================================
    # FIGURE LAYOUT
    # ================================================================
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1.2, 1],
                          left=0.05, right=0.97, top=0.93, bottom=0.05,
                          wspace=0.22, hspace=0.35)
    ax_img    = fig.add_subplot(gs[:, 0])
    ax_signal = fig.add_subplot(gs[0, 1])
    ax_table  = fig.add_subplot(gs[1, 1])

    # Dedicated colorbar axes (avoids removal crash)
    cbar_ax = fig.add_axes([0.46, 0.15, 0.012, 0.35])

    crosshair_artist = [None]

    # ================================================================
    # DRAW MAP
    # ================================================================
    def draw_map():
        ax_img.clear()

        # Black background
        black_bg = np.zeros((ny, nx))  # shape of rot90'd image
        ax_img.imshow(black_bg, cmap="gray", origin="lower",
                      extent=extent, aspect="equal", vmin=0, vmax=1)

        # Parametric overlay
        sl = get_map_slice(st.cur_map)
        sl_rot = np.rot90(sl)
        # Mask: hide zero-valued and out-of-mask voxels
        sl_masked = np.ma.masked_where(
            np.rot90(~mask_sl) | (sl_rot == 0), sl_rot)
        cmap = MAP_CMAPS.get(st.cur_map, "viridis")

        ax_img.imshow(sl_masked, cmap=cmap, origin="lower",
                      extent=extent, aspect="equal", interpolation="nearest")

        # Auto-zoom to brain
        ax_img.set_xlim(xmin_p, xmax_p)
        ax_img.set_ylim(ymin_p, ymax_p)

        ax_img.set_title(f"Slice {sl_idx}  —  {st.cur_map}  [m] cycle",
                         fontsize=11, pad=8)
        ax_img.set_xlabel("mm"); ax_img.set_ylabel("mm")
        ax_img.tick_params(labelsize=7)

        # Colorbar
        cbar_ax.clear()
        vmin = np.nanmin(sl_masked)
        vmax = np.nanmax(sl_masked)
        if vmin == vmax:
            vmax = vmin + 1
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(MAP_LABELS.get(st.cur_map, ""), fontsize=8)

        # Crosshair
        crosshair_artist[0], = ax_img.plot([], [], 'w+', ms=14, mew=2.5)
        if st.selected:
            px, py = vx_to_disp(st.x, st.y)
            crosshair_artist[0].set_data([px], [py])

    draw_map()

    # Placeholders
    ax_signal.set_title("Click a voxel to begin", fontsize=11)
    ax_signal.set_xlabel("b-value (s/mm²)")
    ax_signal.set_ylabel("S(b) / S₀")
    ax_table.set_axis_off()

    # ================================================================
    # MATCHING
    # ================================================================
    def find_top(vx, vy):
        measured = get_signal_vec(vx, vy)
        if np.all(measured < 1e-10):
            return None, measured
        dists = np.sum((lib_sub - measured[np.newaxis, :]) ** 2, axis=1)
        order = np.argsort(dists)[:N_TOP]
        matches = []
        for rank, idx in enumerate(order):
            matches.append({
                'rank': rank + 1,
                'kio': lib_kios[idx], 'rho': lib_rhos[idx], 'V': lib_Vs[idx],
                'residual': dists[idx],
                'pred': lib_sub[idx],
            })
        return matches, measured

    # ================================================================
    # DRAW SIGNAL PLOT
    # ================================================================
    def draw_signal(vx, vy, measured):
        ax_signal.clear()
        if not st.matches:
            ax_signal.set_title(f"Voxel ({vx},{vy}) — no matches")
            return

        di = st.di_show % n_fit
        delta_ms = fit_deltas[di]
        m = st.matches[st.mi]

        obs  = measured[di * n_b : (di + 1) * n_b]
        pred = m['pred'][di * n_b : (di + 1) * n_b]

        # All faint matches first (behind)
        for j, om in enumerate(st.matches):
            if j == st.mi:
                continue
            op = om['pred'][di * n_b : (di + 1) * n_b]
            ax_signal.plot(BVALS_DISPLAY, op, "o-",
                           color=COLORS[j % len(COLORS)],
                           ms=3, lw=0.8, alpha=0.20, zorder=2)

        # Observed
        ax_signal.scatter(BVALS_DISPLAY, obs, c=OBS_COLOR, s=80, zorder=6,
                          label="Observed", edgecolors="white", linewidths=1.0)

        # Selected match (on top)
        c = COLORS[st.mi % len(COLORS)]
        ax_signal.plot(BVALS_DISPLAY, pred, "o-", color=c, ms=10, lw=2.5,
                       alpha=0.9, zorder=5,
                       label=f"#{m['rank']}  kio={m['kio']:.0f}  "
                             f"ρ={m['rho']/1e3:.0f}k  V={m['V']:.2f}")

        # Title with fitted values
        kv = get_map_val("kio", vx, vy)
        rv = get_map_val("rho", vx, vy)
        vv = get_map_val("V", vx, vy)
        ax_signal.set_title(
            f"Voxel ({vx},{vy})    Δ = {delta_ms:.0f} ms    "
            f"[←/→ Δ   ↑/↓ match]\n"
            f"Best fit: kio={kv:.1f}  ρ={rv/1e3:.0f}k  V={vv:.2f}",
            fontsize=10)
        ax_signal.set_xlabel("b-value  (s/mm²)")
        ax_signal.set_ylabel("S(b) / S₀")
        ax_signal.legend(loc="upper right", frameon=True, fontsize=7)
        ax_signal.set_xlim(600, 6800)
        ymax = max(1.05, float(np.max(obs)) * 1.15, float(np.max(pred)) * 1.15)
        ax_signal.set_ylim(-0.03, ymax)

    # ================================================================
    # DRAW TABLE
    # ================================================================
    def draw_table():
        ax_table.clear()
        ax_table.set_axis_off()
        if not st.matches:
            return

        cols = ["#", "k_io (s⁻¹)", "ρ (k/μL)", "V (pL)", "v_i", "SSE"]
        rows = []
        for m in st.matches:
            vi = (m['rho'] / 1e9) * (m['V'] * 1e3)
            rows.append([
                f"{m['rank']}", f"{m['kio']:.1f}",
                f"{m['rho']/1e3:.0f}", f"{m['V']:.2f}",
                f"{vi:.3f}", f"{m['residual']:.6f}",
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
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.35)

        # Header
        for j in range(len(cols)):
            cell = tbl[0, j]
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_edgecolor('#374151')

        # Highlight selected row
        for j in range(len(cols)):
            cell = tbl[st.mi + 1, j]
            cell.set_edgecolor(COLORS[st.mi % len(COLORS)])
            cell.set_text_props(fontweight='bold')

        # Style other cells
        for i in range(len(rows)):
            for j in range(len(cols)):
                if i != st.mi:
                    tbl[i + 1, j].set_edgecolor('#e5e7eb')

        ax_table.set_title(
            f"Top {len(st.matches)} matches   [↑/↓ select, ←/→ cycle Δ]",
            fontsize=9, color='#374151', pad=4)

    # ================================================================
    # UPDATE
    # ================================================================
    def update_voxel(vx, vy):
        st.matches, measured = find_top(vx, vy)
        st.mi = 0
        st.di_show = 0
        if st.matches is None:
            ax_signal.clear()
            ax_signal.set_title(f"Voxel ({vx},{vy}) — no signal")
            ax_table.clear(); ax_table.set_axis_off()
        else:
            draw_signal(vx, vy, measured)
            draw_table()
        fig.canvas.draw_idle()

    # ================================================================
    # EVENT HANDLERS
    # ================================================================
    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None:
            return
        vx, vy = disp_to_vx(event.xdata, event.ydata)
        in_mask = bool(mask_sl[vx, vy])
        kv = get_map_val("kio", vx, vy)
        rv = get_map_val("rho", vx, vy)
        vv = get_map_val("V", vx, vy)
        sig = get_signal_vec(vx, vy)
        sig_sum = np.sum(sig)

        print(f"  Click ({event.xdata:.1f}, {event.ydata:.1f}) mm "
              f"→ voxel ({vx},{vy})  "
              f"mask={in_mask}  sig_sum={sig_sum:.3f}  "
              f"kio={kv:.1f}  rho={rv/1e3:.0f}k  V={vv:.2f}")

        if not in_mask:
            return

        st.x, st.y = vx, vy
        st.selected = True
        px, py = vx_to_disp(vx, vy)
        crosshair_artist[0].set_data([px], [py])
        update_voxel(vx, vy)

    def on_key(event):
        key = event.key or ''

        if key == 'h':
            print_help(); return

        if key == 'm':
            names = [k for k in ["kio", "rho", "V", "residual"] if k in maps]
            ci = names.index(st.cur_map) if st.cur_map in names else -1
            st.cur_map = names[(ci + 1) % len(names)]
            draw_map()
            fig.canvas.draw_idle()
            print(f"  Map → {st.cur_map}"); return

        if key == 's':
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = os.path.join(args.madi_dir, f"viewer_{st.x}_{st.y}_{ts}.png")
            fig.savefig(p, dpi=180, bbox_inches='tight')
            print(f"  Saved: {p}"); return

        if not st.selected:
            return

        # ← → cycle Δ
        if key == 'right' and st.matches:
            st.di_show = (st.di_show + 1) % n_fit
            draw_signal(st.x, st.y, get_signal_vec(st.x, st.y))
            fig.canvas.draw_idle(); return
        if key == 'left' and st.matches:
            st.di_show = (st.di_show - 1) % n_fit
            draw_signal(st.x, st.y, get_signal_vec(st.x, st.y))
            fig.canvas.draw_idle(); return

        # ↑ ↓ cycle match
        if key == 'up' and st.matches:
            st.mi = max(0, st.mi - 1)
            draw_signal(st.x, st.y, get_signal_vec(st.x, st.y))
            draw_table(); fig.canvas.draw_idle(); return
        if key == 'down' and st.matches:
            st.mi = min(len(st.matches) - 1, st.mi + 1)
            draw_signal(st.x, st.y, get_signal_vec(st.x, st.y))
            draw_table(); fig.canvas.draw_idle(); return

        # Ctrl+arrows move voxel
        if 'ctrl' not in key:
            return
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
            st.selected = True
            px, py = vx_to_disp(nvx, nvy)
            crosshair_artist[0].set_data([px], [py])
            print(f"  Move → voxel ({nvx},{nvy})")
            update_voxel(nvx, nvy)

    def print_help():
        print("""
    ┌────────────────────────────────────────────────────────┐
    │            MADI VIEWER — CONTROLS                      │
    ├────────────────────────────────────────────────────────┤
    │  Click              Select voxel                       │
    │  ← / →              Cycle Δ in signal plot             │
    │  ↑ / ↓              Cycle top-10 library matches       │
    │  Ctrl + Arrows      Move selected voxel (±1)           │
    │  Ctrl+Shift+Arrow   Move voxel (±5)                    │
    │  m                  Cycle map (kio → ρ → V → residual) │
    │  s                  Save screenshot                    │
    │  h                  Print this help                    │
    └────────────────────────────────────────────────────────┘
        """)

    # ================================================================
    # CONNECT & LAUNCH
    # ================================================================
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print_help()
    n_mask_voxels = int(mask_sl.sum())
    n_signal_voxels = int(np.any(
        get_slice_2d(signal_vol, ax_dir, sl_idx) > 0, axis=-1).sum())
    print(f"Slice {sl_idx}: {nx}×{ny} voxels, {n_mask_voxels} in mask, "
          f"{n_signal_voxels} with signal")
    print(f"Δ = {fit_deltas} ms, {len(library)} library entries")
    print(f"Auto-zoom: x=[{xmin_p:.1f}, {xmax_p:.1f}] "
          f"y=[{ymin_p:.1f}, {ymax_p:.1f}] mm")
    plt.show()


if __name__ == "__main__":
    main()

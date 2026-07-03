#!/usr/bin/env python3
"""
view_error_landscape_3d.py — Interactive 3D MADI error-landscape viewer
=======================================================================

Builds a self-contained interactive HTML.  Every DISCRETE library entry is
plotted at its (kio, rho, V) coordinate and coloured by the standard MADI
matcher loss (``match_voxels_batch``, fixed-S0) for either a manually
specified normalized decay (MEASURED below) or a REAL voxel's measured
decay exported by ``scripts/fit_data.py --export-voxel``. Open the HTML in
any browser and orbit / pan / zoom — no server, no internet, no Python
needed to view it.

Controls (top bar, so nothing overlaps the plot):
    • Rician        : toggle No-Rician / Rician colouring (legacy "loss"
                      coloring — see loss_per_entry docstring below).
    • Log colour    : colour by log10(loss) for dynamic range across decades.
    • vi min / max  : move the candidate window live; entries with
                      vi=(rho/1e9)*(V*1e3) inside [min,max] are coloured
                      candidates, the rest are "excluded".  The best-fit
                      marker re-finds the minimum within the window.
    • Show excluded : show/hide the out-of-window entries (faint grey).
    • Contrast      : two sliders set the colourbar min / max.  "Reset"
                      restores defaults.
    • Dot size      : marker size.
    • Color by      : "Loss" (legacy, above) or "Posterior weight" — the
                      Bayesian posterior p(parameter_set | data) from
                      madi.fitters.bayes_fit, computed live client-side:
                      w_i ∝ exp(-resid_i / (2·sigma_m²)).  Since sigma_m only
                      rescales an already-computed residual, dragging the
                      sigma_m slider recomputes weights and the weight
                      distribution panel instantly with no server round trip.
                      "Fit S0" switches resid_i to the free-S0 (analytic
                      L2-optimal per-candidate S0) residual instead of the
                      fixed-S0 one. "Rician noise" applies the second-moment
                      Rician debias (E[M^2] = A^2 + 2σ²) to the raw measured
                      signal before computing resid_i, with a settable σ.

Click any point to open the decay-curve panel: the voxel's measured decay
(normalized by S0, or by that candidate's own fitted S0 in Fit-S0 mode)
plotted against the clicked library entry's simulated decay, drawn with
opacity that tracks how well it fits (more error → more transparent).
The panel stays in sync with sigma_m/Fit S0/Rician changes (no need to
re-click), shows the fixed vs. fitted S0 side by side so Fit-S0 mode is
visibly different from fixed-S0, and has a "Raw signal" toggle to view
un-normalized counts instead of S/S0. The weight-distribution histogram
uses a log-scaled count axis (posterior mass is usually concentrated in a
handful of bins) and marks the currently-open candidate's weight with a
dotted line.

Loss & Rician semantics (legacy "loss" coloring) match plot_error_landscape.py:
    optional log-transform:  x -> log(clip(x, s_floor, 1.0))   (if LOG_SPACE)
    loss(entry) = || measured_subset - library_vector_subset ||^2

Requires plotly to GENERATE (pip install plotly).  The output HTML is standalone.

Usage
-----
    # Manually specified decay (old behavior, unchanged):
    python view_error_landscape_3d.py \\
        --library data/libraries/madi_dense_human.npz \\
        --out     error_landscape_3d_out

    python view_error_landscape_3d.py \
        --library /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/data/libraries/madi_dense_human.npz \
        --out     /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/figures/error_3dlandscape_out

    # Real voxel, exported by fit_data.py --export-voxel:
    python scripts/fit_data.py --export-voxel 50 50 27 \\
        --input 50:dwi.nii.gz:dwi.bval:dwi.bvec --mask mask.nii.gz \\
        --library data/libraries/madi_dense_human.npz --out voxel_out \\
        --rician-correct
    python analysis/view_error_landscape_3d.py \\
        --library data/libraries/madi_dense_human.npz \\
        --voxel-data voxel_out/voxel_50_50_27.npz \\
        --out error_landscape_3d_out

        
    python scripts/fit_data.py --export-voxel 62 63 26 --input 50:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data_storage/data/Mayo_Glioma/derivatives/preproc/sub-125/dwi/sub-125_desc-madi-input_dwi.nii.gz:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data_storage/data/Mayo_Glioma/derivatives/preproc/sub-125/dwi/sub-125_desc-madi-input_dwi.bval:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data_storage/data/Mayo_Glioma/derivatives/preproc/sub-125/dwi/sub-125_desc-madi-input_dwi.bvec --mask /mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data_storage/data/Mayo_Glioma/derivatives/preproc/sub-125/dwi/sub-125_desc-brain_mask.nii.gz --library  /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/data/libraries/madi_dense_human.npz --out /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/figures/error_3dlandscape_out --rician-correct
    python analysis/view_error_landscape_3d.py
        --library /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/data/libraries/madi_dense_human.npz
        --voxel-data voxel_out/voxel_62_63_26.npz
        --out error_3dlandscape_out"""

import argparse
import json
import os
import numpy as np


# ===================================================================
# CONFIG
# ===================================================================

# Normalized decay to fit:  (Δ [ms], b [s/mm²], S/S0).  Used only when
# --voxel-data is not given.  Default = Grey-Matter curve from
# b_space_map_125.png.
MEASURED = [
    (50.0,  500.0, 0.680),
    (50.0, 1000.0, 0.470),
    (50.0, 1500.0, 0.360),
    (50.0, 2000.0, 0.270),
    (50.0, 2500.0, 0.205),
]
# White-matter tract:
# MEASURED = [(50.0,500.0,0.640),(50.0,1000.0,0.440),(50.0,1500.0,0.320),
#             (50.0,2000.0,0.245),(50.0,2500.0,0.200)]
# Edema grey matter:
# MEASURED = [(50.0,500.0,0.580),(50.0,1000.0,0.360),(50.0,1500.0,0.240),
#             (50.0,2000.0,0.170),(50.0,2500.0,0.125)]

# Assumed S0 for manually-specified (--voxel-data-less) decays -- a raw
# signal scale is synthesized as MEASURED*S0_SYNTH so the same raw-signal
# Bayesian-weight code path (fit_s0 / Rician-sigma) works whether or not a
# real voxel was loaded. Purely a bookkeeping constant; S0 cancels out of
# every quantity that matters (weights, fixed-S0 residual) except the
# absolute scale of a user-typed Rician sigma.
S0_SYNTH = 1000.0

SNR0 = 30.0                 # b=0 SNR (= S0/sigma) for the legacy "loss" Rician colouring

# Initial candidate window (movable live in the viewer).
VI_MIN    = 0.5
VI_MAX    = 0.95
RHO_MAX   = None            # hard pre-filter (not a live control); None = keep all
LOG_SPACE = False
S_FLOOR   = 1e-3

SHOW_OUT_OF_WINDOW = True    # initial state of the "Show excluded vi" toggle
RHO_LOG_AXIS       = True    # log-spaced rho AXIS (independent of log-colour)
MARKER_SIZE        = 5
DEFAULT_LOG_COLOR  = False
COLOR_CLIP_PCT     = 99.0
COLORSCALE         = "Viridis"

# Bayesian posterior-weight panel defaults.
SIGMA_M_INIT = 0.02          # matches madi.fitters.DEFAULT_SIGMA_M
SIGMA_M_MIN  = 0.002
SIGMA_M_MAX  = 0.30
FIT_S0_DEFAULT     = False
RICIAN_W_DEFAULT   = False   # Bayesian-weight-panel Rician toggle (separate
                             # from the legacy "loss" Rician button above)


# ===================================================================
#  Library I/O + loss math (mirror madi.library)
# ===================================================================

def load_library_npz(path):
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
    meta["b_values"] = list(np.asarray(data["b_values"], dtype=float)) \
        if "b_values" in data.files else None
    if meta["deltas"] is None or meta["n_b"] is None or meta["b_values"] is None:
        raise ValueError(f"Library {path} missing deltas/n_b/b_values metadata.")
    return entries, meta


def load_voxel_npz(path):
    """Load a voxel exported by ``scripts/fit_data.py --export-voxel``."""
    data = np.load(path, allow_pickle=True)
    fit_pairs = list(zip(np.asarray(data["fit_deltas"], dtype=float).tolist(),
                          np.asarray(data["fit_bvals"], dtype=float).tolist()))
    sigma = float(data["sigma"]) if "sigma" in data.files else None
    if sigma is not None and np.isnan(sigma):
        sigma = None
    return dict(
        measured=np.asarray(data["measured"], dtype=float),
        raw=np.asarray(data["raw"], dtype=float),
        fit_pairs=fit_pairs,
        s0=float(data["s0"]),
        sigma=sigma,
        ijk=np.asarray(data["ijk"], dtype=int).tolist() if "ijk" in data.files else None,
    )


def pair_indices(fit_pairs, lib_deltas, lib_b_values, n_b, b_tol=50.0):
    cols = np.empty(len(fit_pairs), dtype=int)
    for k, (d, b) in enumerate(fit_pairs):
        di = next((i for i, ld in enumerate(lib_deltas) if abs(d - ld) < 0.01), None)
        if di is None:
            raise ValueError(f"Δ = {d} ms not in library deltas {list(lib_deltas)}.")
        bi = next((j for j, lb in enumerate(lib_b_values) if abs(b - lb) < b_tol), None)
        if bi is None:
            raise ValueError(f"b = {b} s/mm² not in library b-values {list(lib_b_values)}.")
        cols[k] = di * n_b + bi
    return cols


def rician_normalized(norm_vec, snr0):
    norm_vec = np.asarray(norm_vec, dtype=float)
    nt = 2.0 / (snr0 ** 2)
    denom2 = 1.0 - nt
    if denom2 <= 0:
        raise ValueError(f"SNR0={snr0} too low (needs > sqrt(2) ≈ 1.41).")
    num = np.sqrt(np.clip(norm_vec ** 2 - nt, 0.0, None))
    return num / np.sqrt(denom2)


def loss_per_entry(measured_vec, lib_subset, log_space, s_floor):
    if log_space:
        m = np.log(np.clip(measured_vec, s_floor, 1.0))
        r = np.log(np.clip(lib_subset, s_floor, 1.0))
    else:
        m, r = measured_vec, lib_subset
    diff = r - m[None, :]
    return np.sum(diff * diff, axis=1)


# ===================================================================
#  Build figure (3 traces) + full-data JS payload
# ===================================================================

def build(entries, meta, measured_vec, fit_pairs, raw_vec, s0, sigma_raw):
    import plotly.graph_objects as go

    cols = pair_indices(fit_pairs, meta["deltas"], meta["b_values"], meta["n_b"])
    lib_subset = entries["vectors"][:, cols]
    loss_no = loss_per_entry(measured_vec, lib_subset, LOG_SPACE, S_FLOOR)
    loss_ri = loss_per_entry(rician_normalized(measured_vec, SNR0),
                             lib_subset, LOG_SPACE, S_FLOOR)

    kio = entries["kio"]; rho = entries["rho"]; V = entries["V"]
    vi = (rho / 1e9) * (V * 1e3)

    # Hard pre-filter (rho_max) defines the universe of plotted points.
    universe = np.ones(len(kio), dtype=bool)
    if RHO_MAX is not None:
        universe &= rho <= RHO_MAX
    kio, rho, V, vi = kio[universe], rho[universe], V[universe], vi[universe]
    loss_no, loss_ri = loss_no[universe], loss_ri[universe]
    lib_subset = lib_subset[universe]

    # rho axis (optionally log-spaced)
    if RHO_LOG_AXIS:
        ycoord = np.log10(rho)
        decades = np.arange(np.floor(np.log10(rho.min())),
                            np.ceil(np.log10(rho.max())) + 1)
        tickvals, ticktext = [], []
        for d in decades:
            for mult in (1, 2, 5):
                val = mult * 10 ** d
                if rho.min() / 1.5 <= val <= rho.max() * 1.5:
                    tickvals.append(float(np.log10(val)))
                    ticktext.append(f"{val/1e3:.0f}k" if val < 1e6 else f"{val/1e6:g}M")
        yaxis_kw = dict(title="ρ  [cells/µL]", tickvals=tickvals, ticktext=ticktext)
    else:
        ycoord = rho.astype(float)
        yaxis_kw = dict(title="ρ  [cells/µL]")

    def hov(i):
        return (f"k_io = {kio[i]:g} s⁻¹<br>ρ = {rho[i]/1e3:.0f}k cells/µL<br>"
                f"V = {V[i]:g} pL<br>vi = {vi[i]:.3f}<br>"
                f"loss(noRic) = {loss_no[i]:.4g}<br>loss(Ric) = {loss_ri[i]:.4g}")

    cand0 = (vi >= VI_MIN) & (vi <= VI_MAX)
    out0 = ~cand0
    ci = np.where(cand0)[0]
    oi = np.where(out0)[0]

    cl = loss_no[cand0] if cand0.any() else loss_no
    cmin0 = float(cl.min())
    cmax0 = float(np.percentile(cl, COLOR_CLIP_PCT))
    if cmax0 <= cmin0:
        cmax0 = cmin0 + 1e-9

    # trace 0: candidates (geometry/colour all rewritten in JS on init)
    t_cand = go.Scatter3d(
        x=kio[cand0], y=ycoord[cand0], z=V[cand0], mode="markers", name="candidates",
        marker=dict(size=MARKER_SIZE, color=loss_no[cand0], colorscale=COLORSCALE,
                    cmin=cmin0, cmax=cmax0, opacity=0.9, line=dict(width=0),
                    colorbar=dict(title=dict(text="loss<br>‖m−r‖²"), len=0.75)),
        text=[hov(i) for i in ci], hoverinfo="text",
    )

    # trace 1: best
    jno = int(ci[np.argmin(loss_no[cand0])]) if cand0.any() else 0
    t_best = go.Scatter3d(
        x=[kio[jno]], y=[ycoord[jno]], z=[V[jno]], mode="markers", name="best fit",
        marker=dict(size=MARKER_SIZE + 7, color="red", symbol="diamond",
                    line=dict(width=1.5, color="white")),
        text=["best"], hoverinfo="text",
    )

    # trace 2: excluded (always present so index 2 is stable)
    gx = kio[out0] if SHOW_OUT_OF_WINDOW else np.array([])
    gy = ycoord[out0] if SHOW_OUT_OF_WINDOW else np.array([])
    gz = V[out0] if SHOW_OUT_OF_WINDOW else np.array([])
    gt = [hov(i) for i in oi] if SHOW_OUT_OF_WINDOW else []
    t_out = go.Scatter3d(
        x=gx, y=gy, z=gz, mode="markers", name="excluded (vi out of window)",
        marker=dict(size=max(2, MARKER_SIZE - 2), color="lightgrey",
                    opacity=0.22, line=dict(width=0)),
        text=gt, hoverinfo="text",
    )

    fig = go.Figure(data=[t_cand, t_best, t_out])
    fig.update_layout(
        scene=dict(xaxis=dict(title="k_io  [s⁻¹]"), yaxis=yaxis_kw,
                   zaxis=dict(title="V  [pL]"), aspectmode="cube"),
        legend=dict(itemsizing="constant", x=0.0, y=1.0),
        margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white",
    )

    feat_labels = [f"Δ{d:g}/b{b:g}" for d, b in fit_pairs]

    payload = dict(
        greyIdx=2, sizeDefault=int(MARKER_SIZE), bestSizeOff=7,
        kio=[float(x) for x in kio], rho=[float(x) for x in rho],
        y=[float(x) for x in ycoord], V=[float(x) for x in V],
        vi=[float(x) for x in vi],
        lossNo=[float(x) for x in loss_no], lossRi=[float(x) for x in loss_ri],
        viDataMin=float(vi.min()), viDataMax=float(vi.max()),
        viMinDefault=float(VI_MIN), viMaxDefault=float(VI_MAX),
        showOutDefault=bool(SHOW_OUT_OF_WINDOW),
        clipPct=float(COLOR_CLIP_PCT), defaultLog=bool(DEFAULT_LOG_COLOR),
        snr0=float(SNR0),
        # Bayesian posterior-weight panel data.
        nFeat=int(lib_subset.shape[1]),
        featLabels=feat_labels,
        libSubset=[float(x) for x in lib_subset.ravel()],
        mRaw=[float(x) for x in raw_vec],
        s0=float(s0), sigmaRaw=float(sigma_raw),
        sigmaMInit=float(SIGMA_M_INIT), sigmaMMin=float(SIGMA_M_MIN),
        sigmaMMax=float(SIGMA_M_MAX),
        fitS0Default=bool(FIT_S0_DEFAULT), ricianWDefault=bool(RICIAN_W_DEFAULT),
    )
    stats = (loss_no, loss_ri, cand0, kio, rho, V, vi)
    return fig, payload, stats


# ===================================================================
#  Compose the standalone HTML
# ===================================================================

def write_html(fig, payload, title, subtitle, out_path):
    frag = fig.to_html(full_html=False, include_plotlyjs=True, div_id="gd",
                       config={"responsive": True, "displaylogo": False,
                               "toImageButtonOptions": {"scale": 2}})
    data_json = json.dumps(payload)

    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root{ --bg:#fafafa; --pan:#ffffff; --bd:#dcdcdc; --tx:#222; --mut:#666; --acc:#1f5fb8; }
  html,body{margin:0;height:100%;font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;color:var(--tx);background:var(--bg);}
  #wrap{display:flex;flex-direction:column;height:100vh;}
  #bar{flex:0 0 auto;background:var(--pan);border-bottom:1px solid var(--bd);
       padding:8px 14px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;box-shadow:0 1px 3px rgba(0,0,0,.05);}
  #bar2{flex:0 0 auto;background:var(--pan);border-bottom:1px solid var(--bd);
       padding:6px 14px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;}
  #bar h1{font-size:15px;margin:0 12px 0 0;font-weight:650;}
  #bar .sub{font-size:11.5px;color:var(--mut);font-weight:400;}
  .grp{display:flex;align-items:center;gap:7px;}
  .grp label{font-size:12.5px;color:var(--tx);white-space:nowrap;}
  .seg{display:inline-flex;border:1px solid var(--bd);border-radius:7px;overflow:hidden;}
  .seg button{border:0;background:#fff;padding:5px 12px;font-size:12.5px;cursor:pointer;color:var(--tx);}
  .seg button.on{background:var(--acc);color:#fff;}
  input[type=range]{width:130px;}
  input[type=number]{width:64px;font-size:12px;}
  .val{font-variant-numeric:tabular-nums;font-size:11.5px;color:var(--mut);min-width:54px;display:inline-block;}
  .btn{border:1px solid var(--bd);background:#fff;border-radius:7px;padding:5px 11px;font-size:12.5px;cursor:pointer;}
  .btn:hover{background:#f0f4fb;}
  #main{flex:1 1 auto;min-height:0;display:flex;}
  #gd{flex:2 1 0;min-width:0;}
  #side{flex:1 1 0;min-width:280px;display:flex;flex-direction:column;border-left:1px solid var(--bd);}
  #wdist{flex:1 1 0;min-height:0;border-bottom:1px solid var(--bd);}
  #decayCtl{flex:0 0 auto;padding:5px 10px;display:flex;gap:12px;align-items:center;
            border-bottom:1px solid var(--bd);}
  #decay{flex:1 1 0;min-height:0;}
  .chk{display:flex;align-items:center;gap:6px;font-size:12.5px;cursor:pointer;white-space:nowrap;}
  .sep{width:1px;height:24px;background:var(--bd);}
  #decayHint{font-size:11px;color:var(--mut);padding:6px 10px;}
  #s0Info{font-size:11px;color:var(--mut);}
</style></head>
<body><div id="wrap">
  <div id="bar">
    <div><h1>__TITLE__</h1></div>
    <div class="grp"><span class="sub">__SUB__</span></div>
    <div class="sep"></div>
    <div class="grp"><label>Rician (loss)</label>
      <div class="seg" id="modeSeg">
        <button id="mNo" class="on">No Rician</button>
        <button id="mRi">Rician (SNR&#8320;=__SNR__)</button>
      </div></div>
    <div class="grp"><label class="chk"><input type="checkbox" id="logChk"> Log colour</label></div>
    <div class="sep"></div>
    <div class="grp"><label>vi min</label>
      <input type="range" id="viMin" min="0" max="1000" value="500">
      <span class="val" id="viMinVal"></span></div>
    <div class="grp"><label>vi max</label>
      <input type="range" id="viMax" min="0" max="1000" value="950">
      <span class="val" id="viMaxVal"></span></div>
    <div class="grp"><label class="chk"><input type="checkbox" id="showChk"> Show excluded vi</label></div>
    <div class="sep"></div>
    <div class="grp"><label>Contrast min</label>
      <input type="range" id="cMin" min="0" max="1000" value="0">
      <span class="val" id="cMinVal"></span></div>
    <div class="grp"><label>max</label>
      <input type="range" id="cMax" min="0" max="1000" value="1000">
      <span class="val" id="cMaxVal"></span></div>
    <div class="grp"><label>Dot size</label>
      <input type="range" id="dotSz" min="2" max="18" value="5">
      <span class="val" id="dotSzVal"></span></div>
    <div class="grp"><button class="btn" id="resetBtn">Reset</button></div>
  </div>
  <div id="bar2">
    <div class="grp"><label>Color by</label>
      <div class="seg" id="colorSeg">
        <button id="cbLoss" class="on">Loss</button>
        <button id="cbWeight">Posterior weight</button>
      </div></div>
    <div class="sep"></div>
    <div class="grp"><label>&sigma;<sub>m</sub></label>
      <input type="range" id="sigmaM" min="0" max="1000" value="500">
      <span class="val" id="sigmaMVal"></span></div>
    <div class="grp"><label class="chk"><input type="checkbox" id="fitS0Chk"> Fit S0</label></div>
    <div class="sep"></div>
    <div class="grp"><label class="chk"><input type="checkbox" id="riceWChk"> Rician noise, &sigma;=</label>
      <input type="number" id="riceSigma" step="any"></div>
    <div class="grp"><span class="sub" id="nEffVal"></span></div>
  </div>
  <div id="main">
  __FRAG__
  <div id="side">
    <div id="wdist"></div>
    <div id="decayCtl">
      <label class="chk"><input type="checkbox" id="rawChk"> Raw signal (not normalized)</label>
      <span class="val" id="s0Info"></span>
    </div>
    <div id="decay"></div>
    <div id="decayHint">Click a point in the 3D plot to compare its simulated decay against the measured voxel curve.</div>
  </div>
  </div>
</div>
<script>
(function(){
  const D = __DATA__;
  const gd = document.getElementById('gd');
  const CAND = 0, BEST = 1, GREY = 2;
  let mode = 'no';
  let logc = !!D.defaultLog;
  let touched = false;
  let colorMode = 'loss';           // 'loss' | 'weight'
  let lastCandIdx = [];              // global indices behind trace CAND, in display order
  let lastClickedIdx = null;         // library entry currently shown in the decay panel

  function g(x){ if(x===0) return '0'; const a=Math.abs(x);
    if(a>=1e4||a<1e-3) return x.toExponential(2);
    return (Math.round(x*1000)/1000).toString(); }
  function hov(i){
    return 'k_io = '+g(D.kio[i])+' s\\u207B\\u00B9<br>\\u03C1 = '+(D.rho[i]/1e3).toFixed(0)+
      'k cells/\\u00B5L<br>V = '+g(D.V[i])+' pL<br>vi = '+D.vi[i].toFixed(3)+
      '<br>loss(noRic) = '+D.lossNo[i].toPrecision(4)+'<br>loss(Ric) = '+D.lossRi[i].toPrecision(4);
  }
  function frac(id){ return parseInt(document.getElementById(id).value,10)/1000; }
  function viVal(id){ return D.viDataMin + frac(id)*(D.viDataMax-D.viDataMin); }
  function activeLoss(){ return (mode==='no')?D.lossNo:D.lossRi; }

  // ---------------------------------------------------------------
  // Bayesian posterior-weight math (mirrors madi.fitters.bayes_fit).
  // sigma_m only rescales an already-computed residual, so residuals are
  // recomputed only when fitS0/riceW/riceSigma actually change; the
  // sigma_m slider itself just re-runs the (cheap) softmax.
  // ---------------------------------------------------------------
  const nLib = D.kio.length, nFeat = D.nFeat;
  function libRow(i){ return D.libSubset.slice(i*nFeat, i*nFeat+nFeat); }

  function riceCorrect(vec, sigma){
    return vec.map(v => Math.sqrt(Math.max(v*v - 2*sigma*sigma, 0)));
  }
  function fixedS0Resid(Mnorm){
    const out = new Float64Array(nLib);
    for(let i=0;i<nLib;i++){
      let s=0, base=i*nFeat;
      for(let j=0;j<nFeat;j++){ const d=D.libSubset[base+j]-Mnorm[j]; s+=d*d; }
      out[i]=s;
    }
    return {resid: out, s0Cand: null};
  }
  function freeS0Resid(Mraw){
    const out = new Float64Array(nLib);
    const s0c = new Float64Array(nLib);
    let mm=0; for(let j=0;j<nFeat;j++) mm+=Mraw[j]*Mraw[j];
    for(let i=0;i<nLib;i++){
      let rr=0, mr=0, base=i*nFeat;
      for(let j=0;j<nFeat;j++){ const r=D.libSubset[base+j]; rr+=r*r; mr+=Mraw[j]*r; }
      rr=Math.max(rr,1e-30);
      const s0cand=mr/rr;
      let resid=mm-(mr*mr)/rr;
      if(s0cand<=0){ resid=Infinity; }
      s0c[i]=s0cand;
      out[i]= s0cand>0 ? resid/(s0cand*s0cand) : Infinity;
    }
    return {resid: out, s0Cand: s0c};
  }
  // Cache: recomputed only when fitS0/riceW/riceSigma change.
  let residCache = null, residKey = '';
  function currentResid(){
    const fitS0 = document.getElementById('fitS0Chk').checked;
    const riceW = document.getElementById('riceWChk').checked;
    const riceSigma = parseFloat(document.getElementById('riceSigma').value) || 0;
    const key = fitS0+'|'+riceW+'|'+riceSigma;
    if(residCache && residKey===key) return residCache;
    const M = riceW ? riceCorrect(D.mRaw, riceSigma) : D.mRaw.slice();
    let r;
    if(!fitS0){
      r = fixedS0Resid(M.map(v=>v/D.s0));
    } else {
      r = freeS0Resid(M);
    }
    r.M = M;   // raw (Rician-corrected if toggled) signal, pre-S0-normalization
    residCache = r; residKey = key;
    return r;
  }
  function sigmaMVal(){
    const f = frac('sigmaM');
    // log-spaced slider between sigmaMMin and sigmaMMax
    const lo=Math.log10(D.sigmaMMin), hi=Math.log10(D.sigmaMMax);
    return Math.pow(10, lo+f*(hi-lo));
  }
  function weightsFor(idxArr){
    const r = currentResid();
    const sigmaM = sigmaMVal();
    const denom = 2*sigmaM*sigmaM;
    let vals = idxArr.map(i => -r.resid[i]/denom);
    const mx = vals.length ? Math.max(...vals.filter(v=>isFinite(v))) : 0;
    let w = vals.map(v => isFinite(v) ? Math.exp(v-mx) : 0);
    const sum = w.reduce((a,b)=>a+b,0) || 1e-300;
    return w.map(v=>v/sum);
  }

  function disp(arr){
    if(!logc) return arr.slice();
    let mn=Infinity; for(const v of arr) if(v>0 && v<mn) mn=v;
    const floor=(mn===Infinity?1e-12:mn*1e-3);
    return arr.map(v=>Math.log10(Math.max(v,floor)));
  }
  function percentile(sorted,p){
    if(sorted.length===0) return 0;
    const idx=Math.min(sorted.length-1,Math.max(0,Math.round((p/100)*(sorted.length-1))));
    return sorted[idx];
  }
  function defaultFracs(dvals){
    if(dvals.length===0) return {minF:0,maxF:1};
    const dmin=Math.min(...dvals),dmax=Math.max(...dvals),span=(dmax-dmin)||1e-9;
    const sorted=dvals.slice().sort((a,b)=>a-b);
    return {minF:0, maxF:(percentile(sorted,D.clipPct)-dmin)/span};
  }

  function rebuild(resetContrast){
    const lo=Math.min(viVal('viMin'),viVal('viMax'));
    const hi=Math.max(viVal('viMin'),viVal('viMax'));
    document.getElementById('viMinVal').textContent=lo.toFixed(3);
    document.getElementById('viMaxVal').textContent=hi.toFixed(3);

    const cand=[],out=[];
    for(let i=0;i<D.vi.length;i++){ (D.vi[i]>=lo && D.vi[i]<=hi)?cand.push(i):out.push(i); }
    lastCandIdx = cand;

    let dvals, label, colorbarLabel;
    if(colorMode==='loss'){
      const active=activeLoss();
      dvals=disp(cand.map(i=>active[i]));
      colorbarLabel = logc ? 'log\\u2081\\u2080 loss' : 'loss<br>\\u2016m\\u2212r\\u2016\\u00B2';
    } else {
      const w = weightsFor(cand);
      dvals = disp(w);
      colorbarLabel = logc ? 'log\\u2081\\u2080 weight' : 'posterior<br>weight';
      updateNEff(w);
    }
    const dmin=dvals.length?Math.min(...dvals):0, dmax=dvals.length?Math.max(...dvals):1;
    const span=(dmax-dmin)||1e-9;

    if(resetContrast){
      const df=defaultFracs(dvals);
      document.getElementById('cMin').value=Math.round(df.minF*1000);
      document.getElementById('cMax').value=Math.round(df.maxF*1000);
      touched=false;
    }
    let fMin=frac('cMin'),fMax=frac('cMax'); if(fMin>fMax){const t=fMin;fMin=fMax;fMax=t;}
    let cmin=dmin+fMin*span, cmax=dmin+fMax*span; if(cmax<=cmin) cmax=cmin+1e-9;

    Plotly.restyle(gd,{
      x:[cand.map(i=>D.kio[i])], y:[cand.map(i=>D.y[i])], z:[cand.map(i=>D.V[i])],
      text:[cand.map(i=>hov(i))],
      'marker.color':[dvals], 'marker.cmin':cmin, 'marker.cmax':cmax,
      'marker.colorbar.title.text':colorbarLabel
    },[CAND]);

    const showOut=document.getElementById('showChk').checked;
    const gi=showOut?out:[];
    Plotly.restyle(gd,{ x:[gi.map(i=>D.kio[i])], y:[gi.map(i=>D.y[i])],
      z:[gi.map(i=>D.V[i])], text:[gi.map(i=>hov(i))] },[GREY]);

    if(cand.length){
      let bi, bt;
      if(colorMode==='loss'){
        const active=activeLoss();
        bi=cand[0]; let bl=active[bi];
        for(const i of cand){ if(active[i]<bl){bl=active[i];bi=i;} }
        bt='BEST ('+(mode==='no'?'No Rician':'Rician')+')<br>k_io='+g(D.kio[bi])+
          ', \\u03C1='+(D.rho[bi]/1e3).toFixed(0)+'k, V='+g(D.V[bi])+
          '<br>loss='+active[bi].toPrecision(4)+', vi='+D.vi[bi].toFixed(3);
      } else {
        const w = weightsFor(cand);
        let bj=0; for(let k=1;k<w.length;k++) if(w[k]>w[bj]) bj=k;
        bi=cand[bj];
        bt='MAP WEIGHT<br>k_io='+g(D.kio[bi])+', \\u03C1='+(D.rho[bi]/1e3).toFixed(0)+
          'k, V='+g(D.V[bi])+'<br>weight='+w[bj].toPrecision(4)+', vi='+D.vi[bi].toFixed(3);
      }
      Plotly.restyle(gd,{x:[[D.kio[bi]]],y:[[D.y[bi]]],z:[[D.V[bi]]],text:[[bt]]},[BEST]);
    } else {
      Plotly.restyle(gd,{x:[[]],y:[[]],z:[[]],text:[['']]},[BEST]);
    }

    const showLo=logc?Math.pow(10,cmin):cmin, showHi=logc?Math.pow(10,cmax):cmax;
    document.getElementById('cMinVal').textContent=showLo.toPrecision(3);
    document.getElementById('cMaxVal').textContent=showHi.toPrecision(3);

    updateWeightDist(cand);
    if(lastClickedIdx !== null) showDecay(lastClickedIdx);
  }

  function updateNEff(w){
    const nEff = 1/Math.max(w.reduce((a,b)=>a+b*b,0),1e-300);
    document.getElementById('nEffVal').textContent =
      'n_eff = '+nEff.toFixed(2)+' / '+w.length+' candidates';
  }

  // ---------------------------------------------------------------
  // Weight-distribution side panel (updates with sigma_m / toggles).
  // ---------------------------------------------------------------
  let wdistInit=false;
  function updateWeightDist(cand){
    const w = weightsFor(cand);
    const floor = 1e-12;
    const logw = w.map(v=>Math.log10(Math.max(v,floor)));
    const trace = {x: logw, type:'histogram', nbinsx: 40,
                    marker:{color:'#1f5fb8'}};
    const shapes = [];
    if(lastClickedIdx !== null){
      const pos = cand.indexOf(lastClickedIdx);
      if(pos !== -1){
        const lw = Math.log10(Math.max(w[pos],floor));
        shapes.push({type:'line', x0:lw, x1:lw, y0:0, y1:1, yref:'paper',
                     line:{color:'crimson', width:2, dash:'dot'}});
      }
    }
    const layout = {
      title: {text:'posterior weight distribution ('+w.length+' candidates, \\u03C3_m='+
              sigmaMVal().toPrecision(3)+')', font:{size:11}},
      xaxis:{title:{text:'log\\u2081\\u2080(weight)', font:{size:10}}},
      yaxis:{title:{text:'count (log)', font:{size:10}}, type:'log'},
      shapes:shapes,
      margin:{l:45,r:10,t:34,b:34}, font:{size:9},
    };
    if(!wdistInit){
      Plotly.newPlot('wdist',[trace],layout,{responsive:true,displayModeBar:false});
      wdistInit=true;
    } else {
      Plotly.react('wdist',[trace],layout,{responsive:true,displayModeBar:false});
    }
  }

  // ---------------------------------------------------------------
  // Decay-curve panel (populated on click).
  // ---------------------------------------------------------------
  let decayInit=false;
  function showDecay(i){
    lastClickedIdx = i;
    const r = currentResid();
    const cand = lastCandIdx.length ? lastCandIdx : [i];
    const candResid = cand.map(k=>r.resid[k]).filter(isFinite);
    const rmin = candResid.length?Math.min(...candResid):0;
    const rmax = candResid.length?Math.max(...candResid):1;
    const span = (rmax-rmin)||1e-9;
    const frac0 = Math.min(1,Math.max(0,(r.resid[i]-rmin)/span));
    const opacity = Math.max(0.15, 1-frac0);

    const s0Fit = r.s0Cand ? r.s0Cand[i] : null;
    const s0ForCand = (s0Fit!==null && s0Fit>0) ? s0Fit : D.s0;
    const rawMode = document.getElementById('rawChk').checked;

    let measuredCurve, candCurve, yTitle;
    if(rawMode){
      measuredCurve = r.M;                              // raw counts
      candCurve = libRow(i).map(v=>v*s0ForCand);         // scaled to this candidate's S0
      yTitle = 'raw signal';
    } else {
      measuredCurve = r.M.map(v=>v/s0ForCand);           // normalized by whichever S0 applies
      candCurve = libRow(i);                             // library vectors are already S/S0
      yTitle = 'S/S0';
    }

    const s0Text = s0Fit!==null
      ? ('S0 fixed = '+D.s0.toFixed(1)+'   S0 fit = '+s0Fit.toFixed(1)+
         '   (fit/fixed = '+(s0Fit/D.s0).toFixed(3)+')')
      : ('S0 = '+D.s0.toFixed(1));
    document.getElementById('s0Info').textContent = s0Text;

    const traces = [
      {x:D.featLabels, y:measuredCurve, mode:'lines+markers', name:'measured',
       line:{color:'#222', width:2},
       hovertemplate:'%{x}<br>measured=%{y:.4g}<extra></extra>'},
      {x:D.featLabels, y:candCurve, mode:'lines+markers', name:'simulated',
       line:{color:'crimson', width:3}, opacity:opacity,
       hovertemplate:'%{x}<br>simulated=%{y:.4g}<extra></extra>'},
    ];
    const layout = {
      title:{text:'k_io='+g(D.kio[i])+', \\u03C1='+(D.rho[i]/1e3).toFixed(0)+
             'k, V='+g(D.V[i])+'  (resid='+r.resid[i].toPrecision(4)+')', font:{size:11}},
      yaxis:{title:{text:yTitle, font:{size:10}}, rangemode: rawMode?'tozero':'normal'},
      legend:{font:{size:9}, x:0.02, y:0.06},
      margin:{l:50,r:10,t:34,b:34}, font:{size:9},
    };
    if(!decayInit){
      Plotly.newPlot('decay',traces,layout,{responsive:true,displayModeBar:false});
      decayInit=true;
    } else {
      Plotly.react('decay',traces,layout,{responsive:true,displayModeBar:false});
    }
  }

  function resize(){
    const s=parseInt(document.getElementById('dotSz').value,10);
    document.getElementById('dotSzVal').textContent=s+' px';
    Plotly.restyle(gd,{'marker.size':s},[CAND]);
    Plotly.restyle(gd,{'marker.size':s+(D.bestSizeOff||7)},[BEST]);
    Plotly.restyle(gd,{'marker.size':Math.max(2,s-2)},[GREY]);
  }

  function setMode(m){
    mode=m;
    document.getElementById('mNo').classList.toggle('on',m==='no');
    document.getElementById('mRi').classList.toggle('on',m==='ri');
    rebuild(!touched);
  }
  function setColorMode(m){
    colorMode=m;
    document.getElementById('cbLoss').classList.toggle('on',m==='loss');
    document.getElementById('cbWeight').classList.toggle('on',m==='weight');
    document.getElementById('nEffVal').textContent = (m==='weight')?'':'';
    rebuild(true);
  }

  document.getElementById('mNo').onclick=()=>setMode('no');
  document.getElementById('mRi').onclick=()=>setMode('ri');
  document.getElementById('cbLoss').onclick=()=>setColorMode('loss');
  document.getElementById('cbWeight').onclick=()=>setColorMode('weight');
  document.getElementById('logChk').onchange=function(){ logc=this.checked; rebuild(!touched); };
  document.getElementById('viMin').oninput=()=>rebuild(!touched);
  document.getElementById('viMax').oninput=()=>rebuild(!touched);
  document.getElementById('showChk').onchange=()=>rebuild(false);
  document.getElementById('cMin').oninput=()=>{ touched=true; rebuild(false); };
  document.getElementById('cMax').oninput=()=>{ touched=true; rebuild(false); };
  document.getElementById('resetBtn').onclick=()=>rebuild(true);
  document.getElementById('dotSz').oninput=resize;
  document.getElementById('sigmaM').oninput=function(){
    document.getElementById('sigmaMVal').textContent=sigmaMVal().toPrecision(3);
    rebuild(false);
  };
  document.getElementById('fitS0Chk').onchange=()=>{ residCache=null; rebuild(true); };
  document.getElementById('riceWChk').onchange=()=>{ residCache=null; rebuild(true); };
  document.getElementById('riceSigma').onchange=()=>{ residCache=null; rebuild(false); };
  document.getElementById('rawChk').onchange=()=>{ if(lastClickedIdx!==null) showDecay(lastClickedIdx); };

  gd.on('plotly_click', function(evt){
    const pt = evt.points && evt.points[0];
    if(!pt || pt.curveNumber !== CAND) return;
    const i = lastCandIdx[pt.pointNumber];
    if(i===undefined) return;
    showDecay(i);
  });

  function clamp01(x){ return Math.max(0,Math.min(1,x)); }
  function init(){
    if(!gd || !gd.data){ return setTimeout(init,40); }
    const sp=(D.viDataMax-D.viDataMin)||1e-9;
    document.getElementById('viMin').value=Math.round(clamp01((D.viMinDefault-D.viDataMin)/sp)*1000);
    document.getElementById('viMax').value=Math.round(clamp01((D.viMaxDefault-D.viDataMin)/sp)*1000);
    document.getElementById('showChk').checked=D.showOutDefault;
    document.getElementById('logChk').checked=logc;
    document.getElementById('dotSz').value=D.sizeDefault;
    document.getElementById('fitS0Chk').checked=D.fitS0Default;
    document.getElementById('riceWChk').checked=D.ricianWDefault;
    document.getElementById('riceSigma').value=D.sigmaRaw;
    const slo=Math.log10(D.sigmaMMin), shi=Math.log10(D.sigmaMMax);
    document.getElementById('sigmaM').value=Math.round(
      clamp01((Math.log10(D.sigmaMInit)-slo)/(shi-slo))*1000);
    document.getElementById('sigmaMVal').textContent=sigmaMVal().toPrecision(3);
    rebuild(true);
    resize();
  }
  init();
})();
</script>
</body></html>"""
    html = (html.replace("__TITLE__", title).replace("__SUB__", subtitle)
                .replace("__SNR__", f"{payload['snr0']:g}")
                .replace("__FRAG__", frag).replace("__DATA__", data_json))
    with open(out_path, "w") as f:
        f.write(html)


# ===================================================================
#  Main
# ===================================================================

def main():
    global SIGMA_M_INIT, SIGMA_M_MIN, SIGMA_M_MAX, FIT_S0_DEFAULT, RICIAN_W_DEFAULT
    ap = argparse.ArgumentParser(
        description="Interactive 3D MADI error-landscape viewer (HTML).")
    ap.add_argument("--library", required=True, help="Path to MADI .npz library.")
    ap.add_argument("--voxel-data", default=None,
                    help="Optional .npz from `scripts/fit_data.py --export-voxel` "
                         "with a real voxel's measured decay. If omitted, the "
                         "MEASURED constant at the top of this file is used "
                         "(with a synthesized raw signal at S0=%.0f so the "
                         "Bayesian-weight panel still works)." % S0_SYNTH)
    ap.add_argument("--sigma-m-init", type=float, default=SIGMA_M_INIT)
    ap.add_argument("--sigma-m-min", type=float, default=SIGMA_M_MIN)
    ap.add_argument("--sigma-m-max", type=float, default=SIGMA_M_MAX)
    ap.add_argument("--fit-s0-default", action="store_true",
                    help="Default the 'Fit S0' toggle on in the viewer.")
    ap.add_argument("--rician-default", action="store_true",
                    help="Default the Bayesian-weight-panel Rician toggle on.")
    ap.add_argument("--out", default="error_landscape_3d_out",
                    help="Output directory for the HTML.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    SIGMA_M_INIT, SIGMA_M_MIN, SIGMA_M_MAX = (args.sigma_m_init, args.sigma_m_min,
                                              args.sigma_m_max)
    FIT_S0_DEFAULT = bool(args.fit_s0_default)
    RICIAN_W_DEFAULT = bool(args.rician_default)

    print("=" * 64)
    print("MADI 3D error-landscape viewer")
    print("=" * 64)
    entries, meta = load_library_npz(args.library)
    lib_name = os.path.basename(args.library)
    print(f"  Library:   {args.library}  ({len(entries['kio'])} entries)")
    print(f"  Δ [ms]:    {[f'{d:g}' for d in meta['deltas']]}")
    print(f"  b [s/mm²]: {[f'{b:g}' for b in meta['b_values']]}  (n_b={meta['n_b']})")

    if args.voxel_data:
        vox = load_voxel_npz(args.voxel_data)
        fit_pairs = vox["fit_pairs"]
        measured_vec = vox["measured"]
        raw_vec = vox["raw"]
        s0 = vox["s0"]
        sigma_raw = vox["sigma"] if vox["sigma"] is not None else (s0 / SNR0)
        measured_desc = (f"voxel {vox['ijk']}  S/S0:  " +
                         ",  ".join(f"Δ{d:g}/b{b:g}={s:g}"
                                    for (d, b), s in zip(fit_pairs, measured_vec)))
        print(f"  Voxel data: {args.voxel_data}  ijk={vox['ijk']}  "
              f"S0={s0:.1f}  sigma={vox['sigma']}")
    else:
        fit_pairs = [(d, b) for (d, b, _) in MEASURED]
        measured_vec = np.array([s for (_, _, s) in MEASURED], dtype=float)
        s0 = S0_SYNTH
        sigma_raw = S0_SYNTH / SNR0
        raw_vec = measured_vec * s0
        measured_desc = "S/S0:  " + ",  ".join(
            f"Δ{d:g}/b{b:g}={s:g}" for (d, b, s) in MEASURED)

    print(f"  Features:  {len(fit_pairs)} → {fit_pairs}")
    print(f"  {measured_desc}")
    print(f"  Initial vi window: [{VI_MIN}, {VI_MAX}]  (movable in viewer)   "
          f"rho_max={RHO_MAX}   log_space={LOG_SPACE}   SNR₀={SNR0:g}")
    print(f"  Bayesian weight panel: sigma_m init={SIGMA_M_INIT:g} "
          f"[{SIGMA_M_MIN:g}, {SIGMA_M_MAX:g}]   fit_s0_default={FIT_S0_DEFAULT}   "
          f"rician_default={RICIAN_W_DEFAULT}   sigma_raw={sigma_raw:.4g}")

    fig, payload, stats = build(entries, meta, measured_vec, fit_pairs, raw_vec, s0, sigma_raw)
    loss_no, loss_ri, cand0, kio, rho, V, vi = stats
    print(f"  Plotted entries: {len(kio)}   "
          f"(vi data range [{vi.min():.3f}, {vi.max():.3f}])")
    print(f"  In initial window: {int(cand0.sum())}")
    if cand0.any():
        j = int(np.where(cand0)[0][np.argmin(loss_no[cand0])])
        print(f"  [No-Rician] initial best: kio={kio[j]:g}, rho={rho[j]/1e3:.0f}k, "
              f"V={V[j]:g}  (vi={vi[j]:.3f}, loss={loss_no[j]:.4g})")

    html_path = os.path.join(args.out, "error_landscape_3d.html")
    write_html(fig, payload, title=f"MADI error landscape — {lib_name}",
               subtitle=measured_desc, out_path=html_path)
    print(f"\n  Saved interactive viewer: {html_path}")
    print("  Drag to rotate, scroll to zoom, right-drag to pan.")
    print("  Click a point to compare its simulated decay to the measured curve.")
    print("\nDone.")


if __name__ == "__main__":
    main()

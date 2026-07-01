#!/usr/bin/env python3
"""
view_error_landscape_3d.py — Interactive 3D MADI error-landscape viewer
=======================================================================

Builds a self-contained interactive HTML.  Every DISCRETE library entry is
plotted at its (kio, rho, V) coordinate and coloured by the standard MADI
matcher loss (``match_voxels_batch``, fixed-S0) for a manually specified
normalized decay.  Open the HTML in any browser and orbit / pan / zoom —
no server, no internet, no Python needed to view it.

Controls (top bar, so nothing overlaps the plot):
    • Rician        : toggle No-Rician / Rician colouring.
    • Log colour    : colour by log10(loss) for dynamic range across decades.
    • vi min / max  : move the candidate window live; entries with
                      vi=(rho/1e9)*(V*1e3) inside [min,max] are coloured
                      candidates, the rest are "excluded".  The best-fit
                      marker re-finds the minimum within the window.
    • Show excluded : show/hide the out-of-window entries (faint grey).
    • Contrast      : two sliders set the colourbar min / max.  "Reset"
                      restores defaults.
    • Dot size      : marker size.

Loss & Rician semantics match plot_error_landscape.py exactly:
    optional log-transform:  x -> log(clip(x, s_floor, 1.0))   (if LOG_SPACE)
    loss(entry) = || measured_subset - library_vector_subset ||^2

Requires plotly to GENERATE (pip install plotly).  The output HTML is standalone.

Usage
-----
    python view_error_landscape_3d.py \
        --library data/libraries/madi_dense_human.npz \
        --out     error_landscape_3d_out

    python view_error_landscape_3d.py \
        --library /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/data/libraries/madi_dense_human.npz \
        --out     /mnt/c/miscellaneous/coding_projects/Python/mri_processing/processing/madi_gpu/Custom_MADI/figures/error_3dlandscape_out    
"""

import argparse
import json
import os
import numpy as np


# ===================================================================
# CONFIG
# ===================================================================

# Normalized decay to fit:  (Δ [ms], b [s/mm²], S/S0).
# Default = Grey-Matter curve from b_space_map_125.png.
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

SNR0 = 30.0                 # b=0 SNR (= S0/sigma) for the Rician colouring

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

def build(entries, meta, measured_vec, fit_pairs):
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
  #bar h1{font-size:15px;margin:0 12px 0 0;font-weight:650;}
  #bar .sub{font-size:11.5px;color:var(--mut);font-weight:400;}
  .grp{display:flex;align-items:center;gap:7px;}
  .grp label{font-size:12.5px;color:var(--tx);white-space:nowrap;}
  .seg{display:inline-flex;border:1px solid var(--bd);border-radius:7px;overflow:hidden;}
  .seg button{border:0;background:#fff;padding:5px 12px;font-size:12.5px;cursor:pointer;color:var(--tx);}
  .seg button.on{background:var(--acc);color:#fff;}
  input[type=range]{width:130px;}
  .val{font-variant-numeric:tabular-nums;font-size:11.5px;color:var(--mut);min-width:54px;display:inline-block;}
  .btn{border:1px solid var(--bd);background:#fff;border-radius:7px;padding:5px 11px;font-size:12.5px;cursor:pointer;}
  .btn:hover{background:#f0f4fb;}
  #gd{flex:1 1 auto;min-height:0;}
  .chk{display:flex;align-items:center;gap:6px;font-size:12.5px;cursor:pointer;white-space:nowrap;}
  .sep{width:1px;height:24px;background:var(--bd);}
</style></head>
<body><div id="wrap">
  <div id="bar">
    <div><h1>__TITLE__</h1></div>
    <div class="grp"><span class="sub">__SUB__</span></div>
    <div class="sep"></div>
    <div class="grp"><label>Rician</label>
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
  __FRAG__
</div>
<script>
(function(){
  const D = __DATA__;
  const gd = document.getElementById('gd');
  const CAND = 0, BEST = 1, GREY = 2;
  let mode = 'no';
  let logc = !!D.defaultLog;
  let touched = false;

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

    const active=activeLoss();
    const dvals=disp(cand.map(i=>active[i]));
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
    const label = logc ? 'log\\u2081\\u2080 loss' : 'loss<br>\\u2016m\\u2212r\\u2016\\u00B2';

    Plotly.restyle(gd,{
      x:[cand.map(i=>D.kio[i])], y:[cand.map(i=>D.y[i])], z:[cand.map(i=>D.V[i])],
      text:[cand.map(i=>hov(i))],
      'marker.color':[dvals], 'marker.cmin':cmin, 'marker.cmax':cmax,
      'marker.colorbar.title.text':label
    },[CAND]);

    const showOut=document.getElementById('showChk').checked;
    const gi=showOut?out:[];
    Plotly.restyle(gd,{ x:[gi.map(i=>D.kio[i])], y:[gi.map(i=>D.y[i])],
      z:[gi.map(i=>D.V[i])], text:[gi.map(i=>hov(i))] },[GREY]);

    if(cand.length){
      let bi=cand[0],bl=active[bi];
      for(const i of cand){ if(active[i]<bl){bl=active[i];bi=i;} }
      const bt='BEST ('+(mode==='no'?'No Rician':'Rician')+')<br>k_io='+g(D.kio[bi])+
        ', \\u03C1='+(D.rho[bi]/1e3).toFixed(0)+'k, V='+g(D.V[bi])+
        '<br>loss='+active[bi].toPrecision(4)+', vi='+D.vi[bi].toFixed(3);
      Plotly.restyle(gd,{x:[[D.kio[bi]]],y:[[D.y[bi]]],z:[[D.V[bi]]],text:[[bt]]},[BEST]);
    } else {
      Plotly.restyle(gd,{x:[[]],y:[[]],z:[[]],text:[['']]},[BEST]);
    }

    const showLo=logc?Math.pow(10,cmin):cmin, showHi=logc?Math.pow(10,cmax):cmax;
    document.getElementById('cMinVal').textContent=showLo.toPrecision(3);
    document.getElementById('cMaxVal').textContent=showHi.toPrecision(3);
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

  document.getElementById('mNo').onclick=()=>setMode('no');
  document.getElementById('mRi').onclick=()=>setMode('ri');
  document.getElementById('logChk').onchange=function(){ logc=this.checked; rebuild(!touched); };
  document.getElementById('viMin').oninput=()=>rebuild(!touched);
  document.getElementById('viMax').oninput=()=>rebuild(!touched);
  document.getElementById('showChk').onchange=()=>rebuild(false);
  document.getElementById('cMin').oninput=()=>{ touched=true; rebuild(false); };
  document.getElementById('cMax').oninput=()=>{ touched=true; rebuild(false); };
  document.getElementById('resetBtn').onclick=()=>rebuild(true);
  document.getElementById('dotSz').oninput=resize;

  function clamp01(x){ return Math.max(0,Math.min(1,x)); }
  function init(){
    if(!gd || !gd.data){ return setTimeout(init,40); }
    const sp=(D.viDataMax-D.viDataMin)||1e-9;
    document.getElementById('viMin').value=Math.round(clamp01((D.viMinDefault-D.viDataMin)/sp)*1000);
    document.getElementById('viMax').value=Math.round(clamp01((D.viMaxDefault-D.viDataMin)/sp)*1000);
    document.getElementById('showChk').checked=D.showOutDefault;
    document.getElementById('logChk').checked=logc;
    document.getElementById('dotSz').value=D.sizeDefault;
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
    ap = argparse.ArgumentParser(
        description="Interactive 3D MADI error-landscape viewer (HTML).")
    ap.add_argument("--library", required=True, help="Path to MADI .npz library.")
    ap.add_argument("--out", default="error_landscape_3d_out",
                    help="Output directory for the HTML.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 64)
    print("MADI 3D error-landscape viewer")
    print("=" * 64)
    entries, meta = load_library_npz(args.library)
    lib_name = os.path.basename(args.library)
    print(f"  Library:   {args.library}  ({len(entries['kio'])} entries)")
    print(f"  Δ [ms]:    {[f'{d:g}' for d in meta['deltas']]}")
    print(f"  b [s/mm²]: {[f'{b:g}' for b in meta['b_values']]}  (n_b={meta['n_b']})")

    fit_pairs = [(d, b) for (d, b, _) in MEASURED]
    measured_vec = np.array([s for (_, _, s) in MEASURED], dtype=float)
    measured_desc = "S/S0:  " + ",  ".join(
        f"Δ{d:g}/b{b:g}={s:g}" for (d, b, s) in MEASURED)
    print(f"  Features:  {len(fit_pairs)} → {fit_pairs}")
    print(f"  {measured_desc}")
    print(f"  Initial vi window: [{VI_MIN}, {VI_MAX}]  (movable in viewer)   "
          f"rho_max={RHO_MAX}   log_space={LOG_SPACE}   SNR₀={SNR0:g}")

    fig, payload, stats = build(entries, meta, measured_vec, fit_pairs)
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
    print("\nDone.")


if __name__ == "__main__":
    main()

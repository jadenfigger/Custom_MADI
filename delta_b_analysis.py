"""
delta_b_analysis.py — Multi-Δ × multi-b decay curve diagnostics
================================================================

Generates one multi-panel figure that simultaneously demonstrates:

  1. PURE-WATER SANITY   — for a no-cell ensemble, S(b)/S₀ at every Δ
     must collapse onto exp(−b·D₀). Validates that compute_signals +
     G_from_b are doing the right thing at each Δ independently.

  2. Δ-DEPENDENCE OF S(b) AT EACH FIXED b — how strongly does varying
     Δ (for the same nominal b) change the measured signal? Tells you
     where in the (Δ, b) grid you have the most "Δ leverage".

  3. PARAMETER SENSITIVITY OVER THE (Δ, b) GRID — finite-difference
     log-derivatives |∂ ln S / ∂ ln θ| for θ ∈ {k_io, ρ, V}, drawn as
     three heatmaps. Identifies which acquisition points carry the
     most information for which parameter, and is directly relevant
     to fitting identifiability.

PRIMER ON CT vs CG (Springer et al. NMRBM 2023, MADI I §3.4 + Paper II SI §S.II)
-------------------------------------------------------------------------------
SDE-CT (constant time):
    t_D fixed, G incremented to sweep b;  b = (γGδ)²·t_D = q²·t_D.
    Each Δ-row of your multi-Δ library is one CT sweep at t_D = Δ − δ/3.

SDE-CG (constant gradient):
    G fixed, t_D incremented;  b varies linearly with t_D.
    At small b the CT and CG decays are identical, but at large b the
    CG decay becomes "articulated" and ultimately reaches a
    non-Gaussian/pseudo-Gaussian switchover where its slope = −k_io.
    Your library is NOT a true CG sweep — for each (Δ, b) it computes
    a separate G. This script optionally adds a true CG sweep with
    --cg.

DEFAULTS (edit at the top of this file):
    Baseline tissue : brain GM from Paper II Table 2
    Perturbation    : ±15% central differences
    b grid          : 13 values from 0 to 6000 s/mm²
    Δ grid          : your library's [15, 25, 30, 40] ms
    True CG sweep   : disabled (use --cg to enable)

USAGE
-----
    python delta_b_analysis.py              # default, ~3-10 min
    python delta_b_analysis.py --cg         # add true CG sweep
    python delta_b_analysis.py --quick      # fewer walkers, ~1-3 min
    python delta_b_analysis.py --baseline wm
    python delta_b_analysis.py --out my.png

Heads up: first run will trigger the v_i → (α*, ⟨A/V⟩) lookup-table
build inside madi.ensemble (~10–20 min, cached). Subsequent runs reuse
the cache.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make the madi package importable whether this file is in the repo root
# or one directory above.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from madi.config import SimConfig, D0_UM2_MS
from madi.walker_gpu import run_simulation, HAS_CUDA
from madi.signal import compute_signals, G_from_b


# ===========================================================================
# Defaults — edit these to change the experiment
# ===========================================================================

# Tissue baselines (ρ in cells/μL, V in pL, k_io in s⁻¹)
TISSUES = {
    "gm": dict(name="Brain GM",       kio=6.6,  rho=1.10e5, V=6.00),
    "wm": dict(name="Brain WM",       kio=22.0, rho=6.90e5, V=0.91),
    "th": dict(name="Thalamus",       kio=22.0, rho=6.20e5, V=1.20),
    "pu": dict(name="Putamen",        kio=17.0, rho=2.90e5, V=2.60),
    "pl": dict(name="Prostate lesion",kio=39.0, rho=1.20e6, V=0.54),
}

DELTAS_MS  = [15.0, 25.0, 30.0, 40.0]                      # your library's Δ
B_GRID     = np.linspace(0.0, 6000.0, 13)                  # s/mm²
PERTURB    = 0.15                                          # ±15% central FD
DELTA_PFG  = 6.0                                           # δ [ms]


# ===========================================================================
# Config builder
# ===========================================================================

def build_cfg(quick: bool) -> SimConfig:
    """Build a SimConfig appropriate to the available hardware."""
    if HAS_CUDA:
        # GPU: large batches are essentially free
        n_walkers = 120_000
        n_ens     = 4
    elif quick:
        n_walkers = 3_000
        n_ens     = 1
    else:
        n_walkers = 8_000
        n_ens     = 1

    # 46_000 steps × 1 μs = 46 ms — just covers Δ_max=40 + δ=6
    return SimConfig(
        n_walkers   = n_walkers,
        n_ensembles = n_ens,
        n_steps     = 46_000,
        Deltas      = list(DELTAS_MS),
        delta       = DELTA_PFG,
    )


# ===========================================================================
# Single-condition simulation helper
# ===========================================================================

def simulate_condition(label: str, rho: float, V: float, kio: float,
                       cfg: SimConfig, seed: int) -> dict:
    """Run one walker simulation and return signals on the dense B_GRID."""
    is_water = (rho <= 0 or V <= 0)
    print(f"  [{label:<18}] ρ={rho:>10.0f}  V={V:>5.2f}  k_io={kio:>6.2f}  ",
          end="", flush=True)
    t0 = time.time()
    wr = run_simulation(rho if not is_water else 0.0,
                        V   if not is_water else 0.0,
                        kio,
                        cfg, seed=seed, verbose=False)
    res = compute_signals(wr, cfg, b_values_s_mm2=B_GRID)
    print(f"... {time.time()-t0:5.1f}s   "
          f"(N_per_axis = {wr.n_walkers_per_axis})")
    return res


# ===========================================================================
# Acquisition-grid summary table
# ===========================================================================

def print_acquisition_grid():
    """Show the actual gradient strength at every (Δ, b) cell."""
    print("\nAcquisition (Δ, b) → G(mT/m) grid")
    print(f"  δ_PFG = {DELTA_PFG} ms,  D_0 = {D0_UM2_MS} μm²/ms")
    print(f"  b values [s/mm²]: {[float(b) for b in B_GRID]}")
    print()
    header = "  Δ \\ b  " + "".join(f"{b:>7.0f}" for b in B_GRID)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for d in DELTAS_MS:
        row = "  " + f"{d:>5.0f}  "
        for b in B_GRID:
            G = G_from_b(b, DELTA_PFG, d)        # G in T/m
            row += f"{G*1000:>7.1f}"             # → mT/m
        print(row)
    print()


# ===========================================================================
# Sensitivity (log-derivative via central difference)
# ===========================================================================

def log_sensitivity(s_plus: np.ndarray, s_minus: np.ndarray,
                    s_base: np.ndarray) -> np.ndarray:
    """|d ln S / d ln θ| ≈ |S_+ − S_−| / (2 · δ · |S_base|),
    where δ = PERTURB is the relative perturbation. Returns same shape as input."""
    safe = np.where(np.abs(s_base) > 1e-3, np.abs(s_base), 1e-3)
    return np.abs(s_plus - s_minus) / (2.0 * PERTURB * safe)


# ===========================================================================
# Run everything
# ===========================================================================

def run_all_conditions(baseline: dict, cfg: SimConfig) -> dict:
    """Run pure water + baseline + 6 perturbed (3 params × ±)."""
    results = {}

    print("\n=== Running simulations ===\n")
    seed_base = 0xC0DEB000

    results["water"] = simulate_condition(
        "pure water", 0.0, 0.0, np.inf, cfg, seed=seed_base + 0)

    results["base"] = simulate_condition(
        "baseline tissue", baseline["rho"], baseline["V"], baseline["kio"],
        cfg, seed=seed_base + 100)

    for i, p in enumerate(("kio", "rho", "V")):
        for j, sign in enumerate((+1, -1)):
            mult = 1.0 + sign * PERTURB
            theta = dict(baseline)
            theta[p] = baseline[p] * mult
            tag = f"{p}{'+' if sign > 0 else '-'}"
            results[tag] = simulate_condition(
                tag, theta["rho"], theta["V"], theta["kio"],
                cfg, seed=seed_base + 200 + 10 * i + j)

    return results


def run_cg_sweep(baseline: dict, cfg: SimConfig,
                 G_mT_per_m: float = 200.0,
                 n_delta: int = 12) -> dict:
    """OPTIONAL: true SDE-CG sweep at fixed G, sweeping Δ.

    Builds a custom cfg with many Δ values, runs ONE walker simulation
    on the baseline tissue, then computes signals at the b values that
    correspond to the chosen fixed G.
    """
    print(f"\n=== Running SDE-CG sweep (G={G_mT_per_m:.0f} mT/m, "
          f"{n_delta} Δ values) ===\n")

    # Δ values evenly spaced from 10 to 50 ms
    deltas_cg = np.linspace(10.0, 50.0, n_delta).tolist()

    # Need walker steps to cover the longest Δ + δ
    n_steps_cg = int((max(deltas_cg) + DELTA_PFG + 5.0) / cfg.ts)
    cfg_cg = SimConfig(
        n_walkers   = cfg.n_walkers,
        n_ensembles = cfg.n_ensembles,
        n_steps     = n_steps_cg,
        Deltas      = deltas_cg,
        delta       = DELTA_PFG,
    )

    # b for each Δ at fixed G:  b = (γGδ)² · (Δ − δ/3)
    from madi.config import GAMMA_RAD
    G_si = G_mT_per_m * 1e-3
    delta_si = DELTA_PFG * 1e-3
    b_per_delta_si = []
    for d in deltas_cg:
        tD_si = (d - DELTA_PFG / 3.0) * 1e-3
        b_per_delta_si.append((GAMMA_RAD * G_si * delta_si) ** 2 * tD_si)
    b_per_delta_s_mm2 = np.array(b_per_delta_si) / 1e6

    print(f"  Δ values [ms]: {[f'{d:.1f}' for d in deltas_cg]}")
    print(f"  → b values  [s/mm²]: {[f'{b:.0f}' for b in b_per_delta_s_mm2]}")

    res = simulate_condition(
        "CG (baseline)",
        baseline["rho"], baseline["V"], baseline["kio"],
        cfg_cg, seed=0xCCCC)

    # Pure water at the same Δ grid for reference
    res_w = simulate_condition(
        "CG (water)", 0.0, 0.0, np.inf, cfg_cg, seed=0xDDDD)

    return dict(
        deltas      = deltas_cg,
        b_per_delta = b_per_delta_s_mm2,
        signals_tis = res["signals"],   # (n_delta, n_b_GRID)
        signals_wat = res_w["signals"],
        G_mT_per_m  = G_mT_per_m,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def make_main_figure(results: dict, baseline: dict, cfg: SimConfig,
                     cg_results: dict | None,
                     save_path: str):
    """Six (or seven) panel diagnostic figure."""
    n_d = len(DELTAS_MS)

    sig_water = results["water"]["signals"]   # (n_d, n_b)
    sig_base  = results["base"]["signals"]    # (n_d, n_b)

    # --- compute sensitivities ---
    sens_kio = log_sensitivity(results["kio+"]["signals"],
                               results["kio-"]["signals"], sig_base)
    sens_rho = log_sensitivity(results["rho+"]["signals"],
                               results["rho-"]["signals"], sig_base)
    sens_V   = log_sensitivity(results["V+"]["signals"],
                               results["V-"]["signals"], sig_base)
    vmax = float(max(sens_kio.max(), sens_rho.max(), sens_V.max()))

    # --- figure layout ---
    if cg_results is None:
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.55)
        ax_water = fig.add_subplot(gs[0, 0:3])
        ax_decay = fig.add_subplot(gs[0, 3:6])
        ax_dfan  = fig.add_subplot(gs[1, 0:6])
        ax_sk    = fig.add_subplot(gs[2, 0:2])
        ax_sr    = fig.add_subplot(gs[2, 2:4])
        ax_sv    = fig.add_subplot(gs[2, 4:6])
    else:
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.5, wspace=0.55)
        ax_water = fig.add_subplot(gs[0, 0:3])
        ax_decay = fig.add_subplot(gs[0, 3:6])
        ax_dfan  = fig.add_subplot(gs[1, 0:6])
        ax_sk    = fig.add_subplot(gs[2, 0:2])
        ax_sr    = fig.add_subplot(gs[2, 2:4])
        ax_sv    = fig.add_subplot(gs[2, 4:6])
        ax_cg    = fig.add_subplot(gs[3, 0:6])

    delta_cmap = plt.cm.plasma
    delta_colors = [delta_cmap(i / max(n_d - 1, 1)) for i in range(n_d)]

    # ----------------------------------------------------------------
    # Panel 1 — Pure water sanity: should collapse onto exp(−b·D0)
    # ----------------------------------------------------------------
    b_dense = np.linspace(B_GRID.min(), B_GRID.max(), 200)
    theory = np.exp(-(b_dense / 1e3) * D0_UM2_MS)


    ax_water.semilogy(b_dense, theory, "k--", lw=1.5, alpha=0.5,
                      label=r"theory: $e^{-b D_0}$")
    for di, d in enumerate(DELTAS_MS):
        ax_water.semilogy(B_GRID, np.clip(sig_water[di], 1e-4, None),
                          "o-", color=delta_colors[di], ms=5,
                          label=f"Δ = {d:.0f} ms")
    ax_water.set_xlabel(r"$b$  [s/mm$^2$]")
    ax_water.set_ylabel(r"$S(b)/S_0$")
    ax_water.set_title("(1) Pure water sanity\n"
                       "all Δ should collapse onto $e^{-b D_0}$")
    ax_water.legend(fontsize=8, loc="lower left")
    ax_water.grid(True, alpha=0.3)
    ax_water.set_ylim(1e-3, 1.5)

    # ----------------------------------------------------------------
    # Panel 2 — Baseline tissue CT-style decays at each Δ
    # ----------------------------------------------------------------
    for di, d in enumerate(DELTAS_MS):
        ax_decay.semilogy(B_GRID, np.clip(sig_base[di], 1e-4, None),
                          "o-", color=delta_colors[di], ms=5,
                          label=f"Δ = {d:.0f} ms")
    ax_decay.semilogy(b_dense, theory, "k--", lw=1, alpha=0.4,
                      label="H$_2$O free")
    ax_decay.set_xlabel(r"$b$  [s/mm$^2$]")
    ax_decay.set_ylabel(r"$S(b)/S_0$")
    ax_decay.set_title(f"(2) {baseline['name']} tissue decays at each Δ\n"
                       f"$k_{{io}}={baseline['kio']:.1f}$, "
                       f"$\\rho={baseline['rho']/1e3:.0f}$k, "
                       f"$V={baseline['V']:.1f}$ pL")
    ax_decay.legend(fontsize=8, loc="lower left")
    ax_decay.grid(True, alpha=0.3)
    ax_decay.set_ylim(bottom=max(sig_base.min() * 0.5, 1e-4))

    # ----------------------------------------------------------------
    # Panel 3 — Δ-dependence of S/S0 at each fixed b (answers Q2)
    # Lines are color-coded by b value; points sit at the 4 library Δ.
    # ----------------------------------------------------------------
    b_indices = [i for i, b in enumerate(B_GRID) if b > 0]   # skip b=0 (≡1)
    b_cmap = plt.cm.viridis
    norm_b = plt.Normalize(vmin=B_GRID[b_indices[0]], vmax=B_GRID[-1])

    for bi in b_indices:
        b = B_GRID[bi]
        col = b_cmap(norm_b(b))
        ax_dfan.plot(DELTAS_MS, sig_base[:, bi], "o-",
                     color=col, lw=1.6, ms=5)
    sm = plt.cm.ScalarMappable(cmap=b_cmap, norm=norm_b)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_dfan, fraction=0.025, pad=0.01)
    cb.set_label(r"$b$  [s/mm$^2$]")
    ax_dfan.set_xlabel(r"$\Delta$  [ms]")
    ax_dfan.set_ylabel(r"$S(b)/S_0$")
    ax_dfan.set_title(
        f"(3) Δ-dependence of $S(b)/S_0$ for {baseline['name']}\n"
        f"each curve is one fixed $b$; dispersion across Δ shows the "
        f"non-Gaussian regime")
    ax_dfan.set_xticks(DELTAS_MS)
    ax_dfan.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Panels 4–6 — Sensitivity heatmaps (answers Q3)
    # ----------------------------------------------------------------
    def _heatmap(ax, data, title):
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(B_GRID)))
        ax.set_xticklabels([f"{b:.0f}" for b in B_GRID],
                           rotation=45, fontsize=7)
        ax.set_yticks(np.arange(len(DELTAS_MS)))
        ax.set_yticklabels([f"{d:.0f}" for d in DELTAS_MS])
        ax.set_xlabel(r"$b$  [s/mm$^2$]")
        ax.set_ylabel(r"$\Delta$  [ms]")
        ax.set_title(title)
        # Annotate each cell with its value (only if not too small)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if val > 0.01 * vmax:
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center",
                            color="white" if val < 0.6 * vmax else "black",
                            fontsize=6)
        return im

    im_k = _heatmap(ax_sk, sens_kio,
                    r"(4) $|\partial \ln S / \partial \ln k_{io}|$")
    im_r = _heatmap(ax_sr, sens_rho,
                    r"(5) $|\partial \ln S / \partial \ln \rho|$")
    im_v = _heatmap(ax_sv, sens_V,
                    r"(6) $|\partial \ln S / \partial \ln V|$")

    # Shared colorbar at the right of the heatmap row
    cb_sens = fig.colorbar(im_v, ax=[ax_sk, ax_sr, ax_sv],
                           fraction=0.018, pad=0.02)
    cb_sens.set_label(
        f"sensitivity  (fractional ΔS per fractional Δθ, ±{int(PERTURB*100)}% FD)"
    )

    # ----------------------------------------------------------------
    # OPTIONAL Panel 7 — true SDE-CG sweep
    # ----------------------------------------------------------------
    if cg_results is not None:
        # The cg results contain signals at all Δ in the cg cfg, but only the
        # diagonal element (the b that corresponds to that Δ at fixed G) is
        # the true CG point.
        deltas_cg = np.array(cg_results["deltas"])
        b_per_d = cg_results["b_per_delta"]
        sig_cg_tis = cg_results["signals_tis"]   # (n_delta_cg, n_b_GRID)
        sig_cg_wat = cg_results["signals_wat"]

        # For each Δ, interpolate the signal at its fixed-G b value
        cg_curve_tis = np.array([
            np.interp(b_per_d[i], B_GRID, sig_cg_tis[i])
            for i in range(len(deltas_cg))
        ])
        cg_curve_wat = np.array([
            np.interp(b_per_d[i], B_GRID, sig_cg_wat[i])
            for i in range(len(deltas_cg))
        ])

        ax_cg.semilogy(b_per_d, np.clip(cg_curve_tis, 1e-4, None),
                       "o-", color="darkred", lw=2, ms=6,
                       label=f"CG sweep at G = {cg_results['G_mT_per_m']:.0f} "
                             f"mT/m  ({baseline['name']})")
        ax_cg.semilogy(b_per_d, np.clip(cg_curve_wat, 1e-4, None),
                       "s--", color="gray", ms=5, alpha=0.7,
                       label="CG sweep at G = same  (pure water)")
        # CT reference (all of it)
        for di, d in enumerate(DELTAS_MS):
            ax_cg.semilogy(B_GRID, np.clip(sig_base[di], 1e-4, None),
                           "-", color=delta_colors[di], lw=1, alpha=0.45,
                           label=f"CT Δ = {d:.0f} ms")
        ax_cg.set_xlabel(r"$b$  [s/mm$^2$]")
        ax_cg.set_ylabel(r"$S(b)/S_0$")
        ax_cg.set_title("(7) True SDE-CG vs. CT — at large $b$ the CG decay "
                        "becomes articulated and approaches slope $-k_{io}$")
        ax_cg.legend(fontsize=7, loc="lower left", ncol=2)
        ax_cg.grid(True, alpha=0.3)

    fig.suptitle(
        f"Multi-Δ × multi-b decay analysis  |  baseline: {baseline['name']}",
        fontsize=14, y=0.995)

    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"\n  → saved {save_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("USAGE")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="gm",
                        choices=list(TISSUES.keys()),
                        help="Tissue baseline (default: gm)")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer walkers (faster, noisier)")
    parser.add_argument("--cg", action="store_true",
                        help="Add a true SDE-CG sweep panel")
    parser.add_argument("--cg-G", type=float, default=200.0,
                        help="Fixed G [mT/m] for the CG sweep")
    parser.add_argument("--cg-n", type=int, default=12,
                        help="Number of Δ values in the CG sweep")
    parser.add_argument("--out", default="delta_b_analysis.png",
                        help="Output figure path")
    args = parser.parse_args()

    baseline = TISSUES[args.baseline]
    cfg = build_cfg(quick=args.quick)

    print(f"\n{'='*70}")
    print(f" delta_b_analysis.py")
    print(f"{'='*70}")
    print(f"  CUDA available  : {HAS_CUDA}")
    print(f"  baseline tissue : {baseline['name']}  "
          f"(k_io={baseline['kio']}, ρ={baseline['rho']:.0f}, V={baseline['V']})")
    print(f"  v_i             : {baseline['rho'] * baseline['V'] * 1e-6:.3f}")
    print(f"  Δ values        : {DELTAS_MS} ms  ({len(DELTAS_MS)})")
    print(f"  b values        : {len(B_GRID)} from {B_GRID[0]:.0f} "
          f"to {B_GRID[-1]:.0f} s/mm²")
    print(f"  perturbation    : ±{PERTURB*100:.0f}% (central FD)")
    print(f"  walkers/cond.   : "
          f"{3 * cfg.n_ensembles * cfg.n_walkers:,}  "
          f"({cfg.n_walkers} × {cfg.n_ensembles} ens × 3 axes)")
    print(f"  walk duration   : {cfg.tRW_max:.0f} ms "
          f"({cfg.n_steps} steps × {cfg.ts*1000:.0f} ns)")

    # First-run cache warning
    cache = os.path.expanduser("~/.cache/madi")
    if not os.path.isdir(cache) or not os.listdir(cache):
        print()
        print("  ⚠ No MADI lookup-table cache found in ~/.cache/madi —")
        print("    the first ensemble construction will trigger a ~10–20 min")
        print("    one-time table build. Subsequent runs are fast.")

    print_acquisition_grid()

    t0 = time.time()
    results = run_all_conditions(baseline, cfg)
    cg_results = run_cg_sweep(baseline, cfg,
                              G_mT_per_m=args.cg_G,
                              n_delta=args.cg_n) if args.cg else None
    elapsed = time.time() - t0

    print(f"\n=== Total simulation time: {elapsed:.1f} s ===")
    print(f"\nGenerating figure...")
    make_main_figure(results, baseline, cfg, cg_results, args.out)
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()

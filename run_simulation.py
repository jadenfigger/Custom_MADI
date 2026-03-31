#!/usr/bin/env python3
"""
run_simulation.py — MADI simulation with GPU acceleration
==========================================================

Reproduces Figure 4 of Springer et al. (2023) using YOUR acquisition
parameters: delta=6ms, Delta=15/25/30/40ms, b=1000/2500/4000/6000 s/mm2.

Usage
-----
    python run_simulation.py --minimal       # ~2 min  (GPU)  / ~10 min (CPU)
    python run_simulation.py                 # ~10 min (GPU)  / ~2 hr   (CPU)
    python run_simulation.py --full          # ~1 hr   (GPU)  / ~12 hr  (CPU)
"""

import argparse, time, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from madi.config      import SimConfig
from madi.walker_gpu   import run_simulation as run_sim, HAS_CUDA
from madi.signal       import compute_signals, compute_adc
from madi.plotting     import plot_decays_multidelta, plot_parameter_sensitivity, plot_ensemble_slice
from madi.ensemble     import create_ensemble, estimate_vi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===================================================================
# Parameter sets (Figure 4 analog, mouse-appropriate ranges)
# ===================================================================

PANEL_A_RHO, PANEL_A_V = 400_000, 2.00
PANEL_A_KIOS = [3, 8, 12, 25, 50]

PANEL_B_KIO, PANEL_B_V = 12.0, 2.00
PANEL_B_RHOS = [100_000, 200_000, 300_000, 450_000]

PANEL_C_KIO, PANEL_C_RHO = 12.0, 400_000
PANEL_C_VS = [0.5, 1.0, 1.5, 2.0]


# ===================================================================
# Presets
# ===================================================================

def make_config(preset):
    if preset == "minimal":
        return SimConfig(n_walkers=2_000, n_ensembles=1, n_steps=50_000,
                         L=150.0, buffer=40.0, grid_spacing=1.5)
    elif preset == "default":
        return SimConfig(n_walkers=10_000, n_ensembles=3, n_steps=50_000,
                         L=200.0, buffer=50.0, grid_spacing=1.0)
    elif preset == "full":
        return SimConfig(n_walkers=30_000, n_ensembles=8, n_steps=50_000,
                         L=300.0, buffer=70.0, grid_spacing=0.8)
    raise ValueError(preset)


# ===================================================================
# Helpers
# ===================================================================

def run_one(rho, V, kio, cfg, seed=0, label=""):
    t0 = time.time()
    wr = run_sim(rho, V, kio, cfg, seed=seed, verbose=False)
    res = compute_signals(wr, cfg)
    dt = time.time() - t0
    adc = compute_adc(res['b_values'], res['signals'][1])  # Delta=25ms
    print(f"  {label:40s}  ADC={adc:.3f} um2/ms  ({dt:.1f}s)")
    return res


def run_panel(name, values, fixed, cfg, seed_base=0):
    curves = []
    for i, val in enumerate(values):
        if name == "kio":
            r, v, k = fixed['rho'], fixed['V'], val
            lab = f"kio={val:.0f} s-1"
        elif name == "rho":
            r, v, k = val, fixed['V'], fixed['kio']
            lab = f"rho={val/1e3:.0f}k"
        elif name == "V":
            r, v, k = fixed['rho'], val, fixed['kio']
            lab = f"V={val:.1f} pL"
        else:
            raise ValueError(name)
        res = run_one(r, v, k, cfg, seed=seed_base + i*10000, label=lab)
        curves.append((res, lab))
    return curves


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--minimal", action="store_true")
    ap.add_argument("--output", default="madi_figure4.png")
    args = ap.parse_args()

    preset = "full" if args.full else ("minimal" if args.minimal else "default")
    cfg = make_config(preset)

    print("=" * 60)
    print("MADI-GPU Simulation")
    print("=" * 60)
    print(f"  CUDA available  : {HAS_CUDA}")
    print(f"  Preset          : {preset}")
    print(f"  Walkers         : {cfg.n_walkers} x {cfg.n_ensembles} ensembles")
    print(f"  Steps           : {cfg.n_steps}  ({cfg.tRW_max:.0f} ms)")
    print(f"  Domain          : {cfg.L:.0f} um,  grid {cfg.grid_size}^3")
    print(f"  delta           : {cfg.delta} ms")
    print(f"  Deltas          : {cfg.Deltas} ms")
    print(f"  b-values        : {list(map(int, [1000,2500,4000,6000]))} s/mm2")
    print()

    t_total = time.time()

    # --- Pure water verification ---
    print("=== Pure water verification ===")
    wr0 = run_sim(0, 0, np.inf, cfg, seed=99, verbose=False)
    res0 = compute_signals(wr0, cfg)
    for di, D in enumerate(cfg.Deltas):
        adc = compute_adc(res0['b_values'], res0['signals'][di])
        print(f"  Delta={D:2.0f}ms  ADC={adc:.3f} um2/ms  (expect ~3.0)")

    # --- Ensemble inspection ---
    print(f"\n=== Ensemble (rho={PANEL_A_RHO}, V={PANEL_A_V}) ===")
    ens = create_ensemble(PANEL_A_RHO, PANEL_A_V, cfg, seed=42)
    vi_num = estimate_vi(ens)
    print(f"  Cells: {len(ens.seeds)}, vi_target={ens.vi:.3f}, vi_numerical={vi_num:.3f}")

    fig_e, ax_e = plt.subplots(figsize=(6,6))
    plot_ensemble_slice(ens, ax=ax_e)
    fig_e.savefig("ensemble_slice.png", dpi=150, bbox_inches="tight"); plt.close(fig_e)
    print("  -> ensemble_slice.png")

    # --- Panel simulations ---
    print("\n=== Panel (a): varying kio ===")
    pa = run_panel("kio", PANEL_A_KIOS, {"rho": PANEL_A_RHO, "V": PANEL_A_V}, cfg, 100)

    print("\n=== Panel (b): varying rho ===")
    pb = run_panel("rho", PANEL_B_RHOS, {"kio": PANEL_B_KIO, "V": PANEL_B_V}, cfg, 200)

    print("\n=== Panel (c): varying V ===")
    pc = run_panel("V", PANEL_C_VS, {"kio": PANEL_C_KIO, "rho": PANEL_C_RHO}, cfg, 300)

    # --- Multi-delta plot for the "brain-like" case ---
    print("\n=== Multi-delta decay (kio=12, rho=400k, V=2.0) ===")
    res_brain = run_one(400_000, 2.0, 12.0, cfg, seed=999, label="mouse cortex-like")
    fig_md = plot_decays_multidelta(
        [(res_brain, "kio=12, rho=400k, V=2.0")],
        title="Multi-delta MADI decay (mouse cortex estimate)",
        save_path="madi_multidelta.png",
    )
    plt.close(fig_md)
    print("  -> madi_multidelta.png")

    # --- Three-panel figure ---
    print("\n=== Generating figure ===")
    fig = plot_parameter_sensitivity(
        {"kio": pa, "rho": pb, "V": pc},
        save_path=args.output,
    )
    plt.close(fig)
    print(f"  -> {args.output}")

    print(f"\nTotal: {time.time()-t_total:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()

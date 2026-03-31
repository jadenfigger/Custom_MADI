"""
Plotting utilities for multi-Δ MADI results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from .config import D0_UM2_MS


def plot_decays_multidelta(
    results_list: List[Tuple[dict, str]],
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot S(b)/S₀ decay curves, one subplot per Δ.

    Parameters
    ----------
    results_list : list of (signals_dict, label)
    """
    if not results_list:
        return

    Deltas = results_list[0][0]['Deltas']
    n_deltas = len(Deltas)
    fig, axes = plt.subplots(1, n_deltas, figsize=(4 * n_deltas, 5), sharey=True)
    if n_deltas == 1:
        axes = [axes]

    for di, Delta in enumerate(Deltas):
        ax = axes[di]
        for res, lab in results_list:
            bv = res['b_values'] / 1000.0   # s/mm² → display as ×10³
            sig = res['signals'][di]
            ax.semilogy(bv, np.clip(sig, 1e-4, None), "o-", ms=4, label=lab)

        # Pure water reference
        bv_ref = results_list[0][0]['b_values']
        tD = Delta - results_list[0][0].get('delta', 6.0) / 3.0  # approximate
        S_free = np.exp(-bv_ref / 1e6 * D0_UM2_MS)
        ax.semilogy(bv_ref / 1000, S_free, "k--", lw=1, alpha=0.4, label="H₂O free")

        ax.set_xlabel("b  [×10³ s/mm²]")
        ax.set_title(f"Δ = {Delta:.0f} ms")
        if di == 0:
            ax.set_ylabel("S(b) / S₀")
        ax.legend(fontsize=6, loc="lower left")
        ax.set_ylim(bottom=0.01)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_parameter_sensitivity(
    panels: dict,
    save_path: Optional[str] = None,
):
    """Three-panel figure varying kio, rho, V (Figure 4 analog).

    Parameters
    ----------
    panels : dict with keys 'kio', 'rho', 'V', each containing
             list of (signals_dict, label)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = {
        'kio': "(a) Varying k_io",
        'rho': "(b) Varying ρ",
        'V':   "(c) Varying V",
    }

    for pi, key in enumerate(['kio', 'rho', 'V']):
        ax = axes[pi]
        if key not in panels:
            continue
        for res, lab in panels[key]:
            # Plot the Δ=25ms curve (mid-range) for the overview
            di = 1 if len(res['Deltas']) > 1 else 0
            bv = res['b_values'] / 1000.0
            sig = res['signals'][di]
            ax.semilogy(bv, np.clip(sig, 1e-4, None), "o-", ms=3, label=lab)

        bv_ref = panels[key][0][0]['b_values']
        S_free = np.exp(-bv_ref / 1e6 * D0_UM2_MS)
        ax.semilogy(bv_ref / 1000, S_free, "k--", lw=1, alpha=0.4, label="H₂O free")

        ax.set_xlabel("b  [×10³ s/mm²]")
        ax.set_title(titles[key])
        if pi == 0:
            ax.set_ylabel("S(b) / S₀  (Δ = 25 ms)")
        ax.legend(fontsize=6)
        ax.set_ylim(bottom=0.01)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ensemble_slice(ens, z_level=None, n_grid=400, ax=None):
    """2-D cross-section of an ensemble."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    L = ens.L
    if z_level is None:
        z_level = L / 2.0

    xs = np.linspace(0, L, n_grid)
    ys = np.linspace(0, L, n_grid)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z_level)
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    _, inside = ens.classify_cpu(pts)
    img = inside.reshape(n_grid, n_grid).astype(float)

    ax.imshow(img, origin="lower", extent=[0, L, 0, L],
              cmap="coolwarm", vmin=0, vmax=1, alpha=0.7)
    dz = L / 20
    mask = np.abs(ens.seeds[:, 2] - z_level) < dz
    ax.scatter(ens.seeds[mask, 0], ens.seeds[mask, 1], c="k", s=3, alpha=0.6)
    ax.set_xlabel("x [μm]");  ax.set_ylabel("y [μm]")
    ax.set_title(f"Ensemble slice z={z_level:.0f} μm (blue=intra)")
    return ax

#!/usr/bin/env python3
"""
analyze_library.py — Explore structure in a MADI simulation library
====================================================================

Answers:
  1. How many principal components explain the signal variance?
  2. How smoothly do signals vary along each parameter axis?
  3. Can we interpolate accurately between library entries?
  4. What surrogate model works best?

Usage:
    python analyze_library.py --library madi_library.npz
    python analyze_library.py --library madi_library.npz --output analysis/
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(__file__))
from madi.library import load_library, load_library_meta, library_summary


# ===================================================================
# 1. PCA — How compressible is the signal space?
# ===================================================================

def analyze_pca(vectors, out_dir):
    """SVD of the signal matrix. Shows if signals live in a low-D subspace."""
    # Center
    mean = vectors.mean(axis=0)
    centered = vectors - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, len(S)+1), explained * 100, color="steelblue")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("PCA spectrum")

    axes[1].plot(range(1, len(S)+1), cumulative * 100, "o-", color="darkorange")
    axes[1].axhline(99, ls="--", color="gray", alpha=0.5)
    axes[1].axhline(99.9, ls=":", color="gray", alpha=0.5)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative explained variance")
    axes[1].text(len(S)*0.6, 99.3, "99%", color="gray")
    axes[1].text(len(S)*0.6, 100.2, "99.9%", color="gray")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_spectrum.png"), dpi=150)
    plt.close(fig)

    n99 = np.searchsorted(cumulative, 0.99) + 1
    n999 = np.searchsorted(cumulative, 0.999) + 1
    print(f"\n  PCA Results:")
    print(f"    Signal vector dimension: {vectors.shape[1]}")
    print(f"    Components for 99% variance:  {n99}")
    print(f"    Components for 99.9% variance: {n999}")
    print(f"    Top singular values: {S[:6].round(4)}")
    print(f"    → pca_spectrum.png")

    return mean, S, Vt, n99


# ===================================================================
# 2. Smoothness — How do signals vary along each parameter axis?
# ===================================================================

def analyze_smoothness(kios, rhos, Vs, vectors, out_dir):
    """Plot signal components along each parameter axis while fixing others."""

    unique_kios = sorted(set(kios))
    unique_rhos = sorted(set(rhos))
    unique_Vs = sorted(set(Vs))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Vary kio, fix rho and V at median values ---
    rho_mid = unique_rhos[len(unique_rhos)//2]
    V_mid = unique_Vs[len(unique_Vs)//2]

    mask_rv = (np.abs(rhos - rho_mid) < 1) & (np.abs(Vs - V_mid) < 0.001)
    if mask_rv.sum() > 1:
        idx = np.where(mask_rv)[0]
        order = np.argsort(kios[idx])
        x = kios[idx][order]
        vecs = vectors[idx][order]
        for j in range(vecs.shape[1]):
            axes[0].plot(x, vecs[:, j], "o-", ms=3, alpha=0.6)
        axes[0].set_xlabel("kio (s⁻¹)")
        axes[0].set_title(f"Signal vs kio\n(ρ={rho_mid/1e3:.0f}k, V={V_mid:.1f})")

    # --- Vary rho, fix kio and V ---
    kio_mid = unique_kios[len(unique_kios)//2]

    mask_kv = (np.abs(kios - kio_mid) < 0.01) & (np.abs(Vs - V_mid) < 0.001)
    if mask_kv.sum() > 1:
        idx = np.where(mask_kv)[0]
        order = np.argsort(rhos[idx])
        x = rhos[idx][order]
        vecs = vectors[idx][order]
        for j in range(vecs.shape[1]):
            axes[1].plot(x / 1e3, vecs[:, j], "o-", ms=3, alpha=0.6)
        axes[1].set_xlabel("ρ (×10³ cells/μL)")
        axes[1].set_title(f"Signal vs ρ\n(kio={kio_mid:.0f}, V={V_mid:.1f})")

    # --- Vary V, fix kio and rho ---
    mask_kr = (np.abs(kios - kio_mid) < 0.01) & (np.abs(rhos - rho_mid) < 1)
    if mask_kr.sum() > 1:
        idx = np.where(mask_kr)[0]
        order = np.argsort(Vs[idx])
        x = Vs[idx][order]
        vecs = vectors[idx][order]
        for j in range(vecs.shape[1]):
            axes[2].plot(x, vecs[:, j], "o-", ms=3, alpha=0.6)
        axes[2].set_xlabel("V (pL)")
        axes[2].set_title(f"Signal vs V\n(kio={kio_mid:.0f}, ρ={rho_mid/1e3:.0f}k)")

    for ax in axes:
        ax.set_ylabel("S(b)/S₀")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Smoothness: signal components along each parameter axis", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smoothness.png"), dpi=150)
    plt.close(fig)
    print(f"    → smoothness.png")


# ===================================================================
# 3. Leave-one-out interpolation test
# ===================================================================

def analyze_interpolation(kios, rhos, Vs, vectors, out_dir):
    """Hold out each entry, predict from neighbors, measure error."""
    from scipy.interpolate import RBFInterpolator

    n = len(kios)
    if n < 20:
        print("  Skipping interpolation test (need ≥20 entries)")
        return None

    # Normalise parameter space to [0, 1]
    params = np.column_stack([kios, rhos / 1e6, Vs])  # scale rho down
    p_min = params.min(axis=0)
    p_max = params.max(axis=0)
    p_range = p_max - p_min
    p_range[p_range < 1e-10] = 1.0
    params_norm = (params - p_min) / p_range

    # Random subset for LOO (full LOO is slow for large libraries)
    n_test = min(n, 200)
    rng = np.random.default_rng(42)
    test_idx = rng.choice(n, n_test, replace=False)

    errors_nn = []    # nearest-neighbor (current method)
    errors_linear = []  # linear interpolation via RBF
    errors_rbf = []   # RBF with smoothing

    for i, ti in enumerate(test_idx):
        train_mask = np.ones(n, dtype=bool)
        train_mask[ti] = False
        train_params = params_norm[train_mask]
        train_vecs = vectors[train_mask]
        test_param = params_norm[ti:ti+1]
        true_vec = vectors[ti]

        # Nearest neighbor
        dists = np.sum((train_params - test_param) ** 2, axis=1)
        nn_idx = np.argmin(dists)
        nn_pred = train_vecs[nn_idx]
        errors_nn.append(np.sqrt(np.mean((true_vec - nn_pred) ** 2)))

        # Linear interpolation (RBF with linear kernel)
        try:
            rbf_lin = RBFInterpolator(train_params, train_vecs,
                                       kernel="linear", smoothing=0.0)
            lin_pred = rbf_lin(test_param)[0]
            errors_linear.append(np.sqrt(np.mean((true_vec - lin_pred) ** 2)))
        except Exception:
            errors_linear.append(np.nan)

        # Thin-plate spline RBF
        try:
            rbf_tps = RBFInterpolator(train_params, train_vecs,
                                       kernel="thin_plate_spline", smoothing=1e-4)
            tps_pred = rbf_tps(test_param)[0]
            errors_rbf.append(np.sqrt(np.mean((true_vec - tps_pred) ** 2)))
        except Exception:
            errors_rbf.append(np.nan)

    errors_nn = np.array(errors_nn)
    errors_linear = np.array(errors_linear)
    errors_rbf = np.array(errors_rbf)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Nearest\nneighbor", "Linear\nRBF", "Thin-plate\nspline RBF"]
    data = [errors_nn, errors_linear[~np.isnan(errors_linear)],
            errors_rbf[~np.isnan(errors_rbf)]]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#d9534f", "#5bc0de", "#5cb85c"]
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("RMSE (signal units)")
    ax.set_title(f"Leave-one-out prediction error ({n_test} test points)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interpolation_test.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Interpolation (LOO, n={n_test}):")
    print(f"    Nearest-neighbor RMSE: {np.nanmedian(errors_nn):.5f} "
          f"(median), {np.nanmean(errors_nn):.5f} (mean)")
    print(f"    Linear RBF RMSE:      {np.nanmedian(errors_linear):.5f} "
          f"(median), {np.nanmean(errors_linear):.5f} (mean)")
    print(f"    TPS RBF RMSE:         {np.nanmedian(errors_rbf):.5f} "
          f"(median), {np.nanmean(errors_rbf):.5f} (mean)")
    improvement = np.nanmedian(errors_nn) / np.nanmedian(errors_rbf)
    print(f"    TPS vs NN improvement: {improvement:.1f}×")
    print(f"    → interpolation_test.png")

    return {
        "nn_median": np.nanmedian(errors_nn),
        "rbf_median": np.nanmedian(errors_rbf),
        "improvement": improvement,
    }


# ===================================================================
# 4. Parameter correlation in signal space
# ===================================================================

def analyze_signal_landscape(kios, rhos, Vs, vectors, out_dir):
    """2D slices through the signal landscape using first 2 PCA components."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    scores = U[:, :2] * S[:2]  # project onto first 2 PCs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (c, label, cmap) in zip(axes, [
        (kios, "kio (s⁻¹)", "viridis"),
        (rhos / 1e3, "ρ (×10³ cells/μL)", "plasma"),
        (Vs, "V (pL)", "coolwarm"),
    ]):
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=c, cmap=cmap,
                        s=15, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Colored by {label}")

    fig.suptitle("Library entries in PCA signal space", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_landscape.png"), dpi=150)
    plt.close(fig)
    print(f"    → pca_landscape.png")


# ===================================================================
# 5. Surrogate model test: fit full RBF on all data, test on grid
# ===================================================================

def analyze_surrogate_potential(kios, rhos, Vs, vectors, out_dir):
    """Fit a thin-plate-spline RBF to the entire library and report."""
    from scipy.interpolate import RBFInterpolator

    n = len(kios)
    if n < 30:
        print("  Skipping surrogate analysis (need ≥30 entries)")
        return

    params = np.column_stack([kios, rhos / 1e6, Vs])
    p_min = params.min(axis=0)
    p_max = params.max(axis=0)
    p_range = p_max - p_min
    p_range[p_range < 1e-10] = 1.0
    params_norm = (params - p_min) / p_range

    print(f"\n  Fitting TPS-RBF surrogate on {n} entries ...")
    t0 = __import__("time").time()
    rbf = RBFInterpolator(params_norm, vectors,
                           kernel="thin_plate_spline", smoothing=1e-5)
    t_fit = __import__("time").time() - t0
    print(f"    Fit time: {t_fit:.2f}s")

    # Self-prediction (should be near-zero with low smoothing)
    pred = rbf(params_norm)
    self_err = np.sqrt(np.mean((vectors - pred) ** 2, axis=1))
    print(f"    Self-prediction RMSE: {self_err.mean():.6f} (mean), "
          f"{self_err.max():.6f} (max)")

    # Predict on a fine grid between min/max of each param
    n_fine = 20
    k_fine = np.linspace(kios.min(), kios.max(), n_fine)
    r_fine = np.linspace(rhos.min(), rhos.max(), n_fine)
    v_fine = np.linspace(Vs.min(), Vs.max(), n_fine)

    # Just a 1D sweep for visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rho_mid = np.median(rhos)
    V_mid = np.median(Vs)
    kio_mid = np.median(kios)

    # Sweep kio
    sweep_k = np.column_stack([
        k_fine,
        np.full(n_fine, rho_mid / 1e6),
        np.full(n_fine, V_mid)
    ])
    sweep_k_norm = (sweep_k - p_min / np.array([1, 1e6, 1])) / (p_range / np.array([1, 1e6, 1]))
    # Need to normalise consistently
    sweep_k_params = np.column_stack([k_fine, np.full(n_fine, rho_mid/1e6), np.full(n_fine, V_mid)])
    sweep_k_norm = (sweep_k_params - p_min) / p_range

    pred_k = rbf(sweep_k_norm)
    for j in range(pred_k.shape[1]):
        axes[0].plot(k_fine, pred_k[:, j], "-", alpha=0.5, lw=1.5)
    # Overlay actual library points at this rho/V
    mask = (np.abs(rhos - rho_mid) < 1) & (np.abs(Vs - V_mid) < 0.001)
    if mask.sum() > 0:
        idx = np.where(mask)[0]
        for j in range(vectors.shape[1]):
            axes[0].plot(kios[idx], vectors[idx, j], "ko", ms=4, alpha=0.5)
    axes[0].set_xlabel("kio (s⁻¹)")
    axes[0].set_title(f"RBF interpolation vs kio\n(ρ={rho_mid/1e3:.0f}k, V={V_mid:.1f})")

    # Sweep rho
    sweep_r_params = np.column_stack([np.full(n_fine, kio_mid), r_fine/1e6, np.full(n_fine, V_mid)])
    sweep_r_norm = (sweep_r_params - p_min) / p_range
    pred_r = rbf(sweep_r_norm)
    for j in range(pred_r.shape[1]):
        axes[1].plot(r_fine/1e3, pred_r[:, j], "-", alpha=0.5, lw=1.5)
    mask = (np.abs(kios - kio_mid) < 0.01) & (np.abs(Vs - V_mid) < 0.001)
    if mask.sum() > 0:
        idx = np.where(mask)[0]
        for j in range(vectors.shape[1]):
            axes[1].plot(rhos[idx]/1e3, vectors[idx, j], "ko", ms=4, alpha=0.5)
    axes[1].set_xlabel("ρ (×10³ cells/μL)")
    axes[1].set_title(f"RBF interpolation vs ρ\n(kio={kio_mid:.0f}, V={V_mid:.1f})")

    # Sweep V
    sweep_v_params = np.column_stack([np.full(n_fine, kio_mid), np.full(n_fine, rho_mid/1e6), v_fine])
    sweep_v_norm = (sweep_v_params - p_min) / p_range
    pred_v = rbf(sweep_v_norm)
    for j in range(pred_v.shape[1]):
        axes[2].plot(v_fine, pred_v[:, j], "-", alpha=0.5, lw=1.5)
    mask = (np.abs(kios - kio_mid) < 0.01) & (np.abs(rhos - rho_mid) < 1)
    if mask.sum() > 0:
        idx = np.where(mask)[0]
        for j in range(vectors.shape[1]):
            axes[2].plot(Vs[idx], vectors[idx, j], "ko", ms=4, alpha=0.5)
    axes[2].set_xlabel("V (pL)")
    axes[2].set_title(f"RBF interpolation vs V\n(kio={kio_mid:.0f}, ρ={rho_mid/1e3:.0f}k)")

    for ax in axes:
        ax.set_ylabel("S(b)/S₀")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Continuous surrogate (TPS-RBF): lines = interpolated, dots = simulated",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "surrogate_sweeps.png"), dpi=150)
    plt.close(fig)
    print(f"    → surrogate_sweeps.png")

    # Save the surrogate for potential later use
    surrogate_path = os.path.join(out_dir, "surrogate_rbf.npz")
    np.savez(surrogate_path,
             p_min=p_min, p_range=p_range,
             train_params_norm=params_norm,
             train_vectors=vectors,
             smoothing=1e-5,
             kernel="thin_plate_spline")
    print(f"    → surrogate_rbf.npz (training data for reconstruction)")


# ===================================================================
# 6. Correlation matrix between signal components
# ===================================================================

def analyze_correlations(vectors, out_dir, meta=None):
    """Correlation between the 16 signal components."""
    corr = np.corrcoef(vectors.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")

    n_b = meta.get('n_b', 4) if meta else 4
    deltas = meta.get('deltas', [15, 25, 30, 40]) if meta else [15, 25, 30, 40]
    labels = []
    for d in deltas:
        for bi, bv in enumerate([1000, 2500, 4000, 6000][:n_b]):
            labels.append(f"Δ{d:.0f}\nb{bv/1e3:.0f}k")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Signal component correlation matrix")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"    → correlation_matrix.png")


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description="Analyze MADI library structure")
    ap.add_argument("--library", default="madi_library.npz")
    ap.add_argument("--output", default="madi_analysis")
    args = ap.parse_args()

    if not os.path.exists(args.library):
        print(f"Library not found: {args.library}")
        return

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("MADI Library Structure Analysis")
    print("=" * 60)

    lib = load_library(args.library)
    meta = load_library_meta(args.library)
    print(f"\nLibrary: {args.library}")
    library_summary(lib)

    kios = np.array([e.kio for e in lib])
    rhos = np.array([e.rho for e in lib])
    Vs   = np.array([e.V for e in lib])
    vectors = np.array([e.vector for e in lib])

    print(f"\nSignal matrix shape: {vectors.shape}")
    print(f"  (rows = parameter sets, columns = signal measurements)")

    # # 1. PCA
    # print("\n" + "─" * 40)
    # print("1. PRINCIPAL COMPONENT ANALYSIS")
    # print("─" * 40)
    # mean, S, Vt, n99 = analyze_pca(vectors, args.output)

    # # 2. Smoothness
    # print("\n" + "─" * 40)
    # print("2. SMOOTHNESS ALONG PARAMETER AXES")
    # print("─" * 40)
    # analyze_smoothness(kios, rhos, Vs, vectors, args.output)

    # # 3. Correlation
    # print("\n" + "─" * 40)
    # print("3. SIGNAL COMPONENT CORRELATIONS")
    # print("─" * 40)
    # analyze_correlations(vectors, args.output, meta)

    # 4. PCA landscape
    print("\n" + "─" * 40)
    print("4. PCA SIGNAL LANDSCAPE")
    print("─" * 40)
    analyze_signal_landscape(kios, rhos, Vs, vectors, args.output)

    # # 5. Interpolation test
    # print("\n" + "─" * 40)
    # print("5. LEAVE-ONE-OUT INTERPOLATION TEST")
    # print("─" * 40)
    # interp_results = analyze_interpolation(kios, rhos, Vs, vectors, args.output)

    # # 6. Surrogate model
    # print("\n" + "─" * 40)
    # print("6. CONTINUOUS SURROGATE MODEL")
    # print("─" * 40)
    # analyze_surrogate_potential(kios, rhos, Vs, vectors, args.output)

    # # Summary
    # print("\n" + "=" * 60)
    # print("SUMMARY")
    # print("=" * 60)
    # print(f"  PCA: {n99} components explain 99% of variance (out of {vectors.shape[1]})")
    # if n99 <= 4:
    #     print(f"  → STRONG structure: signals live in a {n99}D subspace")
    #     print(f"    A continuous surrogate should work very well.")
    # elif n99 <= 8:
    #     print(f"  → MODERATE structure: {n99}D effective dimensionality")
    #     print(f"    Surrogate modeling feasible with enough training points.")
    # else:
    #     print(f"  → WEAK structure: high effective dimensionality")

    # if interp_results:
    #     imp = interp_results["improvement"]
    #     if imp > 3:
    #         print(f"  Interpolation: RBF is {imp:.1f}× better than nearest-neighbor")
    #         print(f"    → Continuous surrogate would significantly improve fitting")
    #     elif imp > 1.5:
    #         print(f"  Interpolation: RBF is {imp:.1f}× better than nearest-neighbor")
    #         print(f"    → Surrogate would help, especially in sparse regions")
    #     else:
    #         print(f"  Interpolation: only {imp:.1f}× improvement over NN")
    #         print(f"    → Library is already dense enough, or signals are very flat")

    # print(f"\n  All plots saved to {args.output}/")
    # print(f"\n  Next steps if structure is good:")
    # print(f"    1. Replace NN matching with RBF interpolation for continuous fits")
    # print(f"    2. Or train a small neural network: (kio,ρ,V) → signal vector")
    # print(f"    3. Then use scipy.optimize.minimize to find best (kio,ρ,V) per voxel")


if __name__ == "__main__":
    main()

"""
MADI library: normalized signal-decay vs b-value.

Creates a figure with three subplots. In each subplot one parameter
(kio, rho, or V) is left FREE and swept over a range of values, while
the other two are held FIXED at a common central anchor point that
actually exists in the library grid. Curves are colored by the value
of the free parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

# ----------------------------------------------------------------------
# Load library
# ----------------------------------------------------------------------
NPZ_PATH = "/mnt/c/miscellaneous/coding_projects/python/mri_processing/processing/madi_gpu/custom_madi/data/libraries/madi_dense_human.npz"
d = np.load(NPZ_PATH, allow_pickle=True)

kios = d["kios"]          # (N,)  exchange rate  k_io
rhos = d["rhos"]          # (N,)  density        rho
Vs = d["Vs"]              # (N,)  volume         V
vectors = d["vectors"]    # (N, n_b)  normalized signal at each b-value
b_values = d["b_values"]  # (n_b,)  b-values [s/mm^2]
log_yaxis = False

print("Range of kios", np.unique(kios))
print("Range of rhos", np.unique(rhos))
print("Range of Vs", np.unique(Vs))

# ----------------------------------------------------------------------
# Pick a central anchor point that EXISTS in the (sparse) grid.
# For each parameter, take the unique value closest (in log space) to
# the median of its unique values, then verify the combination exists.
# ----------------------------------------------------------------------
def closest_to_median(vals):
    u = np.unique(vals)
    target = np.median(u)
    return u[np.argmin(np.abs(np.log(u) - np.log(target)))]

# anchor_kio = closest_to_median(kios)
# anchor_rho = closest_to_median(rhos)
# anchor_V = closest_to_median(Vs)

anchor_kio = 35.0
anchor_rho = 1.2e6
anchor_V = 0.3

print(f"Anchor (fixed) point: kio={anchor_kio}, rho={anchor_rho:g}, V={anchor_V}")

# ----------------------------------------------------------------------
# Helper: rows where the two "fixed" params equal the anchor,
# sorted by the free param.
# ----------------------------------------------------------------------
def sweep(free):
    if free == "kio":
        mask = (rhos == anchor_rho) & (Vs == anchor_V)
        fvals = kios[mask]
    elif free == "rho":
        mask = (kios == anchor_kio) & (Vs == anchor_V)
        fvals = rhos[mask]
    elif free == "V":
        mask = (kios == anchor_kio) & (rhos == anchor_rho)
        fvals = Vs[mask]
    else:
        raise ValueError(free)
    order = np.argsort(fvals)
    return fvals[order], vectors[mask][order]

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
panels = [
    ("kio", r"$k_{io}$",  r"$k_{io}$ free   ($\rho=%g,\ V=%g$ fixed)" % (anchor_rho, anchor_V), "viridis"),
    ("rho", r"$\rho$",     r"$\rho$ free   ($k_{io}=%g,\ V=%g$ fixed)" % (anchor_kio, anchor_V), "viridis"),
    ("V",   r"$V$",        r"$V$ free   ($k_{io}=%g,\ \rho=%g$ fixed)" % (anchor_kio, anchor_rho), "viridis"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

for ax, (free, sym, title, cmap_name) in zip(axes, panels):
    fvals, curves = sweep(free)
    cmap = plt.get_cmap(cmap_name)
    norm = LogNorm(vmin=fvals.min(), vmax=fvals.max())

    # print(free, sym, title, cmap_name)
    for fv, curve in zip(fvals, curves):
        # print(fv, curve)
        ax.plot(b_values, curve, "-o", ms=4, lw=1.5,
                color=cmap(norm(fv)))

    if log_yaxis:
        ax.set_yscale("log")
    ax.set_xlabel(r"$b$-value  [s/mm$^2$]")
    ax.set_ylabel("Normalized signal  $S/S_0$")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"{sym}  ({len(fvals)} values)")

fig.suptitle("MADI normalized decay: one parameter free, two fixed",
             fontsize=14, fontweight="bold")

OUT = f"/mnt/c/miscellaneous/coding_projects/python/mri_processing/processing/madi_gpu/custom_madi/figures/madi_decay_subplots_rho-{anchor_rho}_V-{anchor_V}_kio-{anchor_kio}.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved:", OUT)
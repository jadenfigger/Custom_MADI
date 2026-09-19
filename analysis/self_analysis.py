import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Load library
# ------------------------------------------------------------

lib_path = Path("data/libraries/madi_dense_universal_remediated.npz")
lib = np.load(lib_path, allow_pickle=False)

vectors = lib["vectors"]
pair_deltas = lib["pair_deltas"]
pair_Deltas = lib["pair_Deltas"]
b_values = lib["b_values"]

# Inspect these names once to confirm parameter-array names in your artifact
print(lib.files)


# ------------------------------------------------------------
# Choose one (delta, Delta) timing
# ------------------------------------------------------------

delta_target = 20
Delta_target = 50

pair_idx = np.where(
    (pair_deltas == delta_target) &
    (pair_Deltas == Delta_target)
)[0]

if len(pair_idx) != 1:
    raise ValueError(f"Expected one timing pair, found {len(pair_idx)}")

pair_idx = pair_idx[0]


# ------------------------------------------------------------
# Extract this timing's b-value decay curves
#
# Library layout:
#   timing pair 0: all b-values
#   timing pair 1: all b-values
#   ...
# ------------------------------------------------------------

n_b = len(b_values)

start = pair_idx * n_b
stop = start + n_b

# shape: (n_library_entries, n_b)
signals = vectors[:, start:stop]

print("signals shape:", signals.shape)
print("b-values:", b_values)


# ------------------------------------------------------------
# Select entries with very low high-b signal
# ------------------------------------------------------------

low_b = b_values < 4000
high_b = b_values > 4000
threshold = 1e-4

# Require the signal to fall below 1e-4 at AT LEAST ONE
# b-value above 4000.
selected = np.any(signals[:10, low_b] < threshold, axis=1)

selected_indices = np.flatnonzero(selected)

print(f"{selected.sum()} / {len(selected)} entries selected")


# ------------------------------------------------------------
# Don't plot thousands of curves.
# Select a manageable subset.
# ------------------------------------------------------------

n_plot = 10

# evenly sample from all entries satisfying the condition
if len(selected_indices) > n_plot:
    positions = np.linspace(
        0,
        len(selected_indices) - 1,
        n_plot,
        dtype=int
    )
    plot_indices = selected_indices[positions]
else:
    plot_indices = selected_indices


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
for i in plot_indices:
    ax.plot(
        b_values,
        signals[i],
        marker="o",
        markersize=3,
        linewidth=1,
        alpha=0.7,
    )

ax.axhline(
    threshold,
    linestyle="--",
    linewidth=1,
    label=r"$S/S_0 = 10^{-4}$"
)

ax.set_yscale("log")

ax.set_xlabel(r"$b$ [s/mm$^2$]")
ax.set_ylabel(r"$S/S_0$")
ax.set_title(
    rf"Library signal decay: "
    rf"$\delta={delta_target}$ ms, $\Delta={Delta_target}$ ms"
)

ax.legend()
fig.tight_layout()

plt.show()
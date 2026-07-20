#!/usr/bin/env python3
"""
plot_decay_curves.py — Plot signal decay curves for one free parameter
=======================================================================

Fix two of (kio, rho, V) and sweep the third across a set of values,
plotting S/S0 vs b-value (one panel per Δ) for each value of the free
parameter. Values are matched to the nearest available library entries
(the library is a discrete grid) — the matched combinations are printed
before plotting.

Usage:
    python plot_decay_curves.py --library madi_library.npz \
        --free kio --range 2,5,8,12,18,25,35,50,75,100 \
        --fixed rho=400000,V=2.0

    python plot_decay_curves.py --library madi_library.npz \
        --free rho --range 100000:800000:100000 \
        --fixed kio=25,V=2.0 \
        --output decay_vs_rho.png

    python plot_decay_curves.py --library madi_library.npz \
        --free kio --fixed rho=400000,V=2.0 --log


    python analysis/plot_decay_curves.py --library data/libraries/madi_dense.npz --free V --fixed rho=100000,kio=2.0 --log
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from madi.library import load_library, load_library_meta

PARAM_NAMES = ("kio", "rho", "V")


def _parse_values(spec: str) -> list[float]:
    """Parse 'v1,v2,v3' or 'start:stop:step' into a list of floats."""
    spec = spec.strip()
    if ":" in spec:
        parts = [float(p) for p in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Range spec must be start:stop:step, got '{spec}'")
        start, stop, step = parts
        n = int(round((stop - start) / step)) + 1
        return list(np.linspace(start, stop, n))
    return [float(v) for v in spec.split(",")]


def _parse_fixed(spec: str) -> dict:
    """Parse 'rho=400000,V=2.0' into {'rho': 400000.0, 'V': 2.0}."""
    fixed = {}
    for item in spec.split(","):
        name, value = item.split("=")
        name = name.strip()
        if name not in PARAM_NAMES:
            raise ValueError(f"Unknown parameter '{name}' (expected one of {PARAM_NAMES})")
        fixed[name] = float(value)
    return fixed


def _snap(value: float, grid: list[float]) -> float:
    """Return the grid value closest to `value`."""
    return min(grid, key=lambda g: abs(g - value))


def main():
    ap = argparse.ArgumentParser(description="Plot decay curves for one free MADI parameter")
    ap.add_argument("--library", default="madi_library.npz")
    ap.add_argument("--free", required=True, choices=PARAM_NAMES,
                     help="Parameter to sweep")
    ap.add_argument("--range", default=None,
                     help="Comma list 'v1,v2,...' or slice 'start:stop:step' for the free parameter. "
                          "If omitted, sweeps every unique value of --free present in the library.")
    ap.add_argument("--fixed", required=True,
                     help="Comma list of 'name=value' for the two fixed parameters, e.g. 'rho=400000,V=2.0'")
    ap.add_argument("--output", default="decay_curves.png")
    ap.add_argument("--log", action="store_true",
                     help="Plot signal (S/S0) on a log y-axis.")
    args = ap.parse_args()

    if not os.path.exists(args.library):
        print(f"Library not found: {args.library}")
        return

    fixed = _parse_fixed(args.fixed)
    missing = set(PARAM_NAMES) - {args.free} - set(fixed)
    if missing:
        raise ValueError(f"Must fix {set(PARAM_NAMES) - {args.free}}, missing: {missing}")

    lib = load_library(args.library)
    meta = load_library_meta(args.library)
    deltas = meta["deltas"]
    n_b = meta["n_b"]
    b_values = meta["b_values"]

    # Snap each fixed parameter to the nearest value actually on the grid,
    # then keep only entries that match those fixed values exactly. This
    # guarantees a "fixed" parameter never silently drifts to another value
    # just because the requested (kio, rho, V) combination doesn't exist.
    grids = {name: sorted(set(getattr(e, name) for e in lib)) for name in PARAM_NAMES}
    snapped = {}
    for name, requested in fixed.items():
        snapped[name] = _snap(requested, grids[name])
        if abs(snapped[name] - requested) > 1e-9:
            print(f"  note: fixed {name}={requested:g} snapped to nearest grid "
                  f"value {snapped[name]:g}")

    subset = [e for e in lib
              if all(abs(getattr(e, name) - val) < 1e-6 * max(1.0, abs(val))
                     for name, val in snapped.items())]
    if not subset:
        raise ValueError(f"No library entries match fixed values {snapped}.")

    # Available free values within the fixed subset.
    avail = sorted(set(getattr(e, args.free) for e in subset))
    if args.range is None:
        free_values = avail
    else:
        # Snap each requested free value to the nearest one that exists for
        # this fixed combination, dropping duplicates while preserving order.
        free_values, seen = [], set()
        for v in _parse_values(args.range):
            s = _snap(v, avail)
            if s not in seen:
                free_values.append(s); seen.add(s)

    # One entry per free value (fixed params are guaranteed identical now).
    by_free = {getattr(e, args.free): e for e in subset}
    matched_entries = [by_free[v] for v in free_values]

    fixed_str = ", ".join(f"{k}={v:g}" for k, v in snapped.items())
    print(f"\nSweeping '{args.free}' over {len(free_values)} value(s), "
          f"fixed = {fixed_str}\n")
    print(f"{'kio':>10} {'rho':>12} {'V':>8}")
    for entry in matched_entries:
        print(f"{entry.kio:10.3f} {entry.rho:12.1f} {entry.V:8.3f}")

    n_deltas = len(deltas)
    fig, axes = plt.subplots(1, n_deltas, figsize=(5 * n_deltas, 4.5), squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.9, max(len(free_values), 1)))

    # Prepend the b=0 anchor (S/S0 = 1 by definition) to every curve.
    b_base = np.asarray(b_values, dtype=float) if b_values is not None else np.arange(1, n_b + 1)
    x = np.concatenate([[0.0], b_base])

    for di, Delta in enumerate(deltas):
        ax = axes[di]
        col_lo, col_hi = di * n_b, (di + 1) * n_b
        for v, entry, c in zip(free_values, matched_entries, colors):
            curve = np.concatenate([[1.0], entry.vector[col_lo:col_hi]])
            ax.plot(x, curve, "o-", color=c, label=f"{args.free}={v:g}")
        ax.set_xlabel("b (s/mm²)")
        ax.set_ylabel("S/S0")
        ax.set_title(f"Δ = {Delta:g} ms")
        if args.log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, which="both")

    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Decay curves vs {args.free} (fixed: {fixed_str})"
                 + ("  [log S/S0]" if args.log else ""))
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()

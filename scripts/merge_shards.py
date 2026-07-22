#!/usr/bin/env python
"""
Merge per-shard MADI libraries into a single .npz.

Usage
-----
    python merge_shards.py libraries/madi_dense.shard*.npz \
                           -o libraries/madi_dense_full.npz

    python scripts/merge_shards.py libraries/madi_dense.shard*.npz -o madi_universal_dense.npz

Duplicates (same kio/rho/V triplet) are deduped; first occurrence wins.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# Make the repo importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as _np
from madi.config import SimConfig
from madi.library import (load_library, load_library_meta, _save_library,
                          library_summary)
from madi.signal import ColumnGrid


def _columns_from_meta(meta):
    """Reconstruct the (δ,Δ,b) grid metadata from a loaded shard so the
    merged library carries the SHARDS' actual grid, not a default one.
    Only the fields _save_library reads (delta_pairs, b_values, n_b) need
    to be real; the per-column lookup/coeff arrays are unused at save time."""
    pairs = list(meta["delta_pairs"])
    bvals = _np.asarray(meta["b_values"], dtype=float)
    empty = _np.array([], dtype=_np.int32)
    return ColumnGrid(delta_pairs=pairs, b_values=bvals,
                      j_delta=empty, j_Delta=empty, j_sum=empty,
                      phase_coef=_np.array([], dtype=_np.float64),
                      n_pairs=len(pairs), n_b=int(meta["n_b"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+",
                    help="Per-shard .npz files (globs allowed)")
    ap.add_argument("-o", "--out", required=True,
                    help="Output merged library .npz")
    args = ap.parse_args()

    # Expand globs
    paths = []
    for pat in args.shards:
        matched = sorted(glob.glob(pat))
        if not matched:
            print(f"  WARNING: no files match '{pat}'")
        paths.extend(matched)

    if not paths:
        print("ERROR: no shard files found")
        sys.exit(1)

    print(f"Merging {len(paths)} shard files:")
    merged = []
    seen = set()
    for p in paths:
        lib = load_library(p)
        before = len(merged)
        for e in lib:
            key = (round(e.kio, 4), round(e.rho, 1), round(e.V, 6))
            if key in seen:
                continue
            seen.add(key)
            merged.append(e)
        print(f"  {p}: +{len(merged) - before} new  ({len(lib)} in file)")

    # Preserve the (δ,Δ,b) grid metadata from the first shard so the merged
    # library's stored pair/b-value axes match the concatenated vectors.
    meta = load_library_meta(paths[0])
    columns = _columns_from_meta(meta)
    cfg = SimConfig(h_ms=(meta.get("h_ms") or 1.0))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    _save_library(merged, args.out, cfg=cfg, columns=columns)

    print(f"\nMerged library → {args.out}")
    library_summary(merged, meta=load_library_meta(args.out))


if __name__ == "__main__":
    main()

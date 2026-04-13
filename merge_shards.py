#!/usr/bin/env python
"""
Merge per-shard MADI libraries into a single .npz.

Usage
-----
    python merge_shards.py libraries/madi_dense.shard*.npz \
                           -o libraries/madi_dense_full.npz

Duplicates (same kio/rho/V triplet) are deduped; first occurrence wins.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

# Make the repo importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from madi.library import load_library, _save_library, library_summary


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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    _save_library(merged, args.out)

    print(f"\nMerged library → {args.out}")
    library_summary(merged)


if __name__ == "__main__":
    main()

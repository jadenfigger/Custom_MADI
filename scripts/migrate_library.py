#!/usr/bin/env python3
"""
migrate_library.py — Add small_delta and b_values metadata to a library
========================================================================

Older libraries built before the metadata update don't carry small δ or
the explicit b-value list.  The fitter falls back to defaults with a
warning, but it is cleaner to migrate the file once.

Usage
-----
  python migrate_library.py madi_dense.npz
      → uses defaults: --small-delta 6.0 --b-values 1000 2500 4000 6000

  python migrate_library.py madi_dense.npz --small-delta 6.0 \\
      --b-values 1000 2500 4000 6000

  python migrate_library.py madi_dense.npz --output madi_dense.migrated.npz
      → write to a new file instead of overwriting

The Δ list (`deltas`) is kept as-is from the original file.  Only the
two new fields are added.
"""

import argparse
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("library", help="Path to .npz library to migrate")
    ap.add_argument("--small-delta", type=float, default=6.0,
                    help="δ (PFG duration) [ms].  Default: 6.0")
    ap.add_argument("--b-values", type=float, nargs="+",
                    default=[1000.0, 2500.0, 4000.0, 6000.0],
                    help="b-values [s/mm²], one per shell.  "
                         "Default: 1000 2500 4000 6000")
    ap.add_argument("--output", default=None,
                    help="Output path.  Default: overwrite input "
                         "(after writing a .bak copy).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite even if existing metadata is present.")
    args = ap.parse_args()

    if not os.path.exists(args.library):
        print(f"ERROR: {args.library} does not exist."); sys.exit(1)

    data = np.load(args.library)
    fields = list(data.files)
    print(f"Loaded {args.library}")
    print(f"  Existing fields: {fields}")
    print(f"  Entries: {len(data['kios'])}")
    print(f"  Δ values (deltas): {list(data['deltas'])}")
    print(f"  n_b: {int(data['n_b'])}")

    has_small_delta = 'small_delta' in fields
    has_b_values    = 'b_values'    in fields
    if has_small_delta and has_b_values and not args.force:
        print("\n  Already has small_delta and b_values; nothing to do.")
        print("  (Pass --force to overwrite.)")
        return

    if int(data['n_b']) != len(args.b_values):
        print(f"\nERROR: --b-values has {len(args.b_values)} entries but "
              f"library was built with n_b={int(data['n_b'])}.  Pass exactly "
              f"{int(data['n_b'])} b-values."); sys.exit(2)

    # Assemble the migrated payload
    payload = {k: data[k] for k in fields}
    payload['small_delta'] = np.array(float(args.small_delta))
    payload['b_values']    = np.asarray(args.b_values, dtype=float)

    out = args.output or args.library
    if out == args.library:
        bak = args.library + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(args.library, bak)
            print(f"\n  Wrote backup: {bak}")
        else:
            print(f"\n  Backup already exists, leaving it: {bak}")

    np.savez(out, **payload)
    print(f"\n  Wrote: {out}")
    print(f"    small_delta = {args.small_delta} ms")
    print(f"    b_values    = {args.b_values} s/mm²")


if __name__ == "__main__":
    main()

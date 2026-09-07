#!/usr/bin/env python3
"""Extract the Phase-1 selected columns of `vectors` and `signal_variance` once.

Phase 2 needs, at arbitrary nodes and arbitrary columns: the normalized signal
`S` (for the nuisance-amplitude block), and `signal_variance` at both stencil
endpoints (for the Monte-Carlo debias).  NPZ members cannot be memory-mapped
and random row access into a compressed member is expensive, so both are
materialized once into dense `.npy` caches over the Phase-1 column domain.
Nothing is recomputed here: the caches are verbatim slices.

The cache inherits the Phase-1 domain and records it, so a restricted Phase-1
run cannot be mistaken downstream for a universal substrate.  At the full stored
grid the two members are 2.34 GiB each (18,820 entries x 31,125 columns,
float32), plus the column-major transposes Phase 2 builds beside them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from madi.fisher_crlb import (artifact_manifest, assert_safe_output, incomplete_banner,
                              iter_npz_array_chunks, npz_array_info, read_column_domain)

MEMBERS = ("vectors", "signal_variance")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    args = parser.parse_args()

    manifest_json = json.loads((args.phase1 / "phase1_manifest.json").read_text(encoding="utf-8"))
    domain = read_column_domain(manifest_json)
    selected = np.asarray(domain.column_indices, dtype=int)
    print(domain.banner())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.artifact, allow_pickle=False) as data:
        manifest = artifact_manifest(data)
    assert_safe_output(args.cache_dir, manifest)

    report = {"schema": "madi-fisher-phase2-cache-v2", "artifact": str(args.artifact),
              "phase1": str(args.phase1), "banner": incomplete_banner(manifest),
              "column_domain": domain.as_dict(), "column_domain_banner": domain.banner(),
              "selected_columns": int(len(selected)), "members": {}}
    for member in MEMBERS:
        shape, _, dtype = npz_array_info(args.artifact, member)
        destination = args.cache_dir / f"{member}_selected.npy"
        out = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                        shape=(int(shape[0]), len(selected)))
        for start, chunk in iter_npz_array_chunks(args.artifact, member, rows_per_chunk=args.rows_per_chunk):
            out[start:start + len(chunk)] = chunk[:, selected]
        out.flush()
        del out
        report["members"][member] = {"file": destination.name, "rows": int(shape[0]),
                                     "columns": int(len(selected)), "source_dtype": str(dtype),
                                     "bytes": int(destination.stat().st_size)}
        print(f"{member}: {destination} ({destination.stat().st_size / 2**30:.2f} GiB)")

    # The CRN ensemble means exist only for the 200 diagnostic columns.  They are
    # what makes the *exact* Var(J_hat) available at those columns, and so what
    # calibrates how conservative the endpoint-only form is everywhere else.
    subset_shape, _, _ = npz_array_info(args.artifact, "ensemble_means_subset")
    destination = args.cache_dir / "ensemble_means_subset.npy"
    out = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=tuple(subset_shape))
    for start, chunk in iter_npz_array_chunks(args.artifact, "ensemble_means_subset",
                                              rows_per_chunk=args.rows_per_chunk):
        out[start:start + len(chunk)] = chunk
    out.flush()
    del out
    report["members"]["ensemble_means_subset"] = {"file": destination.name,
                                                  "shape": [int(v) for v in subset_shape],
                                                  "bytes": int(destination.stat().st_size)}
    print(f"ensemble_means_subset: {destination} ({destination.stat().st_size / 2**30:.2f} GiB)")
    (args.cache_dir / "phase2_cache_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["banner"])
    print(report["column_domain_banner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

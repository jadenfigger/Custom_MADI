"""One-time transpose of the universal library into a column-major cache.

Why
---
The library matrix is stored row-major: one row (one parameter triple) is
249 KB of adjacent bytes, and one column (one acquisition coordinate) is
18,820 values spread 249 KB apart.  The explorer wants columns, so every
request from the original file costs a full 4.7 GB scan (~3 s).  Transposed,
a column is 150 KB of adjacent bytes (~10 ms).

What it writes (into a cache folder, NEVER beside the library):

    <stem>.columns_f64.npy            (n_columns, n_entries) float64, 4.69 GB
    <stem>.variance_columns_f32.npy   (n_columns, n_entries) float32, 2.34 GB
    <stem>.columns.json               provenance sidecar

float64 keeps the signal cache bit-identical to the source, so the cached and
uncached readers can be tested for exact equality.  The variance cache is
float32 because the source member is float32; that too is exact.

How
---
A naive transpose would write 31,125 tiny scattered pieces per pass.  Instead
this walks BANDS OF COLUMNS: for one band it streams every row, keeps only
that band (contiguous inside each row), transposes the band in RAM, and
appends it to the output as one long sequential write.  Band width is chosen
from a memory budget, so peak RAM is roughly ``--band-bytes`` (default 400 MB).

The source is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .columns import (
    SIGNAL_MEMBER,
    VARIANCE_MEMBER,
    cache_paths,
    memmap_member,
    npy_member_layout,
)

DEFAULT_LIBRARY = Path(
    "C:/Miscellaneous/Coding_Projects/Python/mri_processing/processing/madi_gpu/"
    "Custom_MADI/data/libraries/madi_dense_universal_remediated.npz"
)


def transpose_member(library_path: Path, member: str, destination: Path,
                     out_dtype: np.dtype, band_bytes: int,
                     progress: bool = True) -> dict:
    """Write ``member`` transposed to ``destination`` as a flat .npy.

    Returns a small record of what was written.
    """
    _, (n_entries, n_columns), source_dtype = npy_member_layout(library_path, member)
    source = memmap_member(library_path, member)

    # One band holds `band_columns` columns for all rows, twice over (the read
    # block and its transpose), so budget half the allowance to each.
    bytes_per_column_all_rows = n_entries * source_dtype.itemsize
    band_columns = max(1, int(band_bytes // (2 * bytes_per_column_all_rows)))
    band_columns = min(band_columns, n_columns)

    out = np.lib.format.open_memmap(
        destination, mode="w+", dtype=out_dtype, shape=(n_columns, n_entries)
    )
    started = time.time()
    n_bands = (n_columns + band_columns - 1) // band_columns
    for band_index, start in enumerate(range(0, n_columns, band_columns)):
        stop = min(start + band_columns, n_columns)
        block = np.asarray(source[:, start:stop])        # (entries, band)
        out[start:stop, :] = block.T.astype(out_dtype, copy=False)
        if progress:
            elapsed = time.time() - started
            done = (band_index + 1) / n_bands
            print(
                f"  [{member}] band {band_index + 1}/{n_bands} "
                f"(columns {start}..{stop - 1})  {elapsed:6.1f}s elapsed, "
                f"~{elapsed / done - elapsed:5.1f}s left",
                flush=True,
            )
    out.flush()
    del out
    duration = time.time() - started
    return {
        "member": member,
        "path": str(destination),
        "shape": [int(n_columns), int(n_entries)],
        "source_dtype": str(source_dtype),
        "dtype": str(np.dtype(out_dtype)),
        "band_columns": int(band_columns),
        "bytes": int(destination.stat().st_size),
        "seconds": round(duration, 2),
    }


def build(library_path: Path, cache_dir: Path | None = None,
          with_variance: bool = True, signal_dtype: str = "float64",
          band_bytes: int = 400_000_000, overwrite: bool = False) -> dict:
    library_path = Path(library_path).resolve()
    if not library_path.exists():
        raise SystemExit(f"library not found: {library_path}")
    paths = cache_paths(library_path, cache_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if paths["signal"].exists() and not overwrite:
        raise SystemExit(
            f"refusing to overwrite {paths['signal']} (pass --overwrite to rebuild)"
        )

    _, (n_entries, n_columns), _ = npy_member_layout(library_path, SIGNAL_MEMBER)
    needed = n_entries * n_columns * np.dtype(signal_dtype).itemsize
    if with_variance:
        needed += n_entries * n_columns * 4
    free = _free_disk_bytes(paths["dir"])
    print(f"library : {library_path}")
    print(f"cache   : {paths['dir']}")
    print(f"matrix  : {n_entries} entries x {n_columns} columns")
    print(f"need    : {needed / 1e9:.2f} GB   free: {free / 1e9:.1f} GB")
    if free < needed * 1.05:
        raise SystemExit(
            f"not enough free disk: need {needed / 1e9:.2f} GB, have {free / 1e9:.1f} GB"
        )

    # A sidecar marked incomplete until the end means an interrupted build is
    # never mistaken for a usable cache.
    record = {
        "source": str(library_path),
        "source_size_bytes": library_path.stat().st_size,
        "complete": False,
        "members": [],
    }
    paths["sidecar"].write_text(json.dumps(record, indent=2) + "\n")

    started = time.time()
    record["members"].append(
        transpose_member(library_path, SIGNAL_MEMBER, paths["signal"],
                         np.dtype(signal_dtype), band_bytes)
    )
    if with_variance:
        record["members"].append(
            transpose_member(library_path, VARIANCE_MEMBER, paths["variance"],
                             np.dtype("float32"), band_bytes)
        )
    record["complete"] = True
    record["total_seconds"] = round(time.time() - started, 2)
    record["total_bytes"] = sum(item["bytes"] for item in record["members"])
    record["built_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    paths["sidecar"].write_text(json.dumps(record, indent=2) + "\n")

    print(
        f"\ndone in {record['total_seconds']:.1f}s, "
        f"{record['total_bytes'] / 1e9:.2f} GB written to {paths['dir']}"
    )
    return record


def _free_disk_bytes(directory: Path) -> int:
    usage = os.statvfs(directory) if hasattr(os, "statvfs") else None
    if usage is not None:
        return usage.f_bavail * usage.f_frsize
    import shutil

    return shutil.disk_usage(directory).free


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY,
                        help="universal library .npz (read-only)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="cache folder (default: <library>/../../cache)")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"),
                        help="signal cache dtype; float64 is bit-exact (default)")
    parser.add_argument("--no-variance", action="store_true",
                        help="skip the signal_variance cache")
    parser.add_argument("--band-bytes", type=int, default=400_000_000,
                        help="approximate peak RAM for the transpose buffer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.library, args.cache_dir, with_variance=not args.no_variance,
          signal_dtype=args.dtype, band_bytes=args.band_bytes,
          overwrite=args.overwrite)


if __name__ == "__main__":
    main()

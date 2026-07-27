#!/usr/bin/env python3
"""Create a low-memory protocol-specific derivative of the universal library.

Only the four columns used by the edema cohort's single-Delta acquisition are
retained: (delta=20 ms, Delta=50 ms, b=1000/1500/2000/2500 s/mm^2). Every
universal-library candidate and its (kio, rho, V) values is preserved exactly.

The source .npz is streamed row-by-row from its compressed ``vectors.npy``
member, so the 10 GB universal matrix is never held in RAM. The resulting
library is appropriate only for the stated protocol, but can be used by the
low-memory controlled S0 comparison without changing its candidate curves.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np

from madi import library


FIT_BVALUES = np.array([1000.0, 1500.0, 2000.0, 2500.0])
FIT_TRIPLES = [(20.0, 50.0, float(b)) for b in FIT_BVALUES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/libraries/madi_dense_universal.npz", help="Universal-library input .npz.")
    parser.add_argument(
        "--output",
        default="data/libraries/madi_dense_universal_delta20_D50_b1000-2500.npz",
        help="Protocol-specific output .npz.",
    )
    return parser.parse_args()


def _read_exact(handle: BinaryIO, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise EOFError(f"vectors.npy ended with {remaining} bytes still required")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_npy_header(handle: BinaryIO):
    version = np.lib.format.read_magic(handle)
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
        (3, 0): np.lib.format.read_array_header_2_0,
    }
    if version not in readers:
        raise ValueError(f"unsupported vectors.npy version {version}")
    return readers[version](handle)


def stream_protocol_vectors(source: Path, destination: Path, columns: np.ndarray) -> tuple[int, np.dtype]:
    """Stream selected columns from an npz's vectors.npy without full-array RAM."""
    with zipfile.ZipFile(source) as archive:
        member = next((name for name in archive.namelist() if name.endswith("vectors.npy")), None)
        if member is None:
            raise KeyError("vectors.npy is missing from the input library")
        with archive.open(member) as handle:
            shape, fortran_order, dtype = _read_npy_header(handle)
            if len(shape) != 2 or fortran_order:
                raise ValueError(f"expected a C-order 2D vectors array, got shape={shape}, fortran={fortran_order}")
            n_rows, n_columns = shape
            if columns.min() < 0 or columns.max() >= n_columns:
                raise ValueError(f"requested columns {columns.tolist()} outside vectors shape {shape}")
            output = np.lib.format.open_memmap(destination, mode="w+", dtype=dtype, shape=(n_rows, len(columns)))
            row_bytes = n_columns * dtype.itemsize
            for row in range(n_rows):
                values = np.frombuffer(_read_exact(handle, row_bytes), dtype=dtype, count=n_columns)
                output[row] = values[columns]
            output.flush()
    return int(n_rows), dtype


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if not source.exists():
        raise SystemExit(f"input library is missing: {source}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = library.load_library_meta(str(source))
    columns = library._grid_columns(FIT_TRIPLES, metadata["delta_pairs"], metadata["b_values"], metadata["n_b"])
    temporary_vectors = output.with_suffix(".vectors.tmp.npy")
    n_rows, dtype = stream_protocol_vectors(source, temporary_vectors, columns)
    try:
        with np.load(source) as original:
            kios = np.asarray(original["kios"])
            rhos = np.asarray(original["rhos"])
            volumes = np.asarray(original["Vs"])
        vectors = np.load(temporary_vectors, mmap_mode="r")
        np.savez(
            output,
            kios=kios,
            rhos=rhos,
            Vs=volumes,
            vectors=vectors,
            pair_deltas=np.array([20.0]),
            pair_Deltas=np.array([50.0]),
            b_values=FIT_BVALUES,
            n_b=np.array(len(FIT_BVALUES)),
        )
    finally:
        if temporary_vectors.exists():
            temporary_vectors.unlink()

    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "source": str(source),
                "source_vector_columns": columns.tolist(),
                "protocol": {"delta_ms": 20.0, "Delta_ms": 50.0, "b_s_mm2": FIT_BVALUES.tolist()},
                "candidate_count": n_rows,
                "dtype": str(dtype),
                "note": "All universal-library candidates preserved; only unused protocol columns were removed.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} with {n_rows} candidates x {len(FIT_BVALUES)} protocol columns")
    print(f"source columns: {columns.tolist()}")


if __name__ == "__main__":
    main()

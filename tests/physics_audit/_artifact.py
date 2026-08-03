"""Low-memory helpers for auditing the production universal library.

The production ``.npz`` is roughly 10 GB.  ``numpy.load`` materialises an
entire member when it is indexed, so these helpers read small metadata arrays
normally and memory-map the uncompressed ``vectors.npy`` member directly.
They deliberately validate the ZIP/NPY layout before doing so; a compressed
member is rejected rather than silently copied into RAM.
"""

from __future__ import annotations

import ast
import os
import struct
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = ROOT / "data" / "libraries" / "madi_dense_universal.npz"


def production_library_path() -> Path:
    """Return the audited artifact, overridable for a rebuilt candidate."""
    return Path(os.environ.get("MADI_PRODUCTION_LIBRARY", DEFAULT_LIBRARY))


def require_production_library() -> Path:
    path = production_library_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Production library not found: {path}. Set MADI_PRODUCTION_LIBRARY."
        )
    return path


def metadata(path: Path | None = None) -> dict[str, np.ndarray]:
    """Load only the small metadata/parameter arrays, never ``vectors``."""
    path = require_production_library() if path is None else Path(path)
    wanted = (
        "kios", "rhos", "Vs", "pair_deltas", "pair_Deltas", "b_values",
        "n_b", "h_ms",
    )
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in wanted if name in data.files}


def _npy_member_layout(path: Path, array_name: str) -> tuple[np.dtype, tuple[int, ...], bool, int]:
    """Return dtype, shape, Fortran order and raw data offset for an NPY ZIP member."""
    member = f"{array_name}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(
                f"{path}:{member} is compressed; direct memory mapping is unsafe."
            )

        with path.open("rb") as fh:
            fh.seek(info.header_offset)
            local = fh.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04":
                raise RuntimeError(f"Invalid local ZIP header for {member}")
            name_len, extra_len = struct.unpack("<HH", local[26:30])
            npy_start = info.header_offset + 30 + name_len + extra_len

            fh.seek(npy_start)
            magic = fh.read(6)
            if magic != b"\x93NUMPY":
                raise RuntimeError(f"{member} does not contain an NPY payload")
            major, minor = struct.unpack("BB", fh.read(2))
            if (major, minor) == (1, 0):
                header_len = struct.unpack("<H", fh.read(2))[0]
            elif major in (2, 3):
                header_len = struct.unpack("<I", fh.read(4))[0]
            else:
                raise RuntimeError(f"Unsupported NPY version {(major, minor)}")
            header = ast.literal_eval(fh.read(header_len).decode("latin1"))
            data_offset = fh.tell()

    return (
        np.dtype(header["descr"]),
        tuple(int(x) for x in header["shape"]),
        bool(header["fortran_order"]),
        data_offset,
    )


def vectors_memmap(path: Path | None = None) -> np.memmap:
    """Open the production signal matrix without allocating the 10 GB matrix."""
    path = require_production_library() if path is None else Path(path)
    dtype, shape, fortran_order, offset = _npy_member_layout(path, "vectors")
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def chunks(n_rows: int, chunk_rows: int) -> Iterator[slice]:
    for start in range(0, n_rows, chunk_rows):
        yield slice(start, min(start + chunk_rows, n_rows))


@lru_cache(maxsize=4)
def scan_signal_quality(
    path: Path | None = None,
    *,
    chunk_rows: int = 512,
    trust_floor: float = 0.015,
) -> dict[str, float | int]:
    """Scan all signals while respecting pair-major / b-major storage."""
    meta = metadata(path)
    vec = vectors_memmap(path)
    n_b = int(meta["n_b"])
    if vec.shape[1] % n_b:
        raise AssertionError("vector length is not divisible by n_b")

    n_pairs = vec.shape[1] // n_b
    stats: dict[str, float | int] = {
        "n_values": int(vec.size),
        "nonfinite": 0,
        "negative": 0,
        "b0_not_one": 0,
        "increases_total": 0,
        "increases_at_or_above_floor": 0,
        "max_increase": 0.0,
        "minimum": float("inf"),
    }
    for rows in chunks(vec.shape[0], chunk_rows):
        block = np.asarray(vec[rows])
        cube = block.reshape(block.shape[0], n_pairs, n_b)
        stats["nonfinite"] += int((~np.isfinite(cube)).sum())
        stats["negative"] += int((cube < 0).sum())
        stats["b0_not_one"] += int((cube[:, :, 0] != 1.0).sum())
        stats["minimum"] = min(float(stats["minimum"]), float(np.nanmin(cube)))
        delta = cube[:, :, 1:] - cube[:, :, :-1]
        positive = delta > 0
        stats["increases_total"] += int(positive.sum())
        stats["increases_at_or_above_floor"] += int(
            (positive & (cube[:, :, :-1] >= trust_floor)).sum()
        )
        if delta.size:
            stats["max_increase"] = max(float(stats["max_increase"]), float(delta.max()))
    return stats


def shard_paths(root: Path | None = None) -> list[Path]:
    root = ROOT if root is None else Path(root)
    return sorted((root / "libraries").glob("madi_dense.shard*.npz"))


def triplet_keys(kios: np.ndarray, rhos: np.ndarray, volumes: np.ndarray) -> set[tuple[float, float, float]]:
    """Use the same rounding convention as the production shard merger."""
    return {
        (round(float(k), 4), round(float(r), 1), round(float(v), 6))
        for k, r, v in zip(kios, rhos, volumes)
    }

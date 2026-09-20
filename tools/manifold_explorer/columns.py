"""Read (delta, Delta, b) columns out of the universal MADI library.

Geometry in plain terms
-----------------------
The library is a sampled model manifold.  Each ROW is one point on it: a
parameter triple (rho, V, k_io) together with the signal that triple predicts
at every stored acquisition.  Each COLUMN is one acquisition coordinate
(delta, Delta, b).  So the library matrix is

    vectors[row, column] = S/S0 predicted by row's parameters at column's
                           (delta, Delta, b)

with 18,820 rows and 31,125 columns.  "Slicing" the manifold means fixing a
few columns and asking which rows still agree there, so everything this module
does is: given a handful of column indices, hand back that thin slab of the
matrix without touching the other 4.7 GB.

Two readers provide the same slab:

``NpzColumnReader``
    Memory-maps straight into the original ``.npz``.  Legal because every
    member of the production artifact is stored uncompressed (ZIP_STORED), so
    an NPY member is just a flat run of bytes at a known offset.  No setup, but
    the matrix is row-major, so pulling one column touches a page in every one
    of the 18,820 rows -- a full-file scan, about 3 s per request.

``CachedColumnReader``
    Memory-maps the transposed cache written by ``build_column_cache.py``,
    where one column is 150 KB of contiguous bytes.  About 10 ms per request.

The original ``.npz`` is opened read-only, always.  Nothing here reads
``entry_metadata_json`` (5.3 GB of per-entry JSON) or any full-matrix member.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Members that carry one small value per library row or per acquisition axis.
# These are the only members loaded whole; the largest is 150 KB.
LABEL_MEMBERS = (
    "kios", "rhos", "Vs", "vis",
    "nominal_kios", "nominal_rhos", "nominal_Vs",
    "weights", "is_free_water",
    "pair_deltas", "pair_Deltas", "b_values", "n_b", "h_ms",
)

# The two (n_entries, n_columns) matrices the explorer uses.
SIGNAL_MEMBER = "vectors"
VARIANCE_MEMBER = "signal_variance"

# The builder averages this many independent ensembles per entry, and
# `build_metadata_json` states the consumer contract explicitly:
# "consumer SE is sqrt(signal_variance / n_ensembles)".
N_ENSEMBLES = 40


def npy_member_layout(path: str | Path, member: str) -> tuple[int, tuple[int, ...], np.dtype]:
    """Byte offset, shape and dtype of an uncompressed NPY member of an NPZ.

    Returns the offset of the member's first DATA byte inside the zip, which
    is what ``np.memmap`` needs.  Reads only the local file header and the NPY
    header -- a few hundred bytes -- never the array.
    """
    name = f"{member}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(
            f"{member} is compressed (type {info.compress_type}); it cannot be "
            "memory-mapped. Use build_column_cache.py to make a readable cache."
        )
    with open(path, "rb") as handle:
        handle.seek(info.header_offset)
        local = handle.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError(f"{member}: local file header not found in {path}")
        name_length, extra_length = struct.unpack("<HH", local[26:30])
        data_start = info.header_offset + 30 + name_length + extra_length
        handle.seek(data_start)
        header_bytes = handle.read(4096)

    stream = io.BytesIO(header_bytes)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"{member}: unsupported NPY version {version}")
    if fortran:
        raise ValueError(f"{member}: Fortran-order members are not supported")
    return data_start + stream.tell(), tuple(shape), np.dtype(dtype)


def memmap_member(path: str | Path, member: str) -> np.memmap:
    """Read-only memory map of one uncompressed NPY member of an NPZ."""
    offset, shape, dtype = npy_member_layout(path, member)
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def read_small_member(path: str | Path, member: str) -> np.ndarray:
    """Read one small member whole (axis labels, scalars -- never a matrix)."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as stream:
            return np.lib.format.read_array(stream, allow_pickle=False)


# ---------------------------------------------------------------------------
# Labels: what each row and each column means
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LibraryLabels:
    """Row labels (parameters) and column labels (acquisition) of the library.

    ``rhos``/``Vs`` are the REALISED finite-geometry values the builder
    measured; ``nominal_rhos``/``nominal_Vs`` are the exact canonical grid
    nodes that were requested.  They differ by up to ~1%.  Plot against the
    nominal values when you want the grid to look like a grid (it is
    log-spaced in rho and V); use the realised values when you want the
    parameters the entry actually has.  For k_io the two are identical.
    """

    kios: np.ndarray
    rhos: np.ndarray
    Vs: np.ndarray
    vis: np.ndarray
    nominal_kios: np.ndarray
    nominal_rhos: np.ndarray
    nominal_Vs: np.ndarray
    weights: np.ndarray
    is_free_water: np.ndarray
    pair_deltas: np.ndarray
    pair_Deltas: np.ndarray
    b_values: np.ndarray
    n_b: int
    h_ms: float
    n_entries: int
    n_columns: int

    # -- column bookkeeping -------------------------------------------------

    @property
    def n_pairs(self) -> int:
        return len(self.pair_deltas)

    def pair_index(self, delta: float, Delta: float) -> int:
        """Index of the stored (delta, Delta) pair, which must exist exactly."""
        hits = np.flatnonzero(
            np.isclose(self.pair_deltas, delta) & np.isclose(self.pair_Deltas, Delta)
        )
        if hits.size != 1:
            raise KeyError(f"(delta={delta:g}, Delta={Delta:g}) ms is not a stored pair")
        return int(hits[0])

    def b_index(self, b: float) -> int:
        hits = np.flatnonzero(np.isclose(self.b_values, b))
        if hits.size != 1:
            raise KeyError(f"b={b:g} s/mm^2 is not a stored b-value")
        return int(hits[0])

    def column_index(self, delta: float, Delta: float, b: float) -> int:
        """Flat column index of one acquisition coordinate.

        The flat layout is pair-major: column = pair_index * n_b + b_index.
        """
        return self.pair_index(delta, Delta) * self.n_b + self.b_index(b)

    def column_indices(self, triples) -> np.ndarray:
        return np.array(
            [self.column_index(d, D, b) for d, D, b in triples], dtype=np.int64
        )

    def column_triple(self, column: int) -> tuple[float, float, float]:
        """(delta, Delta, b) of a flat column index -- the inverse lookup."""
        pair, b_index = divmod(int(column), self.n_b)
        return (float(self.pair_deltas[pair]), float(self.pair_Deltas[pair]),
                float(self.b_values[b_index]))

    def column_label(self, column: int) -> str:
        delta, Delta, b = self.column_triple(column)
        return f"d{delta:g}/D{Delta:g}/b{b:g}"

    def columns_for_pair(self, delta: float, Delta: float,
                         b_min: float = -np.inf, b_max: float = np.inf) -> np.ndarray:
        """Every b column at one (delta, Delta), optionally inside a b window."""
        pair = self.pair_index(delta, Delta)
        keep = np.flatnonzero((self.b_values >= b_min) & (self.b_values <= b_max))
        return pair * self.n_b + keep

    def columns_for_b(self, b: float, delta: float | None = None) -> np.ndarray:
        """One b across every stored Delta (optionally at one fixed delta)."""
        b_index = self.b_index(b)
        pairs = np.arange(self.n_pairs)
        if delta is not None:
            pairs = pairs[np.isclose(self.pair_deltas, delta)]
        return pairs * self.n_b + b_index

    def delta_pairs_list(self) -> list[tuple[float, float]]:
        """The (delta, Delta) pairs in stored order, as madi.library wants them."""
        return list(zip(self.pair_deltas.tolist(), self.pair_Deltas.tolist()))

    def deltas(self) -> np.ndarray:
        return np.unique(self.pair_deltas)

    def Deltas_for_delta(self, delta: float) -> np.ndarray:
        return np.sort(self.pair_Deltas[np.isclose(self.pair_deltas, delta)])


def load_labels(library_path: str | Path) -> LibraryLabels:
    """Read every small label member.  Touches ~1.3 MB, not the 4.7 GB matrix."""
    values: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(library_path) as archive:
        available = set(archive.namelist())
        for member in LABEL_MEMBERS:
            if f"{member}.npy" not in available:
                raise KeyError(f"{library_path} is missing the {member} array")
            with archive.open(f"{member}.npy") as stream:
                values[member] = np.lib.format.read_array(stream, allow_pickle=False)
    _, signal_shape, _ = npy_member_layout(library_path, SIGNAL_MEMBER)
    n_entries, n_columns = signal_shape
    if n_columns != len(values["pair_deltas"]) * int(values["n_b"]):
        raise ValueError(
            f"{SIGNAL_MEMBER} has {n_columns} columns but the grid says "
            f"{len(values['pair_deltas'])} pairs x {int(values['n_b'])} b-values"
        )
    return LibraryLabels(
        kios=values["kios"], rhos=values["rhos"], Vs=values["Vs"], vis=values["vis"],
        nominal_kios=values["nominal_kios"],
        nominal_rhos=values["nominal_rhos"],
        nominal_Vs=values["nominal_Vs"],
        weights=values["weights"],
        is_free_water=values["is_free_water"].astype(bool),
        pair_deltas=values["pair_deltas"].astype(float),
        pair_Deltas=values["pair_Deltas"].astype(float),
        b_values=values["b_values"].astype(float),
        n_b=int(values["n_b"]),
        h_ms=float(values["h_ms"]),
        n_entries=int(n_entries),
        n_columns=int(n_columns),
    )


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

class NpzColumnReader:
    """Pull columns straight from the row-major matrix inside the .npz.

    Correct but slow: because the matrix is row-major, every column read walks
    one 8-byte value per 249 KB row, which the OS serves as a full scan of the
    4.7 GB member (~3 s).  Use it when no cache has been built.
    """

    kind = "npz"

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path)
        self._signal = memmap_member(self.library_path, SIGNAL_MEMBER)
        self._variance = memmap_member(self.library_path, VARIANCE_MEMBER)
        self.n_entries, self.n_columns = self._signal.shape

    @property
    def has_variance(self) -> bool:
        return True

    def read(self, columns) -> np.ndarray:
        """(n_entries, len(columns)) float64 block of S/S0 predictions."""
        columns = np.asarray(columns, dtype=np.int64)
        return np.ascontiguousarray(self._signal[:, columns], dtype=np.float64)

    def read_variance(self, columns) -> np.ndarray:
        """Matching block of the between-ensemble sample variance."""
        columns = np.asarray(columns, dtype=np.int64)
        return np.ascontiguousarray(self._variance[:, columns], dtype=np.float64)

    def close(self) -> None:
        self._signal = None
        self._variance = None


class CachedColumnReader:
    """Pull columns from the column-major cache: one column, one contiguous run.

    The cache holds the same numbers transposed, so ``cache[column]`` is the
    whole library's prediction at that acquisition coordinate, 150 KB of
    adjacent bytes.  Reading 50 columns is a few megabytes.
    """

    kind = "cache"

    def __init__(self, signal_cache: str | Path, variance_cache: str | Path | None = None):
        self.signal_path = Path(signal_cache)
        self._signal = np.load(self.signal_path, mmap_mode="r")
        self.n_columns, self.n_entries = self._signal.shape
        self.variance_path = Path(variance_cache) if variance_cache else None
        self._variance = (
            np.load(self.variance_path, mmap_mode="r")
            if self.variance_path is not None and self.variance_path.exists() else None
        )

    @property
    def has_variance(self) -> bool:
        return self._variance is not None

    def read(self, columns) -> np.ndarray:
        columns = np.asarray(columns, dtype=np.int64)
        # Cache is (column, entry); the explorer wants (entry, column).
        return np.ascontiguousarray(self._signal[columns, :].T, dtype=np.float64)

    def read_variance(self, columns) -> np.ndarray:
        if self._variance is None:
            raise RuntimeError(
                f"no variance cache beside {self.signal_path}; rebuild with "
                "build_column_cache.py (variance is on by default)"
            )
        columns = np.asarray(columns, dtype=np.int64)
        return np.ascontiguousarray(self._variance[columns, :].T, dtype=np.float64)

    def close(self) -> None:
        self._signal = None
        self._variance = None


# ---------------------------------------------------------------------------
# Cache naming and selection
# ---------------------------------------------------------------------------

def default_cache_dir(library_path: str | Path) -> Path:
    """``<library parent>/../cache`` -- a separate folder, never data/libraries.

    Keeping the cache out of the library folder means it can never be mistaken
    for a library artifact by anything that globs that directory.
    """
    return Path(library_path).resolve().parent.parent / "cache"


def cache_paths(library_path: str | Path, cache_dir: str | Path | None = None) -> dict:
    directory = Path(cache_dir) if cache_dir else default_cache_dir(library_path)
    stem = Path(library_path).stem
    return {
        "dir": directory,
        "signal": directory / f"{stem}.columns_f64.npy",
        "variance": directory / f"{stem}.variance_columns_f32.npy",
        "sidecar": directory / f"{stem}.columns.json",
    }


def cache_is_valid(library_path: str | Path,
                   cache_dir: str | Path | None = None) -> tuple[bool, str]:
    """Is there a cache, and was it built from this exact library file?"""
    paths = cache_paths(library_path, cache_dir)
    if not paths["signal"].exists():
        return False, f"no cache at {paths['signal']}"
    if not paths["sidecar"].exists():
        return False, f"cache has no sidecar at {paths['sidecar']}"
    record = json.loads(paths["sidecar"].read_text())
    source = Path(library_path).resolve()
    if record.get("source_size_bytes") != source.stat().st_size:
        return False, "cache was built from a different-sized library file"
    if not record.get("complete", False):
        return False, "cache sidecar says the build did not finish"
    return True, "ok"


def open_reader(library_path: str | Path, cache_dir: str | Path | None = None,
                prefer_cache: bool = True):
    """Best available reader: the column cache if it is valid, else the .npz."""
    if prefer_cache:
        valid, _ = cache_is_valid(library_path, cache_dir)
        if valid:
            paths = cache_paths(library_path, cache_dir)
            return CachedColumnReader(paths["signal"], paths["variance"])
    return NpzColumnReader(library_path)

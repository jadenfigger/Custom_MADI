"""Read-only library access shared by exploration workers."""

from __future__ import annotations

import functools
from pathlib import Path
import numpy as np
from . import fisher as fisher_tools
from . import slice as slicing
from .columns import cache_is_valid, load_labels, open_reader
from .runtime import ArrayCache


class ExplorerData:
    """Labels, a memoised window onto the columns, and the canonical node grid."""

    def __init__(self, library_path: Path, cache_dir: Path | None = None,
                 prefer_cache: bool = True, cache_mb: int = 256):
        self.library_path = Path(library_path)
        self.labels = load_labels(self.library_path)
        self.reader = open_reader(self.library_path, cache_dir, prefer_cache)
        self.cache_ok, self.cache_note = cache_is_valid(self.library_path, cache_dir)
        self.arrays = ArrayCache(cache_mb * 1024 * 1024)
        # The (rho, V, k_io) node grid is what finite-difference stencils step
        # through.  Built once: it costs about half a second.
        self.node_grid = fisher_tools.build_node_grid(self.labels)
        self.stencil_coverage = {
            width: fisher_tools.coverage(self.node_grid, width) for width in (1, 2)
        }

    def _read(self, columns: tuple[int, ...]) -> np.ndarray:
        return self.arrays.get(("signal", columns),
                               lambda: self.reader.read(np.asarray(columns, dtype=np.int64)))

    def _read_variance(self, columns: tuple[int, ...]) -> np.ndarray:
        return self.arrays.get(("variance", columns), lambda: self.reader.read_variance(
            np.asarray(columns, dtype=np.int64)))

    def block(self, columns) -> np.ndarray:
        """(n_entries, len(columns)) S/S0, memoised on the column set."""
        return self._read(tuple(int(c) for c in columns))

    def variance(self, columns) -> np.ndarray:
        return self._read_variance(tuple(int(c) for c in columns))

    @functools.lru_cache(maxsize=8)
    def _crn(self, columns: tuple[int, ...]):
        return fisher_tools.build_crn_subset(self.library_path, self.labels,
                                             np.asarray(columns, dtype=np.int64))

    def crn_subset(self, columns):
        return self._crn(tuple(int(c) for c in columns))

    def _batch_jacobian(self, columns: tuple[int, ...], width: int):
        """Jacobians at every node with a complete stencil, cached on the columns.

        Cached on (columns, width) rather than on sigma, because sigma only
        rescales the Fisher matrix afterwards.
        """
        return self.arrays.get(("jacobian", columns, width), lambda:
                              fisher_tools.jacobian_batch(self.node_grid, self.block(columns), width))

    def batch_jacobian(self, columns, width: int):
        return self._batch_jacobian(tuple(int(c) for c in columns), int(width))

    def nearest_entry(self, rho: float, V: float, kio: float,
                      eligible: np.ndarray) -> int:
        """Library row closest to a requested parameter triple.

        Distance is taken in (log rho, log V, k_io) because the rho and V grids
        are log-spaced; k_io is linear, scaled by its range so it counts
        comparably.
        """
        labels = self.labels
        eligible = np.asarray(eligible, dtype=int)
        eligible = eligible[np.isfinite(labels.kios[eligible])
                            & (labels.nominal_rhos[eligible] > 0)
                            & (labels.nominal_Vs[eligible] > 0)]
        if not len(eligible):
            raise ValueError("No cellular entries pass the candidate filter.")
        target = np.array([np.log(max(rho, 1e-12)), np.log(max(V, 1e-12)), kio])
        grid = np.column_stack([
            np.log(np.maximum(labels.nominal_rhos[eligible], 1e-12)),
            np.log(np.maximum(labels.nominal_Vs[eligible], 1e-12)),
            labels.kios[eligible],
        ])
        scale = np.array([1.0, 1.0, 1.0 / max(np.ptp(labels.kios[eligible]), 1e-9)])
        distance = np.sum(((grid - target) * scale) ** 2, axis=1)
        return int(eligible[int(np.argmin(distance))])

    @functools.lru_cache(maxsize=32)
    def eligible(self, vi_min, vi_max, include_free_water=False, rho_max=None):
        if not 0 <= vi_min <= vi_max <= 1:
            raise ValueError("The v_i band must satisfy 0 <= minimum <= maximum <= 1.")
        if rho_max is not None and (not np.isfinite(rho_max) or rho_max <= 0):
            raise ValueError("Maximum rho must be positive, or left empty.")
        rows = np.flatnonzero(slicing.candidate_mask(
            self.labels, vi_min, vi_max, rho_max, include_free_water))
        rows.setflags(write=False)
        return rows



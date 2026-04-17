"""Lazy, shared loaders for NIfTI volumes, libraries, and signals.

Every profile-backed view goes through ``ProfileData`` which knows how to
materialise maps, DWI cubes, library vectors, and per-voxel signal vectors.
A process-wide ``_cache`` is keyed on absolute file path so switching tabs
between profiles that share a mask / library / DWI never re-reads from disk.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib

# Reuse the production library loader
_MADI_DIR = Path(__file__).resolve().parent.parent
if str(_MADI_DIR) not in sys.path:
    sys.path.insert(0, str(_MADI_DIR))
from madi.library import load_library, load_library_meta  # noqa: E402

from .constants import SHELLS, N_SHELLS
from .app_state import OutputProfile


# ---------------------------------------------------------------------
# Process-wide cache
# ---------------------------------------------------------------------

class _Cache:
    def __init__(self) -> None:
        self._nii: dict[str, tuple[np.ndarray, np.ndarray, tuple]] = {}
        self._lib: dict[str, dict] = {}

    def get_nii(self, path: str):
        path = str(Path(path).resolve())
        if path not in self._nii:
            img = nib.load(path)
            data = img.get_fdata()
            affine = img.affine
            zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
            self._nii[path] = (data, affine, zooms)
        return self._nii[path]

    def get_library(self, path: str):
        path = str(Path(path).resolve())
        if path not in self._lib:
            lib = load_library(path)
            meta = load_library_meta(path)
            kios = np.array([e.kio for e in lib], dtype=np.float64)
            rhos = np.array([e.rho for e in lib], dtype=np.float64)
            Vs   = np.array([e.V   for e in lib], dtype=np.float64)
            vecs = np.array([e.vector for e in lib], dtype=np.float64)
            self._lib[path] = {
                "lib": lib,
                "meta": meta,
                "kios": kios,
                "rhos": rhos,
                "Vs": Vs,
                "vecs": vecs,
            }
        return self._lib[path]

    def drop(self, path: str) -> None:
        path = str(Path(path).resolve())
        self._nii.pop(path, None)
        self._lib.pop(path, None)

    def clear(self) -> None:
        self._nii.clear()
        self._lib.clear()


_cache = _Cache()


def cache() -> _Cache:
    return _cache


# ---------------------------------------------------------------------
# Per-profile data
# ---------------------------------------------------------------------

@dataclass
class ProfileData:
    """Materialised data for one output profile.

    All expensive work is behind `ensure_*` methods so the UI can open a
    profile cheaply and pull the pieces it needs on demand.
    """

    profile: OutputProfile

    # Populated on demand
    mask:         Optional[np.ndarray] = None
    affine:       Optional[np.ndarray] = None
    zooms:        Optional[np.ndarray] = None
    maps:         dict = field(default_factory=dict)   # name -> 3d array
    raw_dwi:      dict = field(default_factory=dict)   # delta_ms -> 4d array
    signal_vol:   Optional[np.ndarray] = None          # (X,Y,Z, n_fit*n_b)
    lib_bundle:   Optional[dict] = None                # from cache.get_library
    lib_sub:      Optional[np.ndarray] = None          # Δ-subset matrix
    lib_sub_sig:  Optional[str] = None                 # fingerprint for invalidation
    fit_deltas:   list = field(default_factory=list)
    load_errors:  list[str] = field(default_factory=list)

    # ------------- Mask -------------
    def ensure_mask(self) -> bool:
        if self.mask is not None:
            return True
        if not self.profile.mask_path:
            self.load_errors.append("No mask path set")
            return False
        try:
            data, affine, zooms = _cache.get_nii(self.profile.mask_path)
            self.mask = data.astype(bool)
            self.affine = affine
            self.zooms = np.array(zooms, dtype=float)
            return True
        except Exception as e:
            self.load_errors.append(f"Mask load failed: {e}")
            return False

    # ------------- Maps -------------
    def ensure_map(self, name: str) -> Optional[np.ndarray]:
        if name in self.maps:
            return self.maps[name]
        path = self.profile.map_path(name)
        if not Path(path).exists():
            return None
        try:
            data, affine, zooms = _cache.get_nii(path)
            self.maps[name] = data
            if self.affine is None:
                self.affine = affine
            if self.zooms is None:
                self.zooms = np.array(zooms, dtype=float)
            return data
        except Exception as e:
            self.load_errors.append(f"Map {name} load failed: {e}")
            return None

    def available_maps(self) -> list[str]:
        names = []
        for n in ["kio", "rho", "V", "residual", "s0_fit", "s0_fit_over_measured"]:
            if Path(self.profile.map_path(n)).exists():
                names.append(n)
        return names

    # ------------- Raw DWI -------------
    def ensure_raw(self, delta_ms: float) -> Optional[np.ndarray]:
        path = self.profile.input_dict().get(float(delta_ms))
        if not path:
            return None
        if delta_ms in self.raw_dwi:
            return self.raw_dwi[delta_ms]
        try:
            data, affine, zooms = _cache.get_nii(path)
            self.raw_dwi[delta_ms] = data
            if self.affine is None:
                self.affine = affine
            if self.zooms is None:
                self.zooms = np.array(zooms, dtype=float)
            return data
        except Exception as e:
            self.load_errors.append(f"DWI Δ={delta_ms} load failed: {e}")
            return None

    # ------------- Library -------------
    def ensure_library(self) -> bool:
        if self.lib_bundle is not None:
            return True
        if not self.profile.library_path:
            self.load_errors.append("No library path set")
            return False
        try:
            self.lib_bundle = _cache.get_library(self.profile.library_path)
            return True
        except Exception as e:
            self.load_errors.append(f"Library load failed: {e}")
            return False

    def library_deltas(self) -> list[float]:
        if not self.ensure_library():
            return []
        return list(self.lib_bundle["meta"]["deltas"])

    def _lib_subset_signature(self) -> str:
        return "|".join(f"{d:.2f}" for d in self.fit_deltas)

    def ensure_lib_sub(self, fit_deltas: list[float]) -> bool:
        """Build the Δ-subset library matrix for the current fit deltas."""
        if not self.ensure_library():
            return False
        self.fit_deltas = list(fit_deltas)
        sig = self._lib_subset_signature()
        if self.lib_sub is not None and self.lib_sub_sig == sig:
            return True
        bundle = self.lib_bundle
        lib_deltas = list(bundle["meta"]["deltas"])
        n_b = int(bundle["meta"]["n_b"])
        di_list = []
        for d in self.fit_deltas:
            for i, ld in enumerate(lib_deltas):
                if abs(d - ld) < 0.01:
                    di_list.append(i)
                    break
            else:
                self.load_errors.append(
                    f"Δ={d} ms not in library (available {lib_deltas})")
                return False
        vecs = bundle["vecs"]
        self.lib_sub = np.hstack(
            [vecs[:, di * n_b:(di + 1) * n_b] for di in di_list])
        self.lib_sub_sig = sig
        return True

    # ------------- Signal volume -------------
    def ensure_signal_vol(self) -> bool:
        """Build the normalised S/S0 volume from the DWI inputs."""
        if self.signal_vol is not None:
            return True
        if not self.ensure_mask():
            return False
        inputs = sorted(self.profile.inputs, key=lambda x: float(x[0]))
        if not inputs:
            self.load_errors.append("Profile has no DWI inputs")
            return False
        deltas = [float(d) for d, _ in inputs]
        shape = self.mask.shape
        n_fit = len(deltas)
        vol = np.zeros((*shape, n_fit * N_SHELLS), dtype=np.float32)
        mask = self.mask
        for di, (delta_ms, _path) in enumerate(inputs):
            data = self.ensure_raw(float(delta_ms))
            if data is None:
                return False
            s0 = data[..., 0].astype(np.float32)
            safe_s0 = np.where(s0 > 1e-10, s0, 1.0)
            for si, (_, vol_sl) in enumerate(SHELLS):
                shell_mean = data[..., vol_sl].mean(axis=-1).astype(np.float32)
                ratio = np.clip(shell_mean / safe_s0, 0.0, 1.0)
                ratio = np.where((s0 > 1e-10) & mask, ratio, 0.0)
                vol[..., di * N_SHELLS + si] = ratio
        self.signal_vol = vol
        self.fit_deltas = deltas
        return True

    # ------------- Helpers for viewer -------------
    def signal_at(self, vx: int, vy: int, sl: int, axis: int) -> Optional[np.ndarray]:
        if not self.ensure_signal_vol():
            return None
        if axis == 0:
            return self.signal_vol[sl, vx, vy, :]
        if axis == 1:
            return self.signal_vol[vx, sl, vy, :]
        return self.signal_vol[vx, vy, sl, :]

    def s0_per_delta_at(self, vx: int, vy: int, sl: int, axis: int) -> Optional[np.ndarray]:
        """Return the per-Δ b=0 values at a voxel, ordered by ``fit_deltas``.

        Used by the signal plot's alternative observed-display modes
        (avg-S0 and fit-S0). Returns ``None`` if any Δ scan is missing.
        """
        deltas = self.fit_deltas or [float(d) for d, _ in
                                      sorted(self.profile.inputs,
                                             key=lambda x: float(x[0]))]
        if not deltas:
            return None
        out = np.zeros(len(deltas), dtype=np.float32)
        for i, d in enumerate(deltas):
            data = self.ensure_raw(float(d))
            if data is None:
                return None
            if axis == 0:
                out[i] = float(data[sl, vx, vy, 0])
            elif axis == 1:
                out[i] = float(data[vx, sl, vy, 0])
            else:
                out[i] = float(data[vx, vy, sl, 0])
        return out

    def map_at(self, name: str, vx: int, vy: int, sl: int, axis: int) -> Optional[float]:
        m = self.ensure_map(name)
        if m is None:
            return None
        if axis == 0:
            return float(m[sl, vx, vy])
        if axis == 1:
            return float(m[vx, sl, vy])
        return float(m[vx, vy, sl])

    def raw_volume_slice(self, delta_ms: float, volume_idx: int,
                         axis: int, sl: int) -> Optional[np.ndarray]:
        data = self.ensure_raw(float(delta_ms))
        if data is None:
            return None
        volume_idx = int(np.clip(volume_idx, 0, data.shape[-1] - 1))
        sub = data[..., volume_idx]
        s = [slice(None)] * 3
        s[axis] = sl
        return sub[tuple(s)]

    def dims(self) -> Optional[tuple]:
        if self.ensure_mask():
            return self.mask.shape
        return None

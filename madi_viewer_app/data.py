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
    signal_vol_sig: str = ""                            # preprocessing fingerprint
    lib_bundle:   Optional[dict] = None                # from cache.get_library
    lib_sub:      Optional[np.ndarray] = None          # Δ-subset matrix
    lib_sub_sig:  Optional[str] = None                 # fingerprint for invalidation
    fit_deltas:   list = field(default_factory=list)
    load_errors:  list[str] = field(default_factory=list)

    # Pre-processing (mirrors scripts/fit_data.py).  Defaults match the
    # profile's stored fit configuration so "live" matching uses the same
    # signal transforms the saved maps did.
    rician_correct: bool = False
    noise_sigma:    Optional[float] = None   # None → auto-estimate
    avg_s0:         bool = False
    sigma_used:     Optional[float] = None   # populated after auto-estimate

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

    # ------------- Preprocessing helpers -------------
    def set_preprocessing(self, rician_correct: bool,
                          noise_sigma: Optional[float],
                          avg_s0: bool) -> bool:
        """Apply new preprocessing options. Returns True if anything changed.

        When the signature changes, the cached ``signal_vol`` is dropped so
        the next ``ensure_signal_vol`` rebuilds it with the new transforms.
        """
        new_sig = self._preproc_sig(rician_correct, noise_sigma, avg_s0)
        if new_sig == self.signal_vol_sig and self.signal_vol is not None:
            # same as last build; still keep flags in sync
            self.rician_correct = rician_correct
            self.noise_sigma = noise_sigma
            self.avg_s0 = avg_s0
            return False
        self.rician_correct = rician_correct
        self.noise_sigma = noise_sigma
        self.avg_s0 = avg_s0
        self.signal_vol = None
        self.signal_vol_sig = ""
        self.sigma_used = None
        return True

    @staticmethod
    def _preproc_sig(rician: bool, sigma: Optional[float], avg: bool) -> str:
        s = "auto" if sigma is None else f"{float(sigma):.3f}"
        return f"r={int(bool(rician))}|s={s}|a={int(bool(avg))}"

    def _estimate_sigma(self, data_4d: np.ndarray) -> Optional[float]:
        """Rayleigh-background sigma from the b=0 of the first Δ.

        Uses magnitude voxels OUTSIDE the brain mask, drops zeros
        (FOV padding). Mirrors scripts/fit_data.py::estimate_noise_sigma.
        """
        if self.mask is None:
            return None
        b0 = data_4d[..., 0]
        bg = ~self.mask
        vals = b0[bg]
        vals = vals[vals > 0]
        if vals.size < 100:
            return None
        return float(np.sqrt(np.mean(vals.astype(np.float64) ** 2) / 2.0))

    @staticmethod
    def _rician_debias(arr: np.ndarray, sigma: float) -> np.ndarray:
        """E[M^2] = A^2 + 2σ^2 → A = sqrt(max(M^2 - 2σ^2, 0))."""
        a2 = arr.astype(np.float64) ** 2 - 2.0 * sigma * sigma
        np.maximum(a2, 0.0, out=a2)
        return np.sqrt(a2)

    # ------------- Signal volume -------------
    def ensure_signal_vol(self) -> bool:
        """Build the normalised S/S0 volume from the DWI inputs.

        Respects ``rician_correct``, ``noise_sigma``, and ``avg_s0`` set
        via ``set_preprocessing`` so live matching sees exactly what the
        batch fitter would have produced for the same options.
        """
        target_sig = self._preproc_sig(
            self.rician_correct, self.noise_sigma, self.avg_s0)
        if self.signal_vol is not None and self.signal_vol_sig == target_sig:
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
        mask = self.mask

        # --- Sigma (once per rebuild) ---
        sigma: Optional[float] = None
        if self.rician_correct:
            if self.noise_sigma is not None:
                sigma = float(self.noise_sigma)
            else:
                first = self.ensure_raw(float(deltas[0]))
                if first is None:
                    return False
                sigma = self._estimate_sigma(first)
                if sigma is None:
                    self.load_errors.append(
                        "Rician: could not auto-estimate σ (too few "
                        "background voxels). Disabling correction.")
        self.sigma_used = sigma
        apply_rician = self.rician_correct and sigma is not None

        # --- Collect per-Δ S0 (with Rician correction if requested) ---
        s0_list: list[np.ndarray] = []
        for delta_ms, _ in inputs:
            data = self.ensure_raw(float(delta_ms))
            if data is None:
                return False
            s0_raw = data[..., 0]
            if apply_rician:
                s0 = self._rician_debias(s0_raw, sigma).astype(np.float32)
            else:
                s0 = s0_raw.astype(np.float32)
            s0_list.append(s0)

        if self.avg_s0:
            s0_common = np.mean(np.stack(s0_list, axis=0), axis=0)
            s0_used = [s0_common for _ in inputs]
        else:
            s0_used = s0_list

        # --- Build normalized volume ---
        vol = np.zeros((*shape, n_fit * N_SHELLS), dtype=np.float32)
        for di, (delta_ms, _path) in enumerate(inputs):
            data = self.ensure_raw(float(delta_ms))
            if data is None:
                return False
            s0 = s0_used[di]
            safe_s0 = np.where(s0 > 1e-10, s0, 1.0)
            for si, (_, vol_sl) in enumerate(SHELLS):
                sub = data[..., vol_sl]
                if apply_rician:
                    sub = self._rician_debias(sub, sigma)
                shell_mean = sub.mean(axis=-1).astype(np.float32)
                ratio = np.clip(shell_mean / safe_s0, 0.0, 1.0)
                ratio = np.where((s0 > 1e-10) & mask, ratio, 0.0)
                vol[..., di * N_SHELLS + si] = ratio

        self.signal_vol = vol
        self.signal_vol_sig = target_sig
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

    def raw_signal_at(self, vx: int, vy: int, sl: int,
                      axis: int) -> Optional[np.ndarray]:
        """Un-normalised per-shell means at a voxel, in scanner units.

        Same ordering as :meth:`signal_at` (n_fit × n_b flattened) but each
        entry is the plain mean of the shell's magnitude samples (Rician
        debiased if enabled). Intended for the fit-S0 live matcher so its
        residual sees the same raw vector the batch fitter consumed.
        """
        deltas = self.fit_deltas or [float(d) for d, _ in
                                      sorted(self.profile.inputs,
                                             key=lambda x: float(x[0]))]
        if not deltas:
            return None
        apply_rician = self.rician_correct and self.sigma_used is not None
        out = np.zeros(len(deltas) * N_SHELLS, dtype=np.float32)
        for di, d in enumerate(deltas):
            data = self.ensure_raw(float(d))
            if data is None:
                return None
            # Pull this voxel's time series (all volumes).
            if axis == 0:
                ts = data[sl, vx, vy, :]
            elif axis == 1:
                ts = data[vx, sl, vy, :]
            else:
                ts = data[vx, vy, sl, :]
            ts = np.asarray(ts, dtype=np.float64)
            if apply_rician:
                ts = np.sqrt(np.maximum(
                    ts * ts - 2.0 * self.sigma_used ** 2, 0.0))
            for si, (_, vol_sl) in enumerate(SHELLS):
                out[di * N_SHELLS + si] = float(ts[vol_sl].mean())
        return out

    def s0_per_delta_at(self, vx: int, vy: int, sl: int, axis: int) -> Optional[np.ndarray]:
        """Return the per-Δ b=0 values at a voxel, ordered by ``fit_deltas``.

        Honors the current preprocessing: applies Rician debias to the raw
        b=0 sample (if enabled) so the avg-S0 factor shown in the signal
        plot matches the S0 the matcher actually consumed.
        """
        deltas = self.fit_deltas or [float(d) for d, _ in
                                      sorted(self.profile.inputs,
                                             key=lambda x: float(x[0]))]
        if not deltas:
            return None
        apply_rician = self.rician_correct and self.sigma_used is not None
        out = np.zeros(len(deltas), dtype=np.float32)
        for i, d in enumerate(deltas):
            data = self.ensure_raw(float(d))
            if data is None:
                return None
            if axis == 0:
                s = float(data[sl, vx, vy, 0])
            elif axis == 1:
                s = float(data[vx, sl, vy, 0])
            else:
                s = float(data[vx, vy, sl, 0])
            if apply_rician:
                s = float(np.sqrt(max(s * s - 2.0 * self.sigma_used ** 2, 0.0)))
            out[i] = s
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

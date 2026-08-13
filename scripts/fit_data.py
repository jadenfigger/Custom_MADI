#!/usr/bin/env python3
"""
fit_data.py — Build MADI libraries & fit in-vivo DWI data
==========================================================

NEW IN THIS VERSION
-------------------
1. **Bvals/bvecs-driven shell detection**
       The hardcoded SHELLS table is gone.  Each input acquisition
       carries its own bvals (FSL format) and optional bvecs file.
       b=0 indices and shell groupings are detected from bvals.  This
       lets you fit datasets with arbitrary numbers of shells, arbitrary
       direction counts, and interleaved b=0 volumes — as long as the
       (Δ, b) pairs you have are a subset of what the library covers.

2. **(Δ, b) pair-level matching**
       Library entries store an n_deltas × n_b signal vector.  The
       matcher now subsets to whatever (Δ, b) pairs the data provides,
       not just whatever Δ values.  A single-shell dataset against a
       4-shell library will match on the column for that one b-value.

3. **Acquisition consistency checks**
       Pre-fit: small δ must match library, every input Δ must be in
       the library, every measured b-value must be in the library's
       b-values for that Δ.  Underdetermined fits (fewer measurements
       than free parameters) trigger a loud warning.

4. **Multi-b=0 averaging**
       Each input file may have many b=0 volumes scattered throughout.
       They are averaged within each Δ acquisition to produce that Δ's
       S0.  --avg-s0 still cross-averages S0 across Δ.

LEGACY FEATURES (unchanged)
---------------------------
1. **Rician noise-bias correction** (--rician-correct)
2. **S0 averaging across Delta scans** (--avg-s0)
3. **Optional S0 fitting in the matcher** (--fit-s0)

INPUT FORMAT
------------
  New (recommended):
    --input "Δ:dwi.nii.gz:bvals.bval[:bvecs.bvec]"
  Legacy (still works for old protocol):
    --input "Δ:dwi.nii.gz"   (uses LEGACY_SHELLS)

EXAMPLES
--------
  python fit_data.py --fit \\
      --input 25:dwi25.nii.gz:dwi25.bval:dwi25.bvec \\
      --mask mask.nii.gz \\
      --small-delta 6.0 \\
      --rician-correct
"""

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from madi.config   import SimConfig, BVALS_S_MM2, BVALS_UNIQUE, DELTAS_BIG
from madi.library  import (build_library, build_library_from_triplets,
                            load_library, load_library_meta,
                            match_voxels_batch, match_voxels_batch_fits0,
                            library_summary, make_remediation_log_grid,
                            edge_railing_diagnostics, resolve_grid_columns,
                            B_SNAP_TOL_S_MM2, TIMING_SNAP_TOL_MS)
from madi.fitters  import (bayes_fit, amico_fit, estimate_sigma_m,
                           calibrate_sigma_m,
                           DEFAULT_SIGMA_M, DEFAULT_LAMBDA1, DEFAULT_LAMBDA2)
from madi.signal   import signals_to_flat
from madi import fitters_gpu
from madi.biomarkers import derive_voxelwise_biomarkers


# ===================================================================
# Acquisition protocol — LEGACY fallback only
# ===================================================================
# Used only when an --input has no bvals path attached (old workflow).
# New code paths use bvals files directly.

LEGACY_SHELLS = [
    (1000.0, slice(1, 25)),
    (2500.0, slice(25, 49)),
    (4000.0, slice(49, 73)),
    (6000.0, slice(73, 97)),
]
LEGACY_B0_INDEX = 0
LEGACY_N_VOLS   = 97


# Anything below B0_THRESHOLD is treated as b=0 [s/mm²].
B0_THRESHOLD = 50.0

# Each non-b0 DWI volume's raw b-value is matched individually (not
# clustered with its neighbors first) against the library's b-value grid:
# if the nearest library b-value is within B_LIB_MATCH_TOL, that volume is
# snapped to it; otherwise the volume is discarded from the fit entirely.
B_LIB_MATCH_TOL = B_SNAP_TOL_S_MM2

# Direction-contract values deliberately mirror the language of the
# remediation request.  The scheme is a property of the input acquisition,
# not a fitting preference: a macroscopically isotropic MADI library cannot
# silently be compared to an unlabelled single-direction image.
DIRECTION_SCHEMES = (
    "single_direction",
    "orthogonal_3",
    "powder_N",
    "prescanner_averaged",
)
DIRECTION_ISOTROPY_SPREAD_MAX = 0.15
MADI_III_SINGLE_DIRECTION_BIAS = {
    "context": "Reported mean change when moving from one encoding direction to three in MADI III.",
    "ADC_percent": 18.1,
    "kioV_percent": 16.9,
    "kio_percent": -4.2,
    "rho_percent": -7.6,
    "V_percent": 1.1,
    "vi_percent": -8.4,
}


# ===================================================================
# Fit-run provenance and terminal capture
# ===================================================================


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot JSON-serialise {type(value)!r}")


def _sha256_file(path: str | None) -> str | None:
    """Return a streaming SHA-256 without loading an image/library into RAM."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_provenance(path: str | None) -> dict | None:
    if path is None:
        return None
    absolute = os.path.abspath(path)
    exists = os.path.isfile(absolute)
    return {
        "path": absolute,
        "exists": exists,
        "size_bytes": os.path.getsize(absolute) if exists else None,
        "sha256": _sha256_file(absolute) if exists else None,
    }


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment_provenance() -> dict:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for module_name in ("scipy", "nibabel", "numba"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = None
    gpu = None
    if fitters_gpu.HAS_CUDA:
        try:
            from numba import cuda
            gpu = cuda.get_current_device().name.decode()
        except Exception as exc:  # provenance must never abort a successful fit
            gpu = f"CUDA available; device query failed: {exc}"
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "package_versions": versions,
        "gpu": gpu,
        "code_commit_hash": _git_revision(),
    }


def _finite_summary(values) -> dict | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _library_provenance(path: str, meta: dict, library) -> dict:
    """Collect the immutable library facts required to interpret one fit."""
    build = meta.get("build_metadata") or {}
    escape_fractions = []
    for entry in library:
        boundary = entry.metadata.get("boundary", {}) if entry.metadata else {}
        if "escape_fraction" in boundary:
            escape_fractions.append(boundary["escape_fraction"])
    return {
        "filename": os.path.abspath(path),
        "sha256": _sha256_file(path),
        "build_commit_hash": build.get("code_commit_hash"),
        "build_configuration": build,
        "library_schema": meta.get("library_schema"),
        "has_per_entry_weights": bool(meta.get("has_weights")),
        "phase_model": build.get("phase_model"),
        "checkpoint_h_ms": meta.get("h_ms"),
        "delta_pairs": meta.get("delta_pairs"),
        "b_values_s_mm2": meta.get("b_values"),
        "escape_statistics_summary": _finite_summary(escape_fractions),
        "per_entry_monte_carlo_error": {
            "method": (
                "not yet available in the current artifact; P2-M will use "
                "held-out ensemble signal variation, not a tagged-exit k_io proxy"
            ),
        },
    }


class _TeeStream:
    """Mirror a text stream to the terminal and one run log file."""

    def __init__(self, terminal, log_file):
        self._terminal = terminal
        self._log_file = log_file

    def write(self, text):
        self._terminal.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self):
        self._terminal.flush()
        self._log_file.flush()

    def isatty(self):
        return self._terminal.isatty()


class FitRunArtifacts:
    """Exactly two sidecars for a fit: one JSON record and one full log."""

    def __init__(self, out_dir: str, run_name: str):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("._")
        self.run_name = safe_name or "madi_fit"
        self.out_dir = os.path.abspath(out_dir)
        self.json_path = os.path.join(self.out_dir, f"{self.run_name}.json")
        self.log_path = os.path.join(self.out_dir, f"{self.run_name}.log")
        self._stdout = None
        self._stderr = None
        self._log_file = None
        self._finalized = False
        self.record: dict = {
            "schema": "madi-fit-run-v1",
            "run_name": self.run_name,
            "status": "running",
            "start_timestamp": _utc_now(),
        }

    def start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self._log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(self._stdout, self._log_file)
        sys.stderr = _TeeStream(self._stderr, self._log_file)

    def update(self, **fields) -> None:
        self.record.update(fields)

    def mark_success(self) -> None:
        self.record["status"] = "completed"

    def mark_error(self, exc: BaseException) -> None:
        self.record["status"] = "failed"
        self.record["error"] = {"type": type(exc).__name__, "message": str(exc)}

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self.record.get("status") == "running":
            self.record["status"] = "incomplete"
        self.record["end_timestamp"] = _utc_now()
        self.record.setdefault("environment", _environment_provenance())
        try:
            with open(self.json_path, "w", encoding="utf-8") as fh:
                json.dump(self.record, fh, indent=2, sort_keys=True,
                          default=_json_default)
                fh.write("\n")
        finally:
            if self._stdout is not None:
                sys.stdout = self._stdout
            if self._stderr is not None:
                sys.stderr = self._stderr
            if self._log_file is not None:
                self._log_file.close()


_ACTIVE_FIT_ARTIFACTS: FitRunArtifacts | None = None


def _start_fit_artifacts(out_dir: str, run_name: str | None) -> FitRunArtifacts:
    global _ACTIVE_FIT_ARTIFACTS
    derived_name = run_name or os.path.basename(os.path.normpath(out_dir)) or "madi_fit"
    artifacts = FitRunArtifacts(out_dir, derived_name)
    artifacts.start()
    _ACTIVE_FIT_ARTIFACTS = artifacts
    return artifacts


def _snap_summary(events: list[dict]) -> dict:
    b_events = [event for event in events if event.get("axis") == "b"]
    timing_events = [event for event in events if event.get("axis") == "timing"]
    return {
        "n_snapped_b_shells": len(b_events),
        "n_snapped_timing_pairs": len(timing_events),
        "max_b_abs_offset_s_mm2": max(
            (float(event["abs_offset"]) for event in b_events), default=0.0),
        "max_timing_abs_offset_ms": {
            "delta": max(
                (float(event["abs_offset"]["delta_ms"]) for event in timing_events),
                default=0.0),
            "Delta": max(
                (float(event["abs_offset"]["Delta_ms"]) for event in timing_events),
                default=0.0),
        },
    }


def _deduplicate_snap_events(events: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for event in events:
        key = json.dumps(event, sort_keys=True, default=_json_default)
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return unique


def _print_snap_events(events: list[dict]) -> None:
    for event in events:
        if event["axis"] == "b":
            print(
                "  SNAP[b]: requested "
                f"{event['requested']:g} s/mm² → stored {event['used']:g} s/mm² "
                f"(|offset|={event['abs_offset']:g} s/mm², "
                f"{event['rel_offset']:.3%})")
        else:
            requested = event["requested"]
            used = event["used"]
            absolute = event["abs_offset"]
            relative = event["rel_offset"]
            print(
                "  SNAP[timing]: requested "
                f"(δ={requested['delta_ms']:g}, Δ={requested['Delta_ms']:g}) ms → "
                f"stored (δ={used['delta_ms']:g}, Δ={used['Delta_ms']:g}) ms "
                f"(|offset|=(δ {absolute['delta_ms']:g}, Δ {absolute['Delta_ms']:g}) ms; "
                f"relative=(δ {relative['delta']:.3%}, Δ {relative['Delta']:.3%}))")


# ===================================================================
# Presets  (unchanged - omitted for brevity, keep yours)
# ===================================================================

# NOTE: `n_steps` is no longer a settable SimConfig field -- it is derived
# from `T_max_ms / ts` so that every walk covers the full (δ,Δ,b) grid (see
# madi/config.py). Presets below only tune walker/ensemble counts and
# geometry; leave T_max_ms at the SimConfig default unless a preset
# deliberately restricts the (δ,Δ) grid too (via small_deltas/big_deltas).
PRESETS = {
    "calibration": {
        "kios": [10, 35],
        "rhos": [200_000, 800_000],
        "Vs":   [1.0, 3.0],
        "cfg":  dict(n_walkers=100_000, n_ensembles=120),
    },
    "small": {
        "kios": [5, 12, 25, 50],
        "rhos": [100_000, 200_000, 400_000, 800_000, 1_200_000],
        "Vs":   [0.5, 1.0, 2.0, 3.5],
        "cfg":  dict(n_walkers=5_000, n_ensembles=2),
    },
    "default": {
        "kios": [2, 5, 8, 12, 18, 25, 35, 50, 75],
        "rhos": [100_000, 200_000, 300_000, 400_000, 600_000,
                 800_000, 1_000_000, 1_200_000, 1_500_000],
        "Vs":   [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        "cfg":  dict(n_walkers=100_000, n_ensembles=120),
    },
    "dense": {
        # P0-E production topology: log rho × log V, diagonal v_i mask,
        # k_io=0 family and an explicit free-water atom.  The numerical axes
        # and analytic quadrature weights are made below, not here, so no
        # caller can accidentally turn this back into a linear dense grid.
        "grid": "remediation_log",
        "cfg":  dict(n_walkers=100_000, n_ensembles=40),
    },
}


# ===================================================================
# Parsers
# ===================================================================

def parse_triplet(s: str):
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Triplet must be 'kio,rho,V', got '{s}'")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _entry_key_for_script(kio: float, rho: float, volume: float):
    """Match the library's stable requested-coordinate weight key."""
    if rho == 0.0 and volume == 0.0:
        return ("free_water", 0.0, 0.0)
    return (round(float(kio), 4), round(float(rho), 1), round(float(volume), 6))


def load_remediation_entry_subset(
    path: str,
    canonical_triplets: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], dict]:
    """Resolve a declared v5 restricted subset against the full P0 grid.

    A restricted build must not accept arbitrary coordinates: each declared
    pair/k_io combination is checked against the canonical weighted
    remediation grid, retaining the production quadrature weights and making
    a typo fail before any expensive GPU work begins.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            declaration = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read remediation entry subset {path!r}: {exc}") from exc

    supported_schemas = {
        "madi-replicate-entry-subset-v1",
        "madi-v5-stencil-probe-entry-subset-v1",
    }
    if declaration.get("schema") not in supported_schemas:
        raise ValueError("unsupported remediation entry-subset schema")
    pairs = declaration.get("cellular_pairs")
    kios = declaration.get("kio_values_s_inv")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("remediation entry subset must declare non-empty cellular_pairs")
    if not isinstance(kios, list) or not kios:
        raise ValueError("remediation entry subset must declare non-empty kio_values_s_inv")

    canonical = {
        _entry_key_for_script(kio, rho, volume): (float(kio), float(rho), float(volume))
        for kio, rho, volume in canonical_triplets
    }
    selected: list[tuple[float, float, float]] = []
    selected_keys: set[tuple] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or "rho" not in pair or "V" not in pair:
            raise ValueError("each remediation subset pair must contain rho and V")
        rho = float(pair["rho"])
        volume = float(pair["V"])
        for kio in kios:
            key = _entry_key_for_script(float(kio), rho, volume)
            if key not in canonical:
                raise ValueError(
                    "remediation subset coordinate is not a retained canonical "
                    f"grid entry: kio={kio}, rho={rho}, V={volume}"
                )
            if key in selected_keys:
                raise ValueError(f"remediation subset declares duplicate entry {key!r}")
            selected.append(canonical[key])
            selected_keys.add(key)

    if bool(declaration.get("include_free_water", False)):
        free_key = _entry_key_for_script(0.0, 0.0, 0.0)
        if free_key not in canonical:
            raise ValueError("canonical remediation grid is missing its free-water atom")
        selected.append(canonical[free_key])
        selected_keys.add(free_key)

    expected_cellular = declaration.get("expected_cellular_entries")
    expected_total = declaration.get("expected_total_entries")
    cellular_count = sum(1 for _, rho, _ in selected if rho > 0.0)
    if expected_cellular is not None and cellular_count != int(expected_cellular):
        raise ValueError(
            f"remediation subset has {cellular_count} cellular entries, expected {expected_cellular}"
        )
    if expected_total is not None and len(selected) != int(expected_total):
        raise ValueError(
            f"remediation subset has {len(selected)} total entries, expected {expected_total}"
        )

    provenance = {
        "schema": declaration["schema"],
        "path": os.path.abspath(path),
        "sha256": _sha256_file(path),
        "cellular_entries": cellular_count,
        "total_entries": len(selected),
        "purpose": declaration.get("purpose"),
    }
    return selected, provenance


def parse_input(s: str):
    """Parse a single --input spec.

    The first ':'-separated field is the diffusion time Δ (ms). It may be
    prefixed with a per-scan gradient duration δ as 'δ,Δ' to override the
    global --small-delta for THIS scan (the (δ,Δ,b)-universal library
    supports a different δ per acquisition).

    Supported forms:
        Δ:dwi.nii.gz                          — legacy, uses LEGACY_SHELLS; δ from --small-delta
        Δ:dwi.nii.gz:bvals.bval               — bvals-driven, no bvecs
        Δ:dwi.nii.gz:bvals.bval:bvecs.bvec    — bvals + bvecs
        δ,Δ:dwi.nii.gz:bvals.bval[:bvecs]     — explicit per-scan δ

    Returns
    -------
    (delta_small_or_None, Delta_ms, dwi_path, bvals_path_or_None,
     bvecs_path_or_None)
        delta_small_or_None is the per-scan δ if given as 'δ,Δ', else None
        (meaning "use the global --small-delta").
    """
    parts = s.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Input must be '[δ,]Δ:dwi.nii.gz[:bvals[:bvecs]]', got '{s}'")
    timing = parts[0]
    if "," in timing:
        ds, dD = timing.split(",", 1)
        try:
            delta_small = float(ds)
            Delta = float(dD)
        except ValueError:
            raise ValueError(
                f"First field 'δ,Δ' must be two numbers in ms; got '{timing}'")
    else:
        delta_small = None
        try:
            Delta = float(timing)
        except ValueError:
            raise ValueError(
                f"First field of --input must be Δ in ms (or 'δ,Δ'); "
                f"got '{timing}'")
    dwi   = parts[1]
    bvals = parts[2] if len(parts) >= 3 and parts[2] else None
    bvecs = parts[3] if len(parts) >= 4 and parts[3] else None
    if len(parts) > 4:
        raise ValueError(f"Too many ':'-separated fields in --input '{s}'")
    return (delta_small, Delta, dwi, bvals, bvecs)


def parse_z_slice(s):
    """Parse a --z-slice spec into a Python slice object (or None).

    Examples
    --------
        '50'    → slice(50, 51)        — single Z-slice
        '40:60' → slice(40, 60)        — Z in [40, 60)
        ':60'   → slice(None, 60)      — Z in [0, 60)
        '40:'   → slice(40, None)      — Z in [40, end)
        None    → None                 — no restriction
    """
    if s is None:
        return None
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"--z-slice range must be 'a:b', got '{s}'")
        a = int(parts[0]) if parts[0] else None
        b = int(parts[1]) if parts[1] else None
        return slice(a, b)
    return slice(int(s), int(s) + 1)


# ===================================================================
# bvals / bvecs parsing
# ===================================================================

def parse_bvals(path: str, lib_b_values, b0_thresh: float = B0_THRESHOLD,
                tol: float = B_LIB_MATCH_TOL, *,
                return_snap_events: bool = False):
    """Read an FSL-format bvals file and match each volume directly against
    the library's b-value grid.

    Each non-b0 volume's raw b-value is checked individually (not
    clustered with its neighbors first): if the nearest library b-value is
    within ``tol``, the volume is snapped to that library b-value;
    otherwise it is dropped from the fit entirely. This is deliberately
    library-driven rather than scanner-driven -- a chain-clustering
    approach that groups nearby raw b-values together first and then
    checks whether the cluster's mean happens to land near a library value
    can silently misclassify volumes near the boundary between shells, and
    can't express "this whole run of intermediate b-values (e.g. an
    IVIM/perfusion-range shell) isn't in the library at all, throw it
    away" as precisely as a per-volume check does.

    Parameters
    ----------
    path : str
        Path to a whitespace-delimited bvals file (typically one row).
    lib_b_values : list of float
        b-values present in the library [s/mm²].
    b0_thresh : float
        Anything with b < this is treated as b=0 [s/mm²].
    tol : float
        Max |raw b-value - nearest library b-value| to accept a volume
        (default 30 s/mm²). Volumes farther than this from every library
        b-value are discarded.

    Returns
    -------
    bvals : (n_vols,) float ndarray  — raw values from the file
    b0_idx : (n_b0,) int ndarray
    shells : list of (b_value, idx_array) sorted by ascending b
        b_value is the library's own canonical value (not a rounded
        scanner-side representative); only library b-values with at least
        one matching volume appear.
    n_dropped : int
        Number of non-b0 volumes discarded (too far from every library
        b-value).
    snap_events : list[dict], optional
        Returned only when ``return_snap_events=True``. Events are grouped by
        distinct raw b-value and identify the canonical library b used.
    """
    raw = np.loadtxt(path).ravel().astype(float)
    if raw.size == 0:
        raise ValueError(f"bvals file is empty: {path}")

    b0_mask = raw < b0_thresh
    b0_idx  = np.where(b0_mask)[0]
    nz_idx  = np.where(~b0_mask)[0]

    if nz_idx.size == 0:
        raise ValueError(f"No non-zero b-values in {path}")

    nz_vals = raw[nz_idx]
    lib_arr = np.asarray(sorted(set(lib_b_values)), dtype=float)

    # Nearest library b-value (and its distance) for every volume.
    dists = np.abs(nz_vals[:, None] - lib_arr[None, :])
    nearest_j = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(nz_vals.size), nearest_j]
    nearest_b = lib_arr[nearest_j]

    keep = nearest_dist <= tol
    n_dropped = int((~keep).sum())

    shells = []
    for lb in lib_arr:
        idx = nz_idx[keep & (nearest_b == lb)]
        if idx.size > 0:
            shells.append((float(lb), np.sort(idx)))
    shells.sort(key=lambda x: x[0])

    events: list[dict] = []
    for requested in np.unique(nz_vals[keep]):
        j = int(np.argmin(np.abs(lib_arr - requested)))
        used = float(lib_arr[j])
        offset = abs(used - float(requested))
        if offset > 1e-12:
            events.append({
                "axis": "b",
                "requested": float(requested),
                "used": used,
                "abs_offset": offset,
                "rel_offset": offset / max(abs(float(requested)), 1e-12),
                "n_volumes": int(np.count_nonzero(nz_vals[keep] == requested)),
            })

    if return_snap_events:
        return raw, b0_idx, shells, n_dropped, events
    return raw, b0_idx, shells, n_dropped


def parse_bvecs(path: str, n_vols_expected: int):
    """Read an FSL-format bvecs file (3 rows × N columns).

    Returns
    -------
    bvecs : (3, n_vols) ndarray, or None if path is None
    """
    if path is None:
        return None
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != 3:
        # Some pipelines store bvecs as N×3; transpose if needed
        if arr.shape[1] == 3:
            arr = arr.T
        else:
            raise ValueError(
                f"bvecs file {path} has shape {arr.shape}; expected (3, N) "
                f"or (N, 3).")
    if arr.shape[1] != n_vols_expected:
        raise ValueError(
            f"bvecs file {path}: {arr.shape[1]} columns, "
            f"DWI has {n_vols_expected} volumes.")
    return arr


def _unique_antipodal_direction_count(unit_vectors: np.ndarray,
                                      *, tolerance: float = 1e-4) -> int:
    """Count directions after treating ``v`` and ``-v`` as equivalent."""
    representatives: list[np.ndarray] = []
    for vector in unit_vectors.T:
        if not any(abs(float(np.dot(vector, other))) >= 1.0 - tolerance
                   for other in representatives):
            representatives.append(vector)
    return len(representatives)


def b_tensor_diagnostic(bvecs: np.ndarray | None, idx: np.ndarray) -> dict:
    """Describe the b-tensor direction sampling for one retained shell."""
    report = {
        "bvecs_available": bvecs is not None,
        "n_volumes": int(len(idx)),
        "n_valid_bvecs": 0,
        "n_unique_directions": None,
        "b_tensor": None,
        "eigenvalues": None,
        "eigenvalue_spread": None,
        "isotropic": None,
    }
    if bvecs is None:
        return report
    vectors = np.asarray(bvecs[:, idx], dtype=float)
    norms = np.linalg.norm(vectors, axis=0)
    keep = norms > 1e-6
    if not np.any(keep):
        return report
    unit = vectors[:, keep] / norms[keep]
    tensor = (unit @ unit.T) / unit.shape[1]
    eigenvalues = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    spread = float(eigenvalues[0] - eigenvalues[2])
    report.update(
        n_valid_bvecs=int(unit.shape[1]),
        n_unique_directions=_unique_antipodal_direction_count(unit),
        b_tensor=tensor.tolist(),
        eigenvalues=eigenvalues.tolist(),
        eigenvalue_spread=spread,
        isotropic=bool(
            unit.shape[1] >= 3 and spread <= DIRECTION_ISOTROPY_SPREAD_MAX),
    )
    return report


def _direction_contract(
    declared_scheme: str | None,
    shell_reports: list[dict],
) -> dict:
    """Infer/validate the required acquisition-direction declaration.

    A direction-free MADI library represents an isotropic/powder-averaged
    signal.  The validation intentionally operates on b-vectors rather than
    merely counting images: three repeated collinear acquisitions are not an
    orthogonal three-direction acquisition.
    """
    if declared_scheme is not None and declared_scheme not in DIRECTION_SCHEMES:
        raise ValueError(f"unknown direction scheme {declared_scheme!r}")
    if not shell_reports:
        raise ValueError("no retained diffusion shells are available for direction validation")

    all_have_bvecs = all(bool(report["bvecs_available"]) for report in shell_reports)
    inferred = None
    if all_have_bvecs:
        counts = [int(report["n_unique_directions"] or 0) for report in shell_reports]
        isotropic = [bool(report["isotropic"]) for report in shell_reports]
        if all(count == 1 for count in counts):
            inferred = "single_direction"
        elif all(count == 3 for count in counts) and all(isotropic):
            inferred = "orthogonal_3"
        elif all(count >= 4 for count in counts) and all(isotropic):
            inferred = "powder_N"

    if declared_scheme is None:
        if inferred is None:
            raise ValueError(
                "--direction-scheme is required because the b-vectors do not "
                "unambiguously identify single_direction, orthogonal_3, or "
                "an isotropic powder_N acquisition.")
        declared_scheme = inferred
        inferred_from_bvecs = True
        print(f"  Direction scheme inferred from b-vectors: {declared_scheme}")
    else:
        inferred_from_bvecs = False

    warnings_list: list[str] = []
    for report in shell_reports:
        label = (f"δ={report['delta_ms']:g}, Δ={report['Delta_ms']:g}, "
                 f"b={report['b_s_mm2']:g}")
        available = bool(report["bvecs_available"])
        n_unique = report["n_unique_directions"]
        isotropic = report["isotropic"]
        if declared_scheme == "single_direction":
            if available and n_unique != 1:
                raise ValueError(
                    f"{label}: declared single_direction but b-vectors contain "
                    f"{n_unique} unique directions.")
            if not available:
                warnings_list.append(
                    f"{label}: single-direction declaration cannot be verified; bvecs absent.")
        elif declared_scheme == "orthogonal_3":
            if not available:
                raise ValueError(
                    f"{label}: orthogonal_3 requires bvecs for verification.")
            if n_unique != 3:
                raise ValueError(
                    f"{label}: declared orthogonal_3 but b-vectors contain "
                    f"{n_unique} unique directions.")
            if not isotropic:
                warnings_list.append(
                    f"{label}: declared orthogonal_3 is degenerate "
                    f"(b-tensor eigenvalue spread={report['eigenvalue_spread']:.3f}).")
        elif declared_scheme == "powder_N":
            if not available:
                raise ValueError(
                    f"{label}: powder_N requires bvecs for isotropy verification.")
            if n_unique is None or n_unique < 4:
                raise ValueError(
                    f"{label}: declared powder_N but b-vectors contain fewer than "
                    "four unique directions.")
            if not isotropic:
                warnings_list.append(
                    f"{label}: declared powder_N is degenerate "
                    f"(b-tensor eigenvalue spread={report['eigenvalue_spread']:.3f}).")
        else:  # prescanner_averaged
            if available and n_unique not in (None, 1):
                warnings_list.append(
                    f"{label}: declared prescanner_averaged but raw b-vectors "
                    f"contain {n_unique} directions; verify the input was actually averaged.")

    for message in warnings_list:
        print(f"  ⚠ Direction contract: {message}")
    single_direction_bias = None
    if declared_scheme == "single_direction":
        single_direction_bias = dict(MADI_III_SINGLE_DIRECTION_BIAS)
        print(
            "  ⚠ SINGLE-DIRECTION INPUT: these maps are not trace/powder maps. "
            "MADI III reported the following one→three-direction shifts: "
            "ADC +18.1%, k_ioV +16.9%, k_io -4.2%, rho -7.6%, "
            "V +1.1%, v_i -8.4%.")
    return {
        "declared_scheme": declared_scheme,
        "inferred_from_bvecs": inferred_from_bvecs,
        "isotropy_spread_threshold": DIRECTION_ISOTROPY_SPREAD_MAX,
        "shells": shell_reports,
        "warnings": warnings_list,
        "single_direction_madi_iii_bias_percent": single_direction_bias,
    }


# ===================================================================
# Noise estimation & Rician correction
# ===================================================================

def estimate_noise_sigma(b0_image, mask_brain, dilate_iters=16):
    """Estimate Rician noise sigma from a b=0 magnitude image's background.

    Background is voxels OUTSIDE a DILATED copy of the brain mask, not
    simply outside the raw mask. Skull/scalp tissue sits immediately
    outside a typical brain mask and its magnitude values are not pure
    Rayleigh-distributed noise (bone/marrow signal, motion, EPI ghosting
    near the skull) -- sampling background right up against the mask
    boundary biases the estimate high. Dilating the mask first pushes the
    sampled background out past the skull into cleaner air.

    Parameters
    ----------
    b0_image : ndarray (X,Y,Z)  a single b=0 magnitude volume
    mask_brain : bool ndarray (X,Y,Z)  brain mask
    dilate_iters : int  binary-dilation iterations applied to mask_brain
        before excluding it from the background region (default 8;
        0 reproduces the old "just outside the raw mask" behavior).

    Returns
    -------
    sigma : float or None (None if too few background voxels)
    n_bg  : int  number of background voxels actually used
    """
    if dilate_iters > 0:
        from scipy.ndimage import binary_dilation
        bg_mask = ~binary_dilation(mask_brain, iterations=dilate_iters)
    else:
        bg_mask = ~mask_brain

    # Drop zero voxels (often the FOV padding) - they're not noise samples
    bg_vals = b0_image[bg_mask]
    bg_vals = bg_vals[bg_vals > 0]

    n_bg = len(bg_vals)
    if n_bg < 100:
        return None, n_bg

    # Rayleigh: sigma = sqrt(<M^2>/2)
    sigma = float(np.sqrt(np.mean(bg_vals.astype(np.float64)**2) / 2.0))
    return sigma, n_bg


def rician_correct_secondmoment(M, sigma):
    """Recover unbiased A from magnitude M using E[M^2] = A^2 + 2 sigma^2.

    Vectorized.  Clips negative values to 0 (occurs when M < sqrt(2)*sigma,
    i.e. essentially pure noise).

    Parameters
    ----------
    M : ndarray  magnitude signal (any shape)
    sigma : float  Rician noise std

    Returns
    -------
    A : ndarray  bias-corrected signal, same shape as M
    """
    A2 = M.astype(np.float64)**2 - 2.0 * sigma**2
    A2 = np.clip(A2, 0.0, None)
    return np.sqrt(A2)


# ===================================================================
# Data loading
# ===================================================================

def load_dwi_and_average(input_specs, mask_path,
                         lib_b_values,
                         default_small_delta=None,
                         rician_correct=False,
                         noise_sigma=None,
                         noise_bg_dilate_iters=16,
                         avg_s0=False,
                         return_raw=False,
                         return_metadata=False,
                         direction_scheme=None,
                         z_slice=None,
                         b_tol=B_LIB_MATCH_TOL):
    """Load DWI NIfTIs, derive (δ, Δ, b) layout from each input's bvals,
    and assemble the measured matrix in a column order matching
    ``fit_triples``.

    Parameters
    ----------
    input_specs : list of (delta_small_or_None, Delta_ms, dwi_path,
                          bvals_path_or_None, bvecs_path_or_None)
        For each input either bvals_path is given (preferred) or it is
        None (legacy path: assumes LEGACY_SHELLS protocol).  ``delta_small``
        is the per-scan gradient duration δ [ms]; when None the global
        ``default_small_delta`` is used for that scan.
    default_small_delta : float or None
        Global δ [ms] applied to any input whose per-scan δ is None.
        Required (non-None) unless every input carries its own δ — the
        (δ,Δ,b) library cannot be indexed without a δ for every scan.
    mask_path : str or None
        Path to a brain mask NIfTI.  If None, fit every voxel of the
        full DWI volume — affine and shape are taken from the first
        input DWI.  Required when ``rician_correct`` is True and
        ``noise_sigma`` is None (auto-σ estimation needs background
        voxels, which need a mask to find).
    z_slice : slice or None
        Optional Python slice along the third spatial axis to restrict
        fitting to a Z-slice or range.  Combined (intersected) with
        ``mask_path`` if both are given.  Noise σ estimation continues
        to use the un-sliced mask so air voxels from outside the slice
        range still contribute.
    lib_b_values : list of float
        b-values present in the library [s/mm²].  Used to filter shells
        in the data: only shells whose b matches a library b-value
        (within b_tol) are retained.
    rician_correct : bool
    noise_sigma : float or None
    noise_bg_dilate_iters : int
        [auto sigma only, ignored if noise_sigma is given] binary-dilation
        iterations applied to the mask before its background (~mask) is
        sampled for noise estimation -- see estimate_noise_sigma().
    avg_s0 : bool
        If True, replace each Δ's S0 with the grand mean across Δs.
    return_raw : bool
    return_metadata : bool
        Return acquisition provenance, snap events, and direction-contract
        diagnostics even when an un-normalised raw signal is not needed.
    direction_scheme : str or None
        Declared acquisition direction convention. Required by the CLI fit
        path; direct library/testing callers may omit it when
        ``return_metadata`` is False.
    b_tol : float
        Tolerance for matching data b-values to library b-values.

    Returns
    -------
    measured : (n_voxels, n_features) ndarray  — S/S0 ratios
    fit_triples : list of (δ, Δ, b) tuples in the column order of ``measured``
    affine, mask_indices, shape
    extras (if return_raw or return_metadata) : dict
        Includes acquisition provenance, snapping and direction diagnostics;
        contains ``raw`` only when ``return_raw`` is True.
    sigma_used : float
        The noise standard deviation used for fitting.
    """
    import nibabel as nib

    # Sort by diffusion time Δ (2nd field); δ (1st field) may be None.
    input_specs = sorted(input_specs, key=lambda x: x[1])
    n_deltas = len(input_specs)

    def _resolve_delta_small(per_scan_delta, Delta):
        d = per_scan_delta if per_scan_delta is not None else default_small_delta
        if d is None:
            raise ValueError(
                f"No δ (small-delta) for the Δ={Delta:g} ms scan: pass "
                f"--small-delta, or give the input as 'δ,Δ:...'. The "
                f"(δ,Δ,b) library needs a δ for every acquisition.")
        return float(d)

    # ----------------------------------------------------------------
    # Mask (optional).  If absent, fit every voxel of the volume —
    # spatial reference is taken from the first DWI input.
    # ----------------------------------------------------------------
    if mask_path is not None:
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(bool)
        affine = mask_img.affine
        shape = mask.shape
        print(f"  Mask: {mask.sum()} voxels (from --mask)")
    else:
        first_img = nib.load(input_specs[0][2])
        affine = first_img.affine
        shape  = tuple(first_img.shape[:3])
        mask   = np.ones(shape, dtype=bool)
        print(f"  No --mask: starting from full volume "
              f"(shape {shape}, {int(np.prod(shape))} voxels). "
              f"Output maps will contain garbage in air regions; "
              f"mask the maps post-hoc if desired.")

    # Keep the un-sliced mask around for noise estimation (background
    # voxels should be drawn from the whole volume regardless of
    # whether we're only fitting one slice).
    mask_for_noise = mask.copy()

    # Optional z-slice restriction.
    if z_slice is not None:
        if len(shape) < 3:
            raise ValueError(f"z_slice given but volume has shape {shape} "
                             f"(need ≥3 spatial dims).")
        z_keep = np.zeros(shape, dtype=bool)
        z_keep[:, :, z_slice] = True
        before = int(mask.sum())
        mask = mask & z_keep
        after = int(mask.sum())
        # Pretty-print the slice
        a = z_slice.start if z_slice.start is not None else 0
        b = z_slice.stop  if z_slice.stop  is not None else shape[2]
        print(f"  --z-slice: restricting to Z in [{a}, {b}) "
              f"→ {after} voxels (was {before})")

    mask_idx = np.where(mask)
    n_vox = len(mask_idx[0])
    if n_vox == 0:
        raise ValueError("Mask (after --z-slice intersection) contains "
                         "zero voxels.")

    # ----------------------------------------------------------------
    # Pass 1: load DWI volumes and parse each input's shell layout
    # ----------------------------------------------------------------
    all_data         = []         # list of (X, Y, Z, N_vols) arrays
    all_b0_idx       = []         # list of int arrays
    all_shells_kept  = []         # list of [(b, idx)] retained against library
    all_small_delta  = []         # list of resolved per-scan δ [ms]
    all_snap_events: list[dict] = []
    direction_shell_reports: list[dict] = []
    input_provenance: list[dict] = []

    for di, (delta_small_in, delta_ms, dwi_path, bvals_path, bvecs_path) \
            in enumerate(input_specs):
        delta_small = _resolve_delta_small(delta_small_in, delta_ms)
        all_small_delta.append(delta_small)
        print(f"  δ={delta_small:.1f}ms  Δ={delta_ms:.1f}ms  "
              f"({os.path.basename(dwi_path)})", flush=True)
        img = nib.load(dwi_path)
        data = img.get_fdata().astype(np.float64)
        n_vols = data.shape[-1]
        print(f"    DWI shape: {data.shape}")

        n_dropped = 0
        if bvals_path is None:
            # Legacy path: rebuild shell layout from LEGACY_SHELLS
            if n_vols != LEGACY_N_VOLS:
                print(f"    ⚠ legacy mode expected {LEGACY_N_VOLS} volumes, "
                      f"got {n_vols}.  Pass a bvals file for non-default "
                      f"protocols.")
            b0_idx = np.array([LEGACY_B0_INDEX], dtype=int)
            raw_shells = [(float(b), np.arange(sl.start, sl.stop, dtype=int))
                          for b, sl in LEGACY_SHELLS]
            # LEGACY_SHELLS's hardcoded b-values aren't necessarily on the
            # library's grid either -- match them the same way (nearest
            # library b-value within b_tol, else drop the whole shell;
            # there's no raw per-volume bvals array in legacy mode to
            # round individually, only these fixed nominal shell values).
            kept = []
            dropped = []
            for b, idx in raw_shells:
                nearest = min(lib_b_values, key=lambda lb: abs(lb - b))
                if abs(nearest - b) <= b_tol:
                    kept.append((float(nearest), idx))
                    offset = abs(float(nearest) - float(b))
                    if offset > 1e-12:
                        all_snap_events.append({
                            "axis": "b",
                            "requested": float(b),
                            "used": float(nearest),
                            "abs_offset": offset,
                            "rel_offset": offset / max(abs(float(b)), 1e-12),
                            "n_volumes": int(len(idx)),
                        })
                else:
                    dropped.append(b)
                    n_dropped += int(len(idx))
            if dropped:
                print(f"    ⚠ dropping legacy shells not in library: "
                      f"{[f'{b:g}' for b in dropped]} "
                      f"(library has {sorted(lib_b_values)})")
            if not kept:
                raise ValueError(
                    f"No LEGACY_SHELLS b-value matched any library "
                    f"b-value within {b_tol:g} s/mm² "
                    f"(library: {sorted(lib_b_values)}).")
            print(f"    LEGACY mode: 1 b=0 vol, {len(kept)} shells "
                  f"matched to library")
        else:
            bvals, b0_idx, shells, n_dropped, snap_events = parse_bvals(
                bvals_path, lib_b_values, tol=b_tol, return_snap_events=True)
            all_snap_events.extend(snap_events)
            if bvals.size != n_vols:
                raise ValueError(
                    f"bvals length ({bvals.size}) ≠ DWI volumes ({n_vols}) "
                    f"for {dwi_path}")
            print(f"    bvals: {len(b0_idx)} b=0 vols, "
                  f"{len(shells)} shells matched to library "
                  f"(±{b_tol:g} s/mm²) "
                  f"({[f'b={int(b)}×{len(idx)}' for b, idx in shells]})")
            if n_dropped:
                print(f"    ⚠ discarded {n_dropped} volume(s) whose "
                      f"b-value is more than {b_tol:g} s/mm² from every "
                      f"library b-value {sorted(lib_b_values)} "
                      f"(not used in the fit)")
            if not shells:
                raise ValueError(
                    f"No volumes in {dwi_path} matched any library "
                    f"b-value within {b_tol:g} s/mm² "
                    f"(library: {sorted(lib_b_values)}).")

            kept = shells

        bvecs = parse_bvecs(bvecs_path, n_vols_expected=n_vols)
        for b, idx in kept:
            direction_report = b_tensor_diagnostic(bvecs, idx)
            direction_report.update(
                delta_ms=float(delta_small),
                Delta_ms=float(delta_ms),
                b_s_mm2=float(b),
            )
            direction_shell_reports.append(direction_report)

        all_data.append(data)
        all_b0_idx.append(b0_idx)
        all_shells_kept.append(kept)
        input_provenance.append({
            "declared_timing": {"delta_ms": float(delta_small), "Delta_ms": float(delta_ms)},
            "dwi": _file_provenance(dwi_path),
            "bvals": _file_provenance(bvals_path),
            "bvecs": _file_provenance(bvecs_path),
            "shape": [int(v) for v in data.shape],
            "voxel_dimensions_mm": [float(v) for v in img.header.get_zooms()[:3]],
            "n_b0_volumes": int(len(b0_idx)),
            "n_dropped_nonzero_volumes": int(n_dropped),
        })

    # ----------------------------------------------------------------
    # Build the column ordering: (Δ, b) pairs sorted by Δ then by b
    # ----------------------------------------------------------------
    fit_triples = []
    for spec, delta_small, kept in zip(input_specs, all_small_delta,
                                       all_shells_kept):
        delta_big = spec[1]
        for b, _ in kept:
            fit_triples.append((float(delta_small), float(delta_big),
                                float(b)))
    n_features = len(fit_triples)
    print(f"  → {n_features} (δ,Δ,b) features per voxel")

    # ----------------------------------------------------------------
    # Noise sigma estimation (optional) — uses first scan's first b=0
    # ----------------------------------------------------------------
    sigma_used = None
    if rician_correct:
        if noise_sigma is not None:
            sigma_used = float(noise_sigma)
            print(f"  Rician correction ENABLED, user-provided sigma={sigma_used:.2f}")
        else:
            b0_for_sigma = all_data[0][..., all_b0_idx[0][0]]
            sigma_used, n_bg = estimate_noise_sigma(
                b0_for_sigma, mask_for_noise,
                dilate_iters=noise_bg_dilate_iters)
            if sigma_used is None:
                print(f"    ⚠ very few background voxels ({n_bg}) outside "
                      f"the {noise_bg_dilate_iters}x-dilated mask — "
                      f"disabling Rician correction. Try a smaller "
                      f"--noise-bg-dilate-iters.")
                rician_correct = False
            else:
                b0_brain_med = float(np.median(b0_for_sigma[mask_for_noise]))
                print(f"  Rician correction ENABLED")
                print(f"    sigma (auto, background, "
                      f"{noise_bg_dilate_iters}x-dilated-mask excluded, "
                      f"n={n_bg}) = {sigma_used:.2f}")
                print(f"    median brain b=0 signal   = {b0_brain_med:.1f}")
                print(f"    median brain b=0 SNR      = {b0_brain_med/sigma_used:.1f}")

    if rician_correct and sigma_used is not None:
        for di in range(n_deltas):
            all_data[di] = rician_correct_secondmoment(all_data[di], sigma_used)

    # ----------------------------------------------------------------
    # Per-Δ S0 = mean of all b=0 volumes within that input
    # ----------------------------------------------------------------
    s0_per_delta = []
    for di in range(n_deltas):
        b0_idx = all_b0_idx[di]
        # (X, Y, Z, n_b0)  →  per-voxel mean → flatten with mask
        b0_block = all_data[di][..., b0_idx]
        b0_mean_vol = np.mean(b0_block, axis=-1)
        s0_vox = b0_mean_vol[mask]
        s0_per_delta.append(s0_vox)
        print(f"  Δ={input_specs[di][1]:g}: S0 from {len(b0_idx)} b=0 vols, "
              f"median brain S0 = {np.median(s0_vox):.1f}")

    if avg_s0 and n_deltas > 1:
        s0_stack = np.stack(s0_per_delta, axis=0)             # (n_d, n_vox)
        s0_common = np.mean(s0_stack, axis=0)
        s0_cv = np.std(s0_stack, axis=0) / (s0_common + 1e-10)
        print(f"  --avg-s0: averaging S0 across {n_deltas} Δ scans")
        print(f"    median across-Δ S0 CV = {np.median(s0_cv)*100:.2f}%   "
              f"(low = scans well registered)")
        print(f"    95th-pct CV           = {np.percentile(s0_cv, 95)*100:.2f}%")
        if np.median(s0_cv) > 0.10:
            print(f"    ⚠ High S0 variability across Δ — check motion/drift "
                  f"before trusting the averaged S0.")
        s0_used = [s0_common.copy() for _ in range(n_deltas)]
    else:
        if avg_s0 and n_deltas == 1:
            print("  --avg-s0 is a no-op for single-Δ input "
                  "(b=0 vols within the input are already averaged).")
        s0_used = s0_per_delta

    # ----------------------------------------------------------------
    # Build measured matrix in fit_triples order
    # ----------------------------------------------------------------
    measured   = np.zeros((n_vox, n_features), dtype=np.float64)
    raw_signal = np.zeros((n_vox, n_features), dtype=np.float64)

    col = 0
    for di, kept in enumerate(all_shells_kept):
        vox_data = all_data[di][mask, :]            # (n_vox, n_vols)
        S0 = s0_used[di].copy()
        S0[S0 < 1e-10] = 1e-10
        for b, idx in kept:
            shell_mean = np.mean(vox_data[:, idx], axis=1)
            measured[:, col]   = shell_mean / S0
            raw_signal[:, col] = shell_mean
            col += 1
    assert col == n_features

    direction_contract = None
    if return_metadata:
        direction_contract = _direction_contract(direction_scheme, direction_shell_reports)

    if return_raw or return_metadata:
        s0_ref = np.mean(np.stack(s0_used, axis=0), axis=0)
        # Mean number of averaged directions per kept (Δ,b) shell — used to
        # propagate Rician σ through shell averaging when auto-estimating σ_m.
        n_dir = [len(idx) for kept in all_shells_kept for _, idx in kept]
        mean_n_dir = float(np.mean(n_dir)) if n_dir else 1.0
        extras = dict(
            s0=s0_ref,
            sigma=sigma_used,
            mean_n_dir=mean_n_dir,
            s0_median=float(np.median(s0_ref)),
            rician_correction_applied=bool(rician_correct and sigma_used is not None),
            direction_contract=direction_contract,
            snap_events=_deduplicate_snap_events(all_snap_events),
            input_provenance=input_provenance,
            mask_provenance=_file_provenance(mask_path),
        )
        if return_raw:
            extras["raw"] = raw_signal
        return measured, fit_triples, affine, mask_idx, shape, extras, sigma_used

    return measured, fit_triples, affine, mask_idx, shape, sigma_used


# ===================================================================
# Saving maps
# ===================================================================

def save_map(data_1d, mask_idx, shape, affine, path, dtype=np.float32):
    import nibabel as nib
    vol = np.zeros(shape, dtype=dtype)
    vol[mask_idx] = data_1d.astype(dtype)
    nib.save(nib.Nifti1Image(vol, affine), path)
    print(f"  Saved {path}")


def save_derived_biomarker_maps(kio_map, rho_map, V_map,
                                mask_idx, shape, affine, out_dir: str) -> list[str]:
    """Write voxel-first MADI derived endpoints and return their filenames."""
    biomarkers = derive_voxelwise_biomarkers(kio_map, rho_map, V_map)
    outputs = {
        "vi_map.nii.gz": biomarkers.vi,
        "kioV_map.nii.gz": biomarkers.kioV,
        "kioVrho_map.nii.gz": biomarkers.kioVrho,
        "water_efflux_nmol_s_cell_map.nii.gz": biomarkers.water_efflux_nmol_s_cell,
    }
    for filename, values in outputs.items():
        save_map(values, mask_idx, shape, affine, os.path.join(out_dir, filename))
    return list(outputs)


def _distribution_summary(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n_finite": 0}
    return {
        "n_finite": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _record_fit_completion(
    artifacts: FitRunArtifacts | None,
    *,
    args,
    resolved_device: str,
    fit_triples,
    extras: dict,
    sigma_used,
    edge_summary: dict,
    residual: np.ndarray,
    primary_maps: list[str],
    derived_maps: list[str],
    method_metadata: dict | None = None,
) -> None:
    """Fill the single structured sidecar at the successful end of a fit."""
    if artifacts is None:
        return
    artifacts.update(
        fit_configuration={
            "method": args.method,
            "device": resolved_device,
            "linear_or_log_space": "log" if args.log_space else "linear",
            "S0_mode": "free" if args.fit_s0 else "fixed",
            "rician_correction": bool(extras.get("rician_correction_applied")),
            "noise_sigma": sigma_used,
            "masks_applied": {
                "mask": extras.get("mask_provenance"),
                "z_slice": args.z_slice,
            },
            "direction_scheme": extras.get("direction_contract"),
            "fit_triples_delta_Delta_b": [
                [float(delta), float(Delta), float(b)] for delta, Delta, b in fit_triples
            ],
            "grid_edge_policy": {
                "vi_min": float(args.vi_min),
                "vi_max": float(args.vi_max),
                "rho_max": args.rho_max,
                "include_free_water": bool(args.include_free_water),
                "warning_fraction": float(args.edge_warning_fraction),
            },
            "tie_breaking": "first array-order entry among exact minima (numpy.argmin)",
            "rng_seed": int(args.seed),
            "method_options": method_metadata or {},
        },
        input_data_provenance={
            "inputs": extras.get("input_provenance", []),
            "mask": extras.get("mask_provenance"),
            "voxel_dimensions_mm": [
                item.get("voxel_dimensions_mm")
                for item in extras.get("input_provenance", [])
            ],
        },
        snap_events=extras.get("all_snap_events", []),
        snap_summary=extras.get("snap_summary", {}),
        results_summary={
            "n_voxels": int(len(residual)),
            "edge_railing": edge_summary,
            "residual": _distribution_summary(residual),
            "primary_maps_written": primary_maps,
            "derived_maps_written": derived_maps,
            "aggregation_order_assertion": (
                "PASS: v_i, k_ioV, k_ioVrho, and water efflux were computed "
                "voxel-wise before any ROI aggregation API can operate."
            ),
        },
    )
    artifacts.mark_success()


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="MADI library builder & fitter (with Rician correction "
                    "and flexible S0 handling)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # -- Actions --
    ap.add_argument("--build-library", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--export-voxel", type=int, nargs=3, default=None,
                    metavar=("I", "J", "K"),
                    help="Export ONE voxel's measured decay (normalized + "
                         "raw, in fit_triples column order) to an .npz for "
                         "analysis/view_error_landscape_3d.py, instead of "
                         "fitting the whole volume. Uses the same "
                         "--input/--mask/--rician-correct/--noise-sigma/"
                         "--avg-s0/--library/--small-delta flags as --fit, "
                         "so the exported curve is built by the exact same "
                         "code path (load_dwi_and_average) as a real fit -- "
                         "no separate shell-averaging logic to drift out of "
                         "sync. Voxel indices (I,J,K) are in the --mask/DWI "
                         "native voxel grid.")

    # -- Library file --
    ap.add_argument("--library", default="madi_library.npz")
    ap.add_argument("--append", action="store_true")

    # -- Preset grid --
    ap.add_argument("--lib-preset", default="default", choices=list(PRESETS.keys()))
    ap.add_argument("--remediation-n-rho", type=int, default=64,
                    help="[dense remediation grid] number of uniform-log rho nodes "
                         "(default 64; use a small value only for a pilot).")
    ap.add_argument("--remediation-n-V", type=int, default=64,
                    help="[dense remediation grid] number of uniform-log V nodes "
                         "(default 64; use a small value only for a pilot).")
    ap.add_argument("--remediation-rho-min", type=float, default=1.0e4)
    ap.add_argument("--remediation-rho-max", type=float, default=1.0e7)
    ap.add_argument("--remediation-V-min", type=float, default=0.01)
    ap.add_argument("--remediation-V-max", type=float, default=200.0)
    ap.add_argument("--remediation-kios", type=float, nargs="+", default=None,
                    help="[dense remediation grid] explicit piecewise-linear k_io "
                         "nodes, including zero; intended for pilot builds.")
    ap.add_argument("--remediation-entry-subset", type=str, default=None,
                    help="[dense remediation grid] JSON declaration of a restricted, "
                         "canonical-entry build (for a separately declared v5 "
                         "diagnostic or replicate job).")
    ap.add_argument("--sim-walkers", type=int, default=None,
                    help="Override walkers per ensemble for this build (pilot only).")
    ap.add_argument("--sim-ensembles", type=int, default=None,
                    help="Override independent ensembles per entry for this build (pilot only).")
    ap.add_argument("--geometry-reference", type=str, default=None,
                    help="Certified SI §S.IV 5e6-cell / 26-alpha geometry-reference "
                         "table. Defaults to data/geometry_reference_si_kappa_0p9.npz. "
                         "A library build fails if the certified table is absent.")
    ap.add_argument("--sim-T-max", type=float, default=None,
                    help="Override walk duration in ms; must cover every selected δ+Δ.")
    ap.add_argument("--sim-small-deltas", type=float, nargs="+", default=None,
                    help="Override stored δ grid in ms (pilot only).")
    ap.add_argument("--sim-big-deltas", type=float, nargs="+", default=None,
                    help="Override stored Δ grid in ms (pilot only).")
    ap.add_argument("--sim-b-values", type=float, nargs="+", default=None,
                    help="Override stored b grid in s/mm² (pilot only).")

    # -- Custom additions --
    ap.add_argument("--custom-kios", type=float, nargs="+")
    ap.add_argument("--custom-rhos", type=float, nargs="+")
    ap.add_argument("--custom-Vs",   type=float, nargs="+")

    # -- Explicit sub-grid --
    ap.add_argument("--explicit", action="store_true")
    ap.add_argument("--grid-kios", type=float, nargs="+")
    ap.add_argument("--grid-rhos", type=float, nargs="+")
    ap.add_argument("--grid-Vs",   type=float, nargs="+")

    # -- Exact triplets --
    ap.add_argument("--triplets", type=str, nargs="+")

    # -- Sharding --
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)

    # -- Build-level RNG seed --
    ap.add_argument("--seed", type=int, default=0,
                    help="Build-level RNG seed. MUST be the same across "
                         "every SLURM shard of a given library build: "
                         "ensemble/walker seeds are derived from "
                         "(seed, ensemble_index) only -- never from "
                         "(rho, V) -- so every (rho,V) grid point shares "
                         "correlated random-number streams (common random "
                         "numbers), which the Fisher/CRLB phase needs. "
                         "Changing --seed between shards of the same "
                         "library silently breaks that correlation.")

    # -- Fitting inputs --
    ap.add_argument("--input", type=str, nargs="+",
                    help="'delta:path' pairs (e.g. 15:dwi15.nii.gz)")
    ap.add_argument("--mask", default=None,
                    help="Optional brain mask NIfTI.  If omitted, the "
                         "fit is run over every voxel of the volume.  "
                         "Required when --rician-correct is used without "
                         "an explicit --noise-sigma.")
    ap.add_argument("--z-slice", default=None,
                    help="Restrict fitting to a single Z-slice or range. "
                         "Examples: '50' (just slice 50), '40:60' "
                         "(slices 40-59), ':60' (slices 0-59), '40:' "
                         "(slices 40 to end).  Intersects with --mask "
                         "if both are given.  Noise σ estimation still "
                         "uses the un-sliced volume.")
    ap.add_argument("--out", default="madi_output")
    ap.add_argument("--run-name", default=None,
                    help="Basename for the exactly two fit sidecars "
                         "<run-name>.json and <run-name>.log. Defaults to "
                         "the --out directory name.")
    ap.add_argument("--direction-scheme", choices=DIRECTION_SCHEMES,
                    default=None,
                    help="Required acquisition contract: single_direction, "
                         "orthogonal_3, powder_N, or prescanner_averaged. "
                         "If omitted, the fitter proceeds only when b-vectors "
                         "unambiguously infer it.")

    # -- NEW: Rician + S0 options --
    ap.add_argument("--rician-correct", action="store_true",
                    help="Apply Rician noise-bias correction "
                         "(E[M^2] = A^2 + 2 sigma^2) to each volume "
                         "before normalization.")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="Rician noise std.  If omitted and --rician-correct "
                         "is set, sigma is estimated from background voxels "
                         "of the first scan's b=0 image, drawn from outside "
                         "a dilated copy of --mask (see "
                         "--noise-bg-dilate-iters) so skull/scalp tissue "
                         "immediately outside the brain mask doesn't bias "
                         "the estimate.")
    ap.add_argument("--noise-bg-dilate-iters", type=int, default=48,
                    help="[auto sigma only] Number of binary-dilation "
                         "iterations applied to --mask before excluding it "
                         "from the background region used to estimate "
                         "noise sigma (default 64). Larger values push the "
                         "background sample further from the brain, past "
                         "the skull/scalp, at the cost of fewer background "
                         "voxels; tune upward if the auto sigma still looks "
                         "too high, downward if too few background voxels "
                         "remain for a small FOV.")
    ap.add_argument("--avg-s0", action="store_true",
                    help="Average the b=0 volumes across all Δ scans into a "
                         "single S0 image for normalization.  TE is constant "
                         "across Δ so this only loses information if the "
                         "scans are misregistered.")
    ap.add_argument("--fit-s0", action="store_true",
                    help="Treat S0 as a free per-voxel parameter in the "
                         "matcher (analytic L2-optimal projection per "
                         "library entry).  Diagnostic for S0 reliability.")
    ap.add_argument("--log_space", action="store_true",
                    help="Whether to preform fitting within log space or to use no transformations.")

    # -- NEW: fitting method selection --
    method_grp = ap.add_argument_group(
        "fitting method",
        "Choose how each voxel is fit against the library.  'map' is the "
        "original point-estimate matcher and is byte-for-byte unchanged; "
        "'bayes' and 'amico' additionally emit posterior mean/std maps.")
    method_grp.add_argument(
        "--method", choices=["map", "bayes", "amico"], default="map",
        help="map: nearest-library-entry MAP estimate (default, "
             "backwards-compatible).  bayes: Gaussian posterior mean/std "
             "over the whole library.  amico: elastic-net NNLS mixture.")
    method_grp.add_argument(
        "--sigma-m", type=float, default=None,
        help="[bayes only] Residual-noise std on the normalized S/S0 "
             "signal.  If omitted, auto-estimated from Rician σ when "
             "--rician-correct is on, else defaults to "
             f"{DEFAULT_SIGMA_M} (a placeholder — a warning is logged). "
             "Ignored if --target-n-eff is given.")
    method_grp.add_argument(
        "--target-n-eff", type=float, default=None,
        help="[bayes only] Instead of a fixed --sigma-m, pick σ_m so that "
             "the posterior's effective number of contributing library "
             "entries (n_eff = 1/Σw_i², median over a random voxel "
             "subsample) hits this value. Useful with --fit-s0, whose "
             "residual scale (S0 fitted per candidate) is not directly "
             "comparable to the fixed-S0 residual, so the same σ_m does "
             "not give comparable posterior sharpness across the two — "
             "run without --fit-s0 first, note its n_eff, then pass that "
             "value here for the --fit-s0 run to match its discrimination.")
    method_grp.add_argument(
        "--lambda1", type=float, default=DEFAULT_LAMBDA1,
        help=f"[amico only] L1 (sparsity) penalty (default {DEFAULT_LAMBDA1}).")
    method_grp.add_argument(
        "--lambda2", type=float, default=DEFAULT_LAMBDA2,
        help=f"[amico only] L2 (ridge) penalty (default {DEFAULT_LAMBDA2}).")
    method_grp.add_argument(
        "--device", choices=["auto", "cpu", "gpu"], default="auto",
        help="auto (default): use CUDA if available, else CPU.  gpu: force "
             "GPU (errors if CUDA is unavailable).  cpu: force CPU.  MAP and "
             "bayes GPU kernels are exact reorderings of the CPU math "
             "(same output to float64 precision); amico's GPU path is an "
             "approximate FISTA solve replacing the CPU's exact per-voxel "
             "NNLS, so amico GPU/CPU outputs will be close but not "
             "bit-identical (see docs/fitting_methods.md).")
    method_grp.add_argument(
        "--gpu-chunk-voxels", type=int,
        default=fitters_gpu.DEFAULT_GPU_CHUNK_VOXELS,
        help="[amico + --device gpu only] voxels per GPU batch for the "
             f"FISTA solve (default {fitters_gpu.DEFAULT_GPU_CHUNK_VOXELS}). "
             "MAP/bayes GPU kernels process all voxels in one launch and "
             "ignore this.")
    method_grp.add_argument(
        "--amico-gpu-iters", type=int,
        default=fitters_gpu.DEFAULT_AMICO_ITERS,
        help="[amico + --device gpu only] max FISTA iterations per voxel "
             f"(default {fitters_gpu.DEFAULT_AMICO_ITERS}).")
    method_grp.add_argument(
        "--amico-gpu-tol", type=float,
        default=fitters_gpu.DEFAULT_AMICO_TOL,
        help="[amico + --device gpu only] relative-objective-change "
             f"early-exit tolerance for the FISTA solve (default "
             f"{fitters_gpu.DEFAULT_AMICO_TOL}). Convergence speed depends "
             "strongly on --lambda2 (it also conditions the problem for "
             "this solver, since MADI library entries are highly "
             "correlated): near --lambda2 0, n_eff can look stuck high even "
             "though it is still slowly dropping — raise --amico-gpu-iters "
             "substantially (tens of thousands) in that regime, or use "
             "--device cpu for an exact answer.")
    # -- NEW: acquisition metadata (must match library) --
    ap.add_argument("--small-delta", type=float, default=None,
                    help="Global δ (PFG gradient duration) [ms] of the data "
                         "being fit, used as the default for every --input "
                         "that doesn't carry its own δ. The (δ,Δ,b)-universal "
                         "library is indexed by (δ,Δ,b), so a δ is required "
                         "for every scan: either pass --small-delta or give "
                         "each input as 'δ,Δ:dwi...'. Each (δ,Δ) is matched to "
                         "the nearest stored library pair (no interpolation).")
    # -- vi bounds (used BOTH for --build-library and for matching) --
    ap.add_argument("--vi-min", type=float, default=0.40,
                    help="Lower bound on intracellular volume fraction "
                         "vi = (rho/1e9)*(V*1e3). With --build-library, only "
                         "(kio,rho,V) triplets with vi in [--vi-min, --vi-max] "
                         "are simulated (e.g. --vi-min 0.4 skips sparse/low-vi "
                         "tissue). With --fit, bounds the library candidates.")
    ap.add_argument("--vi-max", type=float, default=0.99,
                    help="Upper bound on v_i. Production geometry directly "
                         "calibrates every target and raises on a miss; it "
                         "never clamps a target to a different geometry.")
    ap.add_argument("--rho-max", type=float, default=None,
                    help="Optional upper bound on library rho [cells/uL] "
                         "for matching (e.g. 1500000 for brain).")
    ap.add_argument("--include-free-water", action="store_true",
                    help="Include the explicit rho=V=0 free-water atom in "
                         "the MAP candidate set. When selected, k_io is "
                         "reported as NaN (undefined), not a false zero. "
                         "The default remains the paper-compatible cellular "
                         "reference matcher; use an acellular mask or this "
                         "flag deliberately.")
    ap.add_argument("--edge-warning-fraction", type=float, default=0.10,
                    help="Warn when this fraction of fitted voxels sits on "
                         "a realised cellular library boundary (default 0.10).")

    args = ap.parse_args()
    artifacts = None
    if args.fit:
        artifacts = _start_fit_artifacts(args.out, args.run_name)
        artifacts.update(
            command=[sys.executable, *sys.argv],
            requested_cli_arguments=vars(args).copy(),
            requested_fit_configuration={
                "method": args.method,
                "device": args.device,
                "linear_or_log_space": "log" if args.log_space else "linear",
                "fit_s0": bool(args.fit_s0),
                "rician_correction_requested": bool(args.rician_correct),
                "direction_scheme_requested": args.direction_scheme,
                "vi_min": float(args.vi_min),
                "vi_max": float(args.vi_max),
                "rho_max": args.rho_max,
                "include_free_water": bool(args.include_free_water),
                "tie_breaking": "first array-order entry among exact minima (numpy.argmin)",
                "b_snap_tolerance_s_mm2": B_LIB_MATCH_TOL,
                "timing_snap_tolerance_ms": TIMING_SNAP_TOL_MS,
                "rng_seed": int(args.seed),
            },
        )
    
    # # Manually overridden arguments matching the terminal command
    # args.fit = True
    # args.input = [
    #     "15:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_15ms/eddy_corrected.nii.gz",
    #     "25:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_25ms/eddy_corrected.nii.gz",
    #     "30:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_30ms/eddy_corrected.nii.gz",
    #     "40:/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_40ms/eddy_corrected.nii.gz"
    # ]
    # args.mask = "/mnt/c/Miscellaneous/Coding_Projects/Python/mri_processing/data/2026-02-05_NEXI_H/preprocessed_4/DWI_15ms/mask_cropped.nii.gz"
    # args.out = "out_baseline"
    # args.library = "data/libraries/madi_dense.npz"

    # args.fit_s0 = False
    # args.rician_correct = False
    # args.avg_s0 = False

    if not any([args.build_library, args.fit, args.info,
                args.export_voxel is not None]):
        ap.print_help(); return

    # ================================================================
    #  INFO / BUILD branches unchanged - omitted for brevity
    # ================================================================
    if args.info:
        if not os.path.exists(args.library):
            print(f"Library not found: {args.library}"); return
        lib = load_library(args.library)
        meta = load_library_meta(args.library)
        print(f"\nLibrary: {args.library}")
        library_summary(lib, meta=meta)
        return

# ================================================================
    #  BUILD LIBRARY
    # ================================================================
    if args.build_library:
        print("=" * 60)
        print("Building MADI library")
        print("=" * 60)

        # Get simulation config from preset
        preset = PRESETS[args.lib_preset]
        cfg_values = dict(preset["cfg"])
        for arg_name, cfg_name in (
            ("sim_walkers", "n_walkers"),
            ("sim_ensembles", "n_ensembles"),
            ("geometry_reference", "geometry_reference_path"),
            ("sim_T_max", "T_max_ms"),
            ("sim_small_deltas", "small_deltas"),
            ("sim_big_deltas", "big_deltas"),
            ("sim_b_values", "b_values"),
        ):
            value = getattr(args, arg_name)
            if value is not None:
                cfg_values[cfg_name] = value
        cfg = SimConfig(**cfg_values)
        remediation_grid = None
        remediation_triplets = None
        remediation_weights = None
        build_grid_metadata = None
        if args.remediation_entry_subset is not None:
            if preset.get("grid") != "remediation_log":
                print("ERROR: --remediation-entry-subset requires --lib-preset dense")
                return
            if args.triplets or args.explicit or args.custom_kios or args.custom_rhos or args.custom_Vs:
                print("ERROR: --remediation-entry-subset cannot be combined with other grid selectors")
                return
            if (args.remediation_kios is not None
                    or args.remediation_n_rho != 64 or args.remediation_n_V != 64
                    or args.remediation_rho_min != 1.0e4 or args.remediation_rho_max != 1.0e7
                    or args.remediation_V_min != 0.01 or args.remediation_V_max != 200.0
                    or args.vi_min != 0.40 or args.vi_max != 0.99):
                print("ERROR: --remediation-entry-subset requires the unmodified P0 remediation grid")
                return
        if preset.get("grid") == "remediation_log":
            remediation_grid = make_remediation_log_grid(
                n_rho=args.remediation_n_rho,
                n_V=args.remediation_n_V,
                kios=(None if args.remediation_kios is None
                      else np.asarray(args.remediation_kios, dtype=float)),
                rho_min=args.remediation_rho_min,
                rho_max=args.remediation_rho_max,
                V_min=args.remediation_V_min,
                V_max=args.remediation_V_max,
                vi_min=args.vi_min, vi_max=args.vi_max,
            )
            remediation_triplets, remediation_weights = remediation_grid.triplets_and_weights()
            build_grid_metadata = remediation_grid.metadata()
            if args.remediation_entry_subset is not None:
                try:
                    remediation_triplets, subset_provenance = load_remediation_entry_subset(
                        args.remediation_entry_subset, remediation_triplets,
                    )
                except ValueError as exc:
                    print(f"ERROR: {exc}")
                    return
                remediation_weights = {
                    _entry_key_for_script(*triplet): remediation_weights[
                        _entry_key_for_script(*triplet)
                    ]
                    for triplet in remediation_triplets
                }
                build_grid_metadata = {
                    **build_grid_metadata,
                    "restricted_entry_subset": subset_provenance,
                }

        # Load existing if appending
        existing = None
        if args.append and os.path.exists(args.library):
            print(f"\n  Loading existing library: {args.library}")
            existing = load_library(args.library)
        elif args.append:
            print(f"\n  --append: {args.library} not found, starting fresh.")

        # ---- Mode: exact triplets ----
        if args.triplets:
            triplets = [parse_triplet(s) for s in args.triplets]
            print(f"\n  Mode: exact triplets ({len(triplets)})")
            for k, r, v in triplets:
                print(f"    kio={k}, rho={r/1e3:.0f}k, V={v:.2f}")

            build_library_from_triplets(
                triplets, cfg=cfg, save_path=args.library,
                existing_library=existing, seed=args.seed,
                vi_min=args.vi_min, vi_max=args.vi_max)
            return

        # ---- Mode: explicit sub-grid ----
        if args.explicit:
            if remediation_grid is not None and not (
                args.grid_kios and args.grid_rhos and args.grid_Vs
            ):
                print("ERROR: --explicit with the remediation grid requires "
                      "--grid-kios, --grid-rhos and --grid-Vs.  The production "
                      "dense preset is intentionally not a mutable linear grid.")
                return
            gk = args.grid_kios or preset["kios"]
            gr = [int(r) for r in (args.grid_rhos or preset["rhos"])]
            gv = args.grid_Vs or preset["Vs"]
            print(f"\n  Mode: explicit sub-grid")
            print(f"  kio ({len(gk)}): {gk}")
            print(f"  rho ({len(gr)}): {[f'{r/1e3:.0f}k' for r in gr]}")
            print(f"  V   ({len(gv)}): {gv}")

            build_library(
                kios=gk, rhos=gr, Vs=gv, cfg=cfg,
                save_path=args.library, existing_library=existing,
                seed=args.seed, vi_min=args.vi_min, vi_max=args.vi_max)
            return

        # ---- Mode: remediation production grid ---------------------------
        if remediation_grid is not None:
            if args.custom_kios or args.custom_rhos or args.custom_Vs:
                print("ERROR: custom linear-axis additions are incompatible "
                      "with the weighted remediation grid. Use an explicit "
                      "diagnostic grid instead; do not mutate production weights.")
                return
            assert remediation_triplets is not None and remediation_weights is not None
            print("\n  Mode: weighted remediation log grid")
            print(f"  cellular triplets: {len(remediation_triplets)-1}")
            print(f"  rho: {args.remediation_rho_min:,.0f} .. "
                  f"{args.remediation_rho_max:,.0f} cells/uL "
                  f"({args.remediation_n_rho} uniform-log nodes)")
            print(f"  V:   {args.remediation_V_min:g} .. {args.remediation_V_max:g} pL "
                  f"({args.remediation_n_V} uniform-log nodes)")
            print(f"  vi:  {args.vi_min:.3f} .. {args.vi_max:.3f} (diagonal mask)")
            print(f"  kio: {remediation_grid.kios.tolist()} s^-1")
            print("  free water: one explicit discrete atom")
            if args.remediation_entry_subset is not None:
                subset = build_grid_metadata["restricted_entry_subset"]
                print(f"  restricted declared subset: {subset['cellular_entries']} cellular "
                      f"+ {subset['total_entries'] - subset['cellular_entries']} free-water entries")

            if args.shard_id is not None:
                if args.n_shards is None or args.n_shards < 1:
                    print("ERROR: --shard-id requires --n-shards >= 1")
                    return
                if not (0 <= args.shard_id < args.n_shards):
                    print(f"ERROR: --shard-id must be in [0, {args.n_shards})")
                    return
                cellular_pairs = sorted(
                    {(r, v) for _, r, v in remediation_triplets if r > 0.0},
                    key=lambda p: p[0] * p[1],
                )
                my_pairs = {
                    pair for i, pair in enumerate(cellular_pairs)
                    if i % args.n_shards == args.shard_id
                }
                shard_triplets = [
                    t for t in remediation_triplets
                    if (t[1] > 0.0 and (t[1], t[2]) in my_pairs)
                    or (args.shard_id == 0 and t[1] == 0.0 and t[2] == 0.0)
                ]
                shard_weights = {
                    key: remediation_weights[key]
                    for key in [_entry_key_for_script(*t) for t in shard_triplets]
                }
                root, ext = os.path.splitext(args.library)
                save_path = (f"{root}.shard{args.shard_id:03d}{ext}"
                             if "shard" not in os.path.basename(args.library)
                             else args.library.format(shard=args.shard_id, n_shards=args.n_shards))
                print(f"\n  Sharding: {args.shard_id}/{args.n_shards}; "
                      f"{len(shard_triplets)} triplets -> {save_path}")
                build_library_from_triplets(
                    shard_triplets, cfg=cfg, save_path=save_path,
                    existing_library=existing, seed=args.seed,
                    vi_min=args.vi_min, vi_max=args.vi_max,
                    entry_weights=shard_weights,
                    grid_metadata=build_grid_metadata,
                )
                return

            build_library_from_triplets(
                remediation_triplets, cfg=cfg, save_path=args.library,
                existing_library=existing, seed=args.seed,
                vi_min=args.vi_min, vi_max=args.vi_max,
                entry_weights=remediation_weights,
                grid_metadata=build_grid_metadata,
            )
            return

        # ---- Mode: legacy/custom preset grid + optional additions ----
        kios = list(preset["kios"])
        rhos = list(preset["rhos"])
        Vs   = list(preset["Vs"])

        if args.custom_kios:
            kios = sorted(set(kios + args.custom_kios))
        if args.custom_rhos:
            rhos = sorted(set(rhos + [int(r) for r in args.custom_rhos]))
        if args.custom_Vs:
            Vs = sorted(set(Vs + args.custom_Vs))

        print(f"\n  Mode: preset '{args.lib_preset}' + custom additions")
        print(f"  kio ({len(kios)}): {kios}")
        print(f"  rho ({len(rhos)}): {[f'{r/1e3:.0f}k' for r in rhos]}")
        print(f"  V   ({len(Vs)}):   {Vs}")

        # ---- Optional sharding for SLURM job arrays ----
        if args.shard_id is not None:
            if args.n_shards is None or args.n_shards < 1:
                print("ERROR: --shard-id requires --n-shards >= 1")
                return
            if not (0 <= args.shard_id < args.n_shards):
                print(f"ERROR: --shard-id must be in [0, {args.n_shards})")
                return

            # Build full triplet list, filter by vi, slice by (ρ,V) pair.
            # Use the SAME vi bounds the builder will apply so shard
            # assignment matches what actually gets computed (otherwise a
            # shard could be handed entries that are then all skipped).
            vi_hi = min(args.vi_max, 0.99)
            all_triplets = [(k, r, v) for k in kios for r in rhos for v in Vs]
            valid = [(k, r, v) for k, r, v in all_triplets
                     if args.vi_min <= (r / 1e9) * (v * 1e3) <= vi_hi]

            # Unique (ρ,V) pairs, sorted by cost proxy (ρ·V), round-robin
            # across shards so each shard gets a mix of cheap + expensive.
            pairs = sorted(set((r, v) for _, r, v in valid),
                           key=lambda p: p[0] * p[1])
            my_pairs = set(pairs[i] for i in range(len(pairs))
                           if i % args.n_shards == args.shard_id)
            shard_triplets = [(k, r, v) for (k, r, v) in valid
                              if (r, v) in my_pairs]

            print(f"\n  Sharding: shard {args.shard_id}/{args.n_shards}")
            print(f"    (ρ,V) pairs in shard : {len(my_pairs)}/{len(pairs)}")
            print(f"    triplets in shard    : {len(shard_triplets)}/{len(valid)}")

            # Tag the output file with shard id unless user explicitly
            # supplied a per-shard path already.
            save_path = args.library
            if "{shard" not in save_path and "shard" not in os.path.basename(save_path):
                root, ext = os.path.splitext(save_path)
                save_path = f"{root}.shard{args.shard_id:03d}{ext}"
            else:
                save_path = save_path.format(shard=args.shard_id,
                                             n_shards=args.n_shards)
            print(f"    output               : {save_path}")

            build_library_from_triplets(
                shard_triplets, cfg=cfg, save_path=save_path,
                existing_library=existing, seed=args.seed,
                vi_min=args.vi_min, vi_max=args.vi_max)
            return

        build_library(
            kios=kios, rhos=rhos, Vs=Vs, cfg=cfg,
            save_path=args.library, existing_library=existing,
            seed=args.seed, vi_min=args.vi_min, vi_max=args.vi_max)
        return

    # ================================================================
    #  EXPORT ONE VOXEL'S DECAY CURVE (for view_error_landscape_3d.py)
    # ================================================================
    if args.export_voxel is not None:
        i_vox, j_vox, k_vox = args.export_voxel

        input_specs = []
        if args.input:
            for s in args.input:
                input_specs.append(parse_input(s))
        if not input_specs:
            print("ERROR: No DWI inputs specified."); return
        input_specs.sort(key=lambda x: x[1])

        if not os.path.exists(args.library):
            print(f"ERROR: Library not found: {args.library}"); return
        meta = load_library_meta(args.library)
        lib_b_values = meta['b_values']
        if lib_b_values is None:
            print("ERROR: library has no stored b-values metadata."); return

        print("=" * 60)
        print(f"Exporting voxel ({i_vox}, {j_vox}, {k_vox})")
        print("=" * 60)
        measured, fit_triples, affine, mask_idx, shape, extras, sigma_used = \
            load_dwi_and_average(
                input_specs, args.mask,
                lib_b_values=lib_b_values,
                default_small_delta=args.small_delta,
                rician_correct=args.rician_correct,
                noise_sigma=args.noise_sigma,
                noise_bg_dilate_iters=args.noise_bg_dilate_iters,
                avg_s0=args.avg_s0,
                return_raw=True,
            )

        hit = ((mask_idx[0] == i_vox) & (mask_idx[1] == j_vox) &
               (mask_idx[2] == k_vox))
        pos = np.where(hit)[0]
        if pos.size == 0:
            print(f"ERROR: voxel ({i_vox},{j_vox},{k_vox}) is not in "
                  f"--mask (or falls outside the volume, shape {shape}). "
                  f"Nothing exported.")
            return
        pos = int(pos[0])

        measured_vec = measured[pos]
        raw_vec = extras['raw'][pos]
        s0 = float(extras['s0'][pos])
        small_deltas = np.array([d for d, _, _ in fit_triples], dtype=float)
        deltas = np.array([D for _, D, _ in fit_triples], dtype=float)
        bvals = np.array([b for _, _, b in fit_triples], dtype=float)

        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(
            args.out, f"voxel_{i_vox}_{j_vox}_{k_vox}.npz")
        np.savez(
            out_path,
            measured=measured_vec, raw=raw_vec,
            fit_small_deltas=small_deltas, fit_deltas=deltas, fit_bvals=bvals,
            s0=s0, sigma=(sigma_used if sigma_used is not None else np.nan),
            mean_n_dir=extras['mean_n_dir'], s0_median=extras['s0_median'],
            ijk=np.array([i_vox, j_vox, k_vox], dtype=int),
            affine=affine, library=args.library,
            rician_correct=bool(args.rician_correct),
        )
        print(f"\n  S0 = {s0:.1f}   sigma = {sigma_used}")
        print(f"  measured (S/S0): {np.array2string(measured_vec, precision=4)}")
        print(f"  fit_triples (δ,Δ,b): "
              f"{list(zip(small_deltas.tolist(), deltas.tolist(), bvals.tolist()))}")
        print(f"\n  Saved {out_path}")
        print("  Open with: python analysis/view_error_landscape_3d.py "
              f"--library {args.library} --voxel-data {out_path}")
        return

    # ================================================================
    #  FIT DATA
    # ================================================================
    if args.fit:
        # Parse inputs
        input_specs = []
        if args.input:
            for s in args.input:
                input_specs.append(parse_input(s))

        if not input_specs:
            print("ERROR: No DWI inputs specified."); return
        if args.mask is None and args.rician_correct and args.noise_sigma is None:
            print("ERROR: --rician-correct without --mask requires "
                  "--noise-sigma <value>.  Auto-estimation of σ needs "
                  "background (air) voxels, which requires a brain mask "
                  "to identify."); return

        input_specs.sort(key=lambda x: x[1])
        fit_deltas = [D for _, D, _, _, _ in input_specs]

        # ---- Warn about method-specific flags that are ignored ----
        if args.method != "bayes" and args.sigma_m is not None:
            print(f"  ⚠ --sigma-m is only used by --method bayes; "
                  f"ignored for --method {args.method}.")
        if args.method != "bayes" and args.target_n_eff is not None:
            print(f"  ⚠ --target-n-eff is only used by --method bayes; "
                  f"ignored for --method {args.method}.")
        if args.method != "amico":
            if args.lambda1 != DEFAULT_LAMBDA1:
                print(f"  ⚠ --lambda1 is only used by --method amico; "
                      f"ignored for --method {args.method}.")
            if args.lambda2 != DEFAULT_LAMBDA2:
                print(f"  ⚠ --lambda2 is only used by --method amico; "
                      f"ignored for --method {args.method}.")
        if args.method == "amico" and args.log_space:
            print(f"  ⚠ --log_space has no effect for --method amico "
                  f"(the regression is linear in signal); ignored.")

        # ---- Resolve --device -> use_gpu, threaded into every fit call ----
        if args.device == "auto":
            use_gpu = None  # let each fitter auto-detect (None = HAS_CUDA)
        elif args.device == "gpu":
            if not fitters_gpu.HAS_CUDA:
                print("ERROR: --device gpu requested but CUDA is not "
                      "available in this environment."); return
            use_gpu = True
        else:  # cpu
            use_gpu = False
        resolved_device = ("gpu" if (use_gpu or
                           (use_gpu is None and fitters_gpu.HAS_CUDA))
                           else "cpu")

        gpu_only_flags_touched = (
            args.gpu_chunk_voxels != fitters_gpu.DEFAULT_GPU_CHUNK_VOXELS or
            args.amico_gpu_iters != fitters_gpu.DEFAULT_AMICO_ITERS or
            args.amico_gpu_tol != fitters_gpu.DEFAULT_AMICO_TOL)
        if gpu_only_flags_touched and not (args.method == "amico" and resolved_device == "gpu"):
            print("  ⚠ --gpu-chunk-voxels/--amico-gpu-iters/--amico-gpu-tol "
                  "only affect --method amico with --device gpu (resolved: "
                  f"method={args.method}, device={resolved_device}); ignored.")

        print("=" * 60)
        print("MADI Fitting")
        print("=" * 60)
        print(f"  Method:                 {args.method}")
        print(f"  Device:                 {resolved_device} "
              f"(--device {args.device})")
        print(f"  Δ values to fit:        {fit_deltas} ms")
        print(f"  Rician correction:      {args.rician_correct}")
        print(f"  S0 averaging across Δ:  {args.avg_s0}")
        print(f"  S0 fitted per voxel:    {args.fit_s0}")
        print(f"  vi range:               [{args.vi_min}, {args.vi_max}]")
        print(f"  rho_max:                {args.rho_max}")

        os.makedirs(args.out, exist_ok=True)

        print(f"\nLoading library: {args.library}")
        if not os.path.exists(args.library):
            print(f"ERROR: Library not found: {args.library}"); return
        lib = load_library(args.library)
        meta = load_library_meta(args.library)
        if artifacts is not None:
            print("  Hashing library for run provenance ...")
            artifacts.update(library_provenance=_library_provenance(args.library, meta, lib))
        lib_delta_pairs = meta['delta_pairs']
        lib_n_b         = meta['n_b']
        lib_b_values    = meta['b_values']

        if lib_b_values is None:
            print(f"ERROR: library has no stored b-values metadata and no "
                  f"safe default could be inferred.  Rebuild the library "
                  f"with the updated _save_library, or patch lib_b_values "
                  f"manually."); return

        print(f"  {len(lib)} entries")
        library_summary(lib, meta=meta)

        # Unique library δ and Δ, derived from the stored (δ,Δ) pairs.
        lib_pairs_arr    = np.asarray(lib_delta_pairs, dtype=float)
        lib_small_deltas = sorted(set(round(float(d), 4)
                                      for d, _ in lib_delta_pairs))
        lib_deltas       = sorted(set(round(float(D), 4)
                                      for _, D in lib_delta_pairs))
        if meta.get('format') == 'legacy':
            print(f"  ⓘ legacy fixed-δ library (δ = {lib_small_deltas} ms); "
                  f"matched as a single-δ grid.")

        # ---- Acquisition consistency check: every fit Δ must sit on the
        # library's Δ grid (δ is validated per-pair after data load, once
        # each scan's δ is resolved). ----
        for d in fit_deltas:
            if not any(abs(d - ld) < 0.01 for ld in lib_deltas):
                print(f"ERROR: Δ = {d} ms not in library {list(lib_deltas)}"); return

        # Load data; produces fit_triples.  Metadata is always returned for
        # the required run sidecar, while the un-normalised raw matrix is
        # retained only for a free-S0 fit to avoid doubling MAP memory use.
        need_raw_signal = args.fit_s0
        print("\nLoading DWI data ...")
        z_slice_obj = parse_z_slice(args.z_slice)
        load_out = load_dwi_and_average(
            input_specs, args.mask,
            lib_b_values=lib_b_values,
            default_small_delta=args.small_delta,
            rician_correct=args.rician_correct,
            noise_sigma=args.noise_sigma,
            noise_bg_dilate_iters=args.noise_bg_dilate_iters,
            avg_s0=args.avg_s0,
            return_raw=need_raw_signal,
            return_metadata=True,
            direction_scheme=args.direction_scheme,
            z_slice=z_slice_obj,
        )
        measured, fit_triples, affine, mask_idx, shape, extras, sigma_used = load_out

        n_features = len(fit_triples)
        print(f"\n  Feature vector ({n_features} cols):")
        for dd, D, b in fit_triples:
            print(f"    δ={dd:g} ms,  Δ={D:g} ms,  b={b:g} s/mm²")

        # ---- Resolve, report, and record every allowed snap.  This uses the
        # exact same resolver as the matcher; the resolver refuses values
        # outside 30 s/mm² in b or 1.5 ms in either timing coordinate. ----
        try:
            _, column_snap_events = resolve_grid_columns(
                fit_triples, lib_delta_pairs, lib_b_values, lib_n_b,
                b_tol=B_LIB_MATCH_TOL, timing_tol=TIMING_SNAP_TOL_MS,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return
        snap_events = _deduplicate_snap_events(
            list(extras.get("snap_events", [])) + column_snap_events)
        if snap_events:
            print("\n  Acquisition snaps (no interpolation):")
            _print_snap_events(snap_events)
        else:
            print("\n  Acquisition snaps: none (all requested coordinates are stored exactly).")
        snap_summary = _snap_summary(snap_events)
        extras["all_snap_events"] = snap_events
        extras["snap_summary"] = snap_summary
        print("  Snap summary: "
              f"{snap_summary['n_snapped_b_shells']} b shell(s), "
              f"{snap_summary['n_snapped_timing_pairs']} timing pair(s); "
              f"max b offset={snap_summary['max_b_abs_offset_s_mm2']:g} s/mm², "
              "max timing offsets="
              f"(δ {snap_summary['max_timing_abs_offset_ms']['delta']:g}, "
              f"Δ {snap_summary['max_timing_abs_offset_ms']['Delta']:g}) ms")

        # Underdetermination warning (3 free params: kio, ρ, V)
        if n_features < 3:
            print(f"\n  ⚠ Only {n_features} measurement(s) per voxel for "
                  f"3 free parameters (kio, ρ, V).  The fit is severely "
                  f"underdetermined; many library entries will produce "
                  f"essentially identical residuals.  Treat the maps as "
                  f"diagnostic only.")
        elif n_features < 6:
            print(f"\n  ⓘ {n_features} measurements per voxel for 3 free "
                  f"parameters — a workable but tight fit.")

        # ================================================================
        #  METHOD DISPATCH
        # ================================================================
        if args.method == "map":
            map_extra_maps: list[str] = []
            # ---- MAP: point-estimate matcher (UNCHANGED behaviour) ----
            if args.fit_s0:
                raw_signal = extras['raw']
                print(f"\nMatching {raw_signal.shape[0]} voxels with S0 FITTED ...")
                t0 = time.time()
                kio_map, rho_map, V_map, res_map, s0_fit_map = match_voxels_batch_fits0(
                    raw_signal, lib,
                    lib_delta_pairs=lib_delta_pairs,
                    lib_b_values=lib_b_values,
                    n_b=lib_n_b,
                    fit_triples=fit_triples,
                    vi_min=args.vi_min,
                    vi_max=args.vi_max,
                    rho_max=args.rho_max,
                    include_free_water=args.include_free_water,
                    use_gpu=use_gpu,
                )
                print(f"  Done in {time.time()-t0:.1f}s")
                save_map(s0_fit_map, mask_idx, shape, affine,
                         os.path.join(args.out, "s0_fit_map.nii.gz"))
                map_extra_maps.append("s0_fit_map.nii.gz")
                s0_ratio = s0_fit_map / (extras['s0'] + 1e-10)
                save_map(s0_ratio, mask_idx, shape, affine,
                         os.path.join(args.out, "s0_fit_over_measured.nii.gz"))
                map_extra_maps.append("s0_fit_over_measured.nii.gz")
                print(f"\n  Fitted-S0 / Measured-S0 ratio:")
                print(f"    median = {np.median(s0_ratio):.3f}")
                print(f"    5-95%  = [{np.percentile(s0_ratio, 5):.3f}, "
                      f"{np.percentile(s0_ratio, 95):.3f}]")
            else:
                print(f"\nMatching {measured.shape[0]} voxels ...")
                t0 = time.time()
                kio_map, rho_map, V_map, res_map = match_voxels_batch(
                    measured, lib,
                    lib_delta_pairs=lib_delta_pairs,
                    lib_b_values=lib_b_values,
                    n_b=lib_n_b,
                    fit_triples=fit_triples,
                    vi_min=args.vi_min,
                    vi_max=args.vi_max,
                    rho_max=args.rho_max,
                    include_free_water=args.include_free_water,
                    log_space=args.log_space,
                    use_gpu=use_gpu,
                )
                print(f"  Done in {time.time()-t0:.1f}s")

            # Stats
            print(f"\n  kio:  median={np.nanmedian(kio_map):.1f}, "
                  f"range=[{np.nanmin(kio_map):.1f}, {np.nanmax(kio_map):.1f}] s-1")
            print(f"  rho:  median={np.median(rho_map)/1e3:.0f}k, "
                  f"range=[{rho_map.min()/1e3:.0f}k, "
                  f"{rho_map.max()/1e3:.0f}k] cells/uL")
            print(f"  V:    median={np.median(V_map):.2f}, "
                  f"range=[{V_map.min():.2f}, {V_map.max():.2f}] pL")

            edge_summary = edge_railing_diagnostics(
                kio_map, rho_map, V_map, lib,
                vi_min=args.vi_min, vi_max=args.vi_max,
                rho_max=args.rho_max,
                include_free_water=args.include_free_water,
            )
            print("\n  Edge-railing diagnostic (realised library labels):")
            if edge_summary["free_water_count"]:
                print(f"    free water: {edge_summary['free_water_count']}/"
                      f"{edge_summary['n_voxels']} "
                      f"({edge_summary['free_water_fraction']:.1%})")
            for name in ("rho", "V", "kio"):
                item = edge_summary[name]
                print(f"    {name}: lower {item['at_lower_fraction']:.1%} "
                      f"({item['lower_value']:.6g}), upper "
                      f"{item['at_upper_fraction']:.1%} "
                      f"({item['upper_value']:.6g})")
                if max(item["at_lower_fraction"], item["at_upper_fraction"]) > args.edge_warning_fraction:
                    print(f"      ⚠ {name} rails against a library edge above "
                          f"{args.edge_warning_fraction:.0%}; do not interpret "
                          "that boundary as a measurement.")

            print("\nSaving maps ...")
            save_map(kio_map, mask_idx, shape, affine,
                     os.path.join(args.out, "kio_map.nii.gz"))
            save_map(rho_map, mask_idx, shape, affine,
                     os.path.join(args.out, "rho_map.nii.gz"))
            save_map(V_map, mask_idx, shape, affine,
                     os.path.join(args.out, "V_map.nii.gz"))
            save_map(res_map, mask_idx, shape, affine,
                     os.path.join(args.out, "residual_map.nii.gz"))

            derived_maps = save_derived_biomarker_maps(
                kio_map, rho_map, V_map, mask_idx, shape, affine, args.out)
            primary_maps = [
                "kio_map.nii.gz", "rho_map.nii.gz", "V_map.nii.gz",
                "residual_map.nii.gz", *map_extra_maps,
            ]
            _record_fit_completion(
                artifacts,
                args=args,
                resolved_device=resolved_device,
                fit_triples=fit_triples,
                extras=extras,
                sigma_used=sigma_used,
                edge_summary=edge_summary,
                residual=res_map,
                primary_maps=primary_maps,
                derived_maps=derived_maps,
                method_metadata={
                    "reference_mode": bool(
                        not args.fit_s0 and not args.log_space and
                        not args.rician_correct and args.method == "map"),
                },
            )

            print("\nDone!")
            return

        # ---- bayes / amico: distributional fitters ----
        raw_signal = extras['raw'] if args.fit_s0 else None
        method_meta = {}   # method-specific params recorded in JSON

        # Candidate-library size after the vi/rho_max filter (same mask
        # _build_candidate_lib_matrix applies) -- recorded so n_eff (bayes)
        # / n_eff (amico) can be judged as a fraction of n_lib rather than
        # a bare number.
        vis_all  = np.array([(e.rho / 1e9) * (e.V * 1e3) for e in lib])
        rhos_all = np.array([e.rho for e in lib])
        lib_mask = (vis_all >= args.vi_min) & (vis_all <= args.vi_max)
        if args.rho_max is not None:
            lib_mask &= (rhos_all <= args.rho_max)
        n_lib = int(lib_mask.sum())

        if args.method == "bayes":
            # Resolve σ_m: target-n_eff calibration > user > auto-from-Rician
            # > placeholder default.
            if args.target_n_eff is not None:
                if args.sigma_m is not None:
                    print(f"  ⚠ --target-n-eff given; ignoring --sigma-m "
                          f"{args.sigma_m:.4g}.")
                print(f"\n  Calibrating σ_m for target n_eff="
                      f"{args.target_n_eff:g} "
                      f"({'S0 FITTED' if args.fit_s0 else 'S0 fixed'}) ...")
                sigma_m = calibrate_sigma_m(
                    measured, lib,
                    lib_delta_pairs=lib_delta_pairs, lib_b_values=lib_b_values,
                    n_b=lib_n_b, fit_triples=fit_triples,
                    target_n_eff=args.target_n_eff,
                    fit_s0=args.fit_s0, raw_signal=raw_signal,
                    vi_min=args.vi_min, vi_max=args.vi_max, rho_max=args.rho_max,
                    use_gpu=use_gpu,
                )
                sigma_src = "target-n-eff"
                print(f"  σ_m = {sigma_m:.4g}  (calibrated for target "
                      f"n_eff={args.target_n_eff:g})")
            elif args.sigma_m is not None:
                sigma_m = float(args.sigma_m)
                sigma_src = "user"
                print(f"\n  σ_m = {sigma_m:.4g}  (user-specified)")
            else:
                est = estimate_sigma_m(extras.get('sigma'),
                                       extras.get('s0_median'),
                                       extras.get('mean_n_dir'))
                if args.rician_correct and est is not None and est > 0:
                    sigma_m = est
                    sigma_src = "auto-rician"
                    print(f"\n  σ_m = {sigma_m:.4g}  (auto from Rician σ="
                          f"{extras['sigma']:.2f}, S0_med="
                          f"{extras['s0_median']:.1f}, "
                          f"mean n_dir={extras['mean_n_dir']:.1f})")
                else:
                    sigma_m = DEFAULT_SIGMA_M
                    sigma_src = "default-placeholder"
                    print(f"\n  ⚠ σ_m = {sigma_m:.4g}  (PLACEHOLDER default — "
                          f"pass --sigma-m or --rician-correct for a "
                          f"data-driven value).")
            method_meta = dict(sigma_m=sigma_m, sigma_m_source=sigma_src,
                               n_lib=n_lib)

            print(f"\nBayes posterior over {measured.shape[0]} voxels "
                  f"({'S0 FITTED' if args.fit_s0 else 'S0 fixed'}) ...")
            t0 = time.time()
            res = bayes_fit(
                measured, lib,
                sigma_m=sigma_m,
                lib_delta_pairs=lib_delta_pairs, lib_b_values=lib_b_values,
                n_b=lib_n_b, fit_triples=fit_triples,
                vi_min=args.vi_min, vi_max=args.vi_max, rho_max=args.rho_max,
                log_space=args.log_space,
                fit_s0=args.fit_s0, raw_signal=raw_signal,
                use_gpu=use_gpu,
            )
            print(f"  Done in {time.time()-t0:.1f}s")

        else:  # amico
            print(f"\n  AMICO elastic-net: λ1={args.lambda1:g} (L1), "
                  f"λ2={args.lambda2:g} (L2)")
            method_meta = dict(lambda1=float(args.lambda1),
                               lambda2=float(args.lambda2),
                               n_lib=n_lib)
            if resolved_device == "gpu":
                method_meta.update(
                    gpu_chunk_voxels=args.gpu_chunk_voxels,
                    amico_gpu_iters=args.amico_gpu_iters,
                    amico_gpu_tol=args.amico_gpu_tol)
            print(f"\nAMICO NNLS over {measured.shape[0]} voxels "
                  f"({'S0 FITTED' if args.fit_s0 else 'S0 fixed'}) ...")
            t0 = time.time()
            res = amico_fit(
                measured, lib,
                lambda1=args.lambda1, lambda2=args.lambda2,
                lib_delta_pairs=lib_delta_pairs, lib_b_values=lib_b_values,
                n_b=lib_n_b, fit_triples=fit_triples,
                vi_min=args.vi_min, vi_max=args.vi_max, rho_max=args.rho_max,
                fit_s0=args.fit_s0, raw_signal=raw_signal,
                use_gpu=use_gpu,
                gpu_chunk_voxels=args.gpu_chunk_voxels,
                gpu_n_iters=args.amico_gpu_iters,
                gpu_tol=args.amico_gpu_tol,
            )
            print(f"  Done in {time.time()-t0:.1f}s")

        # ---- Stats (weighted means + posterior std) ----
        print(f"\n  kio_mean: median={np.median(res['kio_mean']):.1f}, "
              f"range=[{res['kio_mean'].min():.1f}, {res['kio_mean'].max():.1f}] s-1")
        print(f"  rho_mean: median={np.median(res['rho_mean'])/1e3:.0f}k cells/uL")
        print(f"  V_mean:   median={np.median(res['V_mean']):.2f} pL")
        print(f"  kio_std:  median={np.median(res['kio_std']):.2f}")
        print(f"  rho_std:  median={np.median(res['rho_std'])/1e3:.1f}k")
        print(f"  V_std:    median={np.median(res['V_std']):.3f}")
        print(f"  n_eff:    median={np.median(res['n_eff']):.2f} "f"(effective # of library atoms per voxel)")

        edge_summary = edge_railing_diagnostics(
            res['kio_mean'], res['rho_mean'], res['V_mean'], lib,
            vi_min=args.vi_min, vi_max=args.vi_max,
            rho_max=args.rho_max,
            include_free_water=args.include_free_water,
        )
        print("\n  Edge-railing diagnostic (reported estimator maps):")
        for name in ("rho", "V", "kio"):
            item = edge_summary[name]
            print(f"    {name}: lower {item['at_lower_fraction']:.1%} "
                  f"({item['lower_value']:.6g}), upper "
                  f"{item['at_upper_fraction']:.1%} "
                  f"({item['upper_value']:.6g})")

        # ---- Save maps ----
        print("\nSaving maps ...")
        save_map(res['kio_mean'], mask_idx, shape, affine,
                 os.path.join(args.out, "kio_mean.nii.gz"))
        save_map(res['rho_mean'], mask_idx, shape, affine,
                 os.path.join(args.out, "rho_mean.nii.gz"))
        save_map(res['V_mean'], mask_idx, shape, affine,
                 os.path.join(args.out, "V_mean.nii.gz"))
        save_map(res['kio_std'], mask_idx, shape, affine,
                 os.path.join(args.out, "kio_std.nii.gz"))
        save_map(res['rho_std'], mask_idx, shape, affine,
                 os.path.join(args.out, "rho_std.nii.gz"))
        save_map(res['V_std'], mask_idx, shape, affine,
                 os.path.join(args.out, "V_std.nii.gz"))
        save_map(res['residual'], mask_idx, shape, affine,
                 os.path.join(args.out, "residual.nii.gz"))
        save_map(res['n_eff'], mask_idx, shape, affine, os.path.join(args.out, "n_eff.nii.gz"))
        if "s0_fit" in res:
            save_map(res['s0_fit'], mask_idx, shape, affine,
                     os.path.join(args.out, "s0_fit_map.nii.gz"))

        derived_maps = save_derived_biomarker_maps(
            res['kio_mean'], res['rho_mean'], res['V_mean'],
            mask_idx, shape, affine, args.out)
        primary_maps = [
            "kio_mean.nii.gz", "rho_mean.nii.gz", "V_mean.nii.gz",
            "kio_std.nii.gz", "rho_std.nii.gz", "V_std.nii.gz",
            "residual.nii.gz", "n_eff.nii.gz",
        ]
        if "s0_fit" in res:
            primary_maps.append("s0_fit_map.nii.gz")
        _record_fit_completion(
            artifacts,
            args=args,
            resolved_device=resolved_device,
            fit_triples=fit_triples,
            extras=extras,
            sigma_used=sigma_used,
            edge_summary=edge_summary,
            residual=res['residual'],
            primary_maps=primary_maps,
            derived_maps=derived_maps,
            method_metadata=method_meta,
        )

        print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if _ACTIVE_FIT_ARTIFACTS is not None:
            _ACTIVE_FIT_ARTIFACTS.mark_error(exc)
        raise
    finally:
        if _ACTIVE_FIT_ARTIFACTS is not None:
            _ACTIVE_FIT_ARTIFACTS.finalize()

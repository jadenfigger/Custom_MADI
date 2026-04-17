"""Persistent workspace + output-profile models.

The workspace lives by default in ``~/.madi_viewer/`` as:
    workspace.json         — index of profiles + app prefs
    profiles/<uuid>.json   — one file per output profile

Profiles are stored with absolute paths so the app is robust when opened
from different working directories.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def default_workspace_dir() -> Path:
    """Return the default workspace directory, creating it if missing."""
    env = os.environ.get("MADI_VIEWER_WORKSPACE")
    base = Path(env) if env else Path.home() / ".madi_viewer"
    (base / "profiles").mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------
# OutputProfile
# ---------------------------------------------------------------------

@dataclass
class OutputProfile:
    """A MADI fitting configuration + its output directory.

    Covers both existing fits ("point me at this folder") and
    not-yet-run fits ("run this fit when I ask").
    """

    # Identity
    id:          str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name:        str = "Untitled profile"
    description: str = ""

    # Data + library
    library_path: str = ""
    mask_path:    str = ""
    output_dir:   str = ""
    # list of [delta_ms, path] pairs (JSON-friendly)
    inputs:       list = field(default_factory=list)

    # Fitting configuration (mirrors scripts/fit_data.py flags)
    rician_correct: bool = False
    noise_sigma:    Optional[float] = None
    avg_s0:         bool = False
    fit_s0:         bool = False
    log_space:      bool = False
    vi_min:         float = 0.5
    vi_max:         float = 0.95
    rho_max:        Optional[float] = None

    # Status
    fitted:    bool = False     # maps present on disk
    last_fit_at: Optional[str] = None

    # UI defaults
    color:     str = "#1d4ed8"   # used in compare plots
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # --- Convenience ---
    def delta_ms_list(self) -> list[float]:
        return sorted(float(d) for d, _ in self.inputs)

    def input_dict(self) -> dict[float, str]:
        return {float(d): p for d, p in self.inputs}

    def map_path(self, name: str) -> str:
        return str(Path(self.output_dir) / f"{name}_map.nii.gz")

    def has_map(self, name: str) -> bool:
        return Path(self.map_path(name)).exists()

    def detect_fitted(self) -> bool:
        """Return True if at least kio_map + rho_map + V_map exist on disk."""
        required = ["kio", "rho", "V"]
        return all(self.has_map(n) for n in required)

    def required_paths_missing(self) -> list[str]:
        miss = []
        if not self.library_path or not Path(self.library_path).exists():
            miss.append(f"library: {self.library_path or '(empty)'}")
        if not self.mask_path or not Path(self.mask_path).exists():
            miss.append(f"mask: {self.mask_path or '(empty)'}")
        for d, p in self.inputs:
            if not p or not Path(p).exists():
                miss.append(f"Δ={d}: {p or '(empty)'}")
        return miss

    # --- Serialization ---
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutputProfile":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in allowed})


# ---------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------

@dataclass
class Workspace:
    """App-wide state. Persisted to ``<workspace_dir>/workspace.json``."""

    workspace_dir: str
    profile_ids:   list[str] = field(default_factory=list)

    # Global visualization prefs (shared default for new tabs)
    default_map:     str = "kio"
    default_axis:    int = 2
    default_top_n:   int = 10
    default_sort:    str = "lin"    # "lin" | "log" | "kio" | "rho" | "V"
    default_log_y:   bool = False
    default_margin:  float = 1.5
    font_size:       int = 9

    # Window geometry (base64-encoded Qt bytes)
    window_geometry: Optional[str] = None
    window_state:    Optional[str] = None

    # Last session
    open_profile_ids:   list[str] = field(default_factory=list)
    active_profile_id:  Optional[str] = None
    compare_profile_ids: list[str] = field(default_factory=list)

    # Cached so the index can be shown without reading every profile
    profile_index: dict = field(default_factory=dict)  # id -> {"name":..., "fitted":...}

    # --- Disk layout ---
    @property
    def ws_path(self) -> Path:
        return Path(self.workspace_dir)

    @property
    def profile_dir(self) -> Path:
        d = self.ws_path / "profiles"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def json_path(self) -> Path:
        return self.ws_path / "workspace.json"

    # --- Persist ---
    def save(self) -> None:
        self.ws_path.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, workspace_dir: Optional[Path] = None) -> "Workspace":
        base = Path(workspace_dir) if workspace_dir else default_workspace_dir()
        jpath = base / "workspace.json"
        if not jpath.exists():
            ws = cls(workspace_dir=str(base))
            ws.save()
            return ws
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["workspace_dir"] = str(base)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    # --- Profile CRUD ---
    def list_profiles(self) -> list[OutputProfile]:
        profs = []
        for pid in self.profile_ids:
            try:
                profs.append(self.load_profile(pid))
            except FileNotFoundError:
                pass
        return profs

    def _profile_path(self, pid: str) -> Path:
        return self.profile_dir / f"{pid}.json"

    def load_profile(self, pid: str) -> OutputProfile:
        p = self._profile_path(pid)
        if not p.exists():
            raise FileNotFoundError(f"Profile not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return OutputProfile.from_dict(json.load(f))

    def save_profile(self, profile: OutputProfile) -> None:
        profile.modified_at = datetime.utcnow().isoformat()
        profile.fitted = profile.detect_fitted()
        with open(self._profile_path(profile.id), "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
        if profile.id not in self.profile_ids:
            self.profile_ids.append(profile.id)
        self.profile_index[profile.id] = {
            "name": profile.name,
            "fitted": profile.fitted,
            "modified_at": profile.modified_at,
        }
        self.save()

    def delete_profile(self, pid: str) -> None:
        p = self._profile_path(pid)
        if p.exists():
            p.unlink()
        if pid in self.profile_ids:
            self.profile_ids.remove(pid)
        if pid in self.open_profile_ids:
            self.open_profile_ids.remove(pid)
        if pid in self.compare_profile_ids:
            self.compare_profile_ids.remove(pid)
        if self.active_profile_id == pid:
            self.active_profile_id = None
        self.profile_index.pop(pid, None)
        self.save()

    def duplicate_profile(self, pid: str) -> OutputProfile:
        src = self.load_profile(pid)
        copy = OutputProfile.from_dict(src.to_dict())
        copy.id = uuid.uuid4().hex[:12]
        copy.name = f"{src.name} (copy)"
        copy.created_at = datetime.utcnow().isoformat()
        copy.modified_at = copy.created_at
        self.save_profile(copy)
        return copy


# ---------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------

def get_workspace() -> Workspace:
    """Return the workspace singleton-style; a plain loader is enough here."""
    return Workspace.load()

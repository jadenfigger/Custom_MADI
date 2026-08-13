#!/usr/bin/env python3
"""Validate the declared v5 production-grid stencil probe before launch.

The declaration is deliberately resolved against ``make_remediation_log_grid``
rather than trusting its serialized decimal coordinates.  This gives the
launcher and the post-build analysis one canonical source for the cross
topology while leaving the simulator and library schema untouched.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from madi.library import make_remediation_log_grid


SCHEMA = "madi-v5-stencil-probe-entry-subset-v1"
KIO_VALUES = (19.0, 20.0, 21.0)
RHO_NODES = 64
V_NODES = 64
VI_MIN = 0.40
VI_MAX = 0.99
TARGET_VI = math.sqrt(VI_MIN * VI_MAX)


@dataclass(frozen=True)
class ProbeDefinition:
    """Canonicalized declaration used by the launcher and artifact reader."""

    path: Path
    declaration: dict[str, Any]
    rhos: np.ndarray
    Vs: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    kio_values: tuple[float, ...]
    center: tuple[int, int]
    rho_step: float
    V_step: float

    @property
    def expected_cellular_entries(self) -> int:
        return len(self.pairs) * len(self.kio_values)


def _fail(message: str) -> None:
    raise ValueError(f"invalid v5 stencil-probe declaration: {message}")


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=2.0e-12, abs_tol=2.0e-12):
        _fail(f"{name}={actual:.17g}, expected canonical value {expected:.17g}")


def _canonical_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = make_remediation_log_grid()
    if len(grid.rhos) != RHO_NODES or len(grid.Vs) != V_NODES:
        _fail("make_remediation_log_grid no longer returns the mandated 64 x 64 grid")
    return np.asarray(grid.rhos, dtype=float), np.asarray(grid.Vs, dtype=float), np.asarray(grid.kios, dtype=float)


def _in_mask(rho: float, volume: float) -> bool:
    vi = rho * volume * 1.0e-6
    return VI_MIN <= vi <= VI_MAX


def _expected_center(rhos: np.ndarray, volumes: np.ndarray) -> tuple[int, int]:
    candidates = [
        (abs(math.log((rho * volume * 1.0e-6) / TARGET_VI)), i, j)
        for i, rho in enumerate(rhos)
        for j, volume in enumerate(volumes)
        if _in_mask(float(rho), float(volume))
    ]
    _, i, j = min(candidates)
    return int(i), int(j)


def _line_support(
    rhos: np.ndarray,
    volumes: np.ndarray,
    center: tuple[int, int],
    axis: str,
) -> tuple[list[int], list[int]]:
    i0, j0 = center
    if axis == "rho":
        indices = [i for i in range(RHO_NODES) if _in_mask(float(rhos[i]), float(volumes[j0]))]
        half_widths = [
            k for k in range(1, min(i0, RHO_NODES - 1 - i0) + 1)
            if i0 - k in indices and i0 + k in indices
        ]
    elif axis == "V":
        indices = [j for j in range(V_NODES) if _in_mask(float(rhos[i0]), float(volumes[j]))]
        half_widths = [
            k for k in range(1, min(j0, V_NODES - 1 - j0) + 1)
            if j0 - k in indices and j0 + k in indices
        ]
    else:
        _fail(f"unknown line axis {axis!r}")
    return indices, half_widths


def load_probe_definition(path: str | Path) -> ProbeDefinition:
    """Read and fully validate the declared production-grid cross.

    In particular, coordinate values and k_io nodes must resolve to canonical
    retained P0 grid entries.  A malformed or out-of-grid JSON therefore
    fails before the expensive GPU launcher can build an entry.
    """
    declaration_path = Path(path)
    try:
        with declaration_path.open("r", encoding="utf-8") as handle:
            declaration = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {declaration_path}: {exc}")
    if not isinstance(declaration, dict):
        _fail("top-level value must be an object")
    if declaration.get("schema") != SCHEMA:
        _fail(f"schema must be {SCHEMA!r}")

    rhos, volumes, canonical_kios = _canonical_grid()
    parent = declaration.get("parent_grid")
    if not isinstance(parent, dict):
        _fail("parent_grid must be an object")
    expected_parent = {
        "rho_nodes": RHO_NODES,
        "V_nodes": V_NODES,
        "rho_min": float(rhos[0]),
        "rho_max": float(rhos[-1]),
        "V_min": float(volumes[0]),
        "V_max": float(volumes[-1]),
        "vi_min": VI_MIN,
        "vi_max": VI_MAX,
        "rho_log_step_natural": float(math.log(rhos[1] / rhos[0])),
        "V_log_step_natural": float(math.log(volumes[1] / volumes[0])),
    }
    for key, expected in expected_parent.items():
        if key not in parent:
            _fail(f"parent_grid is missing {key!r}")
        if isinstance(expected, int):
            if int(parent[key]) != expected:
                _fail(f"parent_grid[{key!r}]={parent[key]!r}, expected {expected}")
        else:
            _require_close(f"parent_grid[{key!r}]", float(parent[key]), expected)

    raw_kios = declaration.get("kio_values_s_inv")
    if not isinstance(raw_kios, list):
        _fail("kio_values_s_inv must be a list")
    kio_values = tuple(float(value) for value in raw_kios)
    if kio_values != KIO_VALUES:
        _fail(f"kio_values_s_inv={kio_values!r}, expected {KIO_VALUES!r}")
    if any(not np.any(np.isclose(value, canonical_kios, rtol=0.0, atol=1.0e-12)) for value in kio_values):
        _fail("kio_values_s_inv contains a coordinate outside the canonical production k_io grid")

    raw_pairs = declaration.get("cellular_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        _fail("cellular_pairs must be a non-empty list")
    pairs: list[tuple[int, int]] = []
    for position, pair in enumerate(raw_pairs):
        if not isinstance(pair, dict):
            _fail(f"cellular_pairs[{position}] must be an object")
        try:
            i = int(pair["rho_index"])
            j = int(pair["V_index"])
            rho = float(pair["rho"])
            volume = float(pair["V"])
        except (KeyError, TypeError, ValueError) as exc:
            _fail(f"cellular_pairs[{position}] lacks a valid index/rho/V: {exc}")
        if not (0 <= i < RHO_NODES and 0 <= j < V_NODES):
            _fail(f"cellular_pairs[{position}] indices ({i}, {j}) are outside the 64 x 64 grid")
        _require_close(f"cellular_pairs[{position}].rho", rho, float(rhos[i]))
        _require_close(f"cellular_pairs[{position}].V", volume, float(volumes[j]))
        vi = rho * volume * 1.0e-6
        if not _in_mask(rho, volume):
            _fail(f"cellular_pairs[{position}] vi={vi:.9g} is outside [0.40, 0.99]")
        if "vi" in pair:
            _require_close(f"cellular_pairs[{position}].vi", float(pair["vi"]), vi)
        pairs.append((i, j))
    if len(set(pairs)) != len(pairs):
        _fail("cellular_pairs contains a duplicate (rho_index, V_index)")

    selection = declaration.get("selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("center"), dict):
        _fail("selection.center must be an object")
    center_obj = selection["center"]
    try:
        center = (int(center_obj["rho_index"]), int(center_obj["V_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"selection.center lacks valid indices: {exc}")
    if center != _expected_center(rhos, volumes):
        _fail(
            f"selection.center={center!r}, expected closest retained geometric-band centre "
            f"{_expected_center(rhos, volumes)!r}"
        )
    i0, j0 = center
    for key, expected in (
        ("rho", float(rhos[i0])),
        ("V", float(volumes[j0])),
        ("vi", float(rhos[i0] * volumes[j0] * 1.0e-6)),
    ):
        _require_close(f"selection.center.{key}", float(center_obj[key]), expected)
    _require_close("selection.target_vi", float(selection.get("target_vi")), TARGET_VI)

    rho_indices, rho_widths = _line_support(rhos, volumes, center, "rho")
    V_indices, V_widths = _line_support(rhos, volumes, center, "V")
    expected_rho_line = [i for i in range(i0 - 4, i0 + 5) if i in rho_indices]
    expected_V_line = [j for j in range(j0 - 2, j0 + 3) if j in V_indices]
    if selection.get("rho_line_indices") != expected_rho_line:
        _fail(f"rho_line_indices must be {expected_rho_line!r}")
    if selection.get("V_line_indices") != expected_V_line:
        _fail(f"V_line_indices must be {expected_V_line!r}")
    expected_pairs = {(i, j0) for i in expected_rho_line}
    expected_pairs.update((i0, j) for j in expected_V_line)
    if set(pairs) != expected_pairs:
        _fail("cellular_pairs is not exactly the declared rho/V cross")
    expected_widths = {"rho": rho_widths, "V": V_widths, "k_io": [1]}
    if selection.get("supported_half_widths") != expected_widths:
        _fail(f"supported_half_widths must be {expected_widths!r}")

    if bool(declaration.get("include_free_water", False)):
        _fail("the stencil probe must not include the free-water atom")
    expected_entries = len(pairs) * len(kio_values)
    if int(declaration.get("expected_cellular_entries", -1)) != expected_entries:
        _fail(f"expected_cellular_entries must be {expected_entries}")
    if int(declaration.get("expected_total_entries", -1)) != expected_entries:
        _fail(f"expected_total_entries must be {expected_entries}")

    return ProbeDefinition(
        path=declaration_path.resolve(),
        declaration=declaration,
        rhos=rhos,
        Vs=volumes,
        pairs=tuple(pairs),
        kio_values=kio_values,
        center=center,
        rho_step=float(math.log(rhos[1] / rhos[0])),
        V_step=float(math.log(volumes[1] / volumes[0])),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declaration",
        default="data/madi_v5_stencil_probe_entry_subset.json",
        help="declared production-grid stencil cross",
    )
    args = parser.parse_args(argv)
    try:
        definition = load_probe_definition(args.declaration)
    except ValueError as exc:
        print(f"ABORT: {exc}")
        return 2
    i0, j0 = definition.center
    centre_vi = definition.rhos[i0] * definition.Vs[j0] * 1.0e-6
    print(json.dumps({
        "status": "valid",
        "declaration": str(definition.path),
        "center_indices": [i0, j0],
        "center_rho": float(definition.rhos[i0]),
        "center_V": float(definition.Vs[j0]),
        "center_vi": float(centre_vi),
        "rho_log_step_natural": definition.rho_step,
        "V_log_step_natural": definition.V_step,
        "cellular_pairs": len(definition.pairs),
        "cellular_entries": definition.expected_cellular_entries,
        "kio_values_s_inv": list(definition.kio_values),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

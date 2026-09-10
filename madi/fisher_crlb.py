"""Phase-0/1 Fisher-information utilities for v5 MADI libraries.

This module deliberately contains no protocol ranking or CRLB map code.  It
validates an artifact, builds central finite-difference fields, and exposes
the noise/feasibility and Fisher primitives needed by later phases.

It also carries the analysis-domain contract (`ColumnDomain`, `require_columns`)
that keeps the reusable model-derived substrate separate from the conditional
acquisition analyses layered on top of it.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import GAMMA_RAD
from .library import make_remediation_log_grid


PARAMETER_ORDER = ("log_rho", "log_V", "k_io")
REQUIRED_V5_ARRAYS = {
    "library_schema", "kios", "rhos", "Vs", "vectors",
    "nominal_kios", "nominal_rhos", "nominal_Vs", "is_free_water",
    "build_metadata_json", "pair_deltas", "pair_Deltas", "b_values", "n_b",
    "signal_imag", "signal_variance", "ensemble_means_subset",
    "ensemble_subset_pair_deltas", "ensemble_subset_pair_Deltas",
    "ensemble_subset_b_values", "ensemble_subset_n_b",
}


def load_preregistration() -> dict[str, Any]:
    """Load the committed Phase-0 analysis choices, never CLI ad-hoc defaults."""
    path = Path(__file__).with_name("fisher_crlb_preregistration.json")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, set[tuple[int, int]]]:
    """Return canonical axes and the retained `(rho_index, V_index)` mask."""
    grid = make_remediation_log_grid()
    rhos, volumes, kios = map(lambda x: np.asarray(x, dtype=float),
                              (grid.rhos, grid.Vs, grid.kios))
    retained = {
        (ir, iv) for ir, rho in enumerate(rhos) for iv, volume in enumerate(volumes)
        if grid.vi_min <= rho * volume * 1e-6 <= grid.vi_max
    }
    return rhos, volumes, kios, retained


def _axis_index(values: np.ndarray, value: float, label: str) -> int:
    matches = np.flatnonzero(np.isclose(values, value, rtol=1e-12, atol=1e-12))
    if matches.size != 1:
        raise ValueError(f"{label}={value!r} is not a unique canonical node")
    return int(matches[0])


def _json(value: np.ndarray) -> dict:
    return json.loads(str(value))


def read_npz_rows(path: str | Path, array: str, rows: Iterable[int]) -> np.ndarray:
    """Read selected rows of an uncompressed NPZ ``.npy`` member.

    ``np.load(...)[array]`` materialises all 4.4 GiB of `vectors` even when a
    gate needs the first free-water row only.  Production artifacts deliberately
    use ZIP_STORED members, so this small reader avoids that needless local RAM
    and I/O cost.  It supports ordinary C-order rank-2 arrays only.
    """
    with zipfile.ZipFile(path) as archive, archive.open(f"{array}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        if len(shape) != 2 or fortran:
            raise ValueError(f"{array} must be a C-order rank-2 array, got {shape}, fortran={fortran}")
        row_bytes = int(shape[1]) * dtype.itemsize
        answer = []
        for row in rows:
            if not 0 <= int(row) < shape[0]:
                raise IndexError(f"row {row} outside {array} shape {shape}")
            # The members are stored, but seek semantics vary by Python/zip
            # version.  Reopen and skip deterministically for each requested
            # row; Phase-0 uses this for a handful of rows, not a bulk scan.
            with archive.open(f"{array}.npy") as again:
                version2 = np.lib.format.read_magic(again)
                if version2 == (1, 0):
                    np.lib.format.read_array_header_1_0(again)
                else:
                    np.lib.format.read_array_header_2_0(again)
                again.seek(int(row) * row_bytes, 1)
                raw = again.read(row_bytes)
            if len(raw) != row_bytes:
                raise OSError(f"short read for {array} row {row}")
            answer.append(np.frombuffer(raw, dtype=dtype).copy())
    return np.stack(answer, axis=0)


def npz_array_info(path: str | Path, array: str) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Return an NPZ member's NPY shape/order/dtype without materialising it."""
    with zipfile.ZipFile(path) as archive, archive.open(f"{array}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version} for {array}")
    return tuple(shape), bool(fortran), np.dtype(dtype)


def iter_npz_array_chunks(path: str | Path, array: str, rows_per_chunk: int = 16):
    """Yield C-order NPY member chunks along axis 0, including compressed NPZs."""
    shape, fortran, dtype = npz_array_info(path, array)
    if fortran or not shape:
        raise ValueError(f"{array} must be a non-scalar C-order NPY member")
    trailing = int(np.prod(shape[1:], dtype=np.int64))
    row_bytes = trailing * dtype.itemsize
    with zipfile.ZipFile(path) as archive, archive.open(f"{array}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(stream)
        else:
            np.lib.format.read_array_header_2_0(stream)
        for start in range(0, shape[0], rows_per_chunk):
            count = min(rows_per_chunk, shape[0] - start)
            raw = stream.read(count * row_bytes)
            if len(raw) != count * row_bytes:
                raise OSError(f"short sequential read for {array} rows {start}:{start + count}")
            yield start, np.frombuffer(raw, dtype=dtype).reshape((count, *shape[1:]))


@dataclass(frozen=True)
class GridManifest:
    present: set[tuple[int, int]]
    missing: set[tuple[int, int]]
    duplicate: set[tuple[int, int]]
    extras: list[dict[str, float]]
    group_entries: dict[tuple[int, int], np.ndarray]

    @property
    def grid_complete(self) -> bool:
        return not self.missing and not self.duplicate and not self.extras

    def as_dict(self) -> dict[str, Any]:
        rhos, volumes, _, _ = canonical_grid()
        def item(pair: tuple[int, int]) -> dict[str, Any]:
            ir, iv = pair
            return {"rho_index": ir, "V_index": iv,
                    "rho": float(rhos[ir]), "V": float(volumes[iv]),
                    "vi": float(rhos[ir] * volumes[iv] * 1e-6)}
        return {
            "grid_complete": self.grid_complete,
            "present_group_count": len(self.present),
            "missing_group_count": len(self.missing),
            "missing_groups": [item(x) for x in sorted(self.missing)],
            "duplicate_groups": [item(x) for x in sorted(self.duplicate)],
            "extra_groups": self.extras,
        }


def build_grid_manifest(nominal_rhos: np.ndarray, nominal_volumes: np.ndarray,
                        free: np.ndarray) -> GridManifest:
    """Map v5 entries to canonical nodes using *nominal* coordinates.

    v5 `rhos`/`Vs` are finite-geometry provenance and intentionally differ
    from the requested grid.  Matching those realized values literally to the
    canonical grid would falsely report every group as missing.
    """
    rhos, volumes, _, retained = canonical_grid()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    extras: list[dict[str, float]] = []
    for entry, (rho, volume, is_free) in enumerate(zip(nominal_rhos, nominal_volumes, free)):
        if is_free:
            continue
        try:
            pair = (_axis_index(rhos, float(rho), "nominal rho"),
                    _axis_index(volumes, float(volume), "nominal V"))
        except ValueError:
            extras.append({"entry": entry, "nominal_rho": float(rho), "nominal_V": float(volume)})
            continue
        if pair not in retained:
            extras.append({"entry": entry, "nominal_rho": float(rho), "nominal_V": float(volume)})
        else:
            groups[pair].append(entry)
    present = set(groups)
    duplicate = {pair for pair, values in groups.items() if len(values) != 51}
    return GridManifest(present=present, missing=retained - present, duplicate=duplicate,
                        extras=extras,
                        group_entries={key: np.asarray(value, dtype=int) for key, value in groups.items()})


def invalidated_stencils(manifest: GridManifest, widths: dict[str, Iterable[int]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Report centres invalidated specifically by a missing group.

    Natural mask boundaries are not errors and are deliberately excluded.  A
    listed centre would have a valid canonical central stencil after the named
    missing group is restored, making this the concrete re-run diff set.
    """
    rhos, volumes, _, retained = canonical_grid()
    answer: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for axis, values in widths.items():
        answer[axis] = {}
        for width in values:
            invalid = []
            for ir, iv in sorted(retained):
                if axis == "rho":
                    endpoints = ((ir - width, iv), (ir + width, iv))
                elif axis == "V":
                    endpoints = ((ir, iv - width), (ir, iv + width))
                else:
                    # k_io stencils live wholly inside one rho/V group.  A
                    # missing group therefore removes all its k_io centres.
                    endpoints = ()
                canonical_stencil = {(ir, iv), *endpoints}
                missing = canonical_stencil.intersection(manifest.missing)
                if canonical_stencil.issubset(retained) and missing:
                    invalid.append({"rho_index": ir, "V_index": iv,
                                    "rho": float(rhos[ir]), "V": float(volumes[iv]),
                                    "missing_stencil_groups": [
                                        {"rho_index": a, "V_index": b} for a, b in sorted(missing)
                                    ]})
            answer[axis][str(width)] = invalid
    return answer


def artifact_manifest(data: np.lib.npyio.NpzFile) -> GridManifest:
    return build_grid_manifest(np.asarray(data["nominal_rhos"]),
                               np.asarray(data["nominal_Vs"]),
                               np.asarray(data["is_free_water"], dtype=bool))


def incomplete_banner(manifest: GridManifest) -> str:
    if manifest.grid_complete:
        return "GRID COMPLETE"
    groups = ", ".join(
        f"(rho_index={ir}, V_index={iv})" for ir, iv in sorted(manifest.missing)
    )
    return f"INCOMPLETE GRID — missing groups: {groups}"


def assert_safe_output(path: Path, manifest: GridManifest) -> None:
    """Prevent an incomplete artifact from masquerading as a final output."""
    if manifest.grid_complete:
        return
    text = str(path).lower()
    if "partial" not in text and "incomplete" not in text and "smoke" not in text:
        raise ValueError(
            f"{incomplete_banner(manifest)}. Incomplete outputs must use an output "
            "directory or filename containing 'partial', 'incomplete', or 'smoke'."
        )


def metadata_crn_contract(build: dict[str, Any]) -> bool:
    contract = ((build.get("uncertainty") or {})
                .get("ensemble_index_ordering_contract") or {})
    return (contract.get("axis_position_in_ensemble_means_subset") == 1
            and contract.get("index_values") == "0..n_ensembles-1"
            and contract.get("same_order_across_entries") is True
            and contract.get("independent_of") == ["rho", "V", "k_io"])


def verify_v5_artifact(path: str | Path) -> dict[str, Any]:
    """Validate Phase-0.1 requirements and return a JSON-safe report.

    This intentionally reads large arrays one at a time.  It is a validator,
    not a replacement for the final production acceptance validator.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing_arrays = sorted(REQUIRED_V5_ARRAYS.difference(data.files))
        manifest = artifact_manifest(data)
        result: dict[str, Any] = {
            "schema": "madi-fisher-shard-verification-v1", "artifact": str(path),
            **manifest.as_dict(), "errors": [], "groups": [],
        }
        if missing_arrays:
            result["errors"].append(f"missing required arrays: {missing_arrays}")
            result["pass"] = False
            return result
        try:
            build = _json(data["build_metadata_json"])
        except Exception as exc:
            result["errors"].append(f"invalid build metadata: {exc}")
            build = {}
        n_entries = len(data["kios"])
        n_b = int(data["n_b"])
        n_columns = len(data["pair_deltas"]) * n_b
        # NPZ members cannot be memory-mapped. Scan them row-wise so the full
        # production validator fits an ordinary workstation even for a
        # compressed 15-GiB artifact.
        vector_info = npz_array_info(path, "vectors")
        imag_info = npz_array_info(path, "signal_imag")
        variance_info = npz_array_info(path, "signal_variance")
        subset_info = npz_array_info(path, "ensemble_means_subset")
        n_ensembles = subset_info[0][1] if len(subset_info[0]) == 3 else 0
        required_shapes = {
            "vectors": ((n_entries, n_columns), np.dtype("float64")),
            "signal_imag": ((n_entries, n_columns), np.dtype("float32")),
            "signal_variance": ((n_entries, n_columns), np.dtype("float32")),
            "ensemble_means_subset": ((n_entries, n_ensembles,
                                         len(data["ensemble_subset_pair_deltas"]) * int(data["ensemble_subset_n_b"])), np.dtype("float32")),
        }
        result["entries"] = n_entries
        result["columns"] = n_columns
        result["crn_contract_present"] = metadata_crn_contract(build)
        if not result["crn_contract_present"]:
            result["errors"].append("recorded CRN ensemble-index contract is absent")
        subset_pairs = list(zip(np.asarray(data["ensemble_subset_pair_deltas"], dtype=float),
                                np.asarray(data["ensemble_subset_pair_Deltas"], dtype=float)))
        main_pairs = list(zip(np.asarray(data["pair_deltas"], dtype=float),
                              np.asarray(data["pair_Deltas"], dtype=float)))
        subset_b = np.asarray(data["ensemble_subset_b_values"], dtype=float)
        main_b = np.asarray(data["b_values"], dtype=float)
        try:
            subset_columns = np.asarray([main_pairs.index(pair) * n_b + int(np.flatnonzero(main_b == b)[0])
                                         for pair in subset_pairs for b in subset_b], dtype=int)
        except (ValueError, IndexError) as exc:
            result["errors"].append(f"cannot resolve diagnostic subset columns: {exc}")
            subset_columns = np.empty(0, dtype=int)
        if vector_info[0] != required_shapes["vectors"][0] or vector_info[2] != required_shapes["vectors"][1]:
            result["errors"].append(f"vectors: shape={vector_info[0]}, dtype={vector_info[2]}; expected {required_shapes['vectors'][0]}, {required_shapes['vectors'][1]}")
        b0 = np.arange(0, n_columns, n_b)
        subset_ref = np.empty((n_entries, len(subset_columns)), dtype=np.float64)
        high_b = main_b[None, :] >= np.quantile(main_b, 0.8)
        high_b_columns = np.tile(high_b, (len(main_pairs), 1)).ravel()
        vector_finite = True
        b0_exact = True
        negative_high_b = 0
        minimum_high_b = math.inf
        for start, chunk in iter_npz_array_chunks(path, "vectors"):
            vector_finite &= bool(np.all(np.isfinite(chunk)))
            b0_exact &= bool(np.array_equal(chunk[:, b0], np.ones((len(chunk), len(b0)))))
            if subset_columns.size:
                subset_ref[start:start + len(chunk)] = chunk[:, subset_columns]
            high = chunk[:, high_b_columns]
            negative_high_b += int(np.count_nonzero(high < 0.0))
            minimum_high_b = min(minimum_high_b, float(np.min(high)))
        if not vector_finite:
            result["errors"].append("vectors contains non-finite values")
        if not b0_exact:
            result["errors"].append("vectors are not exactly one at every b=0 column")
        result["negative_signal_count_high_b"] = negative_high_b
        result["minimum_signal_high_b"] = minimum_high_b
        if subset_columns.size:
            if subset_info[0] != required_shapes["ensemble_means_subset"][0] or subset_info[2] != required_shapes["ensemble_means_subset"][1]:
                result["errors"].append(f"ensemble_means_subset: shape={subset_info[0]}, dtype={subset_info[2]}; expected {required_shapes['ensemble_means_subset'][0]}, {required_shapes['ensemble_means_subset'][1]}")
            subset_finite = True
            max_abs = max_rel = 0.0
            subset_close = True
            for start, chunk in iter_npz_array_chunks(path, "ensemble_means_subset"):
                subset_finite &= bool(np.all(np.isfinite(chunk)))
                mean = np.mean(chunk, axis=1, dtype=np.float64)
                reference = subset_ref[start:start + len(chunk)]
                abs_err = np.abs(mean - reference)
                max_abs = max(max_abs, float(np.max(abs_err)))
                max_rel = max(max_rel, float(np.max(abs_err / np.maximum(np.abs(reference), 1e-12))))
                subset_close &= bool(np.allclose(mean, reference, rtol=2e-6, atol=2e-6))
            if not subset_finite:
                result["errors"].append("ensemble_means_subset contains non-finite values")
            result["subset_mean_max_abs_error"] = max_abs
            result["subset_mean_max_rel_error"] = max_rel
            if not subset_close:
                result["errors"].append("ensemble subset means fail float32 reconstruction tolerance")
            del subset_ref
        if variance_info[0] != required_shapes["signal_variance"][0] or variance_info[2] != required_shapes["signal_variance"][1]:
            result["errors"].append(f"signal_variance: shape={variance_info[0]}, dtype={variance_info[2]}; expected {required_shapes['signal_variance'][0]}, {required_shapes['signal_variance'][1]}")
        variance_finite = True
        variance_negative = False
        for _, chunk in iter_npz_array_chunks(path, "signal_variance"):
            variance_finite &= bool(np.all(np.isfinite(chunk)))
            variance_negative |= bool(np.any(chunk < 0.0))
        if not variance_finite:
            result["errors"].append("signal_variance contains non-finite values")
        if variance_negative:
            result["errors"].append("signal_variance contains negative values")
        if imag_info[0] != required_shapes["signal_imag"][0] or imag_info[2] != required_shapes["signal_imag"][1]:
            result["errors"].append(f"signal_imag: shape={imag_info[0]}, dtype={imag_info[2]}; expected {required_shapes['signal_imag'][0]}, {required_shapes['signal_imag'][1]}")
        imag_finite = True
        for _, chunk in iter_npz_array_chunks(path, "signal_imag"):
            imag_finite &= bool(np.all(np.isfinite(chunk)))
        if not imag_finite:
            result["errors"].append("signal_imag contains non-finite values")
        result["independent_sampling_floor"] = float(1.0 / math.sqrt(6_000_000))
        nominal_kio = np.asarray(data["nominal_kios"], dtype=float)
        realised_kio = np.asarray(data["kios"], dtype=float)
        rhos = np.asarray(data["rhos"], dtype=float)
        volumes = np.asarray(data["Vs"], dtype=float)
        _, _, expected_kio, _ = canonical_grid()
        free = np.asarray(data["is_free_water"], dtype=bool)
        for pair in sorted(manifest.present):
            indices = manifest.group_entries[pair]
            rho_same = bool(np.array_equal(rhos[indices], np.full(len(indices), rhos[indices[0]])))
            V_same = bool(np.array_equal(volumes[indices], np.full(len(indices), volumes[indices[0]])))
            order = np.argsort(nominal_kio[indices])
            local_nominal = nominal_kio[indices][order]
            local_realised = realised_kio[indices][order]
            kio_ok = bool(np.array_equal(local_nominal, expected_kio) and len(np.unique(local_nominal)) == len(expected_kio))
            group = {"rho_index": pair[0], "V_index": pair[1], "entries": int(len(indices)),
                     "realised_rho": float(rhos[indices[0]]), "realised_V": float(volumes[indices[0]]),
                     "rho_bit_identical": rho_same, "V_bit_identical": V_same,
                     "nominal_kio_grid_complete": kio_ok,
                     "realised_kio_min": float(np.nanmin(local_realised)),
                     "realised_kio_max": float(np.nanmax(local_realised))}
            result["groups"].append(group)
            if not (rho_same and V_same and kio_ok and len(indices) == len(expected_kio)):
                result["errors"].append(f"group {pair} fails group-label or k_io coverage contract")
        configured = load_preregistration()["stencil_half_widths"]
        result["stencil_invalidated_nodes"] = invalidated_stencils(manifest, configured)
        result["pass"] = not result["errors"]
        return result


def analytic_free_water_gate(b_values: np.ndarray, *, D0_um2_ms: float = 3.0,
                             sigma_m: float = 0.02, tolerance: float = 1e-12) -> dict[str, Any]:
    """Gate A: validate derivative/Fisher/CRLB code using synthetic signals only."""
    b = np.asarray(b_values, dtype=float).ravel()
    signal = np.exp(-b * D0_um2_ms * 1e-3)
    analytic_derivative = -b * 1e-3 * signal
    # The pipeline receives this independently constructed synthetic Jacobian;
    # no stored library signal enters Gate A.
    F_expected = float(np.sum(analytic_derivative ** 2 / sigma_m ** 2))
    F_pipeline = float(fisher_matrix(analytic_derivative[:, None], sigma_m)[0, 0])
    crlb_expected = 1.0 / math.sqrt(F_expected)
    crlb_pipeline = 1.0 / math.sqrt(F_pipeline)
    derivative_error = float(np.max(np.abs(analytic_derivative - (-b * 1e-3 * signal))))
    return {
        "schema": "madi-free-water-fisher-gate-a-v1",
        "kind": "synthetic_analytic_pipeline",
        "D0_um2_ms": D0_um2_ms,
        "sigma_m": sigma_m,
        "tolerance": tolerance,
        "columns": int(len(b)),
        "derivative_max_abs_error": derivative_error,
        "fisher_abs_error": abs(F_pipeline - F_expected),
        "crlb_abs_error": abs(crlb_pipeline - crlb_expected),
        "pass": max(derivative_error, abs(F_pipeline - F_expected),
                    abs(crlb_pipeline - crlb_expected)) <= tolerance,
    }


def free_water_gate(path: str | Path, sigma_m: float = 0.02,
                    tolerance: float = 1e-12, statistical_sigma_limit: float = 4.0,
                    observed_to_nominal_se_ratio: float = 0.7) -> dict[str, Any]:
    """Run Gate A and Gate B without assuming the stored free-water row is analytic.

    Gate A is the sole blocking software gate.  Gate B is an informational
    Monte-Carlo convergence check, standardized to the declared independent
    sampling floor and, separately, the measured observed/nominal SE ratio.
    """
    with np.load(path, allow_pickle=False) as data:
        manifest = artifact_manifest(data)
        free = np.asarray(data["is_free_water"], dtype=bool)
        indices = np.flatnonzero(free)
        if len(indices) != 1:
            return {"pass": False, **manifest.as_dict(), "error": f"expected one free-water entry, found {len(indices)}"}
        build = _json(data["build_metadata_json"])
        D0 = float(build["D0_um2_ms"])
        b = np.asarray(data["b_values"], dtype=float)
        n_b = int(data["n_b"])
        observed = read_npz_rows(path, "vectors", [int(indices[0])])[0].reshape(-1, n_b)
        expected = np.exp(-b[None, :] * D0 * 1e-3)
        deviation = np.abs(observed - expected)
        max_signal_error = float(np.max(deviation))
        nominal_se = 1.0 / math.sqrt(6_000_000)
        gate_a = analytic_free_water_gate(b, D0_um2_ms=D0, sigma_m=sigma_m, tolerance=tolerance)
        gate_b = {
            "kind": "stored_monte_carlo_signal",
            "criterion": f"maximum standardized deviation < {statistical_sigma_limit:g} sigma",
            "statistical_sigma_limit": statistical_sigma_limit,
            "nominal_independent_sampling_se": nominal_se,
            "observed_to_nominal_se_ratio": observed_to_nominal_se_ratio,
            "max_signal_abs_error": max_signal_error,
            "max_standardized_deviation_nominal": max_signal_error / nominal_se,
            "max_standardized_deviation_observed": max_signal_error / (nominal_se * observed_to_nominal_se_ratio),
        }
        gate_b["pass"] = bool(gate_b["max_standardized_deviation_observed"] < statistical_sigma_limit)
        return {"schema": "madi-free-water-fisher-gates-v2", **manifest.as_dict(),
                "D0_um2_ms": D0, "gate_a": gate_a, "gate_b": gate_b,
                "pass": gate_a["pass"]}


def gradient_strength_t_per_m(delta_ms: np.ndarray, Delta_ms: np.ndarray,
                              b_s_mm2: np.ndarray) -> np.ndarray:
    b_si = np.asarray(b_s_mm2, dtype=float) * 1e6
    delta_s = np.asarray(delta_ms, dtype=float) * 1e-3
    tD_s = (np.asarray(Delta_ms, dtype=float) - np.asarray(delta_ms, dtype=float) / 3.0) * 1e-3
    answer = np.zeros(np.broadcast(b_si, delta_s, tD_s).shape, dtype=float)
    valid = (b_si > 0) & (delta_s > 0) & (tD_s > 0)
    answer[valid] = np.sqrt(b_si[valid] / ((GAMMA_RAD * delta_s[valid]) ** 2 * tD_s[valid]))
    return answer


def column_arrays(pair_deltas: np.ndarray, pair_Deltas: np.ndarray, b_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.repeat(np.asarray(pair_deltas, dtype=float), len(b_values)),
            np.repeat(np.asarray(pair_Deltas, dtype=float), len(b_values)),
            np.tile(np.asarray(b_values, dtype=float), len(pair_deltas)))


def te_noise_sigma(delta_ms: np.ndarray, Delta_ms: np.ndarray, *, sigma0: float,
                   averages: float = 1.0, T2_ms: float = 80.0, t_epi_ms: float = 0.0) -> np.ndarray:
    if sigma0 <= 0 or averages <= 0 or T2_ms <= 0:
        raise ValueError("sigma0, averages, and T2_ms must be positive")
    return sigma0 / math.sqrt(averages) * np.exp((np.asarray(delta_ms) + np.asarray(Delta_ms) + t_epi_ms) / T2_ms)


def feasibility_masks(signals: np.ndarray, pair_deltas: np.ndarray, pair_Deltas: np.ndarray,
                      b_values: np.ndarray, *, sigma0: float, averages: float = 1.0,
                      T2_ms: float = 80.0, t_epi_ms: float = 0.0,
                      trust_floor: float = 0.015, rician_snr_min: float = 3.0,
                      G_max: float = 0.08) -> dict[str, np.ndarray]:
    """Conditional acquisition masks for one declared scanner and noise model.

    Every mask here is CONDITIONAL: it depends on a gradient ceiling, a TE/T2
    noise model, an averaging allocation and a trust threshold, none of which
    the library knows about.  They are legitimate inputs to a declared
    conditional analysis and legitimate annotations on the substrate.

    They are **not** an extraction filter.  Using any of them to decide which
    stored columns get a derivative, a variance or a cache entry makes the
    modelled information unrecoverable at any other scanner setting.  See
    `ColumnDomain` and docs/fisher_domain_audit.md.
    """
    delta, Delta, b = column_arrays(pair_deltas, pair_Deltas, b_values)
    sigma = te_noise_sigma(delta, Delta, sigma0=sigma0, averages=averages, T2_ms=T2_ms, t_epi_ms=t_epi_ms)
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != len(delta):
        raise ValueError("signals must be (entries, columns) matching the supplied timing grid")
    signal_min = np.min(signals, axis=0)
    gradient = gradient_strength_t_per_m(delta, Delta, b)
    trust_per_entry = signals >= trust_floor
    rician_per_entry = signals / sigma[None, :] >= rician_snr_min
    combined_per_entry = ((gradient <= G_max)[None, :] & trust_per_entry & rician_per_entry)
    # A global minimum is retained solely as a conservative diagnostic.  It is
    # explicitly not the operative measurement mask for Fisher summation.
    masks = {
        "gradient": gradient <= G_max,
        "trust_floor_per_entry": trust_per_entry,
        "rician_per_entry": rician_per_entry,
        "combined_per_entry": combined_per_entry,
        "trust_floor_global_min_diagnostic": signal_min >= trust_floor,
        "rician_global_min_diagnostic": signal_min / sigma >= rician_snr_min,
    }
    masks["combined_global_min_diagnostic"] = (masks["gradient"]
                                                & masks["trust_floor_global_min_diagnostic"]
                                                & masks["rician_global_min_diagnostic"])
    masks["gradient_T_per_m"] = gradient
    masks["sigma"] = sigma
    return masks


# ---------------------------------------------------------------------------
# Analysis-domain contract -- added 2026-09-06
# ---------------------------------------------------------------------------
#
# The Fisher work has two layers and they must not be confused.
#
#   REUSABLE SUBSTRATE.  Signal, finite-difference derivatives, derivative
#   Monte-Carlo variance/covariance, truncation-bias diagnostics.  These follow
#   from the validated library alone.  Nothing about a scanner, a TE, a T2, an
#   SNR, an averaging budget, or a trust threshold enters them, so their domain
#   is the stored acquisition grid: every `(delta, Delta, b)` column the library
#   holds.
#
#   CONDITIONAL ANALYSIS.  Gradient feasibility at a chosen `G_max`, TE/T2
#   noise, Rician validity at a chosen averaging, the `S/S0` trust floor,
#   budgets, protocol optimisation.  Each is a statement about a declared
#   acquisition, is legitimate, and belongs at evaluation time where it is
#   named in the result.
#
# A conditional quantity may be calculated, annotated, reported and stratified
# on the substrate.  It must never decide whether a stored column is extracted
# or cached, because that makes the modelled information unrecoverable without
# regenerating the substrate.  `ColumnDomain` is the machine-checkable form of
# that rule: every cache and every report declares the domain it represents,
# and `require_columns` refuses to let a restricted cache be read as universal.
#
# See docs/fisher_domain_audit.md and the pre-registration block
# `analysis_domain_architecture`.

STORED_COLUMN_DOMAIN = "all_stored_columns"
LEGACY_COLUMN_DOMAIN = "legacy_undeclared"


@dataclass(frozen=True)
class ColumnDomain:
    """Which stored `(delta, Delta, b)` columns an artifact actually represents.

    `column_indices` are indices into the full stored column grid, sorted and
    unique.  `basis` names how they were chosen; `STORED_COLUMN_DOMAIN` is the
    unrestricted substrate and is the only basis that may be read as universal.
    """

    basis: str
    column_indices: np.ndarray
    full_stored_columns: int
    restriction_source: str | None = None
    restriction_note: str | None = None

    def __post_init__(self) -> None:
        indices = np.asarray(self.column_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("column_indices must be one-dimensional")
        if indices.size and (indices.min() < 0 or indices.max() >= self.full_stored_columns):
            raise ValueError("column_indices fall outside the stored column grid")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("column_indices must be strictly increasing and unique")
        object.__setattr__(self, "column_indices", indices)

    @property
    def is_complete(self) -> bool:
        """True only when the domain is the whole stored acquisition grid."""
        return len(self.column_indices) == self.full_stored_columns

    @property
    def position_of(self) -> np.ndarray:
        """Full-grid column index -> position in this domain, or -1 if absent."""
        answer = np.full(self.full_stored_columns, -1, dtype=np.int64)
        answer[self.column_indices] = np.arange(len(self.column_indices), dtype=np.int64)
        return answer

    def covers(self, wanted: Iterable[int]) -> np.ndarray:
        wanted = np.asarray(list(wanted), dtype=np.int64)
        return self.position_of[wanted] >= 0 if wanted.size else np.zeros(0, dtype=bool)

    def banner(self) -> str:
        if self.is_complete:
            return (f"COLUMN DOMAIN COMPLETE — all {self.full_stored_columns} stored "
                    f"(delta, Delta, b) columns")
        return (f"COLUMN DOMAIN RESTRICTED — {len(self.column_indices)} of "
                f"{self.full_stored_columns} stored columns, basis={self.basis!r}. "
                "This artifact is NOT a universal substrate; downstream results "
                "conditioned on columns it omits are unavailable, not zero.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "is_complete_stored_grid": bool(self.is_complete),
            "columns": int(len(self.column_indices)),
            "full_stored_columns": int(self.full_stored_columns),
            "restriction_source": self.restriction_source,
            "restriction_note": self.restriction_note,
            "declares": ("reusable model-derived substrate; no hardware, acquisition or "
                         "trust mask applied" if self.is_complete else
                         "RESTRICTED; a conditional mask was applied at extraction time"),
        }


def stored_column_domain(full_stored_columns: int) -> ColumnDomain:
    """The unrestricted substrate: every stored acquisition column."""
    return ColumnDomain(basis=STORED_COLUMN_DOMAIN,
                        column_indices=np.arange(int(full_stored_columns), dtype=np.int64),
                        full_stored_columns=int(full_stored_columns))


def read_column_domain(manifest: dict[str, Any]) -> ColumnDomain:
    """Recover the declared domain from a Phase-1 manifest, old or new.

    A manifest written before 2026-09-06 carries no declaration.  Rather than
    assume it was universal -- which is exactly the failure this contract
    exists to prevent -- it is reported as `LEGACY_COLUMN_DOMAIN`, so the
    guards below treat it as restricted unless it happens to hold every column.
    """
    block = manifest.get("column_domain")
    selection = manifest.get("column_selection", {})
    indices = np.asarray(selection.get("selected_full_column_indices", []), dtype=np.int64)
    full = int(selection.get("full_stored_columns", len(indices)))
    if block is None:
        return ColumnDomain(basis=LEGACY_COLUMN_DOMAIN, column_indices=indices,
                            full_stored_columns=full,
                            restriction_source=manifest.get("feasibility"),
                            restriction_note=("manifest predates the column-domain contract; "
                                              "its basis is not declared and is not assumed universal"))
    return ColumnDomain(basis=str(block["basis"]), column_indices=indices,
                        full_stored_columns=int(block.get("full_stored_columns", full)),
                        restriction_source=block.get("restriction_source"),
                        restriction_note=block.get("restriction_note"))


def require_columns(domain: ColumnDomain, wanted: Iterable[int], purpose: str,
                    delta: np.ndarray | None = None, Delta: np.ndarray | None = None,
                    b: np.ndarray | None = None) -> np.ndarray:
    """Positions of `wanted` inside `domain`, refusing any silent omission.

    This is the incomplete-cache guard.  Downstream code that quietly drops the
    columns a cache happens not to hold would report a conditional analysis of a
    smaller acquisition under the name of the one that was asked for, which is
    the defect this contract exists to prevent.
    """
    wanted = np.asarray(list(wanted), dtype=np.int64)
    positions = domain.position_of[wanted] if wanted.size else np.zeros(0, dtype=np.int64)
    missing = wanted[positions < 0] if wanted.size else wanted
    if missing.size:
        detail = ""
        if delta is not None and Delta is not None and b is not None:
            head = missing[:5]
            detail = "; first missing (delta, Delta, b) = " + ", ".join(
                f"({float(delta[i]):g}, {float(Delta[i]):g}, {float(b[i]):g})" for i in head)
        raise ValueError(
            f"{purpose}: {missing.size} of {wanted.size} requested stored columns are absent "
            f"from this artifact's column domain (basis={domain.basis!r}, "
            f"{len(domain.column_indices)}/{domain.full_stored_columns} columns){detail}. "
            "Rebuild the substrate over the full stored grid rather than reinterpreting "
            "a restricted cache as universal."
        )
    return positions


def gradient_feasible_columns(delta_ms: np.ndarray, Delta_ms: np.ndarray, b_s_mm2: np.ndarray,
                              G_max: float) -> np.ndarray:
    """Full-grid indices of columns achievable at a stated gradient ceiling.

    A CONDITIONAL quantity: it answers "which stored columns could this scanner
    play", not "which columns does the model describe".  It is the authoritative
    reconstruction of the hardware mask from the unrestricted substrate, so
    Phase 0.4 and every later conditional analysis call this one function.
    """
    return np.flatnonzero(gradient_strength_t_per_m(delta_ms, Delta_ms, b_s_mm2) <= float(G_max))


def fisher_matrix(J: np.ndarray, sigma: np.ndarray | float, variance: np.ndarray | None = None,
                  measurement_mask: np.ndarray | None = None) -> np.ndarray:
    """Return F=J^T Sigma^-1 J, debiasing its diagonal when variance is supplied."""
    J = np.asarray(J, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[0],))
    if measurement_mask is not None:
        keep = np.asarray(measurement_mask, dtype=bool)
        if keep.shape != (J.shape[0],):
            raise ValueError("measurement_mask must have one boolean per column")
        J, sigma = J[keep], sigma[keep]
        if variance is not None:
            variance = np.asarray(variance)[keep]
    F = (J / sigma[:, None]).T @ (J / sigma[:, None])
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        if variance.shape != J.shape:
            raise ValueError("derivative variance must have the same shape as J")
        F[np.diag_indices_from(F)] -= np.sum(variance / sigma[:, None] ** 2, axis=0)
    return F


def fisher_diagnostics(F: np.ndarray, kio_ref: float) -> dict[str, Any]:
    F = np.asarray(F, dtype=float)
    try:
        Finv = np.linalg.inv(F)
        invertible = True
    except np.linalg.LinAlgError:
        Finv = np.full_like(F, np.nan)
        invertible = False
    diag = np.diag(Finv) if invertible else np.full(3, np.nan)
    kappa = np.sqrt(np.clip(diag * np.diag(F), 0, np.inf)) if invertible else np.full(3, np.inf)
    scale = np.diag([1.0, 1.0, float(kio_ref)])
    eigvals, eigvecs = np.linalg.eigh(scale @ F @ scale)
    return {"F": F, "Finv": Finv, "crlb": np.sqrt(np.clip(diag, 0, np.inf)),
            "kappa": kappa, "F_tilde_eigenvalues": eigvals,
            "F_tilde_eigenvectors": eigvecs, "invertible": invertible}


def derivative_variance(signal_variance_minus: np.ndarray, signal_variance_plus: np.ndarray,
                        ensemble_minus: np.ndarray | None, ensemble_plus: np.ndarray | None,
                        denominator: float, n_ensembles: int) -> np.ndarray:
    """Variance of a central derivative; covariance is used only for subset columns."""
    answer = (np.asarray(signal_variance_minus, dtype=float) + np.asarray(signal_variance_plus, dtype=float)) / n_ensembles
    if ensemble_minus is not None and ensemble_plus is not None:
        if ensemble_minus.shape != ensemble_plus.shape or ensemble_minus.shape[0] != n_ensembles:
            raise ValueError("CRN ensemble-index alignment is absent or malformed")
        cov = np.sum((ensemble_minus - ensemble_minus.mean(axis=0)) *
                     (ensemble_plus - ensemble_plus.mean(axis=0)), axis=0) / (n_ensembles - 1)
        answer = answer - 2.0 * cov / n_ensembles
    return np.maximum(answer, 0.0) / denominator ** 2


# ---------------------------------------------------------------------------
# Nuisance amplitude (S0) -- adopted 2026-09-05 from the marginal-S0 handoff
# ---------------------------------------------------------------------------
#
# The Fisher primitives above treat the library's normalized S/S0 curve as the
# forward model, which silently asserts that the amplitude S0 is known exactly.
# No real acquisition knows it exactly.  Modelling the measurement as
#
#     m_c = a * S_c(theta) + eps_c ,   eps_c ~ N(0, sigma_c^2)
#
# with amplitude `a = S0` promotes S0 to a fourth, unwanted ("nuisance")
# parameter.  Writing the augmented information matrix in blocks,
#
#     F_full = [ F_tt   F_ta ]      F_tt = a^2 sum_c J_cj J_ck / sigma_c^2
#              [ F_at   F_aa ]      F_ta = a   sum_c J_cj S_c  / sigma_c^2
#                                   F_aa =     sum_c S_c^2     / sigma_c^2 + lambda
#
# the bound on theta alone is the Schur complement
#
#     F_eff = F_tt - F_ta F_aa^-1 F_at .
#
# `lambda >= 0` is an independent Gaussian prior precision on the amplitude and
# selects the three regimes named in the handoff:
#
#     lambda = 0        amplitude entirely unknown (weakest bound)
#     lambda = n0_eff/sigma0^2   amplitude measured with finite precision
#     lambda -> infinity         amplitude known exactly; F_eff -> F_tt
#
# F_ta F_aa^-1 F_at is a rank-one positive-semidefinite outer product, so
# F_eff <= F_tt in the Loewner order for every lambda: marginalizing over an
# unknown amplitude can only lose information, never add it.  The gap between
# the two bounds is reported as its own quantity because it measures how much
# identifiability is spent on amplitude uncertainty alone, with no reference to
# any particular fitter.
#
# Monte-Carlo debiasing convention.  Only the F_tt diagonal is debiased, exactly
# as `fisher_matrix` already does, because E[J_hat^2] = J^2 + Var(J_hat) is a
# first-order bias that does not cancel.  F_ta is left uncorrected for the same
# reason the cross-axis off-diagonals are: J_hat is the difference of the two
# stencil endpoints, so under common random numbers
# Cov(J_hat, S_centre) = [Cov(S+, S_c) - Cov(S-, S_c)] / h, and the two
# covariances are similar in size and opposite in sign, leaving a residual far
# below the diagonal bias this correction exists to remove.


def amplitude_prior_precision(n0_eff: float, sigma_reference: float,
                              reference_signal: float = 1.0) -> float:
    """Gaussian prior precision on `S0` contributed by a low-b reference shell.

    A true `b = 0` reference measured with `n0_eff` effective averages at noise
    `sigma_reference` pins the amplitude with precision `n0_eff/sigma^2`, since
    the normalized signal there is exactly one.

    An acquisition whose lowest shell is `b = b_ref > 0` -- the Jackson-thesis
    structure, whose lowest shell is `b = 50 s/mm2` -- has no such column.
    Using that shell as the normalizer asserts `S(b_ref) = 1` and supplies
    amplitude precision `n0_eff * S(b_ref)^2 / sigma^2` while discarding the
    shell's tissue-derivative content.  Pass the realized `S(b_ref)` as
    ``reference_signal`` to model that.  The information deliberately dropped
    by the collapse is the difference between this prior treatment and simply
    retaining `b_ref` as an ordinary column, which is the quantity Phase 4
    hypothesis H4 is about.
    """
    if n0_eff < 0:
        raise ValueError("n0_eff must be non-negative")
    if sigma_reference <= 0:
        raise ValueError("sigma_reference must be positive")
    return float(n0_eff) * float(reference_signal) ** 2 / float(sigma_reference) ** 2


def amplitude_marginal_fisher(J: np.ndarray, signal: np.ndarray,
                              sigma: np.ndarray | float, *,
                              amplitude: float = 1.0,
                              variance: np.ndarray | None = None,
                              measurement_mask: np.ndarray | None = None,
                              s0_prior_precision: float = 0.0) -> dict[str, Any]:
    """Fixed-S0 and S0-marginalized Fisher matrices for one node.

    Parameters
    ----------
    J : (columns, parameters) Jacobian of the normalized signal.
    signal : (columns,) normalized signal `S/S0` at the same node and columns.
    sigma : scalar or (columns,) noise standard deviation of `m_c`.
    amplitude : the amplitude `a = S0` the acquisition actually carries.
    variance : optional (columns, parameters) `Var(J_hat)` for diagonal
        debiasing, as in `fisher_matrix`.
    s0_prior_precision : `lambda >= 0`; use `amplitude_prior_precision`.

    Returns a dict holding both bounds and the gap between them.
    """
    J = np.asarray(J, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if J.ndim != 2:
        raise ValueError("J must be (columns, parameters)")
    if signal.shape != (J.shape[0],):
        raise ValueError("signal must have one value per J row")
    if s0_prior_precision < 0:
        raise ValueError("s0_prior_precision must be non-negative")
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[0],))
    if measurement_mask is not None:
        keep = np.asarray(measurement_mask, dtype=bool)
        if keep.shape != (J.shape[0],):
            raise ValueError("measurement_mask must have one boolean per column")
        J, signal, sigma = J[keep], signal[keep], sigma[keep]
        if variance is not None:
            variance = np.asarray(variance)[keep]

    a = float(amplitude)
    # Reuse the audited debiasing path, then scale into amplitude units.  The
    # Jacobian of the *measurement* is a*J, so F_tt carries a^2.
    F_tt = a ** 2 * fisher_matrix(J, sigma, variance)
    F_ta = a * (J / sigma[:, None] ** 2).T @ signal
    F_aa = float(np.sum(signal ** 2 / sigma ** 2)) + float(s0_prior_precision)

    if F_aa > 0:
        correction = np.outer(F_ta, F_ta) / F_aa
    else:
        # No column and no prior constrains the amplitude at all; theta is then
        # unidentifiable in any direction that F_ta touches.  Report it rather
        # than silently returning the fixed-S0 answer.
        correction = np.full_like(F_tt, np.inf)
    F_eff = F_tt - correction

    loewner = np.linalg.eigvalsh(correction) if np.all(np.isfinite(correction)) else np.full(J.shape[1], np.inf)
    return {
        "F_fixed_s0": F_tt,
        "F_theta_s0": F_ta,
        "F_s0_s0": F_aa,
        "F_marginal_s0": F_eff,
        "information_lost_to_amplitude": correction,
        "loewner_gap_eigenvalues": loewner,
        "loewner_ok": bool(np.all(loewner >= -1e-9 * max(1.0, float(np.max(np.abs(F_tt)))))),
        "s0_prior_precision": float(s0_prior_precision),
        "amplitude": a,
    }


def amplitude_marginal_diagnostics(result: dict[str, Any], kio_ref: float) -> dict[str, Any]:
    """Fixed-S0 and marginalized CRLB/kappa diagnostics plus their gap.

    ``crlb_ratio`` is the per-parameter factor by which the achievable standard
    deviation grows once the amplitude is treated as unknown.  It is >= 1
    whenever both matrices invert, and is the headline "cost of not knowing
    S0" number.
    """
    fixed = fisher_diagnostics(result["F_fixed_s0"], kio_ref)
    marginal = fisher_diagnostics(result["F_marginal_s0"], kio_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(marginal["crlb"], fixed["crlb"],
                          out=np.full(len(PARAMETER_ORDER), np.inf, dtype=float),
                          where=fixed["crlb"] > 0)
    return {
        "parameter_order": list(PARAMETER_ORDER),
        "fixed_s0": fixed,
        "marginal_s0": marginal,
        "crlb_ratio_marginal_over_fixed": ratio,
        "loewner_ok": result["loewner_ok"],
        "s0_prior_precision": result["s0_prior_precision"],
    }


# ---------------------------------------------------------------------------
# Packed batched Fisher algebra -- added 2026-09-05 for Phase 2
# ---------------------------------------------------------------------------
#
# Phase 2 evaluates on the order of 10^5 candidate acquisitions at ~10^4 grid
# nodes each, so it needs the 3x3 inverse diagonal for millions of matrices at
# once.  `fisher_diagnostics` calls `np.linalg.inv` on one matrix and is the
# readable reference; these functions are its vectorized form and are pinned to
# it by test, rather than being a second implementation left to drift.
#
# A symmetric 3x3 Fisher matrix is carried "packed" as its six independent
# entries in PARAMETER_ORDER = (log rho, log V, k_io):
#
#     packed = [F_rr, F_rV, F_rk, F_VV, F_Vk, F_kk]
#
# so that a batch of matrices is an array of shape (..., 6) and accumulating
# information over acquisition columns is a plain sum along an axis.

FISHER_PACKED_ORDER = ("rho_rho", "rho_V", "rho_kio", "V_V", "V_kio", "kio_kio")
_PACKED_DIAGONAL = (0, 3, 5)


def pack_fisher(F: np.ndarray) -> np.ndarray:
    """Pack a (..., 3, 3) symmetric matrix into its six independent entries."""
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (3, 3):
        raise ValueError("F must be (..., 3, 3)")
    return np.stack([F[..., 0, 0], F[..., 0, 1], F[..., 0, 2],
                     F[..., 1, 1], F[..., 1, 2], F[..., 2, 2]], axis=-1)


def unpack_fisher(packed: np.ndarray) -> np.ndarray:
    """Expand (..., 6) packed entries back to a (..., 3, 3) symmetric matrix."""
    packed = np.asarray(packed, dtype=float)
    if packed.shape[-1] != 6:
        raise ValueError("packed Fisher must have a trailing axis of length 6")
    a, b, c, d, e, f = (packed[..., i] for i in range(6))
    return np.stack([np.stack([a, b, c], axis=-1),
                     np.stack([b, d, e], axis=-1),
                     np.stack([c, e, f], axis=-1)], axis=-2)


def packed_inverse_diagonal(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `([F^-1]_jj, det F, positive_definite)` for a batch of packed matrices.

    The diagonal of the inverse is the cofactor ratio, so it needs no matrix
    inversion.  `positive_definite` is the leading-minor (Sylvester) test
    STRENGTHENED to require all three cofactors positive as well, which is the
    same thing as requiring every entry of the inverse diagonal to be positive.

    Sylvester's three leading minors already imply positive cofactors for an
    exactly positive-definite matrix, so the extra conditions are redundant in
    arithmetic but not in floating point: a Fisher matrix near the boundary can
    pass the leading-minor test and still return a non-positive cofactor ratio,
    which would be reported as a NaN CRLB rather than as an unidentified node.
    Requiring what the CRLB actually needs -- a positive inverse diagonal --
    makes the test self-consistent, and is strictly conservative.

    A matrix that fails is reported rather than silently inverted, because the
    Monte-Carlo diagonal debias can push a weakly determined node indefinite and
    the resulting "CRLB" would be meaningless.
    """
    packed = np.asarray(packed, dtype=float)
    a, b, c, d, e, f = (packed[..., i] for i in range(6))
    cof_a = d * f - e * e
    cof_d = a * f - c * c
    cof_f = a * d - b * b
    det = a * cof_a - b * (b * f - c * e) + c * (b * e - c * d)
    positive = (a > 0) & (cof_a > 0) & (cof_d > 0) & (cof_f > 0) & (det > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_diagonal = np.stack([cof_a / det, cof_d / det, cof_f / det], axis=-1)
    inverse_diagonal = np.where(positive[..., None], inverse_diagonal, np.nan)
    return inverse_diagonal, det, positive


def packed_amplitude_marginal(packed_tt: np.ndarray, F_theta_s0: np.ndarray,
                              F_s0_s0: np.ndarray,
                              s0_prior_precision: np.ndarray | float = 0.0) -> np.ndarray:
    """Batched Schur complement `F_tt - F_ta (F_aa + lambda)^-1 F_at`, packed.

    `lambda = inf` returns `packed_tt` unchanged, which is the known-amplitude
    limit.  See `amplitude_marginal_fisher` for the single-node reference and
    the derivation.
    """
    packed_tt = np.asarray(packed_tt, dtype=float)
    F_theta_s0 = np.asarray(F_theta_s0, dtype=float)
    if F_theta_s0.shape[-1] != 3:
        raise ValueError("F_theta_s0 must have a trailing axis of length 3")
    prior = np.asarray(s0_prior_precision, dtype=float)
    denominator = np.asarray(F_s0_s0, dtype=float) + prior
    t0, t1, t2 = F_theta_s0[..., 0], F_theta_s0[..., 1], F_theta_s0[..., 2]
    outer = np.stack([t0 * t0, t0 * t1, t0 * t2, t1 * t1, t1 * t2, t2 * t2], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = outer / denominator[..., None]
    correction = np.where(np.isfinite(correction), correction, 0.0)
    return packed_tt - correction


# ---------------------------------------------------------------------------
# Degeneracy geometry (Phase 3) -- added 2026-09-09
# ---------------------------------------------------------------------------
#
# Phase 3 asks a different question from Phase 2.  Phase 2 asked how precisely a
# declared acquisition can measure each parameter; Phase 3 asks which parameter
# COMBINATIONS the model resolves and which it barely sees, and whether the
# unresolved one is the constant-`v_i` hyperbola of plan section 2.4.
#
# Everything here is a property of a Fisher matrix that has already been formed,
# so it carries no acquisition assumption of its own.  The Fisher matrix it is
# handed does: `F = J^T Sigma^-1 J` is defined relative to a column set and a
# column weighting, so a spectrum is only meaningful with both named.  Plan
# section 6 requires that naming, and singles out the sloppy-direction ANGLE as
# the weighting-robust quantity, because an angle is a property of a direction
# and not of a scale.
#
# Two conventions, both stated rather than buried:
#
#   Non-dimensionalization.  Plan section 2.3 eigendecomposes `F_tilde = D F D`
#   with `D = diag(1, 1, k_io_ref)`, so a unit step means something comparable on
#   all three axes: the two log parameters are already fractional, and `k_io` is
#   measured in units of `k_io_ref`.  The eigenvectors therefore live in the
#   non-dimensionalized coordinates `theta_tilde = D^-1 theta`.  The first two
#   axes are untouched by `D`, so the constant-`v_i` direction is `(1, -1, 0)`
#   in both coordinate systems and the angle below is unaffected by the choice
#   of `k_io_ref`.  Only the third component, and hence the leakage, depends on
#   it.
#
#   The `k_io`-profiled companion.  `rho_V_profiled_block` eliminates `k_io` by
#   Schur complement, giving the 2x2 precision of `(log rho, log V)` with `k_io`
#   estimated jointly.  It is the instrument the (rho, V)-plane figure of plan
#   section 3.3 actually needs, and, because both its axes are log parameters,
#   it is free of `D` and of `k_io_ref` entirely.  It is reported ALONGSIDE the
#   pre-registered 3x3 result, never instead of it.

# `log v_i = log rho + log V + const`, so this direction holds `v_i` -- and with
# it the geometry factor `g(v_i)` of plan section 2.4 -- exactly constant.
CONSTANT_VI_DIRECTION = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
# Its in-plane complement, along which `v_i` changes fastest.
VI_CHANGING_DIRECTION = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)


def nondimensionalized_fisher(packed: np.ndarray, kio_ref: np.ndarray | float) -> np.ndarray:
    """`F_tilde = D F D` with `D = diag(1, 1, k_io_ref)`, as (..., 3, 3).

    `kio_ref` broadcasts against the leading axes of `packed`, so a batch of
    nodes may each carry their own reference scale (Phase 2 uses the
    pre-registered `max(k_io, k_io_floor)`).
    """
    F = unpack_fisher(packed)
    scale = np.asarray(kio_ref, dtype=float)
    d = np.stack([np.ones_like(scale), np.ones_like(scale), scale], axis=-1)
    return F * d[..., :, None] * d[..., None, :]


def fisher_spectrum(packed: np.ndarray, kio_ref: np.ndarray | float) -> dict[str, np.ndarray]:
    """Eigen-spectrum of `D F D` for a batch of packed Fisher matrices.

    Eigenvalues are returned in DESCENDING order, so `eigenvalues[..., 0]` is
    the stiff direction and `eigenvalues[..., 2]` the sloppy one, matching plan
    section 2.3's `lambda_1 / lambda_3` condition number.  `eigenvectors[..., :, i]`
    is the unit eigenvector of `eigenvalues[..., i]`.

    A Monte-Carlo-debiased Fisher matrix at a weakly determined node can have a
    non-positive smallest eigenvalue.  That is reported (`positive_definite`),
    not repaired: the eigenvector is still the direction the data sees least,
    and suppressing the node would hide exactly the degeneracy being measured.
    The condition number is left as NaN where `lambda_3 <= 0`, because a ratio
    across zero is not a conditioning statement.

    Two separation measures are returned because they answer different
    questions and only one of them is about the sloppy eigenvector's
    uniqueness.  `eigenvalue_ratio_2_over_3 = lambda_2 / lambda_3` is that one:
    near 1 the smallest two eigenvalues are nearly degenerate, the sloppy
    eigenvector is not uniquely defined, and any angle computed from it must be
    read as an arbitrary choice inside a plane.  It is NaN where `lambda_3 <= 0`.
    `sloppy_span_share = (lambda_2 - lambda_3) / (lambda_1 - lambda_3)` is
    sign-safe and defined everywhere, but it measures where `lambda_2` sits in
    the whole spectral span, so a matrix with one dominant stiff direction makes
    it small even when `lambda_2` and `lambda_3` are an order of magnitude apart.
    It is a spectrum-shape statistic, not a degeneracy flag.
    """
    F_tilde = nondimensionalized_fisher(packed, kio_ref)
    eigenvalues, eigenvectors = np.linalg.eigh(F_tilde)     # ascending
    eigenvalues = np.ascontiguousarray(eigenvalues[..., ::-1])
    eigenvectors = np.ascontiguousarray(eigenvectors[..., ::-1])
    lam1, lam2, lam3 = eigenvalues[..., 0], eigenvalues[..., 1], eigenvalues[..., 2]
    positive = lam3 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.where(positive, lam1 / lam3, np.nan)
        ratio = np.where(positive, lam2 / lam3, np.nan)
        span = lam1 - lam3
        share = np.where(span > 0, (lam2 - lam3) / span, np.nan)
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": np.ascontiguousarray(eigenvectors),
        "stiff_vector": np.ascontiguousarray(eigenvectors[..., 0]),
        "sloppy_vector": np.ascontiguousarray(eigenvectors[..., 2]),
        "condition_number": condition,
        "eigenvalue_ratio_2_over_3": ratio,
        "sloppy_span_share": share,
        "positive_definite": positive,
    }


def direction_angle_deg(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Acute angle in degrees between each row of `vectors` and `reference`.

    An eigenvector has no sign, so the angle between two DIRECTIONS is taken
    through the absolute cosine and lands in [0, 90].  Zero means the two
    directions coincide; 90 means they are orthogonal.  For an isotropically
    random direction in three dimensions the median of this angle is 60 degrees,
    which is the null reference the concentration near zero is read against.
    """
    vectors = np.asarray(vectors, dtype=float)
    reference = np.asarray(reference, dtype=float)
    reference = reference / np.linalg.norm(reference)
    norms = np.linalg.norm(vectors, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.abs(np.tensordot(vectors, reference, axes=([-1], [0])) / norms)
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def in_plane_direction_diagnostics(vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 3-vector into its `(log rho, log V)` plane part and its `k_io` part.

    A three-parameter sloppy direction need not lie in the `(log rho, log V)`
    plane at all, and plan section 2.4's hyperbola hypothesis is a statement
    about that plane.  Reporting the in-plane angle without also reporting how
    much of the direction is in the plane would make a direction that is almost
    entirely `k_io` look like a hyperbola whenever its tiny in-plane residue
    happened to point the right way.  Both are therefore returned.

    `in_plane_fraction` is the norm of the `(log rho, log V)` components of a
    unit vector, so it is 1 when the direction lies wholly in the plane and 0
    when it is pure `k_io`.  `in_plane_angle_deg` is the acute angle between the
    projected direction and the constant-`v_i` direction, and is NaN when the
    projection vanishes.
    """
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = vectors / norms[..., None]
    plane = unit[..., :2]
    fraction = np.linalg.norm(plane, axis=-1)
    reference = CONSTANT_VI_DIRECTION[:2] / np.linalg.norm(CONSTANT_VI_DIRECTION[:2])
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.abs(plane @ reference) / fraction
    angle = np.where(fraction > 0, np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))), np.nan)
    return {
        "in_plane_fraction": fraction,
        "k_io_fraction": np.abs(unit[..., 2]),
        "in_plane_angle_deg": angle,
    }


def rho_V_profiled_block(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The `(log rho, log V)` precision with `k_io` profiled out, and its validity.

    Returns `(block, valid)` where `block` is `(..., 2, 2)`.  Eliminating the
    third parameter by Schur complement,

        S = [[F_rr, F_rV], [F_rV, F_VV]] - (1 / F_kk) * outer([F_rk, F_Vk])

    gives the matrix whose inverse is the `(log rho, log V)` block of `F^-1`.  It
    is therefore the precision of the two parameters when `k_io` is estimated
    jointly rather than assumed known, which is the quantity plan section 2.4's
    hyperbola hypothesis and Phase 4's H1 are both about.

    Both axes are log parameters and so are already commensurable: this
    companion needs no non-dimensionalization and is completely independent of
    the `k_io_ref` convention.  It is undefined where `F_kk <= 0`, which the
    Monte-Carlo debias can produce at a node carrying essentially no `k_io`
    information; `valid` is False there and the block is NaN.
    """
    packed = np.asarray(packed, dtype=float)
    if packed.shape[-1] != 6:
        raise ValueError("packed Fisher must have a trailing axis of length 6")
    F_rr, F_rV, F_rk, F_VV, F_Vk, F_kk = (packed[..., i] for i in range(6))
    valid = F_kk > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = F_rr - F_rk * F_rk / F_kk
        b = F_rV - F_rk * F_Vk / F_kk
        d = F_VV - F_Vk * F_Vk / F_kk
    block = np.stack([np.stack([a, b], axis=-1), np.stack([b, d], axis=-1)], axis=-2)
    return np.where(valid[..., None, None], block, np.nan), valid


def rho_V_profiled_spectrum(packed: np.ndarray) -> dict[str, np.ndarray]:
    """Sloppy/stiff directions of the `k_io`-profiled `(log rho, log V)` block.

    Angles are measured against the constant-`v_i` direction `(1, -1)/sqrt(2)`
    for the sloppy eigenvector and against `(1, 1)/sqrt(2)` for the stiff one,
    which is the pair of predictions plan section 2.4 makes: `rho` and `V` trade
    off along the hyperbola while `v_i` itself stays determined.

    In two dimensions those two angles are **identically equal**: the block's
    eigenvectors are orthogonal and so are the two reference directions, so the
    stiff angle is an arithmetic consistency check on the sloppy one, not an
    independent confirmation of the hypothesis.  Both are returned because a
    reader comparing this block with the three-parameter spectrum, where the two
    angles ARE independent, would otherwise have to derive that themselves.
    """
    block, valid = rho_V_profiled_block(packed)
    filled = np.where(np.isfinite(block), block, 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(filled)
    eigenvalues = np.ascontiguousarray(eigenvalues[..., ::-1])
    eigenvectors = np.ascontiguousarray(eigenvectors[..., ::-1])
    stiff, sloppy = eigenvectors[..., 0], eigenvectors[..., 1]
    reference_sloppy = CONSTANT_VI_DIRECTION[:2] / np.linalg.norm(CONSTANT_VI_DIRECTION[:2])
    reference_stiff = VI_CHANGING_DIRECTION[:2] / np.linalg.norm(VI_CHANGING_DIRECTION[:2])
    lam1, lam2 = eigenvalues[..., 0], eigenvalues[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.where(valid & (lam2 > 0), lam1 / lam2, np.nan)
    nan = np.full(eigenvalues.shape[:-1], np.nan)
    return {
        "eigenvalues": np.where(valid[..., None], eigenvalues, np.nan),
        "sloppy_vector": np.where(valid[..., None], sloppy, np.nan),
        "stiff_vector": np.where(valid[..., None], stiff, np.nan),
        "sloppy_angle_deg": np.where(valid, direction_angle_deg(sloppy, reference_sloppy), nan),
        "stiff_angle_deg": np.where(valid, direction_angle_deg(stiff, reference_stiff), nan),
        "condition_number": condition,
        "positive_definite": valid & (lam2 > 0),
        "valid": valid,
    }


def degeneracy_geometry(packed: np.ndarray, kio_ref: np.ndarray | float) -> dict[str, np.ndarray]:
    """Plan items 3.1 and 3.2 for a batch of packed Fisher matrices.

    Collects the pre-registered 3x3 spectrum of `D F D`, the angle between its
    sloppy eigenvector and the constant-`v_i` direction, the same angle for the
    stiff eigenvector against the `v_i`-changing direction, the in-plane/`k_io`
    split of the sloppy direction, and the `k_io`-profiled 2x2 companion.
    """
    spectrum = fisher_spectrum(packed, kio_ref)
    split = in_plane_direction_diagnostics(spectrum["sloppy_vector"])
    profiled = rho_V_profiled_spectrum(packed)
    return {
        "eigenvalues": spectrum["eigenvalues"],
        "eigenvectors": spectrum["eigenvectors"],
        "condition_number": spectrum["condition_number"],
        "eigenvalue_ratio_2_over_3": spectrum["eigenvalue_ratio_2_over_3"],
        "sloppy_span_share": spectrum["sloppy_span_share"],
        "positive_definite": spectrum["positive_definite"],
        "sloppy_vector": spectrum["sloppy_vector"],
        "stiff_vector": spectrum["stiff_vector"],
        "sloppy_angle_deg": direction_angle_deg(spectrum["sloppy_vector"], CONSTANT_VI_DIRECTION),
        "stiff_angle_deg": direction_angle_deg(spectrum["stiff_vector"], VI_CHANGING_DIRECTION),
        "sloppy_in_plane_fraction": split["in_plane_fraction"],
        "sloppy_k_io_fraction": split["k_io_fraction"],
        "sloppy_in_plane_angle_deg": split["in_plane_angle_deg"],
        "profiled_eigenvalues": profiled["eigenvalues"],
        "profiled_sloppy_vector": profiled["sloppy_vector"],
        "profiled_sloppy_angle_deg": profiled["sloppy_angle_deg"],
        "profiled_stiff_angle_deg": profiled["stiff_angle_deg"],
        "profiled_condition_number": profiled["condition_number"],
        "profiled_positive_definite": profiled["positive_definite"],
    }

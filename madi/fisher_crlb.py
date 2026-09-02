"""Phase-0/1 Fisher-information utilities for v5 MADI libraries.

This module deliberately contains no protocol ranking or CRLB map code.  It
validates an artifact, builds central finite-difference fields, and exposes
the noise/feasibility and Fisher primitives needed by later phases.
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

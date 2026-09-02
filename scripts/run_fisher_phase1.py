#!/usr/bin/env python3
"""Build Phase-1 central derivative fields and their diagnostic-column audit.

Use ``--smoke-nodes`` on an incomplete development artifact.  Omit it only
when intentionally writing the full field: the resulting float32 matrices are
large (roughly 2.2 GiB per axis/stencil before metadata).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from madi.fisher_crlb import (assert_safe_output, artifact_manifest, canonical_grid,
                               incomplete_banner, iter_npz_array_chunks, npz_array_info,
                               invalidated_stencils, load_preregistration,
                               metadata_crn_contract)


def _entry_index(manifest, nominal_kio: np.ndarray) -> dict[tuple[int, int, int], int]:
    _, _, kios, _ = canonical_grid()
    out = {}
    for (ir, iv), indices in manifest.group_entries.items():
        for idx in indices:
            ik = int(np.flatnonzero(np.isclose(kios, nominal_kio[idx], rtol=1e-12, atol=1e-12))[0])
            out[(ir, iv, ik)] = int(idx)
    return out


def _centres(manifest, lookup: dict[tuple[int, int, int], int], axis: str, width: int,
             n_kio: int) -> list[tuple[int, int, int, int, int]]:
    """Return `(ir,iv,ik,minus_entry,plus_entry)` for valid central stencils."""
    _, _, kios, retained = canonical_grid()
    items = []
    for ir, iv in sorted(manifest.present):
        for ik in range(n_kio):
            if axis == "rho":
                minus, plus = (ir - width, iv), (ir + width, iv)
                if minus not in retained or plus not in retained:
                    continue
                a, b = lookup.get((minus[0], minus[1], ik)), lookup.get((plus[0], plus[1], ik))
            elif axis == "V":
                minus, plus = (ir, iv - width), (ir, iv + width)
                if minus not in retained or plus not in retained:
                    continue
                a, b = lookup.get((minus[0], minus[1], ik)), lookup.get((plus[0], plus[1], ik))
            else:
                if ik - width < 0 or ik + width >= len(kios):
                    continue
                a, b = lookup.get((ir, iv, ik - width)), lookup.get((ir, iv, ik + width))
            if a is not None and b is not None:
                items.append((ir, iv, ik, a, b))
    return items


def _denominator(axis: str, minus: int, plus: int, rhos: np.ndarray,
                 volumes: np.ndarray, kios: np.ndarray) -> float:
    if axis == "rho":
        return float(np.log(rhos[plus]) - np.log(rhos[minus]))
    if axis == "V":
        return float(np.log(volumes[plus]) - np.log(volumes[minus]))
    return float(kios[plus] - kios[minus])


def _audit(j_path: Path, var_path: Path, sample_path: Path,
           diagnostic_positions: np.ndarray) -> dict:
    J = np.load(j_path, mmap_mode="r")
    variance = np.load(var_path, mmap_mode="r")
    samples = np.load(sample_path)
    se = np.sqrt(variance)
    J_diagnostic = J[:, diagnostic_positions]
    snr = np.divide(np.abs(J_diagnostic), se, out=np.full_like(se, np.nan), where=se > 0)
    beta = np.divide(variance, J_diagnostic ** 2,
                     out=np.full_like(variance, np.inf), where=J_diagnostic != 0)
    magnitude = np.abs(J_diagnostic)
    threshold = np.nanquantile(magnitude, 0.95)
    top = beta[magnitude >= threshold]
    return {"samples": int(len(samples)), "diagnostic_columns": int(variance.shape[1]),
            "snr_partial_fraction_ge_3": float(np.mean(snr >= 3.0)),
            "beta_top_5pct_derivative_magnitude": {"max": float(np.nanmax(top)),
                                                       "q95": float(np.nanquantile(top, .95)),
                                                       "count": int(np.count_nonzero(magnitude >= threshold))},
            "beta_nonfinite_count": int(np.count_nonzero(~np.isfinite(beta)))}


def _write_richardson(output_dir: Path, axis: str, fine: dict, coarse: dict,
                      selected_columns: int) -> dict:
    """Write `(4 J_h - J_2h)/3` over the common central-stencil centres."""
    fine_J = np.load(output_dir / fine["J"], mmap_mode="r")
    coarse_J = np.load(output_dir / coarse["J"], mmap_mode="r")
    fine_samples = np.load(output_dir / fine["samples"])
    coarse_samples = np.load(output_dir / coarse["samples"])
    fine_rows = {tuple(value.tolist()): row for row, value in enumerate(fine_samples)}
    overlap = [(row, fine_rows[tuple(value.tolist())]) for row, value in enumerate(coarse_samples)
               if tuple(value.tolist()) in fine_rows]
    coarse_rows = np.asarray([row for row, _ in overlap], dtype=int)
    source_rows = np.asarray([row for _, row in overlap], dtype=int)
    overlap_samples = coarse_samples[coarse_rows]
    path = output_dir / f"Richardson_{axis}_k1_k2.npy"
    result = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(overlap_samples), selected_columns))
    squared_error = 0.0
    max_error = 0.0
    for row, (coarse_row, source) in enumerate(zip(coarse_rows, source_rows)):
        improved = (4.0 * fine_J[source] - coarse_J[coarse_row]) / 3.0
        result[row] = improved.astype(np.float32)
        delta = np.abs(fine_J[source] - improved)
        squared_error += float(np.sum(delta ** 2, dtype=np.float64))
        max_error = max(max_error, float(np.max(delta)))
    result.flush()
    del result
    np.save(output_dir / f"samples_Richardson_{axis}_k1_k2.npy", overlap_samples)
    return {"field": path.name, "samples": f"samples_Richardson_{axis}_k1_k2.npy",
            "overlap_samples": int(len(overlap_samples)),
            "formula": "(4*J_k1 - J_k2)/3",
            "truncation_bias_abs_rms": float(np.sqrt(squared_error / (len(overlap_samples) * selected_columns))),
            "truncation_bias_abs_max": max_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True,
                        help="JSON emitted by analyze_fisher_feasibility for this artifact")
    parser.add_argument("--smoke-nodes", type=int, default=0,
                        help="limit each axis/stencil to this many valid (rho,V) centres")
    args = parser.parse_args()
    feasibility = json.loads(args.feasibility.read_text(encoding="utf-8"))
    if Path(feasibility["artifact"]).resolve() != args.artifact.resolve():
        raise ValueError("feasibility JSON belongs to a different artifact")
    selected_from_feasibility = np.asarray(
        feasibility["derivative_column_selection"]["column_indices"], dtype=int
    )
    preregistration = load_preregistration()
    widths = {axis: tuple(values) for axis, values in preregistration["stencil_half_widths"].items()}
    with np.load(args.artifact, allow_pickle=False) as data:
        manifest = artifact_manifest(data)
        assert_safe_output(args.output_dir, manifest)
        build = json.loads(str(data["build_metadata_json"]))
        if not metadata_crn_contract(build):
            raise RuntimeError("v5 CRN ensemble-index contract missing; refusing covariance estimates")
        nominal_kio = np.asarray(data["nominal_kios"], dtype=float)
        lookup = _entry_index(manifest, nominal_kio)
        rhos, volumes, kios = (np.asarray(data[name], dtype=float) for name in ("rhos", "Vs", "kios"))
        pairs = list(zip(np.asarray(data["pair_deltas"]), np.asarray(data["pair_Deltas"])))
        b_values = np.asarray(data["b_values"])
        subset_pairs = list(zip(np.asarray(data["ensemble_subset_pair_deltas"]), np.asarray(data["ensemble_subset_pair_Deltas"])))
        subset_b = np.asarray(data["ensemble_subset_b_values"])
        n_b = int(data["n_b"])
        n_columns = len(pairs) * n_b
        subset_columns = np.asarray([pairs.index(pair) * n_b + int(np.flatnonzero(b_values == b)[0])
                                     for pair in subset_pairs for b in subset_b], dtype=int)
        subset_shape, _, _ = npz_array_info(args.artifact, "ensemble_means_subset")
        n_ensembles = int(subset_shape[1])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if np.any(selected_from_feasibility < 0) or np.any(selected_from_feasibility >= n_columns):
            raise ValueError("feasibility JSON contains out-of-range column indices")
        selected_columns = np.unique(np.concatenate([selected_from_feasibility, subset_columns]))
        diagnostic_positions = np.searchsorted(selected_columns, subset_columns)
        if not np.array_equal(selected_columns[diagnostic_positions], subset_columns):
            raise RuntimeError("diagnostic columns were not retained in the Phase-1 selection")
        report = {"schema": "madi-fisher-phase1-v2", "artifact": str(args.artifact),
                  "banner": incomplete_banner(manifest), **manifest.as_dict(),
                  "smoke_nodes": args.smoke_nodes, "derivatives": {},
                  "stencil_invalidated_nodes": invalidated_stencils(manifest, widths),
                  "feasibility": str(args.feasibility),
                  "column_selection": {
                      "feasible_research_columns": int(len(selected_from_feasibility)),
                      "diagnostic_columns": int(len(subset_columns)),
                      "selected_columns": int(len(selected_columns)),
                      "full_stored_columns": int(n_columns),
                      "selected_full_column_indices": selected_columns.tolist(),
                      "diagnostic_full_column_indices": subset_columns.tolist(),
                  }}
        work: list[tuple[str, int, list[tuple[int, int, int, int, int]], Path, Path, Path]] = []
        # Map each endpoint to its signed contribution.  This lets the large
        # compressed vector member be streamed once rather than materialised.
        vector_actions: dict[int, list[tuple[int, int, float]]] = {}
        variance_actions: dict[int, list[tuple[int, int, float]]] = {}
        for axis, axis_widths in widths.items():
            for width in axis_widths:
                centres = _centres(manifest, lookup, axis, width, len(canonical_grid()[2]))
                if args.smoke_nodes:
                    keep_pairs = set((ir, iv) for ir, iv, *_ in centres[:args.smoke_nodes])
                    centres = [item for item in centres if item[:2] in keep_pairs]
                stem = f"{axis}_k{width}"
                j_path = args.output_dir / f"J_{stem}.npy"
                var_path = args.output_dir / f"VarJ_{stem}_diagnostic.npy"
                sample_path = args.output_dir / f"samples_{stem}.npy"
                J = np.lib.format.open_memmap(j_path, mode="w+", dtype=np.float32,
                                               shape=(len(centres), len(selected_columns)))
                J[:] = 0.0
                samples = np.empty((len(centres), 3), dtype=np.int16)
                for row, (ir, iv, ik, minus, plus) in enumerate(centres):
                    denom = _denominator(axis, minus, plus, rhos, volumes, kios)
                    if not np.isfinite(denom) or denom <= 0:
                        raise RuntimeError(f"non-positive realised {axis} denominator at {(ir, iv, ik)}")
                    work_index = len(work)
                    vector_actions.setdefault(minus, []).append((work_index, row, -1.0 / denom))
                    vector_actions.setdefault(plus, []).append((work_index, row, 1.0 / denom))
                    variance_actions.setdefault(minus, []).append((work_index, row, 1.0 / denom ** 2))
                    variance_actions.setdefault(plus, []).append((work_index, row, 1.0 / denom ** 2))
                    samples[row] = (ir, iv, ik)
                del J
                np.save(sample_path, samples)
                work.append((axis, width, centres, j_path, var_path, sample_path))
        j_maps = [np.load(item[3], mmap_mode="r+") for item in work]
        for start, chunk in iter_npz_array_chunks(args.artifact, "vectors"):
            selected = chunk[:, selected_columns]
            for offset, entry in enumerate(range(start, start + len(chunk))):
                for work_index, row, scale in vector_actions.get(entry, ()):
                    j_maps[work_index][row] += (scale * selected[offset]).astype(np.float32)
        for item in j_maps:
            item.flush()
        del j_maps
        var_maps = []
        for _, _, centres, _, var_path, _ in work:
            VarJ = np.lib.format.open_memmap(var_path, mode="w+", dtype=np.float32,
                                              shape=(len(centres), len(subset_columns)))
            VarJ[:] = 0.0
            var_maps.append(VarJ)
        for start, chunk in iter_npz_array_chunks(args.artifact, "signal_variance"):
            diagnostic = chunk[:, subset_columns]
            for offset, entry in enumerate(range(start, start + len(chunk))):
                for work_index, row, scale in variance_actions.get(entry, ()):
                    var_maps[work_index][row] += (scale * diagnostic[offset]).astype(np.float32)
        for item in var_maps:
            item.flush()
        del var_maps
        # Aligned ensemble means are modest enough to cache on disk.  They are
        # needed only while calculating endpoint covariance for diagnostic
        # columns, then removed as a temporary implementation detail.
        ensemble_cache_path = args.output_dir / ".ensemble_subset_cache.npy"
        ensemble_cache = np.lib.format.open_memmap(ensemble_cache_path, mode="w+", dtype=np.float32,
                                                    shape=subset_shape)
        for start, chunk in iter_npz_array_chunks(args.artifact, "ensemble_means_subset"):
            ensemble_cache[start:start + len(chunk)] = chunk
        ensemble_cache.flush()
        for axis, width, centres, j_path, var_path, sample_path in work:
            VarJ = np.load(var_path, mmap_mode="r+")
            for row, (ir, iv, ik, minus, plus) in enumerate(centres):
                denom = _denominator(axis, minus, plus, rhos, volumes, kios)
                ensemble_minus = ensemble_cache[minus]
                ensemble_plus = ensemble_cache[plus]
                covariance = np.sum((ensemble_minus - ensemble_minus.mean(axis=0)) *
                                    (ensemble_plus - ensemble_plus.mean(axis=0)), axis=0) / (n_ensembles - 1)
                VarJ[row] = np.maximum(VarJ[row] - 2.0 * covariance / (n_ensembles * denom ** 2), 0.0)
            VarJ.flush()
            del VarJ
            item = {"J": j_path.name, "VarJ_diagnostic": var_path.name,
                    "samples": sample_path.name,
                    "audit": _audit(j_path, var_path, sample_path, diagnostic_positions)}
            report["derivatives"][f"{axis}_k{width}"] = item
        del ensemble_cache
        ensemble_cache_path.unlink()
    report["richardson"] = {}
    for axis in ("rho", "V"):
        report["richardson"][axis] = _write_richardson(
            args.output_dir, axis, report["derivatives"][f"{axis}_k1"],
            report["derivatives"][f"{axis}_k2"], len(selected_columns)
        )
    report["storage_bytes"] = int(sum(path.stat().st_size for item in report["derivatives"].values()
                                      for path in (args.output_dir / item["J"],
                                                   args.output_dir / item["VarJ_diagnostic"],
                                                   args.output_dir / item["samples"])))
    report["storage_bytes"] += int(sum((args.output_dir / item["field"]).stat().st_size +
                                        (args.output_dir / item["samples"]).stat().st_size
                                        for item in report["richardson"].values()))
    (args.output_dir / "phase1_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["banner"])
    print(json.dumps({"grid_complete": report["grid_complete"], "derivatives": report["derivatives"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

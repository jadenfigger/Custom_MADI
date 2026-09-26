"""Extend the combined notebook's protocol-statistics field family into tidy tables.

The numerical snapshot is the source of the workbook and every figure. Native
and scaled matrices and variance/SD are deliberately separate. Historical Excel
rows are not relabelled as results under this experiment's new assumptions.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .core import PARAMS, SCHEMA, NOISE_FORMULA, clean, digest, dumps


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def append_table(path, rows, keys):
    """Atomic deterministic upsert; rejects duplicate keys within a new snapshot."""
    path = Path(path)
    incoming = {}
    for row in rows:
        row = clean(row)
        key = tuple(str(row[k]) for k in keys)
        if key in incoming and incoming[key] != row:
            raise ValueError(f"Conflicting duplicate record in {path.name}: {key}")
        incoming[key] = row
    previous = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            previous = list(csv.DictReader(f))
    records = {tuple(str(row[k]) for k in keys): row for row in previous}
    records.update(incoming)
    rows = list(records.values())
    headers = list(dict.fromkeys(k for row in rows for k in row))
    if not headers:
        headers = keys
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".csv"); os.close(fd)
    try:
        with open(temp, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, headers); writer.writeheader(); writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return len(rows)


def common_metadata(config, run_id, timestamp, code_version):
    return dict(run_id=run_id, timestamp_utc=timestamp, analysis_version=SCHEMA,
                code_version=code_version, **config["noise"], noise_formula=NOISE_FORMULA,
                te_mode="protocol_adaptive_common", **config["evaluation"],
                G_max_T_m=config["design"]["G_max_T_m"], k_io_relative_denominator="max(k_io,5 s^-1)")


def result_rows(answer, source, config, meta, name, roles):
    protocol = answer["protocol"]
    spec = source.specification(protocol)
    total = sum(n for _, n in protocol)
    summary_common = dict(protocol_id=answer["protocol_id"], protocol_name=name,
                          roles=";".join(sorted(roles)), evaluation_key=answer["evaluation_key"],
                          total_measurements=total, TE_A_s=answer["TE_A_s"], sigma_single=answer["sigma_single"],
                          n_unique_columns=len(protocol), n_timing_pairs=len({(r["delta_ms"], r["Delta_ms"]) for r in spec}),
                          n_distinct_b_values=len({r["b_s_mm2"] for r in spec}),
                          n_diffusion_shells=len({r["b_s_mm2"] for r in spec if r["b_s_mm2"] > 0}),
                          n_timing_shells=sum(r["b_s_mm2"] > 0 for r in spec),
                          n_b0_measurements=sum(r["averages"] for r in spec if r["b_s_mm2"] == 0),
                          n_diffusion_weighted_measurements=sum(r["averages"] for r in spec if r["b_s_mm2"] > 0),
                          repetitions_json=dumps({c: n for c, n in protocol}), acquisitions_json=dumps(spec),
                          actual_te_ms=1000*answer["TE_A_s"], estimated_min_te_ms=1000*answer["TE_A_s"],
                          requested_rho=source.requested[0], requested_V=source.requested[1], requested_k_io=source.requested[2],
                          selected_node_ids_json=dumps(source.node_ids), node_domain=config["nodes"],
                          cache_fingerprint=source.fingerprint, library_path=str(source.library),
                          phase1_path=str(source.phase1), cache_path=str(source.cache), n_ensembles=source.n_ensembles,
                          **meta)
    summary_common["node_domain"] = dumps(summary_common["node_domain"])
    summaries, nodes, acquisitions = [], [], []
    table = source.table
    index = source.nearest_local
    node_id = source.node_ids[index]
    for amplitude, d in answer["models"].items():
        summary = dict(summary_common, amplitude_model=amplitude, **answer["summaries"][amplitude])
        summary["binding_node_ids_json"] = dumps(source.node_ids[summary.pop("binding_node_local_indices")])
        summary["worst_node_id"] = int(source.node_ids[summary.pop("worst_node_local_index")])
        summary.update(actual_rho=table["rho"][node_id], actual_V=table["V"][node_id],
                       actual_k_io=table["kio"][node_id], node_index=int(node_id),
                       used_measurements_median=np.median(answer["used_measurements"]),
                       used_measurements_min=int(answer["used_measurements"].min()))
        gap = np.divide(answer["models"]["marginal_S0"]["crlb_sd"], answer["models"]["fixed_S0"]["crlb_sd"],
                        out=np.full_like(d["crlb_sd"], np.nan),
                        where=np.isfinite(answer["models"]["marginal_S0"]["crlb_sd"]) & np.isfinite(answer["models"]["fixed_S0"]["crlb_sd"]))
        for j, p in enumerate(PARAMS):
            finite = gap[:, j][np.isfinite(gap[:, j])]
            summary[f"S0_SD_gap_{p}_median_both_valid"] = np.median(finite) if len(finite) else np.nan
        summaries.append(summary)
        for k, actual_id in enumerate(source.node_ids):
            row = dict(protocol_id=answer["protocol_id"], amplitude_model=amplitude,
                       evaluation_key=answer["evaluation_key"], node_index=int(actual_id),
                       actual_rho=table["rho"][actual_id], actual_V=table["V"][actual_id], actual_k_io=table["kio"][actual_id],
                       realised_rho=source.realised[table["centre"][actual_id], 0],
                       realised_V=source.realised[table["centre"][actual_id], 1],
                       realised_k_io=source.realised[table["centre"][actual_id], 2],
                       volume_fraction=table["rho"][actual_id]*table["V"][actual_id]*1e-6,
                       identifiable=bool(d["positive"][k]), invalidity_reason=d["invalidity_reason"][k],
                       native_trace_crlb=d["native_trace_crlb"][k], scaled_trace_crlb=d["scaled_trace_crlb"][k],
                       used_measurements=int(answer["used_measurements"][k]), masked_measurements=total-int(answer["used_measurements"][k]),
                       n_kept_columns=int(answer["n_kept_columns"][k]), kio_ref=source.kref[k],
                       entry_index=int(table["centre"][actual_id]),
                       rho_index=int(table["nodes"][actual_id, 0]), V_index=int(table["nodes"][actual_id, 1]),
                       k_io_index=int(table["nodes"][actual_id, 2]),
                       requested_rho=source.requested[0], requested_V=source.requested[1], requested_k_io=source.requested[2],
                       is_requested_nearest_node=(k == index), total_measurements=total,
                       TE_A_s=answer["TE_A_s"], sigma_single=answer["sigma_single"], **meta)
            for j, p in enumerate(PARAMS):
                for metric in ("crlb_variance", "crlb_sd", "relative_crlb_sd"):
                    row[f"{metric}_{p}"] = d[metric][k, j]
                row[f"relative_crlb_variance_{p}"] = d["relative_crlb_sd"][k, j]**2
                row[f"floor_relative_crlb_sd_{p}"] = d["relative_crlb_sd"][k, j]
                row[f"S0_SD_gap_{p}"] = gap[k, j]
                physical_scale = (table["rho"][actual_id], table["V"][actual_id], 1)[j]
                row[f"physical_crlb_sd_{p}"] = d["crlb_sd"][k, j]*physical_scale
                row[f"physical_crlb_variance_{p}"] = d["crlb_variance"][k, j]*physical_scale**2
                row[f"kappa_{p}"] = np.sqrt(max(0, d["F_native"][k, j, j]*d["crlb_variance"][k, j])) if d["positive"][k] else np.nan
                row[f"step_{p}"] = table[f"step_{('rho', 'V', 'k_io')[j]}"][actual_id]
            for matrix_name in ("F_native", "F_scaled", "Finv_native", "Finv_scaled", "F_undebiased", "F_debias_correction", "F_amplitude_correction"):
                matrix = d[matrix_name][k] if matrix_name in d else answer[matrix_name][k]
                for i, a in enumerate(PARAMS):
                    for j, b in enumerate(PARAMS):
                        row[f"{matrix_name}_{a}__{b}"] = matrix[i, j]
            for coordinate in ("native", "scaled"):
                e = d[f"{coordinate}_eigenvalues"][k]
                for metric in ("determinant", "log_determinant", "determinant_sign", "log_abs_determinant", "rank", "positive_definite", "condition_number"):
                    row[f"{coordinate}_{metric}"] = d[f"{coordinate}_{metric}"][k]
                row[f"{coordinate}_lambda1_over_lambda3"] = e[0]/e[2] if d["positive"][k] else np.nan
                row[f"{coordinate}_lambda3_over_lambda1"] = e[2]/e[0] if d["positive"][k] else np.nan
                for j in range(3):
                    row[f"{coordinate}_lambda{j+1}"] = e[j]
                    for i, p in enumerate(PARAMS):
                        row[f"{coordinate}_eigvec_{p}_lambda{j+1}"] = d[f"{coordinate}_eigenvectors"][k, i, j]
            row["F_S0S0"] = answer["F_S0S0"][k]
            for j, p in enumerate(PARAMS):
                row[f"F_{p}_S0"] = answer["F_theta_S0"][k, j]
            nodes.append(row)
    ids = [c for c, _ in protocol]
    _, _, variance, signals = source.get_columns(ids)
    for i, r in enumerate(spec):
        c, n = r["full_column"], r["averages"]
        row = dict(protocol_id=answer["protocol_id"], evaluation_key=answer["evaluation_key"],
                   protocol_name=name, **r, gradient_T_m=source.gradient[c],
                   delta_s=r["delta_ms"]/1000, Delta_s=r["Delta_ms"]/1000,
                   TE_A_s=answer["TE_A_s"], sigma_single=answer["sigma_single"],
                   sigma_average=answer["sigma_single"]/np.sqrt(n), reference_node_index=int(node_id),
                   signal_at_reference_node=signals[i, index], **meta)
        position = source.domain.position_of[c]
        for j, p in enumerate(PARAMS):
            v = float(source.jmaps[p][table["phase1_rows"][("rho", "V", "k_io")[j]][node_id], position])
            row[f"J_{p}_at_reference_node"] = v
            row[f"var_J_{p}_at_reference_node"] = variance[i, index, j]
            row[f"beta_{p}_at_reference_node"] = variance[i, index, j]/v**2 if v else np.inf
        acquisitions.append(row)
    return summaries, nodes, acquisitions


DEFINITIONS = {
    "workflow": "Independent conditional analysis. Historical Phase-2/3 reports and the source workbook are unchanged.",
    "export_lineage": "Extends analysis/phase2_multi_delta_combined.ipynb export_rows/export_protocol: protocol_stats becomes protocol_summary plus node_metrics; acquisition_columns retained; definitions expanded to definitions_config.",
    "coordinates": "Native theta=(ln rho,ln V,k_io); q=(ln rho,ln V,k_io/max(k_io,5)); natural logarithms. Matrix suffixes always follow this named order, including scaled k_io.",
    "units": "rho: cells/uL; V: pL; k_io: s^-1; timings: ms and explicitly _s fields; b: s/mm^2; gradient: T/m.",
    "noise": NOISE_FORMULA,
    "SNR_reference": "SNR_ref is unweighted single-volume SNR at TE_ref; defaults 50 at TE_ref_s=0 from combined notebook. Every column, including b0, shares protocol TE.",
    "budget": "One repeat is one acquired volume. N includes explicit b0 and masked measurements. No time normalization.",
    "CRLB": "crlb_variance=diag(F^-1); crlb_sd=sqrt(variance). relative_crlb_sd=(SD_logrho,SD_logV,SD_kio/max(k_io,5)). Invalid variance/SD rows are +inf; inverse entries blank.",
    "physical_bounds": "Local first-order transformation: physical SD=(rho*SD_logrho,V*SD_logV,SD_kio). Not an exact lognormal uncertainty interval.",
    "traces": "native_trace_crlb has mixed native coordinates and is diagnostic only. scaled_trace_crlb=sum(relative_crlb_sd**2) is dimensionless A loss.",
    "S0": "Both fixed_S0 and marginal_S0 are exported. Marginal uses authoritative packed_amplitude_marginal with prior=0 and all retained acquired columns. Reserved true-b0 volumes consume budget; no external prior or uncounted references.",
    "debias": "Read cached Phase-1 J directly. Endpoint-only Var(J) from stored signal_variance divided by n_ensembles and realised stencil spacing squared through madi.fisher_crlb.derivative_variance. Correction linear in repeats; no new simulations/derivatives.",
    "missing_covariance": "CRN covariance exists at 200 diagnostic columns only. Exact diagnostic VarJ is not mixed with endpoint-only correction; preserve uniform executed Phase-2 convention. This may be conservative. Truncation error is not corrected.",
    "masks": "G_max restricts the declared design only. Trust and Rician masks operate per node/column during evaluation. Phase2_repeated_mean uses S*sqrt(n)/sigma; single_measurement uses S/sigma. No Rician likelihood implemented. Defaults masks off.",
    "objective": "Default max over parameters of median over ALL declared nodes of relative SD, with invalid nodes +inf. Coverage must meet minimum_coverage>0.5. Identifiable-only means are secondary and never used to rank.",
    "binding": "Binding parameter maximizes the three medians. Binding nodes are central order-statistic node(s) for that parameter; worst_node_id is separate. For A/D/E these binding fields refer to secondary robust score.",
    "secondary_objectives": "Minimize A_scaled_trace median, D_negative_scaled_logdet median, or E_inverse_smallest_scaled_eigenvalue median; invalid nodes are +inf in each. No combined score. Natural log determinant.",
    "spectrum": "Descending eigenvalues; eigenvectors stored by component/lambda. Largest absolute component positive fixes sign; near-degenerate eigenspaces may rotate. PD/inversion gated by scaled minimum eigenvalue > pd_rtol*max(abs(eigenvalues)), plus authoritative positive-minor test.",
    "no_regularization": "No pseudoinverse or ridge. Non-PD, singular, and fully masked rows carry reasons and no finite covariance/CRLB. Spectrum remains visible to diagnose failure.",
    "aggregates": "Quantiles/mean_all_nodes include +inf failures; separately labeled identifiable-only means are diagnostic. Quantiles use linear interpolation with explicit infinite tails. Rank counts absolute eigenvalues above coordinate-specific tolerance.",
    "node_domain": "Default 233 rho/V complete-stencil pairs at k_io=5,20,40: 699 nodes. Config k_io_values=null selects all 11,417 complete-stencil nodes. This declared selection does not alter the 31,125-column source domain.",
    "requested_node": "Requested coordinates are annotations; is_requested_nearest_node flags nearest evaluated library node under range-normalized log-rho/log-V/linear-k distance. All actual canonical indices/coordinates are exported.",
    "optimizer": "Finite multi-start discrete annealing and strict-improvement local column replacements/repetition transfers/timing-group moves. Best found, no global optimum claim outside exhaustive validation space.",
    "persistence": "Scores keyed by canonical protocol, full evaluation settings, source fingerprint and evaluator source hash in SQLite. CSV records upsert by evaluation key plus amplitude/node/column; history by run_id/evaluation_id. One writer; workbook written atomically from CSV snapshots.",
    "figure_data": "All plots read exported CSV products. Each figure has noise/config metadata JSON and embedded PNG/PDF/SVG metadata. Timing sweeps move one named group, retain all other acquisitions, then recompute whole-protocol TE.",
    "infinity": "CSV/XLSX uses literal +inf for nonfinite loss/uncertainty; unavailable covariance/logdet/eigen ratios are blank. These are deliberate statuses, not spreadsheet errors.",
}


def definitions(config, source, run_id, extra):
    result = [dict(run_id=run_id, field=key, definition=value) for key, value in DEFINITIONS.items()]
    for key, value in {"run_config": config, "cache_provenance": source.provenance, **extra}.items():
        content = dumps(value)
        # Keep even an all-node/full-column manifest losslessly within Excel's cell limit.
        chunks = [content[i:i+28000] for i in range(0, len(content), 28000)]
        for i, chunk in enumerate(chunks):
            result.append(dict(run_id=run_id, field=f"{key}.json.part{i+1:03d}_of_{len(chunks):03d}", definition=chunk))
    return result

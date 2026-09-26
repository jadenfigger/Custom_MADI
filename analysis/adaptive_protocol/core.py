"""Read-only substrate adapter and adaptive common-TE protocol evaluation.

No simulator calls, finite-difference recomputation, alternate TE mode,
pseudoinverse, ridge, or time normalization is provided here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from madi.fisher_crlb import (
    PARAMETER_ORDER, column_arrays, derivative_variance, gradient_strength_t_per_m,
    nondimensionalized_fisher, pack_fisher, packed_amplitude_marginal,
    packed_inverse_diagonal, read_column_domain, require_columns, unpack_fisher,
)
from scripts.run_fisher_phase2 import build_node_table

SCHEMA = "adaptive-protocol-v1"
AXES = {"log_rho": "rho", "log_V": "V", "k_io": "k_io"}
PARAMS = tuple(PARAMETER_ORDER)
assert PARAMS == ("log_rho", "log_V", "k_io")
NOISE_FORMULA = "sigma_A=(1/SNR_ref)*exp((max(delta_s+Delta_s)+TE_offset_s-TE_ref_s)/T2_s)"


def clean(value):
    """Strict JSON with explicit nonfinite values (never an Excel error)."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value == math.inf else "-inf" if value == -math.inf else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def dumps(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()[:20]


def integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    if not np.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def canonical(protocol):
    merged = {}
    for c, n in protocol:
        c, n = integer(c, "column"), integer(n, "repetitions")
        if n:
            merged[c] = merged.get(c, 0) + n
    if not merged:
        raise ValueError("A protocol must acquire at least one volume")
    return tuple(sorted(merged.items()))


def file_stamp(path, full_hash=False):
    st = path.stat()
    result = {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if full_hash:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class Substrate:
    """Full-domain source maps; working sets are temporary evaluation views only."""
    def __init__(self, config):
        self.config = config
        self.library = Path(config["library"])
        if not self.library.is_absolute():
            self.library = REPO / self.library
        runs = Path(config["runs"])
        self.phase1, self.cache = runs / "phase1", runs / "cache"
        manifest_path = self.phase1 / "phase1_manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.domain = read_column_domain(self.manifest)
        cache_path = self.cache / "phase2_cache_manifest.json"
        cm = json.loads(cache_path.read_text())
        if not self.domain.is_complete or self.domain.basis != "all_stored_columns":
            raise ValueError("Full stored-domain Phase-1 substrate required: " + self.domain.banner())
        if cm["column_domain"] != self.manifest["column_domain"]:
            raise ValueError("Phase-1 and signal-cache column domains disagree")
        with np.load(self.library, allow_pickle=False) as data:
            self.table = build_node_table(self.phase1, data)
            self.pair_d = data["pair_deltas"].astype(float)
            self.pair_D = data["pair_Deltas"].astype(float)
            self.b_values = data["b_values"].astype(float)
            self.columns = np.stack(column_arrays(self.pair_d, self.pair_D, self.b_values), axis=1)
            self.n_ensembles = int(json.loads(data["build_metadata_json"].item())["ensembles_per_entry"])
            self.realised = np.stack([data["rhos"], data["Vs"], data["kios"]], axis=1)
        if self.domain.full_stored_columns != len(self.columns):
            raise ValueError("Library and substrate column count mismatch")
        self.gradient = gradient_strength_t_per_m(*self.columns.T)
        self.lookup = {tuple(row): c for c, row in enumerate(self.columns)}
        if len(self.lookup) != len(self.columns):
            raise ValueError("Duplicate stored column identities in library metadata")
        self.vectors = np.load(self.cache / "vectors_selected_T.npy", mmap_mode="r")
        self.variance = np.load(self.cache / "signal_variance_selected_T.npy", mmap_mode="r")
        expected = (len(self.columns), len(self.realised))
        if self.vectors.shape != expected or self.variance.shape != expected:
            raise ValueError(f"Cache shapes do not match source library {expected}")
        self.jmaps = {p: np.load(self.phase1 / f"J_{AXES[p]}_k1.npy", mmap_mode="r") for p in PARAMS}
        for p, array in self.jmaps.items():
            if array.shape != (len(np.load(self.phase1 / f"samples_{AXES[p]}_k1.npy")), len(self.columns)):
                raise ValueError(f"Derivative domain/row shape mismatch: {p}")
        wanted_k = config["nodes"].get("k_io_values")
        self.node_ids = np.arange(len(self.table["nodes"]))
        if wanted_k is not None:
            if not set(wanted_k) <= set(self.table["kio"]):
                raise ValueError("Requested k_io slice is absent from the complete-stencil node table")
            self.node_ids = self.node_ids[np.isin(self.table["kio"], wanted_k)]
        if config["nodes"].get("indices") is not None:
            requested = {tuple(x) for x in config["nodes"]["indices"]}
            existing = {tuple(x) for x in self.table["nodes"][self.node_ids]}
            if not requested <= existing:
                raise ValueError("Requested exact parameter node absent from declared node set")
            self.node_ids = np.array([i for i in self.node_ids if tuple(self.table["nodes"][i]) in requested])
        if not len(self.node_ids):
            raise ValueError("Empty evaluation node set")
        self.kref = np.maximum(self.table["kio"][self.node_ids], 5.0)
        target = np.asarray(config["nodes"].get("requested_node", [250000, 4, 20]), float)
        if target.shape != (3,) or not np.all(np.isfinite(target)) or np.any(target[:2] <= 0):
            raise ValueError("requested_node must be finite (rho>0,V>0,k_io)")
        coords = np.stack([self.table[k][self.node_ids] for k in ("rho", "V", "kio")], axis=1)
        transform = coords.copy(); transform[:, :2] = np.log(transform[:, :2])
        transformed_target = target.copy(); transformed_target[:2] = np.log(target[:2])
        scale = np.ptp(transform, axis=0); scale[scale == 0] = 1
        self.requested = target
        self.nearest_local = int(np.argmin(np.sum(((transform - transformed_target) / scale)**2, axis=1)))
        self.memory = {}  # read-through, unweighted, unmasked columns; never saved over source caches
        sources = [self.library, manifest_path, cache_path,
                   self.cache / "vectors_selected_T.npy", self.cache / "signal_variance_selected_T.npy"]
        sources += [self.phase1 / f"J_{a}_k1.npy" for a in AXES.values()]
        sources += [self.phase1 / f"samples_{a}_k1.npy" for a in AXES.values()]
        self.provenance = {
            "sources": [file_stamp(p, p.suffix == ".json") for p in sources],
            "column_domain": self.domain.as_dict(), "node_ids": self.node_ids.tolist(),
            "all_complete_stencil_nodes": len(self.table["nodes"]),
            "parameter_order": PARAMS, "derivatives": "Stored Phase-1 k=1 float32 J, cast to float64, never regenerated",
            "variance": "Phase-2 endpoint-only Var(J), via authoritative derivative_variance; no CRN subtraction",
            "new_quantity_reason": "Endpoint-only Var(J) is not stored for all columns. Diagnostic VarJ includes CRN covariance and cannot substitute for the uniform Phase-2 endpoint-only convention.",
            "n_ensembles": self.n_ensembles,
        }
        self.fingerprint = digest(self.provenance)

    def exact_column(self, triple):
        try:
            c = self.lookup[tuple(map(float, triple))]
        except KeyError as exc:
            raise ValueError(f"Requested acquisition {triple} is not a stored column; no substitution allowed") from exc
        require_columns(self.domain, [c], "adaptive protocol", *self.columns.T)
        return c

    def protocol(self, rows):
        return canonical((self.exact_column(row[:3]), row[3]) for row in rows)

    def specification(self, protocol):
        return [{"full_column": c, "delta_ms": self.columns[c, 0], "Delta_ms": self.columns[c, 1],
                 "b_s_mm2": self.columns[c, 2], "averages": n} for c, n in canonical(protocol)]

    def get_columns(self, ids):
        missing = [int(c) for c in ids if int(c) not in self.memory]
        if missing:
            positions = require_columns(self.domain, missing, "evaluation working set", *self.columns.T)
            table, nodes = self.table, self.node_ids
            centre = table["centre"][nodes]
            # Only existing stored derivative values are read. No finite differences here.
            j = np.stack([np.asarray(self.jmaps[p][np.ix_(table["phase1_rows"][AXES[p]][nodes], positions)], float).T
                          for p in PARAMS], axis=-1)
            signal = np.asarray(self.vectors[np.ix_(positions, centre)], float)
            varj = np.stack([
                derivative_variance(
                    self.variance[np.ix_(positions, table[f"minus_{AXES[p]}"][nodes])],
                    self.variance[np.ix_(positions, table[f"plus_{AXES[p]}"][nodes])],
                    None, None, table[f"step_{AXES[p]}"][nodes][None, :], self.n_ensembles)
                for p in PARAMS], axis=-1)
            if not all(np.all(np.isfinite(x)) for x in (j, signal, varj)) or np.any(varj < 0):
                raise ValueError("Nonfinite stored signal/J or invalid derivative variance")
            raw = pack_fisher(j[..., :, None] * j[..., None, :])
            amp = np.concatenate((j * signal[..., None], signal[..., None]**2), axis=-1)
            for k, c in enumerate(missing):
                self.memory[c] = (raw[k], amp[k], varj[k], signal[k])
        return tuple(np.stack([self.memory[int(c)][i] for c in ids]) for i in range(4))


def noise(protocol, columns, config):
    if config["T2_s"] != 0.040 or config["TE_offset_s"] != 0.014:
        raise ValueError("This workflow requires T2_s=0.040 and TE_offset_s=0.014")
    if not np.isfinite(config["SNR_ref"]) or config["SNR_ref"] <= 0 or not np.isfinite(config["TE_ref_s"]):
        raise ValueError("Reference SNR must be positive and reference TE finite")
    ids = [c for c, _ in canonical(protocol)]
    te = float(np.max(columns[ids, 0] + columns[ids, 1]) / 1000 + config["TE_offset_s"])
    sigma = math.exp((te - config["TE_ref_s"]) / config["T2_s"]) / config["SNR_ref"]
    return te, sigma


def quantile(values, q):
    """Linear order statistic, with an explicit infinite upper tail (no inf-inf NaN)."""
    values = np.sort(np.asarray(values, float), axis=0)
    position = q * (len(values) - 1)
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    if lo == hi:
        return values[lo]
    with np.errstate(invalid="ignore"):
        result = values[lo] * (hi-position) + values[hi] * (position-lo)
    return np.where(np.isposinf(values[hi]), np.inf, result)


def diagnostics(packed, kref, rtol, used, detailed=False):
    F = unpack_fisher(packed)
    scaled = nondimensionalized_fisher(packed, kref)
    eig = np.linalg.eigvalsh(scaled)[..., ::-1]
    tolerance = rtol * np.max(np.abs(eig), axis=-1)
    invdiag, det, positive = packed_inverse_diagonal(packed)
    positive &= (eig[:, -1] > tolerance) & np.all(np.isfinite(F), axis=(1, 2))
    rank = np.sum(np.abs(eig) > tolerance[:, None], axis=1)
    reason = np.full(len(packed), "", dtype=object)
    reason[~positive] = "singular_or_below_scaled_tolerance"
    reason[eig[:, -1] < -tolerance] = "non_positive_definite_after_debias_or_S0_marginalization"
    reason[used == 0] = "all_acquired_columns_masked"
    var = np.where(positive[:, None], invdiag, np.inf)
    sd = np.sqrt(var)
    scale = np.stack([np.ones_like(kref), np.ones_like(kref), kref], axis=-1)
    relative = sd / scale
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_det_sign, scaled_logdet = np.linalg.slogdet(scaled)
        condition = np.where(positive, eig[:, 0] / eig[:, -1], np.inf)
    output = dict(F_native=F, F_scaled=scaled, positive=positive, invalidity_reason=reason,
                  scaled_eigenvalues=eig, rank=rank, native_determinant=det,
                  scaled_log_determinant=np.where(positive, scaled_logdet, np.nan),
                  scaled_condition_number=condition, crlb_variance=var, crlb_sd=sd,
                  relative_crlb_sd=relative, native_trace_crlb=var.sum(axis=-1),
                  scaled_trace_crlb=(relative**2).sum(axis=-1))
    if detailed:
        covariance = np.full_like(F, np.nan)
        scaled_covariance = np.full_like(F, np.nan)
        if np.any(positive):
            L = np.linalg.cholesky(scaled[positive])
            invL = np.linalg.solve(L, np.broadcast_to(np.eye(3), L.shape))
            Cs = np.swapaxes(invL, -2, -1) @ invL
            scaled_covariance[positive] = Cs
            covariance[positive] = Cs * scale[positive, :, None] * scale[positive, None, :]
        output.update(Finv_native=covariance, Finv_scaled=scaled_covariance)
        for label, matrix in (("native", F), ("scaled", scaled)):
            e, Q = np.linalg.eigh(matrix)
            e, Q = e[:, ::-1], Q[:, :, ::-1]
            dominant = np.argmax(np.abs(Q), axis=1)
            signs = np.take_along_axis(Q, dominant[:, None, :], axis=1)[:, 0, :]
            Q *= np.where(signs < 0, -1, 1)[:, None, :]
            sign, logdet = np.linalg.slogdet(matrix)
            tol = rtol * np.max(np.abs(e), axis=1)
            output.update({f"{label}_eigenvalues": e, f"{label}_eigenvectors": Q,
                           f"{label}_determinant": np.linalg.det(matrix),
                           f"{label}_log_determinant": np.where(positive, logdet, np.nan),
                           f"{label}_determinant_sign": sign,
                           f"{label}_log_abs_determinant": logdet,
                           f"{label}_rank": np.sum(np.abs(e) > tol[:, None], axis=1),
                           f"{label}_positive_definite": e[:, -1] > 0,
                           f"{label}_condition_number": np.linalg.cond(matrix)})
    return output


def aggregate(d, kref, settings):
    positive = d["positive"]
    medians = quantile(d["relative_crlb_sd"], 0.5)
    binding = int(np.argmax(medians))
    sorted_nodes = np.argsort(d["relative_crlb_sd"][:, binding], kind="stable")
    # A median objective is bound by central order-statistic nodes, not necessarily the worst node.
    central = sorted_nodes[(len(sorted_nodes)-1)//2:len(sorted_nodes)//2+1]
    coverage = float(positive.mean())
    A = float(quantile(d["scaled_trace_crlb"], .5))
    D = float(quantile(np.where(positive, -d["scaled_log_determinant"], np.inf), .5))
    with np.errstate(divide="ignore", invalid="ignore"):
        E = float(quantile(np.where(positive, 1/d["scaled_eigenvalues"][:, -1], np.inf), .5))
    primary = {"robust_minimax": float(medians.max()), "A_scaled_trace": A,
               "D_negative_scaled_logdet": D, "E_inverse_smallest_scaled_eigenvalue": E}
    if settings["objective"] not in primary:
        raise ValueError(f"Unknown objective {settings['objective']}")
    score = primary[settings["objective"]]
    summary = dict(objective= settings["objective"], objective_value=score,
                   robust_score=float(medians.max()), binding_parameter=PARAMS[binding],
                   binding_node_local_indices=central.tolist(),
                   worst_node_local_index=int(np.argmax(d["relative_crlb_sd"][:, binding])),
                   identifiable_count=int(positive.sum()), node_count=len(positive),
                   identifiable_fraction=coverage, feasible=bool(np.isfinite(score) and coverage >= settings["minimum_coverage"]),
                   A_scaled_trace_median=A, D_negative_scaled_logdet_median=D,
                   E_inverse_smallest_scaled_eigenvalue_median=E)
    for j, p in enumerate(PARAMS):
        values = d["relative_crlb_sd"][:, j]
        for name, q in (("q25", .25), ("median", .5), ("q75", .75), ("q90", .9), ("q95", .95), ("max", 1)):
            summary[f"relative_crlb_sd_{p}_{name}"] = float(quantile(values, q))
        summary[f"relative_crlb_sd_{p}_mean_all_nodes"] = float(np.mean(values))
        summary[f"relative_crlb_sd_{p}_mean_identifiable_only_diagnostic"] = float(np.mean(values[positive])) if positive.any() else np.nan
    for name in ("native_trace_crlb", "scaled_trace_crlb", "scaled_condition_number"):
        for qname, q in (("median", .5), ("q90", .9), ("q95", .95), ("max", 1)):
            summary[f"{name}_{qname}"] = float(quantile(d[name], q))
    summary["scaled_lambda3_median_all_nodes"] = float(quantile(d["scaled_eigenvalues"][:, -1], .5))
    return summary


class Evaluator:
    def __init__(self, substrate, config):
        self.source, self.config = substrate, config
        self.settings = config["evaluation"]
        for key in ("trust_floor", "rician_min", "pd_rtol"):
            if not np.isfinite(self.settings[key]) or self.settings[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not .5 < self.settings["minimum_coverage"] <= 1 or not self.settings["pd_rtol"] < 1:
            raise ValueError("minimum_coverage must exceed .5 and pd_rtol must be below 1")
        if self.settings["rician_basis"] not in ("phase2_repeated_mean", "single_measurement"):
            raise ValueError("Explicit Rician basis required")
        if self.settings["objective_amplitude"] not in ("fixed_S0", "marginal_S0"):
            raise ValueError("objective_amplitude must name fixed_S0 or marginal_S0")
        self.key_settings = {"schema": SCHEMA, "substrate": substrate.fingerprint,
                             "noise": config["noise"], "evaluation": self.settings,
                             "design": config["design"],
                             "code": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__), REPO/"madi/fisher_crlb.py", REPO/"scripts/run_fisher_phase2.py")}}

    def key(self, protocol):
        return digest({"protocol": canonical(protocol), **self.key_settings})

    def evaluate(self, protocol, detailed=False):
        protocol = canonical(protocol)
        ids, repeats = np.array(protocol).T
        te, sigma = noise(protocol, self.source.columns, self.config["noise"])
        raw, amp, varj, signal = self.source.get_columns(ids)
        if self.config["design"]["G_max_T_m"] is not None and np.any(self.source.gradient[ids] > self.config["design"]["G_max_T_m"]):
            raise ValueError("Requested protocol exceeds the declared G_max; no columns dropped")
        threshold = self.settings["rician_min"] * sigma
        if self.settings["rician_basis"] == "phase2_repeated_mean":
            threshold = threshold / np.sqrt(repeats[:, None])
        keep = (signal >= self.settings["trust_floor"]) & (signal >= threshold)
        weights = keep * (repeats[:, None] / sigma**2)
        tissue = raw.copy()
        if self.settings["debias"]:
            tissue[..., [0, 3, 5]] -= varj
        packed = np.einsum("cn,cnp->np", weights, tissue)
        amplitudes = np.einsum("cn,cnp->np", weights, amp)
        marginal = packed_amplitude_marginal(packed, amplitudes[:, :3], amplitudes[:, 3], 0.0)
        marginal[amplitudes[:, 3] <= 0] = 0.0
        used = np.sum(keep * repeats[:, None], axis=0)
        models, summaries = {}, {}
        for label, matrix in (("fixed_S0", packed), ("marginal_S0", marginal)):
            d = diagnostics(matrix, self.source.kref, self.settings["pd_rtol"], used, detailed)
            if label == "marginal_S0":
                # In particular never treat a missing amplitude block as known S0.
                absent = amplitudes[:, 3] <= 0
                d["invalidity_reason"][absent] = "no_amplitude_information"
            models[label] = d
            summaries[label] = aggregate(d, self.source.kref, self.settings)
        return {"protocol": protocol, "protocol_id": "p_"+digest(protocol),
                "evaluation_key": self.key(protocol), "TE_A_s": te, "sigma_single": sigma,
                "models": models, "summaries": summaries, "used_measurements": used,
                "n_kept_columns": keep.sum(axis=0), "F_undebiased": unpack_fisher(np.einsum("cn,cnp->np", weights, raw)),
                "F_debias_correction": unpack_fisher(np.einsum("cn,cnp->np", weights, raw)-packed),
                "F_amplitude_correction": unpack_fisher(packed-marginal),
                "F_S0S0": amplitudes[:, 3], "F_theta_S0": amplitudes[:, :3]}


def replace_timing(protocol, old_pair, new_pair, source):
    rows = []
    for c, n in canonical(protocol):
        d, D, b = source.columns[c]
        if (d, D) == tuple(old_pair):
            d, D = new_pair
        rows.append((source.exact_column((d, D, b)), n))
    return canonical(rows)

"""Numerical/integration checks; never regenerates historical data products."""
from __future__ import annotations
import copy
import itertools
import math
from pathlib import Path

import numpy as np

from .core import Evaluator, canonical, diagnostics, noise, replace_timing
from .search import Design, ScoreStore, optimize
from madi.fisher_crlb import amplitude_marginal_fisher, fisher_matrix, unpack_fisher
from scripts.run_fisher_phase2 import pair_contributions, evaluate as phase2_evaluate


def run_checks(source, config, single, output):
    checks = {}
    evaluator = Evaluator(source, config)
    base = evaluator.evaluate(single, detailed=True)
    doubled = evaluator.evaluate([(c, 2*n) for c, n in single], detailed=True)
    masks_same = np.array_equal(base["used_measurements"]*2, doubled["used_measurements"])
    if not masks_same:
        check_config = copy.deepcopy(config)
        check_config["evaluation"]["rician_min"] = 0.0
        evaluator = Evaluator(source, check_config)
        base = evaluator.evaluate(single, detailed=True)
        doubled = evaluator.evaluate([(c, 2*n) for c, n in single], detailed=True)
    errors = {}
    for model in base["models"]:
        a, b = base["models"][model], doubled["models"][model]
        np.testing.assert_allclose(b["F_native"], 2*a["F_native"], rtol=2e-12, atol=1e-10)
        keep = a["positive"] & b["positive"]
        if not keep.any():
            raise AssertionError("No valid node available for repetition/inverse check")
        np.testing.assert_allclose(b["Finv_native"][keep], a["Finv_native"][keep]/2, rtol=1e-8, atol=1e-8)
        errors[model] = float(np.max(np.abs(b["F_native"]-2*a["F_native"])))
    assert base["TE_A_s"] == doubled["TE_A_s"]
    checks["linear_repetitions"] = {"pass": True, "max_abs_F_errors": errors,
                                    "mask_policy": "unchanged masks; Rician disabled only for check if threshold crossing"}
    expected = max(sum(source.columns[c, :2]) for c, _ in single)/1000+.014
    np.testing.assert_allclose(base["TE_A_s"], expected, rtol=0, atol=1e-15)
    short, long = (4., 15.), (4., 40.)
    test = source.protocol([(*short, 1000, 2), (*long, 2500, 2)])
    changed = replace_timing(test, short, (4., 20.), source)
    later = replace_timing(test, short, (4., 60.), source)
    assert np.isclose(noise(changed, source.columns, config["noise"])[0], .058)
    assert np.isclose(noise(later, source.columns, config["noise"])[0], .078)
    checks["adaptive_common_TE"] = {"pass": True, "base_TE_s": expected,
                                    "retained_long_group_TE_s": .058, "new_longest_group_TE_s": .078}
    packed = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, -1], [0, 0, 0, 0, 0, 0]], float)
    d = diagnostics(packed, np.ones(3)*5, 1e-12, np.array([1, 2, 0]), True)
    assert not d["positive"].any() and np.isnan(d["Finv_native"]).all()
    assert np.isinf(d["relative_crlb_sd"]).all() and all(d["invalidity_reason"])
    checks["singular_non_PD_masked"] = {"pass": True, "reasons": d["invalidity_reason"].tolist()}
    # Local Phase-2 calculation under the SAME adaptive noise: independent accumulation
    # calls the current authoritative helper only for this validation sample. The
    # production evaluator never re-differences stored signals.
    valid = np.flatnonzero(base["models"]["fixed_S0"]["positive"])
    k = int(valid[np.argmin(np.abs(valid-source.nearest_local))])
    node = int(source.node_ids[k])
    single_table = {name: (value[node:node+1] if isinstance(value, np.ndarray) else value)
                    for name, value in source.table.items()}
    tissue, amplitude = np.zeros((1, 6)), np.zeros((1, 4))
    sigma = base["sigma_single"]
    for c, n in single:
        threshold = evaluator.settings["rician_min"]*sigma
        if evaluator.settings["rician_basis"] == "phase2_repeated_mean":
            threshold /= np.sqrt(n)
        t, a, debias = pair_contributions(single_table, source.vectors, source.variance,
            np.array([source.domain.position_of[c]]), 1/sigma**2, source.n_ensembles,
            evaluator.settings["trust_floor"], threshold)
        if not evaluator.settings["debias"]:
            t[..., [0, 3, 5]] += debias
        tissue += n*t[0]; amplitude += n*a[0]
    reference = phase2_evaluate(tissue, amplitude, source.kref[k:k+1], 1.0, np.inf)
    ours = base["models"]["fixed_S0"]["F_native"][k]
    reference_F = unpack_fisher(tissue)[0]
    relative_F_error = np.linalg.norm(ours-reference_F)/np.linalg.norm(reference_F)
    assert relative_F_error < 2e-4, relative_F_error
    np.testing.assert_allclose(base["models"]["fixed_S0"]["relative_crlb_sd"][k], reference["_relative"][0], rtol=2e-3)
    checks["phase2_local_cached_calculation"] = {
        "pass": True, "node_index": node, "relative_F_frobenius_error": float(relative_F_error),
        "reference_fixed_relative_SD": reference["_relative"][0].tolist(),
        "new_fixed_relative_SD": base["models"]["fixed_S0"]["relative_crlb_sd"][k].tolist(),
        "note": "Same protocol, weighting, masks and endpoint-only debias as Phase-2 helper, but new common adaptive TE. Small differences expected: stored Phase-1 float32 J versus Phase-2 float64 endpoint calculation. Not a reproduction of historical noise assumptions."}
    # Also verify Schur complement against an explicit complete 4x4 inverse.
    rng = np.random.default_rng(910)
    j = rng.normal(size=(10, 3)); signal = rng.uniform(.1, 1, 10); sd=.07
    complete = np.column_stack([j, signal])
    four = fisher_matrix(complete, sd)
    result = amplitude_marginal_fisher(j, signal, sd)
    np.testing.assert_allclose(np.linalg.inv(result["F_marginal_s0"]), np.linalg.inv(four)[:3, :3], rtol=1e-10)
    checks["S0_explicit_4x4_inverse"] = {"pass": True}
    # Exhaustive tiny discrete domain is an external oracle for the SAME optimizer.
    reduced_cfg = copy.deepcopy(config)
    reduced_cfg["design"].update(reserved_b0_volumes=2, min_repetitions=1,
                                 max_repetitions=6, max_distinct_columns=5, max_timing_pairs=1)
    reduced_cfg["evaluation"]["rician_min"] = 0.0
    pair = tuple(source.columns[next(c for c, _ in single if source.columns[c, 2] > 0), :2])
    original = [c for c, _ in single if source.columns[c, 2] > 0]
    extras = [source.exact_column((*pair, b)) for b in (500, 1000, 2500, 4000, 6000)]
    pool = np.array(list(dict.fromkeys(original+extras))[:4])
    reduced_cfg["design"].update(delta_values_ms=[pair[0]], Delta_values_ms=[pair[1]],
                                  b_values_s_mm2=source.columns[pool, 2].tolist())
    reduced = Design(source, reduced_cfg["design"], 8)
    reduced_eval = Evaluator(source, reduced_cfg)
    reduced_store = ScoreStore(output / "validation_evaluations.sqlite", reduced_eval, "reduced_oracle")
    b0 = source.exact_column((*pair, 0))
    oracle = []
    for counts in itertools.product(range(7), repeat=len(pool)):
        if sum(counts) != 6:
            continue
        protocol = canonical(list(zip(pool, counts))+[(b0, 2)])
        result = reduced_store.evaluate(protocol, reduced, "exhaustive_oracle")
        if result["feasible"]:
            oracle.append((result["score"], protocol))
    if not oracle:
        raise AssertionError("Reduced validation space has no feasible protocols")
    exact_score, exact_protocol = min(oracle)
    # Independent history and result registry; cached oracle values are reusable,
    # but the search cannot see the list of oracle protocols or the optimum.
    reduced_store.close()
    search_store = ScoreStore(output / "validation_evaluations.sqlite", reduced_eval, "reduced_optimizer")
    search_cfg = dict(config["optimizer"], starts=3, iterations_per_start=250, local_passes=3, top_k=3)
    searched = optimize(reduced, search_store, search_cfg)
    best = searched["top"][0]
    np.testing.assert_allclose(best[1]["score"], exact_score, rtol=1e-10)
    checks["exhaustive_optimizer_comparison"] = dict(pass_=True, total_allocations=84,
        feasible_allocations=len(oracle), best_exhaustive_score=exact_score, optimizer_score=best[1]["score"],
        exhaustive_protocol=source.specification(exact_protocol), optimizer_protocol=source.specification(best[0]),
        budget=8, random_seed=search_cfg["random_seed"], search_calls=search_store.calls)
    checks["exhaustive_optimizer_comparison"]["pass"] = checks["exhaustive_optimizer_comparison"].pop("pass_")
    search_store.close()
    return checks

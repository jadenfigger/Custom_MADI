#!/usr/bin/env python3
"""Print the Phase-2 report as the tables `docs/fisher_phase2.md` is built from.

A reporting tool only.  It reads `phase2_report.json` and formats it; nothing
here recomputes a Fisher matrix.  Kept in `scripts/` so the record's tables are
reproducible from the JSON rather than transcribed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

P = ("log_rho", "log_V", "k_io")


def _f(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if v == float("inf"):
        return "inf"
    if v != v:
        return "nan"
    return f"{v:.{digits}g}"


def _pairs(item):
    return " + ".join(f"({a:g},{b:g})" for a, b in item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--regime", default="known_amplitude")
    args = parser.parse_args()
    r = json.loads(args.report.read_text(encoding="utf-8"))

    print(f"# Phase 2 report: {args.report}")
    print(f"runtime {r.get('runtime_seconds')} s; N={r['budget_images_N']}; SNR={r['snr_at_b0']}; "
          f"T2={r['T2_ms']}; t_epi={r['t_epi_ms']}; node_stride={r['node_stride']}")
    # The domain the substrate represented is part of reading any result: a
    # scenario evaluated against a restricted substrate is not the scenario it
    # names.  Reports written before 2026-09-06 carry no declaration.
    print(r.get("column_domain_banner", "COLUMN DOMAIN UNDECLARED — report predates the "
                                        "2026-09-06 analysis-domain contract"))
    n = r["evaluation_nodes"]
    print(f"\n## Nodes: {n['count']} = {n['rho_V_pairs_retained']} (rho,V) pairs x {len(n['k_io_indices'])} k_io; "
          f"{n['rho_V_pairs_lost_to_incomplete_stencils']} of {n['rho_V_pairs_in_mask']} (rho,V) pairs lost; "
          f"k_io>30 fraction {n['k_io_above_30_node_fraction']:.3f}")
    print("lost (rho,V) pairs:", n["lost_rho_V_pairs"])

    print("\n## Derivative cross-check (recomputed J vs stored Phase-1 J)")
    for axis, v in r["derivative_cross_check"].items():
        print(f"  {axis:5} cells {v['cells']:5}  max|diff| {_f(v['max_abs_difference'],3)}  max rel {_f(v['max_relative_difference'],3)}")

    print("\n## Debias calibration: endpoint-only / exact CRN Var(J), 200 diagnostic columns")
    for axis, v in r["debias_calibration"]["axes"].items():
        s = v["endpoint_only_over_exact"]
        print(f"  {axis:5} median {_f(s.get('median'),3)}  q95 {_f(s.get('q95'),3)}  max {_f(s.get('max'),3)}  "
              f"(mean exact {_f(v['mean_exact'],3)}, mean endpoint {_f(v['mean_endpoint_only'],3)})")

    for scenario, sc in r["scenarios"].items():
        print(f"\n\n# ===== {scenario.upper()} ({sc['gradient_limit_T_per_m']*1e3:.0f} mT/m; {sc['feasible_columns']} feasible columns) =====")
        print("candidate pairs by size:", sc["candidate_pairs_by_subset_size"])
        for size, d in sc["declared_sweep_size"].items():
            print(f"  size {size}: m1 {d['m1_arms']}, m2 {d['m2_arms']}, m3-if-exhaustive {d['m3_arms_if_exhaustive']}, "
                  f"m4-if-exhaustive {d['m4_arms_if_exhaustive']}, b-subsets/pair if exhaustive {d['b_subsets_if_exhaustive_per_pair']}")
        for size, arms in sc["arms"].items():
            print(f"\n## size {size}")
            print(f"{'arm':4} {'timing pairs (delta,Delta) ms':42} {'cols':>4} {'avg':>5} {'ident':>6} {'minimax':>8} {'argmax':>7} "
                  f"{'med logrho':>10} {'med logV':>9} {'med kio':>8} {'rhoV-only':>9} {'A-opt':>8} {'D-opt':>7} {'k rho':>6} {'k V':>6} {'k kio':>6}")
            for label in ("m1", "m2", "m3", "m4"):
                if label not in arms:
                    continue
                e = arms[label]
                rep = e["report"]
                g = rep["regimes"][args.regime]
                km = g["kappa_mean_per_parameter"]
                print(f"{label:4} {_pairs(rep['timing_pairs_ms']):42} {rep['columns_used']:4} {rep['averages_per_column']:5.1f} "
                      f"{g['positive_definite_fraction']:6.3f} {_f(g['minimax_score']):>8} {g['argmax_parameter']:>7} "
                      f"{_f(g['relative_crlb_median_over_all_nodes_per_parameter'][0]):>10} "
                      f"{_f(g['relative_crlb_median_over_all_nodes_per_parameter'][1]):>9} "
                      f"{_f(g['relative_crlb_median_over_all_nodes_per_parameter'][2]):>8} "
                      f"{_f(g['minimax_rho_V_only_DIAGNOSTIC_NOT_A_CRITERION']):>9} "
                      f"{_f(g['a_optimality_mean_sum_squared_relative_crlb']):>8} {_f(g['d_optimality_mean_log10_det'],3):>7} "
                      f"{_f(km[0],3):>6} {_f(km[1],3):>6} {_f(km[2],3):>6}")
            for name, v in arms.get("reference_protocols_context_only", {}).items():
                if "unavailable" in v:
                    print(f"ref  {name:42} {v['unavailable']}")
                    continue
                g = v["regimes"][args.regime]; km = g["kappa_mean_per_parameter"]
                print(f"ref  {name+' '+_pairs(v['timing_pairs_ms']):42} {v['columns_used']:4} {v['averages_per_column']:5.1f} "
                      f"{g['positive_definite_fraction']:6.3f} {_f(g['minimax_score']):>8} {g['argmax_parameter']:>7} "
                      f"{_f(g['relative_crlb_median_over_all_nodes_per_parameter'][0]):>10} "
                      f"{_f(g['relative_crlb_median_over_all_nodes_per_parameter'][1]):>9} "
                      f"{_f(g['relative_crlb_median_over_all_nodes_per_parameter'][2]):>8} "
                      f"{_f(g['minimax_rho_V_only_DIAGNOSTIC_NOT_A_CRITERION']):>9} "
                      f"{_f(g['a_optimality_mean_sum_squared_relative_crlb']):>8} {_f(g['d_optimality_mean_log10_det'],3):>7} "
                      f"{_f(km[0],3):>6} {_f(km[1],3):>6} {_f(km[2],3):>6}")

            m1 = arms["m1"]
            print(f"\n  m1 sweep ({m1['n_arms']} arms): ranked by {m1['ranked_by']}; finite minimax {m1['arms_with_a_finite_minimax']}; "
                  f"argmax counts {m1['argmax_parameter_counts']}")
            print(f"     identifiable fraction: {({k: _f(v,3) for k, v in m1['identifiable_fraction_spread'].items()})}")
            print(f"     score spread: {({k: _f(v,4) for k, v in m1['score_spread'].items()})}")
            print(f"     top 5 (delta, Delta, score, ident): {[[_f(x,3) for x in t] for t in m1['top_10_pairs_ms'][:5]]}")
            if "m2" in arms:
                m2 = arms["m2"]
                print(f"  m2 sweep ({m2['n_arms']} arms): finite minimax {m2['arms_with_a_finite_minimax']}; argmax counts {m2['argmax_parameter_counts']}")
                print(f"     score spread: {({k: _f(v,4) for k, v in m2['score_spread'].items()})}")
                print(f"     identifiable fraction: {({k: _f(v,3) for k, v in m2['identifiable_fraction_spread'].items()})}")
                print(f"     greedy-from-best-m1: {_pairs(m2['greedy_from_best_m1']['pairs_ms'])} score {_f(m2['greedy_from_best_m1']['score'])}; "
                      f"greedy optimality gap {_f(m2['greedy_optimality_gap'],3)}")

            print("\n  Monte-Carlo debias effect (bias trap):")
            for label in ("m1", "m2", "m3", "m4"):
                if label not in arms:
                    continue
                d = arms[label]["report"]["monte_carlo_debias_effect"]
                print(f"     {label}: identifiable {d['identifiable_fraction_undebiased']:.3f} -> {d['identifiable_fraction_debiased']:.3f}; "
                      f"minimax {_f(d['minimax_undebiased'])} -> {_f(d['minimax_debiased'])}; "
                      f"debias/diag {[_f(x,2) for x in d['mean_debias_over_fisher_diagonal']]}")

            print("\n  n0_eff sweep (minimax; per-node median CRLB ratio vs fixed S0):")
            for label in ("m1", "m2", "m3", "m4"):
                if label not in arms:
                    continue
                rep = arms[label]["report"]
                row = []
                for reg, g in rep["regimes"].items():
                    row.append(f"{reg.replace('n0_eff_', 'n0=').replace('known_amplitude', 'known')}: {_f(g['minimax_score'])} "
                               f"[{','.join(_f(x,3) for x in g['crlb_ratio_marginal_over_fixed_median_per_node'])}]")
                print(f"     {label} ref {_pairs([rep['amplitude_reference_pair_ms']])}: " + " | ".join(row))

            for label in ("m1", "m2"):
                if label not in arms or "kappa_worst_node" not in arms[label]["report"]:
                    continue
                rep = arms[label]["report"]
                print(f"\n  {label} kappa distribution (identifiable nodes):")
                for name, s in rep["kappa_distribution_per_parameter"].items():
                    w = rep["kappa_worst_node"][name]
                    print(f"     {name:8} median {_f(s.get('median'),3)} q95 {_f(s.get('q95'),3)} max {_f(s.get('max'),3)}; "
                          f"> threshold {rep['kappa_above_threshold_fraction'][name]:.3f}; worst at rho={w['rho_cells_per_uL']:.3g} "
                          f"V={w['V_pL']:.3g} k_io={w['k_io_s_inv']:g} (v_i={w['v_i']:.3f})")
                print(f"  {label} relative CRLB distribution (identifiable nodes):")
                for name, s in rep["relative_crlb_distribution_per_parameter"].items():
                    print(f"     {name:8} q05 {_f(s.get('q05'),3)} median {_f(s.get('median'),3)} q95 {_f(s.get('q95'),3)}")
                print(f"  {label} k_io region split:")
                for region, s in rep["k_io_region_split"].items():
                    print(f"     {region:10} nodes {s['nodes']} identifiable {s['positive_definite']} "
                          f"median relCRLB {[_f(x,3) for x in s['relative_crlb_median_per_parameter']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

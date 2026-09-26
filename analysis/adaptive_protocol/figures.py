"""Publication plots from exported numerical tables only (no Fisher calls)."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .core import NOISE_FORMULA, PARAMS, clean
from .export import write_json

LABELS = (r"$\rho$", r"$V$", r"$k_{io}$")
COLORS = ["#2166ac", "#b35806", "#1b7837", "#762a83", "#555555"]


def numeric(series):
    return pd.to_numeric(series.replace({"+inf": np.inf, "-inf": -np.inf}), errors="coerce")


def make_figures(output, config, run_id):
    output = Path(output)
    summary = pd.read_csv(output/"protocol_summary.csv")
    summary = summary[summary.run_id == run_id]
    nodes = pd.read_csv(output/"node_metrics.csv", low_memory=False)
    nodes = nodes[nodes.evaluation_key.isin(summary.evaluation_key)]
    history = pd.read_csv(output/"optimization_history.csv")
    history = history[history.run_id == run_id]
    columns = pd.read_csv(output/"acquisition_columns.csv")
    sweeps = pd.read_csv(output/"protocol_sweeps.csv")
    sweeps = sweeps[sweeps.run_id == run_id]
    for frame in (summary, nodes, history, sweeps):
        for col in frame:
            if any(k in col for k in ("score", "crlb", "lambda", "condition_number", "objective_value", "best_so_far")):
                frame[col] = numeric(frame[col])
    root = output/"figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 120})
    files = []

    def save(fig, category, name, metadata):
        folder = root/category; folder.mkdir(parents=True, exist_ok=True)
        metadata = dict(run_id=run_id, **config["noise"], noise_formula=NOISE_FORMULA,
                        config=config, **metadata)
        text = json.dumps(clean(metadata), sort_keys=True)
        for extension in config["figures"]["formats"]:
            path = folder/f"{name}.{extension}"
            embedded = {"Description": text} if extension in ("png", "svg") else {"Subject": text, "Creator": "MADI adaptive protocol workflow"}
            fig.savefig(path, dpi=config["figures"]["dpi"], bbox_inches="tight", metadata=embedded)
            files.append(str(path.relative_to(output)))
        write_json(folder/f"{name}.metadata.json", metadata)
        plt.close(fig)

    primary_model = config["evaluation"]["objective_amplitude"]
    for budget in config["budgets"]:
        sub = summary[(summary.total_measurements == budget) & (summary.amplitude_model == primary_model)].copy()
        selected = sub.sort_values("objective_value")
        if selected.empty:
            continue
        winner = selected.iloc[0]
        pid, key = winner.protocol_id, winner.evaluation_key
        # Parameter maps preserve every measured node; missing grid corners stay gray.
        for amplitude in ("fixed_S0", "marginal_S0"):
            selected_nodes = nodes[(nodes.evaluation_key == key) & (nodes.amplitude_model == amplitude)]
            ks = [k for k in config["figures"]["k_io_values"] if k in set(selected_nodes.actual_k_io)]
            metrics = [(f"relative_crlb_sd_{p}", f"Relative SD bound: {label}", True) for p, label in zip(PARAMS, LABELS)]
            metrics += [("identifiable", "Identifiable coverage", False),
                        ("scaled_trace_crlb", "Dimensionless trace CRLB", True),
                        ("scaled_condition_number", "Scaled condition number", True),
                        ("scaled_lambda3", "Smallest scaled eigenvalue", False)]
            for field, title, logarithmic in metrics:
                fig, axes = plt.subplots(1, len(ks), figsize=(4.3*len(ks), 3.7), squeeze=False, layout="constrained")
                values = numeric(selected_nodes[field]).to_numpy(float)
                if field == "identifiable":
                    values = selected_nodes[field].astype(float).to_numpy()
                finite = values[np.isfinite(values) & ((values > 0) if logarithmic else True)]
                if logarithmic and len(finite):
                    vmin, vmax = np.quantile(finite, [.02, .98]); vmax=max(vmax, vmin*1.01)
                    norm = LogNorm(max(vmin, 1e-30), vmax)
                elif field == "identifiable":
                    norm = Normalize(0, 1)
                else:
                    lim = max(np.quantile(np.abs(finite), .98), 1e-20) if len(finite) else 1
                    norm = Normalize(-lim, lim)
                cmap = plt.get_cmap("viridis" if logarithmic or field == "identifiable" else "RdBu_r").copy()
                cmap.set_bad("#dedede")
                for ax, k in zip(axes[0], ks):
                    part = selected_nodes[selected_nodes.actual_k_io == k].copy()
                    part[field] = numeric(part[field]) if field != "identifiable" else part[field].astype(float)
                    grid = part.pivot(index="actual_V", columns="actual_rho", values=field).sort_index()
                    z = grid.to_numpy(float); z[~np.isfinite(z)] = np.nan
                    if logarithmic:
                        z[z <= 0] = np.nan
                    m = ax.pcolormesh(grid.columns.to_numpy(float), grid.index.to_numpy(float), np.ma.masked_invalid(z),
                                     norm=norm, cmap=cmap, shading="nearest", rasterized=False)
                    ax.set(xscale="log", yscale="log", xlabel=r"$\rho$ (cells/$\mu$L)", ylabel=r"$V$ (pL)", title=rf"$k_{{io}}={k:g}$ s$^{{-1}}$")
                fig.colorbar(m, ax=axes.ravel().tolist(), shrink=.8, label=title, extend="both" if logarithmic else "neither")
                fig.suptitle(f"N={budget}, {amplitude.replace('_', ' ')} | {title}\nTE={1000*winner.TE_A_s:.0f} ms; gray = invalid or outside node domain", fontsize=12)
                save(fig, "parameter_space", f"N{budget}_{amplitude}_{field}", dict(protocol_id=pid, evaluation_key=key, amplitude_model=amplitude, metric=field))
        # Compare distributions with invalid fractions explicitly beside each name.
        compare = selected.drop_duplicates("evaluation_key").head(5)
        names = [r.protocol_name for _, r in compare.iterrows()]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
        for ax, p, label in zip(axes, PARAMS, LABELS):
            distributions = []
            annotated = []
            for _, r in compare.iterrows():
                d = nodes[(nodes.evaluation_key == r.evaluation_key) & (nodes.amplitude_model == primary_model)]
                x = numeric(d[f"relative_crlb_sd_{p}"]).to_numpy()
                distributions.append(x[np.isfinite(x)])
                annotated.append(f"{r.protocol_name}\n{100*r.identifiable_fraction:.1f}% valid")
            ax.boxplot(distributions, tick_labels=annotated, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "#d8e6f2"}, medianprops={"color": "#b35806"})
            ax.set(yscale="log", ylabel="Relative SD bound (finite nodes)", title=label)
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"N={budget}, {primary_model.replace('_',' ')} | Node-wise uncertainty\nInvalid nodes remain +infinity in objective; boxes display finite values", fontsize=12)
        save(fig, "comparison", f"N{budget}_distributions", dict(protocol_ids=compare.protocol_id.tolist(), amplitude_model=primary_model))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for i, (_, r) in enumerate(compare.iterrows()):
            both = summary[(summary.evaluation_key == r.evaluation_key)].set_index("amplitude_model")
            for j, label in enumerate(("fixed_S0", "marginal_S0")):
                v = both.loc[label, "robust_score"]
                if np.isfinite(v):
                    axes[0].bar(i+(j-.5)*.34, v, width=.32, color=(COLORS[0], COLORS[1])[j], label=label if i==0 else None)
            if np.isfinite(r.robust_score):
                axes[1].scatter(r.identifiable_fraction, r.robust_score, s=60, color=COLORS[i%len(COLORS)], label=r.protocol_name)
        axes[0].set(xticks=np.arange(len(compare)), xticklabels=names, ylabel="Robust minimax relative SD", yscale="log")
        axes[0].tick_params(axis="x", labelrotation=25); axes[0].legend(frameon=False)
        axes[1].set(xlabel="Identifiable fraction", ylabel="Robust minimax relative SD", yscale="log", xlim=(0,1))
        axes[1].axvline(config["evaluation"]["minimum_coverage"], ls="--", color="gray", lw=1)
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Equal-budget protocol comparison: N={budget}", fontsize=12)
        save(fig, "comparison", f"N{budget}_coverage_score", dict(protocol_ids=compare.protocol_id.tolist()))
        acquisition = columns[columns.evaluation_key == key].drop_duplicates("full_column")
        fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, .35*len(acquisition)+1.4)), layout="constrained")
        table_data = [[f"{r.delta_ms:g}", f"{r.Delta_ms:g}", f"{r.b_s_mm2:g}", f"{r.averages:g}", f"{r.gradient_T_m:.3f}"] for _, r in acquisition.iterrows()]
        axes[0].axis("off")
        table = axes[0].table(cellText=table_data, colLabels=["δ (ms)", "Δ (ms)", "b (s/mm²)", "Volumes", "G (T/m)"], loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.45)
        labels=[f"{r.delta_ms:g}/{r.Delta_ms:g} ms, b={r.b_s_mm2:g}" for _, r in acquisition.iterrows()]
        axes[1].barh(labels, acquisition.averages, color=[COLORS[1] if b==0 else COLORS[0] for b in acquisition.b_s_mm2])
        axes[1].set(xlabel="Acquired volumes", title="Integer repetition allocation")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True)); axes[1].invert_yaxis()
        fig.suptitle(f"Best found, N={budget} | TE={winner.TE_A_s*1000:.0f} ms | σ={winner.sigma_single:.4g}\n{primary_model.replace('_',' ')}; score={winner.objective_value:.4g}; coverage={winner.identifiable_fraction:.1%}", fontsize=12)
        save(fig, "optimization", f"N{budget}_best_acquisition", dict(protocol_id=pid, acquisitions=table_data))
        h = history[history.budget == budget]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
        best = numeric(h.best_so_far).to_numpy(); best[~np.isfinite(best)] = np.nan
        axes[0].plot(h.evaluation_id, best, color=COLORS[0]); axes[0].set(ylabel="Best feasible objective", yscale="log")
        axes[1].scatter(h.evaluation_id, h.coverage, c=h.feasible.astype(int), s=4, cmap="coolwarm", alpha=.4)
        axes[1].axhline(config["evaluation"]["minimum_coverage"], ls="--", color="gray")
        axes[1].set(xlabel="Candidate evaluation request (cache hits included)", ylabel="Identifiable fraction", ylim=(0,1))
        fig.suptitle(f"N={budget} | Discrete optimizer progress ({primary_model})")
        save(fig, "optimization", f"N{budget}_history", dict(evaluation_requests=len(h), objective=config["evaluation"]["objective"]))
        s = sweeps[sweeps.budget == budget]
        timing = s[s.sweep_kind == "one_timing_group"]
        fig, axes = plt.subplots(1, 2, figsize=(11,4), layout="constrained")
        for ax, field, label in ((axes[0], "score", "Robust objective"), (axes[1], "TE_A_s", "Whole-protocol TE (s)")):
            grid=timing.pivot(index="delta_ms", columns="Delta_ms", values=field)
            z=grid.to_numpy(float); z[~np.isfinite(z)] = np.nan
            im=ax.imshow(np.ma.masked_invalid(z), origin="lower", aspect="auto", cmap="viridis")
            ax.set(xticks=np.arange(len(grid.columns)), xticklabels=[f"{v:g}" for v in grid.columns],
                   yticks=np.arange(len(grid.index)), yticklabels=[f"{v:g}" for v in grid.index], xlabel="Δ (ms)", ylabel="δ (ms)", title=label)
            fig.colorbar(im, ax=ax, shrink=.85)
        fig.suptitle(f"N={budget} | Move one timing group; retain remaining acquisitions", fontsize=12)
        save(fig, "protocol_space", f"N{budget}_one_group_timing", dict(protocol_id=pid, sweep_group=timing.sweep_group_json.iloc[0]))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
        for ax, kind, x, xlabel in ((axes[0],"one_b_column","b_s_mm2","Changed column b (s/mm²)"),
                                    (axes[1],"timing_diversity","n_timing_pairs","Distinct timing pairs"),
                                    (axes[2],"repeat_transfer","allocation","Volumes transferred")):
            part=s[s.sweep_kind == kind].sort_values(x)
            for y, label, color in (("score", "Robust score", COLORS[0]), ("scaled_trace_median", "Scaled trace CRLB", COLORS[1])):
                values=part[y].to_numpy(float); values[~np.isfinite(values)] = np.nan
                ax.plot(part[x], values, "o-", ms=3, label=label, color=color)
            ax.set(xlabel=xlabel, ylabel="Dimensionless loss", yscale="log", title=kind.replace("_"," "))
            ax.grid(alpha=.15)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"N={budget} | Equal-volume protocol sweeps; adaptive TE recomputed", fontsize=12)
        save(fig, "protocol_space", f"N{budget}_b_diversity_allocation", dict(protocol_id=pid, note="B and repeat panels alter one aspect of best protocol. Timing-diversity panel compares complete balanced protocols; actual timing spread in CSV."))
    write_json(output/"figure_manifest.json", {"run_id": run_id, "files": files})
    return files

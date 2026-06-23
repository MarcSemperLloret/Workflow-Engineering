"""
Create manuscript-ready pilot workflow figures from analysis CSVs.

Outputs are saved under data/analysis/pilot_workflow_ablation/figures in PNG,
SVG, and PDF. The figures are intentionally simple: they prioritize readable
axis labels and grayscale-compatible encodings over decorative styling.
"""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
ANALYSIS = ROOT / "data" / "analysis" / "pilot_workflow_ablation"
FIGDIR = ANALYSIS / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#666666"
RED = "#D55E00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_workflow_accuracy() -> None:
    rows = read_csv(ANALYSIS / "figure_workflow_accuracy.csv")
    labels = [fill(row["display_label"], 20) for row in rows]
    acc18 = [float(row["accuracy_18_objective"]) for row in rows]
    acc20 = [float(row["accuracy_20_fields"]) for row in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x, acc18, color=BLUE, marker="o", linewidth=2.0, label="Primary metric: 18 objective fields")
    ax.plot(x, acc20, color=GRAY, marker="s", linewidth=1.4, linestyle="--", label="Including diagnostics")
    ax.set_ylim(0.35, 0.88)
    ax.set_ylabel("Alias-aware exact accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    save_figure(fig, "figure_workflow_accuracy")


def plot_field_group_accuracy() -> None:
    rows = read_csv(ANALYSIS / "figure_field_group_accuracy_final.csv")
    labels = [row["field_group"].replace("_", " ") for row in rows]
    acc = [float(row["accuracy"]) for row in rows]
    y = range(len(rows))
    colors = [GREEN if value >= 0.9 else BLUE if value >= 0.8 else ORANGE if value >= 0.7 else RED for value in acc]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.barh(list(y), acc, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Alias-aware exact accuracy")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, value in enumerate(acc):
        ax.text(value + 0.015, i, f"{value:.2f}", va="center", fontsize=8)
    style_axes(ax)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "figure_field_group_accuracy_final")


def plot_posthoc_triage() -> None:
    rows = [row for row in read_csv(ANALYSIS / "pilot_posthoc_triage_simulation.csv") if row["tier"] != "excluded_diagnostic"]
    labels = [row["tier"].replace("_", " ") for row in rows]
    coverage = [float(row["coverage_excluding_diagnostics"]) for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    x = range(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar([i - width / 2 for i in x], coverage, width=width, color=ORANGE, label="Coverage")
    ax.bar([i + width / 2 for i in x], accuracy, width=width, color=BLUE, label="Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    save_figure(fig, "figure_posthoc_triage_final")


def plot_error_taxonomy() -> None:
    rows = read_csv(ANALYSIS / "pilot_residual_error_taxonomy_summary_final.csv")
    labels = [fill(row["error_category"].replace("_", " "), 24) for row in rows]
    errors = [int(row["errors"]) for row in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.barh(list(y), errors, color=BLUE)
    ax.set_xlabel("Residual errors")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, value in enumerate(errors):
        ax.text(value + 0.6, i, str(value), va="center", fontsize=8)
    style_axes(ax)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "figure_residual_error_taxonomy_final")


def model_family_label(model: str) -> str:
    labels = {
        "gemma3:12b": "Gemma\n12B",
        "llama3.1:8b": "Llama 3.1\n8B",
        "mistral-nemo:12b": "Mistral-Nemo\n12B",
        "qwen3-coder:30b": "Qwen3-Coder\n30B",
    }
    return labels.get(model, model.replace(":", "\n"))


def compact_model_family_label(model: str) -> str:
    labels = {
        "gemma3:12b": "Gemma\n12B",
        "llama3.1:8b": "Llama\n8B",
        "mistral-nemo:12b": "Mistral\n12B",
        "qwen3-coder:30b": "Qwen3\n30B",
    }
    return labels.get(model, model_family_label(model))


def plot_model_sensitivity() -> None:
    rows = sorted(
        read_csv(ANALYSIS / "pilot_model_sensitivity_table.csv"),
        key=lambda row: model_family_label(row["model"]).replace("\n", " "),
    )
    access_workflows = {
        "gemma3:12b": "w10_schema_v4_access_gemma3_12b",
        "llama3.1:8b": "w10_schema_v4_access_llama3_1_8b",
        "mistral-nemo:12b": "w10_schema_v4_access_mistral_nemo_12b",
        "qwen3-coder:30b": "w10_schema_v4_access",
    }
    labels = [compact_model_family_label(row["model"]) for row in rows]
    raw = [float(row["accuracy_18_raw"]) for row in rows]
    access = [float(row["accuracy_18_access"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.34

    fig, (ax, heat) = plt.subplots(
        1,
        2,
        figsize=(7.8, 3.35),
        gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.34},
    )
    ax.bar([i - width / 2 for i in x], raw, width=width, color=GRAY, label="Guided workflow")
    ax.bar([i + width / 2 for i in x], access, width=width, color=BLUE, label="+ access rules")
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("18-field primary accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend(frameon=False, loc="upper left")
    for i, value in enumerate(access):
        ax.text(i + width / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    style_axes(ax)
    ax.text(-0.16, 1.03, "A", transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")

    group_rows = read_csv(ANALYSIS / "pilot_field_group_accuracy_by_workflow.csv")
    group_order = ["bibliographic", "task_dataset", "access", "temporal_horizon", "temporal_frequency", "graph_nodes"]
    group_labels = ["Bibliographic", "Task/dataset", "Access", "Horizon", "Frequency", "Graph nodes"]
    values = []
    heat_labels = []
    for row in rows:
        model = row["model"]
        workflow = access_workflows[model]
        heat_labels.append(model_family_label(model).replace("\n", " "))
        by_group = {
            group_row["field_group"]: float(group_row["accuracy"])
            for group_row in group_rows
            if group_row["workflow"] == workflow
        }
        values.append([by_group[group] for group in group_order])

    image = heat.imshow(values, aspect="auto", vmin=0.45, vmax=1.0, cmap="cividis")
    heat.set_xticks(range(len(group_order)))
    heat.set_xticklabels(group_labels, rotation=40, ha="right")
    heat.set_yticks(range(len(heat_labels)))
    heat.set_yticklabels(heat_labels)
    heat.tick_params(length=0)
    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            color = "white" if value < 0.68 else "black"
            heat.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    for spine in heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=heat, fraction=0.046, pad=0.02)
    cbar.set_label("Field-group accuracy", rotation=90)
    heat.text(-0.14, 1.03, "B", transform=heat.transAxes, fontweight="bold", fontsize=10, va="bottom")
    save_figure(fig, "figure_model_sensitivity")


def plot_workflow_model_field_profiles() -> None:
    group_rows = read_csv(ANALYSIS / "pilot_field_group_accuracy_by_workflow.csv")
    group_order = ["bibliographic", "task_dataset", "access", "temporal_horizon", "temporal_frequency", "graph_nodes"]
    group_labels = ["Bibliographic", "Task/dataset", "Access", "Horizon", "Frequency", "Graph nodes"]
    configs = [
        ("metadata_prior_baseline", "Prior / rules", "Rules", "prior"),
        ("w0_direct_50", "Direct / Qwen2.5", "Qwen2.5 7B", "direct"),
        ("w2_schema_normalized_50", "Schema / Qwen2.5", "Qwen2.5 7B", "schema"),
        ("w3_field_groups_50", "Grouped / Qwen2.5", "Qwen2.5 7B", "grouped"),
        ("w4_hybrid_w2_w3_50", "Hybrid / Qwen2.5", "Qwen2.5 7B", "hybrid"),
        ("w11_cot_grouped", "CoT / Qwen3", "Qwen3 30B", "cot"),
        ("w10_schema_v4_access_gemma3_12b", "Guided+access / Gemma", "Gemma 12B", "guided"),
        ("w10_schema_v4_access_llama3_1_8b", "Guided+access / Llama", "Llama 8B", "guided"),
        ("w10_schema_v4_access_mistral_nemo_12b", "Guided+access / Mistral", "Mistral 12B", "guided"),
        ("w10_schema_v4_access", "Guided+access / Qwen3", "Qwen3 30B", "guided"),
    ]
    model_colors = {
        "Rules": "#666666",
        "Qwen2.5 7B": "#56B4E9",
        "Qwen3 30B": "#0072B2",
        "Gemma 12B": "#009E73",
        "Llama 8B": "#E69F00",
        "Mistral 12B": "#D55E00",
    }
    workflow_styles = {
        "prior": ("--", "o", 1.4, 0.75),
        "direct": (":", "o", 1.2, 0.70),
        "schema": ("-", "s", 1.3, 0.72),
        "grouped": ("-", "^", 1.3, 0.72),
        "hybrid": ("-", "D", 1.6, 0.78),
        "cot": ("-.", "P", 1.5, 0.78),
        "guided": ("-", "o", 2.0, 0.92),
    }
    by_workflow_group = {
        (row["workflow"], row["field_group"]): float(row["accuracy"])
        for row in group_rows
    }
    x = list(range(len(group_order)))
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    for workflow, label, model, style_key in configs:
        values = [by_workflow_group[(workflow, group)] for group in group_order]
        linestyle, marker, linewidth, alpha = workflow_styles[style_key]
        ax.plot(
            x,
            values,
            label=label,
            color=model_colors[model],
            linestyle=linestyle,
            marker=marker,
            markersize=4.2,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_ylim(0.15, 0.90)
    ax.set_ylabel("Field-group accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        fontsize=7,
        columnspacing=1.2,
        handlelength=2.2,
    )
    save_figure(fig, "figure_workflow_model_field_profiles")


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Calibri"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_workflow_accuracy()
    plot_field_group_accuracy()
    plot_posthoc_triage()
    plot_error_taxonomy()
    plot_model_sensitivity()
    plot_workflow_model_field_profiles()
    print(f"Saved figures under {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

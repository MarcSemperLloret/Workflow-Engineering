"""Create the central multi-panel results figure for the manuscript."""

from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent
PILOT = ROOT / "data" / "analysis" / "pilot_workflow_ablation"
BALANCED = ROOT / "data" / "analysis" / "balanced_workflow_model"
FIGDIR = PILOT / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GRAY = "#666666"
DARK = "#222222"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def short_model_label(model: str) -> str:
    labels = {
        "qwen2.5-coder:7b": "Qwen2.5 7B",
        "qwen3-coder:30b": "Qwen3 30B",
        "command-r:35b": "Command R 35B",
        "gemma3:12b": "Gemma 12B",
        "mistral-nemo:12b": "Mistral 12B",
        "llama3.1:8b": "Llama 8B",
    }
    return labels.get(model, model)


def workflow_label(workflow: str) -> str:
    labels = {
        "direct": "Direct",
        "schema": "Schema",
        "field_groups": "Groups",
        "hybrid": "Hybrid",
        "guided": "Guided",
        "guided_access": "Guided+\naccess",
        "cot": "CoT",
        "audit": "Audit",
    }
    return labels[workflow]


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Calibri"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.6, 8.2))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.95, 1.20, 0.95],
        hspace=0.92,
    )

    # Panel A: main Qwen-family ladder.
    ax_a = fig.add_subplot(gs[0, 0])
    ladder_rows = read_csv(PILOT / "figure_workflow_accuracy.csv")
    ladder_labels = [
        "Prior",
        "Direct",
        "Schema",
        "Groups",
        "Hybrid",
        "Guided\n30B",
        "+ access",
        "Audit",
    ]
    acc18 = [float(row["accuracy_18_objective"]) for row in ladder_rows]
    x = list(range(len(acc18)))
    ax_a.plot(x, acc18, color=BLUE, marker="o", linewidth=1.9, markersize=4.0)
    ax_a.scatter([6], [acc18[6]], s=48, color=GREEN, zorder=4)
    ax_a.axvline(4.5, color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax_a.text(4.55, 0.43, "model + prompt", fontsize=7, color=GRAY, va="bottom")
    for idx in [0, 1, 4, 6, 7]:
        ax_a.text(idx, acc18[idx] + 0.018, f"{acc18[idx]:.3f}", ha="center", fontsize=7)
    ax_a.set_ylim(0.36, 0.86)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ladder_labels, rotation=35, ha="right")
    ax_a.set_ylabel("18-field accuracy")
    ax_a.set_title("Main workflow ladder", loc="left", fontsize=9)
    despine(ax_a)
    panel_label(ax_a, "A")

    # Panel B: balanced model x workflow experiment.
    ax_b = fig.add_subplot(gs[1, 0])
    workflows = ["direct", "schema", "field_groups", "hybrid", "guided", "guided_access", "cot", "audit"]
    balanced_rows = read_csv(BALANCED / "balanced_workflow_model_primary_wide.csv")
    colors = {
        "qwen2.5-coder:7b": SKY,
        "qwen3-coder:30b": BLUE,
        "command-r:35b": PURPLE,
        "gemma3:12b": GREEN,
        "mistral-nemo:12b": RED,
        "llama3.1:8b": ORANGE,
    }
    markers = {
        "qwen2.5-coder:7b": "o",
        "qwen3-coder:30b": "s",
        "command-r:35b": "D",
        "gemma3:12b": "^",
        "mistral-nemo:12b": "v",
        "llama3.1:8b": "P",
    }
    bx = list(range(len(workflows)))
    for row in balanced_rows:
        model = row["model"]
        values = [float(row[w]) for w in workflows]
        best_idx = max(range(len(values)), key=lambda i: values[i])
        ax_b.plot(
            bx,
            values,
            color=colors[model],
            marker=markers[model],
            linewidth=1.35,
            markersize=3.6,
            alpha=0.92,
            label=short_model_label(model),
        )
        ax_b.scatter(
            [best_idx],
            [values[best_idx]],
            s=44,
            facecolors="none",
            edgecolors=colors[model],
            linewidths=1.2,
            zorder=5,
        )
    ax_b.set_ylim(0.43, 0.85)
    ax_b.set_xticks(bx)
    ax_b.set_xticklabels([workflow_label(w) for w in workflows], rotation=28, ha="right")
    ax_b.set_ylabel("18-field accuracy")
    ax_b.set_title("Balanced model--workflow experiment", loc="left", fontsize=9, pad=24)
    ax_b.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.3,
        fontsize=6.5,
    )
    despine(ax_b)
    panel_label(ax_b, "B")

    # Panel C: residual error taxonomy.
    ax_d = fig.add_subplot(gs[2, 0])
    error_rows = read_csv(PILOT / "pilot_residual_error_taxonomy_summary_final.csv")
    error_labels = {
        "temporal_setup_extraction": "Temporal setup",
        "graph_node_semantics": "Graph-node semantics",
        "task_target_schema_boundary": "Task/target boundary",
        "bibliographic_metadata": "Bibliographic metadata",
        "access_evidence": "Access evidence",
        "dataset_family_recall_normalization": "Dataset normalization",
    }
    errors = [int(row["errors"]) for row in error_rows]
    labels = [error_labels[row["error_category"]] for row in error_rows]
    ex = list(range(len(errors)))
    ax_d.barh(ex, errors, color=[RED, ORANGE, ORANGE, GRAY, BLUE, BLUE], height=0.72)
    for idx, value in enumerate(errors):
        ax_d.text(value + 1.1, idx, str(value), va="center", fontsize=7)
    ax_d.set_xlim(0, max(errors) + 9)
    ax_d.set_yticks(ex)
    ax_d.set_yticklabels(labels)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Residual field errors")
    ax_d.set_title("What still fails", loc="left", fontsize=9)
    despine(ax_d)
    ax_d.grid(axis="x", color="#dddddd", linewidth=0.55)
    ax_d.grid(axis="y", visible=False)
    panel_label(ax_d, "C")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGDIR / f"figure_central_results.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGDIR / 'figure_central_results.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

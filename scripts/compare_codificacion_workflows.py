"""
Summarize coder-1 workflow evaluations into one comparison CSV.

Example:
  python scripts/compare_codificacion_workflows.py
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_BASE = ROOT / "data" / "model_outputs" / "codificacion_1_ollama"
DEFAULT_WORKFLOWS = [
    "w0_direct",
    "w1_metadata_sections",
    "w2_schema_normalized",
    "w3_field_groups",
    "w4_hybrid_w2_w3",
    "w5_evidence_required",
    "w6_second_pass_audit",
    "w7_retrieval_tables",
    "w8_self_consistency",
    "w7_retrieval_tables_qwen3_coder",
    "w9_multi_model_panel",
    "w10_guideline_fewshot",
    "w11_cot_grouped",
    "w0_direct_qwen3_coder",
    "w10_guideline_fewshot_qwen3_coder",
    "w12_focused_audit_qwen3_coder",
]


def read_report(path: Path) -> dict[str, float]:
    by_field: dict[str, list[int]] = {}
    total_ok = 0
    total = 0
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            field = row["field"]
            if field == "__paper__":
                continue
            ok = row["match"] == "yes"
            stats = by_field.setdefault(field, [0, 0])
            stats[1] += 1
            total += 1
            if ok:
                stats[0] += 1
                total_ok += 1

    summary = {field: ok / count for field, (ok, count) in by_field.items() if count}
    summary["__overall__"] = total_ok / total if total else 0.0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--workflow", action="append", dest="workflows")
    parser.add_argument("--report-name", default="evaluation_report.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    workflows = args.workflows or DEFAULT_WORKFLOWS
    output_path = Path(args.out) if args.out else base_dir / "workflow_comparison.csv"

    summaries = {}
    for workflow in workflows:
        report_path = base_dir / workflow / args.report_name
        if not report_path.exists():
            raise FileNotFoundError(f"Missing evaluation report: {report_path}")
        summaries[workflow] = read_report(report_path)

    fields = ["__overall__"] + sorted(
        set().union(*(set(summary) for summary in summaries.values())) - {"__overall__"}
    )
    fieldnames = ["field"] + workflows
    for previous, current in zip(workflows, workflows[1:]):
        fieldnames.append(f"delta_{current}_vs_{previous}")
    if len(workflows) > 2:
        fieldnames.append(f"delta_{workflows[-1]}_vs_{workflows[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for field in fields:
            row = {"field": field}
            for workflow in workflows:
                row[workflow] = f"{summaries[workflow].get(field, 0):.3f}"
            for previous, current in zip(workflows, workflows[1:]):
                delta = summaries[current].get(field, 0) - summaries[previous].get(field, 0)
                row[f"delta_{current}_vs_{previous}"] = f"{delta:.3f}"
            if len(workflows) > 2:
                delta = summaries[workflows[-1]].get(field, 0) - summaries[workflows[0]].get(field, 0)
                row[f"delta_{workflows[-1]}_vs_{workflows[0]}"] = f"{delta:.3f}"
            writer.writerow(row)

    print(f"Saved comparison: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

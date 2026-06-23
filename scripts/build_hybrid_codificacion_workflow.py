"""
Build a hybrid coder-1 output by combining two completed workflow CSVs.

The default hybrid uses w3 for fields where field-group coding improved over w2,
and w2 for fields where grouped coding degraded.

Example:
  python scripts/build_hybrid_codificacion_workflow.py
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASE = ROOT / "data" / "model_outputs" / "codificacion_1_ollama"

FIELDNAMES = [
    "paper_id", "year", "venue_type", "domain_class", "task_class",
    "target_class", "dataset_family", "has_public_code", "has_public_dataset",
    "horizon_type", "horizon_min_value", "horizon_max_value", "horizon_unit",
    "frequency_value", "frequency_unit", "graph_node_count_min",
    "graph_node_count_max", "node_unit", "graph_node_count_status",
    "coding_status", "confidence",
]

W3_FIELDS = {
    "year",
    "venue_type",
    "domain_class",
    "task_class",
    "target_class",
    "horizon_type",
    "horizon_min_value",
    "frequency_value",
    "frequency_unit",
}


def read_by_id(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["paper_id"]: row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-workflow", default="w2_schema_normalized")
    parser.add_argument("--field-workflow", default="w3_field_groups")
    parser.add_argument("--out-workflow", default="w4_hybrid_w2_w3")
    parser.add_argument("--base-dir", default=str(BASE))
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_rows = read_by_id(base_dir / args.base_workflow / "codificacion_1_ollama.csv")
    field_rows = read_by_id(base_dir / args.field_workflow / "codificacion_1_ollama.csv")

    out_dir = base_dir / args.out_workflow
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "codificacion_1_ollama.csv"

    merged_rows = []
    for paper_id, base_row in base_rows.items():
        field_row = field_rows.get(paper_id, {})
        merged = {}
        for field in FIELDNAMES:
            if field == "paper_id":
                merged[field] = paper_id
            elif field in W3_FIELDS and field in field_row:
                merged[field] = field_row[field]
            else:
                merged[field] = base_row.get(field, "not_reported")
        merged_rows.append(merged)

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged_rows)

    manifest = out_dir / "hybrid_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "source_workflow"])
        writer.writeheader()
        for field in FIELDNAMES:
            source = args.field_workflow if field in W3_FIELDS else args.base_workflow
            writer.writerow({"field": field, "source_workflow": source})

    print(f"Saved hybrid CSV: {out_path}")
    print(f"Saved hybrid manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

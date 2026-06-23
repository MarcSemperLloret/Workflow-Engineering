"""
Assemble codificacion_1_ollama.csv from per-paper JSON records in a workflow output directory.

Use this after resumed/partial runs where the runner's final CSV may contain only
the last subset of papers.
"""

import argparse
import csv
import json
from pathlib import Path

from run_codificacion_1_ollama import CORPUS_CSV, DEFAULT_OUT_DIR, FIELDNAMES, normalize_coded_row


def pilot_order() -> list[str]:
    with open(CORPUS_CSV, newline="", encoding="utf-8-sig") as handle:
        return [
            row["id_short"]
            for row in csv.DictReader(handle)
            if row.get("include_in_pilot", "").lower() == "yes"
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--out-workflow", default=None)
    parser.add_argument("--base-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.base_dir) / args.workflow
    write_dir = Path(args.base_dir) / (args.out_workflow or args.workflow)
    write_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for paper_id in pilot_order():
        path = out_dir / f"{paper_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing JSON for {paper_id}: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        coded = normalize_coded_row(record["coded"])
        rows.append({field: coded.get(field, "not_reported") for field in FIELDNAMES})

    csv_path = write_dir / "codificacion_1_ollama.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

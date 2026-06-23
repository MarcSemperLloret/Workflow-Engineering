"""
Compare closed-label coder-1 model output against the pilot gold CSV.

Example:
  python scripts/evaluate_codificacion_1.py
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_GOLD = ROOT / "data" / "annotations" / "codificacion_1_manual_pilot.csv"
DEFAULT_PRED = ROOT / "data" / "model_outputs" / "codificacion_1_ollama" / "codificacion_1_ollama.csv"
DEFAULT_OUT = ROOT / "data" / "model_outputs" / "codificacion_1_ollama" / "evaluation_report.csv"

DATASET_ALIASES = {
    "pems03": "pemsd3",
    "pems04": "pemsd4",
    "pems07": "pemsd7",
    "pems08": "pemsd8",
    "pems3": "pemsd3",
    "pems4": "pemsd4",
    "pems7": "pemsd7",
    "pems8": "pemsd8",
    "pems_d3": "pemsd3",
    "pems_d4": "pemsd4",
    "pems_d7": "pemsd7",
    "pems_d8": "pemsd8",
    "pems-bay": "pems_bay",
    "pemsbay": "pems_bay",
    "metr-la": "metr_la",
    "metrla": "metr_la",
    "nycbike": "nyc_bike",
    "nyc_bike1": "nyc_bike",
    "nyc_bike2": "nyc_bike",
    "nyctaxi": "nyc_taxi",
    "bjtaxi": "bj_taxi",
    "chicago_bike": "chibike",
    "chi_bike": "chibike",
    "hzme_inflow": "hzme",
    "hzme_outflow": "hzme",
}


def read_by_id(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["paper_id"]: row for row in csv.DictReader(handle)}


def normalize_dataset_value(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    return DATASET_ALIASES.get(normalized, normalized)


def dataset_set(value: str, use_aliases: bool) -> set[str]:
    values = {item.strip() for item in value.split("|") if item.strip()}
    if not use_aliases:
        return values
    return {normalize_dataset_value(item) for item in values}


def compare_values(field: str, gold: str, pred: str, use_dataset_aliases: bool = False) -> tuple[bool, str]:
    if field == "dataset_family":
        exact_gold_set = dataset_set(gold, use_aliases=False)
        exact_pred_set = dataset_set(pred, use_aliases=False)
        if exact_gold_set == exact_pred_set:
            return True, "exact_set"
        if use_dataset_aliases and dataset_set(gold, use_aliases=True) == dataset_set(pred, use_aliases=True):
            return True, "alias_set"
        return False, "set_mismatch"
    if gold == pred:
        return True, "exact"
    return False, "mismatch"


def values_match(field: str, gold: str, pred: str) -> bool:
    return compare_values(field, gold, pred)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--pred", default=str(DEFAULT_PRED))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--dataset-aliases",
        action="store_true",
        help="Normalize known dataset aliases before scoring dataset_family.",
    )
    args = parser.parse_args()

    gold = read_by_id(Path(args.gold))
    pred = read_by_id(Path(args.pred))

    fields = [field for field in next(iter(gold.values())).keys() if field != "paper_id"]
    rows = []
    for paper_id, gold_row in gold.items():
        if paper_id not in pred:
            rows.append({
                "paper_id": paper_id,
                "field": "__paper__",
                "gold": "present",
                "pred": "missing",
                "match": "no",
            })
            continue
        pred_row = pred[paper_id]
        for field in fields:
            g = gold_row.get(field, "")
            p = pred_row.get(field, "")
            match, reason = compare_values(field, g, p, use_dataset_aliases=args.dataset_aliases)
            rows.append({
                "paper_id": paper_id,
                "field": field,
                "gold": g,
                "pred": p,
                "match": "yes" if match else "no",
                "match_reason": reason,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "field", "gold", "pred", "match", "match_reason"])
        writer.writeheader()
        writer.writerows(rows)

    compared = [row for row in rows if row["field"] != "__paper__"]
    matches = sum(1 for row in compared if row["match"] == "yes")
    total = len(compared)
    print(f"Compared fields: {total}")
    print(f"Exact matches: {matches}")
    print(f"Accuracy: {matches / total:.3f}" if total else "Accuracy: n/a")

    by_field = {}
    for row in compared:
        stats = by_field.setdefault(row["field"], [0, 0])
        stats[1] += 1
        if row["match"] == "yes":
            stats[0] += 1
    for field, (ok, count) in sorted(by_field.items()):
        print(f"{field}: {ok}/{count} = {ok / count:.3f}")

    print(f"Saved report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

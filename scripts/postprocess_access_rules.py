"""
Apply conservative deterministic access-availability rules to a coded CSV.

This creates an exploratory post-processed run.  It only changes
`has_public_dataset`; `has_public_code` and all other fields are copied from the
input prediction file.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from run_codificacion_1_ollama import FIELDNAMES, PARSED_DIR, normalize_text

ROOT = Path(__file__).parent.parent
DEFAULT_IN = (
    ROOT
    / "data"
    / "model_outputs"
    / "codificacion_1_ollama"
    / "w10_guideline_fewshot_qwen3_coder"
    / "codificacion_1_ollama.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "data"
    / "model_outputs"
    / "codificacion_1_ollama"
    / "w10_access_rules"
)

PUBLIC_DATASET_FAMILIES = {
    "metr_la",
    "pems_bay",
    "pemsd3",
    "pemsd4",
    "pemsd7",
    "pemsd8",
    "pems03",
    "pems04",
    "pems07",
    "pems08",
    "los_loop",
    "sz_taxi",
    "nyc_bike",
    "nyc_taxi",
    "bj_taxi",
    "chibike",
    "t_drive",
    "seattle_loop",
    "ne_bj",
    "nyt_covid",
    "google_mobility",
    "vevo_music",
    "wiki_traffic",
    "electricity",
    "exchange_rate",
    "solar_energy",
    "largest",
    "california_highway_pems",
}

NON_PUBLIC_OR_OWN_DATASET_FAMILIES = {
    "amap_beijing",
    "yangtze_air_quality",
    "swiss_pv_real",
    "swiss_pv_synthetic",
    "zhuzhou",
    "baoding",
    "beijing",
    "chengdu",
    "chengdu_slag_truck",
}

DATASET_URL_RE = re.compile(
    r"(?:dataset|data set|datasets?|benchmark).{0,160}"
    r"(?:https?://|github\.com|kaggle\.com|aistudio\.baidu\.com)",
    re.IGNORECASE,
)

DATASET_PUBLIC_RE = re.compile(
    r"(?:"
    r"public(?:ly)? (?:available )?(?:dataset|datasets|benchmark)|"
    r"dataset(?:s)? (?:is|are) (?:publicly )?available|"
    r"data(?:set)? availability|"
    r"dataset, code and pretrained models?.{0,80}available"
    r")",
    re.IGNORECASE,
)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parsed_text(paper_id: str) -> str:
    path = PARSED_DIR / f"{paper_id}.json"
    if not path.exists():
        return ""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    sections = parsed.get("sections", {})
    values = []
    for name in ("preamble", "abstract", "introduction", "methodology", "experiments", "discussion", "conclusion", "appendix"):
        if sections.get(name):
            values.append(sections[name])
    captions = [str(item.get("caption", "")) for item in parsed.get("table_captions", []) + parsed.get("figure_captions", [])]
    return normalize_text("\n".join(values + captions))


def dataset_values(value: str) -> set[str]:
    return {part.strip().lower() for part in value.split("|") if part.strip()}


def decide_code(_text: str, current: str) -> tuple[str, str]:
    # Code availability is often determined by external repository checks rather
    # than explicit PDF text, so this exploratory pass leaves model labels intact.
    return current, "unchanged_code_availability_requires_external_audit"


def decide_dataset(text: str, dataset_family: str, current: str) -> tuple[str, str]:
    families = dataset_values(dataset_family)
    if families & PUBLIC_DATASET_FAMILIES:
        return "yes", "known_public_benchmark_family"
    if DATASET_URL_RE.search(text) or DATASET_PUBLIC_RE.search(text):
        return "yes", "dataset_public_phrase_or_url"
    if families and families <= NON_PUBLIC_OR_OWN_DATASET_FAMILIES:
        return "no", "known_private_or_newly_collected_family_without_release"
    if current == "yes":
        return "no", "yes_without_public_dataset_evidence"
    return current, "unchanged_no_dataset_evidence"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=str(DEFAULT_IN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rows = read_rows(Path(args.pred))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    changes = []
    out_rows = []
    for row in rows:
        row = dict(row)
        text = parsed_text(row["paper_id"])
        old_code = row.get("has_public_code", "not_reported")
        old_dataset = row.get("has_public_dataset", "not_reported")
        new_code, code_reason = decide_code(text, old_code)
        new_dataset, dataset_reason = decide_dataset(text, row.get("dataset_family", ""), old_dataset)

        if new_code != old_code:
            changes.append({
                "paper_id": row["paper_id"],
                "field": "has_public_code",
                "old_value": old_code,
                "new_value": new_code,
                "reason": code_reason,
            })
            row["has_public_code"] = new_code
        if new_dataset != old_dataset:
            changes.append({
                "paper_id": row["paper_id"],
                "field": "has_public_dataset",
                "old_value": old_dataset,
                "new_value": new_dataset,
                "reason": dataset_reason,
            })
            row["has_public_dataset"] = new_dataset

        out_rows.append(row)

    with open(out_dir / "codificacion_1_ollama.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    with open(out_dir / "access_rule_changes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "field", "old_value", "new_value", "reason"])
        writer.writeheader()
        writer.writerows(changes)

    print(f"Saved: {out_dir / 'codificacion_1_ollama.csv'}")
    print(f"Saved: {out_dir / 'access_rule_changes.csv'}")
    print(f"Changes: {len(changes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

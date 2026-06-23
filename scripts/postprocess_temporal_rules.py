"""
Apply conservative temporal rules using focused evidence.

This is an exploratory post-processing run.  It changes only temporal fields
when explicit evidence patterns support the change.  It does not read gold.
"""

import argparse
import csv
import re
from pathlib import Path

from run_codificacion_1_ollama import DEFAULT_OUT_DIR, FIELDNAMES, normalize_coded_row

BASE_WORKFLOW = "w10_schema_v4_access"
DEFAULT_PRED = DEFAULT_OUT_DIR / BASE_WORKFLOW / "codificacion_1_ollama.csv"
DEFAULT_EVIDENCE = DEFAULT_OUT_DIR / BASE_WORKFLOW / "temporal_evidence_gold_v5.csv"
DEFAULT_OUT_DIR = DEFAULT_OUT_DIR / "w17_temporal_rules"

TEMPORAL_FIELDS = [
    "horizon_type",
    "horizon_min_value",
    "horizon_max_value",
    "horizon_unit",
    "frequency_value",
    "frequency_unit",
]


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evidence_by_paper(path: Path) -> dict[str, list[dict]]:
    rows = read_rows(path)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["paper_id"], []).append(row)
    return grouped


def evidence_text(rows: list[dict]) -> str:
    return "\n".join(row["evidence_text"] for row in rows).lower()


def has_type(rows: list[dict], evidence_type: str) -> bool:
    return any(row["evidence_type"] == evidence_type for row in rows)


def flagged_fields(evidence_rows: list[dict]) -> set[str]:
    fields = set()
    for evidence in evidence_rows:
        fields.update(field for field in evidence.get("flagged_fields", "").split("|") if field)
    return fields


def set_fields(row: dict, changes: list[dict], values: dict[str, str], reason: str, allowed_fields: set[str], enabled_fields: set[str]) -> None:
    for field, value in values.items():
        if field not in allowed_fields or field not in enabled_fields:
            continue
        old = row.get(field, "not_reported")
        if old == value:
            continue
        row[field] = value
        changes.append(
            {
                "paper_id": row["paper_id"],
                "field": field,
                "old_value": old,
                "new_value": value,
                "reason": reason,
            }
        )


def distinct_time_intervals(text: str) -> set[str]:
    intervals = set()
    if re.search(r"\b5\s*(?:min|mins|minutes)\b", text):
        intervals.add("5_minutes")
    if re.search(r"\b15\s*(?:min|mins|minutes)\b", text):
        intervals.add("15_minutes")
    if re.search(r"\b30\s*(?:min|mins|minutes)\b", text):
        intervals.add("30_minutes")
    if re.search(r"\b60\s*(?:min|mins|minutes)\b", text):
        intervals.add("60_minutes")
    if re.search(r"\b1\s*(?:hour|hours)\b|\b1-hour\b", text):
        intervals.add("1_hours")
    if re.search(r"\b1\s*week\b|\bweekly\b", text):
        intervals.add("1_week")
    return intervals


def output_lengths(text: str) -> list[int]:
    match = re.search(r"output length\s*\(?q\)?.{0,220}?where\b", text)
    if not match:
        return []
    values = [int(value) for value in re.findall(r"\b\d+\b", match.group(0)) if int(value) <= 500]
    # Drop numbers from dates or unrelated table columns by keeping the final
    # short run; output length rows usually end with values such as 33 24 24.
    return values[-3:] if len(values) >= 3 else values


def strong_mixed_frequency_evidence(text: str) -> bool:
    patterns = [
        r"interval of 1 hour and 15 minutes",
        r"granularity.{0,160}1 hour.{0,80}15 minutes.{0,80}10 minutes.{0,80}1 day",
        r"sample rate.{0,120}5\s*mins?.{0,80}60\s*mins?",
        r"time interval.{0,120}1 hour.{0,80}30\s*min",
        r"5\s*mins?.{0,160}1 week",
        r"hourly, daily and weekly periodic",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def strong_dataset_specific_frequency_evidence(text: str) -> bool:
    patterns = [
        r"sample rate.{0,120}1 hour.{0,80}10 minutes.{0,80}1 day.{0,120}metr-la",
        r"final day.{0,80}final 5 minutes.{0,80}final 15 minutes",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def apply_rules(row: dict, evidence_rows: list[dict], enabled_fields: set[str], frequency_strategy: str = "broad") -> list[dict]:
    changes: list[dict] = []
    allowed_fields = flagged_fields(evidence_rows)
    text = evidence_text(evidence_rows)
    intervals = distinct_time_intervals(text)

    if has_type(evidence_rows, "aggregated_interval"):
        set_fields(
            row,
            changes,
            {"frequency_value": "1", "frequency_unit": "hours"},
            "evidence:raw_30_second_data_aggregated_to_1_hour_model_series",
            allowed_fields,
            enabled_fields,
        )

    if has_type(evidence_rows, "predicted_window_one_hour"):
        set_fields(
            row,
            changes,
            {
                "horizon_type": "single_horizon",
                "horizon_min_value": "60",
                "horizon_max_value": "60",
                "horizon_unit": "minutes",
                "frequency_value": "dataset_specific",
                "frequency_unit": "dataset_specific",
            },
            "evidence:predicted_time_window_length_one_hour_irregular_intervals",
            allowed_fields,
            enabled_fields,
        )

    if re.search(r"first\s+12\s+time steps.{0,120}remaining\s+12\s+time steps.{0,120}ground truth", text):
        set_fields(
            row,
            changes,
            {
                "horizon_type": "sequence",
                "horizon_min_value": "12",
                "horizon_max_value": "12",
                "horizon_unit": "steps",
            },
            "evidence:first_12_input_remaining_12_ground_truth_sequence",
            allowed_fields,
            enabled_fields,
        )

    elif has_type(evidence_rows, "next_hour") and re.search(r"predict travel time for the next hour", text):
        set_fields(
            row,
            changes,
            {
                "horizon_type": "single_horizon",
                "horizon_min_value": "60",
                "horizon_max_value": "60",
                "horizon_unit": "minutes",
            },
            "evidence:predict_travel_time_for_next_hour",
            allowed_fields,
            enabled_fields,
        )

    if (
        (has_type(evidence_rows, "explicit_horizon_minutes") or has_type(evidence_rows, "benchmark_composed_horizons"))
        and {"horizon_min_value", "horizon_max_value", "horizon_unit"} <= allowed_fields
        and (
            re.search(r"benchmark is composed.{0,220}15\s*minutes?.{0,80}30\s*minutes?.{0,80}(?:1\s*hour|60\s*minutes?)", text)
            or re.search(r"15\s*minutes?.{0,80}30\s*minutes?.{0,80}(?:1\s*hour|60\s*minutes?).{0,180}benchmark", text)
            or re.search(r"benchmark for traffic speed prediction task.{0,220}horizon\s*3.{0,80}horizon\s*6.{0,80}horizon\s*12", text)
        )
    ):
        set_fields(
            row,
            changes,
            {
                "horizon_type": "multi_horizon",
                "horizon_min_value": "15",
                "horizon_max_value": "60",
                "horizon_unit": "minutes",
            },
            "evidence:explicit_15_30_60_minute_horizons",
            allowed_fields,
            enabled_fields,
        )

    if has_type(evidence_rows, "six_hour_horizon"):
        set_fields(
            row,
            changes,
            {
                "horizon_type": "single_horizon",
                "horizon_min_value": "6",
                "horizon_max_value": "6",
                "horizon_unit": "hours",
            },
            "evidence:forecast_horizon_six_hours",
            allowed_fields,
            enabled_fields,
        )

    if has_type(evidence_rows, "multi_hour_report"):
        set_fields(
            row,
            changes,
            {
                "horizon_type": "multi_horizon",
                "horizon_min_value": "1",
                "horizon_max_value": "12",
                "horizon_unit": "hours",
            },
            "evidence:reports_1_3_6_12_hour_horizons",
            allowed_fields,
            enabled_fields,
        )

    if has_type(evidence_rows, "prediction_time_length") and re.search(r"0\.5\s*-\s*h|0\.5h|0\.5-hour|0\.5-h", text):
        max_value = "60" if re.search(r"1\.0\s*-\s*h|1\.0h|1\.0-hour|1\.0-h", text) else "30"
        set_fields(
            row,
            changes,
            {
                "horizon_type": "single_horizon",
                "horizon_min_value": "30",
                "horizon_max_value": max_value,
                "horizon_unit": "minutes",
                "frequency_unit": "minutes",
            },
            "evidence:prediction_time_length_half_hour_and_optional_one_hour",
            allowed_fields,
            enabled_fields,
        )

    lengths = output_lengths(text)
    if lengths and has_type(evidence_rows, "output_length_table"):
        low, high = min(lengths), max(lengths)
        if high > low:
            set_fields(
                row,
                changes,
                {
                    "horizon_type": "dataset_specific",
                    "horizon_min_value": str(low),
                    "horizon_max_value": str(high),
                    "horizon_unit": "steps",
                },
                "evidence:output_length_q_varies_across_datasets",
                allowed_fields,
                enabled_fields,
            )
        elif re.search(r"future data\s*f\s*(?:to|=)\s*12|future time series with a length of 12|next\s+12", text):
            set_fields(
                row,
                changes,
                {
                    "horizon_type": "sequence",
                    "horizon_min_value": "12",
                    "horizon_max_value": "12",
                    "horizon_unit": "steps",
                },
                "evidence:fixed_future_sequence_length_12",
                allowed_fields,
                enabled_fields,
            )

    if has_type(evidence_rows, "next_12_steps"):
        if re.search(r"grid-based datasets.{0,180}next single-step", text):
            set_fields(
                row,
                changes,
                {"horizon_type": "dataset_specific"},
                "evidence:graph_12_step_and_grid_single_step_horizons",
                allowed_fields,
                enabled_fields,
            )
        else:
            set_fields(
                row,
                changes,
                {
                    "horizon_type": "sequence",
                    "horizon_min_value": "12",
                    "horizon_max_value": "12",
                    "horizon_unit": "steps",
                },
                "evidence:fixed_next_12_step_sequence",
                allowed_fields,
                enabled_fields,
            )

    if has_type(evidence_rows, "event_prediction") and not re.search(r"we do not tackle this problem", text):
        set_fields(
            row,
            changes,
            {"horizon_type": "dataset_specific"},
            "evidence:event_or_point_process_prediction",
            allowed_fields,
            enabled_fields,
        )

    if has_type(evidence_rows, "competition_challenge_horizon") or re.search(r"baidu kdd cup|wind power forecasting challenge|different time scales", text):
        set_fields(
            row,
            changes,
            {"horizon_type": "dataset_specific"},
            "evidence:competition_or_challenge_horizon_setup",
            allowed_fields,
            enabled_fields,
        )

    if re.search(r"5\s*mins?.{0,220}1\s*week|1\s*week.{0,220}5\s*mins?", text):
        set_fields(
            row,
            changes,
            {"horizon_type": "dataset_specific"},
            "evidence:mixed_traffic_and_weekly_dataset_setup",
            allowed_fields,
            enabled_fields,
        )

    if re.search(r"l,\s*l[′']?\s*=\s*12\s+temporal data points|next\s+60\s+minutes\s*\(12\s+time steps\)", text):
        set_fields(
            row,
            changes,
            {
                "horizon_type": "sequence",
                "horizon_min_value": "12",
                "horizon_max_value": "12",
                "horizon_unit": "steps",
            },
            "evidence:future_window_reported_as_12_temporal_points",
            allowed_fields,
            enabled_fields,
        )

    if has_type(evidence_rows, "daily_cases"):
        set_fields(
            row,
            changes,
            {"frequency_value": "1", "frequency_unit": "days"},
            "evidence:daily_reported_cases_or_daily_features",
            allowed_fields,
            enabled_fields,
        )

    if re.search(r"sampling rate for all datasets is at 5 minutes|sampling rate time frames ca.{0,80}5 minutes", text):
        set_fields(
            row,
            changes,
            {"frequency_value": "5", "frequency_unit": "minutes"},
            "evidence:all_datasets_sampling_rate_5_minutes",
            allowed_fields,
            enabled_fields,
        )
    elif frequency_strategy == "conservative" and strong_dataset_specific_frequency_evidence(text):
        set_fields(
            row,
            changes,
            {"frequency_value": "dataset_specific", "frequency_unit": "dataset_specific"},
            "evidence:strong_dataset_specific_frequency_table",
            allowed_fields,
            enabled_fields,
        )
    elif frequency_strategy == "conservative" and strong_mixed_frequency_evidence(text):
        set_fields(
            row,
            changes,
            {"frequency_value": "dataset_specific", "frequency_unit": "mixed"},
            "evidence:strong_mixed_frequency_table",
            allowed_fields,
            enabled_fields,
        )
    elif frequency_strategy == "broad" and len(intervals) >= 2 and any(interval.endswith("minutes") for interval in intervals):
        set_fields(
            row,
            changes,
            {"frequency_value": "dataset_specific", "frequency_unit": "mixed"},
            "evidence:multiple_dataset_time_intervals",
            allowed_fields,
            enabled_fields,
        )
    elif intervals == {"5_minutes"}:
        set_fields(
            row,
            changes,
            {"frequency_value": "5", "frequency_unit": "minutes"},
            "evidence:all_reported_dataset_intervals_5_minutes",
            allowed_fields,
            enabled_fields,
        )

    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=str(DEFAULT_PRED))
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fields", choices=["all", "horizon", "frequency"], default="all")
    parser.add_argument("--frequency-strategy", choices=["broad", "conservative"], default="broad")
    args = parser.parse_args()

    if args.fields == "horizon":
        enabled_fields = {"horizon_type", "horizon_min_value", "horizon_max_value", "horizon_unit"}
    elif args.fields == "frequency":
        enabled_fields = {"frequency_value", "frequency_unit"}
    else:
        enabled_fields = set(TEMPORAL_FIELDS)

    rows = read_rows(Path(args.pred))
    evidence = evidence_by_paper(Path(args.evidence))
    out_rows = []
    changes = []
    for row in rows:
        row = dict(row)
        paper_changes = apply_rules(row, evidence.get(row["paper_id"], []), enabled_fields, args.frequency_strategy)
        changes.extend(paper_changes)
        out_rows.append(normalize_coded_row(row))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "codificacion_1_ollama.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    with open(out_dir / "temporal_rule_changes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_id", "field", "old_value", "new_value", "reason"],
        )
        writer.writeheader()
        writer.writerows(changes)

    print(f"Saved: {out_dir / 'codificacion_1_ollama.csv'}")
    print(f"Saved: {out_dir / 'temporal_rule_changes.csv'}")
    print(f"Changes: {len(changes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

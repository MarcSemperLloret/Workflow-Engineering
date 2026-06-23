"""
Build pilot-only workflow ablation summaries for the manuscript.

The script evaluates selected 50-paper pilot workflows against gold_v5 with the
same alias-aware comparison used by evaluate_codificacion_1.py, then writes
compact tables and paired cluster-bootstrap intervals by paper.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from evaluate_codificacion_1 import compare_values, read_by_id

ROOT = Path(__file__).parent.parent
GOLD = ROOT / "data" / "annotations" / "codificacion_1_manual_gold_v5.csv"
CORPUS = ROOT / "corpus_inicial_50_papers_stgl.csv"
ARXIV_CACHE = ROOT / "data" / "annotations" / "arxiv_metadata_cache.csv"
BASE = ROOT / "data" / "model_outputs" / "codificacion_1_ollama"
OUTDIR = ROOT / "data" / "analysis" / "pilot_workflow_ablation"

WORKFLOWS = [
    ("metadata_prior_baseline", "metadata_prior_rules"),
    ("w0_direct_50", "direct_prompt"),
    ("w1_metadata_sections", "metadata_sections"),
    ("w2_schema_normalized_50", "schema_normalized"),
    ("w3_field_groups_50", "field_groups"),
    ("w4_hybrid_w2_w3_50", "field_family_hybrid"),
    ("w5_evidence_required", "evidence_required"),
    ("w6_second_pass_audit", "second_pass_audit"),
    ("w7_retrieval_tables", "table_caption_retrieval"),
    ("w8_self_consistency", "self_consistency"),
    ("w9_multi_model_panel", "multi_model_panel"),
    ("w10_guideline_fewshot_qwen3_coder", "qwen_guideline_fewshot"),
    ("w11_cot_grouped", "cot_grouped"),
    ("w10_schema_v4_access", "structured_baseline_access"),
    ("w16_temporal_audit_qwen3_coder", "temporal_llm_audit"),
    ("w18_horizon_rules_all_pilot", "horizon_rules"),
    ("w19_frequency_rules_conservative_pilot", "conservative_frequency_rules"),
    ("w10_guideline_fewshot_gemma3_12b", "model_sensitivity_raw_gemma3_12b"),
    ("w10_schema_v4_access_gemma3_12b", "model_sensitivity_access_gemma3_12b"),
    ("w10_guideline_fewshot_mistral_nemo_12b", "model_sensitivity_raw_mistral_nemo_12b"),
    ("w10_schema_v4_access_mistral_nemo_12b", "model_sensitivity_access_mistral_nemo_12b"),
    ("w10_guideline_fewshot_llama3_1_8b", "model_sensitivity_raw_llama3_1_8b"),
    ("w10_schema_v4_access_llama3_1_8b", "model_sensitivity_access_llama3_1_8b"),
]

COMPARISONS = [
    ("metadata_prior_baseline", "w0_direct_50"),
    ("w0_direct_50", "w1_metadata_sections"),
    ("w0_direct_50", "w2_schema_normalized_50"),
    ("w2_schema_normalized_50", "w3_field_groups_50"),
    ("w3_field_groups_50", "w4_hybrid_w2_w3_50"),
    ("w0_direct_50", "w5_evidence_required"),
    ("w0_direct_50", "w7_retrieval_tables"),
    ("w0_direct_50", "w8_self_consistency"),
    ("w4_hybrid_w2_w3_50", "w9_multi_model_panel"),
    ("w9_multi_model_panel", "w11_cot_grouped"),
    ("w4_hybrid_w2_w3_50", "w10_guideline_fewshot_qwen3_coder"),
    ("w0_direct_50", "w3_field_groups_50"),
    ("w3_field_groups_50", "w10_guideline_fewshot_qwen3_coder"),
    ("w10_guideline_fewshot_qwen3_coder", "w10_schema_v4_access"),
    ("w10_schema_v4_access", "w16_temporal_audit_qwen3_coder"),
]

FIELD_GROUPS = {
    "bibliographic": {"year", "venue_type"},
    "task_dataset": {"domain_class", "task_class", "target_class", "dataset_family"},
    "access": {"has_public_code", "has_public_dataset"},
    "temporal_horizon": {"horizon_type", "horizon_min_value", "horizon_max_value", "horizon_unit"},
    "temporal_frequency": {"frequency_value", "frequency_unit"},
    "graph_nodes": {"graph_node_count_min", "graph_node_count_max", "node_unit", "graph_node_count_status"},
    "meta": {"coding_status", "confidence"},
}

DIAGNOSTIC_FIELDS = {"coding_status", "confidence"}

MAIN_WORKFLOWS = {
    "metadata_prior_baseline",
    "w0_direct_50",
    "w2_schema_normalized_50",
    "w3_field_groups_50",
    "w4_hybrid_w2_w3_50",
    "w10_guideline_fewshot_qwen3_coder",
    "w10_schema_v4_access",
    "w16_temporal_audit_qwen3_coder",
}

# In-sample-only temporal rule workflows. Reported as a diagnostic of error
# systematicity, never as part of the deployable system, because the rules were
# hand-developed on these 50 papers and do not generalize.
DIAGNOSTIC_TEMPORAL_WORKFLOWS = [
    "w18_horizon_rules_all_pilot",
    "w19_frequency_rules_conservative_pilot",
]

MODEL_SENSITIVITY_PAIRS = [
    ("qwen3-coder:30b", "w10_guideline_fewshot_qwen3_coder", "w10_schema_v4_access"),
    ("gemma3:12b", "w10_guideline_fewshot_gemma3_12b", "w10_schema_v4_access_gemma3_12b"),
    ("mistral-nemo:12b", "w10_guideline_fewshot_mistral_nemo_12b", "w10_schema_v4_access_mistral_nemo_12b"),
    ("llama3.1:8b", "w10_guideline_fewshot_llama3_1_8b", "w10_schema_v4_access_llama3_1_8b"),
]

WORKFLOW_DISPLAY = {
    "metadata_prior_baseline": "Rules-only metadata/prior baseline",
    "w0_direct_50": "Direct local LLM prompt",
    "w1_metadata_sections": "Metadata/section-scoped prompt",
    "w2_schema_normalized_50": "Schema + output normalization",
    "w3_field_groups_50": "Grouped field extraction",
    "w4_hybrid_w2_w3_50": "Field-family hybrid",
    "w5_evidence_required": "Evidence-required prompt",
    "w6_second_pass_audit": "Generic second-pass audit",
    "w7_retrieval_tables": "Table/caption retrieval",
    "w8_self_consistency": "Self-consistency",
    "w9_multi_model_panel": "Multi-model panel",
    "w10_guideline_fewshot_qwen3_coder": "Guided stronger local model",
    "w11_cot_grouped": "Chain-of-thought grouped extraction",
    "w10_schema_v4_access": "Structured baseline + access rules",
    "w16_temporal_audit_qwen3_coder": "Second-pass temporal LLM audit",
    "w18_horizon_rules_all_pilot": "Deterministic horizon rules",
    "w19_frequency_rules_conservative_pilot": "Conservative frequency rules",
    "w10_guideline_fewshot_gemma3_12b": "Guidelines + few-shot",
    "w10_schema_v4_access_gemma3_12b": "Guidelines + few-shot + access rules",
    "w10_guideline_fewshot_mistral_nemo_12b": "Guidelines + few-shot",
    "w10_schema_v4_access_mistral_nemo_12b": "Guidelines + few-shot + access rules",
    "w10_guideline_fewshot_llama3_1_8b": "Guidelines + few-shot",
    "w10_schema_v4_access_llama3_1_8b": "Guidelines + few-shot + access rules",
}

WORKFLOW_MODEL = {
    "metadata_prior_baseline": "rules only",
    "w0_direct_50": "qwen2.5-coder:7b",
    "w1_metadata_sections": "qwen2.5-coder:7b",
    "w2_schema_normalized_50": "qwen2.5-coder:7b",
    "w3_field_groups_50": "qwen2.5-coder:7b",
    "w4_hybrid_w2_w3_50": "qwen2.5-coder:7b",
    "w5_evidence_required": "qwen2.5-coder:7b",
    "w6_second_pass_audit": "qwen2.5-coder:7b",
    "w7_retrieval_tables": "qwen2.5-coder:7b",
    "w8_self_consistency": "qwen2.5-coder:7b",
    "w9_multi_model_panel": "mixed local models",
    "w10_guideline_fewshot_qwen3_coder": "qwen3-coder:30b",
    "w11_cot_grouped": "qwen3-coder:30b",
    "w10_schema_v4_access": "qwen3-coder:30b",
    "w16_temporal_audit_qwen3_coder": "qwen3-coder:30b",
    "w18_horizon_rules_all_pilot": "qwen3-coder:30b + deterministic rules",
    "w19_frequency_rules_conservative_pilot": "qwen3-coder:30b + deterministic rules",
    "w10_guideline_fewshot_gemma3_12b": "gemma3:12b",
    "w10_schema_v4_access_gemma3_12b": "gemma3:12b + deterministic rules",
    "w10_guideline_fewshot_mistral_nemo_12b": "mistral-nemo:12b",
    "w10_schema_v4_access_mistral_nemo_12b": "mistral-nemo:12b + deterministic rules",
    "w10_guideline_fewshot_llama3_1_8b": "llama3.1:8b",
    "w10_schema_v4_access_llama3_1_8b": "llama3.1:8b + deterministic rules",
}

ERROR_CATEGORY_BY_FIELD = {
    "year": "bibliographic_metadata",
    "venue_type": "bibliographic_metadata",
    "domain_class": "task_target_schema_boundary",
    "task_class": "task_target_schema_boundary",
    "target_class": "task_target_schema_boundary",
    "dataset_family": "dataset_family_recall_normalization",
    "has_public_code": "access_evidence",
    "has_public_dataset": "access_evidence",
    "horizon_type": "temporal_setup_extraction",
    "horizon_min_value": "temporal_setup_extraction",
    "horizon_max_value": "temporal_setup_extraction",
    "horizon_unit": "temporal_setup_extraction",
    "frequency_value": "temporal_setup_extraction",
    "frequency_unit": "temporal_setup_extraction",
    "graph_node_count_min": "graph_node_semantics",
    "graph_node_count_max": "graph_node_semantics",
    "node_unit": "graph_node_semantics",
    "graph_node_count_status": "graph_node_semantics",
    "coding_status": "quality_or_annotation_difficulty",
    "confidence": "quality_or_annotation_difficulty",
}


def evaluate_workflow(gold: dict[str, dict[str, str]], workflow: str) -> list[dict[str, str]]:
    if workflow == "metadata_prior_baseline":
        pred = build_metadata_prior_baseline()
    else:
        pred_path = BASE / workflow / "codificacion_1_ollama.csv"
        pred = read_by_id(pred_path)
    fields = [field for field in next(iter(gold.values())) if field != "paper_id"]
    rows = []
    for paper_id, gold_row in gold.items():
        pred_row = pred.get(paper_id, {})
        for field in fields:
            match, reason = compare_values(
                field,
                gold_row.get(field, ""),
                pred_row.get(field, ""),
                use_dataset_aliases=True,
            )
            rows.append(
                {
                    "workflow": workflow,
                    "paper_id": paper_id,
                    "field": field,
                    "gold": gold_row.get(field, ""),
                    "pred": pred_row.get(field, ""),
                    "match": "yes" if match else "no",
                    "match_reason": reason,
                }
            )
    return rows


def build_metadata_prior_baseline() -> dict[str, dict[str, str]]:
    """Rules-only baseline from corpus metadata plus conservative STGL priors."""
    arxiv_year_by_id = {}
    with open(ARXIV_CACHE, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arxiv_year_by_id[row["arxiv_id"]] = row["year_arxiv"]

    rows = {}
    with open(CORPUS, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            paper_id = row["id_short"]
            domain_label = row["domain"].lower()
            title = row["title"].lower()
            arxiv_id = row["abs_url"].rstrip("/").split("/")[-1]
            year = arxiv_year_by_id.get(arxiv_id, "not_reported")

            domain_class = "traffic"
            task_class = "traffic_forecasting"
            target_class = "traffic_speed_and_flow"
            node_unit = "sensors"
            if "air" in domain_label or "air" in title:
                domain_class = "air_quality"
                task_class = "air_quality_forecasting"
                target_class = "aqi"
                node_unit = "stations"
            elif "covid" in domain_label or "epidem" in domain_label or "covid" in title:
                domain_class = "epidemiology"
                task_class = "covid_forecasting"
                target_class = "covid_cases"
                node_unit = "counties"
            elif "pv" in domain_label or "solar" in title:
                domain_class = "energy_pv"
                task_class = "pv_power_forecasting"
                target_class = "pv_power"
                node_unit = "nodes"
            elif "network" in domain_label or "bandwidth" in title:
                domain_class = "network_traffic"
                task_class = "network_traffic_forecasting"
                target_class = "network_bandwidth"
                node_unit = "nodes"
            elif "multivariate" in domain_label or "timeseries" in domain_label:
                domain_class = "multivariate_timeseries"
                task_class = "multivariate_timeseries_forecasting"
                target_class = "multiple_targets"
                node_unit = "nodes"

            dataset_family = "not_reported"
            title_aliases = {
                "metr-la": "metr_la",
                "metr_la": "metr_la",
                "pems-bay": "pems_bay",
                "pems_bay": "pems_bay",
                "pemsd3": "pemsd3",
                "pemsd4": "pemsd4",
                "pemsd7": "pemsd7",
                "pemsd8": "pemsd8",
            }
            found = [value for key, value in title_aliases.items() if key in title]
            if found:
                dataset_family = "|".join(sorted(set(found)))

            rows[paper_id] = {
                "paper_id": paper_id,
                "year": year,
                "venue_type": "preprint",
                "domain_class": domain_class,
                "task_class": task_class,
                "target_class": target_class,
                "dataset_family": dataset_family,
                "has_public_code": "not_reported",
                "has_public_dataset": "yes",
                "horizon_type": "multi_horizon",
                "horizon_min_value": "15",
                "horizon_max_value": "60",
                "horizon_unit": "minutes",
                "frequency_value": "5",
                "frequency_unit": "minutes",
                "graph_node_count_min": "not_reported",
                "graph_node_count_max": "not_reported",
                "node_unit": node_unit,
                "graph_node_count_status": "partial",
                "coding_status": "partial",
                "confidence": "medium",
            }
    return rows


def prediction_coverage(workflow: str, gold: dict[str, dict[str, str]]) -> tuple[int, int]:
    if workflow == "metadata_prior_baseline":
        pred = build_metadata_prior_baseline()
    else:
        pred = read_by_id(BASE / workflow / "codificacion_1_ollama.csv")
    gold_ids = set(gold)
    return len(gold_ids & set(pred)), len(gold_ids)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def accuracy(rows: list[dict[str, str]], excluded_fields: set[str] | None = None) -> tuple[int, int, float]:
    excluded_fields = excluded_fields or set()
    kept = [row for row in rows if row["field"] not in excluded_fields]
    ok = sum(row["match"] == "yes" for row in kept)
    total = len(kept)
    return ok, total, ok / total if total else 0.0


def summarize_workflows(all_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    summaries = []
    gold = read_by_id(GOLD)
    for workflow, label in WORKFLOWS:
        rows = all_rows[workflow]
        ok20, total20, acc20 = accuracy(rows)
        ok18, total18, acc18 = accuracy(rows, DIAGNOSTIC_FIELDS)
        covered, expected = prediction_coverage(workflow, gold)
        summaries.append(
            {
                "workflow": workflow,
                "label": label,
                "covered_papers": covered,
                "expected_papers": expected,
                "complete_50_paper_output": "yes" if covered == expected else "no",
                "correct_20_fields": ok20,
                "total_20_fields": total20,
                "accuracy_20_fields": f"{acc20:.3f}",
                "correct_18_objective": ok18,
                "total_18_objective": total18,
                "accuracy_18_objective": f"{acc18:.3f}",
                "delta_18_vs_direct": "",
            }
        )
    direct = next(
        float(row["accuracy_18_objective"])
        for row in summaries
        if row["workflow"] == "w0_direct_50"
    )
    for row in summaries:
        row["delta_18_vs_direct"] = f"{float(row['accuracy_18_objective']) - direct:.3f}"
    return summaries


def summarize_fields(rows: list[dict[str, str]], workflow: str) -> list[dict[str, object]]:
    by_field: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_field[row["field"]].append(row)
    out = []
    for field, field_rows in sorted(by_field.items()):
        ok, total, acc = accuracy(field_rows)
        out.append(
            {
                "workflow": workflow,
                "field": field,
                "correct": ok,
                "total": total,
                "errors": total - ok,
                "accuracy": f"{acc:.3f}",
            }
        )
    return out


def summarize_groups(rows: list[dict[str, str]], workflow: str, excluded_fields: set[str] | None = None) -> list[dict[str, object]]:
    excluded_fields = excluded_fields or set()
    out = []
    for group, fields in FIELD_GROUPS.items():
        group_rows = [row for row in rows if row["field"] in fields and row["field"] not in excluded_fields]
        if not group_rows:
            continue
        ok, total, acc = accuracy(group_rows)
        out.append(
            {
                "workflow": workflow,
                "field_group": group,
                "correct": ok,
                "total": total,
                "errors": total - ok,
                "accuracy": f"{acc:.3f}",
            }
        )
    return out


def summarize_posthoc_triage(final_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    field_rows = summarize_fields(final_rows, final_rows[0]["workflow"])
    tier_by_field = {}
    for row in field_rows:
        field = row["field"]
        if field in DIAGNOSTIC_FIELDS:
            tier_by_field[field] = "excluded_diagnostic"
            continue
        acc = float(row["accuracy"])
        if acc >= 0.90:
            tier_by_field[field] = "auto_accept"
        elif acc >= 0.75:
            tier_by_field[field] = "review"
        else:
            tier_by_field[field] = "human_required"

    by_tier: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in final_rows:
        by_tier[tier_by_field[row["field"]]].append(row)

    out = []
    total_instances = sum(len(rows) for tier, rows in by_tier.items() if tier != "excluded_diagnostic")
    for tier in ["auto_accept", "review", "human_required", "excluded_diagnostic"]:
        rows = by_tier.get(tier, [])
        ok, total, acc = accuracy(rows)
        fields = sorted({row["field"] for row in rows})
        denominator = total_instances if tier != "excluded_diagnostic" else len(final_rows)
        out.append(
            {
                "tier": tier,
                "fields": "|".join(fields),
                "field_count": len(fields),
                "instances": total,
                "coverage_excluding_diagnostics": f"{(total / denominator if denominator else 0):.3f}",
                "correct": ok,
                "accuracy": f"{acc:.3f}",
            }
        )
    return out


def main_workflow_rows(workflow_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    order = {workflow: index + 1 for index, (workflow, _) in enumerate(WORKFLOWS)}
    for row in workflow_summary:
        if row["workflow"] not in MAIN_WORKFLOWS:
            continue
        out = dict(row)
        out["order"] = order[row["workflow"]]
        out["display_label"] = WORKFLOW_DISPLAY[row["workflow"]]
        out["model_or_source"] = WORKFLOW_MODEL[row["workflow"]]
        rows.append(out)
    return rows


def figure_workflow_accuracy_rows(workflow_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in main_workflow_rows(workflow_summary):
        rows.append(
            {
                "order": row["order"],
                "workflow": row["workflow"],
                "display_label": row["display_label"],
                "model_or_source": row["model_or_source"],
                "accuracy_20_fields": row["accuracy_20_fields"],
                "accuracy_18_objective": row["accuracy_18_objective"],
            }
        )
    return rows


def model_sensitivity_rows(workflow_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    by_workflow = {str(row["workflow"]): row for row in workflow_summary}
    rows = []
    for model, raw_workflow, access_workflow in MODEL_SENSITIVITY_PAIRS:
        raw = by_workflow[raw_workflow]
        access = by_workflow[access_workflow]
        raw_acc = float(raw["accuracy_18_objective"])
        access_acc = float(access["accuracy_18_objective"])
        rows.append(
            {
                "model": model,
                "raw_workflow": raw_workflow,
                "access_workflow": access_workflow,
                "covered_papers_raw": raw["covered_papers"],
                "covered_papers_access": access["covered_papers"],
                "accuracy_18_raw": f"{raw_acc:.3f}",
                "accuracy_18_access": f"{access_acc:.3f}",
                "access_delta_18": f"{(access_acc - raw_acc):.3f}",
                "accuracy_20_raw": raw["accuracy_20_fields"],
                "accuracy_20_access": access["accuracy_20_fields"],
            }
        )
    return rows


def figure_field_group_rows(final_groups: list[dict[str, object]]) -> list[dict[str, object]]:
    group_order = {
        "temporal_horizon": 1,
        "temporal_frequency": 2,
        "access": 3,
        "graph_nodes": 4,
        "task_dataset": 5,
        "bibliographic": 6,
        "meta": 7,
    }
    rows = []
    for row in sorted(final_groups, key=lambda item: group_order[item["field_group"]]):
        out = dict(row)
        out["order"] = group_order[row["field_group"]]
        rows.append(out)
    return rows


def summarize_residual_error_taxonomy(
    final_rows: list[dict[str, str]],
    excluded_fields: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    excluded_fields = excluded_fields or set()
    error_rows = []
    for row in final_rows:
        if row["field"] in excluded_fields:
            continue
        if row["match"] == "yes":
            continue
        category = ERROR_CATEGORY_BY_FIELD.get(row["field"], "other")
        error_rows.append(
            {
                "paper_id": row["paper_id"],
                "field": row["field"],
                "gold": row["gold"],
                "pred": row["pred"],
                "error_category": category,
            }
        )

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in error_rows:
        by_category[row["error_category"]].append(row)
    summary_rows = []
    total_errors = len(error_rows)
    for category, rows in sorted(by_category.items(), key=lambda item: (-len(item[1]), item[0])):
        fields = sorted({row["field"] for row in rows})
        summary_rows.append(
            {
                "error_category": category,
                "errors": len(rows),
                "share_of_errors": f"{(len(rows) / total_errors if total_errors else 0):.3f}",
                "fields": "|".join(fields),
            }
        )
    return error_rows, summary_rows


def paper_scores(rows: list[dict[str, str]], excluded_fields: set[str]) -> dict[str, tuple[int, int]]:
    scores: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["field"] in excluded_fields:
            continue
        scores[row["paper_id"]][1] += 1
        if row["match"] == "yes":
            scores[row["paper_id"]][0] += 1
    return {paper_id: (ok_total[0], ok_total[1]) for paper_id, ok_total in scores.items()}


def bootstrap_delta(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    excluded_fields: set[str],
    iterations: int,
    seed: int,
) -> dict[str, object]:
    scores_a = paper_scores(rows_a, excluded_fields)
    scores_b = paper_scores(rows_b, excluded_fields)
    paper_ids = sorted(set(scores_a) & set(scores_b))
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sample = [rng.choice(paper_ids) for _ in paper_ids]
        ok_a = sum(scores_a[paper_id][0] for paper_id in sample)
        total_a = sum(scores_a[paper_id][1] for paper_id in sample)
        ok_b = sum(scores_b[paper_id][0] for paper_id in sample)
        total_b = sum(scores_b[paper_id][1] for paper_id in sample)
        deltas.append((ok_b / total_b) - (ok_a / total_a))
    deltas.sort()
    ok_a, total_a, acc_a = accuracy(rows_a, excluded_fields)
    ok_b, total_b, acc_b = accuracy(rows_b, excluded_fields)
    lo = deltas[int(0.025 * iterations)]
    hi = deltas[int(0.975 * iterations)]
    p_two_sided = 2 * min(
        sum(delta <= 0 for delta in deltas) / iterations,
        sum(delta >= 0 for delta in deltas) / iterations,
    )
    return {
        "workflow_a": rows_a[0]["workflow"],
        "workflow_b": rows_b[0]["workflow"],
        "excluded_fields": "|".join(sorted(excluded_fields)) if excluded_fields else "none",
        "accuracy_a": f"{acc_a:.3f}",
        "accuracy_b": f"{acc_b:.3f}",
        "delta_b_minus_a": f"{acc_b - acc_a:.3f}",
        "ci95_low": f"{lo:.3f}",
        "ci95_high": f"{hi:.3f}",
        "bootstrap_p_two_sided": f"{min(p_two_sided, 1.0):.4f}",
        "papers": len(paper_ids),
        "iterations": iterations,
    }


def write_markdown(
    path: Path,
    workflow_summary: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    final_fields: list[dict[str, object]],
    final_groups: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
    taxonomy_summary_rows: list[dict[str, object]],
) -> None:
    def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
        return "\n".join(lines)

    selected_summary = [
        row
        for row in main_workflow_rows(workflow_summary)
    ]
    worst_fields = sorted(final_fields, key=lambda row: (-int(row["errors"]), row["field"]))[:10]
    path.write_text(
        "# Pilot workflow ablation analysis\n\n"
        "Scope: 50-paper expert-coded pilot only, evaluated against `codificacion_1_manual_gold_v5.csv` with dataset-alias normalization. "
        "No independent human-coded validation set is reported in this analysis.\n\n"
        "## Key workflow summary\n\n"
        + md_table(
            selected_summary,
            [
                "workflow",
                "display_label",
                "model_or_source",
                "complete_50_paper_output",
                "accuracy_20_fields",
                "accuracy_18_objective",
                "delta_18_vs_direct",
            ],
        )
        + "\n\n## Paired paper-cluster bootstrap deltas\n\n"
        + md_table(
            bootstrap_rows,
            [
                "workflow_a",
                "workflow_b",
                "excluded_fields",
                "accuracy_a",
                "accuracy_b",
                "delta_b_minus_a",
                "ci95_low",
                "ci95_high",
                "bootstrap_p_two_sided",
            ],
        )
        + "\n\n## Model-family sensitivity for the guided workflow\n\n"
        + md_table(
            model_rows,
            [
                "model",
                "covered_papers_raw",
                "covered_papers_access",
                "accuracy_18_raw",
                "accuracy_18_access",
                "access_delta_18",
            ],
        )
        + "\n\n## Final workflow field groups\n\n"
        + md_table(final_groups, ["field_group", "correct", "total", "errors", "accuracy"])
        + "\n\n## Post-hoc human-review triage simulation\n\n"
        + md_table(
            triage_rows,
            [
                "tier",
                "field_count",
                "instances",
                "coverage_excluding_diagnostics",
                "correct",
                "accuracy",
            ],
        )
        + "\n\n## Residual error taxonomy for final workflow\n\n"
        + md_table(taxonomy_summary_rows, ["error_category", "errors", "share_of_errors", "fields"])
        + "\n\n## Worst final workflow fields\n\n"
        + md_table(worst_fields, ["field", "correct", "total", "errors", "accuracy"])
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--outdir", default=str(OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    gold = read_by_id(GOLD)

    all_rows = {workflow: evaluate_workflow(gold, workflow) for workflow, _ in WORKFLOWS}
    evaluation_rows = [row for rows in all_rows.values() for row in rows]
    write_csv(
        outdir / "pilot_evaluation_long_alias_gold_v5.csv",
        evaluation_rows,
        ["workflow", "paper_id", "field", "gold", "pred", "match", "match_reason"],
    )

    workflow_summary = summarize_workflows(all_rows)
    write_csv(
        outdir / "pilot_workflow_summary_18field.csv",
        workflow_summary,
        [
            "workflow",
            "label",
            "covered_papers",
            "expected_papers",
            "complete_50_paper_output",
            "correct_20_fields",
            "total_20_fields",
            "accuracy_20_fields",
            "correct_18_objective",
            "total_18_objective",
            "accuracy_18_objective",
            "delta_18_vs_direct",
        ],
    )
    main_rows = main_workflow_rows(workflow_summary)
    write_csv(
        outdir / "pilot_main_workflows_table.csv",
        main_rows,
        [
            "order",
            "workflow",
            "display_label",
            "model_or_source",
            "label",
            "covered_papers",
            "expected_papers",
            "complete_50_paper_output",
            "correct_20_fields",
            "total_20_fields",
            "accuracy_20_fields",
            "correct_18_objective",
            "total_18_objective",
            "accuracy_18_objective",
            "delta_18_vs_direct",
        ],
    )
    write_csv(
        outdir / "figure_workflow_accuracy.csv",
        figure_workflow_accuracy_rows(workflow_summary),
        ["order", "workflow", "display_label", "model_or_source", "accuracy_20_fields", "accuracy_18_objective"],
    )
    model_rows = model_sensitivity_rows(workflow_summary)
    write_csv(
        outdir / "pilot_model_sensitivity_table.csv",
        model_rows,
        [
            "model",
            "raw_workflow",
            "access_workflow",
            "covered_papers_raw",
            "covered_papers_access",
            "accuracy_18_raw",
            "accuracy_18_access",
            "access_delta_18",
            "accuracy_20_raw",
            "accuracy_20_access",
        ],
    )

    field_rows = []
    group_rows = []
    for workflow in all_rows:
        field_rows.extend(summarize_fields(all_rows[workflow], workflow))
        group_rows.extend(summarize_groups(all_rows[workflow], workflow))
    write_csv(
        outdir / "pilot_field_accuracy_by_workflow.csv",
        field_rows,
        ["workflow", "field", "correct", "total", "errors", "accuracy"],
    )
    write_csv(
        outdir / "pilot_field_group_accuracy_by_workflow.csv",
        group_rows,
        ["workflow", "field_group", "correct", "total", "errors", "accuracy"],
    )

    bootstrap_rows = []
    for workflow_a, workflow_b in COMPARISONS:
        for excluded in (set(), DIAGNOSTIC_FIELDS):
            bootstrap_rows.append(
                bootstrap_delta(
                    all_rows[workflow_a],
                    all_rows[workflow_b],
                    excluded,
                    args.iterations,
                    args.seed,
                )
            )
    write_csv(
        outdir / "pilot_bootstrap_deltas.csv",
        bootstrap_rows,
        [
            "workflow_a",
            "workflow_b",
            "excluded_fields",
            "accuracy_a",
            "accuracy_b",
            "delta_b_minus_a",
            "ci95_low",
            "ci95_high",
            "bootstrap_p_two_sided",
            "papers",
            "iterations",
        ],
    )

    final_workflow = "w10_schema_v4_access"
    final_fields = summarize_fields(all_rows[final_workflow], final_workflow)
    final_groups = summarize_groups(all_rows[final_workflow], final_workflow, DIAGNOSTIC_FIELDS)
    taxonomy_rows, taxonomy_summary_rows = summarize_residual_error_taxonomy(all_rows[final_workflow], DIAGNOSTIC_FIELDS)
    write_csv(
        outdir / "pilot_residual_error_taxonomy_final.csv",
        taxonomy_rows,
        ["paper_id", "field", "gold", "pred", "error_category"],
    )
    write_csv(
        outdir / "pilot_residual_error_taxonomy_summary_final.csv",
        taxonomy_summary_rows,
        ["error_category", "errors", "share_of_errors", "fields"],
    )
    write_csv(
        outdir / "figure_field_group_accuracy_final.csv",
        figure_field_group_rows(final_groups),
        ["order", "workflow", "field_group", "correct", "total", "errors", "accuracy"],
    )
    triage_rows = summarize_posthoc_triage(all_rows[final_workflow])
    write_csv(
        outdir / "pilot_posthoc_triage_simulation.csv",
        triage_rows,
        [
            "tier",
            "fields",
            "field_count",
            "instances",
            "coverage_excluding_diagnostics",
            "correct",
            "accuracy",
        ],
    )
    write_markdown(
        outdir / "pilot_workflow_ablation_summary.md",
        workflow_summary,
        bootstrap_rows,
        model_rows,
        final_fields,
        final_groups,
        triage_rows,
        taxonomy_summary_rows,
    )
    print(f"Saved analysis artifacts under {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

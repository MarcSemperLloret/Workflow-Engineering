"""
New workflow variants for coder-1 closed-label extraction.

Workflows
---------
w10_guideline_fewshot
    Grouped extraction (same 4 groups as w3/w4) with 3 annotated few-shot
    examples per group embedded in the prompt.
    Hypothesis: explicit examples reduce convention errors for the fields that
    are systematically hard — venue_type, horizon_type, frequency vs horizon
    disambiguation, target_class (speed vs flow vs both), dataset_family labels.

w11_cot_grouped
    Grouped extraction with structured chain-of-thought.
    The model returns {"analysis": {field: "brief reasoning"}, "values": {field: label}}.
    Post-processing extracts the "values" dict; if missing, falls back to treating
    the whole response as a values dict (graceful degradation to w4 behaviour).
    Hypothesis: separating evidence reasoning from label assignment reduces
    hallucination and improves stability for ambiguous fields.

Both workflows
    - use the same field-group structure as w3/w4 (4 LLM calls per paper)
    - apply normalize_coded_row() and validate_row() identically
    - produce CSV + per-paper JSON compatible with evaluate_codificacion_1.py

Examples in w10 are constructed from domain knowledge and do NOT include any of
the 20 pilot papers; there is no leakage from the gold standard.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_codificacion_1_ollama import (  # noqa: E402
    ALLOWED_DATASET_FAMILIES,
    ARXIV_CACHE_CSV,
    CORPUS_CSV,
    DEFAULT_OUT_DIR,
    ENUMS,
    FIELD_GROUPS,
    FIELDNAMES,
    GROUP_INSTRUCTIONS,
    PARSED_DIR,
    SCHEMA_PATH,
    arxiv_id_from_url,
    coerce_group_output,
    extract_json,
    normalize_coded_row,
    paper_context_v1,
    partial_schema,
    read_arxiv_cache,
    read_corpus,
    validate_row,
)

# ---------------------------------------------------------------------------
# Few-shot examples for w10
# Each entry: snippet shown to the model, correct closed-label coding, decision note.
# These are constructed examples — NOT from the 20 pilot papers.
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES: dict[str, list[dict]] = {
    "bibliography_access": [
        {
            "snippet": (
                "Published as a conference paper at ICLR 2020. "
                "Our source code is available at https://github.com/author/model-repo."
            ),
            "values": {
                "year": "2020",
                "venue_type": "conference",
                "has_public_code": "yes",
                "has_public_dataset": "not_reported",
            },
            "note": (
                "ICLR is a conference; the arXiv preprint does not change venue_type. "
                "GitHub link for method code → has_public_code=yes."
            ),
        },
        {
            "snippet": (
                "arXiv:2103.05678 [cs.LG]. "
                "We benchmark on the public METR-LA and PEMS-BAY datasets. "
                "Code is not released."
            ),
            "values": {
                "venue_type": "preprint",
                "has_public_code": "no",
                "has_public_dataset": "yes",
            },
            "note": (
                "Only on arXiv, no journal/conference mention → preprint. "
                "Named public benchmarks → has_public_dataset=yes. "
                "Explicit 'code is not released' → has_public_code=no."
            ),
        },
        {
            "snippet": (
                "Accepted at the 34th AAAI Conference on Artificial Intelligence (AAAI-2020). "
                "Dataset available at https://github.com/data/dataset-repo. "
                "The model implementation is not publicly available."
            ),
            "values": {
                "venue_type": "conference",
                "has_public_code": "no",
                "has_public_dataset": "yes",
            },
            "note": (
                "AAAI is a conference. "
                "A dataset GitHub link does NOT make has_public_code=yes; "
                "only method/source code counts. "
                "'Not publicly available' → has_public_code=no."
            ),
        },
    ],
    "task_target": [
        {
            "snippet": (
                "We propose a graph convolutional network to predict "
                "15-min average traffic speed at 207 loop detectors in Los Angeles."
            ),
            "values": {
                "domain_class": "traffic",
                "task_class": "traffic_forecasting",
                "target_class": "traffic_speed",
            },
            "note": (
                "Explicitly 'traffic speed' only → target_class=traffic_speed. "
                "Do not use traffic_speed_and_flow unless both are predicted."
            ),
        },
        {
            "snippet": (
                "We simultaneously forecast traffic flow volume and average speed "
                "on PeMSD4 (flow) and METR-LA (speed), showing our model generalises "
                "across target types."
            ),
            "values": {
                "domain_class": "traffic",
                "task_class": "traffic_forecasting",
                "target_class": "traffic_speed_and_flow",
            },
            "note": (
                "Both speed and flow are predicted → traffic_speed_and_flow. "
                "task_class stays traffic_forecasting; benchmark_evaluation "
                "is for papers whose main contribution is comparing baselines."
            ),
        },
        {
            "snippet": (
                "We conduct a comprehensive benchmark study comparing 12 spatio-temporal "
                "GNN baselines on METR-LA (speed), PeMSD4 (flow) and PeMSD8 (flow). "
                "Our contribution is the evaluation protocol, not a new model."
            ),
            "values": {
                "domain_class": "traffic",
                "task_class": "benchmark_evaluation",
                "target_class": "traffic_speed_and_flow",
            },
            "note": (
                "Focus is comparing existing baselines → task_class=benchmark_evaluation. "
                "target_class reflects what is predicted, not the paper's contribution."
            ),
        },
    ],
    "data_temporal": [
        {
            "snippet": (
                "We use METR-LA (207 sensors) and PEMS-BAY (325 sensors). "
                "Traffic data are aggregated into 5-minute intervals. "
                "We predict traffic conditions at horizons of 15, 30, and 60 minutes."
            ),
            "values": {
                "dataset_family": "metr_la|pems_bay",
                "frequency_value": "5",
                "frequency_unit": "minutes",
                "horizon_min_value": "15",
                "horizon_max_value": "60",
                "horizon_type": "multi_horizon",
                "horizon_unit": "minutes",
            },
            "note": (
                "5-minute interval = data recording frequency, NOT the horizon. "
                "Three prediction horizons → multi_horizon. "
                "Dataset labels joined by | without spaces."
            ),
        },
        {
            "snippet": (
                "Following prior work, we predict the next 12 time steps "
                "(each step equals one 5-minute interval). "
                "Experiments use METR-LA and PEMS-BAY."
            ),
            "values": {
                "dataset_family": "metr_la|pems_bay",
                "frequency_value": "5",
                "frequency_unit": "minutes",
                "horizon_min_value": "12",
                "horizon_max_value": "12",
                "horizon_type": "sequence",
                "horizon_unit": "steps",
            },
            "note": (
                "Fixed number of future steps (not minutes) → horizon_type=sequence, "
                "horizon_unit=steps. "
                "The 5-minute interval is the frequency, not the horizon unit."
            ),
        },
        {
            "snippet": (
                "COVID-19 daily case counts for 20 US counties from the NYT dataset. "
                "We forecast 1 day ahead using a graph epidemic model. "
                "Mobility data from Google COVID-19 Community Mobility Reports are used as covariates."
            ),
            "values": {
                "dataset_family": "nyt_covid|google_mobility",
                "frequency_value": "1",
                "frequency_unit": "days",
                "horizon_min_value": "1",
                "horizon_max_value": "1",
                "horizon_type": "single_horizon",
                "horizon_unit": "days",
            },
            "note": (
                "Daily data → frequency=1, frequency_unit=days. "
                "One step ahead → single_horizon. "
                "Both NYT and Google Mobility are used → include both labels."
            ),
        },
    ],
    "graph_quality": [
        {
            "snippet": (
                "METR-LA contains 207 loop detectors; PEMS-BAY has 325 sensors. "
                "Both datasets are used in our experiments."
            ),
            "values": {
                "graph_node_count_min": "207",
                "graph_node_count_max": "325",
                "node_unit": "sensors",
                "graph_node_count_status": "found",
            },
            "note": (
                "Multiple datasets → min=207, max=325. "
                "Both counts reported → graph_node_count_status=found. "
                "Sort so that min ≤ max."
            ),
        },
        {
            "snippet": (
                "We evaluate on PeMSD4 (307 sensors) and PeMSD8 (170 sensors), "
                "and also on a proprietary urban dataset whose sensor count is not disclosed."
            ),
            "values": {
                "graph_node_count_min": "170",
                "graph_node_count_max": "307",
                "node_unit": "sensors",
                "graph_node_count_status": "partial",
            },
            "note": (
                "Proprietary dataset lacks a node count → partial. "
                "Use only the known counts for min/max: 170 ≤ 307."
            ),
        },
        {
            "snippet": (
                "The road network in our California highway dataset has 11,160 road segments. "
                "Data from Caltrans PeMS."
            ),
            "values": {
                "graph_node_count_min": "11160",
                "graph_node_count_max": "11160",
                "node_unit": "road_segments",
                "graph_node_count_status": "found",
            },
            "note": (
                "Single dataset → min=max=11160. "
                "Remove commas from numeric strings. "
                "Use road_segments, not roads or segments."
            ),
        },
    ],
}


def format_fewshot_block(group_name: str) -> str:
    examples = FEWSHOT_EXAMPLES.get(group_name, [])
    if not examples:
        return ""
    lines = ["CODING EXAMPLES (follow these conventions):"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"\nExample {i}:")
        lines.append(f'  Text: "{ex["snippet"]}"')
        lines.append(f'  Coding: {json.dumps(ex["values"], ensure_ascii=False)}')
        lines.append(f'  Rule: {ex["note"]}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def fewshot_grouped_prompt(group_name: str, fields: list[str], schema: dict, context: str) -> str:
    dataset_labels = ""
    if group_name == "data_temporal":
        dataset_labels = (
            "\nAllowed dataset_family labels (use | as separator for multiple):\n"
            + ", ".join(ALLOWED_DATASET_FAMILIES)
            + "\n"
        )
    fewshot_block = format_fewshot_block(group_name)
    return (
        "You are coding scientific papers into a closed-label schema.\n\n"
        "Return exactly one JSON object. Do not include Markdown, comments, or evidence text.\n"
        "Use only the allowed labels from the schema. "
        "If a value is not clearly reported, use `not_reported`.\n"
        "If the paper is unclear or conflicting, use `ambiguous` when that value is allowed.\n"
        "Numeric fields must be digit strings, or one of: not_reported, ambiguous, not_applicable, dataset_specific.\n"
        "Multi-label dataset fields must use `|` as the separator (no spaces).\n\n"
        f"FIELD GROUP: {group_name}\n"
        f"{GROUP_INSTRUCTIONS[group_name]}\n"
        + dataset_labels
        + "\n"
        + fewshot_block
        + "\n\nReturn only these fields plus paper_id:\n"
        + ", ".join(["paper_id"] + fields)
        + "\n\nJSON SCHEMA:\n"
        + partial_schema(schema, fields)
        + "\n\nPaper metadata and extracted text follow.\n\n"
        + context
    )


def cot_grouped_prompt(group_name: str, fields: list[str], schema: dict, context: str) -> str:
    dataset_labels = ""
    if group_name == "data_temporal":
        dataset_labels = (
            "\nAllowed dataset_family labels (use | as separator for multiple):\n"
            + ", ".join(ALLOWED_DATASET_FAMILIES)
            + "\n"
        )
    field_list = ", ".join(["paper_id"] + fields)
    return (
        "You are coding scientific papers into a closed-label schema.\n\n"
        "Return exactly one JSON object with two top-level keys:\n"
        '  "analysis": an object where each field has a brief reasoning string '
        "(quote the relevant text span, then state your conclusion).\n"
        '  "values": an object with the final closed-label value for each field.\n\n'
        "In `values`, use only the allowed labels from the schema. "
        "If a value is not clearly reported, use `not_reported`. "
        "If unclear or conflicting, use `ambiguous` when allowed.\n"
        "Numeric fields must be digit strings, or one of: not_reported, ambiguous, not_applicable, dataset_specific.\n"
        "Multi-label dataset fields must use `|` as the separator.\n\n"
        f"FIELD GROUP: {group_name}\n"
        f"{GROUP_INSTRUCTIONS[group_name]}\n"
        + dataset_labels
        + "\nFields to code: " + field_list
        + "\n\nJSON SCHEMA FOR VALUES:\n"
        + partial_schema(schema, fields)
        + "\n\nPaper metadata and extracted text follow.\n\n"
        + context
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    timeout: int,
    temperature: float = 0.0,
    think: bool = False,
) -> str:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": 0.1 if temperature == 0.0 else 0.9,
            "num_ctx": 32768,
        },
    }
    if not think:
        payload["think"] = False
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result.get("response", "")


# ---------------------------------------------------------------------------
# COT response coercion
# ---------------------------------------------------------------------------

def coerce_cot_group_output(raw: dict, paper_id: str, fields: list[str]) -> dict:
    values = raw.get("values", raw)
    if not isinstance(values, dict):
        values = raw
    return coerce_group_output(values, paper_id, fields)


# ---------------------------------------------------------------------------
# Per-paper coding loops
# ---------------------------------------------------------------------------

def code_fewshot(
    paper_id: str,
    context: str,
    schema: dict,
    host: str,
    model: str,
    timeout: int,
) -> tuple[dict, dict, float]:
    coded = {field: "not_reported" for field in FIELDNAMES}
    coded["paper_id"] = paper_id
    raw_responses: dict = {}
    elapsed_total = 0.0

    for group_name, fields in FIELD_GROUPS.items():
        prompt = fewshot_grouped_prompt(group_name, fields, schema, context)
        started = time.time()
        try:
            response = ollama_generate(host, model, prompt, timeout)
            raw = extract_json(response)
            group_coded = coerce_group_output(raw, paper_id, fields)
            coded.update(group_coded)
            raw_responses[group_name] = response
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raw_responses[group_name] = f"runtime_error:{type(exc).__name__}:{exc}"
        elapsed_total += time.time() - started

    return normalize_coded_row(coded), raw_responses, elapsed_total


def code_cot(
    paper_id: str,
    context: str,
    schema: dict,
    host: str,
    model: str,
    timeout: int,
) -> tuple[dict, dict, float]:
    coded = {field: "not_reported" for field in FIELDNAMES}
    coded["paper_id"] = paper_id
    raw_responses: dict = {}
    elapsed_total = 0.0

    for group_name, fields in FIELD_GROUPS.items():
        prompt = cot_grouped_prompt(group_name, fields, schema, context)
        started = time.time()
        try:
            response = ollama_generate(host, model, prompt, timeout)
            raw = extract_json(response)
            group_coded = coerce_cot_group_output(raw, paper_id, fields)
            coded.update(group_coded)
            raw_responses[group_name] = response
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raw_responses[group_name] = f"runtime_error:{type(exc).__name__}:{exc}"
        elapsed_total += time.time() - started

    return normalize_coded_row(coded), raw_responses, elapsed_total


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def selected_papers(args: argparse.Namespace) -> list[dict]:
    corpus = read_corpus()
    if args.paper_id:
        wanted = set(args.paper_id)
        papers = [row for row in corpus if row["id_short"] in wanted]
    elif args.pilot:
        papers = [row for row in corpus if row.get("include_in_pilot", "").lower() == "yes"]
    else:
        papers = corpus
    return papers[: args.limit] if args.limit else papers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

WORKFLOW_FN = {
    "w10_guideline_fewshot": code_fewshot,
    "w11_cot_grouped": code_cot,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run new coder-1 workflow variants (w10, w11)."
    )
    parser.add_argument(
        "--workflow",
        required=True,
        choices=list(WORKFLOW_FN),
        help="Which workflow to run.",
    )
    parser.add_argument("--pilot", action="store_true", help="Run pilot papers only.")
    parser.add_argument("--paper-id", action="append", help="Run specific paper IDs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-name", default=None, help="Override output subdirectory name.")
    parser.add_argument("--max-chars", type=int, default=28000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Enable thinking tokens for qwen3 models (off by default).",
    )
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    arxiv_cache = read_arxiv_cache()
    papers = selected_papers(args)

    out_dir = Path(args.output_dir) / (args.run_name or args.workflow)
    out_dir.mkdir(parents=True, exist_ok=True)

    code_fn = WORKFLOW_FN[args.workflow]
    rows: list[dict] = []

    for index, paper in enumerate(papers, start=1):
        paper_id = paper["id_short"]
        parsed_path = PARSED_DIR / f"{paper_id}.json"
        if not parsed_path.exists():
            print(f"[{index}/{len(papers)}] MISSING parsed text: {paper_id}", file=sys.stderr)
            continue

        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        context = paper_context_v1(paper, parsed, arxiv_cache, args.max_chars)

        print(
            f"[{index}/{len(papers)}] coding {paper_id} "
            f"workflow={args.workflow} model={args.model}"
        )

        try:
            coded, raw_responses, elapsed = code_fn(
                paper_id, context, schema, args.ollama_host, args.model, args.timeout
            )
            errors = validate_row(coded)
        except Exception as exc:
            coded = {field: "not_reported" for field in FIELDNAMES}
            coded["paper_id"] = paper_id
            errors = [f"runtime_error:{type(exc).__name__}:{exc}"]
            raw_responses = {}
            elapsed = 0.0

        rows.append(coded)
        record = {
            "paper_id": paper_id,
            "model": args.model,
            "workflow": args.workflow,
            "elapsed_seconds": round(elapsed, 2),
            "validation_errors": errors,
            "coded": coded,
            "raw_response": raw_responses,
        }
        (out_dir / f"{paper_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  saved {paper_id}.json errors={len(errors)} elapsed={round(elapsed, 2)}s")

    csv_path = out_dir / "codificacion_1_ollama.csv"
    write_csv(csv_path, rows)
    print(f"Saved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

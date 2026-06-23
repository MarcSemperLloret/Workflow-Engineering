"""
Run advanced coder-1 workflow variants.

Workflows:
  w5_evidence_required: value + short evidence span per field.
  w6_second_pass_audit: audit the current hybrid output on selected fields.
  w7_retrieval_tables: schema extraction from table/caption-heavy retrieved context.
  w8_self_consistency: consensus over repeated w7-style samples.
  w9_multi_model_panel: field-wise consensus across model outputs.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_codificacion_1_ollama import (  # noqa: E402
    ALLOWED_DATASET_FAMILIES,
    ARXIV_CACHE_CSV,
    CORPUS_CSV,
    DEFAULT_OUT_DIR,
    ENUMS,
    FIELDNAMES,
    PARSED_DIR,
    PROMPT_PATH,
    SCHEMA_PATH,
    arxiv_id_from_url,
    extract_json,
    focused_snippets,
    normalize_coded_row,
    normalize_text,
    paper_context_v1,
    read_arxiv_cache,
    read_corpus,
    validate_row,
)

HYBRID_CSV = DEFAULT_OUT_DIR / "w4_hybrid_w2_w3" / "codificacion_1_ollama.csv"

AUDIT_FIELDS = [
    "dataset_family", "has_public_code", "has_public_dataset",
    "horizon_max_value", "horizon_unit",
    "graph_node_count_min", "graph_node_count_max", "node_unit",
    "graph_node_count_status", "coding_status", "confidence",
]


def ollama_generate(host: str, model: str, prompt: str, timeout: int, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": 0.9 if temperature > 0 else 0.1,
            "num_ctx": 32768,
        },
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8")).get("response", "")


def read_csv_by_id(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["paper_id"]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def base_context(row: dict, parsed: dict, arxiv_cache: dict, max_chars: int) -> str:
    return paper_context_v1(row, parsed, arxiv_cache, max_chars)


def captions_context(parsed: dict) -> str:
    parts = []
    for key, title in (("table_captions", "TABLE CAPTIONS"), ("figure_captions", "FIGURE CAPTIONS")):
        items = parsed.get(key, [])
        if items:
            lines = []
            for item in items[:30]:
                caption = normalize_text(str(item.get("caption", "")))
                if caption:
                    lines.append(f"- {caption}")
            if lines:
                parts.append(title + ":\n" + "\n".join(lines))
    return "\n\n".join(parts)


def retrieval_context(row: dict, parsed: dict, arxiv_cache: dict, max_chars: int) -> str:
    sections = parsed.get("sections", {})
    evidence = [captions_context(parsed), focused_snippets(parsed)]
    retrieval_patterns = [
        r".{0,220}(?:METR-LA|PEMS-BAY|PeMSD\d|PeMSD7|PeMSD8|Los-loop|Seattle-Loop|NE-BJ|Xiamen|Tysons?|Radflow|COVID|Google Mobility|NYT).{0,260}",
        r".{0,220}(?:5 minutes|15 min|30 min|45 min|60 min|1 hour|daily|sampling|aggregat(?:e|ed|ion)|horizon|forecasting horizon).{0,260}",
        r".{0,220}(?:nodes|sensors|stations|counties|road segments|links|# Nodes|Number of nodes).{0,260}",
        r".{0,220}(?:github|source code|code is|available at|publicly available).{0,260}",
    ]
    snippets = []
    for section_name, text in sections.items():
        if section_name == "references":
            continue
        text = normalize_text(text)
        for pattern in retrieval_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                snippet = re.sub(r"\s+", " ", match.group(0)).strip()
                item = f"[{section_name}] {snippet}"
                if item.lower() not in {s.lower() for s in snippets}:
                    snippets.append(item)
                if len(snippets) >= 40:
                    break
            if len(snippets) >= 40:
                break
        if len(snippets) >= 40:
            break
    evidence.append("RETRIEVED FRAGMENTS:\n" + "\n".join(f"- {s}" for s in snippets))
    metadata = {
        "paper_id": row["id_short"],
        "title": row["title"],
        "domain_hint": row.get("domain", ""),
        "abs_url": row.get("abs_url", ""),
        "arxiv_metadata": arxiv_cache.get(arxiv_id_from_url(row.get("abs_url", "")), {}),
    }
    context = "STRUCTURED METADATA:\n" + json.dumps(metadata, ensure_ascii=False, indent=2)
    context += "\n\nRETRIEVAL-FIRST CONTEXT:\n" + "\n\n".join(part for part in evidence if part)
    return context[:max_chars] + ("\n[TRUNCATED]" if len(context) > max_chars else "")


def schema_prompt(context: str) -> str:
    return (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\n\nJSON SCHEMA:\n"
        + SCHEMA_PATH.read_text(encoding="utf-8")
        + "\n\n"
        + context
    )


def evidence_prompt(context: str) -> str:
    return (
        "You are coding scientific papers into a closed-label schema.\n"
        "Return exactly one JSON object with key `fields`.\n"
        "For each field, return an object with `value` and `evidence`.\n"
        "The `value` must use the same closed labels and numeric conventions as the schema.\n"
        "The `evidence` must be a short copied or closely paraphrased text span from the paper. "
        "If no evidence exists, use `not_reported` as the value and an empty evidence string.\n"
        "Do not include Markdown or explanations.\n\n"
        "Allowed dataset_family labels: "
        + ", ".join(ALLOWED_DATASET_FAMILIES)
        + "\n\nJSON SCHEMA FOR VALUES:\n"
        + SCHEMA_PATH.read_text(encoding="utf-8")
        + "\n\n"
        + context
    )


def audit_prompt(context: str, current: dict) -> str:
    subset = {field: current[field] for field in ["paper_id"] + AUDIT_FIELDS if field in current}
    return (
        "You are auditing a closed-label extraction from a scientific paper.\n"
        "Return exactly one JSON object containing only the audited fields plus paper_id.\n"
        "Change a value only if the paper evidence clearly contradicts the current value.\n"
        "Prefer the current value when evidence is weak or absent.\n"
        "Use the closed labels and numeric conventions from the schema.\n\n"
        "CURRENT VALUES:\n"
        + json.dumps(subset, ensure_ascii=False, indent=2)
        + "\n\nJSON SCHEMA:\n"
        + SCHEMA_PATH.read_text(encoding="utf-8")
        + "\n\n"
        + context
    )


def coerce_full(raw: dict, paper_id: str) -> dict:
    row = {field: str(raw.get(field, "not_reported")).strip() for field in FIELDNAMES}
    row["paper_id"] = paper_id
    return normalize_coded_row(row)


def coerce_evidence(raw: dict, paper_id: str) -> tuple[dict, dict]:
    fields = raw.get("fields", raw)
    row = {"paper_id": paper_id}
    evidence = {}
    for field in FIELDNAMES:
        if field == "paper_id":
            continue
        item = fields.get(field, {})
        if isinstance(item, dict):
            row[field] = str(item.get("value", "not_reported")).strip()
            evidence[field] = str(item.get("evidence", "")).strip()
        else:
            row[field] = str(item).strip()
            evidence[field] = ""
    return normalize_coded_row(row), evidence


def consensus(rows: list[dict]) -> dict:
    merged = {"paper_id": rows[0]["paper_id"]}
    for field in FIELDNAMES:
        if field == "paper_id":
            continue
        counts = Counter(row.get(field, "not_reported") for row in rows)
        merged[field] = counts.most_common(1)[0][0]
    return merged


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


def run_llm_workflow(args: argparse.Namespace) -> int:
    arxiv_cache = read_arxiv_cache()
    out_dir = Path(args.output_dir) / (args.run_name or args.workflow)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, paper in enumerate(selected_papers(args), start=1):
        paper_id = paper["id_short"]
        parsed = json.loads((PARSED_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
        context = retrieval_context(paper, parsed, arxiv_cache, args.max_chars) if args.workflow == "w7_retrieval_tables" else base_context(paper, parsed, arxiv_cache, args.max_chars)
        started = time.time()
        raw_response = ""
        evidence = {}
        try:
            if args.workflow == "w5_evidence_required":
                raw_response = ollama_generate(args.ollama_host, args.model, evidence_prompt(context), args.timeout)
                coded, evidence = coerce_evidence(extract_json(raw_response), paper_id)
            elif args.workflow == "w7_retrieval_tables":
                raw_response = ollama_generate(args.ollama_host, args.model, schema_prompt(context), args.timeout)
                coded = coerce_full(extract_json(raw_response), paper_id)
            elif args.workflow == "w8_self_consistency":
                samples = []
                raw_response = []
                for sample_index in range(args.samples):
                    response = ollama_generate(args.ollama_host, args.model, schema_prompt(retrieval_context(paper, parsed, arxiv_cache, args.max_chars)), args.timeout, temperature=0.4)
                    raw_response.append(response)
                    samples.append(coerce_full(extract_json(response), paper_id))
                coded = consensus(samples)
            else:
                raise ValueError(f"Unsupported LLM workflow: {args.workflow}")
            errors = validate_row(coded)
        except Exception as exc:
            coded = {field: "not_reported" for field in FIELDNAMES}
            coded["paper_id"] = paper_id
            errors = [f"runtime_error:{type(exc).__name__}:{exc}"]
        elapsed = round(time.time() - started, 2)
        rows.append(coded)
        record = {
            "paper_id": paper_id,
            "model": args.model,
            "workflow": args.workflow,
            "elapsed_seconds": elapsed,
            "validation_errors": errors,
            "coded": coded,
            "evidence": evidence,
            "raw_response": raw_response,
        }
        (out_dir / f"{paper_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}] {paper_id} {args.workflow} errors={len(errors)} elapsed={elapsed}s")
    write_csv(out_dir / "codificacion_1_ollama.csv", rows)
    return 0


def run_audit(args: argparse.Namespace) -> int:
    arxiv_cache = read_arxiv_cache()
    current_rows = read_csv_by_id(Path(args.input_csv))
    out_dir = Path(args.output_dir) / (args.run_name or args.workflow)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, paper in enumerate(selected_papers(args), start=1):
        paper_id = paper["id_short"]
        current = dict(current_rows[paper_id])
        parsed = json.loads((PARSED_DIR / f"{paper_id}.json").read_text(encoding="utf-8"))
        context = retrieval_context(paper, parsed, arxiv_cache, args.max_chars)
        started = time.time()
        try:
            response = ollama_generate(args.ollama_host, args.model, audit_prompt(context, current), args.timeout)
            audited = extract_json(response)
            coded = dict(current)
            for field in AUDIT_FIELDS:
                if field in audited:
                    coded[field] = str(audited[field]).strip()
            coded = normalize_coded_row(coded)
            errors = validate_row(coded)
        except Exception as exc:
            response = ""
            coded = current
            errors = [f"runtime_error:{type(exc).__name__}:{exc}"]
        elapsed = round(time.time() - started, 2)
        rows.append(coded)
        record = {
            "paper_id": paper_id,
            "model": args.model,
            "workflow": args.workflow,
            "elapsed_seconds": elapsed,
            "validation_errors": errors,
            "coded": coded,
            "raw_response": response,
        }
        (out_dir / f"{paper_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}] {paper_id} {args.workflow} errors={len(errors)} elapsed={elapsed}s")
    write_csv(out_dir / "codificacion_1_ollama.csv", rows)
    return 0


def run_multi_model_panel(args: argparse.Namespace) -> int:
    workflow_dirs = [Path(args.output_dir) / workflow for workflow in args.panel_workflow]
    sources = []
    for workflow_dir in workflow_dirs:
        csv_path = workflow_dir / "codificacion_1_ollama.csv"
        if csv_path.exists():
            sources.append(read_csv_by_id(csv_path))
    if len(sources) < 2:
        raise FileNotFoundError("Need at least two existing workflow CSVs for panel consensus")
    out_dir = Path(args.output_dir) / (args.run_name or args.workflow)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for paper_id in sources[0]:
        candidates = [source[paper_id] for source in sources if paper_id in source]
        rows.append(consensus(candidates))
    write_csv(out_dir / "codificacion_1_ollama.csv", rows)
    (out_dir / "panel_sources.json").write_text(json.dumps(args.panel_workflow, indent=2), encoding="utf-8")
    print(f"Saved panel consensus: {out_dir / 'codificacion_1_ollama.csv'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True, choices=[
        "w5_evidence_required", "w6_second_pass_audit", "w7_retrieval_tables",
        "w8_self_consistency", "w9_multi_model_panel",
    ])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--paper-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--input-csv", default=str(HYBRID_CSV))
    parser.add_argument("--panel-workflow", action="append", default=[])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=28000)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if args.workflow == "w6_second_pass_audit":
        return run_audit(args)
    if args.workflow == "w9_multi_model_panel":
        return run_multi_model_panel(args)
    return run_llm_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())

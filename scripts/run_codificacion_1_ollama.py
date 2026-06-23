"""
Run coder-1 closed-label extraction with Ollama.

This is intended to be launched either directly or from n8n.

Examples:
  python scripts/run_codificacion_1_ollama.py --pilot --limit 1
  python scripts/run_codificacion_1_ollama.py --paper-id DCRNN --model qwen2.5-coder:7b
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
CORPUS_CSV = ROOT / "corpus_inicial_50_papers_stgl.csv"
PARSED_DIR = ROOT / "data" / "parsed_text"
PROMPT_PATH = ROOT / "prompts" / "codificacion_1_prompt.md"
SCHEMA_PATH = ROOT / "schemas" / "codificacion_1_schema.json"
DEFAULT_OUT_DIR = ROOT / "data" / "model_outputs" / "codificacion_1_ollama"
ARXIV_CACHE_CSV = ROOT / "data" / "annotations" / "arxiv_metadata_cache.csv"

FIELDNAMES = [
    "paper_id", "year", "venue_type", "domain_class", "task_class",
    "target_class", "dataset_family", "has_public_code", "has_public_dataset",
    "horizon_type", "horizon_min_value", "horizon_max_value", "horizon_unit",
    "frequency_value", "frequency_unit", "graph_node_count_min",
    "graph_node_count_max", "node_unit", "graph_node_count_status",
    "coding_status", "confidence",
]

ENUMS = {
    "venue_type": {"conference", "journal", "preprint", "competition_workshop", "not_reported"},
    "domain_class": {"traffic", "air_quality", "energy_pv", "epidemiology", "network_traffic", "general_network_timeseries", "multivariate_timeseries"},
    "task_class": {"traffic_forecasting", "travel_time_forecasting", "air_quality_forecasting", "pv_power_forecasting", "covid_forecasting", "network_traffic_forecasting", "network_timeseries_prediction", "multivariate_timeseries_forecasting", "benchmark_evaluation"},
    "target_class": {"traffic_speed", "traffic_flow", "traffic_speed_and_flow", "travel_time", "aqi", "pv_power", "covid_cases", "network_bandwidth", "multiple_targets"},
    "has_public_code": {"yes", "no", "not_reported"},
    "has_public_dataset": {"yes", "no", "not_reported"},
    "horizon_type": {"single_horizon", "multi_horizon", "sequence", "dataset_specific", "not_reported"},
    "horizon_unit": {"minutes", "hours", "steps", "days", "weeks", "mixed", "dataset_specific", "not_reported", "not_applicable"},
    "frequency_unit": {"seconds", "minutes", "hours", "days", "mixed", "dataset_specific", "not_reported", "not_applicable"},
    "node_unit": {"sensors", "stations", "road_segments", "counties", "links", "nodes", "mixed", "not_reported"},
    "graph_node_count_status": {"found", "partial", "not_reported", "ambiguous"},
    "coding_status": {"complete", "partial"},
    "confidence": {"high", "medium", "low"},
}

FIELD_GROUPS = {
    "bibliography_access": [
        "year", "venue_type", "has_public_code", "has_public_dataset",
    ],
    "task_target": [
        "domain_class", "task_class", "target_class",
    ],
    "data_temporal": [
        "dataset_family", "horizon_type", "horizon_min_value", "horizon_max_value",
        "horizon_unit", "frequency_value", "frequency_unit",
    ],
    "graph_quality": [
        "graph_node_count_min", "graph_node_count_max", "node_unit",
        "graph_node_count_status", "coding_status", "confidence",
    ],
}

ALLOWED_DATASET_FAMILIES = [
    "metr_la", "pems_bay", "pemsd3", "pemsd4", "pemsd7", "pemsd8",
    "bjer4", "xiamen", "california_highway_pems", "amap_beijing",
    "nyt_covid", "google_mobility", "yangtze_air_quality", "vevo_music",
    "wiki_traffic", "los_loop", "sz_taxi", "traffic", "solar_energy",
    "electricity", "exchange_rate", "ne_bj", "swiss_pv_real",
    "swiss_pv_synthetic", "tysons_corner", "seattle_loop", "not_reported",
]

GROUP_INSTRUCTIONS = {
    "bibliography_access": (
        "Code bibliographic/access fields only. Use arXiv metadata for year and preprint status, "
        "but prefer explicit publication venue statements in the paper when present. "
        "Code availability requires source/method code, not only a dataset URL. "
        "Use has_public_code=no when no method code is reported. "
        "Use has_public_dataset=yes when the paper uses named public benchmark datasets such as METR-LA, "
        "PEMS-BAY, PeMSD datasets, Los-loop, NE-BJ, or similar public datasets."
    ),
    "task_target": (
        "Code task and target fields only. Distinguish the prediction task from the model family. "
        "For traffic papers, target_class must separate speed, flow, speed_and_flow, and travel_time."
    ),
    "data_temporal": (
        "Code dataset and temporal setup only. Do not confuse prediction horizon with sampling frequency. "
        "Frequency is the data recording/aggregation interval; horizon is the future prediction span."
    ),
    "graph_quality": (
        "Code graph/node and quality fields only. For multiple datasets, use min and max node counts. "
        "Use graph_node_count_status=partial when only some datasets have node counts."
    ),
}


def read_corpus(path: Path = CORPUS_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_arxiv_cache() -> dict[str, dict]:
    if not ARXIV_CACHE_CSV.exists():
        return {}
    with open(ARXIV_CACHE_CSV, newline="", encoding="utf-8-sig") as handle:
        return {row["arxiv_id"]: row for row in csv.DictReader(handle)}


def normalize_text(text: str) -> str:
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "\u0000": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def paper_context(row: dict, parsed: dict, max_chars: int) -> str:
    sections = parsed.get("sections", {})
    selected = []
    for name in ("preamble", "abstract", "introduction", "methodology", "experiments", "conclusion", "appendix"):
        value = sections.get(name, "")
        if value:
            selected.append(f"\n[{name.upper()}]\n{normalize_text(value)}")
    text = "\n".join(selected)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[TRUNCATED]"
    metadata = {
        "paper_id": row["id_short"],
        "title": row["title"],
        "domain_hint": row.get("domain", ""),
        "abs_url": row.get("abs_url", ""),
        "pdf_url": row.get("pdf_url", ""),
    }
    return "METADATA:\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n\nTEXT:\n" + text


def arxiv_id_from_url(url: str) -> str:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s/]+?)(?:\.pdf)?$", url)
    return match.group(1) if match else ""


def focused_snippets(parsed: dict) -> str:
    sections = parsed.get("sections", {})
    text_by_section = {
        name: normalize_text(value)
        for name, value in sections.items()
        if name != "references" and value
    }
    patterns = {
        "venue": [
            r"published as a conference paper.{0,120}",
            r"\b(ICLR|IJCAI|AAAI|KDD|CIKM|WWW|The Web Conference|Neural Networks|IEEE Transactions|PVLDB)\b.{0,160}",
        ],
        "code": [
            r"source code.{0,180}",
            r"code (?:is|are|will be).{0,180}",
            r"https?://github\.com/[^\s,;)]+",
        ],
        "dataset": [
            r"we (?:conduct|evaluate|verify).{0,240}datasets?.{0,240}",
            r"datasets? (?:used|adopted|contains?|consists?).{0,260}",
            r"\b(METR-LA|PEMS-BAY|PeMSD\d|PeMSD7|PeMSD8|Los-loop|Seattle-Loop|NE-BJ|Xiamen|Tyson).{0,260}",
        ],
        "horizon": [
            r"(?:15|30|45|60)[ -]?(?:minutes?|min).{0,160}",
            r"\b1 hour.{0,160}",
            r"next \d+ (?:steps?|time steps|hours?|days?|minutes?).{0,160}",
            r"forecast(?:ing)? horizon.{0,180}",
            r"Q\s*=\s*\d+.{0,120}",
        ],
        "frequency": [
            r"\d+[- ]?(?:seconds?|minutes?|min|hours?).{0,120}(?:interval|window|resolution|record|aggregate|aggregated|granularity)",
            r"aggregated into \d+[- ]?(?:minutes?|min).{0,120}",
            r"every \d+ (?:seconds?|minutes?|hours?|days?)",
            r"daily reports?.{0,120}",
        ],
        "nodes": [
            r"\d{2,5}\s+(?:nodes|sensors|stations|counties|segments|roads|road segments|links|mile-segments).{0,180}",
            r"# Nodes.{0,220}",
            r"Number of nodes.{0,220}",
        ],
        "target": [
            r"predict.{0,180}(?:traffic speed|traffic flow|travel time|AQI|PV|cases|bandwidth|speed and flow)",
            r"(?:traffic speed|traffic flow|travel time|AQI|PV power|daily new cases|bandwidth).{0,180}",
        ],
    }
    blocks = []
    for label, pats in patterns.items():
        hits = []
        for section_name, text in text_by_section.items():
            for pattern in pats:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    snippet = text[max(0, match.start() - 80): match.end() + 120]
                    snippet = re.sub(r"\s+", " ", snippet).strip()
                    key = snippet.lower()
                    if key not in {h.lower() for h in hits}:
                        hits.append(f"[{section_name}] {snippet}")
                    if len(hits) >= 8:
                        break
                if len(hits) >= 8:
                    break
            if len(hits) >= 8:
                break
        if hits:
            blocks.append(f"{label.upper()} SNIPPETS:\n" + "\n".join(f"- {hit}" for hit in hits))
    return "\n\n".join(blocks)


EVIDENCE_PATTERNS = {
    "venue": [
        r"published as a conference paper.{0,180}",
        r"\b(ICLR|IJCAI|AAAI|KDD|CIKM|WWW|The Web Conference|Neural Networks|IEEE Transactions|PVLDB|VLDB|SIGSPATIAL|CIKM|ICASSP|ICDM|WSDM|TKDD|TITS|TNNLS)\b.{0,220}",
        r"arXiv:\d{4}\.\d+.{0,120}",
    ],
    "code": [
        r"(?:source )?code(?:base)? (?:is|are|will be|has been|can be|was).{0,240}",
        r"(?:publicly )?available at.{0,220}",
        r"implementation.{0,180}(?:github|available|code)",
        r"https?://github\.com/[^\s,;)]+",
        r"https?://[^\s,;)]+(?:code|github|repository|repo)[^\s,;)]*",
    ],
    "dataset": [
        r"we (?:conduct|evaluate|verify|test|train).{0,320}datasets?.{0,320}",
        r"datasets? (?:used|adopted|contains?|consists?|include|includes|are|were).{0,360}",
        r"(?:METR[- ]?LA|PEMS[- ]?BAY|PeMSD\d|PeMSD7|PeMSD8|Los[- ]?loop|Seattle[- ]?Loop|NE[- ]?BJ|Xiamen|Tyson|ERA5|MERRA[- ]?2|ChaosNetBench|Electricity|Exchange Rate|Solar|COVID|NYT|Google Mobility).{0,320}",
        r"(?:Table|Tab\.)\s*\d+.{0,260}(?:dataset|nodes|sensors|stations|frequency)",
    ],
    "horizon": [
        r"(?:15|30|45|60)[ -]?(?:minutes?|min).{0,220}",
        r"\b1 hour.{0,220}",
        r"(?:next|future|predict(?:ing)?).{0,140}\d+ (?:steps?|time steps|hours?|days?|weeks?|minutes?|min).{0,160}",
        r"forecast(?:ing)? horizon.{0,260}",
        r"prediction horizon.{0,260}",
        r"\bQ\s*=\s*\d+.{0,160}",
        r"horizon\s*(?:=|:).{0,180}",
    ],
    "frequency": [
        r"\d+[- ]?(?:seconds?|minutes?|min|hours?|days?).{0,180}(?:interval|window|resolution|record|aggregate|aggregated|granularity|sampling|sampled)",
        r"(?:interval|resolution|granularity|sampling rate|sampled every|recorded every).{0,180}\d+[- ]?(?:seconds?|minutes?|min|hours?|days?)",
        r"aggregated into \d+[- ]?(?:minutes?|min|hours?).{0,160}",
        r"every \d+ (?:seconds?|minutes?|hours?|days?)",
        r"daily reports?.{0,160}",
    ],
    "nodes": [
        r"\d{2,6}\s+(?:nodes|sensors|stations|counties|segments|roads|road segments|links|mile-segments|variables|regions|states|PV systems).{0,240}",
        r"(?:# Nodes|Number of nodes|nodes|sensors|stations).{0,260}\d{2,6}",
        r"(?:graph|network).{0,200}(?:nodes|sensors|stations|links|segments).{0,200}",
    ],
    "target": [
        r"predict.{0,240}(?:traffic speed|traffic flow|traffic volume|travel time|AQI|air quality|PV|power|cases|bandwidth|speed and flow|multivariate time series)",
        r"(?:traffic speed|traffic flow|traffic volume|travel time|AQI|PM2\.5|PV power|daily new cases|bandwidth|multivariate time series).{0,240}",
        r"(?:task|problem).{0,220}(?:forecasting|prediction|classification)",
    ],
}


GROUP_EVIDENCE_LABELS = {
    "bibliography_access": ["venue", "code", "dataset"],
    "task_target": ["target", "dataset", "nodes"],
    "data_temporal": ["dataset", "horizon", "frequency", "nodes"],
    "graph_quality": ["nodes", "dataset", "target"],
}


GROUP_SECTION_FALLBACKS = {
    "bibliography_access": ["preamble", "abstract", "experiments", "conclusion"],
    "task_target": ["abstract", "introduction", "methodology", "experiments"],
    "data_temporal": ["experiments", "methodology", "appendix", "abstract"],
    "graph_quality": ["experiments", "methodology", "appendix"],
}


def collect_evidence_snippets(parsed: dict, labels: list[str], max_hits_per_label: int = 12) -> str:
    sections = parsed.get("sections", {})
    text_by_section = {
        name: normalize_text(value)
        for name, value in sections.items()
        if name != "references" and value
    }
    blocks = []
    for label in labels:
        hits = []
        seen = set()
        for section_name, text in text_by_section.items():
            for pattern in EVIDENCE_PATTERNS[label]:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    snippet = text[max(0, match.start() - 120): match.end() + 180]
                    snippet = re.sub(r"\s+", " ", snippet).strip()
                    key = snippet.lower()
                    if key not in seen:
                        seen.add(key)
                        hits.append(f"[{section_name}] {snippet}")
                    if len(hits) >= max_hits_per_label:
                        break
                if len(hits) >= max_hits_per_label:
                    break
            if len(hits) >= max_hits_per_label:
                break
        if hits:
            blocks.append(f"{label.upper()} EVIDENCE:\n" + "\n".join(f"- {hit}" for hit in hits))
    return "\n\n".join(blocks)


def section_fallback_text(parsed: dict, section_names: list[str], max_chars: int) -> str:
    sections = parsed.get("sections", {})
    blocks = []
    remaining = max_chars
    for name in section_names:
        value = sections.get(name, "")
        if not value or remaining <= 0:
            continue
        text = normalize_text(value)
        excerpt = text[: min(len(text), remaining)]
        blocks.append(f"\n[{name.upper()}]\n{excerpt}")
        remaining -= len(excerpt)
    return "\n".join(blocks)


def captions_text(parsed: dict) -> str:
    blocks = []
    for key, label in (("table_captions", "TABLE CAPTIONS"), ("figure_captions", "FIGURE CAPTIONS")):
        captions = parsed.get(key) or []
        if captions:
            lines = []
            for item in captions[:12]:
                caption = normalize_text(item.get("caption", ""))
                if caption:
                    lines.append(f"- {caption}")
            if lines:
                blocks.append(label + ":\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def localized_group_context(
    row: dict,
    parsed: dict,
    arxiv_cache: dict,
    group_name: str,
    max_chars: int,
) -> str:
    arxiv_id = arxiv_id_from_url(row.get("abs_url", ""))
    arxiv_meta = arxiv_cache.get(arxiv_id, {})
    metadata = {
        "paper_id": row["id_short"],
        "title": row["title"],
        "domain_hint": row.get("domain", ""),
        "abs_url": row.get("abs_url", ""),
        "pdf_url": row.get("pdf_url", ""),
        "arxiv_metadata": arxiv_meta,
        "parse_quality": parsed.get("parse_quality", ""),
    }
    evidence = collect_evidence_snippets(parsed, GROUP_EVIDENCE_LABELS[group_name])
    captions = captions_text(parsed)
    fallback_budget = max(6000, max_chars // 2)
    fallback = section_fallback_text(parsed, GROUP_SECTION_FALLBACKS[group_name], fallback_budget)
    if len(evidence) < 1200:
        fallback += "\n\n[FALLBACK BROAD CONTEXT]\n" + section_fallback_text(
            parsed,
            ["preamble", "abstract", "introduction", "methodology", "experiments", "appendix"],
            max_chars,
        )
    context = (
        "STRUCTURED METADATA:\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n\nLOCALIZED FIELD EVIDENCE:\n"
        + (evidence or "No high-sensitivity evidence snippets were found.")
        + "\n\n"
        + captions
        + "\n\nRELEVANT SECTION FALLBACK:\n"
        + fallback
    )
    return context[:max_chars] + ("\n[TRUNCATED]" if len(context) > max_chars else "")


def localized_contexts(row: dict, parsed: dict, arxiv_cache: dict, max_chars: int) -> dict[str, str]:
    return {
        group_name: localized_group_context(row, parsed, arxiv_cache, group_name, max_chars)
        for group_name in FIELD_GROUPS
    }


def paper_context_v1(row: dict, parsed: dict, arxiv_cache: dict, max_chars: int) -> str:
    arxiv_id = arxiv_id_from_url(row.get("abs_url", ""))
    arxiv_meta = arxiv_cache.get(arxiv_id, {})
    sections = parsed.get("sections", {})
    metadata = {
        "paper_id": row["id_short"],
        "title": row["title"],
        "domain_hint": row.get("domain", ""),
        "abs_url": row.get("abs_url", ""),
        "pdf_url": row.get("pdf_url", ""),
        "arxiv_metadata": arxiv_meta,
        "parse_quality": parsed.get("parse_quality", ""),
    }
    section_text = []
    for name in ("preamble", "abstract", "experiments", "methodology", "appendix", "conclusion"):
        value = sections.get(name, "")
        if value:
            value = normalize_text(value)
            section_text.append(f"\n[{name.upper()}]\n{value[:9000]}")
    body = "\n".join(section_text)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[TRUNCATED]"
    return (
        "STRUCTURED METADATA:\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n\nFIELD-FOCUSED EVIDENCE SNIPPETS:\n"
        + focused_snippets(parsed)
        + "\n\nSECTION TEXT:\n"
        + body
    )


def ollama_generate(host: str, model: str, prompt: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": 32768,
        },
    }
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


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_row(row: dict) -> list[str]:
    errors = []
    for field in FIELDNAMES:
        if field not in row:
            errors.append(f"missing:{field}")
    for field, allowed in ENUMS.items():
        value = str(row.get(field, ""))
        if value not in allowed:
            errors.append(f"invalid_enum:{field}={value}")
    for field in ("year", "horizon_min_value", "horizon_max_value", "frequency_value", "graph_node_count_min", "graph_node_count_max"):
        value = str(row.get(field, ""))
        if value not in {"not_reported", "ambiguous", "not_applicable", "dataset_specific"} and not value.isdigit():
            errors.append(f"invalid_numeric:{field}={value}")
    return errors


def coerce_output(raw: dict, paper_id: str) -> dict:
    output = {field: str(raw.get(field, "not_reported")).strip() for field in FIELDNAMES}
    output["paper_id"] = paper_id
    return output


def coerce_group_output(raw: dict, paper_id: str, fields: list[str]) -> dict:
    output = {"paper_id": paper_id}
    for field in fields:
        output[field] = str(raw.get(field, "not_reported")).strip()
    return output


def partial_schema(schema: dict, fields: list[str]) -> str:
    required = ["paper_id"] + fields
    properties = {"paper_id": schema["properties"]["paper_id"]}
    for field in fields:
        properties[field] = schema["properties"][field]
    return json.dumps({
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }, ensure_ascii=False, indent=2)


def grouped_prompt(group_name: str, fields: list[str], schema: dict, context: str) -> str:
    dataset_labels = ""
    if group_name == "data_temporal":
        dataset_labels = (
            "\nAllowed dataset_family labels. Use only these labels, separated by `|` for multiple datasets:\n"
            + ", ".join(ALLOWED_DATASET_FAMILIES)
            + "\n"
        )
    return (
        "You are coding scientific papers into a closed-label schema.\n\n"
        "Return exactly one JSON object. Do not include Markdown, comments, explanations, or evidence text.\n"
        "Use only the allowed labels from the schema. If a value is not clearly reported, use `not_reported`.\n"
        "If the paper is unclear or conflicting, use `ambiguous` when that value is allowed.\n"
        "Numeric fields must be strings with digits only, or one of the allowed non-numeric labels.\n"
        "Multi-label dataset fields must use `|` as the separator.\n\n"
        f"FIELD GROUP: {group_name}\n"
        f"{GROUP_INSTRUCTIONS[group_name]}\n\n"
        + dataset_labels
        + "\n"
        "Return only these fields plus paper_id:\n"
        + ", ".join(["paper_id"] + fields)
        + "\n\nJSON SCHEMA:\n"
        + partial_schema(schema, fields)
        + "\n\nPaper metadata and extracted text follow.\n\n"
        + context
    )


def code_grouped(
    paper_id: str,
    context: str | dict[str, str],
    schema: dict,
    host: str,
    model: str,
    timeout: int,
) -> tuple[dict, dict, float]:
    coded = {field: "not_reported" for field in FIELDNAMES}
    coded["paper_id"] = paper_id
    raw_responses = {}
    elapsed_total = 0.0

    for group_name, fields in FIELD_GROUPS.items():
        group_context = context[group_name] if isinstance(context, dict) else context
        prompt = grouped_prompt(group_name, fields, schema, group_context)
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


def normalize_number(value: str) -> str:
    value = str(value).strip().lower()
    if value == "daily":
        return "1"
    if value in {"", "none", "null", "n/a", "na", "unknown", "not found"}:
        return "not_reported"
    if value in {"not_reported", "ambiguous", "not_applicable", "dataset_specific"}:
        return value
    value = value.replace(",", "")
    compact = re.search(r"(\d+(?:\.\d+)?)\s*([km])\b", value)
    if compact:
        multiplier = 1000 if compact.group(2) == "k" else 1000000
        return str(int(float(compact.group(1)) * multiplier))
    match = re.search(r"\d+", value)
    return match.group(0) if match else value


def normalize_enum(value: str, mapping: dict[str, str], mixed_value: str | None = None) -> str:
    raw = str(value).strip()
    if mixed_value and re.search(r"[|,;/]", raw):
        parts = [normalize_enum(part, mapping) for part in re.split(r"[|,;/]+", raw) if part.strip()]
        unique = {part for part in parts if part}
        if len(unique) == 1:
            return next(iter(unique))
        return mixed_value
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return mapping.get(key, str(value).strip())


def normalize_dataset_family(value: str) -> str:
    value = str(value).strip().lower()
    if not value or value in {"none", "null", "unknown"}:
        return "not_reported"
    detector_patterns = [
        ("metr_la", r"\bmetr[-_ ]?la\b"),
        ("pems_bay", r"\bpems[-_ ]?bay\b"),
        ("pemsd3", r"\bpemsd[-_ ]?3\b"),
        ("pemsd4", r"\bpemsd[-_ ]?4\b"),
        ("pemsd7", r"\bpemsd[-_ ]?7|\bpems[-_ ]?d7\b"),
        ("pemsd8", r"\bpemsd[-_ ]?8|\bpems[-_ ]?d8\b"),
        ("bjer4", r"\bbjer4\b|\bbeijing\s+er\b"),
        ("xiamen", r"\bxiamen\b"),
        ("california_highway_pems", r"\bcalifornia\b.*\bpems\b|\bcaltrans\b"),
        ("amap_beijing", r"\bamap\b|\bbeijing\b.*\btaxi\b"),
        ("nyt_covid", r"\bnyt\b|\bnew york times\b"),
        ("google_mobility", r"\bgoogle\b.*\bmobility\b"),
        ("yangtze_air_quality", r"\byangtze\b"),
        ("vevo_music", r"\bvevo\b"),
        ("wiki_traffic", r"\bwiki\b"),
        ("los_loop", r"\blos[-_ ]?loop\b"),
        ("sz_taxi", r"\bsz[-_ ]?taxi\b|\bshenzhen\b"),
        ("solar_energy", r"\bsolar\b"),
        ("electricity", r"\belectricity\b"),
        ("exchange_rate", r"\bexchange\b"),
        ("ne_bj", r"\bne[-_ ]?bj\b"),
        ("swiss_pv_real", r"\bswiss\b.*\breal\b"),
        ("swiss_pv_synthetic", r"\bswiss\b.*\bsynthetic\b"),
        ("tysons_corner", r"\btysons?\b"),
        ("seattle_loop", r"\bseattle[-_ ]?loop\b"),
    ]
    detected = []
    for label in ALLOWED_DATASET_FAMILIES:
        if label != "not_reported" and label in value and label not in detected:
            detected.append(label)
    for label, pattern in detector_patterns:
        if re.search(pattern, value) and label not in detected:
            detected.append(label)
    if detected:
        return "|".join(detected)
    replacements = {
        "metr-la": "metr_la",
        "metr la": "metr_la",
        "pems-bay": "pems_bay",
        "pems bay": "pems_bay",
        "pemsd-7": "pemsd7",
        "pemsd 7": "pemsd7",
        "pemsd-8": "pemsd8",
        "pemsd 8": "pemsd8",
        "los loop": "los_loop",
        "los-loop": "los_loop",
        "ne-bj": "ne_bj",
        "ne bj": "ne_bj",
    }
    parts = re.split(r"[|,;/]+", value)
    normalized = []
    for part in parts:
        part = part.strip()
        part = replacements.get(part, part.replace("-", "_").replace(" ", "_"))
        if part and part not in normalized:
            normalized.append(part)
    return "|".join(normalized) if normalized else "not_reported"


def normalize_coded_row(row: dict) -> dict:
    row = dict(row)
    horizon_unit_mapping = {
        "minute": "minutes",
        "min": "minutes",
        "mins": "minutes",
        "hour": "hours",
        "hr": "hours",
        "hrs": "hours",
        "day": "days",
        "week": "weeks",
        "weeks": "weeks",
        "sec": "seconds",
        "second": "seconds",
        "time_step": "steps",
        "time_steps": "steps",
        "step": "steps",
    }
    frequency_unit_mapping = {
        **horizon_unit_mapping,
        "week": "days",
        "weeks": "days",
    }
    node_mapping = {
        "sensor": "sensors",
        "sensor_station": "stations",
        "sensor_stations": "stations",
        "weather_station": "stations",
        "weather_stations": "stations",
        "station": "stations",
        "stations": "stations",
        "segment": "road_segments",
        "segments": "road_segments",
        "road_segment": "road_segments",
        "road_segments": "road_segments",
        "mile_segment": "road_segments",
        "mile_segments": "road_segments",
        "road": "road_segments",
        "roads": "road_segments",
        "county": "counties",
        "counties": "counties",
        "link": "links",
        "links": "links",
        "node": "nodes",
        "nodes": "nodes",
        "variable": "nodes",
        "variables": "nodes",
        "region": "nodes",
        "regions": "nodes",
        "state": "nodes",
        "states": "nodes",
        "pv_system": "nodes",
        "pv_systems": "nodes",
        "wind_turbine": "nodes",
        "wind_turbines": "nodes",
    }
    venue_mapping = {
        "workshop": "competition_workshop",
        "competition": "competition_workshop",
        "conference_paper": "conference",
        "journal_article": "journal",
        "arxiv": "preprint",
        "arxiv_preprint": "preprint",
    }
    graph_status_mapping = {
        "found": "found",
        "complete": "found",
        "completed": "found",
        "partial": "partial",
        "partially_found": "partial",
        "ambiguous": "ambiguous",
        "not_found": "not_reported",
        "missing": "not_reported",
        "unknown": "not_reported",
    }
    coding_status_mapping = {
        "complete": "complete",
        "completed": "complete",
        "found": "complete",
        "high": "complete",
        "partial": "partial",
        "partially_found": "partial",
        "not_found": "not_reported",
        "missing": "not_reported",
        "unknown": "not_reported",
    }

    row["dataset_family"] = normalize_dataset_family(row.get("dataset_family", ""))
    row["venue_type"] = normalize_enum(row.get("venue_type", ""), venue_mapping)
    row["horizon_unit"] = normalize_enum(row.get("horizon_unit", ""), horizon_unit_mapping, mixed_value="mixed")
    row["frequency_unit"] = normalize_enum(row.get("frequency_unit", ""), frequency_unit_mapping, mixed_value="mixed")
    row["node_unit"] = normalize_enum(row.get("node_unit", ""), node_mapping, mixed_value="mixed")
    row["graph_node_count_status"] = normalize_enum(row.get("graph_node_count_status", ""), graph_status_mapping)
    row["coding_status"] = normalize_enum(row.get("coding_status", ""), coding_status_mapping)

    if row["coding_status"] not in ENUMS["coding_status"]:
        row["coding_status"] = "partial"

    for field in ("year", "horizon_min_value", "horizon_max_value", "frequency_value", "graph_node_count_min", "graph_node_count_max"):
        row[field] = normalize_number(row.get(field, "not_reported"))

    if row.get("frequency_unit") == "steps":
        row["frequency_unit"] = "not_reported"

    for min_field, max_field in (("horizon_min_value", "horizon_max_value"), ("graph_node_count_min", "graph_node_count_max")):
        if str(row.get(min_field, "")).isdigit() and str(row.get(max_field, "")).isdigit():
            low, high = sorted((int(row[min_field]), int(row[max_field])))
            row[min_field], row[max_field] = str(low), str(high)

    for field in ("has_public_code", "has_public_dataset"):
        value = str(row.get(field, "")).strip().lower()
        if value in {"true", "available", "public", "yes"}:
            row[field] = "yes"
        elif value in {"false", "unavailable", "no"}:
            row[field] = "no"
        elif value in {"", "unknown", "not found", "not_found"}:
            row[field] = "not_reported"

    if row.get("confidence") == "not_reported":
        row["confidence"] = "low"

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run papers marked include_in_pilot=yes")
    parser.add_argument("--corpus", default=str(CORPUS_CSV))
    parser.add_argument("--paper-id", action="append", help="Run one or more paper IDs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-name", default=None, help="Override output subdirectory name")
    parser.add_argument(
        "--workflow",
        choices=["w0_direct", "w1_metadata_sections", "w2_schema_normalized", "w3_field_groups", "w4_evidence_localized"],
        default="w0_direct",
    )
    parser.add_argument("--max-chars", type=int, default=28000)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    corpus = read_corpus(Path(args.corpus))
    arxiv_cache = read_arxiv_cache()

    if args.paper_id:
        wanted = set(args.paper_id)
        papers = [row for row in corpus if row["id_short"] in wanted]
    elif args.pilot:
        papers = [row for row in corpus if row.get("include_in_pilot", "").lower() == "yes"]
    else:
        papers = corpus

    if args.limit is not None:
        papers = papers[: args.limit]

    out_dir = Path(args.output_dir) / (args.run_name or args.workflow)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, paper in enumerate(papers, start=1):
        paper_id = paper["id_short"]
        parsed_path = PARSED_DIR / f"{paper_id}.json"
        if not parsed_path.exists():
            print(f"[{index}/{len(papers)}] MISSING parsed text: {paper_id}", file=sys.stderr)
            continue

        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        if args.workflow == "w4_evidence_localized":
            context = localized_contexts(paper, parsed, arxiv_cache, args.max_chars)
        elif args.workflow in {"w1_metadata_sections", "w2_schema_normalized", "w3_field_groups"}:
            context = paper_context_v1(paper, parsed, arxiv_cache, args.max_chars)
        else:
            context = paper_context(paper, parsed, args.max_chars)

        prompt = ""
        if not isinstance(context, dict):
            prompt = (
                prompt_template
                + "\n\nJSON SCHEMA:\n"
                + schema_text
                + "\n\n"
                + context
            )

        print(f"[{index}/{len(papers)}] coding {paper_id} with {args.model} workflow={args.workflow}")
        started = time.time()
        try:
            if args.workflow in {"w3_field_groups", "w4_evidence_localized"}:
                coded, response, grouped_elapsed = code_grouped(
                    paper_id, context, schema, args.ollama_host, args.model, args.timeout
                )
                elapsed_override = grouped_elapsed
            else:
                response = ollama_generate(args.ollama_host, args.model, prompt, args.timeout)
                raw = extract_json(response)
                coded = coerce_output(raw, paper_id)
                elapsed_override = None
            if args.workflow in {"w2_schema_normalized", "w3_field_groups", "w4_evidence_localized"}:
                coded = normalize_coded_row(coded)
            errors = validate_row(coded)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            coded = {field: "not_reported" for field in FIELDNAMES}
            coded["paper_id"] = paper_id
            errors = [f"runtime_error:{type(exc).__name__}:{exc}"]
            response = ""
            elapsed_override = None

        elapsed = round(elapsed_override if elapsed_override is not None else time.time() - started, 2)
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
        print(f"  saved {paper_id}.json errors={len(errors)} elapsed={elapsed}s")

    csv_path = out_dir / "codificacion_1_ollama.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

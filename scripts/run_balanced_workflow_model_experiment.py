"""
Run a balanced workflow x model experiment for coder-1 extraction.

The experiment is deliberately separated from the exploratory ablation scripts:
it creates consistent output directories, reuses compatible completed runs, runs
only missing paper-level jobs, and evaluates every completed cell against
gold_v5 with dataset aliases enabled.

Design:
  models    = qwen2.5-coder:7b, qwen3-coder:30b, command-r:35b,
              gemma3:12b, mistral-nemo:12b, llama3.1:8b
  workflows = direct, schema, field_groups, hybrid, guided, guided_access,
              cot, audit

Derived workflows:
  hybrid        = deterministic field-family merge of schema + field_groups
  guided_access = conservative access-rule postprocess of guided
  audit         = second-pass audit over guided_access
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_BASE = ROOT / "data" / "model_outputs" / "codificacion_1_ollama"
GOLD_V5 = ROOT / "data" / "annotations" / "codificacion_1_manual_gold_v5.csv"
SUMMARY_DIR = ROOT / "data" / "analysis" / "balanced_workflow_model"
CORPUS = ROOT / "corpus_inicial_50_papers_stgl.csv"


MODELS = [
    "qwen2.5-coder:7b",
    "qwen3-coder:30b",
    "command-r:35b",
    "gemma3:12b",
    "mistral-nemo:12b",
    "llama3.1:8b",
]

WORKFLOWS = [
    "direct",
    "schema",
    "field_groups",
    "hybrid",
    "guided",
    "guided_access",
    "cot",
    "audit",
]

LLM_WORKFLOW_MAP = {
    "direct": ("scripts/run_codificacion_1_ollama.py", "w0_direct"),
    "schema": ("scripts/run_codificacion_1_ollama.py", "w2_schema_normalized"),
    "field_groups": ("scripts/run_codificacion_1_ollama.py", "w3_field_groups"),
    "guided": ("scripts/run_new_codificacion_workflows.py", "w10_guideline_fewshot"),
    "cot": ("scripts/run_new_codificacion_workflows.py", "w11_cot_grouped"),
}

DERIVED_WORKFLOWS = {"hybrid", "guided_access", "audit"}

LEGACY_SEEDS = {
    ("qwen2.5-coder:7b", "direct"): "w0_direct_50",
    ("qwen2.5-coder:7b", "schema"): "w2_schema_normalized_50",
    ("qwen2.5-coder:7b", "field_groups"): "w3_field_groups_50",
    ("qwen2.5-coder:7b", "hybrid"): "w4_hybrid_w2_w3_50",
    ("qwen2.5-coder:7b", "guided"): "w10_guideline_fewshot_50",
    ("qwen2.5-coder:7b", "cot"): "w11_cot_grouped",
    ("qwen3-coder:30b", "guided"): "w10_guideline_fewshot_qwen3_coder",
    ("qwen3-coder:30b", "guided_access"): "w10_schema_v4_access",
    ("gemma3:12b", "guided"): "w10_guideline_fewshot_gemma3_12b",
    ("gemma3:12b", "guided_access"): "w10_schema_v4_access_gemma3_12b",
    ("mistral-nemo:12b", "guided"): "w10_guideline_fewshot_mistral_nemo_12b",
    ("mistral-nemo:12b", "guided_access"): "w10_schema_v4_access_mistral_nemo_12b",
    ("llama3.1:8b", "guided"): "w10_guideline_fewshot_llama3_1_8b",
    ("llama3.1:8b", "guided_access"): "w10_schema_v4_access_llama3_1_8b",
}


@dataclass(frozen=True)
class Cell:
    model: str
    workflow: str

    @property
    def model_slug(self) -> str:
        return (
            self.model.replace(":", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    @property
    def run_name(self) -> str:
        return f"balanced__{self.model_slug}__{self.workflow}"

    @property
    def out_dir(self) -> Path:
        return OUT_BASE / self.run_name

    @property
    def csv_path(self) -> Path:
        return self.out_dir / "codificacion_1_ollama.csv"


def run(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    if dry_run:
        print(f"DRY-RUN {printable}")
        return
    print(f"RUN {printable}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def pilot_ids() -> list[str]:
    with open(CORPUS, newline="", encoding="utf-8-sig") as handle:
        return [
            row["id_short"]
            for row in csv.DictReader(handle)
            if row.get("include_in_pilot", "").lower() == "yes"
        ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def has_complete_csv(cell: Cell, ids: list[str]) -> bool:
    rows = read_csv_rows(cell.csv_path)
    return {row.get("paper_id", "") for row in rows} >= set(ids)


def existing_json_ids(cell: Cell) -> set[str]:
    if not cell.out_dir.exists():
        return set()
    return {path.stem for path in cell.out_dir.glob("*.json")}


def is_seed_complete(seed: str, ids: list[str]) -> bool:
    seed_dir = OUT_BASE / seed
    rows = read_csv_rows(seed_dir / "codificacion_1_ollama.csv")
    if {row.get("paper_id", "") for row in rows} >= set(ids):
        return True
    return {path.stem for path in seed_dir.glob("*.json")} >= set(ids)


def seed_cell(cell: Cell, ids: list[str], dry_run: bool = False) -> bool:
    seed = LEGACY_SEEDS.get((cell.model, cell.workflow))
    if not seed or not is_seed_complete(seed, ids):
        return False
    src = OUT_BASE / seed
    if dry_run:
        print(f"DRY-RUN seed {cell.run_name} from {seed}")
        return True
    if cell.out_dir.exists():
        shutil.rmtree(cell.out_dir)
    shutil.copytree(src, cell.out_dir)
    (cell.out_dir / "balanced_seed.json").write_text(
        json.dumps(
            {
                "source_workflow": seed,
                "model": cell.model,
                "workflow": cell.workflow,
                "run_name": cell.run_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SEEDED {cell.run_name} from {seed}", flush=True)
    return True


def assemble_from_json(cell: Cell, dry_run: bool = False) -> None:
    run(
        [
            sys.executable,
            "scripts/assemble_codificacion_output.py",
            "--workflow",
            cell.run_name,
        ],
        dry_run=dry_run,
    )


def evaluate(cell: Cell, dry_run: bool = False) -> None:
    run(
        [
            sys.executable,
            "scripts/evaluate_codificacion_1.py",
            "--gold",
            str(GOLD_V5),
            "--pred",
            str(cell.csv_path),
            "--out",
            str(cell.out_dir / "evaluation_report_gold_v5_alias.csv"),
            "--dataset-aliases",
        ],
        dry_run=dry_run,
    )


def run_llm_cell(cell: Cell, ids: list[str], timeout: int, dry_run: bool = False) -> None:
    script, workflow = LLM_WORKFLOW_MAP[cell.workflow]
    done = existing_json_ids(cell)
    missing = [paper_id for paper_id in ids if paper_id not in done]
    print(f"CELL {cell.run_name}: {len(ids) - len(missing)}/{len(ids)} JSON complete", flush=True)
    for paper_id in missing:
        run(
            [
                sys.executable,
                script,
                "--paper-id",
                paper_id,
                "--model",
                cell.model,
                "--workflow",
                workflow,
                "--run-name",
                cell.run_name,
                "--timeout",
                str(timeout),
            ],
            dry_run=dry_run,
        )
    if missing or not cell.csv_path.exists():
        assemble_from_json(cell, dry_run=dry_run)


def build_hybrid(cell: Cell, dry_run: bool = False) -> None:
    schema = Cell(cell.model, "schema")
    groups = Cell(cell.model, "field_groups")
    run(
        [
            sys.executable,
            "scripts/build_hybrid_codificacion_workflow.py",
            "--base-workflow",
            schema.run_name,
            "--field-workflow",
            groups.run_name,
            "--out-workflow",
            cell.run_name,
        ],
        dry_run=dry_run,
    )


def build_guided_access(cell: Cell, dry_run: bool = False) -> None:
    guided = Cell(cell.model, "guided")
    run(
        [
            sys.executable,
            "scripts/postprocess_access_rules.py",
            "--pred",
            str(guided.csv_path),
            "--out-dir",
            str(cell.out_dir),
        ],
        dry_run=dry_run,
    )


def run_audit(cell: Cell, ids: list[str], timeout: int, dry_run: bool = False) -> None:
    base = Cell(cell.model, "guided_access")
    done = existing_json_ids(cell)
    missing = [paper_id for paper_id in ids if paper_id not in done]
    print(f"CELL {cell.run_name}: {len(ids) - len(missing)}/{len(ids)} JSON complete", flush=True)
    for paper_id in missing:
        run(
            [
                sys.executable,
                "scripts/run_advanced_codificacion_workflows.py",
                "--workflow",
                "w6_second_pass_audit",
                "--paper-id",
                paper_id,
                "--model",
                cell.model,
                "--run-name",
                cell.run_name,
                "--input-csv",
                str(base.csv_path),
                "--timeout",
                str(timeout),
            ],
            dry_run=dry_run,
        )
    if missing or not cell.csv_path.exists():
        assemble_from_json(cell, dry_run=dry_run)


def ensure_cell(cell: Cell, ids: list[str], timeout: int, dry_run: bool = False) -> None:
    if has_complete_csv(cell, ids):
        print(f"SKIP {cell.run_name}: complete CSV exists", flush=True)
        evaluate(cell, dry_run=dry_run)
        return
    if seed_cell(cell, ids, dry_run=dry_run):
        evaluate(cell, dry_run=dry_run)
        return
    if cell.workflow in LLM_WORKFLOW_MAP:
        run_llm_cell(cell, ids, timeout, dry_run=dry_run)
    elif cell.workflow == "hybrid":
        ensure_cell(Cell(cell.model, "schema"), ids, timeout, dry_run=dry_run)
        ensure_cell(Cell(cell.model, "field_groups"), ids, timeout, dry_run=dry_run)
        build_hybrid(cell, dry_run=dry_run)
    elif cell.workflow == "guided_access":
        ensure_cell(Cell(cell.model, "guided"), ids, timeout, dry_run=dry_run)
        build_guided_access(cell, dry_run=dry_run)
    elif cell.workflow == "audit":
        ensure_cell(Cell(cell.model, "guided_access"), ids, timeout, dry_run=dry_run)
        run_audit(cell, ids, timeout, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown workflow: {cell.workflow}")
    evaluate(cell, dry_run=dry_run)


def summarize(ids: list[str], models: list[str], workflows: list[str], dry_run: bool = False) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        for workflow in workflows:
            cell = Cell(model, workflow)
            report = cell.out_dir / "evaluation_report_gold_v5_alias.csv"
            compared = [
                row for row in read_csv_rows(report)
                if row.get("field") not in {"", "__paper__"}
            ]
            matches = sum(1 for row in compared if row.get("match") == "yes")
            rows.append(
                {
                    "model": model,
                    "workflow": workflow,
                    "run_name": cell.run_name,
                    "complete_csv": "yes" if has_complete_csv(cell, ids) else "no",
                    "n_fields": str(len(compared)),
                    "matches": str(matches),
                    "accuracy": f"{matches / len(compared):.3f}" if compared else "",
                }
            )
    out = SUMMARY_DIR / "balanced_workflow_model_summary.csv"
    if dry_run:
        print(f"DRY-RUN write {out}")
        for row in rows:
            print(row)
        return
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "workflow", "run_name", "complete_csv", "n_fields", "matches", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--workflow", action="append", choices=WORKFLOWS)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ids = pilot_ids()
    models = args.model or MODELS
    workflows = args.workflow or WORKFLOWS
    print(f"Balanced experiment: {len(models)} models x {len(workflows)} workflows x {len(ids)} papers")

    for model in models:
        for workflow in workflows:
            ensure_cell(Cell(model, workflow), ids, args.timeout, dry_run=args.dry_run)
    summarize(ids, models, workflows, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

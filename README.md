# Local LLM Workflow Engineering for Scientific Evidence Coding

This repository contains the minimal reproducibility package for a pilot study on local LLM-assisted structured coding of scientific papers in spatio-temporal graph learning.

The package is intentionally scoped to the 50-paper expert-coded pilot used in the study. It excludes the manuscript source/PDF, discarded validation material, LLM-as-second-coder diagnostics, source PDFs, parsed full texts, and any files that should not be treated as human validation evidence.

## Contents

- `data/gold/`: expert-coded 50-paper pilot in CSV and JSON.
- `data/corpus/`: metadata for the 50-paper pilot corpus.
- `data/analysis/pilot_workflow_ablation/`: analysis outputs, bootstrap deltas, field-group results, model-family sensitivity results, residual error taxonomy, triage simulation, and figure-ready data.
- `data/analysis/balanced_workflow_model/`: balanced six-model by eight-workflow summary tables for the 18-field primary metric and secondary metrics.
- `workflows/n8n/`: n8n workflow exports for local Ollama-based coding workflows.
- `schemas/`: coding schema.
- `prompts/`: prompt materials.
- `scripts/`: scripts used to run workflows, post-process outputs, evaluate predictions, analyze the pilot, and plot figures.
- `docs/`: codebook notes and selected workflow reports.

## Main Data

The primary gold file is:

```text
data/gold/codificacion_1_manual_gold_v5.csv
```

A JSON version for programmatic use is also provided:

```text
data/gold/codificacion_1_manual_gold_v5.json
```

The primary metric in the manuscript is alias-aware exact field accuracy over 18 objective fields, excluding `coding_status` and `confidence`.

## Reproduce The Article Analysis

From the repository root:

```bash
python scripts/analyze_pilot_workflow_ablation.py
python scripts/plot_pilot_workflow_figures.py
python scripts/plot_central_results_figure.py
```

The scripts expect the local project layout used in this package and write outputs under:

```text
data/analysis/pilot_workflow_ablation/
```

The balanced model--workflow experiment is summarized under:

```text
data/analysis/balanced_workflow_model/
```

## n8n Workflows

The n8n JSON files are in:

```text
workflows/n8n/
```

They assume a local Python environment and local Ollama models. Commands may need path/model adjustments depending on the host machine.

## Interpretation

This is a workflow-development pilot, not an independent benchmark. The reported system is `w10_schema_v4_access`; temporal-rule workflows are retained as in-sample diagnostics, not as the deployed system. The package does not report a second-human agreement statistic or a human performance ceiling.
Additional workflow motifs (`w1`, `w5`--`w9`, and `w11`) are included as complete 50-paper runs and should be interpreted as single implementations rather than optimized workflow families. The manuscript also reports a balanced six-model by eight-workflow experiment using `qwen2.5-coder:7b`, `qwen3-coder:30b`, `command-r:35b`, `gemma3:12b`, `mistral-nemo:12b`, and `llama3.1:8b`. This experiment supports model-specific workflow validation; it is not a definitive model benchmark.
The analysis outputs include the figure-ready tables and generated figures needed to audit the reported workflow ladder, balanced model--workflow experiment, field-group reliability, and residual error taxonomy. The manuscript itself is intentionally not included in this repository.

## License

No license is assigned yet. Add a license before making the repository public.

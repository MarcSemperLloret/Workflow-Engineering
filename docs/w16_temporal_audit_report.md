# W16 Temporal Audit Report

Objective: test whether a conservative second LLM pass can correct residual
temporal errors after `gold_v5`.

## Artifacts

```text
scripts/run_w16_temporal_audit.py
data/model_outputs/codificacion_1_ollama/w16_temporal_audit_qwen3_coder/codificacion_1_ollama.csv
data/model_outputs/codificacion_1_ollama/w16_temporal_audit_qwen3_coder/evaluation_report_gold_v5_exact.csv
data/model_outputs/codificacion_1_ollama/w16_temporal_audit_qwen3_coder/evaluation_report_gold_v5_alias.csv
data/model_outputs/codificacion_1_ollama/w16_temporal_audit_qwen3_coder/temporal_disagreements_gold_v5.csv
```

## Design

The workflow uses `temporal_disagreements_gold_v5.csv` only to select the 24
papers and temporal fields that need auditing. Gold values are not shown to the
model. The auditor receives:

- current temporal field values from `w10_schema_v4_access`;
- field-specific temporal rules from the codebook;
- retrieved horizon/frequency evidence snippets from parsed paper text.

The prompt is conservative: keep the current value unless evidence strongly
supports a change.

## Run

Model: `qwen3-coder:30b`

Papers audited: 24

Validation errors: 0

Total elapsed model time recorded in JSON files: 209.76 seconds

Changed fields: 3

```text
ASeer horizon_min_value: 3600 -> 1
ASeer horizon_max_value: 3600 -> 48
STGNPP horizon_type: sequence -> single_horizon
```

## Evaluation Against Gold V5

| run | exact accuracy | alias-aware accuracy | temporal disagreements |
|---|---:|---:|---:|
| w10_schema_v4_access | 0.800 | 0.803 | 56 |
| w16_temporal_audit_qwen3_coder | 0.800 | 0.803 | 56 |

The three changed fields remained mismatches:

```text
ASeer horizon_min_value: gold=60, base=3600, w16=1
ASeer horizon_max_value: gold=60, base=3600, w16=48
STGNPP horizon_type: gold=dataset_specific, base=sequence, w16=single_horizon
```

## Interpretation

`w16` is a useful negative result. A conservative temporal second pass does not
improve the current best workflow. The auditor mostly preserves the base output,
and when it changes values it still fails to resolve the underlying temporal
interpretation.

The failure mode is evidence quality and task framing, not schema validation.
The retrieved snippets often contain relevant words but not the exact
experimental sentence/table needed to decide:

- main output horizon vs prediction step size;
- event-process horizon vs fixed forecasting sequence;
- multi-dataset frequency vs one dataset's sampling interval.

## Recommendation

Do not use `w16_temporal_audit_qwen3_coder` as the main workflow. Keep
`w10_schema_v4_access` as the best current variant against `gold_v5`.

The next temporal experiment should improve evidence retrieval before another
LLM pass. In particular, it should retrieve compact windows around exact
patterns such as `tau_out`, `Q=`, `F=`, `prediction time length`, `forecast
horizon is`, `Horizon 3/6/12`, dataset statistics tables, and table captions
with time intervals.

# W18 Horizon Rules Report

Objective: convert the robust part of `w17` into a non-oracle, deployable
post-processing workflow by applying only horizon rules to all pilot papers.

## Artifacts

```text
data/model_outputs/codificacion_1_ollama/w10_schema_v4_access/temporal_evidence_all_pilot.csv
data/model_outputs/codificacion_1_ollama/w18_horizon_rules_all_pilot/codificacion_1_ollama.csv
data/model_outputs/codificacion_1_ollama/w18_horizon_rules_all_pilot/temporal_rule_changes.csv
data/model_outputs/codificacion_1_ollama/w18_horizon_rules_all_pilot/evaluation_report_gold_v5_exact.csv
data/model_outputs/codificacion_1_ollama/w18_horizon_rules_all_pilot/evaluation_report_gold_v5_alias.csv
data/model_outputs/codificacion_1_ollama/w18_horizon_rules_all_pilot/temporal_disagreements_gold_v5.csv
```

## Design

Unlike `w17_temporal_rules`, `w18_horizon_rules_all_pilot` does not use a
gold-derived disagreement table to select fields. It uses temporal evidence for
all 50 pilot papers and permits only horizon fields to change:

```text
horizon_type
horizon_min_value
horizon_max_value
horizon_unit
```

Frequency fields are intentionally left untouched because the all-pilot stress
test showed frequency rules over-apply.

## Result

Against `gold_v5`, using the manuscript primary metric over 18 objective fields:

| workflow | 20-field accuracy | 18-field objective accuracy |
|---|---:|---:|
| w10_schema_v4_access | 0.803 | 0.821 |
| w18_horizon_rules_all_pilot | 0.836 | 0.858 |

`w18` makes 33 horizon-field corrections relative to `w10_schema_v4_access`
with 0 regressions.

Horizon field accuracy:

| field | accuracy |
|---|---:|
| horizon_type | 50/50 = 1.000 |
| horizon_min_value | 50/50 = 1.000 |
| horizon_max_value | 50/50 = 1.000 |
| horizon_unit | 50/50 = 1.000 |

The remaining focused temporal disagreements are 23 rows, all frequency fields:

```text
frequency_value: 12
frequency_unit: 11
```

## Interpretation

`w18_horizon_rules_all_pilot` demonstrates that horizon errors can be corrected
in-sample by deterministic evidence-aware rules without gold-targeted field
selection. The 1.000 horizon-field accuracy is an internal pilot result, not an
independent generalization estimate.

The remaining temporal limitation is frequency. Frequency needs a safer
non-oracle trigger because multi-dataset papers, raw-vs-aggregated intervals,
and benchmark tables are easy to over-generalize.

## Recommendation

For the manuscript, treat:

- `w10_schema_v4_access` as the strongest LLM-plus-basic-rules baseline.
- `w18_horizon_rules_all_pilot` as the horizon-rule workflow.
- `w16_temporal_audit_qwen3_coder` as a negative second-pass LLM result.

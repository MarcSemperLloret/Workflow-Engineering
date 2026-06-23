# W19 Conservative Frequency Rules Report

`w19_frequency_rules_conservative_pilot` extends the deployable W18 workflow by
adding conservative frequency-specific rules. It uses the all-pilot temporal
evidence file and does not use gold-derived disagreement targeting.

Input workflow:

```text
data/model_outputs/codificacion_1_ollama/w18_horizon_rules_all_pilot/codificacion_1_ollama.csv
```

Output artifacts:

```text
data/model_outputs/codificacion_1_ollama/w19_frequency_rules_conservative_pilot/codificacion_1_ollama.csv
data/model_outputs/codificacion_1_ollama/w19_frequency_rules_conservative_pilot/temporal_rule_changes.csv
data/model_outputs/codificacion_1_ollama/w19_frequency_rules_conservative_pilot/evaluation_report_gold_v5_exact.csv
data/model_outputs/codificacion_1_ollama/w19_frequency_rules_conservative_pilot/evaluation_report_gold_v5_alias.csv
data/model_outputs/codificacion_1_ollama/w19_frequency_rules_conservative_pilot/temporal_disagreements_gold_v5.csv
```

## Rationale

The broad frequency probe over-applied mixed-frequency rules and reduced
alias-aware accuracy to 0.819. The main failure was confusing prediction
horizons such as 15/30/60 minutes with dataset sampling frequency.

W19 therefore disables the broad "multiple intervals" rule and only permits
frequency corrections when evidence contains high-specificity patterns:

- explicit mixed dataset-frequency tables;
- explicit dataset-specific benchmark sample-rate tables;
- daily reported cases;
- all-dataset 5-minute sampling-rate statements;
- irregular one-hour prediction windows;
- half-hour prediction-length statements.

## Results

Against `gold_v5`, using the manuscript primary metric over 18 objective fields:

| workflow | 20-field accuracy | 18-field objective accuracy |
|---|---:|---:|
| w10_schema_v4_access | 0.803 | 0.821 |
| w18_horizon_rules_all_pilot | 0.836 | 0.858 |
| w19_frequency_rules_conservative_pilot | **0.855** | **0.879** |

W19 makes 19 frequency-field corrections relative to W18:

- fixed: 19
- regressions: 0
- neutral: 0

Temporal field accuracy:

| field | accuracy |
|---|---:|
| horizon_type | 50/50 = 1.000 |
| horizon_min_value | 50/50 = 1.000 |
| horizon_max_value | 50/50 = 1.000 |
| horizon_unit | 50/50 = 1.000 |
| frequency_value | 48/50 = 0.960 |
| frequency_unit | 48/50 = 0.960 |

The remaining focused temporal disagreements are 4 rows:

```text
LightTS   frequency_value, frequency_unit
PDFormer  frequency_value, frequency_unit
```

## Interpretation

W19 is the strongest reported workflow in the pilot. Its objective accuracy is
an internal pilot estimate; the temporal rules were developed on the same
50-paper corpus and therefore should not be presented as an independent
validation result.

This result changes the status of the project: the temporal subproblem is no
longer broadly exploratory. Horizon is solved in the pilot, and frequency is
nearly solved with conservative evidence rules. The remaining two papers should
be handled by manual adjudication or by an explicitly reported residual-error
analysis rather than by adding broader automatic rules.

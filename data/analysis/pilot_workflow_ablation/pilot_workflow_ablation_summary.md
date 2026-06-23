# Pilot workflow ablation analysis

Scope: 50-paper expert-coded pilot only, evaluated against `codificacion_1_manual_gold_v5.csv` with dataset-alias normalization. No independent human-coded validation set is reported in this analysis.

## Key workflow summary

| workflow | display_label | model_or_source | complete_50_paper_output | accuracy_20_fields | accuracy_18_objective | delta_18_vs_direct |
| --- | --- | --- | --- | --- | --- | --- |
| metadata_prior_baseline | Rules-only metadata/prior baseline | rules only | yes | 0.409 | 0.412 | -0.039 |
| w0_direct_50 | Direct local LLM prompt | qwen2.5-coder:7b | yes | 0.462 | 0.451 | 0.000 |
| w2_schema_normalized_50 | Schema + output normalization | qwen2.5-coder:7b | yes | 0.574 | 0.568 | 0.117 |
| w3_field_groups_50 | Grouped field extraction | qwen2.5-coder:7b | yes | 0.602 | 0.610 | 0.159 |
| w4_hybrid_w2_w3_50 | Field-family hybrid | qwen2.5-coder:7b | yes | 0.637 | 0.638 | 0.187 |
| w10_guideline_fewshot_qwen3_coder | Guided stronger local model | qwen3-coder:30b | yes | 0.796 | 0.813 | 0.362 |
| w10_schema_v4_access | Structured baseline + access rules | qwen3-coder:30b | yes | 0.803 | 0.821 | 0.370 |
| w16_temporal_audit_qwen3_coder | Second-pass temporal LLM audit | qwen3-coder:30b | yes | 0.803 | 0.821 | 0.370 |

## Paired paper-cluster bootstrap deltas

| workflow_a | workflow_b | excluded_fields | accuracy_a | accuracy_b | delta_b_minus_a | ci95_low | ci95_high | bootstrap_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_prior_baseline | w0_direct_50 | none | 0.409 | 0.462 | 0.053 | -0.010 | 0.118 | 0.1112 |
| metadata_prior_baseline | w0_direct_50 | coding_status|confidence | 0.412 | 0.451 | 0.039 | -0.020 | 0.099 | 0.2236 |
| w0_direct_50 | w1_metadata_sections | none | 0.462 | 0.561 | 0.099 | 0.046 | 0.156 | 0.0000 |
| w0_direct_50 | w1_metadata_sections | coding_status|confidence | 0.451 | 0.553 | 0.102 | 0.044 | 0.163 | 0.0000 |
| w0_direct_50 | w2_schema_normalized_50 | none | 0.462 | 0.574 | 0.112 | 0.059 | 0.168 | 0.0000 |
| w0_direct_50 | w2_schema_normalized_50 | coding_status|confidence | 0.451 | 0.568 | 0.117 | 0.060 | 0.178 | 0.0000 |
| w2_schema_normalized_50 | w3_field_groups_50 | none | 0.574 | 0.602 | 0.028 | -0.011 | 0.064 | 0.1488 |
| w2_schema_normalized_50 | w3_field_groups_50 | coding_status|confidence | 0.568 | 0.610 | 0.042 | 0.003 | 0.078 | 0.0352 |
| w3_field_groups_50 | w4_hybrid_w2_w3_50 | none | 0.602 | 0.637 | 0.035 | 0.006 | 0.066 | 0.0220 |
| w3_field_groups_50 | w4_hybrid_w2_w3_50 | coding_status|confidence | 0.610 | 0.638 | 0.028 | -0.001 | 0.058 | 0.0684 |
| w0_direct_50 | w5_evidence_required | none | 0.462 | 0.588 | 0.126 | 0.079 | 0.176 | 0.0000 |
| w0_direct_50 | w5_evidence_required | coding_status|confidence | 0.451 | 0.584 | 0.133 | 0.083 | 0.186 | 0.0000 |
| w0_direct_50 | w7_retrieval_tables | none | 0.462 | 0.576 | 0.114 | 0.060 | 0.171 | 0.0000 |
| w0_direct_50 | w7_retrieval_tables | coding_status|confidence | 0.451 | 0.568 | 0.117 | 0.058 | 0.178 | 0.0000 |
| w0_direct_50 | w8_self_consistency | none | 0.462 | 0.574 | 0.112 | 0.057 | 0.170 | 0.0000 |
| w0_direct_50 | w8_self_consistency | coding_status|confidence | 0.451 | 0.566 | 0.114 | 0.056 | 0.177 | 0.0000 |
| w4_hybrid_w2_w3_50 | w9_multi_model_panel | none | 0.637 | 0.644 | 0.007 | -0.005 | 0.019 | 0.2896 |
| w4_hybrid_w2_w3_50 | w9_multi_model_panel | coding_status|confidence | 0.638 | 0.647 | 0.009 | -0.004 | 0.022 | 0.2276 |
| w9_multi_model_panel | w11_cot_grouped | none | 0.644 | 0.726 | 0.082 | 0.052 | 0.113 | 0.0000 |
| w9_multi_model_panel | w11_cot_grouped | coding_status|confidence | 0.647 | 0.733 | 0.087 | 0.056 | 0.119 | 0.0000 |
| w4_hybrid_w2_w3_50 | w10_guideline_fewshot_qwen3_coder | none | 0.637 | 0.796 | 0.159 | 0.118 | 0.200 | 0.0000 |
| w4_hybrid_w2_w3_50 | w10_guideline_fewshot_qwen3_coder | coding_status|confidence | 0.638 | 0.813 | 0.176 | 0.129 | 0.221 | 0.0000 |
| w0_direct_50 | w3_field_groups_50 | none | 0.462 | 0.602 | 0.140 | 0.088 | 0.193 | 0.0000 |
| w0_direct_50 | w3_field_groups_50 | coding_status|confidence | 0.451 | 0.610 | 0.159 | 0.106 | 0.213 | 0.0000 |
| w3_field_groups_50 | w10_guideline_fewshot_qwen3_coder | none | 0.602 | 0.796 | 0.194 | 0.144 | 0.244 | 0.0000 |
| w3_field_groups_50 | w10_guideline_fewshot_qwen3_coder | coding_status|confidence | 0.610 | 0.813 | 0.203 | 0.152 | 0.254 | 0.0000 |
| w10_guideline_fewshot_qwen3_coder | w10_schema_v4_access | none | 0.796 | 0.803 | 0.007 | 0.002 | 0.013 | 0.0140 |
| w10_guideline_fewshot_qwen3_coder | w10_schema_v4_access | coding_status|confidence | 0.813 | 0.821 | 0.008 | 0.002 | 0.014 | 0.0140 |
| w10_schema_v4_access | w16_temporal_audit_qwen3_coder | none | 0.803 | 0.803 | 0.000 | 0.000 | 0.000 | 1.0000 |
| w10_schema_v4_access | w16_temporal_audit_qwen3_coder | coding_status|confidence | 0.821 | 0.821 | 0.000 | 0.000 | 0.000 | 1.0000 |

## Model-family sensitivity for the guided workflow

| model | covered_papers_raw | covered_papers_access | accuracy_18_raw | accuracy_18_access | access_delta_18 |
| --- | --- | --- | --- | --- | --- |
| qwen3-coder:30b | 50 | 50 | 0.813 | 0.821 | 0.008 |
| gemma3:12b | 50 | 50 | 0.672 | 0.674 | 0.002 |
| mistral-nemo:12b | 50 | 50 | 0.588 | 0.601 | 0.013 |
| llama3.1:8b | 50 | 50 | 0.677 | 0.686 | 0.009 |

## Final workflow field groups

| field_group | correct | total | errors | accuracy |
| --- | --- | --- | --- | --- |
| bibliographic | 78 | 100 | 22 | 0.780 |
| task_dataset | 163 | 200 | 37 | 0.815 |
| access | 85 | 100 | 15 | 0.850 |
| temporal_horizon | 167 | 200 | 33 | 0.835 |
| temporal_frequency | 77 | 100 | 23 | 0.770 |
| graph_nodes | 169 | 200 | 31 | 0.845 |

## Post-hoc human-review triage simulation

| tier | field_count | instances | coverage_excluding_diagnostics | correct | accuracy |
| --- | --- | --- | --- | --- | --- |
| auto_accept | 4 | 200 | 0.222 | 181 | 0.905 |
| review | 10 | 500 | 0.556 | 412 | 0.824 |
| human_required | 4 | 200 | 0.222 | 146 | 0.730 |
| excluded_diagnostic | 2 | 100 | 0.100 | 64 | 0.640 |

## Residual error taxonomy for final workflow

| error_category | errors | share_of_errors | fields |
| --- | --- | --- | --- |
| temporal_setup_extraction | 56 | 0.348 | frequency_unit|frequency_value|horizon_max_value|horizon_min_value|horizon_type|horizon_unit |
| graph_node_semantics | 31 | 0.193 | graph_node_count_max|graph_node_count_min|graph_node_count_status|node_unit |
| task_target_schema_boundary | 24 | 0.149 | domain_class|target_class|task_class |
| bibliographic_metadata | 22 | 0.137 | venue_type|year |
| access_evidence | 15 | 0.093 | has_public_code|has_public_dataset |
| dataset_family_recall_normalization | 13 | 0.081 | dataset_family |

## Worst final workflow fields

| field | correct | total | errors | accuracy |
| --- | --- | --- | --- | --- |
| confidence | 27 | 50 | 23 | 0.540 |
| horizon_type | 35 | 50 | 15 | 0.700 |
| coding_status | 37 | 50 | 13 | 0.740 |
| dataset_family | 37 | 50 | 13 | 0.740 |
| target_class | 37 | 50 | 13 | 0.740 |
| venue_type | 37 | 50 | 13 | 0.740 |
| frequency_value | 38 | 50 | 12 | 0.760 |
| frequency_unit | 39 | 50 | 11 | 0.780 |
| has_public_code | 39 | 50 | 11 | 0.780 |
| node_unit | 40 | 50 | 10 | 0.800 |

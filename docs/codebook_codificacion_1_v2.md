# Codebook Codificacion 1 v2

This codebook defines closed-label decisions for `codificacion_1_manual_gold_v2.csv`.
It supersedes the implicit conventions used in the pilot file, but the original
pilot CSV is kept unchanged for auditability.

## General Rule

Code only what is supported by the paper text, metadata, or clearly identified
benchmark conventions. Prefer stable closed labels over free text. If a field is
not supported by evidence, use `not_reported`; if the evidence exists but does
not fit the schema cleanly, use `partial`/`medium` or `dataset_specific` where
available.

## Bibliography And Access

### `venue_type`

Use the strongest formal venue evidence available.

- `conference`: the paper is in, accepted to, or clearly copyrighted by a named
  conference or proceedings venue such as AAAI, IJCAI, CIKM, KDD, NeurIPS, ICLR,
  ICML, WWW, VLDB, ACM conference proceedings, etc.
- `journal`: the paper header or bibliographic metadata clearly identifies a
  journal or transactions venue, including IEEE/ACM transactions or journal
  accepted manuscripts.
- `preprint`: only arXiv/preprint evidence is available, or the PDF uses a
  generic template without reliable venue information.
- `competition_workshop`: the work is primarily a competition report, challenge
  submission, or workshop/competition paper.
- `not_reported`: venue evidence is absent.

When arXiv and a formal venue both appear, code the formal venue. Do not treat
an arXiv line alone as evidence against a formal venue.

### `has_public_code`

This field refers to method/source/baseline implementation code, not merely
dataset access.

- `yes`: the paper states that source code, implementation code, project code,
  or baseline implementation is publicly available.
- `no`: the paper explicitly states that code is not available, not released, or
  not publicly available.
- `not_reported`: the paper is silent about code availability, or only mentions
  external/base/dataset repositories.

A dataset GitHub link counts for `has_public_dataset=yes`, but not necessarily
for `has_public_code=yes`.

### `has_public_dataset`

Use `yes` when the paper uses named public benchmarks or releases a dataset.
Use `no` only if the dataset is clearly private/unreleased. Otherwise use
`not_reported`.

## Task And Target

`task_class` describes the main empirical task, not the model architecture.
Use `benchmark_evaluation` when the paper's main contribution is a benchmark,
dataset, or systematic evaluation protocol rather than a new forecasting model.

`target_class` describes what is predicted:

- `traffic_speed`: speed only.
- `traffic_flow`: flow/volume only.
- `traffic_speed_and_flow`: both speed and flow datasets/tasks are evaluated.
- `multiple_targets`: targets are heterogeneous or outside a single listed class.

## Temporal Fields

### `frequency_value` and `frequency_unit`

Frequency is the data recording or aggregation interval. Do not confuse it with
the forecast horizon. For PeMS/METR-LA style traffic benchmarks, frequency is
usually `5` and `minutes`.

### `horizon_type`, `horizon_min_value`, `horizon_max_value`, `horizon_unit`

Code the prediction horizon as reported in the experimental setup. Prefer the
unit used by the paper.

- `sequence`: the model predicts a fixed output sequence, e.g. "predict the next
  12 time steps". Use `horizon_min_value=12`, `horizon_max_value=12`,
  `horizon_unit=steps`.
- `multi_horizon`: the paper explicitly evaluates or defines multiple horizons,
  e.g. "15, 30, and 60 minutes" or "Horizon 3, 6, and 12". Use the smallest and
  largest reported horizon and the reported unit.
- `single_horizon`: the paper has one fixed prediction horizon/window.
- `dataset_specific`: event, irregular, competition, or benchmark settings do
  not map cleanly to a single sequence/single/multi horizon.
- `not_reported`: no usable horizon evidence is found.

Do not convert steps to minutes unless the paper itself reports only minutes.
If both are reported, use the unit used in the main experiment table/setup.

## Graph/Node Fields

### `graph_node_count_min` and `graph_node_count_max`

For multiple datasets, code the minimum and maximum node counts across the
datasets that are actually evaluated.

### `node_unit`

Use the semantic unit represented by graph nodes.

- `sensors`: traffic loop detectors or sensor datasets such as METR-LA,
  PEMS-BAY, PeMSD3/4/7/8, PeMSD7(M/L), LargeST sensors.
- `stations`: monitoring stations or PV/weather/air-quality station-like sites.
- `road_segments`: road links/segments when the node is explicitly a road
  segment rather than a detector.
- `links`: road links in event/process datasets.
- `counties`: county-level epidemiological nodes.
- `nodes`: generic graph nodes, grid cells, regions, turbines, variables, or
  entities not covered above.
- `mixed`: only when the paper truly combines datasets with different semantic
  node units, not merely several sensor datasets.
- `not_reported`: node semantics are absent.

## Coding Quality

Use `coding_status=complete` when all required fields are supported or can be
mapped by explicit codebook rules. Use `partial` when important fields are
missing, ambiguous, or schema-forced.

Use `confidence=high` only for straightforward traffic benchmark papers with
clear metadata and dataset tables. Use `medium` for schema boundary cases, and
`low` for substantial missing evidence.

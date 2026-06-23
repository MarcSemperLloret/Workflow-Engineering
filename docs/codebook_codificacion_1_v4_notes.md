# Codebook Notes For Gold v5

These notes extend `codebook_codificacion_1_v3_notes.md` for the targeted
temporal adjudication applied in `codificacion_1_manual_gold_v5.csv`.

## Frequency Fields

When a paper reports both the raw collection interval and an aggregation interval
used for modeling, code the interval of the time series that enters the main
experimental model. For example, if measurements are collected every 30 seconds
but then aggregated to 1-hour observations for experiments, code:

- `frequency_value=1`
- `frequency_unit=hours`

For multi-dataset benchmark papers, use:

- `frequency_value=dataset_specific`
- `frequency_unit=mixed`

when the evaluated datasets use different time intervals.

If the frequency unit is already `dataset_specific`, the paired value should
also be `dataset_specific`, not `not_applicable`.

## Horizon Fields

When a table reports output length as `Q`, `F`, or another sequence-length
variable, code the horizon unit as `steps` unless the main experiment reports
the horizon only in temporal units.

When a paper has incompatible horizon setups across evaluated datasets, use
`horizon_type=dataset_specific`. This includes papers where one dataset group
uses a fixed multi-step sequence and another dataset group uses single-step
prediction.

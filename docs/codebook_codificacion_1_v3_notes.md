# Codebook Notes For Gold v4

These notes extend `codebook_codificacion_1_v2.md` for the changes applied in
`data/annotations/codificacion_1_manual_gold_v4.csv`.

## Temporal Fields

When a paper explicitly predicts `t+1`, "the next time step", or equivalent
single-step output, code:

- `horizon_type=single_horizon`
- `horizon_min_value=1`
- `horizon_max_value=1`
- `horizon_unit=steps`

When a paper reports horizons such as "1-week, 2-week, and 3-week ahead", keep
the reported unit:

- `horizon_type=multi_horizon`
- `horizon_min_value=1`
- `horizon_max_value=3`
- `horizon_unit=weeks`

Do not convert reported weeks to days unless the paper itself reports days as
the unit of the experimental horizon.

## Dataset Families

Use dataset families that cover all datasets evaluated in the main experiments,
not only the graph-based subset. For papers mixing graph and grid traffic
datasets, include both groups when they are part of the main evaluation.

Normalize common PeMS aliases before scoring:

- `pems03` -> `pemsd3`
- `pems04` -> `pemsd4`
- `pems07` -> `pemsd7`
- `pems08` -> `pemsd8`

For paired inflow/outflow metro datasets from Hangzhou, use `hzme` as a compact
family label unless a later schema separates inflow and outflow.

## Public Dataset Availability

Competition or benchmark dataset URLs count as `has_public_dataset=yes` when
the paper gives a dataset link or an official competition dataset page.

A method/project GitHub link does not automatically imply public dataset
availability unless the paper text or repository explicitly indicates dataset
release. Keep such cases as `no` or `not_reported` according to the evidence.

## Node Counts

For grid-region datasets, multiply the reported grid dimensions when the paper
defines each grid cell as a spatial region. For example:

- `16x8` regions -> `128`
- `10x20` regions -> `200`
- `32x32` regions -> `1024`

For large web/network datasets, prefer the exact filtered node count reported in
the dataset construction section over rounded abstract language such as `366K`.

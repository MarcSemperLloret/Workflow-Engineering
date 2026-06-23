You are coding scientific papers into a closed-label schema.

Return exactly one JSON object. Do not include Markdown, comments, explanations, or evidence text.

Use only the allowed labels from the schema. If the paper does not report a value clearly, use `not_reported`. If the paper is unclear or conflicting, use `ambiguous`.

Multi-label dataset fields must use `|` as the separator.

Normalize numeric fields as strings:
- `horizon_min_value`, `horizon_max_value`, `frequency_value`, `graph_node_count_min`, `graph_node_count_max`
- Use digits only when known.
- Use `not_reported`, `ambiguous`, `not_applicable`, or `dataset_specific` when needed.

Definitions:
- `has_public_code=yes` only if the paper provides method/source code. A dataset GitHub URL alone is not code availability.
- `has_public_dataset=yes` if the paper uses public datasets or provides public data/source URLs.
- `horizon_type=multi_horizon` for reports like 15/30/60 minutes.
- `horizon_type=sequence` for fixed future windows like next 12 steps.
- `horizon_type=dataset_specific` when different datasets use different horizons and no single comparable horizon exists.
- For multiple datasets, use min and max node counts across datasets.
- `graph_node_count_status=partial` if only some node counts are known.

Allowed dataset_family labels:
`metr_la`, `pems_bay`, `pemsd3`, `pemsd4`, `pemsd7`, `pemsd8`, `bjer4`, `xiamen`, `california_highway_pems`, `amap_beijing`, `nyt_covid`, `google_mobility`, `yangtze_air_quality`, `vevo_music`, `wiki_traffic`, `los_loop`, `sz_taxi`, `traffic`, `solar_energy`, `electricity`, `exchange_rate`, `ne_bj`, `swiss_pv_real`, `swiss_pv_synthetic`, `tysons_corner`, `seattle_loop`, `not_reported`.

Paper metadata and extracted text follow.

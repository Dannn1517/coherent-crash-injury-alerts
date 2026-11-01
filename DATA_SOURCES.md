# Data sources and provenance

All three datasets are city-level public crash records. Raw records are not redistributed in this
repository. The released files are derived, non-identifying grid-hour counts. Users who rebuild
from raw records should follow the terms of the corresponding city portal.

| City | Source | Dataset | Period | Raw rows | In-grid events |
|---|---|---|---|---:|---:|
| New York City | New York City Police Department, Motor Vehicle Collisions | NYC Open Data `h9gi-nx95` | 2021–2025 | 2,301,644 | 449,606 |
| Chicago | City of Chicago, Traffic Crashes – Crashes | Chicago Data Portal `85ca-t3if` | 2021–2025 | 1,108,694 | 543,956 |
| Taichung | Police Bureau, Taichung City Government, road traffic accident data | Taichung City Open Data Platform | 2021–2024 | 229,425 | 229,425 |

Source portals:

- New York City: https://data.cityofnewyork.us/d/h9gi-nx95
- Chicago: https://data.cityofchicago.org/d/85ca-t3if
- Taichung: Taichung City Open Data Platform (city portal; A1/A2 per-case records)

Taichung's public records cover A1/A2 crashes (injury or fatal). Property-damage-only A3 records are
not public, so the Taichung crash channel approximates injury crashes. This is a semantic
difference from New York City and Chicago, not a filtering choice.

## Common grid and time contract

- Grid: 20x20 UTM cells per city (EPSG:32618 for New York City, EPSG:32616 for Chicago,
  EPSG:32651 for Taichung). The 10x10 New York City variant is an exact block sum of the 20x20
  labels; the 40x40 variant is built directly on a finer UTM grid over the same bounding box (its
  builder is not part of this release).
- Time: hourly counts from 2021-01-01. New York City and Chicago cover 2021–2025; Taichung covers
  2021–2024.
- Splits: New York City and Chicago train 2021–2023, decision 2024, locked test 2025. Taichung
  trains 2021–2022, decision 2023, locked test 2024.
- Model windows: 12 hours of history, 12 hours ahead, stride 3. Windows crossing a split boundary
  are dropped.

## Released artifacts

`data/processed/` contains `labels.npz` (`region_hour`, `region_hour_injured`,
`region_hour_killed`), `hour_index.parquet`, `day_index.parquet`, and `meta.json` for each city and
resolution. `data/model_inputs/` contains the exact `{train,val,test}.npz` arrays consumed by the
model. `data/manifests/` contains build manifests, SHA-256 prefixes, the Chicago filter audit, and
the raw-source checksums.

All windowed inputs use the midpoint time-of-day encoding `(hour+0.5)/24`. The released arrays are
the exact ones used in the paper; rebuilding them with `src/data/build_t3_stllm_format.py`
reproduces the same encoding.

## Checksums

`data/manifests/raw_sources.sha256` lists the SHA-256 of the three raw crash files used for the
label build. The build manifests record short hashes for the source columns consumed by each
builder. Raw files are not part of this repository.

## License and attribution

The derived aggregates are provided for research reproducibility. Redistribution and reuse of the
underlying records remain governed by the source city portal terms. Please cite the three source
agencies and the paper when using these files.

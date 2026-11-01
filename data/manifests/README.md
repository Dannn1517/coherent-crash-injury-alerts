# Manifests

- `T3_region_hour*.json`: build manifests for the grid-hour labels (source dataset IDs, short
  source hashes, counts, and cross-validation against the shared city aggregates).
- `chicago_filter_decomposition_audit.json`: decomposition of the difference between the grid
  labels and the road-eligible export for Chicago.
- `raw_sources.sha256`: full SHA-256 of the three raw crash files used for the label build; paths
  are relative to the raw-data root described in `DATA_SOURCES.md`. The raw files themselves are
  not redistributed.

The `qa_against_shared_daily_labels` blocks record the cross-validation reported in the paper.
They are provenance values; the shared road-by-day exports are not part of this release.

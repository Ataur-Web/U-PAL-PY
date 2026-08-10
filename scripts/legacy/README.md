# Legacy scripts

Superseded during the evaluation phase. Kept for provenance; not used to
produce any figure in the submitted report.

| Script | Status |
|---|---|
| `benchmark_sources_csv.py` | Superseded by `scripts/benchmark_sources.py`, which adds the index-provenance and URL-validity checks reported in Section 5.4. |
| `summarise_benchmarks.py` | Reads `benchmark_results/retrieval.json` and `source_attribution.json` at their pre-reorganisation paths. Those files now live under `benchmark_results/5.1_retrieval/` and `_superseded_earlier_runs/`, so this script no longer runs unmodified. |
| `monitor_scrape.py` | One-off helper for watching Scrapy progress during corpus collection. |

# Benchmark results


**Relevance rule: fact entry OR the institutional page it cites**

| Subset | Configuration | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---:|---:|---:|---:|
| All (n=20) | Dense-only (ChromaDB) | 0.40 | 0.55 | 0.60 | 0.47 |
| All (n=20) | Sparse-only (BM25) | 0.30 | 0.70 | 0.75 | 0.50 |
| All (n=20) | Hybrid (EnsembleRetriever, RRF) | 0.45 | 0.75 | 0.85 | 0.61 |
| English (n=10) | Dense-only (ChromaDB) | 0.60 | 0.80 | 0.90 | 0.71 |
| English (n=10) | Sparse-only (BM25) | 0.30 | 0.60 | 0.70 | 0.47 |
| English (n=10) | Hybrid (EnsembleRetriever, RRF) | 0.50 | 0.80 | 1.00 | 0.67 |
| Welsh (n=10) | Dense-only (ChromaDB) | 0.20 | 0.30 | 0.30 | 0.23 |
| Welsh (n=10) | Sparse-only (BM25) | 0.30 | 0.80 | 0.80 | 0.53 |
| Welsh (n=10) | Hybrid (EnsembleRetriever, RRF) | 0.40 | 0.70 | 0.70 | 0.55 |

**Relevance rule: the canonical fact entry only**

| Subset | Configuration | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---:|---:|---:|---:|
| All (n=20) | Dense-only (ChromaDB) | 0.05 | 0.30 | 0.45 | 0.19 |
| All (n=20) | Sparse-only (BM25) | 0.10 | 0.35 | 0.50 | 0.25 |
| All (n=20) | Hybrid (EnsembleRetriever, RRF) | 0.15 | 0.40 | 0.50 | 0.30 |
| English (n=10) | Dense-only (ChromaDB) | 0.10 | 0.40 | 0.70 | 0.30 |
| English (n=10) | Sparse-only (BM25) | 0.10 | 0.40 | 0.50 | 0.28 |
| English (n=10) | Hybrid (EnsembleRetriever, RRF) | 0.20 | 0.40 | 0.60 | 0.35 |
| Welsh (n=10) | Dense-only (ChromaDB) | 0.00 | 0.20 | 0.20 | 0.08 |
| Welsh (n=10) | Sparse-only (BM25) | 0.10 | 0.30 | 0.50 | 0.23 |
| Welsh (n=10) | Hybrid (EnsembleRetriever, RRF) | 0.10 | 0.40 | 0.40 | 0.25 |

**Source attribution**

| Subset | Queries | Answered | Sources | Sources/query | UWTSD institutional |
|---|---:|---:|---:|---:|---:|
| All | 20 | 20/20 | 58 | 2.9 | 58 (100%) |
| English | 10 | 10/10 | 29 | 2.9 | 29 (100%) |
| Welsh | 10 | 10/10 | 29 | 2.9 | 29 (100%) |

**Response latency (ms)**

| Subset | Mean | Median | 95th pct | Min | Max |
|---|---:|---:|---:|---:|---:|
| All | 3858 | 3722 | 5350 | 2884 | 6104 |
| English | 3780 | 3574 | 6104 | 3057 | 6104 |
| Welsh | 3936 | 3893 | 5350 | 2884 | 5350 |

**Sources cited per answer**

| Sources shown | Queries |
|---:|---:|
| 2 | 2 |
| 3 | 18 |

_Per-query chart data written to `chart_data.csv`_

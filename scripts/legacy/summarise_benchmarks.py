#!/usr/bin/env python3
"""
consolidates the two benchmark outputs into report-ready tables

reads benchmark_results/retrieval.json and benchmark_results/source_attribution.json
and emits markdown tables plus a flat csv for charting. run this after both
benchmarks so the chapter 5 figures come from the same data rather than being
transcribed by hand, which is where transcription errors creep in.

usage:
    python scripts/summarise_benchmarks.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)
UWTSD = "https://www.uwtsd.ac.uk"


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def retrieval_tables() -> list[str]:
    path = OUT_DIR / "retrieval.json"
    if not path.exists():
        return ["_retrieval.json not found — run scripts/benchmark_retrieval.py_"]
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for mode, caption in (
        ("relaxed", "Relevance rule: fact entry OR the institutional page it cites"),
        ("strict", "Relevance rule: the canonical fact entry only"),
    ):
        out.append(f"\n**{caption}**\n")
        out.append("| Subset | Configuration | Hit@1 | Hit@3 | Hit@5 | MRR |")
        out.append("|---|---|---:|---:|---:|---:|")
        for subset, configs in data["summary"][mode].items():
            for cfg, m in configs.items():
                out.append(
                    f"| {subset} | {cfg} | {m['Hit@1']:.2f} | {m['Hit@3']:.2f} "
                    f"| {m['Hit@5']:.2f} | {m['MRR']:.2f} |"
                )
    return out


def sources_tables() -> list[str]:
    path = OUT_DIR / "source_attribution.json"
    if not path.exists():
        return ["_source_attribution.json not found — run scripts/benchmark_sources_csv.py_"]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["queries"]
    out: list[str] = []

    def stats(subset: list[dict]) -> dict:
        srcs = [s for r in subset for s in r.get("sources", [])]
        inst = [s for s in srcs if str(s.get("url") or "").startswith(UWTSD)]
        times = [r["elapsed_ms"] for r in subset if r["success"]]
        return {
            "n": len(subset),
            "ok": sum(1 for r in subset if r["success"]),
            "sources": len(srcs),
            "per_query": len(srcs) / len(subset) if subset else 0,
            "inst": len(inst),
            "times": times,
        }

    subsets = {
        "All": rows,
        "English": [r for r in rows if r["language"] == "en"],
        "Welsh": [r for r in rows if r["language"] == "cy"],
    }

    out.append("\n**Source attribution**\n")
    out.append("| Subset | Queries | Answered | Sources | Sources/query | UWTSD institutional |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for label, sub in subsets.items():
        s = stats(sub)
        out.append(
            f"| {label} | {s['n']} | {s['ok']}/{s['n']} | {s['sources']} | "
            f"{s['per_query']:.1f} | {s['inst']} ({_pct(s['inst'], s['sources'])}) |"
        )

    out.append("\n**Response latency (ms)**\n")
    out.append("| Subset | Mean | Median | 95th pct | Min | Max |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for label, sub in subsets.items():
        t = sorted(stats(sub)["times"])
        if not t:
            continue
        p95 = t[min(len(t) - 1, int(round(0.95 * (len(t) - 1))))]
        out.append(
            f"| {label} | {statistics.mean(t):.0f} | {statistics.median(t):.0f} "
            f"| {p95:.0f} | {min(t):.0f} | {max(t):.0f} |"
        )

    # distribution of how many sources each answer carried
    dist: dict[int, int] = {}
    for r in rows:
        dist[len(r.get("sources", []))] = dist.get(len(r.get("sources", [])), 0) + 1
    out.append("\n**Sources cited per answer**\n")
    out.append("| Sources shown | Queries |")
    out.append("|---:|---:|")
    for k in sorted(dist):
        out.append(f"| {k} | {dist[k]} |")

    # flat csv for charting
    with open(OUT_DIR / "chart_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Query #", "Language", "Latency (ms)", "Sources", "Institutional sources"])
        for i, r in enumerate(rows, 1):
            srcs = r.get("sources", [])
            inst = sum(1 for s in srcs if str(s.get("url") or "").startswith(UWTSD))
            w.writerow([i, r["language"], f"{r['elapsed_ms']:.0f}", len(srcs), inst])
    out.append("\n_Per-query chart data written to `chart_data.csv`_")
    return out


def main() -> int:
    lines = ["# Benchmark results", ""]
    lines += retrieval_tables()
    lines += sources_tables()
    text = "\n".join(lines)
    (OUT_DIR / "RESULTS.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

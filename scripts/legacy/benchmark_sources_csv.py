#!/usr/bin/env python3
"""
comprehensive source attribution benchmark with CSV export

generates a detailed CSV report that includes:
- query + language
- response time
- number of sources
- each source: title, category, url, excerpt (so you can manually verify relevance)
- success/failure status

output: benchmark_results_detailed.csv (easy to review and fact-check in Excel/Google Sheets)
"""

import json
import logging
import time
import requests
import csv
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# same output folder the retrieval benchmark uses, so chapter 5 cites one place
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"
OUT_DIR.mkdir(exist_ok=True)

# test queries - balanced: 10 english, 10 welsh
# covers: courses, admissions, support, fees, research, international
TEST_QUERIES = [
    # ENGLISH (10 queries)
    ("What courses does UWTSD offer?", "en"),
    ("How do I apply to UWTSD?", "en"),
    ("What student support services are available?", "en"),
    ("How much do courses cost?", "en"),
    ("International student services", "en"),
    ("Entry requirements for students", "en"),
    ("Student accommodation options", "en"),
    ("Counselling and wellbeing services", "en"),
    ("Research opportunities at UWTSD", "en"),
    ("Study abroad opportunities", "en"),

    # WELSH (10 queries)
    ("Beth yw costau'r cwrs?", "cy"),
    ("Cymorth myfyrwyr", "cy"),
    ("Cymhwyster mynediad", "cy"),
    ("Gwybodaeth am lety myfyrwyr", "cy"),
    ("Sut i wneud cais i UWTSD?", "cy"),
    ("Cymhwystrau mynediad", "cy"),
    ("Cymorth ariannol", "cy"),
    ("Gwasanaethau rhyngwladol", "cy"),
    ("Cyfleoedd astudio dramor", "cy"),
    ("Ymchwil ac arloesedd", "cy"),
]

def test_query(backend_url: str, query: str, language: str) -> dict:
    """send a single test query and collect detailed metrics"""
    start_time = time.time()

    try:
        response = requests.post(
            f"{backend_url}/api/chat",
            json={
                "message": query,
                "language": language,
                "session_id": "benchmark-test"
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        elapsed = time.time() - start_time

        sources = data.get("sources", [])

        return {
            "query": query,
            "language": language,
            "success": True,
            "elapsed_ms": elapsed * 1000,
            "num_sources": len(sources),
            "sources": sources,  # keep full source objects for CSV export
            "response": data.get("reply", "")[:200]  # first 200 chars of response
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "query": query,
            "language": language,
            "success": False,
            "error": str(e),
            "elapsed_ms": elapsed * 1000,
            "num_sources": 0,
            "sources": [],
            "response": ""
        }

def run_benchmark(backend_url: str = "http://localhost:3001"):
    """run full benchmark and generate CSV"""
    log.info("=" * 70)
    log.info("SOURCE ATTRIBUTION BENCHMARK - CSV EXPORT")
    log.info("=" * 70)
    log.info(f"Backend URL: {backend_url}")
    log.info(f"Total queries: {len(TEST_QUERIES)}")
    log.info("")

    results = []

    for idx, (query, lang) in enumerate(TEST_QUERIES, 1):
        log.info(f"[{idx}/{len(TEST_QUERIES)}] Testing: {query[:50]}...")
        result = test_query(backend_url, query, lang)
        results.append(result)

        if result["success"]:
            log.info(f"  ✓ {result['num_sources']} sources | {result['elapsed_ms']:.0f}ms")
        else:
            log.error(f"  ✗ Failed: {result['error']}")

        time.sleep(0.5)  # rate limit

    # generate CSV report
    log.info("")
    log.info("=" * 70)
    log.info("GENERATING CSV REPORT")
    log.info("=" * 70)

    csv_file = OUT_DIR / "source_attribution_per_query.csv"

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # header
        writer.writerow([
            "Query #",
            "Query Text",
            "Language",
            "Status",
            "Response Time (ms)",
            "Number of Sources",
            "Source 1 Title",
            "Source 1 Category",
            "Source 1 URL",
            "Source 1 Excerpt",
            "Source 2 Title",
            "Source 2 Category",
            "Source 2 URL",
            "Source 2 Excerpt",
            "Source 3 Title",
            "Source 3 Category",
            "Source 3 URL",
            "Source 3 Excerpt",
            "Response (first 200 chars)",
            "Error/Notes"
        ])

        # data rows
        for idx, result in enumerate(results, 1):
            row = [
                idx,
                result["query"],
                result["language"],
                "✓ SUCCESS" if result["success"] else "✗ FAILED",
                f"{result['elapsed_ms']:.0f}",
                result["num_sources"],
            ]

            # add source data (up to 3 sources)
            for source_idx in range(3):
                if source_idx < len(result["sources"]):
                    src = result["sources"][source_idx]
                    row.extend([
                        src.get("title", ""),
                        src.get("category", ""),
                        src.get("url", ""),
                        src.get("excerpt", "")[:100],  # first 100 chars
                    ])
                else:
                    row.extend(["", "", "", ""])

            row.append(result["response"])
            row.append(result.get("error", ""))

            writer.writerow(row)

    log.info(f"✓ CSV exported to {csv_file}")
    log.info(f"  Total rows: {len(results) + 1} (including header)")

    # print summary
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    log.info(f"Success rate: {successful}/{len(results)} ({100*successful//len(results)}%)")
    log.info(f"Failed: {failed}")

    total_sources = sum(r.get("num_sources", 0) for r in results if r["success"])
    avg_sources = total_sources / successful if successful > 0 else 0

    log.info(f"Total sources returned: {total_sources}")
    log.info(f"Average sources per query: {avg_sources:.1f}")

    times = [r["elapsed_ms"] for r in results if r["success"]]
    if times:
        log.info(f"Response time: {min(times):.0f}ms - {max(times):.0f}ms (avg: {sum(times)/len(times):.0f}ms)")

    # language breakdown
    en_queries = sum(1 for q, l in TEST_QUERIES if l == "en")
    cy_queries = sum(1 for q, l in TEST_QUERIES if l == "cy")
    en_success = sum(1 for r in results if r["language"] == "en" and r["success"])
    cy_success = sum(1 for r in results if r["language"] == "cy" and r["success"])

    log.info("")
    log.info("Language Support:")
    log.info(f"  English: {en_success}/{en_queries} successful")
    log.info(f"  Welsh: {cy_success}/{cy_queries} successful")

    # save JSON too for reference
    json_file = OUT_DIR / "source_attribution.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "backend_url": backend_url,
            "total_queries": len(TEST_QUERIES),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{100*successful//len(results)}%",
            "avg_sources_per_query": f"{avg_sources:.1f}",
            "queries": results
        }, f, ensure_ascii=False, indent=2)

    log.info(f"✓ JSON also saved to {json_file}")
    log.info("")
    log.info("=" * 70)
    log.info("✓ BENCHMARK COMPLETE!")
    log.info("=" * 70)
    log.info("")
    log.info("📊 CSV REPORT: benchmark_results_detailed.csv")
    log.info("   Open in Excel or Google Sheets to review and verify sources")
    log.info("")
    log.info("🔍 HOW TO VERIFY SOURCES:")
    log.info("   1. Open benchmark_results_detailed.csv")
    log.info("   2. For each query, check if the returned sources are relevant")
    log.info("   3. Click source URLs to verify they contain relevant content")
    log.info("   4. Check excerpt text to confirm it matches the query topic")
    log.info("")

    return results

if __name__ == "__main__":
    import sys
    backend_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3001"
    run_benchmark(backend_url)

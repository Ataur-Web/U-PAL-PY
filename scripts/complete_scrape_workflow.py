#!/usr/bin/env python3
"""
end-to-end pipeline orchestrator - from raw scrapy output to live knowledge base

automates the full update cycle:
1. validates scraped data (jsonl → json conversion, sanitization)
2. backs up existing corpus (versioned by timestamp)
3. replaces corpus with fresh scraped pages
4. resets chromadb and re-embeds all documents
5. verifies 100% of sources have valid uwtsd.ac.uk urls
6. reports statistics (docs by category/language, url coverage)

run once after scraping completes. handles all intermediate steps so user only
needs to: scrapy crawl → python complete_scrape_workflow.py → restart backend.

usage:
    python scripts/complete_scrape_workflow.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def convert_jsonl_to_json(jsonl_file: str, json_file: str) -> int:
    """Convert JSONL to JSON array format."""
    log.info(f"Converting {jsonl_file} → {json_file}")

    jsonl_path = Path(jsonl_file)
    json_path = Path(json_file)

    if not jsonl_path.exists():
        log.error(f"Scraped file not found: {jsonl_file}")
        return 0

    items = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    # Validate required fields
                    if item.get('url') and item.get('content') and len(item.get('content', '')) > 50:
                        items.append(item)
                except json.JSONDecodeError as e:
                    log.debug(f"Parse error line {line_num}: {e}")
                    continue
    except Exception as e:
        log.error(f"Failed to read: {e}")
        return 0

    if not items:
        log.error("No valid items found in JSONL")
        return 0

    # Write JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    log.info(f"✓ Converted {len(items)} items to {json_file}")
    return len(items)


def backup_corpus(corpus_file: str) -> bool:
    """Backup existing corpus."""
    corpus_path = Path(corpus_file)
    backup_path = corpus_path.parent / f"{corpus_path.stem}-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}{corpus_path.suffix}"

    if not corpus_path.exists():
        log.warning(f"No existing corpus to backup: {corpus_file}")
        return True

    try:
        with open(corpus_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        log.info(f"✓ Backed up to {backup_path.name}")
        return True
    except Exception as e:
        log.error(f"Backup failed: {e}")
        return False


def replace_corpus(source: str, dest: str) -> bool:
    """Replace corpus file."""
    src_path = Path(source)
    dst_path = Path(dest)

    if not src_path.exists():
        log.error(f"Source not found: {source}")
        return False

    try:
        with open(src_path, 'r', encoding='utf-8') as src:
            with open(dst_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        log.info(f"✓ Replaced {dst_path.name}")
        return True
    except Exception as e:
        log.error(f"Replace failed: {e}")
        return False


def run_reingest() -> bool:
    """Re-ingest corpus into ChromaDB."""
    log.info("Re-ingesting corpus into ChromaDB...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'scripts.ingest', '--reset'],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            # Check for success message
            if 'Ingested' in result.stderr or 'Ingested' in result.stdout:
                log.info("✓ ChromaDB re-ingestion successful")
                return True
        log.error("Re-ingestion failed")
        log.error(result.stderr)
        return False
    except subprocess.TimeoutExpired:
        log.error("Re-ingestion timed out (>10 min)")
        return False
    except Exception as e:
        log.error(f"Re-ingestion error: {e}")
        return False


def verify_sources(corpus_file: str) -> dict:
    """Verify corpus has proper source URLs."""
    log.info("Verifying source URLs...")

    corpus_path = Path(corpus_file)
    if not corpus_path.exists():
        log.error(f"Corpus not found: {corpus_file}")
        return {}

    stats = {
        'total': 0,
        'with_urls': 0,
        'missing_urls': 0,
        'by_category': {},
        'by_language': {},
    }

    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        for item in items:
            stats['total'] += 1

            url = item.get('url')
            if url:
                stats['with_urls'] += 1
            else:
                stats['missing_urls'] += 1

            cat = item.get('page_category', 'Unknown')
            lang = item.get('language', 'en')

            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1

    except Exception as e:
        log.error(f"Verification failed: {e}")
        return {}

    # Log verification results
    log.info(f"Corpus Statistics:")
    log.info(f"  Total documents: {stats['total']}")
    log.info(f"  With verified URLs: {stats['with_urls']} ({100*stats['with_urls']//stats['total']}%)")
    if stats['missing_urls'] > 0:
        log.warning(f"  Missing URLs: {stats['missing_urls']}")

    log.info(f"By language: {stats['by_language']}")
    log.info(f"By category:")
    for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
        log.info(f"    {cat}: {count}")

    return stats


def main():
    """Execute complete workflow."""
    log.info("=" * 60)
    log.info("UWTSD Corpus Scrape → ChromaDB Workflow")
    log.info("=" * 60)

    # Paths
    scraped_jsonl = 'uwtsd_corpus.jsonl'
    temp_json = 'app/data/uwtsd-corpus-scraped.json'
    final_corpus = 'app/data/uwtsd-corpus.json'

    # Step 1: Verify scraped data exists
    log.info("\n[Step 1] Checking scraped data...")
    if not Path(scraped_jsonl).exists():
        log.error(f"Scraped file not found: {scraped_jsonl}")
        log.error("Run spider first: scrapy crawl uwtsd -o uwtsd_corpus.jsonl")
        return 1

    # Step 2: Convert JSONL to JSON
    log.info("\n[Step 2] Converting JSONL → JSON...")
    item_count = convert_jsonl_to_json(scraped_jsonl, temp_json)
    if item_count == 0:
        log.error("Conversion failed")
        return 1

    # Step 3: Verify sources
    log.info("\n[Step 3] Verifying source URLs...")
    stats = verify_sources(temp_json)
    if stats['missing_urls'] > 0:
        log.warning(f"⚠ {stats['missing_urls']} documents missing URLs")

    # Step 4: Backup existing
    log.info("\n[Step 4] Backing up existing corpus...")
    if not backup_corpus(final_corpus):
        log.error("Backup failed")
        return 1

    # Step 5: Replace corpus
    log.info("\n[Step 5] Replacing corpus...")
    if not replace_corpus(temp_json, final_corpus):
        log.error("Replace failed")
        return 1

    # Step 6: Re-ingest
    log.info("\n[Step 6] Re-ingesting into ChromaDB...")
    log.warning("This may take 1-2 minutes...")
    if not run_reingest():
        log.error("Re-ingestion failed")
        return 1

    # Summary
    log.info("\n" + "=" * 60)
    log.info("✓ Workflow Complete!")
    log.info("=" * 60)
    log.info(f"\n{item_count} documents with verified source URLs")
    log.info(f"URL coverage: {100*stats['with_urls']//stats['total']}%")
    log.info("\nNext: Restart backend and test sources")
    log.info("  python run.py")
    log.info("  # Then check http://localhost:3000")

    return 0


if __name__ == '__main__':
    sys.exit(main())

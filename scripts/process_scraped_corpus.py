#!/usr/bin/env python3
"""
Process scraped JSONL corpus into ingestion-ready format.

Converts Scrapy output (JSONL) to:
1. JSON array with all fields
2. Proper metadata structure for ChromaDB
3. Verified source URLs

Usage:
    python scripts/process_scraped_corpus.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def process_scraped_corpus(jsonl_file: str, output_file: str) -> int:
    """
    Convert Scrapy JSONL output to ingestion-ready JSON.

    Args:
        jsonl_file: Path to scraped_corpus.jsonl from Scrapy
        output_file: Path to save processed corpus.json

    Returns:
        Number of documents processed
    """
    jsonl_path = Path(jsonl_file)
    output_path = Path(output_file)

    if not jsonl_path.exists():
        log.error(f"Input file not found: {jsonl_file}")
        return 0

    log.info(f"Reading scraped corpus from {jsonl_file}")

    items = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse line {line_num}: {e}")
                    continue
    except Exception as e:
        log.error(f"Failed to read file: {e}")
        return 0

    log.info(f"Loaded {len(items)} items from JSONL")

    # Process and validate
    processed = []
    urls_seen = set()

    for idx, item in enumerate(items):
        # Extract required fields
        url = item.get('url', '')
        title = item.get('title', '')
        content = item.get('content', '')
        language = item.get('language', 'en')

        # Validate required fields
        if not url:
            log.warning(f"Item {idx}: missing URL, skipping")
            continue

        if not content or len(content.strip()) < 50:
            log.debug(f"Item {idx}: content too short, skipping")
            continue

        # Deduplicate by URL
        if url in urls_seen:
            log.debug(f"Item {idx}: duplicate URL {url}, skipping")
            continue

        urls_seen.add(url)

        # Clean and structure
        processed_item = {
            'url': url,
            'source': url,  # For compatibility with existing code
            'title': title.strip() if title else 'UWTSD Information',
            'content': content.strip(),
            'lang': language,
            'page_category': item.get('page_category', item.get('page_section', 'General')),
            'source_type': item.get('source_type', 'Information Page'),
            'topics': item.get('topics', []),
            'scraped_at': item.get('scraped_at', datetime.now().isoformat()),
        }

        processed.append(processed_item)

    if not processed:
        log.error("No valid items found after processing")
        return 0

    # Write output
    log.info(f"Writing {len(processed)} processed items to {output_file}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    # Summary by category
    categories = {}
    languages = {}
    for doc in processed:
        cat = doc.get('page_category', 'Unknown')
        lang = doc.get('lang', 'en')
        categories[cat] = categories.get(cat, 0) + 1
        languages[lang] = languages.get(lang, 0) + 1

    log.info("Processed corpus summary:")
    log.info(f"  Total documents: {len(processed)}")
    log.info(f"  Languages: {languages}")
    log.info("  By category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        log.info(f"    {cat}: {count}")

    return len(processed)


if __name__ == '__main__':
    import sys

    scraped_file = 'uwtsd_corpus.jsonl'
    output_corpus = 'app/data/uwtsd-corpus-scraped.json'

    # Check if scraped file exists
    if not Path(scraped_file).exists():
        log.error(f"Scraped file not found: {scraped_file}")
        log.error("Please run the Scrapy spider first:")
        log.error("  scrapy crawl uwtsd -o uwtsd_corpus.jsonl")
        sys.exit(1)

    count = process_scraped_corpus(scraped_file, output_corpus)

    if count > 0:
        log.info(f"✓ Successfully processed {count} documents")
        log.info(f"✓ Output saved to {output_corpus}")
        log.info("")
        log.info("Next steps:")
        log.info(f"  1. Backup: cp app/data/uwtsd-corpus.json app/data/uwtsd-corpus-old.json")
        log.info(f"  2. Replace: mv {output_corpus} app/data/uwtsd-corpus.json")
        log.info(f"  3. Re-ingest: python -m scripts.ingest --reset")
        sys.exit(0)
    else:
        log.error("Failed to process scraped corpus")
        sys.exit(1)

#!/usr/bin/env python3
"""Monitor Scrapy spider progress."""

import json
from pathlib import Path
import time

def monitor_scrape(jsonl_file='uwtsd_corpus.jsonl', interval=10):
    """Monitor scraping progress."""
    path = Path(jsonl_file)

    if not path.exists():
        print("Waiting for scraper to start...")
        return

    last_count = 0
    while True:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            count = len([l for l in lines if l.strip()])
            size_mb = path.stat().st_size / (1024*1024)

            if count > last_count:
                print(f"✓ Scraped: {count} items (~{size_mb:.2f}MB)")
                last_count = count

            # Check last line for success
            if lines:
                try:
                    last = json.loads(lines[-1])
                    print(f"  Latest: {last.get('title', 'Unknown')[:60]}")
                except:
                    pass

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

if __name__ == '__main__':
    monitor_scrape()

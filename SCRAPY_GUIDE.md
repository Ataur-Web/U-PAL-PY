# Knowledge Base Reorganization Guide

This guide explains how to use Scrapy to properly organize the U-Pal UWTSD corpus with professional, trustworthy source attribution.

## Problem We're Solving

Previously, sources were displayed with raw internal IDs like "corpus_18" or "courses_general", which looked unprofessional and didn't convince users these were validated sources. This guide reorganizes the corpus to display:

- **Title**: Human-readable page/section title (e.g., "Entry Requirements", "Research Integrity")
- **Category**: Topic category (e.g., "Admissions", "Student Support")
- **Excerpt**: First 150 characters of content for preview
- **URL**: Direct link to the official UWTSD page
- **Language**: Welsh (cy) or English (en)

## Architecture

The system has two paths:

### Path 1: Reorganize Existing Corpus (Recommended for now)

Use `scripts/organize_corpus.py` to enhance your existing `app/data/uwtsd-corpus.json` with better metadata:

```
existing corpus → organize_corpus.py → enriched corpus with titles/categories
                                       ↓
                                    ingest.py
                                       ↓
                                   ChromaDB
                                       ↓
                                   /api/chat displays professional sources
```

### Path 2: Scrape Fresh Corpus (Future)

Use Scrapy to scrape the UWTSD website fresh:

```
UWTSD website → Scrapy spiders → structured JSON items
                                 ↓
                              organize_corpus.py (optional)
                                 ↓
                              ingest.py
                                 ↓
                              ChromaDB
```

## Quick Start (Existing Corpus)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure Scrapy is installed (added for future use):
```bash
pip install scrapy
```

### Step 2: Reorganize Your Existing Corpus

```bash
cd app/data

# Backup original
cp uwtsd-corpus.json uwtsd-corpus-original.json

# Reorganize with enriched metadata
python ../../scripts/organize_corpus.py
```

This creates `uwtsd-corpus-organized.json` with new fields:
- `title`: Inferred from URL
- `category`: Categorized by page type
- `source_type`: FAQ, News, Policy, etc.
- `excerpt`: First 150 chars
- `organized_at`: Timestamp

### Step 3: Verify the Output

```bash
# Check first few items
head -c 2000 uwtsd-corpus-organized.json | python -m json.tool
```

Look for entries like:

```json
{
  "url": "https://www.uwtsd.ac.uk/student-support/",
  "title": "Student Support Services",
  "category": "Student Support",
  "source_type": "Information Page",
  "content": "The Student Support team provides...",
  "excerpt": "The Student Support team provides...",
  "language": "en",
  "topics": ["Wellbeing", "Academic Help"],
  "organized_at": "2024-01-15T10:30:00"
}
```

### Step 4: Replace and Re-Ingest

```bash
# Replace original with organized
mv uwtsd-corpus-organized.json uwtsd-corpus.json

# Go back to project root
cd ../..

# Reset ChromaDB to start fresh
python app/scripts/ingest.py reset

# Re-ingest with new metadata
python app/scripts/ingest.py
```

### Step 5: Test the Chatbot

Start your backend and frontend:

```bash
# Terminal 1: Backend
python app/run.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

Ask a question: **"What student support is available?"**

You should now see sources displayed like:

```
Sources (3)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Student Support Services    [STUDENT SUPPORT]
  The Student Support team provides counselling, wellbeing...
  View full page →
  
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Wellbeing & Counselling     [STUDENT SUPPORT]
  Free counselling services available to all students...
  View full page →
```

Instead of the old unprofessional format:

```
Sources (3)
  CORPUS                      [COURSES_GENERAL]
  Lorem ipsum dolor sit amet consectetur adipiscing...
```

## Advanced: Scraping Fresh Data (Optional)

If you want to scrape fresh UWTSD data with Scrapy:

### Step 1: Install Scrapy (if not already done)

```bash
pip install scrapy
```

### Step 2: Run the Spider

```bash
# Crawl UWTSD and save to uwtsd_corpus.jsonl
scrapy crawl uwtsd -o uwtsd_corpus.jsonl

# Inspect output
head -5 uwtsd_corpus.jsonl | python -m json.tool
```

### Step 3: Convert JSONL to JSON Array

```python
# Convert output format
python -c "
import json

items = []
with open('uwtsd_corpus.jsonl', 'r') as f:
    for line in f:
        items.append(json.loads(line))

with open('app/data/uwtsd-corpus.json', 'w') as f:
    json.dump(items, f, indent=2)

print(f'Converted {len(items)} items')
"
```

### Step 4: Proceed with Steps 2-5 Above

The rest of the workflow is identical.

## Metadata Fields Explained

### Core Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `url` | Official UWTSD page URL | `https://www.uwtsd.ac.uk/study-with-us/entry-requirements/` |
| `title` | Human-readable page title | `"Entry Requirements"` |
| `content` | Full extracted text | `"Typical entry requirements are: A-Levels..."` |
| `language` | Language code | `"en"` or `"cy"` |

### Metadata Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `category` | Topic category | `"Admissions"`, `"Student Support"`, `"Courses"` |
| `source_type` | Type of content | `"FAQ"`, `"Policy Document"`, `"News/Article"` |
| `excerpt` | Preview text (~150 chars) | `"Entry requirements are A-Levels or equivalent..."` |
| `topics` | Keywords/tags | `["Computing", "Engineering", "STEM"]` |
| `organized_at` | Processing timestamp | `"2024-01-15T10:30:00"` |

## How the Frontend Displays Sources

The `/api/chat` response now includes a `sources` array with enhanced metadata:

```python
class SourceRef(BaseModel):
    title: str           # "Student Support Services"
    category: str        # "Student Support"
    excerpt: str         # Preview text
    language: str        # "en" or "cy"
    url: str | None      # https://www.uwtsd.ac.uk/...
```

The frontend renders this in a collapsible `<details>` section with:
- Title and category badge
- Text excerpt
- "View full page" link to the official URL

## Troubleshooting

### "Import file not found: uwtsd-corpus.json"

Ensure you're running the script from the project root:

```bash
python scripts/organize_corpus.py
```

Not from `app/data/`:

```bash
# Wrong ❌
cd app/data && python ../../scripts/organize_corpus.py

# Right ✓
python scripts/organize_corpus.py
```

### "Failed to re-ingest after reorganization"

The ingest script might be caching the old corpus. Clear ChromaDB:

```bash
python app/scripts/ingest.py reset
```

Then re-run ingest:

```bash
python app/scripts/ingest.py
```

### "Sources still show as 'CORPUS' in chat"

This means the reorganized corpus wasn't picked up. Check:

1. Did you replace the original file?
   ```bash
   # Verify the file has the new fields
   grep "title" app/data/uwtsd-corpus.json
   ```

2. Did you reset ChromaDB?
   ```bash
   python app/scripts/ingest.py reset
   ```

3. Did you restart the backend?
   The app caches metadata on startup.

## Next Steps

### For Better Coverage

Add more UWTSD pages to scrape by editing `uwtsd_scraper/spiders/uwtsd_spider.py`:

```python
start_urls = [
    'https://www.uwtsd.ac.uk/study-with-us/',
    'https://www.uwtsd.ac.uk/student-support/',
    'https://www.uwtsd.ac.uk/research/',
    # Add more sections here
]
```

### For Richer Metadata

Customize the inference functions in `scripts/organize_corpus.py`:

- `infer_title_from_url()`: Improve title extraction
- `infer_category()`: Add more category patterns
- `infer_source_type()`: Classify content types better

### For Production

- Set `ROBOTSTXT_OBEY = True` in `uwtsd_scraper/settings.py` (already set)
- Configure `DOWNLOAD_DELAY` to be respectful (already 1 second)
- Monitor crawl with `--loglevel DEBUG`

## Files Modified/Created

**New Files:**
- `scrapy.cfg` — Scrapy config
- `uwtsd_scraper/` — Python package for Scrapy project
  - `settings.py` — Scrapy settings
  - `items.py` — Item definitions
  - `pipelines.py` — Data processing pipelines
  - `spiders/uwtsd_spider.py` — Main spider
- `scripts/organize_corpus.py` — Corpus reorganization script

**Modified Files:**
- `app/routes/chat.py` — Updated `SourceRef` model with title, category, url
- `frontend/pages/index.js` — Updated sources rendering
- `frontend/styles/globals.css` — Enhanced source styling
- `tests/test_sources.py` — Updated source field tests
- `requirements.txt` — Added Scrapy

## Performance Notes

- **Scrapy crawl**: ~5-10 min for 500 pages (with 1s delay)
- **Reorganization**: <1s for 4,000 docs
- **Re-ingestion**: ~30s-2min depending on embedding model
- **Frontend rendering**: Unchanged (still collapsible `<details>`)

## Testing

Run the source tests to verify:

```bash
pytest tests/test_sources.py -v
```

Should output:

```
test_sources.py::test_chat_sources_shape PASSED
test_sources.py::test_chat_sources_excerpt_length PASSED
```

## Questions?

Refer to:
- Scrapy docs: https://docs.scrapy.org/
- ChromaDB docs: https://docs.trychroma.com/
- FastAPI/Pydantic: https://fastapi.tiangolo.com/

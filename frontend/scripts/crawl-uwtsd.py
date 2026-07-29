#!/usr/bin/env python3
"""
crawl-uwtsd.py — Recursive UWTSD website crawler + Morphik ingestion

Crawls all pages on uwtsd.ac.uk (English and Welsh), extracts clean text,
and ingests each page into Morphik for RAG retrieval.

Usage (on Oracle VM):
    python3 crawl-uwtsd.py

Requirements (standard library only — no pip needed):
    Python 3.8+

Config (edit below or pass as env vars):
    MORPHIK_URL  — defaults to http://localhost:8000
    MAX_PAGES    — max pages to crawl per domain (default 400)
"""

import sys
import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser
from collections import deque

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import os

MORPHIK_URL = os.environ.get("MORPHIK_URL", "http://localhost:8000")
MAX_PAGES   = int(os.environ.get("MAX_PAGES", "400"))
DELAY       = 0.4   # seconds between requests (be polite)
TIMEOUT     = 15    # request timeout in seconds

# Seed URLs — English + Welsh entry points
SEEDS = [
    # English
    "https://www.uwtsd.ac.uk",
    "https://www.uwtsd.ac.uk/undergraduate/",
    "https://www.uwtsd.ac.uk/postgraduate/",
    "https://www.uwtsd.ac.uk/how-to-apply/",
    "https://www.uwtsd.ac.uk/fees-and-funding/",
    "https://www.uwtsd.ac.uk/accommodation/",
    "https://www.uwtsd.ac.uk/student-support/",
    "https://www.uwtsd.ac.uk/it-services/",
    "https://www.uwtsd.ac.uk/swansea/",
    "https://www.uwtsd.ac.uk/carmarthen/",
    "https://www.uwtsd.ac.uk/lampeter/",
    "https://www.uwtsd.ac.uk/international/",
    "https://www.uwtsd.ac.uk/contact/",
    "https://library.uwtsd.ac.uk",
    # Welsh — UWTSD has a full Welsh version under /cymraeg/ or /cy/
    "https://www.uwtsd.ac.uk/cy/",
    "https://www.uwtsd.ac.uk/cymraeg/",
    "https://www.uwtsd.ac.uk/cy/israddedig/",
    "https://www.uwtsd.ac.uk/cy/ol-raddedig/",
    "https://www.uwtsd.ac.uk/cy/sut-i-ymgeisio/",
    "https://www.uwtsd.ac.uk/cy/ffioedd-ac-ariannu/",
    "https://www.uwtsd.ac.uk/cy/llety/",
    "https://www.uwtsd.ac.uk/cy/cymorth-i-fyfyrwyr/",
    "https://www.uwtsd.ac.uk/cy/gwasanaethau-tg/",
    "https://www.uwtsd.ac.uk/cy/abertawe/",
    "https://www.uwtsd.ac.uk/cy/caerfyrddin/",
    "https://www.uwtsd.ac.uk/cy/llambed/",
    "https://www.uwtsd.ac.uk/cy/cysylltu/",
]

# Allowed domains to follow links into
ALLOWED_DOMAINS = {
    "www.uwtsd.ac.uk",
    "uwtsd.ac.uk",
    "library.uwtsd.ac.uk",
}

# URL patterns to skip
SKIP_PATTERNS = [
    r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|tar|gz|jpg|jpeg|png|gif|svg|mp4|mp3|avi|mov|wmv|ico|css|js|woff|woff2|ttf|eot)(\?.*)?$",
    r"#",
    r"mailto:",
    r"tel:",
    r"javascript:",
    r"/wp-admin",
    r"/wp-login",
    r"\?.*print=",
    r"\?.*replytocom=",
]


# ---------------------------------------------------------------------------
# HTML parser — extracts text and links
# ---------------------------------------------------------------------------
class PageParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "iframe", "form"}

    def __init__(self, base_url):
        super().__init__()
        self.base_url  = base_url
        self.links     = []
        self.title     = ""
        self.lang      = "en"
        self._skip     = 0
        self._in_title = False
        self._chunks   = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.SKIP_TAGS:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "html" and attrs.get("lang"):
            self.lang = "cy" if attrs["lang"].startswith("cy") else "en"
        if tag == "a" and "href" in attrs:
            href = attrs["href"].strip()
            if href:
                abs_url = urllib.parse.urljoin(self.base_url, href)
                self.links.append(abs_url)

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += text
            return
        if self._skip == 0:
            self._chunks.append(text)

    def get_text(self):
        raw = " ".join(self._chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def should_skip(url):
    for pat in SKIP_PATTERNS:
        if re.search(pat, url, re.IGNORECASE):
            return True
    return False


def in_allowed_domain(url):
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
        return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)
    except Exception:
        return False


def normalise(url):
    """Strip fragments and trailing slashes for deduplication."""
    p = urllib.parse.urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((p.scheme, p.netloc.lower(), path, "", p.query, ""))


def fetch(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "UPal-UWTSD-Crawler/1.0 (dissertation research bot)",
            "Accept-Language": "en-GB,cy;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            content_type = r.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None, None
            raw = r.read()
            encoding = "utf-8"
            ct_enc = re.search(r"charset=([\w-]+)", content_type)
            if ct_enc:
                encoding = ct_enc.group(1)
            return raw.decode(encoding, errors="replace"), r.url  # r.url = final URL after redirects
    except Exception as e:
        return None, None


def ingest(title, text, url, lang):
    if len(text) < 80:
        return False
    # Truncate very large pages to avoid Morphik timeout
    text = text[:12000]
    metadata = {
        "source": "uwtsd_web",
        "title":  title or url,
        "url":    url,
        "lang":   lang,
    }
    payload = json.dumps({"content": text, "metadata": metadata}).encode("utf-8")
    req = urllib.request.Request(
        f"{MORPHIK_URL}/ingest/text",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status in (200, 201)
    except urllib.error.HTTPError as e:
        return e.code in (200, 201)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main crawl loop
# ---------------------------------------------------------------------------
def crawl():
    print(f"\n{'='*60}")
    print(" UWTSD Full-Site Crawler + Morphik Ingestion")
    print(f"{'='*60}\n")
    print(f"  Morphik: {MORPHIK_URL}")
    print(f"  Max pages: {MAX_PAGES}")
    print(f"  Seeds: {len(SEEDS)}\n")

    # Check Morphik is up
    try:
        with urllib.request.urlopen(f"{MORPHIK_URL}/health", timeout=6) as r:
            print(f"  Morphik health: OK\n")
    except Exception as e:
        print(f"  ERROR: Morphik not reachable at {MORPHIK_URL}: {e}")
        sys.exit(1)

    visited     = set()
    queue       = deque()
    ok_count    = 0
    skip_count  = 0
    fail_count  = 0

    # Seed the queue
    for url in SEEDS:
        n = normalise(url)
        if n not in visited:
            visited.add(n)
            queue.append(url)

    print(f"{'─'*60}")
    print(f"  {'#':>4}  {'Lang':>4}  {'Chars':>6}  URL")
    print(f"{'─'*60}")

    while queue and (ok_count + skip_count + fail_count) < MAX_PAGES:
        url = queue.popleft()

        if should_skip(url) or not in_allowed_domain(url):
            skip_count += 1
            continue

        html, final_url = fetch(url)
        time.sleep(DELAY)

        if not html:
            fail_count += 1
            continue

        parser = PageParser(final_url or url)
        try:
            parser.feed(html)
        except Exception:
            fail_count += 1
            continue

        text  = parser.get_text()
        title = parser.title.strip() or url
        lang  = parser.lang

        # Detect Welsh by URL path
        parsed_path = urllib.parse.urlparse(url).path
        if "/cy/" in parsed_path or "/cymraeg/" in parsed_path:
            lang = "cy"

        n = ok_count + skip_count + fail_count + 1
        chars = len(text)

        if chars < 80:
            skip_count += 1
            print(f"  {n:>4}  {lang:>4}  {chars:>6}  SKIP (too short) {url[:70]}")
            continue

        success = ingest(title, text, final_url or url, lang)
        if success:
            ok_count += 1
            print(f"  {n:>4}  {lang:>4}  {chars:>6}  OK  {url[:70]}")
        else:
            fail_count += 1
            print(f"  {n:>4}  {lang:>4}  {chars:>6}  FAIL {url[:70]}")

        # Enqueue new links found on this page
        for link in parser.links:
            norm = normalise(link)
            if norm not in visited and in_allowed_domain(link) and not should_skip(link):
                visited.add(norm)
                queue.append(link)

    print(f"\n{'='*60}")
    print(f"  Crawl complete")
    print(f"  Ingested:  {ok_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Failed:    {fail_count}")
    print(f"  Total URLs seen: {len(visited)}")
    print(f"{'='*60}\n")
    print(f"  View documents: curl {MORPHIK_URL}/documents | python3 -m json.tool | head -40")
    print()


if __name__ == "__main__":
    crawl()

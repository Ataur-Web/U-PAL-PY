"""
scrapy item processing pipelines for corpus enrichment and deduplication

these pipelines run after the spider extracts raw page data, enriching with:
- inferred page categories (courses, admissions, research, etc)
- source type classification (faq, policy, news, contact info, etc)
- language detection (en vs cy)
- deduplication by url to prevent double-scraping
"""

import logging
from datetime import datetime
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)


class StructuredMetadataPipeline:
    """Enhance items with structured metadata for better source attribution."""

    def __init__(self):
        self.seen_urls = set()

    def _infer_category(self, url, item):
        """Infer page category from URL path."""
        path = urlparse(url).path.lower()

        if 'course' in path or 'programme' in path or 'degree' in path:
            return 'Courses & Programmes'
        elif 'admission' in path or 'apply' in path or 'entry' in path:
            return 'Admissions'
        elif 'fee' in path or 'tuition' in path or 'cost' in path:
            return 'Fees & Funding'
        elif 'accommodation' in path or 'housing' in path or 'student-accommodation' in path:
            return 'Accommodation'
        elif 'support' in path or 'wellbeing' in path or 'counselling' in path:
            return 'Student Support'
        elif 'library' in path or 'academic' in path or 'learning' in path:
            return 'Academic Services'
        elif 'research' in path or 'ymchwil' in path:
            return 'Research'
        elif 'international' in path or 'exchange' in path:
            return 'International'
        elif 'career' in path or 'graduate' in path or 'employment' in path:
            return 'Careers'
        else:
            return 'General Information'

    def _infer_source_type(self, url, item):
        """Infer content type from URL patterns."""
        path = urlparse(url).path.lower()

        if '/faq' in path or '/questions' in path:
            return 'FAQ'
        elif '/news' in path or '/blog' in path or '/article' in path:
            return 'News'
        elif '/policy' in path or '/regulation' in path or '/code' in path:
            return 'Policy'
        elif '/contact' in path or '/location' in path:
            return 'Contact Information'
        else:
            return 'Course/Program Information'

    def process_item(self, item, spider):
        """Enrich item with inferred metadata."""
        url = item.get('url', '')

        # Set standard metadata if not already present
        if not item.get('page_category'):
            item['page_category'] = self._infer_category(url, item)

        if not item.get('source_type'):
            item['source_type'] = self._infer_source_type(url, item)

        # Ensure language is set
        if not item.get('language'):
            # Default to English, but detect Welsh from URL path
            item['language'] = 'cy' if '/cy/' in url else 'en'

        # Set scrape timestamp
        item['scraped_at'] = datetime.now().isoformat()

        # Ensure title is present
        if not item.get('title'):
            item['title'] = item.get('page_section', 'UWTSD Information')

        logger.info(
            'Scraped: %s [%s] (%s)',
            item['title'],
            item['page_category'],
            item['language']
        )

        return item


class DuplicateFilterPipeline:
    """Filter duplicate content by URL."""

    def __init__(self):
        self.seen_urls = set()
        self.duplicate_count = 0

    def process_item(self, item, spider):
        """Skip items we've already scraped."""
        url = item.get('url', '')
        if url in self.seen_urls:
            logger.debug('Skipping duplicate: %s', url)
            self.duplicate_count += 1
            raise DropItem(f'Duplicate URL: {url}')
        else:
            self.seen_urls.add(url)
            return item

    def close_spider(self, spider):
        """Log summary at end of spider."""
        if self.duplicate_count > 0:
            logger.info(
                'DuplicateFilterPipeline: Filtered %d duplicates',
                self.duplicate_count
            )


class DropItem(Exception):
    """Exception to signal that an item should be dropped."""
    pass

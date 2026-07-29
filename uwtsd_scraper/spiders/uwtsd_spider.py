import scrapy
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime


class UWTSDSpider(scrapy.Spider):
    # web scraper for uwtsd.ac.uk - crawls 336+ pages to build knowledge corpus
    # extracts structured content from both english and welsh versions
    # respects robots.txt and uses rate limiting (1s delay) to avoid overload
    # each page extracted with title, category, language, and source url

    name = 'uwtsd'
    allowed_domains = ['uwtsd.ac.uk']

    # seed urls for major content sections across 8 landing pages
    start_urls = [
        'https://www.uwtsd.ac.uk/study-with-us/',
        'https://www.uwtsd.ac.uk/student-support/',
        'https://www.uwtsd.ac.uk/student-life/',
        'https://www.uwtsd.ac.uk/international/',
        'https://www.uwtsd.ac.uk/research/',
        'https://www.uwtsd.ac.uk/cy/astudiaethu-gyda-ni/',
        'https://www.uwtsd.ac.uk/cy/cymorth-myfyrwyr/',
        'https://www.uwtsd.ac.uk/cy/bywyd-myfyrwyr/',
    ]

    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,
        'USER_AGENT': 'U-PAL UWTSD Corpus Scraper',
        'CONCURRENT_REQUESTS': 4,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_urls = set()

    def parse(self, response):
        """Parse landing page and follow links to content."""
        # Extract main page content if it exists
        yield from self._extract_page_content(response)

        # Follow links to subpages (courses, FAQs, support pages, etc.)
        # Limit to reasonable depth and domains
        for link in response.css('a::attr(href)'):
            if not link:
                continue

            # Normalize URL
            url = urljoin(response.url, link.extract())
            parsed = urlparse(url)

            # Only follow UWTSD URLs
            if 'uwtsd.ac.uk' not in parsed.netloc:
                continue

            # Skip common non-content URLs
            if any(skip in parsed.path.lower() for skip in [
                '/admin', '/search', '/tag/', '/category/', '/feed', '.pdf',
                '/redirect', '/login', '/register', '/logout'
            ]):
                continue

            # Deduplicate and respect URL limits
            if url not in self.seen_urls:
                self.seen_urls.add(url)
                if len(self.seen_urls) <= 500:  # Reasonable crawl depth
                    yield scrapy.Request(
                        url,
                        callback=self.parse_content_page,
                        meta={'source_url': response.url}
                    )

    def parse_content_page(self, response):
        """Extract content from an individual page."""
        yield from self._extract_page_content(response)

    def _extract_page_content(self, response):
        """Extract structured content from a page."""
        try:
            url = response.url

            # Extract title (use page title or h1)
            title = response.xpath('//title/text()').get('').strip()
            if not title:
                title = response.xpath('//h1/text()').get('').strip()
            if not title:
                title = urlparse(url).path.replace('/', ' ').strip()

            # Clean up title
            title = self._clean_text(title)

            # Extract main content - try multiple selectors for robustness
            content_selectors = [
                '//main//text()',
                '//article//text()',
                '//*[@class*="content"]//text()',
                '//*[@class*="body"]//text()',
            ]

            content_texts = []
            for selector in content_selectors:
                try:
                    texts = response.xpath(selector).getall()
                    if texts:
                        content_texts = texts
                        break
                except Exception as e:
                    self.logger.debug(f"Selector {selector} failed: {e}")
                    continue

            # Clean and join content
            content = self._extract_text_content(content_texts)

            # Skip pages with minimal content
            if not content or len(content.strip()) < 100:
                return

            # Extract page sections and description
            page_section = self._infer_section(response.url)
            description = self._extract_description(content)

            # Detect language
            language = 'cy' if '/cy/' in url else 'en'

            yield {
                'url': url,
                'title': title[:200],  # Cap title length
                'description': description,
                'content': content,
                'page_section': page_section,
                'language': language,
                'scraped_at': datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error extracting page content from {response.url}: {e}")

    def _extract_text_content(self, texts):
        """Extract and clean text content from list of text fragments."""
        cleaned = []
        try:
            for text in texts:
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if text and len(text) > 2:  # Skip very short fragments
                    cleaned.append(text)
        except Exception as e:
            self.logger.warning(f"Error extracting text: {e}")
            pass

        # Join with spaces, clean up whitespace
        content = ' '.join(cleaned)
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        return content[:5000]  # Cap content length for reasonable chunks

    def _clean_text(self, text):
        """Clean up text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common website chrome
        text = text.replace(' | UWTSD', '').replace('UWTSD', '').strip()
        return text

    def _extract_description(self, content):
        """Extract first sentence as description."""
        # Take first sentence (up to 150 chars)
        sentences = content.split('.')
        if sentences:
            desc = sentences[0].strip()
            return desc[:150]
        return content[:150]

    def _infer_section(self, url):
        """Infer page section from URL path."""
        path = url.split('uwtsd.ac.uk')[-1].lower()

        if 'study' in path or 'course' in path or 'programme' in path:
            return 'Study at UWTSD'
        elif 'student' in path and 'support' in path:
            return 'Student Support'
        elif 'student' in path or 'life' in path:
            return 'Student Life'
        elif 'international' in path:
            return 'International'
        elif 'research' in path:
            return 'Research'
        elif 'news' in path or 'blog' in path:
            return 'News & Events'
        else:
            return 'Information'

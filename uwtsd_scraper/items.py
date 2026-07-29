import scrapy


class UWTSDCorpusItem(scrapy.Item):
    """A structured piece of UWTSD content with rich metadata."""

    # Core content
    url = scrapy.Field()
    title = scrapy.Field()  # Page/section title
    description = scrapy.Field()  # Short description or summary
    content = scrapy.Field()  # Full extracted text

    # Metadata
    page_category = scrapy.Field()  # e.g., "Courses", "Student Support", "Research"
    page_section = scrapy.Field()  # e.g., "Undergraduate", "Postgraduate"
    language = scrapy.Field()  # "en" or "cy"
    source_type = scrapy.Field()  # "course_info", "faq", "policy", "news", etc.

    # For chunking/retrieval
    chunk_index = scrapy.Field()  # If content is split into chunks
    keywords = scrapy.Field()  # List of keywords

    # Timestamps
    scraped_at = scrapy.Field()
    last_updated = scrapy.Field()  # If available on page

BOT_NAME = "uwtsd_scraper"

SPIDER_MODULES = ["uwtsd_scraper.spiders"]
NEWSPIDER_MODULE = "uwtsd_scraper.spiders"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Set download delay to be respectful
DOWNLOAD_DELAY = 1

# User agent
USER_AGENT = "U-PAL UWTSD Chatbot Corpus Scraper"

# Concurrent requests
CONCURRENT_REQUESTS = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Disable cookies
COOKIES_ENABLED = False

# Pipelines
ITEM_PIPELINES = {
    'uwtsd_scraper.pipelines.StructuredMetadataPipeline': 300,
    'uwtsd_scraper.pipelines.DuplicateFilterPipeline': 400,
}

# Output settings
FEED_FORMAT = 'json'
FEED_URI = 'uwtsd_corpus.jsonl'

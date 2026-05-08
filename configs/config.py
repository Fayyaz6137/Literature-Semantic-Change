DECADES = {
    '1800s': (1800, 1820),
    '1820s': (1820, 1840),
    '1840s': (1840, 1860),
    '1860s': (1860, 1880),
    '1880s': (1880, 1900),
    '1900s': (1900, 1920),
    '1920s': (1920, 1940),
    '1940s': (1940, 1960),
    '1960s': (1960, 1980),
}

MAX_BOOKS_PER_DECADE = 50  # how many books to download per period 150
DELAY = 0.5  # seconds between requests (be polite)
MIN_LENGTH = 5000  # skip files shorter than this (bad downloads)
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; academic-nlp-project/1.0)'}
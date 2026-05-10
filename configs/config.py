import os

# ---------------------------- PARAMS ------------------------------- #
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
DECADES_LIST = ['1800s', '1820s', '1840s', '1860s', '1880s',
                '1900s', '1920s', '1940s', '1960s']

# TARGET_WORDS = ['awful', 'nice', 'gay', 'pioneer', 'villain', 'heroine', 'fall', 'seeds', 'matter']

# TARGET_WORDS = ['champions', 'weep','pretty', 'rough','oppression','servant','battle','highlands']


TARGET_WORDS = [
    'awful',
    'gay',
    'nice',
    'broadcast',
    'virtue',
    'romance',
    'industrial',
    'wireless'
]

# ---------------------------- SWITCHES ------------------------------- #
DEBUG_SWITCH = 0  # DEBUG SWITCH
MAX_BOOKS_PER_DECADE = 6 if DEBUG_SWITCH else 200  # how many books to download per period 150
DELAY = 0.5  # seconds between requests (be polite)
MIN_LENGTH = 5000  # skip files shorter than this (bad downloads)
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; academic-nlp-project/1.0)'}

# ---------------------------- PATHS ------------------------------- #

# --- CATALOG ---
DATA_DIR = 'data'
CATALOG_DIR = os.path.join(DATA_DIR, 'catalog')
CATALOG_PATH = os.path.join(CATALOG_DIR, 'pg_catalog.csv')

# --- DATA ---
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
ENG_BOOKS_EXCEL = 'output.xlsx'
BOOKS_JSON_PATH = os.path.join(DATA_DIR, 'catalog', 'book_ids_by_decade.json')
DATA_RAW_TEXT_PATH = os.path.join(DATA_RAW_DIR, 'lit_all.txt')
DATA_CLEAN_TEXT_PATH = os.path.join(DATA_PROCESSED_DIR, 'lit_all_clean.txt')

# --- MODELS ---
MODELS_DIR = 'models'
MODEL_WORD2VE_PATH = os.path.join(MODELS_DIR, 'word2vec.model')
MODEL_COMPASS_PATH = os.path.join(MODELS_DIR, 'cad_compass.model')
MODEL_CAD_SLICES_DIR = os.path.join(MODELS_DIR, 'cad_slices')

# --- RESULTS ---
RESULTS_DIR = 'results'
CHANGE_SCORE_PATH = os.path.join(RESULTS_DIR, 'all_change_scores.csv')
DRIFT_TIMELINE_PATH = os.path.join(RESULTS_DIR, 'drift_timeline.csv')
FIGURES_DIR = os.path.join('results', 'figures')

# ---------------------------- WEAT WORD SETS ------------------------------- #.
AGENCY_TRAITS = [
    'brave', 'strong', 'rational', 'bold', 'independent',
    'intelligent', 'ambitious', 'courageous',
]
COMMUNAL_TRAITS = [
    'gentle', 'beautiful', 'emotional', 'frail', 'delicate',
    'modest', 'tender', 'meek',
]
MALE_NAMES = [
    'henry', 'william', 'edward', 'charles',
    'george', 'thomas', 'arthur', 'james',
]
FEMALE_NAMES = [
    'mary', 'elizabeth', 'jane', 'catherine',
    'emma', 'agnes', 'clara', 'harriet',
]

N_PERMUTATIONS = 1000  # for permutation p-value

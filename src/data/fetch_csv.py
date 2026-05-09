import pandas as pd
import urllib.request
import os

import requests
from configs.config import HEADERS, CATALOG_PATH, CATALOG_DIR


# ── Step 1: Download the catalog ──────────────────────────────────────
def get_catalog():
    os.makedirs(CATALOG_DIR, exist_ok=True)
    if not os.path.exists(CATALOG_PATH):
        print("Downloading Gutenberg catalog (~3 MB)...")
        r = requests.get(
            'https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv',
            headers=HEADERS, timeout=60, stream=True
        )
        r.raise_for_status()
        with open(CATALOG_PATH, 'wb') as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print("Catalog saved.\n")
    else:
        print("Catalog already downloaded.\n")

    df = pd.read_csv(CATALOG_PATH, low_memory=False)
    print(f"Catalog loaded: {len(df):,} total entries")
    return df
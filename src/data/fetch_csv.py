import pandas as pd
import urllib.request
import os

import requests
from configs.config import HEADERS




# ── Step 1: Download the catalog ──────────────────────────────────────
def get_catalog():
    os.makedirs('data/catalog', exist_ok=True)
    path = 'data/catalog/pg_catalog.csv'
    if not os.path.exists(path):
        print("Downloading Gutenberg catalog (~3 MB)...")
        r = requests.get(
            'https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv',
            headers=HEADERS, timeout=60, stream=True
        )
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print("Catalog saved.\n")
    else:
        print("Catalog already downloaded.\n")

    df = pd.read_csv(path, low_memory=False)
    print(f"Catalog loaded: {len(df):,} total entries")
    return df

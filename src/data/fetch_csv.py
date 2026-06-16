import os
import time
import pandas as pd
import requests

from configs.config import HEADERS, CATALOG_PATH, CATALOG_DIR


URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"


def download_catalog():

    temp = CATALOG_PATH + ".part"

    for attempt in range(5):

        try:

            print(f"Download attempt {attempt + 1}/5")

            with requests.get(
                URL,
                headers=HEADERS,
                stream=True,
                timeout=(10, 300)
            ) as r:

                r.raise_for_status()

                with open(temp, "wb") as f:

                    for chunk in r.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if chunk:
                            f.write(chunk)

            os.replace(temp, CATALOG_PATH)

            return

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        ) as e:

            print(f"Download interrupted: {e}")

            if os.path.exists(temp):
                os.remove(temp)

            if attempt == 4:
                raise

            wait = 2 ** attempt
            print(f"Retrying in {wait}s...\n")

            time.sleep(wait)


def get_catalog():

    os.makedirs(CATALOG_DIR, exist_ok=True)

    if not os.path.exists(CATALOG_PATH):

        print("Downloading Gutenberg catalog (~21 MB)...")

        download_catalog()

        print("Catalog saved.\n")

    else:
        print("Catalog already downloaded.\n")

    df = pd.read_csv(
        CATALOG_PATH,
        low_memory=False
    )

    print(f"Catalog loaded: {len(df):,} total entries")

    return df
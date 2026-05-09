import re
import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from configs.config import DECADES, MAX_BOOKS_PER_DECADE, DELAY, MIN_LENGTH, HEADERS, DATA_RAW_DIR, ENG_BOOKS_EXCEL, \
    BOOKS_JSON_PATH, DATA_RAW_TEXT_PATH


def get_corpus_data(df):
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    book_ids = build_book_ids(df)
    download_all(book_ids)
    combine()
    print_summary()


# ── Step 2: Get estimated publication year from author lifespan ────────
def pub_year(authors_str):
    """
    Extract birth and death years from strings like "Austen, Jane, 1775-1817"
    and return the midpoint as the estimated publication year.
    """
    if pd.isna(authors_str):
        return None

    years = [int(y) for y in re.findall(r'\b(\d{4})\b', str(authors_str))
             if 1700 < int(y) < 2000]

    if len(years) >= 2:
        return int((years[0] + years[1]) / 2) + 10  # midpoint of birth-death
    if len(years) == 1:
        return years[0] + 28  # birth year + 30
    return None


# ── Step 3: Build list of book IDs per decade ──────────────────────────

def build_book_ids(df):
    # Keep only English plain-text books
    df = df[(df['Language'] == 'en') & (df['Type'] == 'Text')].copy()
    print(f"English text entries: {len(df):,}\n")

    # Add estimated publication year  <- THE FIX

    df['pub_year'] = df['Authors'].apply(pub_year)

    df = df.dropna(subset=['pub_year'])
    df['pub_year'] = df['pub_year'].astype(int)
    df.to_excel(ENG_BOOKS_EXCEL, index=False)

    book_ids = {}
    for decade, (start, end) in DECADES.items():
        ids = df[(df['pub_year'] >= start) & (df['pub_year'] < end)]['Text#'] \
            .astype(int).tolist()
        # shuffle so we get variety, not just alphabetical order
        import random
        random.seed(42)
        random.shuffle(ids)
        book_ids[decade] = ids[:MAX_BOOKS_PER_DECADE]
        print(f"  {decade} ({start}-{end}): {len(book_ids[decade])} books queued")

    with open(BOOKS_JSON_PATH, 'w') as f:
        json.dump(book_ids, f, indent=2)

    print("\nBook IDs saved to data/catalog/book_ids_by_decade.json")
    return book_ids


# ── Step 4: Download one book ──────────────────────────────────────────
def download_book(book_id):
    """Try multiple URL patterns for one Gutenberg book."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and len(r.text) > MIN_LENGTH:
                return r.text
        except requests.RequestException:
            pass
    return None


# ── Step 5: Download all decades ──────────────────────────────────────
def download_all(book_ids):
    for decade, ids in book_ids.items():
        out_path = f'{DATA_RAW_DIR}/lit_{decade}.txt'

        # Skip if already done
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"\n{decade}: already exists ({size_mb:.1f} MB) — skipping.")
            continue

        print(f"\n{'─' * 50}")
        print(f"Downloading {decade} ({len(ids)} books)...")
        print(f"{'─' * 50}")

        texts = []
        ok = 0
        for book_id in tqdm(ids, desc=f"  {decade}", unit="book"):
            text = download_book(book_id)
            if text:
                texts.append(text)
                ok += 1
            time.sleep(DELAY)

        with open(out_path, 'a', encoding='utf-8', errors='ignore') as f:
            f.write('\n\n'.join(texts))

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  ok {ok}/{len(ids)} books -> {size_mb:.1f} MB saved")


# ── Step 6: Combine into one file for the CADE compass ────────────────
def combine():
    out_path = 'data/raw/lit_all.txt'
    print(f"\nCombining all decades into {DATA_RAW_TEXT_PATH}...")
    with open(out_path, 'a', encoding='utf-8', errors='ignore') as out:
        for decade in DECADES:
            src = f'{DATA_RAW_DIR}/lit_{decade}.txt'
            if os.path.exists(src) and os.path.getsize(src) > 1000:
                with open(src, 'r', encoding='utf-8', errors='ignore') as f:
                    out.write(f.read())
                    out.write('\n\n')
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"lit_all.txt created: {size_mb:.1f} MB")


# ── Summary ───────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 50)
    print("  DOWNLOAD SUMMARY")
    print("=" * 50)
    for decade in DECADES:
        path = f'{DATA_RAW_DIR}/lit_{decade}.txt'
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1024 / 1024
            flag = "OK" if mb > 0.1 else "EMPTY - rerun script"
            print(f"  {decade}: {mb:6.1f} MB  {flag}")
        else:
            print(f"  {decade}: NOT FOUND")
    print("=" * 50)

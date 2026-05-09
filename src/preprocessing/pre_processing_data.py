import os
import re
# import nltk
# from nltk.tokenize import word_tokenize

from configs.config import DECADES, DECADES_LIST, DATA_PROCESSED_DIR, DATA_CLEAN_TEXT_PATH, DATA_RAW_DIR

# Gutenberg header/footer sentinel strings
HEADER_END = [
    "*** START OF THE PROJECT GUTENBERG",
    "***START OF THE PROJECT GUTENBERG",
    "*END*THE SMALL PRINT",
    "This etext was produced",
]
FOOTER_START = [
    "*** END OF THE PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of Project Gutenberg",
]


def strip_gutenberg_boilerplate(text):
    """Remove legal header and footer from a Gutenberg book."""
    # Find start of actual content
    start_idx = 0
    for marker in HEADER_END:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.find('\n', idx) + 1
            break
    # Find end of actual content
    end_idx = len(text)
    for marker in FOOTER_START:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
    return text[start_idx:end_idx]


def preprocess_text(text):
    """Clean and tokenise literary text. Returns list of lowercase tokens."""
    # 1. Strip Gutenberg boilerplate
    text = strip_gutenberg_boilerplate(text)

    # 2. Lowercase everything
    text = text.lower()

    # 3. Remove Roman numerals and chapter headings (e.g. CHAPTER I, II, III)
    text = re.sub(r'\bchapter\s+[ivxlcdm]+\b', '', text)
    text = re.sub(r'\b[ivxlcdm]{2,}\b', '', text)

    # 4. Remove numbers and digits
    text = re.sub(r'\d+', '', text)

    # 5. Remove punctuation and special characters (keep only letters)
    text = re.sub(r'[^a-z\s]', '', text)

    # 6. Tokenise by splitting on whitespace
    tokens = text.split()

    # 7. Remove very short words (1-2 chars, usually noise)
    tokens = [t for t in tokens if len(t) > 2]

    # For distributional semantics, we do NOT remove stopwords.
    # Stopwords provide crucial co-occurrence context for Word2Vec.
    # Removing them only for WEAT word sets, won't remove during embedding training.

    return tokens


def combine(output_path):
    # out_path = 'data/raw/lit_all.txt'
    print(f"\nCombining all decades into {output_path}...")
    with open(output_path, 'a', encoding='utf-8', errors='ignore') as out:
        for decade in DECADES:
            src = f'{DATA_PROCESSED_DIR}/lit_{decade}_clean.txt'
            if os.path.exists(src) and os.path.getsize(src) > 1000:
                with open(src, 'r', encoding='utf-8', errors='ignore') as f:
                    out.write(f.read())
                    out.write('\n\n')
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"lit_all_clean.txt created: {size_mb:.1f} MB")


def preprocess_file(input_path, output_path, chunk_size=20):
    """Process a decade corpus file, writing one sentence-chunk per line."""
    print(f"Processing {input_path}...")
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    tokens = preprocess_text(text)
    # Split into chunks of ~20 tokens (pseudo-sentences)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    lines = [' '.join(c) for c in chunks if len(c) > 10]

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"  → {len(tokens):,} tokens → {len(lines):,} chunks saved to {output_path}")


def pre_process_data():
    # Run preprocessing for all decades

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    for decade in DECADES_LIST:
        preprocess_file(
            f'{DATA_RAW_DIR}/lit_{decade}.txt',
            f'{DATA_PROCESSED_DIR}/lit_{decade}_clean.txt'
        )

    combine(DATA_CLEAN_TEXT_PATH)

    # Also process the combined "all decades" file for the compass
    # preprocess_file('data/raw/lit_all.txt', 'data/processed/lit_all_clean.txt')

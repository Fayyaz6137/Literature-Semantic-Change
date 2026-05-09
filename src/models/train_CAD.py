from cade.cade import CADE
import os
import re
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.utils import RULE_KEEP, RULE_DISCARD

from configs.config import DATA_CLEAN_TEXT_PATH, MODEL_COMPASS_PATH, DECADES_LIST, MODEL_CAD_SLICES_DIR, \
    DATA_PROCESSED_DIR

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING  # set to INFO for verbose gensim output
)

# ── Hyperparameters ────────────────────────────────────────────────────
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 10  # words below this frequency get no vector
SG = 1  # 1 = Skip-gram (better for rare/archaic words)
WORKERS = 4
SEED = 42
COMPASS_EPOCHS = 10  # more epochs on compass = better reference space
SLICE_EPOCHS = 5  # fewer epochs on slices (we're fine-tuning)


def train_compass():
    compass_text = DATA_CLEAN_TEXT_PATH
    if not os.path.exists(compass_text):
        raise FileNotFoundError(
            f'Combined corpus not found at {compass_text}\n'
            'Run 01_download_corpus.py and 01_preprocessing.py first.'
        )

    print('=' * 55)
    print('Training COMPASS on full combined corpus ...')
    print(f'  File: {compass_text}')
    print(f'  vector_size={VECTOR_SIZE}, window={WINDOW}, '
          f'min_count={MIN_COUNT}, epochs={COMPASS_EPOCHS}')
    print('=' * 55)

    compass = Word2Vec(
        corpus_file=compass_text,  # memory-efficient streaming
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        epochs=COMPASS_EPOCHS,
        workers=WORKERS,
        seed=SEED,
    )

    compass.save(MODEL_COMPASS_PATH)
    print(f'\n✓ Compass saved  →  {MODEL_COMPASS_PATH}')
    print(f'  Vocabulary: {len(compass.wv):,} words')
    return compass


def train_slice(decade, compass):
    save_path = os.path.join(MODEL_CAD_SLICES_DIR, f'slice_{decade}.model')

    slice_text = os.path.join(DATA_PROCESSED_DIR, f'lit_{decade}_clean.txt')
    if not os.path.exists(slice_text):
        print(f'  {decade}: ⚠ file not found — skipping ({slice_text})')
        return None

    print(f'\n  Training slice: {decade} ...')

    # ── 1. Load sentences ────────────────────────────────────────────
    sentences = list(LineSentence(slice_text))
    if not sentences:
        print(f'  {decade}: ⚠ empty file — skipping')
        return None

    # ── 2. Build a new model shell ───────────────────────────────────
    compass_vocab = set(compass.wv.key_to_index.keys())

    def trim_rule(word, count, min_count):
        """Keep a word only if the compass knows it."""
        return RULE_KEEP if word in compass_vocab else RULE_DISCARD

    slice_m = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        epochs=SLICE_EPOCHS,
        workers=WORKERS,
        seed=SEED,
    )

    # Build vocabulary using the trim rule so every word
    # in the slice model also exists in the compass
    slice_m.build_vocab(sentences, trim_rule=trim_rule)

    if len(slice_m.wv) == 0:
        print(f'  {decade}: ⚠ no shared vocabulary with compass — skipping')
        return None

    # ── 3. Initialise word vectors from the compass  ←  THE KEY STEP ─
    for word in slice_m.wv.index_to_key:
        if word in compass.wv:
            slice_m.wv[word] = compass.wv[word].copy()

    # ── 4. Train (fine-tune) on this decade's text ───────────────────
    slice_m.train(
        sentences,
        total_examples=slice_m.corpus_count,
        epochs=SLICE_EPOCHS,
    )

    slice_m.save(save_path)

    n_words = sum(len(s) for s in sentences)
    print(f'  ✓ {decade}: {len(slice_m.wv):,} words | '
          f'{n_words:,} tokens | saved → {save_path}')
    return slice_m


def train_slices():
    # Step 1: Compass
    compass = Word2Vec.load(MODEL_COMPASS_PATH)

    # Step 2: One aligned slice per decade
    os.makedirs(MODEL_CAD_SLICES_DIR, exist_ok=True)
    slice_models = {}
    for decade in DECADES_LIST:
        m = train_slice(decade, compass)
        if m is not None:
            slice_models[decade] = m

    # Summary
    print('\n' + '=' * 55)
    print('CADE TRAINING COMPLETE')
    print('=' * 55)
    print(f'Compass:  {len(compass.wv):,} words')
    for decade, m in slice_models.items():
        print(f'{decade}:   {len(m.wv):,} words')

    # Quick sanity check
    target = 'awful'
    print(f'\nSanity check — most similar to "{target}" in each slice:')
    for decade, m in list(slice_models.items())[::3]:  # every 3rd decade
        if target in m.wv:
            nn = [w for w, _ in m.wv.most_similar(target, topn=5)]
            print(f'  {decade}: {nn}')

    print(f'\nNext step: run 08_semantic_change.py')

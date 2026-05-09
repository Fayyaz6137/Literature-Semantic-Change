"""
07_semantic_change.py
=====================
Semantic change analysis using CADE-aligned slice models.

Answers RQ1, RQ2, RQ3:
  - Which words changed meaning most between 1800s and 1960s?
  - For top changers: what exactly shifted (nearest neighbours)?
  - How did target words (awful, nice, gay, pioneer) evolve?

"""

import os
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity

from configs.config import RESULTS_DIR, TARGET_WORDS, DECADES_LIST, MODEL_CAD_SLICES_DIR


# ── Paths ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def load_slices():
    """Load all saved slice models. Skip missing ones."""
    models = {}
    for decade in DECADES_LIST:
        path = os.path.join(MODEL_CAD_SLICES_DIR, f'slice_{decade}.model')
        if os.path.exists(path):
            models[decade] = Word2Vec.load(path)
            print(f'  Loaded {decade}: {len(models[decade].wv):,} words')
        else:
            print(f'  ⚠ {decade}: model not found — skipping')
    return models


def cosine_dist(v1, v2):
    """Cosine distance (1 - cosine similarity) between two 1-D arrays."""
    return 1.0 - float(cosine_similarity(
        v1.reshape(1, -1), v2.reshape(1, -1)
    )[0][0])


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Compute change scores for every word in shared vocabulary
# ══════════════════════════════════════════════════════════════════════

def compute_change_scores(slice_models):
    """
    For every word present in BOTH the earliest and latest slice,
    compute cosine distance between its 1800s and 1960s vectors.

    change_score = 1 - cosine_similarity(v_1800s, v_1960s)

    High score → meaning changed a lot.
    Low score  → meaning stayed stable.
    """
    # Use earliest and latest available slices
    available = [d for d in DECADES_LIST if d in slice_models]
    if len(available) < 2:
        raise RuntimeError('Need at least 2 slice models to compute change scores.')

    early_decade = available[0]
    late_decade = available[-1]
    early_m = slice_models[early_decade]
    late_m = slice_models[late_decade]

    print(f'\nComputing change scores: {early_decade} → {late_decade}')
    print(f'  Early vocab: {len(early_m.wv):,}  |  Late vocab: {len(late_m.wv):,}')

    # Shared vocabulary
    early_vocab = set(early_m.wv.index_to_key)
    late_vocab = set(late_m.wv.index_to_key)
    shared = early_vocab & late_vocab
    print(f'  Shared vocabulary: {len(shared):,} words')

    records = []
    for word in shared:

        # Remove stopwords like "the", "and", etc.
        if word in STOPWORDS:
            continue

        # Remove tiny junk words
        if len(word) < 3:
            continue

        score = cosine_dist(early_m.wv[word], late_m.wv[word])
        records.append({'word': word, 'change_score': round(score, 6)})

    df = pd.DataFrame(records).sort_values('change_score', ascending=False).reset_index(drop=True)
    return df, early_decade, late_decade


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Nearest-neighbour drill-down for top changers and targets
# ══════════════════════════════════════════════════════════════════════

def nearest_neighbours(word, slice_models, topn=8):
    """
    Return nearest neighbours for a word in each decade slice.
    Returns dict: { '1800s': [w1, w2, ...], '1840s': [...], ... }
    """
    result = {}
    for decade, m in slice_models.items():
        if word in m.wv:
            result[decade] = [w for w, _ in m.wv.most_similar(word, topn=topn)]
        else:
            result[decade] = []
    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Cumulative semantic drift timeline (for Plot 2)
# ══════════════════════════════════════════════════════════════════════

def compute_drift_timeline(words, slice_models):
    """
    For each word and each decade, compute cosine distance from
    the EARLIEST slice baseline vector.

    Returns a DataFrame with columns: word, decade, drift
    """
    available = [d for d in DECADES_LIST if d in slice_models]
    if not available:
        return pd.DataFrame()

    baseline_decade = available[0]
    baseline_model = slice_models[baseline_decade]

    records = []
    for word in words:
        if word not in baseline_model.wv:
            continue
        v_base = baseline_model.wv[word]
        for decade in available:
            m = slice_models[decade]
            if word in m.wv:
                drift = cosine_dist(v_base, m.wv[word])
            else:
                drift = None
            records.append({'word': word, 'decade': decade, 'drift': drift})

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_semantic_change_analysis():
    # Load models
    print('\nLoading slice models ...')
    slice_models = load_slices()
    if len(slice_models) < 2:
        print('ERROR: Need at least 2 trained slice models.')
        print('Train CAD Compass and Slices first.')
        return

    # ── Change scores ────────────────────────────────────────────────
    df_scores, early_d, late_d = compute_change_scores(slice_models)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Save full ranking
    out_all = os.path.join(RESULTS_DIR, 'all_change_scores.csv')
    df_scores.to_csv(out_all, index=False)
    print(f'\n✓ Full change scores saved → {out_all}')

    # Print top 20 changed
    print(f'\n{"─" * 55}')
    print(f'TOP 20 MOST CHANGED WORDS ({early_d} → {late_d})')
    print(f'{"─" * 55}')
    print(df_scores.head(20).to_string(index=False))

    # Print top 20 most stable
    print(f'\n{"─" * 55}')
    print('TOP 20 MOST STABLE WORDS')
    print(f'{"─" * 55}')
    print(df_scores.tail(20).iloc[::-1].to_string(index=False))

    # ── Target-word change scores ────────────────────────────────────
    print(f'\n{"─" * 55}')
    print('TARGET WORDS — change scores')
    print(f'{"─" * 55}')
    target_rows = df_scores[df_scores['word'].isin(TARGET_WORDS)]
    if not target_rows.empty:
        print(target_rows.to_string(index=False))
    else:
        print('(None of the target words appear in the shared vocabulary)')

    # ── Nearest-neighbour drill-down ─────────────────────────────────
    print(f'\n{"─" * 55}')
    print('NEAREST-NEIGHBOUR DRILL-DOWN')
    print(f'{"─" * 55}')

    nn_data = {}
    for word in TARGET_WORDS:
        nn = nearest_neighbours(word, slice_models, topn=8)
        nn_data[word] = nn
        if any(nn.values()):
            print(f'\n  "{word}"')
            for decade in [d for d in DECADES_LIST if d in nn and nn[d]]:
                print(f'    {decade}: {nn[decade]}')

    # Save nearest neighbours as JSON
    out_nn = os.path.join(RESULTS_DIR, 'nearest_neighbours.json')
    with open(out_nn, 'w') as f:
        json.dump(nn_data, f, indent=2)
    print(f'\n✓ Nearest neighbours saved → {out_nn}')

    # ── Drift timeline ────────────────────────────────────────────────
    df_drift = compute_drift_timeline(TARGET_WORDS, slice_models)
    if not df_drift.empty:
        out_drift = os.path.join(RESULTS_DIR, 'drift_timeline.csv')
        df_drift.to_csv(out_drift, index=False)
        print(f'✓ Drift timeline saved → {out_drift}')

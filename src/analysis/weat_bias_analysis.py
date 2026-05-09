"""
08_weat_bias.py
===============
WEAT (Word Embedding Association Test) and SWEAT (Sliced WEAT)
gender bias analysis across all CADE-aligned decade models.

Answers RQ4 and RQ5:
  - Are agency adjectives more associated with male names in
    19th-century literary embeddings?
  - Has this gender bias changed from the 1800s to the 1960s?


"""

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine as scipy_cosine

from configs.config import DECADES_LIST, MODEL_CAD_SLICES_DIR, RESULTS_DIR, AGENCY_TRAITS, MALE_NAMES, FEMALE_NAMES, \
    COMMUNAL_TRAITS, N_PERMUTATIONS



# ══════════════════════════════════════════════════════════════════════
# WEAT core functions
# ══════════════════════════════════════════════════════════════════════

def cos_sim(v1, v2):
    """Cosine similarity between two vectors (1 - scipy cosine distance)."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return 1.0 - scipy_cosine(v1, v2)


def mean_sim_to_group(word_vec, group_words, wv):
    """
    Mean cosine similarity of word_vec to all words in group_words
    that exist in the model vocabulary.
    """
    sims = []
    for w in group_words:
        if w in wv:
            sims.append(cos_sim(word_vec, wv[w]))
    return float(np.mean(sims)) if sims else 0.0


def association_score(word, attr1_words, attr2_words, wv):
    """
    s(word, A1, A2) = mean_cos(word, A1) - mean_cos(word, A2)

    Positive → word is closer to A1 (male names).
    Negative → word is closer to A2 (female names).
    """
    if word not in wv:
        return None
    v = wv[word]
    return mean_sim_to_group(v, attr1_words, wv) - \
           mean_sim_to_group(v, attr2_words, wv)


def weat_effect_size(target_words, attr1_words, attr2_words, wv):
    """
    Compute WEAT effect size (Cohen's d) and permutation p-value.

    Effect size d = mean(s) / std(s)
    where s(t) = mean_cos(t, A1) - mean_cos(t, A2) for each target t.

    Permutation test: shuffle attr1 + attr2 labels N times,
    compute mean(s) each time, p = P(permuted >= observed).

    Returns
    -------
    d : float
        Effect size. Positive = targets closer to A1 (male names).
    p : float
        One-tailed permutation p-value.
    n_targets : int
        Number of target words found in vocabulary.
    """
    # Compute association scores for all target words in vocab
    scores = []
    for t in target_words:
        s = association_score(t, attr1_words, attr2_words, wv)
        if s is not None:
            scores.append(s)

    if len(scores) < 2:
        return 0.0, 1.0, len(scores)

    scores = np.array(scores)
    obs_mean = float(np.mean(scores))
    obs_std  = float(np.std(scores, ddof=0))

    d = obs_mean / obs_std if obs_std > 0 else 0.0

    # Permutation test
    all_attrs = attr1_words + attr2_words
    n_a1 = len(attr1_words)
    rng  = np.random.default_rng(42)
    perm_means = []

    for _ in range(N_PERMUTATIONS):
        shuffled = list(all_attrs)
        rng.shuffle(shuffled)
        perm_a1 = shuffled[:n_a1]
        perm_a2 = shuffled[n_a1:]
        perm_scores = []
        for t in target_words:
            s = association_score(t, perm_a1, perm_a2, wv)
            if s is not None:
                perm_scores.append(s)
        if perm_scores:
            perm_means.append(float(np.mean(perm_scores)))

    p = float(np.mean([pm >= obs_mean for pm in perm_means])) if perm_means else 1.0

    return d, p, len(scores)


# ══════════════════════════════════════════════════════════════════════
# SWEAT: run WEAT on every decade slice
# ══════════════════════════════════════════════════════════════════════

def run_sweat(slice_models, target_words, attr1_words, attr2_words, label='agency'):
    """
    Run WEAT on each decade slice and return a DataFrame of results.
    """
    print(f'\nRunning SWEAT for "{label}" targets ...')
    print(f'  Target words: {target_words}')
    print(f'  Attr1 (male names):   {attr1_words}')
    print(f'  Attr2 (female names): {attr2_words}\n')

    rows = []
    for decade in DECADES_LIST:
        if decade not in slice_models:
            continue
        wv = slice_models[decade].wv

        # Check how many target/attr words are in vocab
        t_found = [t for t in target_words if t in wv]
        a1_found = [a for a in attr1_words  if a in wv]
        a2_found = [a for a in attr2_words  if a in wv]

        d, p, n = weat_effect_size(target_words, attr1_words, attr2_words, wv)

        sig = 'Yes' if p < 0.05 else 'No'
        print(f'  {decade}:  d={d:+.4f}  p={p:.4f}  '
              f'sig={sig}  '
              f'(targets found: {len(t_found)}/{len(target_words)}, '
              f'names: {len(a1_found)}/{len(attr1_words)} male, '
              f'{len(a2_found)}/{len(attr2_words)} female)')

        rows.append({
            'decade':        decade,
            f'd_{label}':    round(d, 6),
            f'p_{label}':    round(p, 6),
            'significant':   sig,
            'n_targets':     n,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_weat_sweat_bias_analysis():

    # Load slice models
    print('\nLoading slice models ...')
    slice_models = {}
    for decade in DECADES_LIST:
        path = os.path.join(MODEL_CAD_SLICES_DIR, f'slice_{decade}.model')
        if os.path.exists(path):
            slice_models[decade] = Word2Vec.load(path)
            print(f'  {decade}: {len(slice_models[decade].wv):,} words')
        else:
            print(f'  ⚠ {decade}: not found — skipping')

    if not slice_models:
        print('ERROR: No slice models found. Train CAD Slice Models first.')
        return

    # ── SWEAT 1: Agency traits vs male/female names ──────────────────
    df_agency = run_sweat(
        slice_models,
        target_words = AGENCY_TRAITS,
        attr1_words  = MALE_NAMES,
        attr2_words  = FEMALE_NAMES,
        label        = 'agency',
    )
    out_agency = os.path.join(RESULTS_DIR, 'sweat_agency_results.csv')
    df_agency.to_csv(out_agency, index=False)
    print(f'\n✓ Agency SWEAT results saved → {out_agency}')

    # ── SWEAT 2: Communal traits vs male/female names ────────────────
    df_communal = run_sweat(
        slice_models,
        target_words = COMMUNAL_TRAITS,
        attr1_words  = MALE_NAMES,
        attr2_words  = FEMALE_NAMES,
        label        = 'communal',
    )
    out_communal = os.path.join(RESULTS_DIR, 'sweat_communal_results.csv')
    df_communal.to_csv(out_communal, index=False)
    print(f'✓ Communal SWEAT results saved → {out_communal}')

    # ── Summary table ─────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print('SWEAT SUMMARY — Agency traits (positive d = male-biased)')
    print(f'{"─"*55}')
    print(df_agency.to_string(index=False))

    print(f'\n{"─"*55}')
    print('SWEAT SUMMARY — Communal traits (positive d = male-biased)')
    print(f'{"─"*55}')
    print(df_communal.to_string(index=False))

    print(f'\nNext step: run 10_visualise.py')



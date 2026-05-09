"""
10_visualise.py
===============
Generates all 4 project figures from the saved results CSVs
and the trained CADE slice models.

Figures produced
----------------
  plot1_top_changed_words.png   — horizontal bar chart, top 20 changers
  plot2_word_evolution.png      — line chart, drift timeline for targets
  plot3_sweat_bias.png          — SWEAT effect size across 9 periods
  plot4_pca_cluster.png         — PCA neighbourhood plot for 'awful'

    python 09_visualise.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from configs.config import RESULTS_DIR, CHANGE_SCORE_PATH, DRIFT_TIMELINE_PATH, FIGURES_DIR, MODEL_CAD_SLICES_DIR, \
    DECADES_LIST, TARGET_WORDS





# ── Style ─────────────────────────────────────────────────────────────
PALETTE = {
    'navy': '#1E3A5F',
    'blue': '#2563EB',
    'teal': '#0891B2',
    'sky': '#7DD3FC',
    'green': '#059669',
    'red': '#DC2626',
    'amber': '#D97706',
    'gray': '#64748B',
    'lgray': '#CBD5E1',
    'offwhite': '#F8FAFC',
}

LINE_COLOURS = ['#0891B2', '#DC2626', '#059669', '#D97706', '#7C3AED']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#E2E8F0',
    'grid.linewidth': 0.6,
    'figure.dpi': 150,
})


# ══════════════════════════════════════════════════════════════════════
# PLOT 1 — Top 20 most changed words (horizontal bar chart)
# ══════════════════════════════════════════════════════════════════════

def plot1_top_changed_words():
    if not os.path.exists(CHANGE_SCORE_PATH):
        print(f'  ⚠ Plot 1 skipped — {CHANGE_SCORE_PATH} not found')
        return

    df = pd.read_csv(CHANGE_SCORE_PATH)
    top20 = df.nlargest(20, 'change_score').iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(
        top20['word'],
        top20['change_score'],
        color=PALETTE['teal'],
        edgecolor='white',
        linewidth=0.5,
    )

    # Highlight any target words in amber
    target_set = set(TARGET_WORDS)
    for bar, word in zip(bars, top20['word']):
        if word in target_set:
            bar.set_color(PALETTE['amber'])

    # Value labels
    for bar in bars:
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.3f}',
            va='center', ha='left', fontsize=8.5, color=PALETTE['gray']
        )

    ax.set_xlabel('Cosine Distance  (1800s → 1960s)', fontsize=11, color=PALETTE['navy'])
    ax.set_title(
        'Top 20 Most Semantically Changed Words\nin English Literature (1800–1960s)',
        fontsize=13, fontweight='bold', color=PALETTE['navy'], pad=14
    )
    ax.set_xlim(0, top20['change_score'].max() * 1.15)
    ax.tick_params(axis='both', labelsize=10)

    legend_patches = [
        mpatches.Patch(color=PALETTE['teal'], label='Vocabulary word'),
        mpatches.Patch(color=PALETTE['amber'], label='Target word'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'plot1_top_changed_words.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Plot 1 saved → {out}')


# ══════════════════════════════════════════════════════════════════════
# PLOT 2 — Word evolution timeline (line chart)
# ══════════════════════════════════════════════════════════════════════

def plot2_word_evolution():
    if not os.path.exists(DRIFT_TIMELINE_PATH):
        print(f'  ⚠ Plot 2 skipped — {DRIFT_TIMELINE_PATH} not found')
        return

    df = pd.read_csv(DRIFT_TIMELINE_PATH)

    # Filter to target words actually present
    words_present = [w for w in TARGET_WORDS if w in df['word'].values]
    if not words_present:
        print('  ⚠ Plot 2 skipped — no target words in drift timeline')
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, word in enumerate(words_present):
        sub = df[df['word'] == word].sort_values('decade')
        sub = sub.dropna(subset=['drift'])
        if sub.empty:
            continue
        ax.plot(
            sub['decade'], sub['drift'],
            marker='o', linewidth=2.2, markersize=6,
            color=LINE_COLOURS[i % len(LINE_COLOURS)],
            label=f'"{word}"',
        )
        # Label the end point
        last = sub.iloc[-1]
        ax.annotate(
            word,
            (last['decade'], last['drift']),
            xytext=(5, 2), textcoords='offset points',
            fontsize=9, color=LINE_COLOURS[i % len(LINE_COLOURS)],
        )

    ax.set_ylabel('Cosine Distance from 1800s Baseline', fontsize=11, color=PALETTE['navy'])
    ax.set_xlabel('Literary Period', fontsize=11, color=PALETTE['navy'])
    ax.set_title(
        'Semantic Drift of Target Words Across Literary Periods\n'
        '(distance from each word\'s 1800s vector)',
        fontsize=13, fontweight='bold', color=PALETTE['navy'], pad=14
    )
    ax.legend(fontsize=10, loc='upper left')
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, 'plot2_word_evolution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Plot 2 saved → {out}')


# ══════════════════════════════════════════════════════════════════════
# PLOT 3 — SWEAT gender bias over time
# ══════════════════════════════════════════════════════════════════════

def plot3_sweat_bias():
    agency_path = os.path.join(RESULTS_DIR, 'sweat_agency_results.csv')
    communal_path = os.path.join(RESULTS_DIR, 'sweat_communal_results.csv')

    has_agency = os.path.exists(agency_path)
    has_communal = os.path.exists(communal_path)

    if not has_agency:
        print(f'  ⚠ Plot 3 skipped — {agency_path} not found')
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Agency line
    df_a = pd.read_csv(agency_path)
    ax.plot(
        df_a['decade'], df_a['d_agency'],
        marker='o', linewidth=2.5, color=PALETTE['teal'],
        label='Agency traits (brave, strong, rational…)',
    )
    # Mark significant points
    sig = df_a[df_a['p_agency'] < 0.05]
    ax.scatter(
        sig['decade'], sig['d_agency'],
        color=PALETTE['red'], s=90, zorder=5,
        label='p < 0.05 (significant)', edgecolors='white', linewidth=0.8
    )

    # Communal line (if available)
    if has_communal:
        df_c = pd.read_csv(communal_path)
        if 'd_communal' in df_c.columns:
            ax.plot(
                df_c['decade'], df_c['d_communal'],
                marker='s', linewidth=2.0, color=PALETTE['amber'],
                linestyle='--', label='Communal traits (gentle, beautiful…)',
            )

    # Zero reference line
    ax.axhline(0, color=PALETTE['gray'], linewidth=1.0, linestyle='--', alpha=0.7)

    # Threshold lines
    ax.axhline(0.8, color=PALETTE['navy'], linewidth=0.7, linestyle=':', alpha=0.5)
    ax.axhline(-0.8, color=PALETTE['navy'], linewidth=0.7, linestyle=':', alpha=0.5)
    ax.text(0.01, 0.83, 'd = 0.8 (large effect)',
            transform=ax.get_yaxis_transform(), fontsize=8.5, color=PALETTE['gray'])

    ax.set_ylabel('WEAT Effect Size (d)\nPositive = closer to male names', fontsize=11, color=PALETTE['navy'])
    ax.set_xlabel('Literary Period', fontsize=11, color=PALETTE['navy'])
    ax.set_title(
        'Gender Bias in Literary Language Across Nine Periods\n'
        '(agency vs communal adjectives — male vs female character names)',
        fontsize=13, fontweight='bold', color=PALETTE['navy'], pad=14
    )
    ax.legend(fontsize=10, loc='best')
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, 'plot3_sweat_bias.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Plot 3 saved → {out}')


# ══════════════════════════════════════════════════════════════════════
# PLOT 4 — PCA word cluster for 'awful' (1800s vs 1960s)
# ══════════════════════════════════════════════════════════════════════

def plot4_pca_cluster(focus_word='awful', topn=9):
    """
    Side-by-side PCA scatter plots showing the nearest-neighbour
    cluster around focus_word in the earliest vs latest slice.
    """
    # Load earliest and latest slices
    available = [d for d in DECADES_LIST
                 if os.path.exists(os.path.join(MODEL_CAD_SLICES_DIR, f'slice_{d}.model'))]
    if len(available) < 2:
        print('  ⚠ Plot 4 skipped — not enough slice models')
        return

    early_d = available[0]
    late_d = available[-1]
    m_early = Word2Vec.load(os.path.join(MODEL_CAD_SLICES_DIR, f'slice_{early_d}.model'))
    m_late = Word2Vec.load(os.path.join(MODEL_CAD_SLICES_DIR, f'slice_{late_d}.model'))

    if focus_word not in m_early.wv or focus_word not in m_late.wv:
        print(f'  ⚠ Plot 4 skipped — "{focus_word}" not in one or both slice vocabularies')
        return

    def get_cluster(model, word, topn):
        neighbours = [w for w, _ in model.wv.most_similar(word, topn=topn)]
        words = [word] + neighbours
        vecs = np.array([model.wv[w] for w in words])
        return words, vecs

    words_e, vecs_e = get_cluster(m_early, focus_word, topn)
    words_l, vecs_l = get_cluster(m_late, focus_word, topn)

    # Fit PCA on combined vectors so both panels use the same coordinate space
    all_vecs = np.vstack([vecs_e, vecs_l])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_vecs)
    coords_e = coords[:len(words_e)]
    coords_l = coords[len(words_e):]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f'PCA Word-Cluster Visualisation: "{focus_word}"\n'
        f'Nearest neighbours in {early_d} (left) vs {late_d} (right)',
        fontsize=13, fontweight='bold', color=PALETTE['navy'], y=1.01
    )

    for ax, coords_subset, words, title, accent in [
        (axes[0], coords_e, words_e, f'"{focus_word}" — {early_d}', PALETTE['teal']),
        (axes[1], coords_l, words_l, f'"{focus_word}" — {late_d}', PALETTE['red']),
    ]:
        # Background
        ax.set_facecolor(PALETTE['offwhite'])
        ax.grid(color='white', linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Scatter
        ax.scatter(
            coords_subset[1:, 0], coords_subset[1:, 1],
            c=PALETTE['sky'], s=90, edgecolors=PALETTE['teal'],
            linewidth=0.8, zorder=3, label='Neighbours'
        )
        # Focus word
        ax.scatter(
            coords_subset[0, 0], coords_subset[0, 1],
            c=accent, s=160, edgecolors='white',
            linewidth=1.2, zorder=5, marker='*', label=f'"{focus_word}"'
        )

        # Annotations
        for i, word in enumerate(words):
            c = coords_subset[i]
            fontweight = 'bold' if i == 0 else 'normal'
            colour = accent if i == 0 else PALETTE['navy']
            ax.annotate(
                word, (c[0], c[1]),
                xytext=(5, 4), textcoords='offset points',
                fontsize=9.5, color=colour, fontweight=fontweight
            )

        ax.set_title(title, fontsize=12, fontweight='bold', color=accent, pad=10)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=9)
        ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'plot4_pca_cluster.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Plot 4 saved → {out}')


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def visualize():
    print('=' * 55)
    print(' Visualisation — generating all 4 plots')
    print('=' * 55 + '\n')

    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot1_top_changed_words()
    plot2_word_evolution()
    plot3_sweat_bias()
    plot4_pca_cluster(focus_word='soil')

    print(f'\nAll figures saved to:  {FIGURES_DIR}/')
    print('\nFiles produced:')
    for f in sorted(os.listdir(FIGURES_DIR)):
        fp = os.path.join(FIGURES_DIR, f)
        kb = os.path.getsize(fp) / 1024
        print(f'  {f}  ({kb:.0f} KB)')




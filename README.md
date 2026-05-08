
# 🗂 Project Structure
```bash
literature_semantics_project/
│
├── README.md
├── requirements.txt
├── main.py
├── .gitignore
│
├── configs/
│   ├── config.py
│   └── verify_setup.py
│
├── data/
│   ├── catalog/                          # metadata (catalog, CSV, etc.)
│   ├── processed/                        # final tokenized corpus per decade
│   └── raw/                              # original Gutenberg texts (unprocessed)
│
├── src/
│   │
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── fetch_csv.py                  # Download Gutenberg Catalog
│   │   └── fetch_corpus.py               # Download Raw Data of Books
│   │
│   ├── preprocessing/
│   │   ├── pre_processing_data.py        # boilerplate removal + tokenization 
│   │   └── verify_preprocessing.py       
│   │
│   ├── models/
│   │   ├── word2vec.py           # baseline Word2Vec training
│   │   ├── cade_model.py         # CADE wrapper
│   │   └── embeddings.py         # embedding utilities
│   │
│   ├── analysis/
│       ├── semantic_change.py    # cosine drift computation
│       │   ├── weat.py           # WEAT + SWEAT implementation
│       │   └── similarity.py     # cosine similarity helpers
│       │
│       ├── evaluation/
│       │   ├── metrics.py         # statistical tests
│       │   └── validation.py      # sanity checks
│       │
│       ├── visualization/
│       │   ├── plots.py           # all plotting functions
│       │   ├── pca.py             # dimensionality reduction
│       │   └── timelines.py       # temporal plots
│       │
│       └── utils/
│           ├── logging.py
│           ├── io.py              # save/load models, csv, etc.
│           └── helpers.py
│
├── scripts/                       # CLI entry points (important!)
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_train_word2vec.py
│   ├── 04_train_cade.py
│   ├── 05_semantic_change.py
│   ├── 06_weat_analysis.py
│   └── 07_generate_plots.py
│
├── notebooks/                     # ONLY for exploration
│   ├── 00_exploration.ipynb
│   ├── 01_testing_pipeline.ipynb
│   └── 02_results_visualization.ipynb
│
├── models/
│   ├── word2vec_baseline/
│   ├── cade_compass/
│   └── cade_slices/
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── weat_outputs/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   └── test_weat.py
│
└── docs/
    ├── report.md
    ├── slides_outline.md
    └── viva_notes.md
```

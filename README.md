
# 🗂 Project Structure
```python
literature_semantics_project/
│
├── README.md
├── requirements.txt
├── main.py
├── .gitignore
│
├── configs/
│   ├── config.py
│   ├── clean_files.py
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
│   │   ├── train_word2vec.py           # baseline Word2Vec training
│   │   └── train_CAD.py                # CADE wrapper
│   │
│   ├── analysis/
│       ├── data_explore.py   
│       ├── semantic_change.py
│       ├── weat_bias.py
│       └── visualizations.py
│
├── models/
│   ├── cad_slices/
│   ├── cade_compass.model
│   └── word2vec.model
│
├── results/
│   ├── figures/
│   ├── all_change_score.csv
│   ├── drift_timeline.csv
│   ├── sweat_agency_result.csv
│   ├── sweat_communal_result.csv
│   └── nearest_neighbours.json
│
└── Reports/
    ├── Project Report
    └──Project Slides
```

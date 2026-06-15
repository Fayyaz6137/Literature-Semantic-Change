
# 🗂 Project Structure
**Note: data**, **results** and **models** will be generated when the program is run.
```python
literature_semantics_project/
│
├── README.md
├── requirements.txt
├── main.py                               #---------------- Starting Point ----------------#
├── .gitignore
│
├── configs/                              #---------------- Phase 0 ----------------#
│   ├── config.py
│   ├── clean_files.py
│   └── verify_setup.py
│
├── data/
│   ├── catalog/                          
│   ├── processed/                        
│   └── raw/                              
│
├── src/
│   ├── data/                             #---------------- Phase 1 ----------------#
│   │   ├── fetch_csv.py                  
│   │   └── fetch_corpus.py               
│   │
│   ├── preprocessing/                    #---------------- Phase 2 ----------------#
│   │   ├── pre_processing_data.py        
│   │   └── verify_preprocessing.py       
│   │
│   ├── models/                           #---------------- Phase 3 ----------------#
│   │   ├── train_word2vec.py             
│   │   └── train_CAD.py                  
│   │
│   ├── analysis/                         #---------------- Phase 4 ----------------#
│       ├── data_explore.py   
│       ├── semantic_change.py
│       ├── weat_bias.py
│       └── visualizations.py
│
├── models/                               #---------------- Outputs ----------------#
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
    └── Project Slides
```

from configs.config import DECADES_LIST, DATA_RAW_DIR


def data_exploration():
    import os

    for decade in DECADES_LIST:
        path = f'{DATA_RAW_DIR}/lit_{decade}.txt'
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            words = sum(len(l.split()) for l in lines)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"{decade}: {words:,} words | {size_mb:.1f} MB")
        else:
            print(f"{decade}: FILE NOT FOUND — re-run download script")
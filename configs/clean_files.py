import os
import shutil

from debugpy.launcher import output

from configs.config import DATA_DIR, MODELS_DIR, ENG_BOOKS_EXCEL, RESULTS_DIR


def clean_files():
    for file in (DATA_DIR, MODELS_DIR, ENG_BOOKS_EXCEL, RESULTS_DIR):
        if os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)
                print(f"{file} Removed")
            else:
                shutil.rmtree(file)
                print(f"{file} Removed")

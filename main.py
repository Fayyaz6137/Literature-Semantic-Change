from configs.config import DECADES, MAX_BOOKS_PER_DECADE, DELAY, MIN_LENGTH, HEADERS


def main():
    print("Starting ...")

    # ---------------------------- 1. Get and Prepare Data ------------------------------- #
    from src.data import fetch_csv, fetch_corpus
    df = fetch_csv.get_catalog()
    fetch_corpus.get_corpus_data(df)

    # ---------------------------- 2. Data Exploration ------------------------------- #
    from src.analysis import data_exploratory_analysis
    data_exploratory_analysis.data_exploration()


if __name__ == "__main__":
    main()

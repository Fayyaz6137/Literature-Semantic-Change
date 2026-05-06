def main():
    print("Starting ...")

    # ---------------------------- 1. Get and Prepare Data ------------------------------- #

    from src.data import fetch_csv, fetch_corpus
    df = fetch_csv.get_catalog()
    fetch_corpus.get_corpus_data(df)

    # ---------------------------- 2. Data Exploration ------------------------------- #
    from src.analysis import data_exploratory_analysis
    data_exploratory_analysis.data_exploration()
    print("Pre-Processing Verification End ...")

    # ---------------------------- 3. Setup Verification ------------------------------- #
    print("Setup Verification Starting ...")
    from configs import verify_setup
    verify_setup.verify_libraries()
    print("Setup Verification End ...")


if __name__ == "__main__":
    main()

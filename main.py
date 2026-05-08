def main():
    print("Starting ...")

    run_al=0
    run_all=True if run_al==1 else False


    if run_all:
        # ---------------------------- 1. Get and Prepare Data ------------------------------- #
        from src.data import fetch_csv, fetch_corpus
        df = fetch_csv.get_catalog()
        fetch_corpus.get_corpus_data(df)

        # ---------------------------- 2. Data Exploration ------------------------------- #
        from src.analysis import data_exploratory_analysis
        data_exploratory_analysis.data_exploration()

        # ---------------------------- 3. Setup Verification ------------------------------- #
        print("\nSetup Verification START ...\n")

        from configs import verify_setup
        verify_setup.verify_libraries()

        print("\nSetup Verification END\n")

        # ---------------------------- 4. Pre-Processing Data ------------------------------- #
        print("\nPre-Processing Data START ...\n")
        from src.preprocessing import pre_processing_data, verify_preprocessing

        pre_processing_data.pre_process_data()
        verify_preprocessing.verify_pre_processing_output()
        verify_preprocessing.check_word_frequency()

        print("\nPre-Processing Data END\n")

    # ---------------------------- 5. Word2Vec Model Training ------------------------------- #
    print("\nWord2Vec Model Training START ...\n")
    from src.models import train_word2vec_baseline

    train_word2vec_baseline.word2vec_training()

    print("\nWord2Vec Model Training END\n")


if __name__ == "__main__":
    main()

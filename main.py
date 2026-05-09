from src import analysis


def main():
    print("Starting ...")

    run_all = 0
    if run_all:
        # ---------------------------- 0. Remove Old Data ------------------------------- #
        print("\nRemoving Old Files ...\n")
        from configs import clean_files
        clean_files.clean_files()
        print("\nRemoved Old Files \n")

        # ---------------------------- 1. Get and Prepare Data ------------------------------- #
        from src.data import fetch_csv, fetch_corpus
        df = fetch_csv.get_catalog()
        fetch_corpus.get_corpus_data(df)

        # ---------------------------- 2. Data Exploration ------------------------------- #
        from src.analysis import data_exploratory_analysis
        data_exploratory_analysis.data_exploration()

        # ---------------------------- 3. Pre-Processing Data ------------------------------- #
        print("\nPre-Processing Data START ...\n")
        from src.preprocessing import pre_processing_data, verify_preprocessing

        pre_processing_data.pre_process_data()
        verify_preprocessing.verify_pre_processing_output()
        verify_preprocessing.check_word_frequency()

        print("\nPre-Processing Data END\n")

        # ---------------------------- 4. Setup Verification ------------------------------- #
        print("\nSetup Verification START ...\n")

        from configs import verify_setup
        verify_setup.verify_libraries()

        print("\nSetup Verification END\n")

        # ---------------------------- 5. Word2Vec Model Training ------------------------------- #
        print("\nWord2Vec Model Training START ...\n")
        from src.models import train_word2vec_baseline

        train_word2vec_baseline.word2vec_training()
        train_word2vec_baseline.word2vec_test()

        print("\nWord2Vec Model Training END\n")

        # ---------------------------- 6. CAD Model Training ------------------------------- #
        print("\nCAD Compass Model Training START ...\n")
        from src.models import train_CAD

        train_CAD.train_compass()
        train_CAD.train_slices()

        print("\nCAD Compass Model Training END\n")

        # ---------------------------- 7. Semantic Change Analysis ------------------------------- #
        print("\nSemantic Change Analysis START ...\n")
        from src.analysis import semantic_change_analysis

        semantic_change_analysis.run_semantic_change_analysis()

        print("\nSemantic Change Analysis END\n")

        # ---------------------------- 8. Weat Bias Analysis ------------------------------- #
        print("\nWeat Bias Analysis START ...\n")
        from src.analysis import weat_bias_analysis

        weat_bias_analysis.run_weat_sweat_bias_analysis()

        print("\nWeat Bias Analysis END\n")

        # ---------------------------- 9. Visualizations ------------------------------- #
        print("\nVisualizations START ...\n")
        from src.analysis import visualizations

        visualizations.visualize()

        print("\nVisualizations END\n")


if __name__ == "__main__":
    main()

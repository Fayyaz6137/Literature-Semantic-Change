def verify_pre_processing_output():
    with open('data/processed/lit_1840s_clean.txt') as f:
        sample = [next(f) for _ in range(5)]

    print("Sample of cleaned 1840s text:")
    for line in sample:
        print("  ", line.strip())


def check_word_frequency():
    from collections import Counter

    with open('data/processed/lit_all_clean.txt') as f:
        all_words = f.read().split()

    freq = Counter(all_words)
    print(f"Total tokens: {len(all_words):,}")
    print(f"Unique words: {len(freq):,}")
    print("Top 20 words:", freq.most_common(20))

    # Check your key target words are frequent enough
    target_words = ['awful', 'nice', 'gay', 'pioneer', 'villain', 'heroine']
    print("\nFrequency of target words:")
    for w in target_words:
        print(f"  {w}: {freq.get(w, 0):,}")

import os

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def word2vec_training():
    corpus = LineSentence('data/processed/lit_all_clean.txt')

    model = Word2Vec(
        sentences=corpus,
        vector_size=100,  # Each word = 100-dimensional vector
        window=5,  # Look at 5 words on each side
        min_count=10,  # Ignore words appearing fewer than 10 times
        sg=1,  # 1 = Skip-gram, 0 = CBOW
        workers=4,
        epochs=5,
        seed=42
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/word2vec_baseline.model')
    print("Baseline model saved!")
    print(f"Vocabulary size: {len(model.wv):,} words")

    wv = model.wv

    # Explore the baseline model — sanity check
    # print("\nMost similar to 'heroine':")
    # print(wv.most_similar('heroine', topn=10))

    print("\nMost similar to 'awful':")
    print(wv.most_similar('awful', topn=3))

    print("\nMost similar to 'brave':")
    print(wv.most_similar('brave', topn=10))

    # Classic analogy test on literary vocabulary
    result = wv.most_similar(positive=['queen', 'man'], negative=['woman'])
    print("\nwoman:queen :: man:?")
    print(result[:5])

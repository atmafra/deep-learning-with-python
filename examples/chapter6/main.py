from corpora.imdb import imdb
from embeddings.glove import glove
from examples.chapter6 import one_hot_encoding, word_embeddings, recurrent_toy, recurrent_imdb

default_glove_embeddings_filename = 'glove.6B.100d.txt'
default_glove_dir = '../../embeddings/glove/data/glove.6B'

if __name__ == '__main__':

    experiments = [
        # 'one_hot_encoding',
        # 'word_embeddings',
        # 'recurrent_toy',
        'recurrent_imdb'
    ]

    if 'one_hot_encoding' in experiments:
        one_hot_encoding.run()

    if 'word_embeddings' in experiments:
        word_embeddings.run(build=True,
                            maximum_tokens_per_text=100,
                            vocabulary_size=10000)

    if 'recurrent_toy' in experiments:
        recurrent_toy.run()

    if 'recurrent_imdb' in experiments:
        recurrent_imdb.run()


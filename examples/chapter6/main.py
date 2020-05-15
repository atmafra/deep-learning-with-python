from corpora.imdb import imdb
from embeddings.glove import glove
from examples.chapter6 import one_hot_encoding, word_embeddings

default_glove_embeddings_filename = 'glove.6B.100d.txt'
default_glove_dir = '../../embeddings/glove/data/glove.6B'

if __name__ == '__main__':

    experiments = [
        # 'one_hot_encoding',
        'word_embeddings'
    ]

    if 'one_hot_encoding' in experiments:
        one_hot_encoding.run()

    if 'word_embeddings' in experiments:
        word_embeddings.run(build=True,
                            maximum_tokens_per_text=100,
                            vocabulary_size=10000)

    # embeddings_matrix = glove.load_embeddings_matrix(glove_path=default_glove_dir,
    #                                                  glove_embeddings_filename=default_glove_embeddings_filename,
    #                                                  vocabulary_size=10000,
    #                                                  embeddings_dimension=100)

from corpora.imdb import imdb
from examples.chapter6.recurrent_imdb_configurations import load_experiment_plan


def load_imdb_corpus(maximum_tokens_per_text: int,
                     vocabulary_size: int):
    return imdb.build_corpus(imdb_path='../../corpora/imdb/data/aclImdb',
                             maximum_tokens_per_text=maximum_tokens_per_text,
                             vocabulary_size=vocabulary_size,
                             randomize=True)


def run():
    maxlen = 500
    vocabulary_size = 10000
    embeddings_dimension = 32

    corpus = load_imdb_corpus(maximum_tokens_per_text=maxlen,
                              vocabulary_size=vocabulary_size)

    experiment_plan = load_experiment_plan(corpus=corpus)
    experiment_plan.run(train=True,
                        test=True,
                        print_results=True,
                        plot_training_loss=True,
                        plot_validation_loss=True,
                        plot_training_accuracy=True,
                        plot_validation_accuracy=True,
                        display_progress_bars=True,
                        save_models=False)

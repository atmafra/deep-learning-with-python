# maximum
from corpora.imdb import imdb
from examples.chapter6.word_embeddings_configurations import load_experiment_plan

default_maxlen = 20


def run(build: bool = False,
        vocabulary_size: int = 10000,
        maximum_tokens_per_text: int = default_maxlen):
    """ Runs the Word Embeddings experiments

    :param build: build the IMDB Sentiment Analysis corpus from source files
    :param vocabulary_size: vocabulary size for tokenization
    :param maximum_tokens_per_text: maximum tokens per review
    """
    if build:
        corpus = imdb.build_corpus(imdb_path='../../corpora/imdb/data/aclImdb',
                                   maximum_tokens_per_text=maximum_tokens_per_text,
                                   vocabulary_size=vocabulary_size,
                                   randomize=True)
    else:
        corpus = imdb.load_corpus(corpus_name='IMDB')

    embeddings_dimension = 100
    experiment_plan = load_experiment_plan(corpus=corpus,
                                           input_length=maximum_tokens_per_text,
                                           vocabulary_size=vocabulary_size,
                                           embeddings_dimension=embeddings_dimension)

    experiment_plan.run(train=True,
                        test=True,
                        print_results=True,
                        plot_training_loss=True,
                        plot_validation_loss=True,
                        plot_training_accuracy=True,
                        plot_validation_accuracy=True,
                        display_progress_bars=True,
                        save_models=False,
                        models_path='models')

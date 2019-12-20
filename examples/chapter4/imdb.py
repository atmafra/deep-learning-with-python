import numpy as np
from keras.datasets import imdb

from examples.chapter4.imdb_configurations import *
from utils import dataset_utils as dsu

word_index = {}
reverse_word_index = {}


def load_corpus(words: int = 10000, verbose: bool = True) -> Corpus:
    """"Loads the IMDB dataset into a corpus object

    Args:
        words (int): word limit in the reverse index
        verbose (bool): outputs progress messages

    """
    if verbose:
        print("Loading IMDB dataset...")

    corpus = imdb.load_data(num_words=words)
    (train_samples, train_labels), (test_samples, test_labels) = dsu.separate_corpus(corpus)

    # one-hot encode the phrases
    vector_dimension = words
    training_inputs = dsu.one_hot_encode(train_samples, vector_dimension)
    test_inputs = dsu.one_hot_encode(test_samples, vector_dimension)

    # vectorize the labels
    training_outputs = np.asarray(train_labels).astype('float32')
    test_outputs = np.asarray(test_labels).astype('float32')

    if verbose:
        print("{} train reviews loaded".format(len(train_samples)))
        print("{} test reviews loaded".format(len(test_samples)))

    # create the corpus
    return Corpus.from_datasets(training_inputs, training_outputs, test_inputs, test_outputs)


def run(plan: str = 'comparison'):
    """Runs the selected experiment plan

    Args:
        plan (str): key of the experiment to run. Possible values are
            'dropout'
            'weight_regularization_l1'
            'weight_regularization_l2'
            'comparison'

    """
    # loads the corpus and the experiment plans
    corpus = load_corpus(words=num_words)
    experiments = load_experiments(corpus=corpus)

    # runs the selected experiment plan
    experiment_plan = experiments[plan]

    experiment_plan.run(print_results=True,
                        plot_training_loss=False,
                        plot_training_accuracy=False,
                        display_progress_bars=True)

    experiment_plan.save_models(path='models')

    # plots the results
    experiment_plan.plot_loss("Training Loss", training=True, validation=False)
    experiment_plan.plot_loss("Validation Loss", training=False, validation=True)
    experiment_plan.plot_accuracy("Training Accuracy", training=True, validation=False)
    experiment_plan.plot_accuracy("Validation Accuracy", training=False, validation=True)

import numpy as np
from keras import regularizers
from keras.datasets import imdb

from core import network as net
from core.experiment import Experiment
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils import history_utils as hutl
from examples.chapter4.imdb_configurations import *

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
    # global vector_dimension
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


def load_experiments(corpus: Corpus):
    """Defines the experiments that will be run
    """
    _small = Experiment(corpus=corpus,
                        layers_configuration_list=imdb_small,
                        training_configuration=training_configuration_global)

    _medium = Experiment(corpus=corpus,
                         layers_configuration_list=imdb_medium,
                         training_configuration=training_configuration_global)

    _large = Experiment(corpus=corpus,
                        layers_configuration_list=imdb_large,
                        training_configuration=training_configuration_global)

    _medium_dropout = Experiment(corpus=corpus,
                                 layers_configuration_list=imdb_medium_dropout,
                                 training_configuration=training_configuration_global)

    _medium_wreg_l1 = Experiment(corpus=corpus,
                                 layers_configuration_list=imdb_medium_wreg_l1,
                                 training_configuration=training_configuration_global)

    _medium_wreg_l2 = Experiment(corpus=corpus,
                                 layers_configuration_list=imdb_medium_wreg_l2,
                                 training_configuration=training_configuration_global)

    _medium_wreg_l1_l2 = Experiment(corpus=corpus,
                                    layers_configuration_list=imdb_medium_wreg_l1_l2,
                                    training_configuration=training_configuration_global)

    _medium_dropout_wreg_l2 = Experiment(corpus=corpus,
                                         layers_configuration_list=imdb_medium_dropout_wreg_l2,
                                         training_configuration=training_configuration_global)

    experiments = {'small': _small,
                   'medium': _medium,
                   'large': _large,
                   'medium_dropout': _medium_dropout,
                   'medium_wreg_l1': _medium_wreg_l1,
                   'medium_wreg_l2': _medium_wreg_l2,
                   'medium_wreg_l1_l2': _medium_wreg_l1_l2,
                   'medium_dropout_wreg_l2': _medium_dropout_wreg_l2}

    return experiments


def run():
    # loads the corpus
    corpus = load_corpus(words=num_words)
    experiments = load_experiments(corpus=corpus)

    # experiment_small.run()
    # experiment_medium.run()
    # experiment_large.run()
    # experiment_medium_dropout.run()
    # experiment_medium_wreg_l1.run()
    # experiment_medium_wreg_l2.run()
    # experiment_medium_wreg_l1_l2.run()
    experiments['medium_dropout_wreg_l2'].run()
    exit()

    # metrics = [history_small, history_medium, history_large]
    # legends = ['small network', 'medium network', 'large network']
    metrics = [history_medium, history_medium_dropout, history_medium_wreg_l1, history_medium_wreg_l2,
               history_medium_wreg_l1_l2, history_medium_dropout_wreg_l2]

    legends = ['medium network', 'medium with dropout',
               'medium with weight regularization L1',
               'medium with weight regularization L2',
               'medium with weight regularization L1 and L2',
               'medium with dropout and weight regularization L2']

    hutl.plot_loss_list(history_metrics_list=metrics,
                        labels_list=legends,
                        title='Training Loss',
                        plot_training=True,
                        plot_validation=False)

    hutl.plot_loss_list(history_metrics_list=metrics,
                        labels_list=legends,
                        title='Validation Loss',
                        plot_training=False,
                        plot_validation=True)

    hutl.plot_accuracy_list(history_metrics_list=metrics,
                            labels_list=legends,
                            title='Training Accuracy',
                            plot_training=True,
                            plot_validation=False)

    hutl.plot_accuracy_list(history_metrics_list=metrics,
                            labels_list=legends,
                            title='Validation Accuracy',
                            plot_training=False,
                            plot_validation=True)

import numpy as np
from keras.datasets import imdb

from core.file_structures import CorpusFileStructure
from examples.chapter4.imdb_configurations import *
from utils import dataset_utils as dsu

word_index = {}
reverse_word_index = {}


def build_corpus(name: str,
                 words: int = 10000,
                 save: bool = True,
                 verbose: bool = True) -> Corpus:
    """"Loads the IMDB dataset into a corpus object

    Args:
        name (str): corpus name
        words (int): word limit in the reverse index
        save (bool): save the pre-processed datafiles
        verbose (bool): outputs progress messages

    """
    if verbose:
        print("Loading IMDB dataset...")

    corpus_datasets = dsu.separate_corpus(imdb.load_data(num_words=words))
    train_samples = corpus_datasets[0][0]
    train_labels = corpus_datasets[0][1]
    test_samples = corpus_datasets[1][0]
    test_labels = corpus_datasets[1][1]

    # one-hot encode the phrases
    vector_dimension = words
    training_inputs = dsu.one_hot_encode(train_samples, vector_dimension)
    test_inputs = dsu.one_hot_encode(test_samples, vector_dimension)

    # vectorize the labels
    training_outputs = np.asarray(train_labels).astype('float32')
    test_outputs = np.asarray(test_labels).astype('float32')

    # create the corpus
    corpus = Corpus.from_datasets(training_input=training_inputs,
                                  training_output=training_outputs,
                                  test_input=test_inputs,
                                  test_output=test_outputs,
                                  name=name)

    if verbose:
        print("{} train reviews loaded".format(corpus.training_set.length))
        print("{} test reviews loaded".format(corpus.test_set.length))

    if save:
        save_corpus(corpus)

    return corpus


def save_corpus(corpus: Corpus,
                corpus_file_structure: CorpusFileStructure = None):
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus.name,
                                                                  base_path='data')
    corpus_file_structure.save_corpus(corpus)


def load_corpus(corpus_name: str,
                corpus_file_structure: CorpusFileStructure = None):
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus_name,
                                                                  base_path='data')
    return corpus_file_structure.load_corpus(corpus_name=corpus_name, datasets_base_name=corpus_name)


def run(plan: str = 'comparison', build: bool = True):
    """Runs the selected experiment plan

    Args:
        plan (str): key of the experiment to run. Possible values are
            'dropout'
            'weight_regularization_l1'
            'weight_regularization_l2'
            'comparison'
        build (bool): force rebuild of the datasets

    """
    # loads the corpus and the experiment plans
    corpus = None
    corpus_name = 'IMDB'
    if build:
        corpus = build_corpus(name=corpus_name, words=num_words, save=True)
    else:
        corpus = load_corpus(corpus_name=corpus_name)

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

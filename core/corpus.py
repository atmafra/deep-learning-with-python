from enum import Enum

import numpy as np

import utils.dataset_utils as dsu
from core.sets import Set, SetGenerator


class CorpusType(Enum):
    CORPUS_DATASET = 1
    CORPUS_GENERATOR = 2


class Corpus:
    """A Corpus is a group of two sets: Training and Test. An additional set
       can be created for training enhancemente purposes: the Validation set.

        Args:
            training_set (Set): training set
            test_set (Set): test set

        Members:
            training_set (Set): training set
            test_set (Set): test set
            validation_set (Set): a split from the training set, used to perform cross-validation

    """

    def __init__(self, training_set: Set, test_set: Set):
        self.__training_set = training_set
        self.__test_set = test_set

    @classmethod
    def from_datasets(cls,
                      training_inputs: np.array,
                      training_outputs: np.array,
                      test_inputs: np.array,
                      test_outputs: np.array):
        """Creates a corpus from the 4 datasets: training x test, input x output

        Args:
            training_inputs (np.array): training set inputs
            training_outputs (np.array): training set outputs
            test_inputs (np.array): test set inputs
            test_outputs (np.array): test set outputs

        """
        training_set = Set(training_inputs, training_outputs)
        test_set = Set(test_inputs, test_outputs)
        return Corpus(training_set, test_set)

    @classmethod
    def from_tuple(cls, corpus: tuple):
        """Creates a new Corpus from a pair (tuple) of two sets
           Each of these sets must have two subsets: input and output sets

        Args:
            corpus: a pair of arrays that represent training and test sets

        """
        (training_inputs, training_outputs), (test_inputs, test_outputs) = dsu.separate_corpus(corpus)
        return Corpus.from_datasets(training_inputs, training_outputs, test_inputs, test_outputs)

    @property
    def training_set(self):
        return self.__training_set

    @property
    def test_set(self):
        return self.__test_set

    @property
    def input_size(self):
        """Returns the size of the input elements
        """
        return self.training_set.input_size

    @property
    def output_size(self):
        """Returns the size of the output elements
        """
        return self.training_set.output_size

    @property
    def count_categories(self):
        """Returns the number of distinct labels in the output data
        """
        return self.training_set.count_unique_values

    @property
    def min_output(self):
        return self.training_set.min_output

    @property
    def max_output(self):
        return self.training_set.max_output

    @property
    def average_output(self):
        return self.training_set.average_output

    def get_validation_set(self, size: int, start: int = 0):
        """Splits the training set in order to split a validation dataset

        Args:
            size  (int) : validation set size
            start (int) : split training set from this position on

        """
        training_set_copy = self.training_set.copy()
        return training_set_copy.split(size=size, start=start)

    def get_validation_set_k_fold(self, fold: int, k: int):
        """Splits the training set to extract a validation set according to
           the k-fold rule

        Args:
            fold (int): current fold
            k (int): number of folds

        """
        return self.training_set.split_k_fold(fold=fold, k=k)


class CorpusGenerator:
    """A corpus generator contains three set generators:
       1. training
       2. validation
       3. test

    """

    def __init__(self,
                 training_set_generator: SetGenerator,
                 validation_set_generator: SetGenerator,
                 test_set_generator: SetGenerator):
        """Creates a new Corpus based on Set Generators

        Args:
            training_set_generator (SetGenerator): training set generator
            validation_set_generator (SetGenerator): validation set generator
            test_set_generator (SetGenerator): test set generator
        """
        self.__training_set_generator = training_set_generator
        self.__validation_set_generator = validation_set_generator
        self.__test_set_generator = test_set_generator

    @property
    def training_set_generator(self):
        return self.__training_set_generator

    @property
    def validation_set_generator(self):
        return self.__validation_set_generator

    @property
    def test_set_generator(self):
        return self.__test_set_generator


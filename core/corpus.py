from enum import Enum

import numpy as np

import utils.dataset_utils as dsu
from core.sets import Set, SetFiles


class CorpusType(Enum):
    CORPUS_DATASET = 1
    CORPUS_GENERATOR = 2


class Corpus:
    """A Corpus is a group of two sets: Training and Test. An additional set
       can be created for train enhancemente purposes: the Validation set.

        Args:
            training_set (Set): train set
            test_set (Set): test set

        Members:
            training_set (Set): train set
            test_set (Set): test set
            validation_set (Set): a split from the train set, used to perform cross-validation

    """

    def __init__(self, training_set: Set,
                 test_set: Set,
                 validation_set: Set = None):
        self.__training_set = training_set
        self.__test_set = test_set
        self.__validation_set = validation_set or None

    @classmethod
    def from_datasets(cls,
                      training_inputs: np.ndarray,
                      training_outputs: np.ndarray,
                      test_inputs: np.ndarray,
                      test_outputs: np.ndarray,
                      validation_inputs: np.ndarray = None,
                      validation_outputs: np.ndarray = None):
        """Creates a corpus from the 4 datasets: train x test, input x output

        Args:
            training_inputs (ndarray): train set inputs
            training_outputs (ndarray): train set outputs
            test_inputs (ndarray): test set inputs
            test_outputs (ndarray): test set outputs
            validation_inputs (ndarray): validation set inputs
            validation_outputs (ndarray): validation set outputs

        """
        training_set = Set(training_inputs, training_outputs)
        test_set = Set(test_inputs, test_outputs)
        validation_set = None

        if validation_inputs is not None and validation_outputs is not None:
            validation_set = Set(validation_inputs, validation_outputs)

        return Corpus(training_set=training_set,
                      test_set=test_set,
                      validation_set=validation_set)

    @classmethod
    def from_tuple(cls, corpus: tuple):
        """Creates a new Corpus from a pair (tuple) of two sets
           Each of these sets must have two subsets: input and output sets

        Args:
            corpus: a pair of arrays that represent train and test sets

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
    def validation_set(self):
        return self.__validation_set

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
        """Splits the train set in order to split a validation dataset

        Args:
            size  (int) : validation set size
            start (int) : split train set from this position on

        """
        training_set_copy = self.training_set.copy()
        return training_set_copy.split(size=size, start=start)

    def get_validation_set_k_fold(self, fold: int, k: int):
        """Splits the train set to extract a validation set according to
           the k-fold rule

        Args:
            fold (int): current fold
            k (int): number of folds

        """
        return self.training_set.split_k_fold(fold=fold, k=k)


class CorpusFiles:
    """A corpus generator contains three file sets: train, validation, and test

    """

    def __init__(self,
                 training_set_files: SetFiles,
                 validation_set_files: SetFiles,
                 test_set_files: SetFiles):
        """Creates a new Corpus based on Set of files

        Args:
            training_set_files (SetFiles): train set files
            validation_set_files (SetFiles): validation set files
            test_set_files (SetFiles): test set files
        """
        self.__training_set_files = training_set_files
        self.__validation_set_files = validation_set_files
        self.__test_set_files = test_set_files

    @property
    def training_set_files(self):
        return self.__training_set_files

    @property
    def validation_set_files(self):
        return self.__validation_set_files

    @property
    def test_set_files(self):
        return self.__test_set_files

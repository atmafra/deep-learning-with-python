from enum import Enum

import numpy as np

import utils.dataset_utils as dsu
from core.datasets import Dataset, DatasetFileIterator


class CorpusType(Enum):
    CORPUS_DATASET = 1
    CORPUS_GENERATOR = 2


class Corpus:
    """A Corpus is a group of two sets: Training and Test. An additional set
       can be created for train enhancemente purposes: the Validation set.


        Members:
            training_set (Set): train set
            test_set (Set): test set
            validation_set (Set): part of the training set used to perform cross-validation

    """

    def __init__(self, training_set: Dataset,
                 test_set: Dataset,
                 validation_set: Dataset = None,
                 name: str = ''):
        """Creates a new corpus

        Args:
            training_set (Dataset): train set
            test_set (Dataset): test set
            validation_set (Dataset): validation set
            name (str): corpus name

        """
        self.training_set = training_set
        self.test_set = test_set
        self.validation_set = validation_set or None
        self.__name = name

    @classmethod
    def from_datasets(cls,
                      training_input: np.ndarray,
                      training_output: np.ndarray,
                      test_input: np.ndarray,
                      test_output: np.ndarray,
                      validation_input: np.ndarray = None,
                      validation_output: np.ndarray = None,
                      name: str = ''):
        """Creates a corpus from the 4 datasets: train x test, input x output

        Args:
            training_input (ndarray): train set inputs
            training_output (ndarray): train set outputs
            test_input (ndarray): test set inputs
            test_output (ndarray): test set outputs
            validation_input (ndarray): validation set inputs
            validation_output (ndarray): validation set outputs
            name (str): corpus name

        """
        training_set = Dataset(training_input, training_output)
        test_set = Dataset(test_input, test_output)
        validation_set = None

        if validation_input is not None:
            validation_set = Dataset(validation_input, validation_output)

        return Corpus(training_set=training_set,
                      test_set=test_set,
                      validation_set=validation_set,
                      name=name)

    @classmethod
    def from_tuple(cls, corpus: tuple,
                   name: str = ''):
        """Creates a new Corpus from a pair (tuple) of two or three sets
           Each of these sets must have two subsets: input and output sets

        Args:
            corpus (tuple): a pair of arrays that represent train and test sets
            name (str): corpus name

        """
        if corpus is None:
            raise ValueError('Cannot create a corpus from a null tuple of datasets')

        corpus_datasets = dsu.separate_corpus(corpus)
        return Corpus.from_datasets(training_input=corpus_datasets[0][0],
                                    training_output=corpus_datasets[0][1],
                                    test_input=corpus_datasets[1][0],
                                    test_output=corpus_datasets[1][1],
                                    validation_input=corpus_datasets[2][0],
                                    validation_output=corpus_datasets[2][1],
                                    name=name)

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, training_set: Dataset):
        self.__training_set = training_set

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, test_set: Dataset):
        self.__test_set = test_set

    @property
    def validation_set(self):
        return self.__validation_set

    @validation_set.setter
    def validation_set(self, validation_set: Dataset):
        self.__validation_set = validation_set

    @property
    def name(self):
        return self.__name

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
                 training_set_files: DatasetFileIterator,
                 validation_set_files: DatasetFileIterator,
                 test_set_files: DatasetFileIterator):
        """Creates a new Corpus based on Set of files

        Args:
            training_set_files (DatasetFileIterator): train set files
            validation_set_files (DatasetFileIterator): validation set files
            test_set_files (DatasetFileIterator): test set files
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

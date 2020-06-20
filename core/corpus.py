from enum import Enum

import numpy as np

import utils.dataset_utils as dsu
from core.datasets import Dataset, DatasetFileIterator


class CorpusType(Enum):
    CORPUS_DATASET = 1
    CORPUS_GENERATOR = 2


class Corpus:
    """ A Corpus is a group of three datasets:

    - Training Set: used to train a neural network model
    - Test Set: used to evaluate the trained neural network model's performance metrics
    - Validation Set: used to evaluate the model's current performance metrics during training
    """

    def __init__(self,
                 training_set: Dataset = None,
                 test_set: Dataset = None,
                 validation_set: Dataset = None,
                 name: str = ''):
        """ Creates a new corpus

        :param training_set: training set
        :param test_set: test set
        :param validation_set: validation set
        :param name: corpus name
        """
        self.__input_size = 0
        self.__output_size = 0
        self.training_set = training_set or None
        self.test_set = test_set or None
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
        """ Creates a corpus from the 4 datasets: train x test, input x output

        :param training_input: train set inputs
        :param training_output: train set outputs
        :param test_input: test set inputs
        :param test_output: test set outputs
        :param validation_input: validation set inputs
        :param validation_output: validation set outputs
        :param name: corpus name
        :return: new Corpus
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
        """ Creates a new Corpus from a pair (tuple) of two or three sets. Each of these sets must have
        two subsets: input and output.

        :param corpus: a pair of arrays that represent train and test sets
        :param name: corpus name
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

    def resize(self,
               training_set_size: int = -1,
               validation_set_size: int = -1,
               test_set_size: int = -1,
               shuffle: bool = False):
        """ Resizes the internal Datasets to smaller sizes

        :param training_set_size: new size of the training set
        :param validation_set_size: new size of the validation set
        :param test_set_size: new size of the test set
        :param shuffle: shuffle sets before cutting
        :returns: nothing
        """
        if training_set_size > 0:
            if self.training_set is None:
                raise RuntimeError('Corpus has no training set to be resized')
            if shuffle:
                self.training_set.shuffle()
            self.training_set, training_remain = self.training_set.split(size=training_set_size)

        if validation_set_size > 0:
            if self.validation_set is None:
                raise RuntimeError('Corpus has no validation set to be resized')
            if shuffle:
                self.validation_set.shuffle()
            self.validation_set, validation_remain = self.validation_set.split(size=validation_set_size)

        if test_set_size > 0:
            if self.test_set is None:
                raise RuntimeError('Corpus has no test set to be resized')
            if shuffle:
                self.test_set.shuffle()
            self.test_set, test_remain = self.test_set.split(size=test_set_size)

    def copy(self,
             training_set_size: int = -1,
             validation_set_size: int = -1,
             test_set_size: int = -1):
        """ Creates a copy of the current corpus, resizing the datasets if requested

        :param training_set_size: size of the training set
        :param validation_set_size: size of the validation set
        :param test_set_size: size of the test set
        :return: a resized copy of the corpus
        """
        training_set = self.training_set.copy()

        validation_set = None
        if self.validation_set is not None:
            validation_set = self.validation_set.copy()

        test_set = None
        if self.test_set is not None:
            test_set = self.test_set.copy()

        copy = Corpus(training_set=training_set,
                      validation_set=validation_set,
                      test_set=test_set,
                      name=self.name)

        copy.resize(training_set_size=training_set_size,
                    validation_set_size=validation_set_size,
                    test_set_size=test_set_size)

        return copy

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, training_set: Dataset):
        self.__training_set = training_set
        if training_set is not None:
            self.input_size = training_set.input_size
            self.output_size = training_set.output_size

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, test_set: Dataset):
        self.__test_set = test_set
        if test_set is not None:
            self.input_size = test_set.input_size
            self.output_size = test_set.output_size

    @property
    def validation_set(self):
        return self.__validation_set

    @validation_set.setter
    def validation_set(self, validation_set: Dataset):
        self.__validation_set = validation_set
        if validation_set is not None:
            self.input_size = validation_set.input_size
            self.output_size = validation_set.output_size

    @property
    def name(self):
        return self.__name

    @property
    def input_size(self):
        return self.__input_size

    @input_size.setter
    def input_size(self, input_size: int):
        if self.__input_size == 0:
            self.__input_size = input_size
        else:
            if input_size != self.__input_size:
                raise RuntimeError('Error checking input size: current is {}, trying to set to {}',
                                   self.__input_size, input_size)

    @property
    def output_size(self):
        return self.__output_size

    @output_size.setter
    def output_size(self, output_size: int):
        if self.__output_size == 0:
            self.__output_size = output_size
        else:
            if output_size != self.__output_size:
                raise RuntimeError('Error checking output size: current is {}, trying to set to {}',
                                   self.__output_size, output_size)

    @property
    def length(self):
        total_size = self.training_set.length + self.validation_set.length + self.test_set.length
        return total_size

    @property
    def count_categories(self):
        """ Returns the number of distinct labels in the output data
        """
        if len(self.training_set.output_data.shape) == 1:
            dimension = 1
        else:
            dimension = self.training_set.output_data.shape[1]
        if dimension == 1:
            return self.training_set.count_unique_values
        else:
            return dimension

    @property
    def min_output(self):
        return self.training_set.min_output

    @property
    def max_output(self):
        return self.training_set.max_output

    @property
    def average_output(self):
        return self.training_set.average_output

    def split_training_set(self, size: int, start: int = 0):
        """ Splits the training set into two new subsets whose elements are copied from the original training set

        :param size: split size (number of elements of the split dataset)
        :param start: split start position
        :return: two new datasets: split, remain
        """
        return self.training_set.split(size=size, start=start)

    def get_validation_from_training_set(self,
                                         validation_size: int,
                                         start_position: int = 0,
                                         merge_first: bool = False,
                                         preserve_training_set: bool = False):
        """ Creates a validation set by splitting the training set, overwriting the original datasets

        :param validation_size: validation set size (must be smaller than training set)
        :param start_position: split start position
        :param merge_first: merge training and validation sets before split
        :param preserve_training_set: preserve the original training set (don't override with the split remain)
        """
        if merge_first and self.validation_set is not None:
            self.training_set.merge(self.validation_set)

        split, remain = self.split_training_set(size=validation_size, start=start_position)

        if self.validation_set is None:
            self.validation_set = split
        else:
            self.validation_set.merge(split)

        if not preserve_training_set:
            self.training_set = remain

    def get_validation_set_k_fold(self, fold: int, k: int):
        """
        Splits the training set to extract a validation set according to the k-fold rule
        :param fold: current fold
        :param k: number of folds
        """
        return self.training_set.split_k_fold(fold=fold, k=k)


class CorpusGenerator:
    """ A corpus generator contains three file sets: train, validation, and test
    """

    def __init__(self,
                 training_set_files: DatasetFileIterator,
                 validation_set_files: DatasetFileIterator,
                 test_set_files: DatasetFileIterator):
        """ Creates a new Corpus based on Set of files

        :param training_set_files: train set files
        :param validation_set_files: validation set files
        :param test_set_files: test set files
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

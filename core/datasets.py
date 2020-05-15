from __future__ import annotations

import os.path

import numpy as np
from keras_preprocessing.image import DirectoryIterator

from utils.dataset_utils import count_unique_values, merge_datasets


class Dataset:
    """ A Dataset is a pair of two arrays of the same dimension: input and output data
    """

    def __init__(self,
                 input_data: np.ndarray,
                 output_data: np.ndarray = None,
                 name: str = ''):
        """ Creates a new Dataset instance

        :param input_data: input data array
        :param output_data: output data array (optional)
        :param name: dataset name
        """
        if input_data is None:
            raise ValueError('Cannot instantiate dataset: no input data array')

        if output_data is not None:
            if len(output_data) != len(input_data):
                raise ValueError('Input and output data arrays must share the same length')

        self.input_data = input_data
        self.output_data = output_data
        self.name = name

    @property
    def input_data(self):
        return self.__input_data

    @input_data.setter
    def input_data(self, input_data):
        self.__input_data = input_data
        self.__length = len(input_data)
        self.__input_size = input_data.shape[1]

    @property
    def output_data(self):
        return self.__output_data

    @output_data.setter
    def output_data(self, output_data):
        self.__output_data = output_data
        self.__output_size = 0
        if output_data is not None:
            if len(output_data.shape) < 2:
                self.__output_size = 1
            else:
                self.__output_size = output_data.shape[1]

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def length(self):
        return self.__length

    @property
    def input_size(self):
        return self.__input_size

    @property
    def output_size(self):
        return self.__output_size

    @property
    def count_unique_values(self):
        return count_unique_values(self.output_data)

    @property
    def min_output(self):
        return np.min(self.output_data)

    @property
    def max_output(self):
        return np.max(self.output_data)

    @property
    def average_output(self):
        return np.average(self.output_data)

    def copy(self, name: str = None):
        """ Generates a deep copy of the dataset by duplicating the input and output data

        :param name: name of the copy dataset
        :return: a new dataset which is a deep copy of the original dataset
        """
        input_copy = np.copy(self.input_data)
        output_copy = np.copy(self.output_data)
        if name:
            new_name = name
        else:
            new_name = self.name

        return Dataset(input_data=input_copy, output_data=output_copy, name=new_name)

    def shuffle(self):
        """ Shuffles the order of the elements
        """
        p = np.random.permutation(self.length)
        self.input_data = self.input_data[p]
        self.output_data = self.output_data[p]

    def split(self, size: int, start: int = 0):
        """ Splits the elements of the input and output data into two subsets:

        - split set: a copy of the original dataset with the given size, from the given start position
        - remaining set: a copy of the remaining elements of the original set
        :param size: first subset size
        :param start: split position for the first subset
        :returns: a pair of datasets, the split set first and the remaining set second
        """
        if size > self.length:
            raise ValueError('Split size ({}) must be smaller than the dataset size ({})'
                             .format(size, self.length))

        stop = start + size

        if stop <= self.length:
            # split ends before the end of the array
            split_range = list(range(start, stop))
            remain_range = list(range(0, start)) + list(range(stop, self.length))

        else:
            # split after the end of the array
            stop -= self.length
            split_range = list(range(start, self.length)) + list(range(0, stop))
            remain_range = list(range(stop, start))

        split_set = Dataset(self.input_data[split_range], self.output_data[split_range])
        remain_set = Dataset(self.input_data[remain_range], self.output_data[remain_range])

        return split_set, remain_set

    def split_k_fold(self, fold: int, k: int):
        """ Splits the set in two, according to the k-fold partition rule

        :param fold: current fold to be split (n-th fold in k total folds)
        :param k: total number of folds
        """
        if fold > k:
            raise ValueError('Fold ({}) cannot be greater than K ({}) in k-fold validation'.format(fold, k))

        set_length = self.length
        num_samples_fold = set_length // k
        start = fold * num_samples_fold

        return self.split(size=num_samples_fold, start=start)

    def merge(self, second_dataset):
        """ Merges the current dataset with the second dataset input and output data, overriding the current data

        :param second_dataset: dataset to be merged
        """
        if second_dataset is None:
            raise RuntimeError('Cannot merge datasets: no dataset passed to merge with')

        if self.input_size != second_dataset.input_size:
            raise RuntimeError('Cannot merge datasets: input dimensions are different')

        if self.output_size != second_dataset.output_size:
            raise RuntimeError('Cannot merge datasets: output dimensions are different')

        self.input_data = merge_datasets(self.input_data, second_dataset.input_data)
        self.output_data = merge_datasets(self.output_data, second_dataset.output_data)

    def to_array_pair(self):
        """ Returns the pair of input and output sets
        """
        return self.input_data, self.output_data

    def flatten_input_data(self):
        """ Flattens input data, assuming the first dimension is the number of samples (which is preserved)
        """
        new_shape = self.input_data.shape[0], np.prod(self.input_data.shape[1:])
        self.input_data = self.input_data.reshape(new_shape)


class DatasetFileIterator:
    """ Contains a directory iterator that can be sequentially iterated over to get a set of files
    """

    def __init__(self,
                 directory_iterator: DirectoryIterator,
                 name: str):
        """ Creates a new DatasetFileIterator, embedding a directory iterator
        :param directory_iterator: DirectoryIterator to be embedded
        :param name: dataset file iterator name
        """
        if directory_iterator is None:
            raise RuntimeError('No directory iterator passed creating GenSet')

        self.name = name
        self.__directory_iterator = directory_iterator

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def directory_iterator(self):
        return self.__directory_iterator

    @property
    def path(self):
        return self.directory_iterator.directory

    @property
    def batch_size(self):
        return self.directory_iterator.batch_size

    @property
    def file_list(self):
        return os.listdir(self.path)

    @property
    def length(self):
        return self.directory_iterator.samples

    @property
    def image_shape(self):
        return self.directory_iterator.image_shape

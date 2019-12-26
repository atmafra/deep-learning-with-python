import os
from enum import Enum

import numpy as np
from keras_preprocessing.image import DirectoryIterator

import utils.dataset_utils as dsu
from utils.file_utils import str_to_filename


class SetDataFormat(Enum):
    TXT = 1
    BIN = 2
    NPY = 3


class Set:
    """A Set is a pair of two arrays: input and output data

    Args:
        input_data (np.ndarray): input data array
        output_data (np.ndarray): output data array

    """

    def __init__(self,
                 input_data: np.ndarray,
                 output_data: np.ndarray = None):
        """Create a new Set instance
        """
        assert input_data is not None, 'Cannot create Set with no input data'

        if len(output_data) != len(input_data):
            raise ValueError('Input and Output data must share the same length')

        self.__input_data = input_data
        self.__output_data = output_data

    @property
    def input_data(self):
        return self.__input_data

    @input_data.setter
    def input_data(self, input_data):
        self.__input_data = input_data

    @property
    def output_data(self):
        return self.__output_data

    @output_data.setter
    def output_data(self, output_data):
        self.__output_data = output_data

    @property
    def length(self):
        return len(self.input_data)

    @property
    def input_size(self):
        """Returns the size of the input elements
        """
        return self.input_data.shape[1]

    @property
    def output_size(self):
        """Returns the size of the output elements
        """
        shape = self.output_data.shape
        if shape is None:
            return 0
        if len(shape) == 1:
            return 1
        return shape[1]

    @property
    def count_unique_values(self):
        return dsu.count_unique_values(self.output_data)

    @property
    def min_output(self):
        return np.min(self.output_data)

    @property
    def max_output(self):
        return np.max(self.output_data)

    @property
    def average_output(self):
        return np.average(self.output_data)

    def copy(self):
        """Generates a copy of the set, by duplicating the input and output data
        """
        input_copy = np.copy(self.input_data)
        output_copy = np.copy(self.output_data)
        return Set(input_copy, output_copy)

    def shuffle(self):
        """Shuffles the order of the elements
        """
        p = np.random.permutation(self.length)
        self.input_data = self.input_data[p]
        self.output_data = self.output_data[p]

    def split(self, size: int, start: int = 0):
        """Splits the two arrays into two subsets:
           1. the first of the given size and starting from the given start element
           2. the remain of the original set

        Args:
            size (int): first subset size
            start (int): split position for the first subset

         """
        if size > self.length:
            raise ValueError('Split size ({}) must be smaller than the dataset size ({})'
                             .format(size, self.length))

        stop = start + size

        if stop <= self.length:
            # split before the end of the array
            split_range = list(range(start, stop))
            remain_range = list(range(0, start)) + list(range(stop, self.length))

        else:
            # split after the end of the array
            stop -= self.length
            split_range = list(range(start, self.length)) + list(range(0, stop))
            remain_range = list(range(stop, start))

        split_set = Set(self.input_data[split_range], self.output_data[split_range])
        remain_set = Set(self.input_data[remain_range], self.output_data[remain_range])

        return split_set, remain_set

    def split_k_fold(self, fold: int, k: int):
        """Splits the set in two, according to the k-fold partition rule

        Args:
            fold (int) : current fold to be split (n-th fold in k total folds)
            k    (int) : total number of folds

        """
        if fold > k:
            raise ValueError('Fold ({}) cannot be greater than K ({}) in k-fold validation'.format(fold, k))

        set_length = self.length
        num_samples_fold = set_length // k
        start = fold * num_samples_fold

        return self.split(size=num_samples_fold, start=start)

    def to_datasets(self):
        """Returns the pair of input and output sets
        """
        return self.input_data, self.output_data

    def flatten_input_data(self):
        """Flattens input data, asuming the first dimension is the number of samples (which is preserved)
        """
        new_shape = self.input_data.shape[0], np.prod(self.input_data.shape[1:])
        self.input_data = self.input_data.reshape(new_shape)

    def save(self, path: str,
             filename: str,
             file_format: SetDataFormat = SetDataFormat.TXT):
        """Saves the input and output data to files

        Args:
            path (str): system path of the save directory
            filename (str): root file name
            file_format (SetDataFormat): file format: text (.txt), binary (.h5) or numpy (.npy)

        """
        root_name = str_to_filename(filename)
        input_filepath = os.path.join(path, root_name) + '-input'
        output_filepath = os.path.join(path, root_name) + '-output'

        if file_format == SetDataFormat.TXT:
            np.savetxt(fname=input_filepath + '.txt', X=self.input_data)
            np.savetxt(fname=output_filepath + '.txt', X=self.output_data)

        elif file_format == SetDataFormat.BIN:
            np.save(file=input_filepath + '.h5', arr=self.input_data, allow_pickle=True)
            np.save(file=output_filepath + '.h5', arr=self.output_data, allow_pickle=True)

        elif file_format == SetDataFormat.NPY:
            np.save(file=input_filepath + '.npy', arr=self.input_data, allow_pickle=False)
            np.save(file=output_filepath + '.npy', arr=self.output_data, allow_pickle=False)


class SetFiles:
    """Contains a directory iterator that can be sequentially iterated over to get a set of files
    """

    def __init__(self, directory_iterator: DirectoryIterator):
        if directory_iterator is None:
            raise RuntimeError('No directory iterator passed creating GenSet')

        self.__directory_iterator = directory_iterator

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

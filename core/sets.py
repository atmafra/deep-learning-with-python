import numpy as np
from keras.utils import Sequence

import utils.dataset_utils as dsu


class Set:
    """A Set is a pair of two arrays: input and output data

    Args:
        input_data (np.array): input data array
        output_data (np.array): output data array

    """

    def __init__(self,
                 input_data: np.array,
                 output_data: np.array = None):
        """Create a new Set instance
        """
        assert input_data is not None, 'Cannot create Set with no input data'

        self.input_data = input_data

        if len(output_data) != len(self.input_data):
            raise ValueError('Input data and Output data must have the same length')

        self.output_data = output_data

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


class SetGenerator:
    """Contains a generator that can be sequentially iterated over to get a set
    """

    def __init__(self, generator: Sequence):
        if generator is None:
            raise RuntimeError('No generator passed creating GenSet')

        self.__generator = generator

    @property
    def generator(self):
        return self.__generator

from __future__ import annotations

import os.path

import numpy as np
from keras_preprocessing.image import DirectoryIterator

from utils.dataset_utils import count_unique_values, merge_datasets, one_hot_encode, three_way_split


class Dataset:
    """ A Dataset is a pair of two arrays of the same dimension: input and output data
    """

    def __init__(self,
                 input_data: np.ndarray,
                 output_data: np.ndarray = None,
                 sample_weights: np.ndarray = None,
                 name: str = ''):
        """ Creates a new Dataset instance

        :param input_data: input data array
        :param output_data: output data array (optional)
        :param sample_weights: array of relative weights for each (input, output) sample pair
        :param name: dataset name
        """
        if input_data is None:
            raise ValueError('Cannot instantiate dataset: no input data array')

        if output_data is not None:
            if len(output_data) != len(input_data):
                raise ValueError('Input and output data arrays must share the same length')

        if sample_weights is not None:
            if len(sample_weights) != len(input_data):
                raise ValueError('Sample weights and input data arrays must share the same length')

        self.input_data = input_data
        self.output_data = output_data
        self.sample_weights = sample_weights
        self.name = name

    @property
    def input_data(self):
        return self.__input_data

    @input_data.setter
    def input_data(self, input_data):
        self.__input_data = input_data
        self.__length = len(input_data)
        if len(input_data.shape) > 1:
            self.__input_size = input_data.shape[1]
        else:
            self.__input_size = 1

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
    def sample_weights(self):
        return self.__sample_weights

    @sample_weights.setter
    def sample_weights(self, sample_weights):
        self.__sample_weights = sample_weights

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
        if self.output_data is not None:
            output_copy = np.copy(self.output_data)
        else:
            output_copy = None

        if self.sample_weights is not None:
            sample_weights_copy = np.copy(self.sample_weights)
        else:
            sample_weights_copy = None

        if name:
            new_name = name
        else:
            new_name = self.name

        return Dataset(input_data=input_copy,
                       output_data=output_copy,
                       sample_weights=sample_weights_copy,
                       name=new_name)

    def shuffle(self):
        """ Shuffles the order of the elements
        """
        p = np.random.permutation(self.length)
        self.input_data = self.input_data[p]
        self.output_data = self.output_data[p]
        self.sample_weights = self.sample_weights[p]

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

        # split
        input_split = self.input_data[split_range]

        if self.output_data is not None:
            output_split = self.output_data[split_range]
        else:
            output_split = None

        if self.sample_weights is not None:
            sample_weights_split = self.sample_weights[split_range]
        else:
            sample_weights_split = None

        split_set = Dataset(input_data=input_split,
                            output_data=output_split,
                            sample_weights=sample_weights_split,
                            name=self.name)

        # remain
        input_remain = self.input_data[remain_range]

        if self.output_data is not None:
            output_remain = self.output_data[remain_range]
        else:
            output_remain = None

        if self.sample_weights is not None:
            sample_weights_remain = self.sample_weights[remain_range]
        else:
            sample_weights_remain = None

        remain_set = Dataset(input_data=input_remain,
                             output_data=output_remain,
                             sample_weights=sample_weights_remain,
                             name=self.name)

        return split_set, remain_set

    def three_way_split(self,
                        sampling_rate: float = 1.,
                        partition_1_share: float = 0.5,
                        partition_2_share: float = 0.25,
                        partition_3_share: float = 0.25) -> tuple:
        """

        :param sampling_rate:
        :param partition_1_share:
        :param partition_2_share:
        :param partition_3_share:
        :return:
        """
        input1, input2, input3 = None, None, None
        output1, output2, output3 = None, None, None
        weights1, weights2, weights3 = None, None, None

        if self.input_data is not None:
            input1, input2, input3 = three_way_split(self.input_data,
                                                     sampling_rate=sampling_rate,
                                                     partition_1_share=partition_1_share,
                                                     partition_2_share=partition_2_share,
                                                     partition_3_share=partition_3_share)

        if self.output_data is not None:
            output1, output2, output3 = three_way_split(self.output_data,
                                                        sampling_rate=sampling_rate,
                                                        partition_1_share=partition_1_share,
                                                        partition_2_share=partition_2_share,
                                                        partition_3_share=partition_3_share)

        if self.sample_weights is not None:
            weights1, weights2, weights3 = three_way_split(self.sample_weights,
                                                           sampling_rate=sampling_rate,
                                                           partition_1_share=partition_1_share,
                                                           partition_2_share=partition_2_share,
                                                           partition_3_share=partition_3_share)
        dataset1 = Dataset(input_data=input1,
                           output_data=output1,
                           sample_weights=weights1,
                           name=self.name + ' (1/3)')

        dataset2 = Dataset(input_data=input2,
                           output_data=output2,
                           sample_weights=weights2,
                           name=self.name + ' (2/3)')

        dataset3 = Dataset(input_data=input3,
                           output_data=output3,
                           sample_weights=weights3,
                           name=self.name + ' (3/3)')

        return dataset1, dataset2, dataset3

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
        if self.output_data is not None:
            self.output_data = merge_datasets(self.output_data, second_dataset.output_data)
        if self.sample_weights is not None:
            self.sample_weights = merge_datasets(self.sample_weights, second_dataset.sample_weights)

    def to_array_pair(self):
        """ Returns the pair of input and output sets
        """
        return self.input_data, self.output_data

    def flatten_input_data(self):
        """ Flattens input data, assuming the first dimension is the number of samples (which is preserved)
        """
        new_shape = self.input_data.shape[0], np.prod(self.input_data.shape[1:])
        self.input_data = self.input_data.reshape(new_shape)

    def generator(self,
                  offset: int = 0,
                  block_size: int = 1):
        """ Returns a generator that yields the Dataset samples in blocks

        :param offset: first sample index offset
        :param block_size: number of samples per block
        :return: (input, output, sample_weights) block
        """
        blocks = (self.length - offset) // block_size
        for i in range(blocks):
            block_start = i * block_size
            if i < blocks:
                block_end = (i + 1) * block_size
            else:
                block_end = self.length

            input_block = self.input_data[block_start:block_end]

            if self.output_data is not None:
                output_block = self.output_data[block_start:block_end]
            else:
                output_block = None

            if self.sample_weights is not None:
                sample_weights_block = self.sample_weights[block_start:block_end]
            else:
                sample_weights_block = None

            yield input_block, output_block, sample_weights_block

    def one_hot_encode(self, encode_input: bool = False, encode_output: bool = False):
        """ Executes one-hot vector encoding of the input and output data

        :param encode_input: encode input data
        :param encode_output: encode output data
        """
        if encode_input:
            self.input_data = one_hot_encode(sequence=self.input_data, categories=self.length)
        if encode_output:
            self.output_data = one_hot_encode(sequence=self.output_data, categories=self.length)


class DatasetFileIterator:
    """ Contains a directory iterator that can be sequentially iterated over to get a set of files
    """

    def __init__(self, directory_iterator: DirectoryIterator, name: str):
        """ Creates a new DatasetFileIterator, embedding a directory iterator

        :param directory_iterator: DirectoryIterator to be embedded
        :param name: dataset file iterator name
        """
        if directory_iterator is None:
            raise RuntimeError('No directory iterator passed creating GenSet')

        super().__init__(name)
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

import numpy as np

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
        self.length = len(self.input_data)

        if len(output_data) != self.length:
            raise ValueError('Input data and Output data must have the same length')

        self.output_data = output_data

    def copy(self):
        """Duplicates the set
        """
        input_copy = np.copy(self.input_data)
        output_copy = np.copy(self.output_data)
        return Set(input_copy, output_copy)

    def shuffle(self):
        """Shuffles the element order
        """
        p = np.random.permutation(self.length)
        self.input_data = self.input_data[p]
        self.output_data = self.output_data[p]

    def split(self, size: int, start: int = 0):
        """Splits the two arrays into a two subsets, the first of the given size
           and starting from the given start element

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


class Corpus:
    """A Corpus is a pair of two subsets: Training Set and Test Set

        Args:
            training_set (Set): training set
            test_set (Set): test set

        Members:
            training_set (Set): training set
            test_set (Set): test set
            validation_set (Set): a split from the training set, used to perform cross-validation

    """

    def __init__(self, training_set: Set, test_set: Set):
        self.training_set = training_set
        self.test_set = test_set

    @staticmethod
    def from_datasets(
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

    @staticmethod
    def from_tuple(corpus: tuple):
        """Creates a new Corpus from a pair (tuple) of two sets
           Each of these sets must have two subsets: input and output sets

        Args:
            corpus: a pair of arrays that represent training and test sets

        """
        (training_inputs, training_outputs), (test_inputs, test_outputs) = dsu.separate_corpus(corpus)
        return Corpus.from_datasets(training_inputs, training_outputs, test_inputs, test_outputs)

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
    def min_ouptut(self):
        return self.training_set.min_output

    @property
    def max_output(self):
        return self.training_set.max_output

    @property
    def average_output(self):
        return self.training_set.average_output

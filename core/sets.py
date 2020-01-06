import os

import numpy as np
from keras_preprocessing.image import DirectoryIterator

import utils.dataset_utils as dsu
from core.file_structures import SetFileStructure, DatafileFormat


class Set:
    """A Set is a pair of two arrays of the same dimension: input and output data

    Args:
        input_data (ndarray): input data array
        output_data (ndarray): output data array
        name (str): set name

    """

    def __init__(self,
                 input_data: np.ndarray,
                 output_data: np.ndarray = None,
                 name: str = ''):
        """Create a new Set instance
        """
        if input_data is None:
            raise ValueError('Cannot create set with no input data')

        if output_data is not None:
            if len(output_data) != len(input_data):
                raise ValueError('Input and Output data must share the same length')

        self.input_data = input_data
        self.output_data = output_data
        self.__name = name

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
    def name(self):
        return self.__name

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
             filename_input: str,
             filename_output: str,
             file_format: DatafileFormat):
        """Saves the input and output data to files

        Args:
            path (str): system path of the save directory
            filename_input (str): input data filename
            filename_output (str): output data filename
            file_format (DatafileFormat): data file format

        """
        # root_name = str_to_filename(filename)
        input_filepath = os.path.join(path, filename_input)
        output_filepath = os.path.join(path, filename_output)

        if file_format == DatafileFormat.TXT:
            np.savetxt(fname=input_filepath, X=self.input_data)
            np.savetxt(fname=output_filepath, X=self.output_data)

        elif file_format == DatafileFormat.NPY:
            np.save(file=input_filepath, arr=self.input_data, allow_pickle=False)
            np.save(file=output_filepath, arr=self.output_data, allow_pickle=False)

    def save_file_structure(self,
                            set_file_structure: SetFileStructure):
        """"Saves the set data files according to the set file structure

        Args:
            set_file_structure (SetFileStructure): paths and filenames of the data files

        """
        self.save(path=set_file_structure.path,
                  filename_input=set_file_structure.input_data_filename,
                  filename_output=set_file_structure.output_data_filename,
                  file_format=set_file_structure.file_format)

    @classmethod
    def from_files(cls, path: str,
                   input_data_filename: str,
                   output_data_filename: str,
                   file_format: DatafileFormat = DatafileFormat.TXT,
                   name: str = ''):
        """Creates a new set by reading input and output data files

        Args:
            path (str): system path of the data files directory
            input_data_filename (str): input file name
            output_data_filename (str): output file name
            file_format (DatafileFormat): data files format
            name (str): set name

        """
        input_data = None
        output_data = None
        input_filepath = os.path.join(path, input_data_filename)
        output_filepath = os.path.join(path, output_data_filename)

        if file_format == DatafileFormat.TXT:
            input_data = np.loadtxt(fname=input_filepath)
            output_data = np.loadtxt(fname=output_filepath)

        elif file_format == DatafileFormat.NPY:
            input_data = np.load(file=input_filepath)
            output_data = np.load(file=output_filepath)

        return Set(input_data=input_data,
                   output_data=output_data,
                   name=name)

    @classmethod
    def from_file_structure(cls, set_file_structure: SetFileStructure, name: str = ''):
        return Set.from_files(path=set_file_structure.path,
                              input_data_filename=set_file_structure.input_data_filename,
                              output_data_filename=set_file_structure.output_data_filename,
                              file_format=set_file_structure.file_format,
                              name=name)

    def get_canonical_file_structure(self, path: str,
                                     file_format: DatafileFormat):
        """Creates a new Set File Structure according to canonical definitions

        Args:
            path (str): system path of data files directory
            file_format (DatafileFormat): data files format

        """
        return SetFileStructure.get_canonical(path=path, set_name=self.name, file_format=file_format)


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

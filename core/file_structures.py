import os.path

import numpy as np

from core.corpus import Corpus
from core.datasets import Dataset
from utils.file_utils import str_to_filename


# class DatafileFormat(Enum):
#     TXT = 1
#     NPY = 2
#     NPZ = 3


class DatasetFileStructure:
    """A structure consisting of a path and relative file locations
       plus methods for saving and loading datasets

    Members:
        path: system path of data files

    """

    def __init__(self, path: str):
        self.__path: str = path

    @property
    def path(self):
        return self.__path

    # @property
    # def file_format(self):
    #     raise NotImplemented('Must be implemented in the subclass')

    @classmethod
    def get_canonical(cls, path: str, set_name: str):
        """Returns a canonical file structure, based on the path and on the set name
        """
        raise NotImplemented('Must be implemented in the subclass')

    def save_dataset(self, dataset: Dataset):
        """Saves the dataset according to the file structure
        """
        raise NotImplemented('Must be implemented in the subclass')

    def load_dataset(self, name: str):
        """Creates a new dataset from previously saved data files
        """
        raise NotImplemented('Must be implemented in the subclass')


class DatasetFileStructureMultipleFiles(DatasetFileStructure):
    """A structure of path and file names (input and output)
       used to save and load set data files

    Members:
        path: system path of data files
        input_data_filename: input data file name
        output_data_filename: output data file name

    """

    def __init__(self, path: str,
                 input_data_filename: str,
                 output_data_filename: str):
        """Creates a new SetFileStructure
        """
        DatasetFileStructure.__init__(self, path)
        self.__input_data_filename: str = input_data_filename
        self.__output_data_filename: str = output_data_filename

    # @DatasetFileStructure.file_format.getter
    # def file_format(self):
    #     raise NotImplemented('Must be implemented in the subclass')

    @property
    def input_data_filename(self):
        return self.__input_data_filename

    @property
    def output_data_filename(self):
        return self.__output_data_filename

    @property
    def input_data_filepath(self):
        return os.path.join(self.path, self.input_data_filename)

    @property
    def output_data_filepath(self):
        return os.path.join(self.path, self.output_data_filename)

    def get_canonical(cls, path: str, set_name: str):
        raise NotImplemented('Must be implemented in the subclass')

    def save_dataset(self, dataset: Dataset):
        raise NotImplemented('Must be implemented in the subclass')

    def load_dataset(self, name: str):
        if not os.path.exists(self.input_data_filepath):
            raise FileNotFoundError('input data file: \"{}\"'.format(self.input_data_filepath))
        if not os.path.exists(self.output_data_filepath):
            raise FileNotFoundError('output data file: \"{}\"'.format(self.output_data_filepath))


class DatasetFileStructureSingleFile(DatasetFileStructure):
    """A structure of path and file names (input and output)
       used to save and load set data files

    Members:
        path: system path of data files
        data_filename: data file name

    """

    def __init__(self, path: str,
                 data_filename: str):
        """Creates a new SetFileStructure
        """
        DatasetFileStructure.__init__(self, path)
        self.__data_filename: str = data_filename

    # @DatasetFileStructure.file_format.getter
    # def file_format(self):
    #     raise NotImplemented('Must be implemented in the subclass')

    @property
    def data_filename(self):
        return self.__data_filename

    @property
    def data_filepath(self):
        return os.path.join(self.path, self.data_filename)

    @classmethod
    def get_canonical(cls, path: str, set_name: str):
        raise NotImplemented('Must be implemented in the subclass')

    def save_dataset(self, dataset: Dataset):
        raise NotImplemented('Must be implemented in the subclass')

    def load_dataset(self, name: str):
        if not os.path.exists(self.data_filepath):
            raise FileNotFoundError('input data file: \"{}\"'.format(self.data_filepath))


class DatasetFileStructureText(DatasetFileStructureMultipleFiles):
    """A structure of path and file names (input and output)
       used to save and load set data files

    Members:
        path: system path of data files
        input_data_filename: input data file name
        output_data_filename: output data file name

    """

    def __init__(self, path: str,
                 input_data_filename: str,
                 output_data_filename: str):
        """Creates a new SetFileStructure
        """
        DatasetFileStructureMultipleFiles.__init__(self, path=path,
                                                   input_data_filename=input_data_filename,
                                                   output_data_filename=output_data_filename)

    @classmethod
    def get_canonical(cls, path: str, set_name: str):
        """Creates a new Set File Structure according to canonical definitions

        Args:
            path (str): system path of data files directory
            set_name (str): set name

        """
        input_data_filename = str_to_filename(set_name) + '-input.txt'
        output_data_filename = str_to_filename(set_name) + '-output.txt'

        return DatasetFileStructureText(path=path,
                                        input_data_filename=input_data_filename,
                                        output_data_filename=output_data_filename)

    def save_dataset(self, dataset: Dataset):
        np.savetxt(fname=self.input_data_filepath, X=dataset.input_data)
        np.savetxt(fname=self.output_data_filepath, X=dataset.output_data)

    def load_dataset(self, name: str):
        try:
            DatasetFileStructureMultipleFiles.load_dataset(self, name=name)
        except FileNotFoundError:
            return None
        input_data = np.loadtxt(fname=self.input_data_filepath)
        output_data = np.loadtxt(fname=self.output_data_filepath)
        return Dataset(input_data=input_data, output_data=output_data, name=name)


class DatasetFileStructureNumpy(DatasetFileStructureMultipleFiles):
    """A structure of path and file names (input and output)
       used to save and load set data files

    Members:
        path: system path of data files
        input_data_filename: input data file name
        output_data_filename: output data file name

    """

    def __init__(self, path: str,
                 input_data_filename: str,
                 output_data_filename: str):
        """Creates a new SetFileStructure
        """
        DatasetFileStructureMultipleFiles.__init__(self, path=path,
                                                   input_data_filename=input_data_filename,
                                                   output_data_filename=output_data_filename)

    @classmethod
    def get_canonical(cls, path: str, set_name: str):
        """Creates a new Set File Structure according to canonical definitions

        Args:
            path (str): system path of data files directory
            set_name (str): set name

        """
        input_data_filename = str_to_filename(set_name) + '-input.npy'
        output_data_filename = str_to_filename(set_name) + '-output.npy'

        return DatasetFileStructureNumpy(path=path,
                                         input_data_filename=input_data_filename,
                                         output_data_filename=output_data_filename)

    def save_dataset(self, dataset: Dataset):
        np.save(file=self.input_data_filepath, arr=dataset.input_data, allow_pickle=False)
        np.save(file=self.output_data_filepath, arr=dataset.output_data, allow_pickle=False)

    def load_dataset(self, name: str):
        try:
            DatasetFileStructureMultipleFiles.load_dataset(self, name=name)
        except FileNotFoundError:
            return None
        input_data = np.load(file=self.input_data_filepath)
        output_data = np.load(file=self.output_data_filepath)
        return Dataset(input_data=input_data, output_data=output_data, name=name)


class DatasetFileStructureNumpyCompressed(DatasetFileStructureSingleFile):
    """A structure of path and file names (input and output)
       used to save and load set data files

    Members:
        path: system path of data files
        data_filename: data file name

    """

    def __init__(self, path: str,
                 data_filename: str):
        """Creates a new SetFileStructure
        """
        DatasetFileStructureSingleFile.__init__(self, path=path,
                                                data_filename=data_filename)

    @classmethod
    def get_canonical(cls, path: str, set_name: str):
        """Creates a new Set File Structure according to canonical definitions

        Args:
            path (str): system path of data files directory
            set_name (str): set name

        """
        data_filename = str_to_filename(set_name) + '.npz'

        return DatasetFileStructureNumpyCompressed(path=path,
                                                   data_filename=data_filename)

    def save_dataset(self, dataset: Dataset):
        os.makedirs(os.path.dirname(self.data_filepath), exist_ok=True)
        np.savez_compressed(file=self.data_filepath,
                            input=dataset.input_data,
                            output=dataset.output_data)

    def load_dataset(self, name: str):
        try:
            DatasetFileStructureSingleFile.load_dataset(self, name=name)
        except FileNotFoundError:
            return None
        datasets = np.load(file=self.data_filepath)
        return Dataset(input_data=datasets['input'],
                       output_data=datasets['output'],
                       name=name)


class CorpusFileStructure:
    """A set of dataset file structures: training, test (optional), and validation (optional)
    """

    def __init__(self,
                 training_file_structure: DatasetFileStructure,
                 test_file_structure: DatasetFileStructure,
                 validation_file_structure: DatasetFileStructure = None):
        """Creates a new corpus file structure
        """
        self.__training_file_structure: DatasetFileStructure = training_file_structure
        self.__test_file_structure: DatasetFileStructure = test_file_structure
        self.__validation_file_structure: DatasetFileStructure = validation_file_structure

    @property
    def training_file_structure(self):
        return self.__training_file_structure

    @property
    def test_file_structure(self):
        return self.__test_file_structure

    @property
    def validation_file_structure(self):
        return self.__validation_file_structure

    @classmethod
    def get_canonical(cls, corpus_name: str, base_path: str):
        """Creates a new CorpusFileStructure according to the conventions

        Args:
            corpus_name (str): corpus name
            base_path (str): system path of the base directory (subdirectories will be created here)

        """
        training_set_name = corpus_name + ' - train'
        training_file_structure = \
            DatasetFileStructureNumpyCompressed.get_canonical(path=base_path, set_name=training_set_name)

        test_set_name = corpus_name + ' - test'
        test_file_structrure = \
            DatasetFileStructureNumpyCompressed.get_canonical(path=base_path, set_name=test_set_name)

        validation_set_name = corpus_name + ' - validation'
        validation_file_structure = \
            DatasetFileStructureNumpyCompressed.get_canonical(path=base_path, set_name=validation_set_name)

        return CorpusFileStructure(training_file_structure=training_file_structure,
                                   test_file_structure=test_file_structrure,
                                   validation_file_structure=validation_file_structure)

    def save_corpus(self, corpus: Corpus):
        """Saves the corpus data, by saving its set to data files according do their file structures

        Args:
            corpus (Corpus): corpus to be saved according to the file structure

        """
        if corpus.training_set is not None:
            self.training_file_structure.save_dataset(dataset=corpus.training_set)

        if corpus.test_set is not None:
            self.test_file_structure.save_dataset(dataset=corpus.test_set)

        if corpus.validation_set is not None:
            self.validation_file_structure.save_dataset(dataset=corpus.validation_set)

    def load_corpus(self, corpus_name: str,
                    datasets_base_name: str):
        """Creates a corpus by loading feature files directly from disk

        Args:
            corpus_name (str): name of the corpus
            datasets_base_name (str): common part (basename) of the datasets

        """
        training_set = self.training_file_structure.load_dataset(name=datasets_base_name + ' - train')
        test_set = self.test_file_structure.load_dataset(name=datasets_base_name + ' - test')
        validation_set = self.validation_file_structure.load_dataset(name=datasets_base_name + ' - validation')

        return Corpus(training_set=training_set,
                      test_set=test_set,
                      validation_set=validation_set,
                      name=corpus_name)

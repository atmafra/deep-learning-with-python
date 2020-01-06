import os.path
from enum import Enum

from utils.file_utils import str_to_filename


class DatafileFormat(Enum):
    TXT = 1
    NPY = 2


class SetFileStructure:
    """A structure of path and file names (input and output)
       used to save and load set data files

    Members:
        path: system path of data files
        input_data_filename: input data file name
        output_data_filename: output data file name
        file_format: data file format

    """

    def __init__(self, path: str,
                 input_data_filename: str,
                 output_data_filename: str,
                 file_format: DatafileFormat):
        """Creates a new SetFileStructure
        """
        self.__path: str = path
        self.__input_data_filename: str = input_data_filename
        self.__output_data_filename: str = output_data_filename
        self.__file_format: DatafileFormat = file_format

    @property
    def path(self):
        return self.__path

    @property
    def input_data_filename(self):
        return self.__input_data_filename

    @property
    def output_data_filename(self):
        return self.__output_data_filename

    @property
    def file_format(self):
        return self.__file_format

    @property
    def input_data_filepath(self):
        return os.path.join(self.path, self.input_data_filename)

    @property
    def output_data_filepath(self):
        return os.path.join(self.path, self.output_data_filename)

    @classmethod
    def get_canonical(cls, path: str,
                      set_name: str,
                      file_format: DatafileFormat):
        """Creates a new Set File Structure according to canonical definitions

        Args:
            path (str): system path of data files directory
            set_name (str): set name
            file_format (DatafileFormat): data file format

        """
        input_data_filename = str_to_filename(set_name) + '-input'
        output_data_filename = str_to_filename(set_name) + '-output'

        if file_format == DatafileFormat.TXT:
            input_data_filename += '.txt'
            output_data_filename += '.txt'

        elif file_format == DatafileFormat.NPY:
            input_data_filename += '.npy'
            output_data_filename += '.npy'

        return SetFileStructure(path=path,
                                input_data_filename=input_data_filename,
                                output_data_filename=output_data_filename,
                                file_format=file_format)


class CorpusFileStructure:
    def __init__(self,
                 training_file_structure: SetFileStructure,
                 test_file_structure: SetFileStructure,
                 validation_file_structure: SetFileStructure = None,
                 file_format: DatafileFormat = DatafileFormat.NPY):
        self.__training_file_structure = training_file_structure
        self.__test_file_structure = test_file_structure
        self.__validation_file_structure = validation_file_structure
        self.__file_format = file_format

    @property
    def training_file_structure(self):
        return self.__training_file_structure

    @property
    def test_file_structure(self):
        return self.__test_file_structure

    @property
    def validation_file_structure(self):
        return self.__validation_file_structure

    @property
    def file_format(self):
        return self.__file_format

    @classmethod
    def get_canonical(cls,
                      corpus_name: str,
                      base_path: str,
                      file_format: DatafileFormat = DatafileFormat.NPY):
        """Creates a new CorpusFileStructure according to the conventions

        Args:
            corpus_name (str): corpus name
            base_path (str): system path of the base directory (subdirectories will be created here)
            file_format (DatafileFormat): data files format


        """
        training_set_name = corpus_name + ' - train'
        training_path = os.path.join(base_path, 'train')
        training_file_structure = \
            SetFileStructure.get_canonical(path=training_path, set_name=training_set_name, file_format=file_format)

        test_set_name = corpus_name + ' - test'
        test_path = os.path.join(base_path, 'test')
        test_file_structrure = \
            SetFileStructure.get_canonical(path=test_path, set_name=test_set_name, file_format=file_format)

        validation_set_name = corpus_name + ' - validation'
        validation_path = os.path.join(base_path, 'validation')
        validation_file_structure = \
            SetFileStructure.get_canonical(path=validation_path, set_name=validation_set_name, file_format=file_format)

        return CorpusFileStructure(training_file_structure=training_file_structure,
                                   test_file_structure=test_file_structrure,
                                   validation_file_structure=validation_file_structure,
                                   file_format=file_format)

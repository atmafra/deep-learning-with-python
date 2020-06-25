""" One-hot encoding of the sequences
"""
import json
import os

import numpy as np
from keras.utils import to_categorical


class OneHotEncoder:

    def __init__(self, sequence: list = None):
        self.__category_list = []
        self.__frequency_list = []
        self.__category_map = {}

        if sequence is not None:
            self.encode(sequence)

    @property
    def category_list(self):
        return self.__category_list

    @property
    def frequency_list(self):
        return self.__frequency_list

    @property
    def category_map(self):
        return self.__category_map

    def encode(self, sequence: list, rebuild: bool = True):
        """

        :param sequence:
        :param rebuild:
        :return:
        """
        # TODO: better dealing with out of category encoding
        assert sequence is not None, 'No sequences to one-hot encode'

        if rebuild:
            unique_result = np.unique(ar=sequence,
                                      return_index=False,
                                      return_inverse=True,
                                      return_counts=True)

            self.__category_list = unique_result[0].tolist()
            self.__category_map = {category: i for i, category in enumerate(self.__category_list)}
            self.__frequency_list = unique_result[2].tolist()
            encoded_sequence = unique_result[1]

        else:
            encoded_sequence = [self.__category_map[i] for i in sequence if i in self.__category_map]

        return to_categorical(encoded_sequence)

    def decode(self, encoded_sequence: list):
        """

        :param encoded_sequence:
        :return:
        """
        code_sequence = np.argmax(encoded_sequence, axis=1)
        sequence = [self.category_list[code] for code in code_sequence]
        return sequence

    def get_config(self):
        """ Gets the encoder configuration as Python dictionary.

        :return: a Python dictionary containing the enconder configuration (state)
        """
        return {
            'category_list': json.dumps(self.category_list),
            'category_map': json.dumps(self.category_map),
            'frequency_list': json.dumps(self.frequency_list)
        }

    def to_json(self):
        """ Returns a JSON string containing the encoder configuration.
        To load an encoder from a JSON string, use 'encoder_from_json(json_string)'

        :return: a JSON string containing the encoder configuration.
        """
        config = self.get_config()
        encoder_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }

        return json.dumps(encoder_config)

    @classmethod
    def from_json(cls, json_string: str):
        """ Parses a JSON encoder configuration file and returns an encoder instance.

        :param json_string: JSON string containing the encoder configuration
        :return: a tokenizer instance
        """
        encoder_config = json.loads(json_string)
        config = encoder_config.get('config')
        encoder = cls()
        encoder.__category_list = json.loads(config.pop('category_list'))
        encoder.__category_map = json.loads(config.pop('category_map'))
        encoder.__frequency_list = json.loads(config.pop('frequency_list'))

        return encoder

    def save_json_file(self, path: str, filename: str, verbose: bool = True):
        """ Saves the encoder configuration as a JSON file

        :param path: system path of the JSON configuration file
        :param filename: encoder configuration JSON file
        :param verbose: output progress messages in terminal
        """
        if not filename:
            raise RuntimeError('Cannot save encoder: no filename passed')

        filepath = os.path.join(path, filename)
        if verbose:
            print('Saving encoder to JSON file "{}"'.format(filepath))

        with open(filepath, 'w') as file:
            json.dump(self.to_json(), file, indent=4)
        file.close()

    @classmethod
    def from_json_file(cls, path: str, filename: str, verbose: bool = True):
        """ Returns a new instance created from the encoder configurations saved in a JSON file

        :param path: system path of the JSON configuration file
        :param filename: JSON file containing the encoder configuration
        :param verbose: output progress messages in terminal
        :return: new encoder instance, with its status loaded from configurations
        """
        if not filename:
            raise RuntimeError('Cannot load encoder: no filename passed')

        filepath = os.path.join(path, filename)
        if verbose:
            print('Loading encoder from JSON file "{}"'.format(filepath))

        with open(filepath, 'r') as file:
            encoder_configuration = json.load(file)
        file.close()

        return cls.from_json(encoder_configuration)

""" One-hot encoding of the sequences
"""
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

            self.__category_list = unique_result[0]
            self.__category_map = {category: i for i, category in enumerate(self.__category_list)}
            self.__frequency_list = unique_result[2]
            encoded_sequence = unique_result[1]

        else:
            encoded_sequence = [self.__category_map[i] for i in sequence if i in self.__category_map]

        return to_categorical(encoded_sequence)

    def decode(self, encoded_sequence: list):
        """

        :param encoded_sequence:
        :return:
        """
        sequence = []
        for x in encoded_sequence:
            code = np.argmax(x, axis=1)
            category = self.category_list[code]
            sequence.append(category)
        return sequence

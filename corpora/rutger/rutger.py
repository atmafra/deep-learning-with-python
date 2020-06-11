import csv
import os
from random import random

import numpy as np
from keras_preprocessing.sequence import pad_sequences

from core.corpus import Corpus
from core.datasets import Dataset
from core.file_structures import CorpusFileStructure
from utils.dataset_utils import one_hot_encode, tokenize_text, one_hot_encode_categories

default_rutger_path = 'corpora/rutger/data'
default_filename = 'rutger-2020-06-03.csv'
default_vocabulary_size = 10000
default_input_length = 100
default_sampling_rate = 1.
default_training_set_share = 0.6
default_validation_set_share = 0.2
default_test_set_share = 0.2


def __rutger_csv_generator(path: str,
                           filename: str,
                           block_size: int = 1000,
                           sampling_rate: float = 1.):
    """ Loads the Rutger dataset from system files

    :param path: dataset system path
    :param filename: dataset filename
    :param sampling_rate: line sampling rate
    :return: numpy array with the temperature time series
    """
    filepath = os.path.join(path, filename)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        block = {}
        for column in reader.fieldnames:
            block[column] = []
        line_count = 0
        for line in reader:
            if sampling_rate < 1.:
                if random() > sampling_rate:
                    continue
            for column in reader.fieldnames:
                block[column].append(line[column])
            line_count += 1
            if line_count == block_size:
                yield block
                line_count = 0
                for column in reader.fieldnames:
                    block[column].clear()
        yield block


def __load_raw_data(rutger_path: str,
                    filename: str,
                    sampling_rate: float = 1.):
    """

    :param rutger_path:
    :param filename:
    :param sampling_rate:
    :return:
    """
    raw_data = {}
    for block in __rutger_csv_generator(path=rutger_path,
                                        filename=filename,
                                        sampling_rate=sampling_rate):
        for column in block.keys():
            if column not in raw_data:
                raw_data[column] = []
            raw_data[column] += block[column]
    return raw_data


def __preprocess(input_data: np.array,
                 vocabulary_size: int,
                 input_length: int):
    """ Tokenizes the words and truncates/pads the sequences to the input length

    :param input_data: input data
    :param vocabulary_size: number of most frequent words to include in the index
    :param input_length: input length (for padding or truncating)
    :return: preprocessed numpy array
    """
    input_encoded = tokenize_text(texts=list(input_data), fit=True, max_words=vocabulary_size)
    input_padded = pad_sequences(sequences=input_encoded,
                                 maxlen=input_length,
                                 padding='pre',
                                 truncating='pre')
    return input_padded


def build_corpus(rutger_path: str = default_rutger_path,
                 filename: str = default_filename,
                 vocabulary_size: int = default_vocabulary_size,
                 input_length: int = default_input_length,
                 sampling_rate: float = default_sampling_rate,
                 training_set_share: float = default_training_set_share,
                 validation_set_share: float = default_validation_set_share,
                 test_set_share: float = default_test_set_share,
                 verbose: bool = True):
    """ Builds the Rutger intent detection corpus from the source data files

    :param rutger_path: system path for the Rutger corpus data
    :param filename: Rutger data filename
    :param vocabulary_size: maximum number of words (tokens) in the dictionary
    :param input_length: maximum number of words (tokens) per phrase
    :param sampling_rate: fraction of samples to take from the entire dataset
    :param training_set_share: percentage of samples to be used as the training set
    :param validation_set_share: percentage of samples to be used as the validation set
    :param test_set_share: percentage of samples to be used as the test set
    :param verbose: display log messages in the terminal as the corpus is built
    :return: the Rutger intent detection Corpus object
    """
    if verbose:
        print('Loading raw data from file \'{}\' in system path \'{}\' (sampling rate: {:.0%})'.
              format(filename, rutger_path, sampling_rate))

    raw_data = __load_raw_data(rutger_path=rutger_path,
                               filename=filename,
                               sampling_rate=sampling_rate)

    input_data = np.array(raw_data['message'])
    if verbose:
        print('Preprocessing {} loaded messages (input length: {}, vocabulary size: {})'.
              format(len(input_data), input_length, vocabulary_size))

    input_set = __preprocess(input_data=input_data,
                             vocabulary_size=vocabulary_size,
                             input_length=input_length)

    # if verbose:
    #     print('One-hot encoding to vocabulary size of {}'.format(vocabulary_size))
    # input_set = one_hot_encode(sequence=input_preprocessed, categories=vocabulary_size)

    output_data = np.array(raw_data['intent'])
    if verbose:
        print('Processing {} output intentions'.format(len(output_data)))

    output_set = None
    if output_data is not None:
        if verbose:
            print('One-hot encoding output categories')
        output_set = one_hot_encode_categories(sequence=output_data)

    sample_weights_set = np.array(raw_data['frequency']).astype(np.int)
    if verbose:
        print('Processing {} sample weight values'.format(len(sample_weights_set)))

    rutger_set = Dataset(input_data=input_set,
                         output_data=output_set,
                         sample_weights=sample_weights_set,
                         name='Rutger')

    split_sampling_rate = 1.
    if verbose:
        print('Splitting the dataset (sampling rate: {:.0%}, training: {:.0%}, validation: {:.0%}, test: {:.0%})'.
              format(split_sampling_rate, training_set_share, validation_set_share, test_set_share))

    training_set, validation_set, test_set = rutger_set.three_way_split(sampling_rate=split_sampling_rate,
                                                                        partition_1_share=training_set_share,
                                                                        partition_2_share=validation_set_share,
                                                                        partition_3_share=test_set_share)

    if verbose:
        print('Training dataset created')
    del rutger_set

    corpus = Corpus(training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    name='Rutger')
    if verbose:
        print('Corpus created')

    return corpus

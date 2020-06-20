import csv
import os
from random import random

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from core.corpus import Corpus
from core.datasets import Dataset
from text.one_hot_encoder import OneHotEncoder
from text.preprocessing import clean_text, tokenize_text, build_tokenizer
from utils.dataset_utils import one_hot_encode_categories

default_rutger_path = 'corpora/rutger/data'
default_rutger_golden_path = default_rutger_path + '/golden'
default_filename = 'rutger-2020-06-03.csv'
default_golden_filename = 'Version33.csv'
default_clean = True
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


def __clean(input_data: np.array):
    for i, text in enumerate(input_data):
        input_data[i] = clean_text(text)


def __fit_tokenizer(input_data: list,
                    vocabulary_size: int,
                    verbose: bool = True):
    """ Builds the global tokenizer

    :param input_data: list of input phrases
    :param vocabulary_size: maximum vocabulary size
    :param verbose: display log messages during execution
    :return: tokenizer object, fit to input data
    """
    global global_tokenizer
    global_tokenizer = build_tokenizer(phrase_list=input_data,
                                       vocabulary_size=vocabulary_size,
                                       verbose=verbose)
    return global_tokenizer


def __encode(input_data: np.array,
             input_length: int,
             tokenizer: Tokenizer,
             verbose: bool = True):
    """ Tokenizes the messages and truncates/pads the sequences to the input length

    :param input_data: input data
    :param input_length: input length (for padding or truncating)
    :param tokenizer: pre-calculated tokenizer (encoder)
    :param verbose: display log messages on terminal during execution
    :returns: preprocessed numpy array and the generated tokenizer
    """
    if verbose:
        print('Encoding input data')
    input_encoded = tokenize_text(texts=list(input_data), tokenizer=tokenizer)

    if verbose:
        print('Padding data sequences to input length {}'.format(input_length))
    input_padded = pad_sequences(sequences=input_encoded,
                                 maxlen=input_length,
                                 padding='pre',
                                 truncating='pre')

    return input_padded


def get_name(clean: bool = default_clean,
             vocabulary_size: int = default_vocabulary_size,
             input_length: int = default_input_length,
             sampling_rate: float = default_sampling_rate,
             training_set_share: float = default_training_set_share,
             validation_set_share: float = default_validation_set_share,
             test_set_share: float = default_test_set_share):
    """ Gets a standard dataset name based on the build parameters

    :param clean: clean phrases before tokenization
    :param vocabulary_size: maximum number of words (tokens) in the dictionary
    :param input_length: maximum number of words (tokens) per phrase
    :param sampling_rate: fraction of samples to take from the entire dataset
    :param sampling_rate: fraction of samples to take from the entire dataset
    :param training_set_share: percentage of samples to be used as the training set
    :param validation_set_share: percentage of samples to be used as the validation set
    :param test_set_share: percentage of samples to be used as the test set
    :return: dataset name
    """
    return 'Rutger CLN={} VCSZ={} INLN={} SMPR={:.0%} TVT={:.0%}/{:.0%}/{:.0%}'. \
        format(clean, vocabulary_size, input_length, sampling_rate,
               training_set_share, validation_set_share, test_set_share)


def build_corpus(rutger_path: str = default_rutger_path,
                 filename: str = default_filename,
                 clean: bool = default_clean,
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
    :param clean: clean phrases before tokenization
    :param vocabulary_size: maximum number of words (tokens) in the dictionary
    :param input_length: maximum number of words (tokens) per phrase
    :param sampling_rate: fraction of samples to take from the entire dataset
    :param training_set_share: percentage of samples to be used as the training set
    :param validation_set_share: percentage of samples to be used as the validation set
    :param test_set_share: percentage of samples to be used as the test set
    :param verbose: display log messages in the terminal as the corpus is built
    :returns: the Rutger intent detection Corpus object and the Tokenizer object
    """
    if verbose:
        print('Loading raw data from file \'{}\' in system path \'{}\' (sampling rate: {:.0%})'.
              format(filename, rutger_path, sampling_rate))

    dataset_name = get_name(clean=clean,
                            vocabulary_size=vocabulary_size,
                            input_length=input_length,
                            sampling_rate=sampling_rate,
                            training_set_share=training_set_share,
                            validation_set_share=validation_set_share,
                            test_set_share=test_set_share)
    if verbose:
        print('Dataset name is "{}"'.format(dataset_name))

    if verbose:
        filepath = os.path.join(rutger_path, filename)
        print('Loading raw data from file "{}"'.format(filepath))
    raw_data = __load_raw_data(rutger_path=rutger_path,
                               filename=filename,
                               sampling_rate=sampling_rate)

    messages_raw = np.array(raw_data['message'])
    if verbose:
        print('{} messages loaded'.format(len(messages_raw)))

    if clean:
        if verbose:
            print('Cleaning loaded messages')
        __clean(messages_raw)

    intents_raw = np.array(raw_data['intent'])
    if verbose:
        print('Processing {} output intentions'.format(len(intents_raw)))

    if verbose:
        print('One-hot encoding output categories')
    output_encoder = OneHotEncoder()
    output_encoded = output_encoder.encode(sequence=list(intents_raw), rebuild=True)

    sample_weights_set = np.array(raw_data['frequency']).astype(np.int)
    if verbose:
        print('Processing {} sample weight values'.format(len(sample_weights_set)))

    raw_set = Dataset(input_data=messages_raw,
                      output_data=output_encoded,
                      sample_weights=sample_weights_set,
                      name='Rutger RAW')

    if verbose:
        print('Shuffling raw dataset')
    raw_set.shuffle()

    split_sampling_rate = 1.
    if verbose:
        print('Splitting the dataset (sampling rate: {:.0%}, training: {:.0%}, validation: {:.0%}, test: {:.0%})'.
              format(split_sampling_rate, training_set_share, validation_set_share, test_set_share))

    training_clean, validation_clean, test_clean = raw_set.three_way_split(sampling_rate=split_sampling_rate,
                                                                           partition_1_share=training_set_share,
                                                                           partition_2_share=validation_set_share,
                                                                           partition_3_share=test_set_share)

    tokenizer = __fit_tokenizer(input_data=training_clean.input_data,
                                vocabulary_size=vocabulary_size,
                                verbose=verbose)

    training_input_encoded = __encode(input_data=training_clean.input_data,
                                      input_length=input_length,
                                      tokenizer=tokenizer,
                                      verbose=verbose)

    validation_input_encoded = __encode(input_data=validation_clean.input_data,
                                        input_length=input_length,
                                        tokenizer=tokenizer,
                                        verbose=verbose)

    test_input_encoded = __encode(input_data=test_clean.input_data,
                                  input_length=input_length,
                                  tokenizer=tokenizer,
                                  verbose=verbose)

    training_set = Dataset(input_data=training_input_encoded,
                           output_data=training_clean.output_data,
                           sample_weights=training_clean.sample_weights,
                           name=dataset_name + ' - training')

    validation_set = Dataset(input_data=validation_input_encoded,
                             output_data=validation_clean.output_data,
                             sample_weights=validation_clean.sample_weights,
                             name=dataset_name + ' - validation')

    test_set = Dataset(input_data=test_input_encoded,
                       output_data=test_clean.output_data,
                       sample_weights=test_clean.sample_weights,
                       name=dataset_name + ' - test')

    corpus = Corpus(training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    name=dataset_name)
    if verbose:
        print('Corpus "{}" successfully created'.format(dataset_name))

    return corpus, tokenizer, output_encoder


def build_golden_dataset(rutger_golden_path: str = default_rutger_golden_path,
                         rutger_golden_filename: str = default_golden_filename,
                         clean: bool = default_clean,
                         vocabulary_size: int = default_vocabulary_size,
                         input_length: int = default_input_length,
                         sampling_rate: float = default_sampling_rate,
                         tokenizer: Tokenizer = None,
                         output_encoder: OneHotEncoder = None,
                         verbose: bool = True):
    """ Builds the Golden Dataset from the source data files

    :param rutger_golden_path:
    :param rutger_golden_filename:
    :param clean:
    :param vocabulary_size:
    :param input_length:
    :param sampling_rate:
    :param tokenizer:
    :param output_encoder:
    :param verbose:
    :return: Golden Dataset
    """
    if verbose:
        filepath = os.path.join(rutger_golden_path, rutger_golden_filename)
        print('Loading raw data from file \'{}\' (sampling rate: {:.0%})'.
              format(filepath, sampling_rate))

    dataset_name = get_name(clean=clean,
                            vocabulary_size=vocabulary_size,
                            input_length=input_length,
                            sampling_rate=sampling_rate,
                            training_set_share=0.,
                            validation_set_share=0.,
                            test_set_share=1.)
    if verbose:
        print('Dataset name is "{}"'.format(dataset_name))

    raw_data = __load_raw_data(rutger_path=rutger_golden_path,
                               filename=rutger_golden_filename,
                               sampling_rate=sampling_rate)

    input_data = np.array(raw_data['resolved_query'])
    if verbose:
        print('Preprocessing {} loaded messages (input length: {}, vocabulary size: {})'.
              format(len(input_data), input_length, vocabulary_size))

    if clean:
        print('Cleaning {} messages'.format(len(input_data)))
        __clean(input_data)

    input_encoded = __encode(input_data=input_data,
                             input_length=input_length,
                             tokenizer=tokenizer,
                             verbose=verbose)

    output_data = np.array(raw_data['intent_name'])
    if verbose:
        print('Processing {} output intentions'.format(len(output_data)))

    if output_data is None:
        raise RuntimeError('Cannot build Golden dataset with no output data')

    if verbose:
        print('One-hot encoding output categories')

    if output_encoder is not None:
        output_encoded = output_encoder.encode(sequence=list(output_data), rebuild=False)
    else:
        output_encoded = one_hot_encode_categories(sequence=output_data)

    golden_dataset = Dataset(input_data=input_encoded,
                             output_data=output_encoded,
                             name='Rutger Golden Dataset')

    return golden_dataset

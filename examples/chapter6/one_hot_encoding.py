import string

import numpy as np
from keras_preprocessing.text import Tokenizer

default_samples = ['The cat sat on the mat.', 'The dog ate my homework.']


def word_level_one_hot_encoding(samples: list) -> np.array:
    token_index = {}

    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_length = 10

    results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    return results


def word_level_one_hot_encoding_keras(samples: list) -> np.array:
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)

    sequences = tokenizer.texts_to_sequences(texts=samples)
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    return one_hot_results


def word_level_one_hot_encoding_hashing(samples: list) -> np.array:
    dimensionality = 1000
    max_length = 10

    results = np.zeros((len(samples), max_length, dimensionality))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1.

    return results


def character_level_one_hot_encoding(samples: list):
    characters = string.printable
    token_index = dict(zip(range(1, len(characters) + 1), characters))
    # invert token index
    token_index = {v: k for k, v in token_index.items()}

    max_length = 50
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1.

    return results


def run():
    word_level_results = word_level_one_hot_encoding(samples=default_samples)
    print('word level results =', word_level_results)

    word_level_results_keras = word_level_one_hot_encoding_keras(samples=default_samples)
    print('word level results (Keras) =', word_level_results_keras)

    word_level_results_hashing = word_level_one_hot_encoding_hashing(samples=default_samples)
    print('word level results (hashing trick) =', word_level_results_hashing)

    character_level_results = character_level_one_hot_encoding(samples=default_samples)
    print('character level results = ', character_level_results)

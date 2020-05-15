import numpy as np


def separate_corpus(corpus: tuple):
    """ Splits a corpus into train, test, and validation (optional) data sets
    """
    assert corpus is not None, 'No corpus passed to split'

    if len(corpus) < 2:
        raise ValueError('Corpus must have at least two sets: the training and test sets')

    training_inputs = None
    training_outputs = None
    test_inputs = None
    test_outputs = None
    validation_inputs = None
    validation_outputs = None

    # training set (mandatory)
    if len(corpus[0]) < 1:
        raise RuntimeError('Training set must have at least input data')

    training_inputs = corpus[0][0]
    if len(corpus[0]) > 1:
        training_outputs = corpus[0][1]

    # test set (mandatory)
    if len(corpus) > 1:

        if len(corpus[1]) < 1:
            raise RuntimeError('Test set must have at least input data')

        test_inputs = corpus[1][0]
        if len(corpus[1]) > 1:
            test_outputs = corpus[1][1]

    # validation set (optional)
    if len(corpus) > 2:

        if len(corpus[2]) < 1:
            raise RuntimeError('Validation set must have at least input data')

        test_inputs = corpus[2][0]
        if len(corpus[2] > 1):
            test_outputs = corpus[2][1]

    training_set = (training_inputs, training_outputs)
    test_set = (test_inputs, test_outputs)
    validation_set = (validation_inputs, validation_outputs)

    return training_set, test_set, validation_set


def split_dataset(dataset: np.array,
                  split_size: int,
                  split_start: int = 0,
                  shuffle: bool = False):
    """ Splits a dataset into two subsets
    :param dataset: dataset to be split
    :param split_size: number of elements to be split
    :param split_start: split point
    :param shuffle: shuffle element order before split
    :return: split datasets: split, remain
    """
    assert dataset is not None, 'Empty train set trying to split validation set'

    dataset_len = len(dataset)

    if split_size > dataset_len:
        raise RuntimeError('validation set size bigger ({}) than train set size ({})'
                           .format(split_size, dataset_len))

    if shuffle:
        np.random.shuffle(dataset)

    split_stop = split_start + split_size

    if split_stop <= dataset_len:
        # split before the end of the array
        split_left = dataset[0:split_start]
        split_right = dataset[split_stop:]
        split = dataset[split_start:split_stop]
        remain = np.concatenate([split_left, split_right])
    else:
        # split after the end of the array
        split_stop = split_stop - dataset_len
        split_left = dataset[split_start:]
        split_right = dataset[0:split_stop]
        split = np.concatenate([split_left, split_right])
        remain = dataset[split_stop:split_start]

    return split, remain


def merge_datasets(dataset1: np.array,
                   dataset2: np.array):
    """ Returns a new dataset that merges the input and output data of the two datasets
    :param dataset1: first dataset
    :param dataset2: second dataset
    :return: merged dataset
    """
    empty1 = dataset1 is None or len(dataset1) == 0
    empty2 = dataset2 is None or len(dataset2) == 0

    if empty1 and empty2:
        raise ValueError('No valid datasets passed for merge')

    if empty1 and not empty2:
        return dataset2.copy()

    if empty2 and not empty1:
        return dataset1.copy()

    if len(dataset1) != len(dataset2):
        raise RuntimeError('Cannot merge arrays: input dimensions are different')

    return np.append([dataset1, dataset2])


def count_unique_values(sequence: np.array):
    """ Counts the number of unique values in the sequence
    """
    assert sequence is not None, 'No sequence passed'
    return len(np.unique(sequence))


def one_hot_encode(sequence: np.array, categories: int):
    """ One-hot encoding of the sequences
    """
    assert sequence is not None, 'No sequences to one-hot encode'

    results = np.zeros((len(sequence), categories))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1.

    return results


def normalize(training_data: np.array, test_data: np.array):
    """ Normalizes the train and test datasets
    """
    mean = training_data.mean(axis=0)
    stddev = training_data.std(axis=0)

    # train dataset
    training_data -= mean
    training_data /= stddev

    # test dataset
    test_data -= mean
    test_data /= stddev

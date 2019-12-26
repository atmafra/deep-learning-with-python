import numpy as np


def separate_corpus(corpus: tuple):
    """ Splits a corpus into train and test sets
    """
    assert corpus is not None, 'No corpus passed to split'

    if len(corpus) != 2:
        raise RuntimeError('Corpus must have 2 sets: train and test')

    # train set
    training_set = corpus[0]
    if len(training_set) != 2:
        raise RuntimeError('Training set must have 2 series: data and labels')
    training_data = training_set[0]
    training_labels = training_set[1]

    # test set
    test_set = corpus[1]
    if len(test_set) != 2:
        raise RuntimeError('Test set must have 2 series: data and labels')
    test_data = test_set[0]
    test_labels = test_set[1]

    return (training_data, training_labels), (test_data, test_labels)


def split_dataset(dataset: np.array,
                  split_size: int,
                  split_start: int = 0,
                  shuffle: bool = False):
    """ Splits a dataset into two subsets
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

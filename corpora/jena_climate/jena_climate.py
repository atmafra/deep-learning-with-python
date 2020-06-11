import os
import matplotlib.pyplot as plt

import numpy as np

from core.corpus import CorpusGenerator

default_jena_path = 'corpora/jena_climate/data'
default_filename = 'jena_climate_2009_2016.csv'


def __load_raw_data(path: str,
                    filename: str):
    """ Loads the Jena Climate dataset from system files

    :param path: dataset system path
    :param filename: dataset filename
    :return: numpy array with the temperature time series
    """
    filepath = os.path.join(path, filename)
    f = open(filepath)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    return float_data


def __preprocess(data, sample_size: int = 200000):
    """ Samples the dataset and normalizes the sample

    :param data: dataset
    :param sample_size: sample size
    :return: normalized sample of the dataset
    """
    mean = data[:sample_size].mean(axis=0)
    data -= mean
    stdev = data[:sample_size].std(axis=0)
    data /= stdev
    return data


def __plot(data, days: int = None):
    """ Plots a sample of the last days of the dataset

    :param data: dataset array
    :param days: number of days
    """
    if days is not None:
        points = days * 144
    else:
        points = len(data)

    temp = data[-points:, 1]

    plt.plot(range(points), temp)
    plt.grid()
    plt.show()


def build_corpus(jena_path: str = default_jena_path,
                 jena_filename: str = default_filename):
    """ Loads the data series from the file system and then builds a Corpus object

    :param jena_path: system path of the Jena Climate dataset
    :param jena_filename: dataset filename
    :return:
    """
    float_data = __load_raw_data(path=jena_path,
                                 filename=jena_filename)

    data = __preprocess(data=float_data,
                        sample_size=200000)

    # __plot(data)
    return data


def generator(data,
              lookback: int,
              delay: int,
              min_index: int,
              max_index: int,
              shuffle: bool = False,
              batch_size: int = 128,
              step: int = 6):
    """ Generator that returns the data in batches

    :param data: the original array of floating-point data, already normalized
    :param lookback: how many timestapes back the input data should go
    :param delay: how many timesteps in the future the target should be
    :param min_index: lower limit in the data array that delimit which timesteps to draw from
    :param max_index: upper limit in the data array that delimit which timesteps to draw from
    :param shuffle: thether to shuffle the samples or draw them in chronological order
    :param batch_size: the number of samples per batch
    :param step: the period, in timestemps, at which to sample data
    :return:
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while True:
        if shuffle:
            rows = np.random.randint(low=min_index + lookback, high=max_index, size=batch_size)
        else:
            if i + batch_size > max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


def get_generators(float_data,
                   lookback: int = 1440,
                   step: int = 6,
                   delay: int = 144,
                   batch_size: int = 128,
                   train_size: int = 200000,
                   validation_size: int = 100000,
                   test_size: int = None):
    """

    :param float_data:
    :param lookback:
    :param step:
    :param delay:
    :param batch_size:
    :return:
    """
    train_generator = generator(data=float_data,
                                lookback=lookback,
                                delay=delay,
                                min_index=0,
                                max_index=train_size,
                                shuffle=True,
                                step=step,
                                batch_size=batch_size)

    validation_generator = generator(data=float_data,
                                     lookback=lookback,
                                     delay=delay,
                                     min_index=train_size + 1,
                                     max_index=train_size + test_size,
                                     shuffle=True,
                                     step=step,
                                     batch_size=batch_size)

    test_generator = generator(data=float_data,
                               lookback=lookback,
                               delay=delay,
                               min_index=train_size + test_size + 1,
                               shuffle=True,
                               step=step,
                               batch_size=batch_size)


    Dataset
    corpus_generator = CorpusGenerator(training_set_files=train_generator,
                                       validation_set_files=validation_generator,
                                       test_set_files=test_generator)

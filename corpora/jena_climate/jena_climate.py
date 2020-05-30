import os
import matplotlib.pyplot as plt

import numpy as np

default_jena_path = 'corpora/jena_climate/data'
default_filename = 'jena_climate_2009_2016.csv'


def __load_raw_data(path: str,
                    filename: str):
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
    mean = data[:sample_size].mean(axis=0)
    data -= mean
    stdev = data[:sample_size].std(axis=0)
    data /= stdev
    return data


def __plot(data, days: int = -1):
    if days > 0:
        points = days * 144
    else:
        points = len(data)

    temp = data[:points, 1]

    plt.plot(range(points), temp)
    plt.show()


def build_corpus(jena_path: str = default_jena_path,
                 jena_filename: str = default_filename):
    """

    :param jena_path:
    :param jena_filename:
    :return:
    """
    float_data = __load_raw_data(path=jena_path,
                                 filename=jena_filename)

    data = __preprocess(data=float_data,
                        sample_size=200000)
    __plot(data)  # , days=10)


def generator(data,
              lookback: int,
              delay: int,
              min_index: int,
              max_index: int,
              shuffle: bool = False,
              batch_size: int = 128,
              step: int = 6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
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
                   batch_size: int = 128):
    train_generator = generator(data=float_data,
                                lookback=lookback,
                                delay=delay,
                                min_index=0,
                                max_index=20000)

import numpy as np
from keras.datasets import imdb

from core import hyperparameters as hpp
from core import network as net
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils import history_utils as hutl

vector_dimension = 0

if __name__ == '__main__':
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def load_corpus(num_words: int = 10000, verbose: bool = True) -> Corpus:
    """"Loads the IMDB dataset into a corpus object

    Args:
        num_words (int): word limit in the reverse index
        verbose (bool): outputs progress messages

    """
    if verbose:
        print("Loading IMDB dataset...")

    corpus = imdb.load_data(num_words=num_words)
    (train_samples, train_labels), (test_samples, test_labels) = dsu.separate_corpus(corpus)

    # one-hot encode the phrases
    global vector_dimension
    vector_dimension = num_words
    training_inputs = dsu.one_hot_encode(train_samples, vector_dimension)
    test_inputs = dsu.one_hot_encode(test_samples, vector_dimension)

    # vectorize the labels
    training_outputs = np.asarray(train_labels).astype('float32')
    test_outputs = np.asarray(test_labels).astype('float32')

    if verbose:
        print("{} train reviews loaded".format(len(train_samples)))
        print("{} test reviews loaded".format(len(test_samples)))

    # create the corpus
    return Corpus.from_datasets(training_inputs, training_outputs, test_inputs, test_outputs)


def hyperparameters(input_size: int = 10000,
                    activation: str = 'relu',
                    output_activation: str = 'sigmoid',
                    layer_units=None,
                    loss: str = 'binary_crossentropy'):
    """Defines the IMDB neural model hyper parameters

    Args:
        input_size (int): input layer size (in units)
        activation (str): hidden layers activation function string
        output_activation (str): output activation function string
        layer_units (list): list of number of units per layer
        loss (str): loss function string

    """
    if layer_units is None:
        layer_units = [16, 16, 1]

    total_layers = len(layer_units)
    hidden_layers = total_layers - 2

    # layer hyper parameters list
    input_layer_hparm = hpp.LayerHyperparameters(layer_units[0], activation, input_size)
    hidden_layers_hparm = []
    for units in layer_units[1:1 + hidden_layers]:
        hidden_layers_hparm.append(hpp.LayerHyperparameters(units=units, activation=activation))
    output_layer_hparm = hpp.LayerHyperparameters(layer_units[-1], output_activation)
    layer_hparm_list = [input_layer_hparm] + hidden_layers_hparm + [output_layer_hparm]

    # network hyper parameters
    hparm = hpp.NetworkHyperparameters(input_size=input_size, output_size=1,
                                       output_type=hpp.OutputType.BOOLEAN,
                                       layer_hyperparameters_list=layer_hparm_list,
                                       optimizer='rmsprop',
                                       learning_rate=0.001,
                                       loss=loss,
                                       metrics=['accuracy']
                                       )
    return hparm


def run():
    num_words = 10000
    corpus = load_corpus(num_words=num_words)

    imdb_hyperparameters_1 = hyperparameters(input_size=num_words,
                                             activation='relu',
                                             layer_units=[16, 16, 1],
                                             loss='binary_crossentropy')

    imdb_nnet_1 = net.create_network(imdb_hyperparameters_1)

    epochs = 20
    batch_size = 512
    shuffle = True
    validation_set_size = 10000

    validation_set, training_set_remaining = corpus.get_validation_set(validation_set_size)

    history_1 = net.train_network(network=imdb_nnet_1,
                                  training_set=training_set_remaining,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  validation_set=validation_set)

    hutl.plot_accuracy_dict(history_1.history, title='IMDB 1: Training and Validation Accuracies')
    (test_loss_1, test_accuracy_1) = net.test_network(imdb_nnet_1, corpus.test_set)

    imdb_hyperparameters_2 = hyperparameters(input_size=num_words,
                                             activation='relu',
                                             layer_units=[16, 16, 16, 1],
                                             loss='binary_crossentropy')
    imdb_nnet_2 = net.create_network(imdb_hyperparameters_2)

    history_2 = net.train_network(network=imdb_nnet_2,
                                  training_set=training_set_remaining,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  validation_set=validation_set)

    hutl.plot_accuracy_dict(history_2.history, title='IMDB 2: Training and Validation Accuracies')
    (test_loss_2, test_accuracy_2) = net.test_network(imdb_nnet_2, corpus.test_set)

    # print results
    print("\nNetwork 1")
    print("loss     =", test_loss_1)
    print("accuracy = {:.2%}".format(test_accuracy_1))

    print("\nNetwork 2")
    print("loss     =", test_loss_2)
    print("accuracy = {:.2%}".format(test_accuracy_2))

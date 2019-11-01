import numpy as np
from keras.datasets import imdb

from core import network as net
from core.hyperparameters import LayerPosition, LayerHyperparameters, NetworkHyperparameters, NetworkOutputType
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils import history_utils as hutl

# vector_dimension = 0
num_words = 10000
input_size = num_words
output_size = 1
hidden_activation = 'relu'
output_activation = 'sigmoid'

network_configuration_global = {
    'input_size': input_size,
    'output_size': output_size,
    'output_type': NetworkOutputType.BOOLEAN,
    'optimizer': 'rmsprop',
    'learning_rate': 0.001,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy']}

layer_hyperparameters_1_global = [
    {'units': input_size, 'position': LayerPosition.INPUT, 'activation': 'linear'},
    {'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'units': output_size, 'position': LayerPosition.OUTPUT, 'activation': output_activation}]

layer_hyperparameters_2_global = [
    {'units': input_size, 'position': LayerPosition.INPUT, 'activation': 'linear'},
    {'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'units': output_size, 'position': LayerPosition.OUTPUT, 'activation': output_activation}]

if __name__ == '__main__':
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def load_corpus(words: int = 10000, verbose: bool = True) -> Corpus:
    """"Loads the IMDB dataset into a corpus object

    Args:
        words (int): word limit in the reverse index
        verbose (bool): outputs progress messages

    """
    if verbose:
        print("Loading IMDB dataset...")

    corpus = imdb.load_data(num_words=words)
    (train_samples, train_labels), (test_samples, test_labels) = dsu.separate_corpus(corpus)

    # one-hot encode the phrases
    # global vector_dimension
    vector_dimension = words
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


def hyperparameters(network_hyperparameters: dict, layer_hyperparameters: list):
    """Defines the IMDB neural model hyper parameters

    Args:
        network_hyperparameters (dict): neural network hyperparameters
        layer_hyperparameters (list): list of layer hyperparameters

    """
    # layer hyper parameters list
    layer_hyperparameter_list = []

    for layer in layer_hyperparameters:
        layer_hyperparameter_list.append(
            LayerHyperparameters(units=layer['units'],
                                 position=layer['position'],
                                 activation=layer['activation']))

    # network hyper parameters
    net_hparm = NetworkHyperparameters(input_size=network_hyperparameters['input_size'],
                                       output_size=network_hyperparameters['output_size'],
                                       output_type=network_hyperparameters['output_type'],
                                       optimizer=network_hyperparameters['optimizer'],
                                       learning_rate=network_hyperparameters['learning_rate'],
                                       loss=network_hyperparameters['loss'],
                                       metrics=network_hyperparameters['metrics'],
                                       layer_hyperparameters_list=layer_hyperparameter_list)
    return net_hparm


def run():
    global num_words
    global network_configuration_global
    global layer_hyperparameters_1_global
    global layer_hyperparameters_2_global

    corpus = load_corpus(words=num_words)

    imdb_nnet_1 = net.create_network(
        hyperparameters(network_hyperparameters=network_hyperparameters_global,
                        layer_hyperparameters=layer_hyperparameters_1_global))

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

    imdb_nnet_2 = net.create_network(
        hyperparameters(network_hyperparameters=network_hyperparameters_global,
                        layer_hyperparameters=layer_hyperparameters_2_global))

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

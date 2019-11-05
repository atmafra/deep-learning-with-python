import numpy as np
from keras.datasets import imdb

from core import network as net
from core.hyperparameters import LayerPosition, LayerHyperparameters, NetworkHyperparameters, NetworkOutputType, \
    LayerType
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils import history_utils as hutl

# vector_dimension = 0
num_words = 10000
input_size = num_words
output_size = 1
hidden_activation = 'relu'
output_activation = 'sigmoid'

word_index = {}
reverse_word_index = {}

network_configuration_global = {
    'input_size': input_size,
    'output_size': output_size,
    'output_type': NetworkOutputType.BOOLEAN,
    'optimizer': 'rmsprop',
    'learning_rate': 0.001,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy']}

layers_configuration_small = [
    {'layer_type': LayerType.DENSE, 'units': input_size, 'position': LayerPosition.INPUT, 'activation': 'linear'},
    {'layer_type': LayerType.DENSE, 'units': 4, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': 4, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'position': LayerPosition.OUTPUT,
     'activation': output_activation}]
layers_configuration_medium = [
    {'layer_type': LayerType.DENSE, 'units': input_size, 'position': LayerPosition.INPUT, 'activation': 'linear'},
    {'layer_type': LayerType.DENSE, 'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'position': LayerPosition.OUTPUT,
     'activation': output_activation}]

layers_configuration_large = [
    {'layer_type': LayerType.DENSE, 'units': input_size, 'position': LayerPosition.INPUT, 'activation': 'linear'},
    {'layer_type': LayerType.DENSE, 'units': 512, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': 512, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'position': LayerPosition.OUTPUT,
     'activation': output_activation}]

layers_configuration_medium_dropout = [
    {'layer_type': LayerType.DENSE, 'units': input_size, 'position': LayerPosition.INPUT, 'activation': 'linear'},
    {'layer_type': LayerType.DENSE, 'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'dropout_rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': 16, 'position': LayerPosition.HIDDEN, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'dropout_rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'position': LayerPosition.OUTPUT,
     'activation': output_activation}]

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
            LayerHyperparameters(layer_type=LayerType.DENSE,
                                 units=layer['units'],
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


def run_configuration(network_configuration: dict,
                      layers_configuration: list,
                      corpus: Corpus,
                      validation_set_size: int = 10000,
                      epochs: int = 20,
                      batch_size: int = 512,
                      shuffle: bool = True):
    """Runs a particular configuration of network and layer parameters
    """
    # create the neural network
    neural_network = net.create_network(
        hyperparameters(network_hyperparameters=network_configuration,
                        layer_hyperparameters=layers_configuration))

    # split the training set to get the validation set
    validation_set, training_set_remaining = corpus.get_validation_set(validation_set_size)

    # train the neural network
    history = net.train_network(network=neural_network,
                                training_set=training_set_remaining,
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                validation_set=validation_set)

    # evaluate the neural network
    (test_loss, test_accuracy) = net.test_network(neural_network, corpus.test_set)

    return test_loss, test_accuracy, history


def run():
    global num_words
    global network_configuration_global
    global layers_configuration_small
    global layers_configuration_medium
    global layers_configuration_large

    corpus = load_corpus(words=num_words)

    # training parameters
    validation_set_size = 10000
    epochs = 20
    batch_size = 512
    shuffle = True

    # small configuration
    test_loss_small, test_accuracy_small, history_small = \
        run_configuration(network_configuration=network_configuration_global,
                          layers_configuration=layers_configuration_small,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # medium configuration
    test_loss_medium, test_accuracy_medium, history_medium = \
        run_configuration(network_configuration=network_configuration_global,
                          layers_configuration=layers_configuration_medium,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # large configuration
    test_loss_large, test_accuracy_large, history_large = \
        run_configuration(network_configuration=network_configuration_global,
                          layers_configuration=layers_configuration_large,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # print results
    print("\nSmall Network")
    print("loss     =", test_loss_small)
    print("accuracy = {:.2%}".format(test_accuracy_small))

    print("\nMedium Network")
    print("loss     =", test_loss_medium)
    print("accuracy = {:.2%}".format(test_accuracy_medium))

    print("\nLarge Network")
    print("loss     =", test_loss_large)
    print("accuracy = {:.2%}".format(test_accuracy_large))

    metrics = [history_small, history_medium, history_large]
    legends = ['small network', 'medium network', 'large network']

    hutl.plot_loss_list(history_metrics_list=metrics,
                        labels_list=legends,
                        title='Training Loss',
                        plot_training=True,
                        plot_validation=False)

    hutl.plot_loss_list(history_metrics_list=metrics,
                        labels_list=legends,
                        title='Validation Loss',
                        plot_training=False,
                        plot_validation=True)

    hutl.plot_accuracy_list(history_metrics_list=metrics,
                            labels_list=legends,
                            title='Training Accuracy',
                            plot_training=True,
                            plot_validation=False)

    hutl.plot_accuracy_list(history_metrics_list=metrics,
                            labels_list=legends,
                            title='Validation Accuracy',
                            plot_training=False,
                            plot_validation=True)

    # hutl.plot_accuracy_dict(history_small.history, title='IMDB SMALL: Training and Validation Accuracies')
    # hutl.plot_accuracy_dict(history_medium.history, title='IMDB MEDIUM: Training and Validation Accuracies')
    # hutl.plot_accuracy_dict(history_medium.history, title='IMDB LARGE: Training and Validation Accuracies')

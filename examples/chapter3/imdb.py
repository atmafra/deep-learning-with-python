import numpy as np
from keras import regularizers
from keras.datasets import imdb

from core import network as net
from core.network import NetworkOutputType, LayerType
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

layers_config_1 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

layers_config_2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]


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


def run_configuration(network_configuration: dict,
                      layer_configuration_list: list,
                      corpus: Corpus,
                      validation_set_size: int = 10000,
                      epochs: int = 20,
                      batch_size: int = 512,
                      shuffle: bool = True):
    """Runs a particular configuration of network and layer parameters
    """
    # create the neural network
    neural_network = net.create_network(network_configuration=network_configuration,
                                        layer_configuration_list=layer_configuration_list)

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
    global layers_config_1
    global layers_config_2

    corpus = load_corpus(words=num_words)

    # training parameters
    validation_set_size = 10000
    epochs = 20
    batch_size = 512
    shuffle = True

    # configuration 1
    test_loss_1, test_accuracy_1, history_1 = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=layers_config_1,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # configuration 2
    test_loss_2, test_accuracy_2, history_2 = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=layers_config_2,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    print("\nNetwork Configuration 1")
    print("loss     =", test_loss_1)
    print("accuracy = {:.2%}".format(test_accuracy_1))

    print("\nNetwork Configuration 2")
    print("loss     =", test_loss_2)
    print("accuracy = {:.2%}".format(test_accuracy_2))

    metrics = [history_1, history_2]
    legends = ['configuration 1', 'configuration 2']

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

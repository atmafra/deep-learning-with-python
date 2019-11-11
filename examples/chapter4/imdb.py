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

config_small = [
    {'layer_type': LayerType.DENSE, 'units': 4, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 4, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

config_medium = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

config_large = [
    {'layer_type': LayerType.DENSE, 'units': 512, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 512, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

config_medium_dropout = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

config_medium_wreg_l1 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1(0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1(0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

config_medium_wreg_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

config_medium_wreg_l1_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1_l2(l1=0.001, l2=0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1_l2(l1=0.001, l2=0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

config_medium_dropout_wreg_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
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
    global config_small
    global config_medium
    global config_large

    corpus = load_corpus(words=num_words)

    # training parameters
    validation_set_size = 10000
    epochs = 20
    batch_size = 512
    shuffle = True

    # small configuration
    # test_loss_small, test_accuracy_small, history_small = \
    #     run_configuration(network_configuration=network_configuration_global,
    #                       layer_configuration=layer_configuration_small,
    #                       corpus=corpus,
    #                       validation_set_size=validation_set_size,
    #                       epochs=epochs,
    #                       batch_size=batch_size,
    #                       shuffle=shuffle)

    # medium configuration
    test_loss_medium, test_accuracy_medium, history_medium = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=config_medium,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # large configuration
    # test_loss_large, test_accuracy_large, history_large = \
    #     run_configuration(network_configuration=network_configuration_global,
    #                       layer_configuration=layer_configuration_large,
    #                       corpus=corpus,
    #                       validation_set_size=validation_set_size,
    #                       epochs=epochs,
    #                       batch_size=batch_size,
    #                       shuffle=shuffle)

    # medium configuration with dropout
    test_loss_medium_dropout, test_accuracy_medium_dropout, history_medium_dropout = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=config_medium_dropout,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # medium configuration with weight regularization L1
    test_loss_medium_wreg_l1, test_accuracy_medium_wreg_l1, history_medium_wreg_l1 = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=config_medium_wreg_l1,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # medium configuration with weight regularization L2
    test_loss_medium_wreg_l2, test_accuracy_medium_wreg_l2, history_medium_wreg_l2 = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=config_medium_wreg_l2,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # medium configuration with weight regularization L1 and L2
    test_loss_medium_wreg_l1_l2, test_accuracy_medium_wreg_l1_l2, history_medium_wreg_l1_l2 = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=config_medium_wreg_l1_l2,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # medium configuration with dropout and weight regularization L2
    test_loss_medium_dropout_wreg_l2, test_accuracy_medium_dropout_wreg_l2, history_medium_dropout_wreg_l2 = \
        run_configuration(network_configuration=network_configuration_global,
                          layer_configuration_list=config_medium_dropout_wreg_l2,
                          corpus=corpus,
                          validation_set_size=validation_set_size,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle)

    # print results
    # print("\nSmall Network")
    # print("loss     =", test_loss_small)
    # print("accuracy = {:.2%}".format(test_accuracy_small))

    print("\nMedium Network")
    print("loss     =", test_loss_medium)
    print("accuracy = {:.2%}".format(test_accuracy_medium))

    # print("\nLarge Network")
    # print("loss     =", test_loss_large)
    # print("accuracy = {:.2%}".format(test_accuracy_large))

    print("\nMedium Network with Dropout")
    print("loss     =", test_loss_medium_dropout)
    print("accuracy = {:.2%}".format(test_accuracy_medium_dropout))

    print("\nMedium Network with Weight Regularization L1")
    print("loss     =", test_loss_medium_wreg_l1)
    print("accuracy = {:.2%}".format(test_accuracy_medium_wreg_l1))

    print("\nMedium Network with Weight Regularization L2")
    print("loss     =", test_loss_medium_wreg_l2)
    print("accuracy = {:.2%}".format(test_accuracy_medium_wreg_l2))

    print("\nMedium Network with Weight Regularizations L1 and L2")
    print("loss     =", test_loss_medium_wreg_l1_l2)
    print("accuracy = {:.2%}".format(test_accuracy_medium_wreg_l1_l2))

    print("\nMedium Network with Dropout and Weight Regularization L2")
    print("loss     =", test_loss_medium_dropout_wreg_l2)
    print("accuracy = {:.2%}".format(test_accuracy_medium_dropout_wreg_l2))

    # metrics = [history_small, history_medium, history_large]
    # legends = ['small network', 'medium network', 'large network']
    metrics = [history_medium, history_medium_dropout, history_medium_wreg_l1, history_medium_wreg_l2,
               history_medium_wreg_l1_l2, history_medium_dropout_wreg_l2]

    legends = ['medium network', 'medium with dropout',
               'medium with weight regularization L1',
               'medium with weight regularization L2',
               'medium with weight regularization L1 and L2',
               'medium with dropout and weight regularization L2']

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

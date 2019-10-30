import numpy as np
from keras.datasets import imdb

from core import network as net
from core.hyperparameters import LayerHyperparameters, NetworkHyperparameters, OutputType, LayerPosition
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


def hyperparameters(input_size: int,
                    hidden_size: list,
                    output_size: int,
                    hidden_activation: str,
                    output_activation: str,
                    loss: str):
    """Defines the IMDB neural model hyper parameters

    Args:
        input_size        (int) : input layer size (in units)
        hidden_size       (list): number of units in each hidden layer
        output_size       (int) : output layer size (in units)
        hidden_activation (str) : hidden layers activation function string
        output_activation (str) : output activation function string
        loss              (str) : loss function string

    """
    hidden_layers = len(hidden_size)

    # layer hyper parameters list
    input_layer_hparm = LayerHyperparameters(units=input_size,
                                             position=LayerPosition.INPUT,
                                             activation='linear')
    hidden_layers_hparm = []

    for size in hidden_size:
        hidden_layers_hparm.append(LayerHyperparameters(units=size,
                                                        position=LayerPosition.HIDDEN,
                                                        activation=hidden_activation))

    output_layer_hparm = LayerHyperparameters(units=output_size,
                                              position=LayerPosition.OUTPUT,
                                              activation=output_activation)

    layer_hparm_list = [input_layer_hparm] + hidden_layers_hparm + [output_layer_hparm]

    # network hyper parameters
    hparm = NetworkHyperparameters(input_size=input_size, output_size=1,
                                   output_type=OutputType.BOOLEAN,
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
                                             hidden_size=[16, 16],
                                             output_size=1,
                                             hidden_activation='relu',
                                             output_activation='sigmoid',
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
                                             hidden_size=[16, 16, 16],
                                             output_size=1,
                                             hidden_activation='relu',
                                             output_activation='sigmoid',
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

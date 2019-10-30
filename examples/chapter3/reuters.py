import numpy as np
from keras.datasets import reuters

from core import network as net
from core.hyperparameters import LayerHyperparameters, NetworkHyperparameters, OutputType, LayerPosition
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils import history_utils as hplt

if __name__ == '__main__':
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def load(num_words: int = 10000, encoding_schema: str = 'one-hot', verbose: bool = True):
    if verbose:
        print('Loading Reuters dataset...')

    corpus = reuters.load_data(num_words=num_words)
    (train_data, train_labels), (test_data, test_labels) = dsu.separate_corpus(corpus)

    # vectorization of the input data
    train_data = dsu.one_hot_encode(train_data, num_words)
    test_data = dsu.one_hot_encode(test_data, num_words)

    # vectorization of the labels
    categories = dsu.count_unique_values(train_labels)

    if encoding_schema == 'one-hot':
        train_labels = dsu.one_hot_encode(train_labels, categories)
        test_labels = dsu.one_hot_encode(test_labels, categories)

    elif encoding_schema == 'int-array':
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

    if verbose:
        print('Training phrases:', len(train_data))
        print('Test phrases    :', len(test_data))
        print('Categories      :', categories)

    return (train_data, train_labels), (test_data, test_labels)


def hyperparameters(input_size: int,
                    hidden_size: list,
                    output_size: int,
                    hidden_activation: str = 'relu',
                    output_activation: str = 'softmax',
                    loss: str = 'categorical_crossentropy'):
    """ IMDB neural network hyperparameters
    """

    # layers hyperparameters
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

    # network hyperparameters
    hparm = NetworkHyperparameters(input_size=input_size,
                                   output_size=output_size,
                                   output_type=OutputType.CATEGORICAL,
                                   layer_hyperparameters_list=layer_hparm_list,
                                   optimizer='rmsprop',
                                   learning_rate=0.001,
                                   loss=loss,
                                   metrics=['accuracy']
                                   )
    return hparm


def run(num_words: int = 10000, encoding_schema: str = 'one-hot'):
    # load corpus
    loss = 'categorical_crossentropy'
    if encoding_schema == 'int-array':
        loss = 'sparse_categorical_crossentropy'
    corpus = Corpus.from_tuple(load(num_words=num_words, encoding_schema=encoding_schema))

    # split validation set
    split_size = 1000
    validation_set, training_set_remaining = corpus.training_set.split(size=split_size)

    # define hyper parameters and create the neural network
    # categories = len(train_labels[0])
    categories = 46
    reuters_hparm = hyperparameters(input_size=num_words,
                                    hidden_size=[64, 64, 64],
                                    output_size=categories,
                                    hidden_activation='relu',
                                    output_activation='softmax',
                                    loss=loss)

    reuters_nnet = net.create_network(reuters_hparm)

    # train the neural network
    history = net.train_network(network=reuters_nnet,
                                training_set=training_set_remaining,
                                epochs=20,
                                batch_size=128,
                                shuffle=True,
                                validation_set=validation_set)

    (test_loss, test_accuracy) = net.test_network(reuters_nnet, corpus.test_set)
    print("loss     =", test_loss)
    print("accuracy = {:.2%}".format(test_accuracy))

    hplt.plot_accuracy_dict(history.history,
                            title='Reuters {}: Training and Validation Accuracies'.format(encoding_schema))

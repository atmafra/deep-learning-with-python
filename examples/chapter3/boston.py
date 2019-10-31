import numpy as np
from keras.datasets import boston_housing as boston

from core import network as net
from core.hyperparameters import LayerPosition, LayerHyperparameters, NetworkHyperparameters, NetworkOutputType
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils import history_utils as hutl


def load(num_words: int = 10000, encoding_schema: str = 'one-hot', verbose: bool = True):
    if verbose:
        print('Loading Boston Housing dataset...')

    corpus = boston.load_data()
    (train_inputs, train_outputs), (test_inputs, test_outputs) = dsu.separate_corpus(corpus)

    # normalization of the training and test data
    dsu.normalize(train_inputs, test_inputs)

    if verbose:
        print('Training examples:', len(train_inputs))
        print('Test examples    :', len(test_inputs))
        print('Minimum price    : {:.2f}'.format(np.min(train_outputs)))
        print('Average price    : {:.2f}'.format(np.average(train_outputs)))
        print('Maximum price    : {:.2f}'.format(np.max(train_outputs)))

    return (train_inputs, train_outputs), (test_inputs, test_outputs)


def hyperparameters(input_size: int,
                    hidden_size: list,
                    output_size: int,
                    hidden_activation: str,
                    output_activation: str,
                    loss: str = 'mse'):
    """ Boston Housing neural network hyperparameters
    """

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
    hparm = NetworkHyperparameters(input_size=input_size,
                                   output_size=output_size,
                                   output_type=NetworkOutputType.DECIMAL,
                                   layer_hyperparameters_list=layer_hparm_list,
                                   optimizer='rmsprop',
                                   learning_rate=0.001,
                                   loss=loss,
                                   metrics=['mae'])
    return hparm


def run():
    # load corpus
    corpus = Corpus.from_tuple(load())

    # define hyper parameters and create the neural network
    # input_size = train_data.shape[1]
    input_size = corpus.input_size()
    hidden_size = [64, 64]
    output_size = corpus.output_size()

    boston_hparm = hyperparameters(input_size=input_size,
                                   hidden_size=hidden_size,
                                   output_size=output_size,
                                   hidden_activation='relu',
                                   output_activation='linear',
                                   loss='mse')

    # create and compile the network
    boston_nnet = net.create_network(boston_hparm)

    # train the neural network
    history_list = net.train_network_k_fold(network=boston_nnet,
                                            epochs=80, batch_size=16, k=5, shuffle=True,
                                            training_set=corpus.training_set)

    # average fold results
    merged_metrics = hutl.merge_history_metrics(history_list)
    hutl.smooth_metrics_dict(merged_metrics, factor=0.9)

    # plots the Mean Absolute Error
    hutl.plot_loss_dict(merged_metrics,
                        title='Boston Housing: Training and Validation Mean Squared Error (MSE)')

    hutl.plot_mae_dict(merged_metrics,
                       title='Boston Housing: Training and Validation Mean Absolute Error (MAE)')

    # test the resulting model
    (test_mse_score, test_mae_score) = net.test_network(boston_nnet, corpus.test_set)
    print("loss =", test_mse_score)
    print("mae  =", test_mae_score)

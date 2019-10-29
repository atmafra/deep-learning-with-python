import numpy as np
from keras.datasets import boston_housing as boston

from core import hyperparameters as hpp
from core import network as net
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils import history_utils as hutl


def load(num_words: int = 10000, encoding_schema: str = 'one-hot', verbose: bool = True):
    if verbose:
        print('Loading Boston Housing dataset...')

    corpus = boston.load_data()
    (train_data, train_values), (test_data, test_output) = dsu.separate_corpus(corpus)

    # normalization of the training and test data
    dsu.normalize(train_data, test_data)

    if verbose:
        print('Training examples:', len(train_data))
        print('Test examples    :', len(test_data))
        print('Minimum price    : {:.2f}'.format(np.min(train_values)))
        print('Average price    : {:.2f}'.format(np.average(train_values)))
        print('Maximum price    : {:.2f}'.format(np.max(train_values)))

    return (train_data, train_values), (test_data, test_output)


def hyperparameters(input_size: int,
                    output_size: int = 1,
                    hidden_activation: str = 'relu',
                    output_activation: str = 'linear',
                    layer_units=None,
                    loss: str = 'mse'):
    """ Boston Housing neural network hyperparameters
    """
    if layer_units is None:
        layer_units = [64, 64, output_size]

    total_layers = len(layer_units)
    hidden_layers = total_layers - 2

    # layer hyper parameters list
    input_layer_hparm = hpp.LayerHyperparameters(layer_units[0], hidden_activation, input_size)
    hidden_layers_hparm = []
    for units in layer_units[1:1 + hidden_layers]:
        hidden_layers_hparm.append(hpp.LayerHyperparameters(units=units, activation=hidden_activation))
    output_layer_hparm = hpp.LayerHyperparameters(layer_units[-1], output_activation)
    layer_hparm_list = [input_layer_hparm] + hidden_layers_hparm + [output_layer_hparm]

    # network hyper parameters
    hparm = hpp.NetworkHyperparameters(input_size=input_size,
                                       output_size=output_size,
                                       output_type=hpp.OutputType.DECIMAL,
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
    output_size = corpus.output_size()
    boston_hparm = hyperparameters(input_size=input_size,
                                   output_size=output_size,
                                   hidden_activation='relu',
                                   output_activation='linear',
                                   layer_units=[64, 64, 1],
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

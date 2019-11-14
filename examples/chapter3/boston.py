import numpy as np
from keras.datasets import boston_housing as boston

from core import network as net
from core.network import LayerType, NetworkOutputType
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


def hyperparameters(input_size: int, output_size: int):
    """ Boston Housing neural network hyperparameters
    """
    # network hyper parameters
    network_configuration = {
        'input_size': input_size,
        'output_size': output_size,
        'output_type': NetworkOutputType.DECIMAL,
        'optimizer': 'rmsprop',
        'learning_rate': 0.001,
        'loss': 'mse',
        'metrics': ['mae']}

    # layer hyperparameters list
    layers_configuration = [
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': 'relu', 'input_shape': (input_size,)},
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': 'relu'},
        {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': 'linear'}]

    return network_configuration, layers_configuration


def run():
    # load corpus
    corpus = Corpus.from_tuple(load())

    # define hyper parameters and create the neural network
    input_size = corpus.input_size
    output_size = corpus.output_size

    network_configuration, layers_configuration = \
        hyperparameters(input_size=input_size, output_size=output_size)

    # create and compile the network
    boston_nnet = net.create_network(network_configuration=network_configuration,
                                     layer_configuration_list=layers_configuration)

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

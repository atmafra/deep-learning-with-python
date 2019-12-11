import os
import sys
from enum import Enum

from keras import models, Model
from keras.engine.saving import load_model
from keras.utils import Sequence

import utils.parameter_utils as putl
from core.sets import Set, SetGenerator
from utils.file_utils import str_to_filename, root_name
from utils.parameter_utils import extract_parameter


class NetworkOutputType(Enum):
    BOOLEAN = 1
    CATEGORICAL = 2
    DECIMAL = 3


class LayerPosition(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class ValidationStrategy(Enum):
    NO_VALIDATION = 1
    CROSS_VALIDATION = 2
    K_FOLD_CROSS_VALIDATION = 3


default_optimizer: str = 'rmsprop'
default_learning_rate: float = 0.001
default_loss: dict = {
    NetworkOutputType.BOOLEAN: 'binary_crossentropy',
    NetworkOutputType.CATEGORICAL: 'categorical_crossentropy',
    NetworkOutputType.DECIMAL: 'mean_squared_error'}


def create_layer(parameters: dict):
    """Creates a layer according to the hyperparameters

    Args:
        parameters (dict): hyperparameters dictionary

    """
    parameters_copy = parameters.copy()
    layer_type = extract_parameter(parameters=parameters_copy,
                                   key='layer_type',
                                   mandatory=True)

    layer = getattr(sys.modules['keras.layers'], layer_type)
    if layer is None:
        raise RuntimeError('Invalid layer type: \'{}\''.format(layer_type))

    return layer(**parameters_copy)


def create_network(layer_configuration_list: list):
    """Creates a neural network according to its hyper parameters

    Args:
        layer_configuration_list (list): list of layer hyperparameters

    """
    network = models.Sequential()

    # layers
    for layer_configuration in layer_configuration_list:
        layer = create_layer(layer_configuration)
        if layer is not None:
            network.add(layer)

    network.build()

    return network


def train_network(network: Model,
                  training_configuration: dict,
                  training_set: Set,
                  validation_set: Set = None,
                  verbose: bool = True):
    """Train the neural network, returning the evolution of the training metrics

    Args:
        network (Model): neural network model to be trained
        training_configuration (dict): training algorithm parameters
        training_set (Set): training set
        validation_set (Set): validation set
        verbose (bool): display training progress bars if True

    """
    validation = putl.get_parameter(training_configuration, 'validation')
    validation_strategy = putl.get_parameter(validation, 'strategy')
    working_training_set = training_set

    validation_data = None
    if validation_set is not None:
        validation_data = validation_set.to_datasets()

    keras_parameters = putl.get_parameter(training_configuration, 'keras')
    compile_parameters = putl.get_parameter(keras_parameters, 'compile')
    network.compile(**compile_parameters)

    fit_parameters = putl.get_parameter(keras_parameters, 'fit')
    history = network.fit(x=working_training_set.input_data,
                          y=working_training_set.output_data,
                          validation_data=validation_data,
                          verbose=verbose,
                          **fit_parameters)

    return history


def train_network_generator(network: Model,
                            training_generator: Sequence,
                            training_configuration: dict,
                            validation_generator: Sequence = None,
                            verbose: bool = True):
    """Train the neural network, returning the evolution of the training metrics

    Args:
        network (Model): neural network model to be trained
        training_generator (Sequence): training set generator
        training_configuration (dict): training algorithm parameters
        validation_generator (Sequence): validation set generator
        verbose (bool): display training progress bars if True

    """
    validation = putl.get_parameter(training_configuration, 'validation')
    validation_strategy = putl.get_parameter(validation, 'strategy')

    keras_parameters = putl.get_parameter(training_configuration, 'keras')
    compile_parameters = putl.get_parameter(keras_parameters, 'compile')
    network.compile(**compile_parameters)

    fit_generator_parameters = putl.get_parameter(keras_parameters, 'fit_generator')
    history = network.fit_generator(generator=training_generator,
                                    validation_data=validation_generator,
                                    verbose=verbose,
                                    **fit_generator_parameters)

    return history


def train_network_k_fold(network: Model,
                         training_configuration: dict,
                         training_set: Set,
                         k: int,
                         shuffle: bool,
                         verbose: bool = True):
    """Train the neural network model using k-fold cross-validation

    Args:
        network (Model): neural network model to be trained
        training_configuration (dict): training parameters
        training_set (Set): training data set
        k (int): number of partitions in k-fold cross-validation
        shuffle (bool): shuffle the training set before k splitting
        verbose (bool): display training progress bars if True

    """
    all_histories = []
    working_training_set = training_set.copy()

    if shuffle:
        working_training_set.shuffle()

    for fold in range(k):
        print('processing fold #', fold)
        validation_set, training_set_remaining = working_training_set.split_k_fold(fold, k)

        # build the model and train the current fold
        network.build()

        fold_history = train_network(network=network,
                                     training_set=training_set_remaining,
                                     validation_set=validation_set,
                                     training_configuration=training_configuration,
                                     verbose=verbose)

        all_histories.append(fold_history)

    return all_histories


def test_network(network: Model,
                 test_set: Set,
                 verbose: bool = True) -> dict:
    """Evaluates all the test inputs according to the current network

    Args:
        network (Model): neural network model to be tested
        test_set (Set): test set to be used for metrics evaluation
        verbose (bool): display evaluation progress bar if True

    """
    result = network.evaluate(x=test_set.input_data,
                              y=test_set.output_data,
                              batch_size=None,
                              verbose=verbose,
                              sample_weight=None,
                              steps=None,
                              callbacks=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False)

    metrics = {}
    i = 0
    for name in network.metrics_names:
        metrics[name] = result[i]
        i = i + 1

    return metrics


def test_network_generator(network: Model,
                           test_set_generator: SetGenerator,
                           verbose: bool = True) -> dict:
    """Evaluates all the test inputs according to the current network

    Args:
        network (Model): neural network model to be tested
        test_set_generator (SetGenerator): test set generator to be used for metrics evaluation
        verbose (bool): display evaluation progress bar if True

    """
    result = network.evaluate_generator(generator=test_set_generator.generator,
                                        steps=None,
                                        callbacks=None,
                                        max_queue_size=10,
                                        workers=1,
                                        use_multiprocessing=False,
                                        verbose=verbose)

    metrics = {}
    i = 0
    for name in network.metrics_names:
        metrics[name] = result[i]
        i = i + 1

    return metrics


def save_network_json(network: Model,
                      path: str,
                      filename: str,
                      verbose: bool = True):
    """Saves the model in the current state to a JSON file

    Args:
        network (Model): model to be saved
        path (str): system path of the save directory
        filename (str): saved model file name
        verbose (bool): show save message in terminal

    """
    if filename == '':
        filename = str_to_filename(network.name)
    rootname = root_name(filename)
    model_name = os.path.join(path, rootname + '.json')
    weights_name = os.path.join(path, rootname + '.h5')

    model_json = network.to_json()
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)
        if verbose:
            print('Saving network architecture \"{}\" to file \"{}\"'.format(network.name, model_name))
    network.save_weights(filepath=weights_name, overwrite=True)
    if verbose:
        print('Saving network weights to file \"{}\"'.format(weights_name))


def save_network_hdf5(network: Model,
                      path: str,
                      file_name: str,
                      verbose: bool = True):
    """Saves the model in the current state to a H5PY file

    Args:
        network (Model): model to be saved
        path (str): system path of the save directory
        file_name (str): saved model file name
        verbose (bool): show save message in terminal

    """
    file_path = os.path.join(path, file_name)
    network.save(filepath=file_path)
    if verbose:
        print('Saved neural network model \"{}\" to file \"{}\"'.format(network.name, file_path))


def load_network_hdf5(path: str,
                      file_name: str,
                      verbose: bool = True):
    """Loads a model in the state it was saved

    Args:
        path (str): system path of the load directory
        file_name (str): file name of the model to be loaded
        verbose (bool): show load message in terminal

    """
    file_path = os.path.join(path, file_name)
    loaded_network = load_model(filepath=file_path, compile=False)
    if verbose:
        print('Loaded neural network model from file \"{}\"'.format(file_path))
    return loaded_network

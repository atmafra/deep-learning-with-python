import os
import sys
from enum import Enum

from keras import Model, Sequential
from keras.engine.saving import load_model, model_from_json
from keras.layers.wrappers import Wrapper
from keras.utils import Sequence, layer_utils

import utils.parameter_utils as putl
from core.datasets import Dataset, DatasetFileIterator
from utils.file_utils import str_to_filename
from utils.parameter_utils import extract_parameter

default_architecture_extension='-arch.json'
default_weights_extension='-wght.h5'


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
    """ Creates a layer according to the hyperparameters

    :param parameters: hyperparameters dictionary
    :returns: new layer
    """
    layer_parameters = parameters.copy()
    layer_class = extract_parameter(parameters=layer_parameters,
                                    key='class_name',
                                    mandatory=True)

    layer = getattr(sys.modules['keras.layers'], layer_class)
    if layer is None:
        raise RuntimeError('Invalid layer type: \'{}\''.format(layer_class))

    if not issubclass(layer, Wrapper):
        return layer(**layer_parameters)

    sublayer_parameters = extract_parameter(parameters=layer_parameters,
                                            key='layer',
                                            mandatory=True)

    sublayer = create_layer(sublayer_parameters)

    return layer(layer=sublayer, **layer_parameters)


def create_model(name: str,
                 layer_configuration_list: list):
    """ Creates a neural network according to its hyper parameters

    :param name: neural network name
    :param layer_configuration_list: list of layer hyperparameters
    :return: Keras sequential model
    """
    model = Sequential()
    model.name = name

    if layer_configuration_list is not None:
        add_layers(model=model, layer_configuration_list=layer_configuration_list)

    return model


def add_layers(model: Sequential,
               layer_configuration_list: list):
    """ Append layers to an existing sequential model

    :param model: previously created sequential model
    :param layer_configuration_list: list of layer hyperparameters
    """
    if model is None:
        raise RuntimeError('Null sequential model trying to append layers')

    for layer_configuration in layer_configuration_list:
        layer = create_layer(layer_configuration)
        if layer is not None:
            model.add(layer)

    model.build()


def append_model(base_model: Sequential,
                 model: Model):
    """ Appends a model to the current sequential model (in-place)

    :param base_model: current sequential model
    :param model: model to be appended
    """
    if base_model is None:
        raise ValueError('Empty base_model')

    if model is None:
        raise ValueError('Empty model to append to base model')

    base_model.add(model)
    base_model.build()


def create_model_from_file(filepath: str,
                           verbose: bool = True):
    """ Creates a new model from a JSON architecture file

    :param filepath: fully qualified path to JSON file
    :param verbose: display load messages in terminal
    """
    json_file = open(filepath, 'r')
    network_architecture_json = json_file.read()
    json_file.close()
    model = model_from_json(network_architecture_json)
    model.build()

    if verbose:
        print('Loaded model \"{}\" from file \"{}\"'.format(model.name, filepath))

    return model


def count_trainable_parameters(model: Model) -> int:
    if not Model:
        raise ValueError('Empty model to count trainable parameters')

    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = layer_utils.count_params(model._collected_trainable_weights)
    else:
        trainable_count = layer_utils.count_params(model.trainable_weights)

    return trainable_count


def count_non_trainable_parameters(model: Model) -> int:
    if not Model:
        raise ValueError('Empty model to count non-trainable parameters')

    return layer_utils.count_params(model.non_trainable_weights)


def count_parameters(model: Model) -> int:
    if not Model:
        raise ValueError('Empty model to count non-trainable parameters')

    return model.count_params()


def train_network(network: Model,
                  training_configuration: dict,
                  training_set: Dataset,
                  validation_set: Dataset = None,
                  use_sample_weights: bool = False,
                  verbose: bool = True):
    """ Train the neural network, returning the evolution of the train metrics

    :param network: neural network model to be trained
    :param training_configuration: train algorithm parameters
    :param training_set: train set
    :param validation_set: validation set
    :param use_sample_weights: use sample weights in traininig if defined
    :param verbose: display train progress bars if True
    """
    compile_parameters = putl.get_parameter(parameters=training_configuration,
                                            key='keras.compile',
                                            mandatory=True)

    fit_parameters = putl.get_parameter(parameters=training_configuration,
                                        key='keras.fit',
                                        mandatory=True)

    validation_data = None
    if validation_set is not None:
        validation_data = validation_set.to_array_pair()

    network.compile(**compile_parameters)

    working_training_set = training_set

    sample_weight = None
    if use_sample_weights:
        sample_weight = training_set.sample_weights

    history = network.fit(x=working_training_set.input_data,
                          y=working_training_set.output_data,
                          validation_data=validation_data,
                          sample_weight=sample_weight,
                          verbose=verbose,
                          **fit_parameters)

    return history


def train_network_generator(network: Model,
                            training_generator: Sequence,
                            training_configuration: dict,
                            validation_generator: Sequence = None,
                            verbose: bool = True):
    """Train the neural network, returning the evolution of the train metrics

    Args:
        network (Model): neural network model to be trained
        training_generator (Sequence): train set generator
        training_configuration (dict): train algorithm parameters
        validation_generator (Sequence): validation set generator
        verbose (bool): display train progress bars if True

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
                         training_set: Dataset,
                         k: int,
                         shuffle: bool,
                         verbose: bool = True):
    """Train the neural network model using k-fold cross-validation

    Args:
        network (Model): neural network model to be trained
        training_configuration (dict): train parameters
        training_set (Dataset): train data set
        k (int): number of partitions in k-fold cross-validation
        shuffle (bool): shuffle the train set before k splitting
        verbose (bool): display train progress bars if True

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


def evaluate_dataset(network: Model,
                     test_set: Dataset,
                     use_sample_weights: bool = False,
                     verbose: bool = True) -> dict:
    """ Evaluates all the test inputs according to the given network

    :param network: neural network model to be tested
    :param test_set: test set to be used for metrics evaluation
    :param use_sample_weights: use sample weights in evaluation
    :param verbose: display evaluation progress bar if True
    :return: map of performance metrics
    """
    metrics = {}

    sample_weight = None
    if use_sample_weights:
        sample_weight = test_set.sample_weights

    if test_set.length > 0:
        result = network.evaluate(x=test_set.input_data,
                                  y=test_set.output_data,
                                  batch_size=None,
                                  verbose=verbose,
                                  sample_weight=sample_weight,
                                  steps=None,
                                  callbacks=None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False)

        i = 0
        for name in network.metrics_names:
            metrics[name] = result[i]
            i = i + 1

    return metrics


def evaluate_dataset_generator(network: Model,
                               test_set_generator: DatasetFileIterator,
                               verbose: bool = True) -> dict:
    """ Evaluates all the test inputs according to the current network

    :param network: neural network model to be tested
    :param test_set_generator: test set generator to be used for metrics evaluation
    :param verbose: display evaluation progress bar if True
    """
    result = network.evaluate_generator(generator=test_set_generator.directory_iterator,
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


def predict_dataset(network: Model,
                    dataset: Dataset,
                    verbose: bool = True) -> dict:
    """ Gives the predictions for each entry in the dataset

    :param network: neural network model to be tested
    :param dataset: test set to be used for metrics evaluation
    :param verbose: display evaluation progress bar if True
    :return: map of performance metrics
    """
    predictions = network.predict(x=dataset.input_data,
                                  batch_size=None,
                                  verbose=verbose,
                                  steps=None,
                                  callbacks=None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False)

    return predictions


def predict_dataset_generator(network: Model,
                              dataset_generator: DatasetFileIterator,
                              verbose: bool = True) -> dict:
    """ Gives the predictions for each entry from the dataset generator

    :param network: neural network model to be tested
    :param dataset_generator: test set to be used for metrics evaluation
    :param verbose: display evaluation progress bar if True
    :return: map of performance metrics
    """
    predictions = network.predict_generator(generator=dataset_generator,
                                            batch_size=None,
                                            verbose=verbose,
                                            steps=None,
                                            callbacks=None,
                                            max_queue_size=10,
                                            workers=1,
                                            use_multiprocessing=False)

    return predictions


def build_filepath(filename: str,
                   path: str,
                   extension: str = '',
                   default_filename: str = ''):
    """ Builds the file path from filename, path, and extension

    :param filename: operating system filename
    :param path: system path of the file
    :param extension: filename extension
    :param default_filename: default filename
    :return: complete system file path
    """
    if not filename:
        if default_filename:
            filename = str_to_filename(default_filename) + extension
        else:
            raise ValueError('Cannot define filename')

    if not path:
        filepath = filename
    else:
        filepath = os.path.join(path, filename)

    return filepath


def save_architecture_json(network: Model,
                           path: str,
                           filename: str = '',
                           verbose: bool = True):
    """ Saves the model architecture as a JSON file

    :param network: model whose architecture must be saved
    :param path: system path of the save directory
    :param filename: architecture file name
    :param verbose: show save message in terminal
    """
    model_filepath = build_filepath(filename=filename,
                                    path=path,
                                    extension=default_architecture_extension,
                                    default_filename=network.name)
    model_json = network.to_json()

    if verbose:
        print('Saving network architecture \"{}\" to file \"{}\"...'.format(network.name, model_filepath))

    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)

    if verbose:
        print('Save architecture OK')


def save_weights_h5(network: Model,
                    filename: str = '',
                    path: str = '',
                    verbose: bool = True):
    """ Saves the model weights to a h5 file

    :param network: neural network model
    :param verbose: show save message in terminal
    :param filename: weights file name
    :param path: system path of the save directory
    """
    weights_filepath = build_filepath(filename=filename,
                                      path=path,
                                      extension=default_weights_extension,
                                      default_filename=network.name)

    if verbose:
        print('Saving model \"{}\" weights to file \"{}\"...'.format(network.name, weights_filepath))

    network.save_weights(filepath=weights_filepath, overwrite=True)

    if verbose:
        print('Save weights OK')


def save_network(network: Model,
                 path: str,
                 root_name: str,
                 verbose: bool = True):
    """ Saves the model in the current state in 2 files:

    - a JSON file, containing the architecture (.json)
    - the binary H5 (pickle) file, containing the weights (.h5)

    :param network: model to be saved
    :param path: system path of the save directory
    :param root_name: root of the model and weights filenames
    :param verbose: show save message in terminal
    """
    if root_name is None:
        root_name = str_to_filename(network.name)

    save_architecture_json(network=network,
                           filename=root_name,
                           path=path,
                           verbose=verbose)

    save_weights_h5(network=network,
                    filename=root_name,
                    path=path,
                    verbose=verbose)


def load_network(architecture_filepath: str,
                 weights_filepath: str,
                 verbose: bool = True):
    """ Loads a file containing a representation of a JSON object
    that describes the architecture of a neural network model

    :param architecture_filepath: architecture filename
    :param weights_filepath: weights filename
    :param verbose: show load message on terminal
    """
    model = create_model_from_file(filepath=architecture_filepath, verbose=verbose)

    if weights_filepath is not None:
        if weights_filepath != '':
            model.load_weights(weights_filepath)
        else:
            raise RuntimeError('A full path must be specified in order to load pre-trained model weights')

        if verbose:
            print('Loaded weights from file \"{}\"'.format(weights_filepath))

    return model


def save_network_hdf5(network: Model,
                      path: str,
                      filename: str,
                      verbose: bool = True):
    """ Saves the model in the current state to a H5PY file

    :param network: model to be saved
    :param path: system path of the save directory
    :param filename: saved model file name
    :param verbose: show save message in terminal
    """
    filepath = os.path.join(path, filename)
    network.save(filepath=filepath)
    if verbose:
        print('Saved neural network model \"{}\" to file \"{}\"'.format(network.name, filepath))


def load_network_hdf5(path: str,
                      filename: str,
                      verbose: bool = True):
    """ Loads a model in the state it was saved

    :param path: system path of the load directory
    :param filename: file name of the model to be loaded
    :param verbose: show load message in terminal
    """
    filepath = os.path.join(path, filename)
    loaded_network = load_model(filepath=filepath, compile=False)
    if verbose:
        print('Loaded neural network model from file \"{}\"'.format(filepath))
    return loaded_network

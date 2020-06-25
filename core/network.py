import os
import sys
from enum import Enum

from keras import Model, Sequential
from keras.engine.network import Network
from keras.engine.saving import load_model, model_from_json
from keras.layers.wrappers import Wrapper
from keras.utils import Sequence, layer_utils

import utils.parameter_utils as putl
from core.datasets import Dataset, DatasetFileIterator
from utils.file_utils import str_to_filename
from utils.parameter_utils import extract_parameter

default_architecture_extension = '-arch.json'
default_weights_extension = '-wght.h5'


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
    """ Train the neural network, returning the evolution of the training metrics

    :param network: neural network model to be trained
    :param training_generator: train set generator
    :param training_configuration: train algorithm parameters
    :param validation_generator: validation set generator
    :param verbose: display train progress bars if True
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
    """ Train the neural network model using k-fold cross-validation

    :param network: neural network model to be trained
    :param training_configuration: train parameters
    :param training_set: train data set
    :param k: number of partitions in k-fold cross-validation
    :param shuffle: shuffle the train set before k splitting
    :param verbose: display train progress bars if True
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


def build_filepath(path: str,
                   filename: str,
                   extension: str = '',
                   default_filename: str = ''):
    """ Builds the file path from filename, path, and extension

    :param path: system path of the file
    :param filename: operating system filename
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


def default_architecture_filename(path: str, model_name: str):
    """ Returns the default architecture filename for the model

    :param path: system path of the file
    :param model_name: Keras model name
    :return: complete system filepath
    """
    return build_filepath(path=path,
                          extension=default_architecture_extension,
                          default_filename=model_name)


def default_weights_filename(path: str, model_name: str):
    """ Returns the default weights filename for the model

    :param path: system path of the file
    :param model_name: Keras model name
    :return: complete system filepath
    """
    return build_filepath(path=path,
                          extension=default_weights_extension,
                          default_filename=model_name)


def save_architecture_json(model: Model,
                           path: str,
                           architecture_filename: str = '',
                           verbose: bool = True):
    """ Saves the model architecture as a JSON file

    :param model: model whose architecture must be saved
    :param path: system path of the save directory
    :param architecture_filename: architecture file name
    :param verbose: show save message in terminal
    """
    model_filepath = build_filepath(filename=architecture_filename,
                                    path=path,
                                    extension=default_architecture_extension,
                                    default_filename=model.name)
    model_json = model.to_json()

    if verbose:
        print('Saving network architecture \"{}\" to file \"{}\"...'.format(model.name, model_filepath))

    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)

    if verbose:
        print('Save architecture OK')


def load_architecture_json(path: str,
                           architecture_filename: str,
                           verbose: bool = True):
    """ Creates and builds a new model from a JSON architecture file

    :param path: system path of the configuration file
    :param architecture_filename: JSON file with the model architecture
    :param verbose: display load messages in terminal
    :return: Keras untrained model according to the JSON file architecture
    """
    if not architecture_filename:
        raise RuntimeError('Cannot load architecture: no filepath passed')

    filepath = os.path.join(path, architecture_filename)
    if verbose:
        print('Loading network architecture from file "{}"'.format(filepath))

    json_file = open(filepath, 'r')
    network_architecture_json = json_file.read()
    json_file.close()
    model = model_from_json(network_architecture_json)
    model.build()

    if verbose:
        print('Loaded the archtecture of model \"{}\" from file \"{}\"'.format(model.name, filepath))

    return model


def save_weights(model: Model,
                 path: str = '',
                 weights_filename: str = '',
                 verbose: bool = True):
    """ Saves the model weights to a h5 file

    :param model: neural network model
    :param path: system path of the save directory
    :param weights_filename: weights file name
    :param verbose: show save message in terminal
    """
    weights_filepath = build_filepath(filename=weights_filename,
                                      path=path,
                                      extension=default_weights_extension,
                                      default_filename=model.name)

    if verbose:
        print('Saving model \"{}\" weights to file \"{}\"...'.format(model.name, weights_filepath))

    model.save_weights(weights_filepath, overwrite=True)

    if verbose:
        print('Save weights OK')


def load_weights(model: Model,
                 path: str,
                 weights_filename: str,
                 verbose: bool = True):
    """ Loads a model with weights previously saved

    :param model: pre-loaded Keras model
    :param path: system path of the weights file
    :param weights_filename: name of the file containing the weight values
    :param verbose: display processing messages in the terminal output
    :return: model loaded and built from the file
    """
    if not weights_filename:
        raise RuntimeError('Cannot load weights: no weights filename passed')

    filepath = os.path.join(path, weights_filename)
    if verbose:
        print('Loading model weights from file "{}"'.format(filepath))

    model.load_weights(filepath)


def save_architecture_and_weights(model: Model,
                                  path: str,
                                  architecture_filename: str,
                                  weights_filename: str,
                                  verbose: bool = True):
    """ Saves the architecture and the weights in the current state in 2 files:

    - a JSON file, containing the architecture (.json)
    - the binary H5 (pickle) file, containing the weights (.h5)

    :param model: model to be saved
    :param path: system path of the save directory
    :param architecture_filename: architecture filename
    :param weights_filename: weights filename
    :param verbose: show save message in terminal
    """
    if not architecture_filename:
        architecture_filename = default_architecture_filename(path=path, model_name=model.name)

    if not weights_filename:
        weights_filename = default_weights_filename(path=path, model_name=model.name)

    save_architecture_json(model=model,
                           path=path,
                           architecture_filename=architecture_filename,
                           verbose=verbose)

    save_weights(model=model,
                 path=path,
                 weights_filename=weights_filename,
                 verbose=verbose)


def load_architecture_and_weights(path: str,
                                  architecture_filename: str,
                                  weights_filename: str,
                                  verbose: bool = True):
    """ Loads a file containing a representation of a JSON object
    that describes the architecture of a neural network model

    :param architecture_filepath: architecture filename
    :param path: weights filename
    :param architecture_filename: JSON file containing the model architecture
    :param weights_filename: filename of the saved weights
    :param verbose: show load message on terminal
    """
    model = load_architecture_json(path=path,
                                   architecture_filename=architecture_filename,
                                   verbose=verbose)

    load_weights(model=model,
                 path=path,
                 weights_filename=weights_filename,
                 verbose=verbose)

    if verbose:
        print('Architecture and weights successfully loaded')

    return model


def save_model_hdf5(model: Model,
                    path: str,
                    filename: str = None,
                    verbose: bool = True):
    """ Saves the model in the current state to a H5PY file

    :param model: model to be saved
    :param path: system path of the save directory
    :param filename: saved model file name (if None, will be built using model name)
    :param verbose: show save message in terminal
    """
    if not filename:
        filename = str_to_filename(model.name) + '.h5py'
    filepath = os.path.join(path, filename)
    if verbose:
        print('Saving neural network model "{}" to file "{}"'.format(model.name, filepath))

    model.save(filepath=filepath)

    if verbose:
        print('Model saved successfully')


def load_model_hdf5(path: str,
                    filename: str,
                    verbose: bool = True):
    """ Loads a model in the state it was saved

    :param path: system path of the load directory
    :param filename: file name of the model to be loaded
    :param verbose: show load message in terminal
    """
    filepath = os.path.join(path, filename)
    loaded_model = load_model(filepath, compile=True)
    if verbose:
        print('Loaded neural network model from file "{}"'.format(filepath))

    return loaded_model

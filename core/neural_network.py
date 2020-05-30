import os.path

import numpy as np
from keras import Sequential, Model
from keras.engine import Layer

import core.network as net
from core.corpus import CorpusFiles
from core.datasets import Dataset, DatasetFileIterator
from core.training_configuration import TrainingConfiguration
from utils.history_utils import merge_history_metrics
from utils.parameter_utils import get_parameter


class NeuralNetwork():

    def __init__(self, model: Sequential = Sequential(),
                 name: str = None):
        """ Instantiates a neural network object that encapsulates the Keras sequential model

        :param model: Keras sequential model
        :param name: neural network name
        """
        if not isinstance(model, Sequential):
            raise TypeError('Base model must be Sequential')

        self.__layer_map: map = {}
        self.model: Sequential = model

        if name:
            self.name = name

    @property
    def model(self) -> Sequential:
        return self.__model

    def __map_layers(self, model: Sequential):
        for layer in model.layers:
            if isinstance(layer, Model) or isinstance(layer, Sequential):
                self.__map_layers(layer)
            self.__layer_map[layer.name] = layer

    @model.setter
    def model(self, model: Sequential):
        self.__map_layers(model)
        self.__model = model

    @property
    def name(self) -> str:
        if self.model is None:
            return ''
        return self.model.name

    @name.setter
    def name(self, name: str):
        if self.model is not None:
            self.model.name = name

    @classmethod
    def from_configurations(cls, name: str, layers_configuration: list):
        """ Creates a new neural network from configurations

        :param name: neural network name
        :param layers_configuration: neural network architecture
        :return: a new NeuralNetwork object
        """
        model = net.create_model(name=name, layer_configuration_list=layers_configuration)
        return NeuralNetwork(model=model)

    @classmethod
    def from_file(cls, path: str, filename: str, verbose: bool = True):
        """ Creates a new neural network from JSON architecture file

        :param path: file path
        :param filename: file name
        :param verbose: display load messages in terminal
        :return: a new NeuralNetwork object
        """
        filepath = os.path.join(path, filename)
        model = net.create_model_from_file(filepath=filepath, verbose=verbose)
        return NeuralNetwork(model=model)

    def append_layers(self, layers_configuration: list):
        """ Appends a list of layers according to their configurations

        :param layers_configuration: list of layer configurations to be appended
        """
        if layers_configuration is None:
            raise ValueError('Null model passed, trying to append neural network')

        net.add_layers(self.model, layers_configuration)

    def append_model(self, model: Model):
        """ Appends a pre-existing model to the current model in a sequential way

        :param model: model to be appended to the base model
        """
        if model is None:
            raise ValueError('Error trying to append model to the current neural network: Null model passed to append')

        net.append_model(base_model=self.model, model=model)
        self.__map_layers(model=model)

    def get_layer(self, layer_name: str) -> Layer:
        """ Gets a layer by its name

        :param layer_name: requested layer name
        :return: layer (if found) or None
        """
        if layer_name:
            if layer_name in self.__layer_map:
                return self.__layer_map[layer_name]
            else:
                raise ValueError('No layer with name "{}"'.format(layer_name))
        else:
            raise ValueError('No layer name passed to get_layer()')

    def get_weights(self, layer_name: str) -> list:
        """ Gets the weights of a layer, given its name

        :param layer_name: layer name
        :return: weights as a matrix
        """
        layer = self.get_layer(layer_name)
        if layer is not None:
            return layer.get_weights()

    def set_weights(self, layer_name: str, weights: np.array):
        """ Sets the weights of a layer, given its name and the weight matrix

        :param layer_name: layer name
        :param weights: weight matrix
        """
        layer = self.get_layer(layer_name)
        layer.set_weights(weights=weights)

    def set_layers_trainable(self, layer_names: list, trainable: bool):
        """ Freezes or unfreezes the layers in the list,
        avoiding or allowing its weights to be updated during training

        :param layer_names: list of layer names
        :param trainable: sets if the layer is trainable or not
        """
        if layer_names is not None:
            for layer_name in layer_names:
                layer = self.get_layer(layer_name)
                if layer is not None:
                    layer.trainable = trainable
                else:
                    raise ValueError('Layer not found, trying to freeze layer: {}'.format(layer_name))
        else:
            raise ValueError('Empty layer name vector trying to freeze layers')

    def freeze_layer(self, layer_name: str):
        """ Freezes the layer, avoiding its weights to be updated during training

        :param layer_name: layer name
        """
        self.set_layers_trainable([layer_name], False)

    def unfreeze_layer(self, layer_name: str):
        """ Unfreezes the layer, allowing its weights to be updated during training

        :param layer_name: layer name
        """
        self.set_layers_trainable([layer_name], True)

    def save_architecture(self, path: str, filename: str, verbose: bool = True):
        """ Saves the neural network architecture as a JSON file

        :param path: system path of the save directory
        :param filename: architecture file name
        :param verbose:  display save messages on terminal
        """
        net.save_architecture_json(network=self.model, path=path, filename=filename, verbose=verbose)

    def save_weights(self, path: str, filename: str, verbose: bool = True):
        """ Saves the neural network weights (status) as a H5 file

        :param path: system path of the save directory
        :param filename: weights file name
        :param verbose: show save messages in terminal
        """
        net.save_weights_h5(network=self.model, path=path, filename=filename, verbose=verbose)

    def save_model(self, path: str, root_filename: str, verbose: bool = True):
        """ Saves the neural network architecture and weights in the same file system path

        :param path: system path of the save directory
        :param root_filename: root file name (without extension)
        :param verbose: show save messages in terminal
        """
        architecture_filename = ''
        weights_filename = ''
        if len(root_filename) > 0:
            architecture_filename = root_filename + '.json'
            weights_filename = root_filename + '.h5'

        self.save_architecture(path=path, filename=architecture_filename, verbose=verbose)
        self.save_weights(path=path, filename=weights_filename, verbose=verbose)

    @property
    def trainable_parameters_count(self):
        return net.count_trainable_parameters(self.model)

    @property
    def non_trainable_parameters_count(self):
        return net.count_non_trainable_parameters(self.model)

    @property
    def parameters_count(self):
        return net.count_parameters(self.model)

    def train(self,
              training_set: Dataset,
              training_configuration: TrainingConfiguration,
              validation_set: Dataset = None,
              display_progress_bars: bool = True):
        """ Trains the neural network

        :param training_set: training set
        :param training_configuration: train configuration hyperparameters
        :param validation_set: validation set (optional, depending on validation strategy)
        :param display_progress_bars: display progress bars in terminal during train
        """
        if training_configuration is None:
            raise RuntimeError('No train configuration defined')

        strategy = training_configuration.validation_strategy
        training_history = None

        if strategy == net.ValidationStrategy.NO_VALIDATION:
            training_history = net.train_network(network=self.model,
                                                 training_configuration=training_configuration.configuration,
                                                 training_set=training_set,
                                                 validation_set=None,
                                                 verbose=display_progress_bars)

        elif strategy == net.ValidationStrategy.CROSS_VALIDATION:
            training_history = net.train_network(network=self.model,
                                                 training_configuration=training_configuration.configuration,
                                                 training_set=training_set,
                                                 validation_set=validation_set,
                                                 verbose=display_progress_bars)

        elif strategy == net.ValidationStrategy.K_FOLD_CROSS_VALIDATION:
            validation_configuration = training_configuration.validation_configuration
            k = get_parameter(validation_configuration, 'k')
            shuffle = get_parameter(validation_configuration, 'shuffle')
            all_histories = net.train_network_k_fold(network=self.model,
                                                     training_configuration=training_configuration.configuration,
                                                     training_set=training_set,
                                                     k=k, shuffle=shuffle,
                                                     verbose=display_progress_bars)

            training_history = merge_history_metrics(all_histories)

        return training_history

    def train_generator(self,
                        corpus_files: CorpusFiles,
                        training_configuration: TrainingConfiguration,
                        display_progress_bars: bool = True):
        """ Trains the neural network using a file training set generator

        :param corpus_files: corpus files configuration
        :param training_configuration: training configuration
        :param display_progress_bars: display progress bars during training
        :return: training history metrics object
        """
        if training_configuration is None:
            raise RuntimeError('No train configuration defined')

        training_set_files = corpus_files.training_set_files
        if training_set_files is None:
            raise RuntimeError('Training Set Generator not defined')

        training_history = None
        validation_strategy = training_configuration.validation_strategy

        if validation_strategy == net.ValidationStrategy.NO_VALIDATION:
            training_history = net.train_network_generator(network=self.model,
                                                           training_generator=training_set_files.directory_iterator,
                                                           training_configuration=training_configuration.configuration,
                                                           verbose=display_progress_bars)

        elif validation_strategy == net.ValidationStrategy.CROSS_VALIDATION:
            validation_files = corpus_files.validation_set_files
            training_history = net.train_network_generator(network=self.model,
                                                           training_generator=training_set_files.directory_iterator,
                                                           training_configuration=training_configuration.configuration,
                                                           validation_generator=validation_files.directory_iterator,
                                                           verbose=display_progress_bars)
        return training_history

    def evaluate(self, test_set: Dataset,
                 display_progress_bars: bool = True):
        """ Evaluate the neural network

        :param test_set: validation set, used to assess performance metrics
        :param display_progress_bars: display progress bars in terminal during evaluation
        :return: test metrics object
        """
        return net.test_network(network=self.model,
                                test_set=test_set,
                                verbose=display_progress_bars)

    def evaluate_generator(self, test_set_generator: DatasetFileIterator,
                           display_progress_bars: bool = True):
        """ Evaluate the neural network using a test set generator

        :param test_set_generator: test set generator
        :param display_progress_bars: display progress bars in terminal during evaluation
        :return: test metrics object
        """
        return net.test_network_generator(network=self.model,
                                          test_set_generator=test_set_generator,
                                          verbose=display_progress_bars)

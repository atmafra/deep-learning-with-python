import os.path

from keras import Sequential, Model

import core.network as net
from core.corpus import CorpusFiles
from core.datasets import Dataset, DatasetFileIterator
from core.training_configuration import TrainingConfiguration
from utils.history_utils import merge_history_metrics
from utils.parameter_utils import get_parameter


class NeuralNetwork():

    def __init__(self, model: Sequential = Sequential(),
                 name: str = None):
        """Instantiates a neural network object that encapsulates the neural network model

        Args:
            model (Sequential): Keras sequential model

        """
        if not isinstance(model, Sequential):
            raise TypeError('Base model must be sequential')

        self.__model: Sequential = model

        if name is not None:
            self.name = name

    @property
    def model(self) -> Sequential:
        return self.__model

    @property
    def name(self) -> str:
        if self.model is not None:
            return self.model.name
        return ''

    @name.setter
    def name(self, name: str):
        if self.model is not None:
            self.model.name = name

    @classmethod
    def from_configurations(cls, name: str,
                            layers_configuration: list):
        """Creates a new neural network from configurations

        Args:
            name (str): neural network name
            layers_configuration (list): neural network architecture

        """
        model = net.create_model(name=name, layer_configuration_list=layers_configuration)
        return NeuralNetwork(model=model)

    @classmethod
    def from_file(cls, path: str,
                  filename: str,
                  verbose: bool = True):
        """Creates a new neural network from JSON architecture file

        Args:
            path (str): file path
            filename (str): file name
            verbose (bool): display load messages in terminal

        """
        filepath = os.path.join(path, filename)
        model = net.create_model_from_file(filepath=filepath, verbose=verbose)
        return NeuralNetwork(model=model)

    def append_layers(self, layers_configuration: list):
        """Appends a list of layers according to their configurations

        Args:
            layers_configuration (list): list of layer configurations to be appended

        """
        if layers_configuration is None:
            raise ValueError('Null model passed, trying to append neural network')

        net.add_layers(self.model, layers_configuration)

    def append_model(self, model: Model):
        """Appends a pre-existing model to the current model in a sequential way

        Args:
            model (Model): model to be appended to the base model

        """
        if model is None:
            raise ValueError('Error trying to append model to the current neural network: Null model passed to append')

        net.append_model(base_model=self.model, model=model)

    def save_archtecture(self, path: str,
                         filename: str,
                         verbose: bool = True):
        """Saves the neural network architecture as a JSON file

        Args:
            path (str): system path of the save directory
            filename (str): architecture file name
            verbose (bool): display save messages on terminal

        """
        net.save_architecture_json(network=self.model, path=path, filename=filename, verbose=verbose)

    def save_weights(self, path: str,
                     filename: str,
                     verbose: bool = True):
        """Saves the neural network weights (status) as a H5 file

        Args:
            path (str): system path of the save directory
            filename (str): weights file name
            verbose (bool): show save messages in terminal

        """
        net.save_weights_h5(network=self.model, path=path, filename=filename, verbose=verbose)

    def save_model(self, path: str,
                   root_filename: str,
                   verbose: bool = True):
        """Saves the neural network architecture and weights

        Args:
            path (str): system path of the save directory
            root_filename (str): root file name (without extension)
            verbose (bool): show save messages in terminal

        """
        architecture_filename = ''
        weights_filename = ''
        if len(root_filename) > 0:
            architecture_filename = root_filename + '.json'
            weights_filename = root_filename + 'h5'

        self.save_archtecture(path=path, filename=architecture_filename, verbose=verbose)
        self.save_weights(path=path, filename=weights_filename, verbose=verbose)

    def train(self, training_set: Dataset,
              training_configuration: TrainingConfiguration,
              validation_set: Dataset = None,
              display_progress_bars: bool = True):
        """Trains the neural network

        Args:
            training_set (Dataset): train set
            training_configuration (TrainingConfiguration): train configuration hyperparameters
            validation_set (Dataset): validation set (optional, depending on validation strategy)
            display_progress_bars (bool): display progress bars in terminal during train

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

    def train_generator(self, corpus_files: CorpusFiles,
                        training_configuration: TrainingConfiguration,
                        display_progress_bars: bool = True):
        """Trains the neural network using a train set generator
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
        """Evaluate the neural network

        Args:
            test_set (Dataset): validation set, used to assess performance metrics
            display_progress_bars (bool): display progress bars in terminal during evaluation

        """
        return net.test_network(network=self.model,
                                test_set=test_set,
                                verbose=display_progress_bars)

    def evaluate_generator(self, test_set_generator: DatasetFileIterator,
                           display_progress_bars: bool = True):
        """Evaluate the neural network using a test set generator

        Args:
            test_set_generator (DatasetFileIterator): test set generator
            display_progress_bars (bool): display progress bars in terminal during evaluation

        """
        return net.test_network_generator(network=self.model,
                                          test_set_generator=test_set_generator,
                                          verbose=display_progress_bars)

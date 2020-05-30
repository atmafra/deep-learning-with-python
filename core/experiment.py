from typing import Type

import numpy as np
from keras.callbacks import History

from core.convolutional_neural_network import ConvolutionalNeuralNetwork
from core.corpus import CorpusType, Corpus, CorpusFiles
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from utils.file_utils import str_to_filename
from utils.history_utils import plot_loss, plot_accuracy, plot_loss_list, plot_accuracy_list, \
    concatenate_history_metrics
from utils.parameter_utils import get_parameter


class Experiment:
    """ An Experiment is a combination of a Neural Network model, plus a Corpus (training and test datasets)
    plus all the hyperparameters that can be tuned to execute the training of that model
    """

    def __init__(self,
                 name: str,
                 neural_network: NeuralNetwork,
                 training_configuration: TrainingConfiguration,
                 id: str = None,
                 fine_tuning_configuration: TrainingConfiguration = None,
                 corpus_type: CorpusType = CorpusType.CORPUS_DATASET,
                 corpus: Corpus = None,
                 corpus_files: CorpusFiles = None):
        """ Creates a new Experiment to evaluate the performance of a specific combination
        of data, model, and training hyperparameters

        :param name: name of the experiment
        :param training_configuration: train hyperparameters
        :param id: experiment ID (code)
        :param fine_tuning_configuration: fine-tuning training stage hyperparameters
        :param corpus_type: defines if data comes from in-memory sets or from directory iterators (generators)
        :param corpus: the train and test sets to be used
        :param corpus_files: corpus files that represent the training, test, and validation sets
        """
        self.__name: str = name
        self.__neural_network = neural_network
        self.__training_configuration = training_configuration
        self.__id: str = id or None
        self.__fine_tuning_configuration = fine_tuning_configuration
        self.__epochs = []
        self.__training_metrics = {}
        self.__training_history = None
        self.__fine_tuning_history = None
        self.__test_results = None

        # Corpus definition
        self.__corpus_type: CorpusType = corpus_type
        self.__corpus = None
        self.__corpus_files = None
        if corpus_type == CorpusType.CORPUS_DATASET:
            if corpus is None:
                raise RuntimeError('No corpus passed to create experiment')
            else:
                self.__corpus = corpus
        elif corpus_type == CorpusType.CORPUS_GENERATOR:
            if corpus_files is None:
                raise RuntimeError('No corpus generator passed to create experiment')
            else:
                self.__corpus_files = corpus_files

        # Sets
        self.__training_set = None
        self.__validation_set = None
        self.__test_set = None
        self.__validation_strategy = ValidationStrategy.NO_VALIDATION

    @property
    def name(self):
        return self.__name

    @property
    def corpus_type(self):
        return self.__corpus_type

    @property
    def corpus(self):
        return self.__corpus

    @property
    def corpus_files(self):
        return self.__corpus_files

    @property
    def neural_network(self):
        return self.__neural_network

    @property
    def training_configuration(self):
        return self.__training_configuration

    @property
    def id(self):
        return self.__id

    @property
    def fine_tuning_configuration(self):
        return self.__fine_tuning_configuration

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs: list):
        self.__epochs = epochs

    @property
    def training_metrics(self):
        return self.__training_metrics

    @training_metrics.setter
    def training_metrics(self, metrics: dict):
        self.__training_metrics = metrics

    @property
    def training_history(self) -> History:
        return self.__training_history

    @training_history.setter
    def training_history(self, history: History):
        self.__training_history = history
        self.epochs = self.training_history.epoch
        self.training_metrics = self.training_history.history

    @property
    def fine_tuning_history(self):
        return self.__fine_tuning_history

    @fine_tuning_history.setter
    def fine_tuning_history(self, history: History):
        self.__fine_tuning_history = history
        self.training_history = concatenate_history_metrics([self.training_history, history])

    @property
    def test_results(self):
        return self.__test_results

    @property
    def test_loss(self):
        return get_parameter(parameters=self.test_results, key='loss', mandatory=False)

    @property
    def test_accuracy(self):
        return get_parameter(parameters=self.test_results, key='accuracy', mandatory=False)

    @property
    def test_mae(self):
        return get_parameter(parameters=self.test_results, key='mae', mandatory=False)

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, training_set):
        self.__training_set = training_set

    @property
    def validation_set(self):
        return self.__validation_set

    @validation_set.setter
    def validation_set(self, validation_set):
        self.__validation_set = validation_set

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, test_set):
        self.__test_set = test_set

    def prepare_sets(self):
        """ Prepare the train and the validation sets for train
        """
        validation_strategy = self.training_configuration.validation_strategy
        validation_set_size = self.training_configuration.validation_set_size

        if validation_strategy in (ValidationStrategy.NO_VALIDATION, ValidationStrategy.K_FOLD_CROSS_VALIDATION):
            self.training_set = self.corpus.training_set
            self.validation_set = None

        elif validation_strategy == ValidationStrategy.CROSS_VALIDATION:
            if validation_set_size > 0:
                if self.corpus.validation_set is None or self.corpus.validation_set.length == 0:
                    split, remain = self.corpus.split_training_set(size=validation_set_size)
                    self.training_set = remain
                    self.validation_set = split
                else:
                    if self.corpus.validation_set.length != validation_set_size:
                        raise RuntimeError('Incompatible validation set sizes')
                    self.training_set = self.corpus.training_set
                    self.validation_set = self.corpus.validation_set

        self.test_set = self.corpus.test_set

    def training(self, display_progress_bars: bool = True):
        """ Train the neural network model

        :param display_progress_bars: display progress bars in terminal during evaluation
        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.training_history = \
                self.neural_network.train(training_set=self.training_set,
                                          training_configuration=self.training_configuration,
                                          validation_set=self.validation_set,
                                          display_progress_bars=display_progress_bars)

        else:
            self.training_history = \
                self.neural_network.train_generator(corpus_files=self.corpus_files,
                                                    training_configuration=self.training_configuration,
                                                    display_progress_bars=display_progress_bars)

    def fine_tuning(self, layer_names: set, display_progress_bars: bool = True):
        """ Executes the fine tuning of a Convolutional Neural Network by training the last layers
        of the convolutional base after the classifier is trained.

        :param layer_names: list of layer names to be unfrozen for fine tuning
        :param display_progress_bars: display progress bars during training
        """
        if not isinstance(self.neural_network, ConvolutionalNeuralNetwork):
            raise TypeError('To run fine tuning, neural network model must be Convolutional')

        convolutional_network: ConvolutionalNeuralNetwork = self.neural_network

        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.fine_tuning_history = \
                convolutional_network.fine_tuning(training_set=self.training_set,
                                                  training_configuration=self.fine_tuning_configuration,
                                                  validation_set=self.validation_set,
                                                  fine_tuning_layers=layer_names,
                                                  display_progress_bars=display_progress_bars)
        else:
            self.fine_tuning_history = \
                convolutional_network.fine_tuning_generator(corpus_files=self.corpus_files,
                                                            training_configuration=self.fine_tuning_configuration,
                                                            fine_tuning_layers=layer_names,
                                                            display_progress_bars=display_progress_bars)

    def evaluation(self, display_progress_bars: bool = True):
        """ Evaluates the neural network performance metrics against the test set

        :param display_progress_bars: display progress bars in terminal during evaluation
        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.__test_results = \
                self.neural_network.evaluate(test_set=self.corpus.test_set,
                                             display_progress_bars=display_progress_bars)

        elif self.corpus_type == CorpusType.CORPUS_GENERATOR:
            self.__test_results = \
                self.neural_network.evaluate_generator(test_set_generator=self.corpus_files.test_set_files,
                                                       display_progress_bars=display_progress_bars)

    def run(self,
            train: bool = True,
            test_after_training: bool = True,
            print_training_results: bool = True,
            fine_tune: bool = False,
            unfreeze_layers: set = {},
            plot_training_loss: bool = False,
            plot_training_accuracy: bool = False,
            plot_fine_tuning_loss: bool = False,
            plot_fine_tuning_accuracy: bool = False,
            training_plot_smooth_factor: float = 0.,
            validation_plot_smooth_factor: float = 0.,
            test: bool = True,
            print_test_results: bool = True,
            save: bool = True,
            model_path: str = None,
            display_progress_bars: bool = True):
        """ Runs the experiment

        :param train: executes the training step of the neural network
        :param test_after_training: evaluates the test dataset after the training stage is complete
        :param print_training_results: print a summary of the test results after training
        :param fine_tune: executes fine tuning on the last convolutional layer after training the classifier
        :param unfreeze_layers: list of layer names to be unfrozen in the convolutional base for fine tuning
        :param plot_training_loss: plot the loss graphic after the training stage
        :param plot_training_accuracy: plot the accuracy graphic after the training stage
        :param plot_fine_tuning_loss: plot the loss graphic after the fine tuning stage
        :param plot_fine_tuning_accuracy: plot the loss graphic after the fine tuning stage
        :param training_plot_smooth_factor: exponential smooth factor to be applied to the training curves
        :param validation_plot_smooth_factor: exponential smooth factor to be applied to the validation curves
        :param test: evaluates the test dataset after the fine tuning stage is complete
        :param print_test_results: print a summary of the results after training and tests
        :param save: save the model (architecture and weights) after training and fine tuning
        :param model_path: system path for the saved model
        :param display_progress_bars: display progress bars during the training process
        """
        if (train or fine_tune or test) and self.corpus_type == CorpusType.CORPUS_DATASET:
            self.prepare_sets()

        if train:
            print("Starting training")
            self.training(display_progress_bars=display_progress_bars)
            print("Training finished successfully")

            if test_after_training:
                print("Starting test after training")
                self.evaluation(display_progress_bars=display_progress_bars)
                print("Test finished successfully")
                if print_training_results:
                    print("Test results after training")
                    self.print_test_results()

        if fine_tune:
            print("Starting fine tuning")
            self.fine_tuning(layer_names=unfreeze_layers,
                             display_progress_bars=display_progress_bars)
            print("Fine tuning finished successfully")

        if save:
            self.save_model(path=model_path,
                            root_filename=str_to_filename(self.name),
                            verbose=True)

        if test:
            print("Starting final tests")
            self.evaluation(display_progress_bars=display_progress_bars)
            print("Final tests finished successfully")
            if print_test_results:
                print("Final test results")
                self.print_test_results()

        # Plotting the results
        if train:
            if plot_training_loss:
                print("Plotting training loss")
                self.plot_training_loss(training_smooth_factor=training_plot_smooth_factor,
                                        validation_smooth_factor=validation_plot_smooth_factor)

            if plot_training_accuracy:
                print("Plotting training accuracy")
                self.plot_training_accuracy(training_smooth_factor=training_plot_smooth_factor,
                                            validation_smooth_factor=validation_plot_smooth_factor)

        if fine_tune:
            if plot_fine_tuning_loss:
                print("Plotting fine tuning training loss")
                self.plot_fine_tuning_loss(training_smooth_factor=training_plot_smooth_factor,
                                           validation_smooth_factor=validation_plot_smooth_factor)

            if plot_fine_tuning_accuracy:
                print("Plotting fine tuning training accuracy")
                self.plot_fine_tuning_accuracy(training_smooth_factor=training_plot_smooth_factor,
                                               validation_smooth_factor=validation_plot_smooth_factor)

    def save_architecture(self, path: str, filename: str = '', verbose: bool = True):
        """ Saves the neural network in its current status to the file path and name

        :param path: system path of the save directory
        :param filename: model file name
        :param verbose: show save messages on terminal
        """
        # save_network_hdf5(network=self.neural_network, path=path, file_name=name)
        self.neural_network.save_architecture(path=path, filename=filename, verbose=verbose)

    def save_weights(self, path: str, filename: str = '', verbose: bool = True):
        """ Saves the model status (weights) to a binary H5 file

        :param path: system path of the save directory
        :param filename: weights file name
        :param verbose: show save message on terminal
        """
        self.neural_network.save_weights(path=path, filename=filename, verbose=verbose)

    def save_model(self, path: str, root_filename: str = '', verbose: bool = True):
        """ Saves the neural network architecture and weights

        :param path: system path of the save directory
        :param root_filename: model file name
        :param verbose: show save messages on terminal
        """
        # save_network_hdf5(network=self.neural_network, path=path, file_name=name)
        self.neural_network.save_model(path=path, root_filename=root_filename, verbose=verbose)

    def plot_training_loss(self,
                           training_smooth_factor: float = 0.,
                           validation_smooth_factor: float = 0.):
        """ Plot the Loss function evolution during the training phase

        :param training_smooth_factor: exponential smooth factor for the training curve
        :param validation_smooth_factor: exponential smooth factor for the validation curve
        """
        plot_loss(history=self.training_history,
                  title='Training Loss',
                  training_smooth_factor=training_smooth_factor,
                  validation_smooth_factor=validation_smooth_factor)

    def plot_fine_tuning_loss(self,
                              training_smooth_factor: float = 0.,
                              validation_smooth_factor: float = 0.):
        """ Plot the Loss function evolution during the fine tuning phase

        :param training_smooth_factor: exponential smooth factor for the training curve
        :param validation_smooth_factor: exponential smooth factor for the validation curve
        """
        plot_loss(history=self.fine_tuning_history,
                  title='Fine Tuning Loss',
                  training_smooth_factor=training_smooth_factor,
                  validation_smooth_factor=validation_smooth_factor)

    def plot_training_accuracy(self,
                               training_smooth_factor: float = 0.,
                               validation_smooth_factor: float = 0.):
        """ Plot the evolution of the accuracy function along the training process

        :param training_smooth_factor: exponential smooth factor for the training curve
        :param validation_smooth_factor: exponential smooth factor for the validation curve
        """
        plot_accuracy(history=self.training_history,
                      title='Training Accuracy',
                      training_smooth_factor=training_smooth_factor,
                      validation_smooth_factor=validation_smooth_factor)

    def plot_fine_tuning_accuracy(self,
                                  training_smooth_factor: float = 0.,
                                  validation_smooth_factor: float = 0.):
        """ Plot the evolution of the accuracy function along the fine tuning process

        :param training_smooth_factor: exponential smooth factor for the training curve
        :param validation_smooth_factor: exponential smooth factor for the validation curve
        """
        plot_accuracy(history=self.fine_tuning_history,
                      title='Fine Tuning Accuracy',
                      training_smooth_factor=0.,
                      validation_smooth_factor=validation_smooth_factor)

    def print_test_results(self):
        """  Print the performance of the model evaluated against the test dataset
        """
        print("\n{}".format(self.name))
        if self.test_loss is not None:
            print("Test loss     = {:.6}".format(self.test_loss))
        else:
            print("Test loss     = undefined")

        if self.test_accuracy is not None:
            print("Test accuracy = {:.2%}".format(self.test_accuracy))
        else:
            print("Test accuracy = undefined")

        if self.test_mae is not None:
            print("Test MAE      = {}".format(self.test_mae))
        else:
            print("Test MAE      = undefined")


class ExperimentPlan:

    def __init__(self, name: str, experiments: list):
        """ Creates a new Experiment Plan from a list of Experiments

        :param name: name of the experiment plan
        :param experiments: list of experiments that take part of the plan
        """
        self.name = name
        self.experiment_list = experiments

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def experiment_list(self):
        return self.__experiment_list

    @experiment_list.setter
    def experiment_list(self, experiment_list: list):
        self.__experiment_list = experiment_list
        self.__experiment_map = {}
        for experiment in self.experiment_list:
            if experiment.id:
                self.__experiment_map[experiment.id] = experiment

    def get_experiment(self, id: str):
        """ Gets an experiment by its name

        :param id: experiment ID
        :return: matching experiment
        """
        if id not in self.__experiment_map:
            raise ValueError('Experiment ID \'{}\' not found in experiment plan'.format(id))

        return self.__experiment_map[id]

    def run(self,
            train: bool = True,
            test: bool = True,
            print_results: bool = False,
            plot_training_loss: bool = False,
            plot_validation_loss: bool = False,
            plot_training_accuracy: bool = False,
            plot_validation_accuracy: bool = False,
            display_progress_bars: bool = True,
            save_models: bool = False,
            models_path: str = None):
        """ Runs all the experiments

        :param train: execute the training phase
        :param test: execute the test phase
        :param print_results: print a summary of the test results after each phase
        :param plot_training_loss: plots a summary of the comparative loss during the training of all models
        :param plot_validation_loss: plots a summary of the comparative loss during the validation of all models
        :param plot_training_accuracy: plots a summary of the comparative accuracy during the training of all models
        :param plot_validation_accuracy: plots a summary of the comparative accuracy during the validation of all models
        :param display_progress_bars: display progress bars during the training process
        :param save_models: save the trained models
        :param models_path: system path to save the trained models
        """
        for experiment in self.experiment_list:
            experiment.run(train=train,
                           test_after_training=False,
                           print_training_results=print_results,
                           fine_tune=False,
                           unfreeze_layers={},
                           plot_training_loss=False,
                           plot_training_accuracy=False,
                           plot_fine_tuning_loss=False,
                           plot_fine_tuning_accuracy=False,
                           test=test,
                           print_test_results=print_results,
                           save=save_models,
                           model_path=models_path,
                           display_progress_bars=display_progress_bars)

        if plot_training_loss:
            self.plot_loss(title='Training Loss',
                           plot_training_series=True,
                           plot_validation_series=False,
                           smooth_factor=0.)

        if plot_validation_loss:
            self.plot_loss(title='Validation Loss',
                           plot_training_series=False,
                           plot_validation_series=True,
                           smooth_factor=0.)

        if plot_training_accuracy:
            self.plot_accuracy(title='Training Accuracy',
                               plot_training_series=True,
                               plot_validation_series=False,
                               smooth_factor=0.)

        if plot_validation_accuracy:
            self.plot_accuracy(title='Validation Accuracy',
                               plot_training_series=False,
                               plot_validation_series=True,
                               smooth_factor=0.)

    def get_history_list(self):
        """ Gets a list of all the training history objects of the experiments
        """
        history_list = []
        for experiment in self.experiment_list:
            if experiment.training_history is not None:
                history_list.append(experiment.training_history)
        return history_list

    def get_labels_list(self):
        """ Gets a list of all the experiment names
        """
        labels_list = []
        for experiment in self.experiment_list:
            if experiment.training_history is not None:
                labels_list.append(experiment.name)
        return labels_list

    def plot_loss(self,
                  title: str = "Training Loss",
                  plot_training_series: bool = True,
                  plot_validation_series: bool = False,
                  smooth_factor: float = 0.):
        """ Plots the evolution of Loss during train

        :param title: plot title
        :param plot_training_series: plot training series
        :param plot_validation_series: plot validation series
        :param smooth_factor: exponential smooth factor
        """
        plot_loss_list(history_metrics_list=self.get_history_list(),
                       labels_list=self.get_labels_list(),
                       title=title,
                       plot_training=plot_training_series,
                       plot_validation=plot_validation_series,
                       smooth_factor=smooth_factor)

    def plot_accuracy(self,
                      title: str = "Training Accuracy",
                      plot_training_series: bool = True,
                      plot_validation_series: bool = False,
                      smooth_factor: float = 0.):
        """ Plots the evolution of Accuracy during train

        :param title: plot title
        :param plot_training_series: plot training series
        :param plot_validation_series: plot validation series
        :param smooth_factor: exponential smooth factor
        """
        plot_accuracy_list(history_metrics_list=self.get_history_list(),
                           labels_list=self.get_labels_list(),
                           title=title,
                           plot_training=plot_training_series,
                           plot_validation=plot_validation_series,
                           smooth_factor=smooth_factor)

    def save_models(self,
                    path: str,
                    filename_list: list = None,
                    save_architecture: bool = True,
                    save_weights: bool = True):
        """ Saves all models architectures to files

        :param path: system path of the save directory
        :param filename_list: list of file names
        :param save_architecture: save architecture of each model
        :param save_weights: save weights of each model
        """
        if not save_architecture and not save_weights:
            return

        has_file_names = filename_list is not None
        if has_file_names and len(filename_list) != len(self.experiment_list):
            raise RuntimeError('File names list has a different size of experiment list')

        architecture_filename = ''
        weights_filename = ''

        if has_file_names:
            root_filename = next(filename_list)
            architecture_filename = root_filename + '.json'
            weights_filename = root_filename + '.h5'

        for experiment in self.experiment_list:
            if not has_file_names:
                architecture_filename = str_to_filename(experiment.name) + '.json'
                weights_filename = str_to_filename(experiment.name) + '.h5'

            if save_architecture:
                experiment.save_architecture(path=path, filename=architecture_filename)

            if save_weights:
                experiment.save_weights(path=path, filename=weights_filename)

            if has_file_names:
                filename = next(filename_list)

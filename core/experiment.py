import utils.history_utils as hutl
from core.convolutional_neural_network import ConvolutionalNeuralNetwork
from core.corpus import CorpusType, Corpus, CorpusFiles
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from utils.file_utils import str_to_filename
from utils.parameter_utils import get_parameter


class Experiment:

    def __init__(self,
                 name: str,
                 neural_network: NeuralNetwork,
                 training_configuration: TrainingConfiguration,
                 fine_tuning_configuration: TrainingConfiguration,
                 corpus_type: CorpusType = CorpusType.CORPUS_DATASET,
                 corpus: Corpus = None,
                 corpus_files: CorpusFiles = None):
        """Creates a new Experiment to evaluate the performance of a specific
           combination of data and train hyperparameters

        Args:
            name (str): name of the experiment
            neural_network (NeuralNetwork): neural network architecture associated to this particular experiment
            training_configuration (TrainingConfiguration): train hyperparameters
            fine_tuning_configuration (TrainingConfiguration): fine-tuning training stage hyperparameters
            corpus_type (CorpusType): defines if data comes from in-memory sets
               or from directory iterators (generators)
            corpus (Corpus): the train and test sets to be used
            corpus_files (CorpusFiles): corpus files that represent the training, test, and validation sets

        """
        self.__neural_network = neural_network
        self.__training_configuration = training_configuration
        self.__fine_tuning_configuration = fine_tuning_configuration
        self.__name: str = name
        self.__training_history = None
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
    def fine_tuning_configuration(self):
        return self.__fine_tuning_configuration

    @property
    def training_history(self):
        return self.__training_history

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
        """Prepare the train and the validation sets for train
        """
        validation_strategy = self.training_configuration.validation_strategy

        if validation_strategy in (ValidationStrategy.NO_VALIDATION, ValidationStrategy.K_FOLD_CROSS_VALIDATION):
            self.training_set = self.corpus.training_set
            self.validation_set = None

        elif validation_strategy == ValidationStrategy.CROSS_VALIDATION:
            if self.corpus.validation_set is None:
                validation_configuration = self.training_configuration.validation_configuration
                validation_set_size = get_parameter(validation_configuration, 'set_size')
                self.validation_set, self.training_set = self.corpus.get_validation_set(validation_set_size)
            else:
                self.training_set = self.corpus.training_set
                self.validation_set = self.corpus.validation_set

        self.test_set = self.corpus.test_set

    def training(self, display_progress_bars: bool = True):
        """Train the neural network model

        Args:
            display_progress_bars (bool): display progress bars in terminal during evaluation

        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.__training_history = \
                self.neural_network.train(training_set=self.training_set,
                                          training_configuration=self.training_configuration,
                                          validation_set=self.validation_set,
                                          display_progress_bars=display_progress_bars)

        else:
            self.__training_history = \
                self.neural_network.train_generator(corpus_files=self.corpus_files,
                                                    training_configuration=self.training_configuration,
                                                    display_progress_bars=display_progress_bars)

    def fine_tuning(self, layer_names: set,
                    display_progress_bars: bool = True):
        """Executes the fine tuning of a Convolutional Neural Network
           by training the last layers of the convolutional base after
           the classifier is traineed

        Args:
            layer_names (list): list of layer names to be unfrozen for fine tuning
            display_progress_bars (bool):
        """
        if not isinstance(self.neural_network, ConvolutionalNeuralNetwork):
            raise TypeError('To run fine tuning, neural network model must be Convolutional')

        self.neural_network.set_convolutional_base_trainable(True)
        self.neural_network.set_convolutional_layer_trainable(layer_names={'block5_conv1'}, trainable=True)
        self.__training_history = \
            self.neural_network.train_generator(corpus_files=self.corpus_files,
                                                training_configuration=self.fine_tuning_configuration,
                                                display_progress_bars=display_progress_bars)

    def evaluation(self, display_progress_bars: bool = True):
        """Evaluate the neural network

        Args:
            display_progress_bars (bool): display progress bars in terminal during evaluation

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
            plot_training_loss: bool = True,
            plot_training_accuracy: bool = True,
            test_after_training: bool = True,
            print_training_results: bool = True,
            fine_tune: bool = False,
            unfreeze_layers: set = {},
            plot_fine_tuning_loss: bool = True,
            plot_fine_tuning_accuracy: bool = True,
            test_after_fine_tuning: bool = True,
            print_fine_tuning_results: bool = True,
            display_progress_bars: bool = True):
        """Runs the experiment

        Args:
            train (bool): executes the training step of the neural network
            plot_training_loss (bool): plot the loss graphic after the training stage
            plot_training_accuracy (bool): plot the accuracy graphic after the training stage
            test_after_training (bool): evaluates the test dataset after the training stage is complete
            print_training_results (bool): print a summary of the test results after training
            fine_tune (bool): executes fine tuning on the last convolutional layer after training the classifier
            unfreeze_layers (list): list of layer names to be unfrozen in the convolutional base for fine tuning
            plot_fine_tuning_loss (bool): plot the loss graphic after the fine tuning stage
            plot_fine_tuning_accuracy (bool): plot the loss graphic after the fine tuning stage
            test_after_fine_tuning (bool): evaluates the test dataset after the fine tuning stage is complete
            print_fine_tuning_results (bool): print a summary of the results after training and tests
            display_progress_bars (bool):

        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.prepare_sets()

        if train:
            print("Starting training")
            self.training(display_progress_bars=display_progress_bars)
            print("Training finished successfully")
            if plot_training_loss:
                print("Plotting training loss")
                self.plot_loss()
            if plot_training_accuracy:
                print("Plotting training accuracy")
                self.plot_accuracy()

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
            if plot_fine_tuning_loss:
                print("Plotting fine tuning training loss")
                self.plot_loss()
            if plot_fine_tuning_accuracy:
                print("Plotting fine tuning training accuracy")
                self.plot_accuracy()

            if test_after_fine_tuning:
                print("Starting test after fine tuning")
                self.evaluation(display_progress_bars=display_progress_bars)
                print("Test after fine tuning finished successfully")
                if print_fine_tuning_results:
                    print("Test results after fine tuning")
                    self.print_test_results()

    def save_architecture(self, path: str, filename: str = '', verbose: bool = True):
        """Saves the neural network in its current status to the file path and name

        Args:
            path (str): system path of the save directory
            filename (str): model file name
            verbose (bool): show save messages on terminal

        """
        # save_network_hdf5(network=self.neural_network, path=path, file_name=name)
        self.neural_network.save_archtecture(path=path, filename=filename, verbose=verbose)

    def save_weights(self, path: str, filename: str = '', verbose: bool = True):
        """Saves the model status (weights) to a binary H5 file

        Args:
            path (str): system path of the save directory
            filename (str): weights file name
            verbose (bool): show save message on terminal

        """
        self.neural_network.save_weights(path=path, filename=filename, verbose=verbose)

    def save_model(self, path: str, root_filename: str = '', verbose: bool = True):
        """Saves the neural network architecture and weights

        Args:
            path (str): system path of the save directory
            root_filename (str): model file name
            verbose (bool): show save messages on terminal

        """
        # save_network_hdf5(network=self.neural_network, path=path, file_name=name)
        self.neural_network.save_model(path=path, root_filename=root_filename, verbose=verbose)

    def plot_loss(self):
        hutl.plot_loss(history=self.training_history, title='Training and Validation Loss')

    def plot_accuracy(self):
        hutl.plot_accuracy(history=self.training_history, title='Training and Validation Accuracy')

    def print_test_results(self):
        """Print the result of the train session
        """
        print("\n{}".format(self.name))
        print("Test loss     = {:.6}".format(self.test_loss))
        if self.test_accuracy is not None:
            print("Test accuracy = {:.2%}".format(self.test_accuracy))
        if self.test_mae is not None:
            print("Test MAE      = {}".format(self.test_mae))


class ExperimentPlan:

    def __init__(self, name: str, experiments: list):
        """Creates a new Experiment Plan from a list of Experiments

        Args:
            name (str): name of the experiment plan
            experiments (list): list of experiments that take part of the plan

        """
        self.__name = name
        self.__experiment_list = experiments

    @property
    def name(self):
        return self.__name

    @property
    def experiment_list(self):
        return self.__experiment_list

    def run(self,
            print_results: bool = False,
            plot_training_loss: bool = False,
            plot_training_accuracy: bool = False,
            display_progress_bars: bool = True):

        """Runs all the experiments
        """
        for experiment in self.__experiment_list:
            experiment.run()

        if plot_training_loss:
            self.plot_loss(title="Training Loss", training=True, validation=False)

        if plot_training_accuracy:
            self.plot_accuracy(title="Training Accuracy", training=True, validation=False)

    def plot_loss(self,
                  title: str = "Training Loss",
                  training: bool = True,
                  validation: bool = False):
        """Plots the evolution of Loss during train
        """
        history_list = []
        labels_list = []

        for experiment in self.experiment_list:
            if experiment.training_history is not None:
                history_list.append(experiment.training_history)
                labels_list.append(experiment.name)

        hutl.plot_loss_list(history_metrics_list=history_list,
                            labels_list=labels_list,
                            title=title,
                            plot_training=training,
                            plot_validation=validation)

    def plot_accuracy(self,
                      title: str = "Training Accuracy",
                      training: bool = True,
                      validation: bool = False):
        """Plots the evolution of Accuracy during train
        """
        history_list = []
        labels_list = []

        for trial in self.__experiment_list:
            history_list.append(trial.training_history)
            labels_list.append(trial.name)

        hutl.plot_accuracy_list(history_metrics_list=history_list,
                                labels_list=labels_list,
                                title=title,
                                plot_training=training,
                                plot_validation=validation)

    def save_models(self, path: str,
                    filename_list: list = None,
                    save_architecture: bool = True,
                    save_weights: bool = True):
        """Saves all models architectures to files

        Args:
            path (str): system path of the save directory
            filename_list (str): list of file names
            save_architecture (bool): save architecture of each model
            save_weights (bool): save weights of each model

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

import utils.history_utils as hutl
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
                 corpus_type: CorpusType = CorpusType.CORPUS_DATASET,
                 corpus: Corpus = None,
                 corpus_generator: CorpusFiles = None):
        """Creates a new Experiment to evaluate the performance of a specific
           combination of data and training hyperparameters

        Args:
            name (str): name of the experiment
            neural_network (NeuralNetwork): neural network architecture associated to this particular experiment
            training_configuration (TrainingConfiguration): training hyperparameters
            corpus_type (CorpusType): defines if data comes from in-memory sets
               or from directory iterators (generators)
            corpus (Corpus): the training and test sets to be used
            corpus_generator (CorpusFiles): corpus generator

        """
        self.__neural_network = neural_network
        self.__training_configuration = training_configuration
        self.__name: str = name
        self.__training_history = None
        self.__test_results = None

        # Corpus definition
        self.__corpus_type: CorpusType = corpus_type
        self.__corpus = None
        self.__corpus_generator = None
        if corpus_type == CorpusType.CORPUS_DATASET:
            if corpus is None:
                raise RuntimeError('No corpus passed to create experiment')
            else:
                self.__corpus = corpus
        elif corpus_type == CorpusType.CORPUS_GENERATOR:
            if corpus_generator is None:
                raise RuntimeError('No corpus generator passed to create experiment')
            else:
                self.__corpus_generator = corpus_generator

        # Sets
        self.__training_set = None
        self.__validation_strategy = ValidationStrategy.NO_VALIDATION
        self.__validation_set = None
        self.__test_set = None

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
    def corpus_generator(self):
        return self.__corpus_generator

    @property
    def neural_network(self):
        return self.__neural_network

    @property
    def training_configuration(self):
        return self.__training_configuration

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

    @property
    def validation_set(self):
        return self.__validation_set

    @property
    def test_set(self):
        return self.__test_set

    def prepare_sets(self):
        """Prepare the training and the validation sets for training
        """
        validation_strategy = self.training_configuration.validation_strategy

        if validation_strategy in (ValidationStrategy.NO_VALIDATION, ValidationStrategy.K_FOLD_CROSS_VALIDATION):
            self.__training_set = self.corpus.training_set
            self.__validation_set = None

        elif validation_strategy == ValidationStrategy.CROSS_VALIDATION:
            validation_configuration = self.training_configuration.validation_configuration
            validation_set_size = get_parameter(validation_configuration, 'set_size')
            self.__validation_set, self.__training_set = self.corpus.get_validation_set(validation_set_size)

        self.__test_set = self.corpus.test_set

    def train(self, display_progress_bars: bool = True):
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
                self.neural_network.train_generator(corpus_generator=self.corpus_generator,
                                                    training_configuration=self.training_configuration,
                                                    display_progress_bars=display_progress_bars)

    def evaluate(self, display_progress_bars: bool = True):
        """Evaluate the neural network

        Args:
            display_progress_bars (bool): display progress bars in terminal during evaluation

        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.__test_results = self.neural_network.evaluate(
                test_set=self.corpus.test_set,
                display_progress_bars=display_progress_bars)

        elif self.corpus_type == CorpusType.CORPUS_GENERATOR:
            self.__test_results = self.neural_network.evaluate_generator(
                test_set_generator=self.corpus_generator.test_set_files,
                display_progress_bars=display_progress_bars)

    def run(self, print_results: bool = True,
            plot_history: bool = False,
            display_progress_bars: bool = True):
        """Runs the experiment

        Args:
            print_results (bool):
            plot_history (bool):
            display_progress_bars (bool):

        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.prepare_sets()

        self.train(display_progress_bars=display_progress_bars)
        self.evaluate(display_progress_bars=display_progress_bars)

        if print_results:
            self.print_test_results()

        if plot_history:
            self.plot_loss()

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
        """Print the result of the training session
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
        """Plots the evolution of Loss during training
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
        """Plots the evolution of Accuracy during training
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

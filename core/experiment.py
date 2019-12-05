from keras.callbacks import History

import utils.history_utils as hutl
from core.network import *
from core.sets import Corpus, CorpusGenerator


class CorpusType(Enum):
    CORPUS_DATASET = 1
    CORPUS_GENERATOR = 2


class Experiment:

    def __init__(self,
                 name: str,
                 layers_configuration: list,
                 training_configuration: dict,
                 corpus_type: CorpusType = CorpusType.CORPUS_DATASET,
                 corpus: Corpus = None,
                 corpus_generator: CorpusGenerator = None):
        """Creates a new Experiment to evaluate the performance of a specific
           combination of data and training hyperparameters

        Args:
            name (str): name of the
            layers_configuration (list): list of layer configurations that represent the network
            training_configuration (dict): training configuration parameters
            corpus_type (CorpusType): defines if data comes from in-memory sets
               or from directory iterators (generators)
            corpus (Corpus): the training and test sets to be used
            corpus_generator (CorpusGenerator): corpus generator

        """
        self.__name: str = name
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
        self.__layers_configuration: list = layers_configuration
        self.__training_configuration: dict = training_configuration
        self.__training_set = None
        self.__validation_strategy = ValidationStrategy.NO_VALIDATION
        self.__validation_set = None
        self.__test_set = None
        self.__neural_network = None
        self.__history: History = None
        self.__test_loss = None
        self.__test_accuracy = None
        self.__test_mae = None

    @property
    def name(self):
        return self.__name

    @property
    def corpus(self):
        return self.__corpus

    @property
    def corpus_generator(self):
        return self.__corpus_generator

    @property
    def corpus_type(self):
        return self.__corpus_type

    @property
    def neural_network(self):
        return self.__neural_network

    @property
    def training_configuration(self):
        return self.__training_configuration

    @property
    def layers_configuration(self):
        return self.__layers_configuration

    @property
    def history(self):
        return self.__history

    @history.setter
    def history(self, history: History):
        self.__history = history

    @property
    def test_loss(self):
        return self.__test_loss

    @property
    def test_accuracy(self):
        return self.__test_accuracy

    @property
    def test_mae(self):
        return self.__test_mae

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
        validation = putl.get_parameter(self.training_configuration, 'validation')
        validation_strategy = putl.get_parameter(validation, 'strategy')

        if validation_strategy in \
                (ValidationStrategy.NO_VALIDATION, ValidationStrategy.K_FOLD_CROSS_VALIDATION):
            self.__training_set = self.corpus.training_set
            self.__validation_set = None

        elif validation_strategy == ValidationStrategy.CROSS_VALIDATION:
            validation_set_size = putl.get_parameter(validation, 'set_size')
            self.__validation_set, self.__training_set = self.corpus.get_validation_set(validation_set_size)

        self.__test_set = self.corpus.test_set

    def create_network(self):
        """Creates the neural network according to the layers configuration list
        """
        self.__neural_network = create_network(layer_configuration_list=self.layers_configuration)

    def train(self, display_progress_bars: bool = True):
        """Trains the neural network
        """
        validation = putl.get_parameter(self.training_configuration, 'validation')
        strategy = putl.get_parameter(validation, 'strategy')

        if strategy == ValidationStrategy.NO_VALIDATION:
            self.__history = train_network(network=self.neural_network,
                                           training_configuration=self.training_configuration,
                                           training_set=self.training_set,
                                           validation_set=None,
                                           verbose=display_progress_bars)

        elif strategy == ValidationStrategy.CROSS_VALIDATION:
            self.__history = train_network(network=self.neural_network,
                                           training_configuration=self.training_configuration,
                                           training_set=self.training_set,
                                           validation_set=self.validation_set,
                                           verbose=display_progress_bars)

        elif strategy == ValidationStrategy.K_FOLD_CROSS_VALIDATION:
            k = putl.get_parameter(validation, 'k')
            shuffle = putl.get_parameter(validation, 'shuffle')
            all_histories = train_network_k_fold(network=self.neural_network,
                                                 training_configuration=self.training_configuration,
                                                 training_set=self.training_set,
                                                 k=k, shuffle=shuffle,
                                                 verbose=display_progress_bars)
            self.__history = hutl.merge_history_metrics(all_histories)

    def train_generator(self, display_progress_bars: bool = True):
        """Trains the neural network using a training set generator
        """
        if self.corpus_type != CorpusType.CORPUS_GENERATOR:
            raise RuntimeError('Corpus type is not from Generator type')

        if self.corpus_generator is None:
            raise RuntimeError('Training Generator not defined')

        training_set_generator = self.corpus_generator.training_set_generator
        if training_set_generator is None:
            raise RuntimeError('Training Set Generator not defined')

        validation = putl.get_parameter(self.training_configuration, 'validation')
        strategy = putl.get_parameter(validation, 'strategy')

        if strategy == ValidationStrategy.NO_VALIDATION:
            self.__history = train_network_generator(network=self.neural_network,
                                                     training_generator=training_set_generator.generator,
                                                     training_configuration=self.training_configuration,
                                                     verbose=display_progress_bars)

        elif strategy == ValidationStrategy.CROSS_VALIDATION:
            validation_generator = self.corpus_generator.validation_set_generator
            self.__history = train_network_generator(network=self.neural_network,
                                                     training_generator=training_set_generator.generator,
                                                     training_configuration=self.training_configuration,
                                                     validation_generator=validation_generator.generator,
                                                     verbose=display_progress_bars)

        # elif strategy == ValidationStrategy.K_FOLD_CROSS_VALIDATION:
        #     k = putl.get_parameter(validation, 'k')
        #     shuffle = putl.get_parameter(validation, 'shuffle')
        #     all_histories = train_network_k_fold(network=self.__neural_network,
        #                                          training_set=self.__training_set,
        #                                          training_configuration=self.__training_configuration,
        #                                          k=k, shuffle=shuffle,
        #                                          verbose=display_progress_bars)
        #     self.__history = hutl.merge_history_metrics(all_histories)

    def evaluate(self, display_progress_bars: bool = True):
        """Evaluate the neural network
        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            test_results = test_network(network=self.neural_network,
                                        test_set=self.test_set,
                                        verbose=display_progress_bars)

        elif self.corpus_type == CorpusType.CORPUS_GENERATOR:
            test_results = test_network_generator(network=self.neural_network,
                                                  test_set_generator=self.corpus_generator.test_set_generator,
                                                  verbose=display_progress_bars)

        self.__test_loss = putl.get_parameter(parameters=test_results, key='loss', mandatory=True)
        self.__test_accuracy = putl.get_parameter(parameters=test_results, key='accuracy', mandatory=False)
        self.__test_mae = putl.get_parameter(parameters=test_results, key='mae', mandatory=False)

    def plot_loss(self):
        hutl.plot_loss(history=self.history, title='Training and Validation Loss')

    def plot_accuracy(self):
        hutl.plot_accuracy(history=self.history, title='Training and Validation Accuracy')

    def print_test_results(self):
        """Print the result of the training session
        """
        print("\n{}".format(self.name))
        print("Test loss     = {:.6}".format(self.test_loss))
        if self.__test_accuracy is not None:
            print("Test accuracy = {:.2%}".format(self.__test_accuracy))
        if self.__test_mae is not None:
            print("Test MAE      = {}".format(self.__test_mae))

    def run(self,
            print_results: bool = True,
            plot_history: bool = False,
            display_progress_bars: bool = True):
        """Runs the experiment
        """
        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.prepare_sets()
        self.create_network()

        if self.corpus_type == CorpusType.CORPUS_DATASET:
            self.train(display_progress_bars=display_progress_bars)
        else:
            self.train_generator(display_progress_bars=display_progress_bars)

        self.evaluate()

        if print_results:
            self.print_test_results()

        if plot_history:
            self.plot_loss()


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

        for trial in self.__experiment_list:
            history_list.append(trial.history)
            labels_list.append(trial.name)

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
            history_list.append(trial.history)
            labels_list.append(trial.name)

        hutl.plot_accuracy_list(history_metrics_list=history_list,
                                labels_list=labels_list,
                                title=title,
                                plot_training=training,
                                plot_validation=validation)

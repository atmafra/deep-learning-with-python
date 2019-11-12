import core.network as net
import utils.history_utils as hutl
from core.sets import Corpus
from utils.history_utils import plot_loss, plot_accuracy


class Trial:

    def __init__(self,
                 name: str,
                 corpus: Corpus,
                 layers_configuration_list: list,
                 training_configuration: dict):
        """Creates a new Trial to evaluate the performance of a specific
           combination of data and training hyperparameters

        Args:
            corpus (Corpus): the training and test sets to be used
            layers_configuration_list (list): list of layer configurations that represent the network
            training_configuration (dict): training configuration parameters

        """
        self.__name: str = name
        self.__corpus: Corpus = corpus
        self.__layers_configuration_list: list = layers_configuration_list
        self.__training_configuration: dict = training_configuration
        self.__training_set = None
        self.__validation_set = None
        self.__neural_network = None
        self.__history = None
        self.__test_loss = None
        self.__test_accuracy = None

    @property
    def name(self):
        return self.__name

    @property
    def history(self):
        return self.__history

    @property
    def test_loss(self):
        return self.__test_loss

    @property
    def test_accuracy(self):
        return self.__test_accuracy

    def prepare_sets(self):
        """Prepare the training and the validation sets for training
        """
        validation_set_size = net.get_parameter(self.__training_configuration, 'validation_set_size')
        self.__validation_set, self.__training_set = self.__corpus.get_validation_set(validation_set_size)

    def create_network(self):
        """Creates the neural network according to the layers configuration list
        """
        self.__neural_network = net.create_network(layer_configuration_list=self.__layers_configuration_list)

    def train(self):
        self.__history = net.train_network(network=self.__neural_network,
                                           training_configuration=self.__training_configuration,
                                           training_set=self.__training_set,
                                           validation_set=self.__validation_set)

    def evaluate(self):
        """Evaluate the neural network
        """
        (self.__test_loss, self.__test_accuracy) = \
            net.test_network(self.__neural_network, self.__corpus.test_set)

    def plot_loss(self):
        plot_loss(history=self.__history)

    def plot_accuracy(self):
        plot_accuracy(history=self.__history)

    def print_result(self):
        print("\n{}".format(self.__name))
        print("loss     = {:.6}".format(self.__test_loss))
        print("accuracy = {:.2%}".format(self.__test_accuracy))

    def run(self, print_results: bool = True, plot_history: bool = False):
        """Runs the experiment
        """
        self.prepare_sets()
        self.create_network()
        self.train()
        self.evaluate()

        if print_results:
            self.print_result()

        if plot_history:
            self.plot_loss()
            self.plot_accuracy()

        return self.__test_loss, self.__test_accuracy, self.__history


class Experiment:

    def __init__(self, name: str, trials: list):
        """Creates a new Experiment from a list of trials

        Args:
            name (str): name of the experiment
            trials (dict): list of Trial objects that are part of the Experiment

        """
        self.__name = name
        self.__trial_list = trials

    @property
    def name(self):
        return self.__name

    def run(self, print_results: bool = False,
            plot_training_loss: bool = False,
            plot_training_accuracy: bool = False):

        """Runs all the experiments
        """
        for trial in self.__trial_list:
            trial.run()

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

        for trial in self.__trial_list:
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

        for trial in self.__trial_list:
            history_list.append(trial.history)
            labels_list.append(trial.name)

        hutl.plot_accuracy_list(history_metrics_list=history_list,
                                labels_list=labels_list,
                                title=title,
                                plot_training=training,
                                plot_validation=validation)

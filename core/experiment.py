from keras import Model
from keras.callbacks import History

from core.sets import Corpus
import core.network as net
from utils.history_utils import plot_loss_list, plot_loss, plot_accuracy


class Experiment:

    def __init__(self,
                 corpus: Corpus,
                 layers_configuration_list: list,
                 training_configuration: dict):
        """Creates a new Experiment
        """
        self._corpus: Corpus = corpus
        self._layers_configuration_list: list = layers_configuration_list
        self._training_configuration: dict = training_configuration
        self._training_set = None
        self._validation_set = None
        self._neural_network = None
        self._history = None
        self._test_loss = None
        self._test_accuracy = None

    def prepare_sets(self):
        """Prepare the training and the validation sets for training
        """
        validation_set_size = net.get_parameter(self._training_configuration, 'validation_set_size')
        self._training_set, self._validation_set = self._corpus.get_validation_set(validation_set_size)

    def create_network(self):
        self._neural_network = net.create_network(layer_configuration_list=self._layers_configuration_list)

    def train(self):
        self._history = net.train_network(network=self._neural_network,
                                          training_configuration=self._training_configuration,
                                          training_set=self._training_set,
                                          validation_set=self._validation_set)

    def evaluate(self):
        # evaluate the neural network
        (self._test_loss, self._test_accuracy) = net.test_network(self._neural_network, self._corpus.test_set)

    def plot_loss(self):
        plot_loss(history=self._history)

    def plot_accuracy(self):
        plot_accuracy(history=self._history)

    def print_result(self, title='RESULTS'):
        print("\nMedium Network with Dropout")
        print("loss     =", self._test_loss)
        print("accuracy = {:.2%}".format(self._test_accuracy))

    def run(self):
        """Runs the experiment
        """
        self.prepare_sets()
        self.create_network()
        self.train()
        self.evaluate()
        self.print_result()
        self.plot_loss()
        self.plot_accuracy()

        return self._test_loss, self._test_accuracy, self._history

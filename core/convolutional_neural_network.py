from keras import Model, Sequential, layers

import core.network as net
from core.corpus import CorpusFiles
from core.datasets import Dataset
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from utils.history_utils import concatenate_history_metrics


class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self,
                 name: str,
                 convolutional_base: Model,
                 classifier: Model):
        """ Instantiates a new ConvolutionalNeuralNetwork, by concatenating a convolutional base model
        with a classifier in a Sequential way

        :param name: neural network name
        :param convolutional_base: convolutional base model
        :param classifier: classifier model
        """
        if convolutional_base is None:
            raise ValueError('Null convolutional base creating Convolutional Neural Network')

        if classifier is None:
            raise ValueError('Null classifier network creationg Convolutional Neural Network')

        model = Sequential()
        net.append_model(model, convolutional_base)
        net.append_model(model, layers.Flatten())
        net.append_model(model, classifier)
        super().__init__(model=model, name=name)
        self.__convolutional_base = convolutional_base
        self.__classifier = classifier
        self.set_convolutional_base_trainable(trainable=False, cascade_layers=True)

    @property
    def convolutional_base(self) -> Model:
        return self.__convolutional_base

    @property
    def classifier(self) -> Model:
        return self.__classifier

    def set_convolutional_base_trainable(self, trainable: bool, cascade_layers: bool):
        """ Defines the value of the trainable property for the entire convolutional base of the network

        :param trainable: True for trainable, False for non-trainable
        :param cascade_layers: cascade update to all the layers in the convolutional base model
        """
        self.convolutional_base.trainable = trainable
        if cascade_layers:
            for layer in self.convolutional_base.layers:
                layer.trainable = trainable

    def set_convolutional_layers_trainable(self, layer_names: set, trainable: bool):
        """ Defines the value of the trainable property for a specific set of layers
        in the convolutional base of the network

        :param layer_names: list of layer names
        :param trainable: True for trainable, False for non-trainable
        """
        for layer in self.convolutional_base.layers:
            if layer.name in layer_names:
                layer.trainable = trainable

    def fine_tuning(self,
                    training_set: Dataset,
                    training_configuration: TrainingConfiguration,
                    fine_tuning_layers: set,
                    validation_set: Dataset = None,
                    display_progress_bars: bool = True):
        """ Execute the fine tuning of the Convolutional Neural Network.

        :param training_set: training set
        :param training_configuration: training configuration hyperparameters
        :param validation_set: validation set
        :param fine_tuning_layers: list of layers to unfreeze for fine tuning
        :param display_progress_bars: display progress bars during fine tuning
        """
        if fine_tuning_layers is None:
            fine_tuning_layers = []
        self.set_convolutional_base_trainable(trainable=False, cascade_layers=True)
        self.set_convolutional_base_trainable(trainable=True, cascade_layers=False)
        self.set_layers_trainable(layer_names=fine_tuning_layers, trainable=True)

        return NeuralNetwork.train(self,
                                   training_set=training_set,
                                   training_configuration=training_configuration,
                                   validation_set=validation_set,
                                   display_progress_bars=display_progress_bars)

    def fine_tuning_generator(self,
                              corpus_files: CorpusFiles,
                              training_configuration: TrainingConfiguration,
                              fine_tuning_layers: set,
                              display_progress_bars: bool = True):
        """ Execute the fine tuning of the Convolutional Neural Network using generators as datasets.

        :param corpus_files: corpus files configuration
        :param training_configuration: training configuration
        :param fine_tuning_layers: list of layers to unfreeze for fine tuning
        :param display_progress_bars: display progress bars during training
        :return: training history metrics object
        """
        fine_tuning_history = None
        if fine_tuning_layers is None:
            fine_tuning_layers = []
        self.set_convolutional_base_trainable(trainable=True, cascade_layers=False)
        self.set_layers_trainable(layer_names=fine_tuning_layers, trainable=True)

        return NeuralNetwork.train_generator(self,
                                             corpus_files=corpus_files,
                                             training_configuration=training_configuration,
                                             display_progress_bars=display_progress_bars)

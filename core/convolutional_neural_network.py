from keras import Model, Sequential, layers

import core.network as net
from core.neural_network import NeuralNetwork


class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self,
                 name: str,
                 convolutional_base: Model,
                 classifier: Model):
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
        self.set_convolutional_base_trainable(False)

    @property
    def convolutional_base(self) -> Model:
        return self.__convolutional_base

    @property
    def classifier(self) -> Model:
        return self.__classifier

    def set_convolutional_base_trainable(self, trainable: bool):
        """Defines the value of the trainable property for the entire convolutional base of the network
        """
        self.convolutional_base.trainable = trainable

    def set_convolutional_layer_trainable(self, layer_names: set, trainable: bool):
        """Sets the trainable property for a list of specific layers in the network
        """
        for layer in self.convolutional_base.layers:
            layer.trainable = False
            if layer.name in layer_names:
                layer.trainable = trainable
            # print('Layer {} trainable: {}'.format(layer.name, layer.trainable))

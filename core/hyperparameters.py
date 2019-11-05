import json
from enum import Enum


class NetworkOutputType(Enum):
    BOOLEAN = 1
    CATEGORICAL = 2
    DECIMAL = 3

    class EnumEncoder(json.JSONEncoder):
        def default(self, obj):
            if type(obj) in NetworkOutputType.values():
                return {"__enum__": str(obj)}
            return json.JSONEncoder.default(self, obj)


class LayerPosition(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class LayerType(Enum):
    DENSE = 1
    DROPOUT = 2


class KernelRegularizer(Enum):
    NONE = 1
    L1 = 2
    L1L2 = 3


default_learning_rate: float = 0.001
default_optimizer: str = 'rmsprop'
default_loss: dict = {
    NetworkOutputType.BOOLEAN: 'binary_crossentropy',
    NetworkOutputType.CATEGORICAL: 'categorical_crossentropy',
    NetworkOutputType.DECIMAL: 'mean_squared_error'
}


class LayerHyperparameters:
    """Layer hyper parameters
    """

    def __init__(self,
                 layer_type: LayerType,
                 units: int,
                 position: LayerPosition,
                 activation: str,
                 **kwargs):
        """Creates a new layer hyper parameters definition object

        Args:
            layer_type (LayerType): Layer Type (dense, dropout,...)
            units (int): number of layer units (neurons)
            position (LayerPosition): layer position
            activation (str): type of activation function
            parameters (dict): additional parameters
            **kwargs: additional configurations

        """
        self.layer_type = layer_type
        self.units = units
        self.position = position
        self.activation = activation
        self.kwargs = kwargs

    def to_dict(self):
        """Returns the layer hyperparameters as a dictionary
        """
        layer_dict = {'layer_type': self.layer_type.name,
                      'units': self.units,
                      'position': self.position.name,
                      'activation': self.activation}

        return layer_dict

    def to_json(self):
        return json.dumps(self.to_dict())


class NetworkHyperparameters:

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 output_type: NetworkOutputType,
                 layer_hyperparameters_list: list,
                 optimizer: str = default_optimizer,
                 learning_rate: float = default_learning_rate,
                 loss: str = None, metrics: list = None):

        """ Parameterized constructor
        """
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.loss = None
        if loss is None:
            self.loss = default_loss[output_type]
        else:
            self.loss = loss
        self.layer_hyperparameters_list = layer_hyperparameters_list
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics

    def to_dict(self):
        network_dict = {'input_size': self.input_size,
                        'output_size': self.output_size,
                        # 'output_type': self.output_type,
                        'optimizer': self.optimizer,
                        'learning_rate': self.learning_rate}
        if self.loss is not None:
            network_dict['loss'] = self.loss

        return network_dict

    def to_json(self):
        return json.dumps(self.to_dict())

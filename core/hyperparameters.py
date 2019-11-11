import json

from core.network import NetworkOutputType


class NetworkHyperparameters:

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 output_type: NetworkOutputType,
                 layer_hyperparameters_list: list,
                 optimizer: str = None,
                 learning_rate: float = None,
                 loss: str = None,
                 metrics: list = None):

        """ Parameterized constructor
        """
        self.input_size = input_size
        self.output_size = output_size
        self.output_type = output_type
        self.loss = None
        if loss is None:
            self.loss = None #default_loss[output_type]
        else:
            self.loss = loss
        self.layer_hyperparameters_list = layer_hyperparameters_list
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics

    def to_dict(self):
        network_dict = {'input_size': self.input_size,
                        'output_size': self.output_size,
                        'optimizer': self.optimizer,
                        'learning_rate': self.learning_rate}
        if self.loss is not None:
            network_dict['loss'] = self.loss

        return network_dict

    def to_json(self):
        return json.dumps(self.to_dict())

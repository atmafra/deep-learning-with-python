from utils.parameter_utils import get_parameter


class TrainingConfiguration:

    def __init__(self, configuration: dict):
        self.__configuration = configuration

    @property
    def configuration(self):
        return self.__configuration

    @property
    def validation_configuration(self):
        return get_parameter(parameters=self.__configuration,
                             key='validation',
                             mandatory=True)

    @property
    def validation_strategy(self):
        return get_parameter(parameters=self.validation_configuration,
                             key='strategy',
                             mandatory=True)

    @property
    def validation_set_size(self):
        return get_parameter(parameters=self.validation_configuration,
                             key='set_size',
                             mandatory=False)

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
                             mandatory=False)

    @property
    def validation_strategy(self):
        validation_configuration = self.validation_configuration
        return get_parameter(parameters=validation_configuration,
                             key='strategy', mandatory=False)

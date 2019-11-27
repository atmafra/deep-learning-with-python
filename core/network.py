from enum import Enum

from keras import models, layers, Model

from core.sets import Set


class NetworkOutputType(Enum):
    BOOLEAN = 1
    CATEGORICAL = 2
    DECIMAL = 3


class LayerPosition(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class LayerType(Enum):
    INPUT = 1
    DENSE = 2
    DROPOUT = 3
    CONV_2D = 4
    MAX_POOLING_2D = 5
    FLATTEN = 6
    OUTPUT = 7


class ValidationStrategy(Enum):
    NO_VALIDATION = 1
    CROSS_VALIDATION = 2
    K_FOLD_CROSS_VALIDATION = 3


default_optimizer: str = 'rmsprop'
default_learning_rate: float = 0.001
default_loss: dict = {
    NetworkOutputType.BOOLEAN: 'binary_crossentropy',
    NetworkOutputType.CATEGORICAL: 'categorical_crossentropy',
    NetworkOutputType.DECIMAL: 'mean_squared_error'}


def get_parameters(parameters: dict, parameter_list: list, delete_parameter: bool = False) -> dict:
    """Get the list of parameter values by its keys

    Args:
        parameters: parameters dictionary
        parameter_list: list of parameters to be retrieved
        delete_parameter: delete parameter from parameters dictionary after retrieval

    """
    parameter_values = {}
    for parameter in parameter_list:
        if parameter not in parameters:
            parameter_values[parameter] = None
            continue

        parameter_values[parameter] = parameters.get(parameter)
        if delete_parameter:
            del parameters[parameter]

    return parameter_values


def get_parameter(parameters: dict, parameter: str, mandatory: bool = True):
    """Gets a single parameter from a parameters dictionary

    Args:
        parameters: parameters dictionary
        parameter: parameter key to be retrieved
        mandatory: raise an error if parameter is not found

    """
    if parameter not in parameters:
        if mandatory:
            raise RuntimeError('parameter {} is not in parameters list'.format(parameter))
        else:
            return None

    return parameters.get(parameter)


def extract_parameters(parameters: dict, parameter_list: list) -> dict:
    """Extracts a list of parameter values from a parameters dictionary
       deleting from the original parameter dictionary

    Args:
        parameters (dict): parameter dictionary
        parameter_list (list): parameter key list to be extracted

    """
    return get_parameters(parameters, parameter_list, delete_parameter=True)


def extract_parameter(parameters: dict, parameter: str):
    """Extracts a particular parameter from a parameters dictionary
       deleting the parameter from the dictionary

    Args:
        parameters: parameter dictionary
        parameter: parameter key to be extracted

    """
    parameter_values = extract_parameters(parameters, [parameter])
    return parameter_values[parameter]


def create_layer(parameters: dict):
    """Creates a layer according to the hyperparameters

    Args:
        parameters (dict): hyperparameters dictionary

    """
    if 'layer_type' not in parameters:
        raise ValueError('layer_type not defined')

    layer_type = extract_parameter(parameters, 'layer_type')

    # Input
    if layer_type == LayerType.INPUT:
        pass

    # Dense
    elif layer_type == LayerType.DENSE:
        return layers.Dense(**parameters)

    # Dropout
    elif layer_type == LayerType.DROPOUT:
        rate = extract_parameter(parameters, 'rate')
        return layers.Dropout(rate=rate, **parameters)

    # Convolutional 2D
    elif layer_type == LayerType.CONV_2D:
        filters = extract_parameter(parameters, 'filters')
        kernel_size = extract_parameter(parameters, 'kernel_size')
        return layers.Conv2D(filters=filters, kernel_size=kernel_size)

    # Max Pooling 2D
    elif layer_type == LayerType.MAX_POOLING_2D:
        pool_size = extract_parameter(parameters, 'pool_size')
        return layers.MaxPooling2D(pool_size=pool_size)

    # Flatten
    elif layer_type == LayerType.FLATTEN:
        return layers.Flatten()

    # Output
    elif layer_type == LayerType.OUTPUT:
        pass

    # Unknown
    else:
        raise NotImplementedError('Unknown layer type')


def create_network(layer_configuration_list: list):
    """Creates a neural network according to its hyper parameters

    Args:
        layer_configuration_list (list): list of layer hyperparameters

    """
    network = models.Sequential()

    # layers
    for layer_configuration in layer_configuration_list:
        layer = create_layer(layer_configuration)
        if layer is not None:
            network.add(layer)

    return network


def train_network(network: Model,
                  training_configuration: dict,
                  training_set: Set,
                  validation_set: Set = None,
                  verbose: bool = True):
    """Train the neural network, returning the evolution of the training metrics

    Args:
        network (Model): neural network model to be trained
        training_configuration (dict): training algorithm parameters
        training_set (Set): training set
        validation_set (Set): validation set
        verbose (bool): display training progress bars if True

    """
    shuffle = extract_parameter(training_configuration, 'shuffle')
    validation = get_parameter(training_configuration, 'validation')
    validation_strategy = get_parameter(validation, 'strategy')

    if shuffle:
        working_training_set = training_set.copy()
        working_training_set.shuffle()
    else:
        working_training_set = training_set

    validation_data = None
    if validation_set is not None:
        validation_data = validation_set.to_datasets()

    keras_parameters = get_parameter(training_configuration, 'keras')
    compile_parameters = get_parameter(keras_parameters, 'compile')
    network.compile(**compile_parameters)

    fit_parameters = get_parameter(keras_parameters, 'fit')
    history = network.fit(x=working_training_set.input_data,
                          y=working_training_set.output_data,
                          validation_data=validation_data,
                          verbose=verbose,
                          **fit_parameters)

    return history


def train_network_k_fold(network: Model,
                         training_configuration: dict,
                         training_set: Set,
                         k: int,
                         shuffle: bool,
                         verbose: bool = True):
    """Train the neural network model using k-fold cross-validation

    Args:
        network (Model): neural network model to be trained
        training_configuration (dict): training parameters
        training_set (Set): training data set
        k (int): number of partitions in k-fold cross-validation
        shuffle (bool): shuffle the training set before k splitting
        verbose (bool): display training progress bars if True

    """
    all_histories = []
    working_training_set = training_set.copy()

    if shuffle:
        working_training_set.shuffle()

    for fold in range(k):
        print('processing fold #', fold)
        validation_set, training_set_remaining = working_training_set.split_k_fold(fold, k)

        # build the model and train the current fold
        network.build()

        fold_history = train_network(network=network,
                                     training_set=training_set_remaining,
                                     validation_set=validation_set,
                                     training_configuration=training_configuration,
                                     verbose=verbose)

        all_histories.append(fold_history)

    return all_histories


def test_network(network: Model,
                 test_set: Set,
                 verbose: bool = True) -> dict:
    """Evaluates all the test inputs according to the current network

    Args:
        network (Model): neural network model to be tested
        test_set (Set): test set to be used for metrics evaluation
        verbose (bool): display evaluation progress bar if True

    """
    result = network.evaluate(x=test_set.input_data,
                              y=test_set.output_data,
                              batch_size=None,
                              verbose=verbose,
                              sample_weight=None,
                              steps=None,
                              callbacks=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False)
    metrics = {}
    i = 0
    for name in network.metrics_names:
        metrics[name] = result[i]
        i = i + 1

    return metrics

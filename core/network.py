from keras import models, layers, Model
from keras.callbacks import History

from core.hyperparameters import LayerType
from core.sets import Set


def create_layer(**kwargs):
    """Creates a layer according to the hyperparameters
    """
    layer_type = kwargs.get('layer_type')

    # Input
    if layer_type == LayerType.INPUT:
        # return layers.Input(layer_configuration)
        pass

    # Dense
    elif layer_type == LayerType.DENSE:
        return layers.Dense(kwargs)

    # Dropout
    elif layer_type == LayerType.DROPOUT:
        return layers.Dropout(kwargs)

    # Unknown
    else:
        raise NotImplementedError('Unknown layer type')


def create_network(network_configuration: dict,
                   layer_configuration_list: list):
    """Creates a neural network according to its hyper parameters

    Args:
        network_configuration (dict): neural network hyperparameters
        layer_configuration_list (list): list of layer hyperparameters

    """
    network = models.Sequential()

    # layers
    for layer_configuration in layer_configuration_list:
        network.add(create_layer(layer_configuration))

    # compile the network
    network.compile(optimizer=network_configuration.get('optimizer'),
                    loss=network_configuration.get('loss'),
                    metrics=network_configuration.get('metrics'))

    return network


def train_network(
        network: Model,
        epochs: int,
        batch_size: int,
        shuffle: bool,
        training_set: Set,
        validation_set: Set = None) -> History:
    """Train the neural network, returning the evolution of the training metricsj

    Args:
        network        (Model) : neural network model to be trained
        epochs         (int)   : number of training epochs
        batch_size     (int)   : training batch size
        shuffle        (bool)  : shuffle training set before training
        training_set   (Set)   : training set
        validation_set (Set)   : validation set

    """
    if shuffle:
        working_training_set = training_set.copy()
        working_training_set.shuffle()
    else:
        working_training_set = training_set

    if validation_set is None:
        validation_data = None
    else:
        validation_data = validation_set.to_datasets()

    history = network.fit(
        x=working_training_set.input_data,
        y=working_training_set.output_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data)

    return history


def train_network_k_fold(
        network: Model,
        epochs: int,
        batch_size: int,
        k: int,
        shuffle: bool,
        training_set: Set) -> list:
    """Train the neural network model using k-fold cross-validation

    Args:
        network      (Model) : neural network model to be trained
        epochs       (int)   : number of passes through the training set
        batch_size   (int)   : training batch size
        k            (int)   : number of partitions in k-fold cross-validation
        shuffle      (bool)  : shuffle the training set before k splitting
        training_set (Set)   : training data set

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

        fold_history = train_network(
            network=network,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            training_set=training_set_remaining,
            validation_set=validation_set)

        all_histories.append(fold_history)

    return all_histories


def test_network(network: Model, test_set: Set):
    """Evaluates all the test inputs according to the current network

    Args:
        network (Model): neural network model to be tested
        test_set (Set): test set to be used for metrics evaluation

    """
    return network.evaluate(test_set.input_data, test_set.output_data)

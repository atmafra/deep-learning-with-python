from keras import models, layers, Model
from keras.callbacks import History

from core.hyperparameters import NetworkHyperparameters
from core.sets import Set


def create_network(network_hparm: NetworkHyperparameters):
    """Creates a neural network according to its hyper parameters

    Args:
        network_hparm (NetworkHyperparameters): neural network hyper parameters

    """
    network = models.Sequential()

    # layers
    for layer_hparm in network_hparm.layer_hyperparameters_list:

        if layer_hparm.input_shape is not None:
            network.add(layers.Dense(
                units=layer_hparm.units,
                activation=layer_hparm.activation,
                input_shape=layer_hparm.input_shape))

        else:
            network.add(layers.Dense(
                units=layer_hparm.units,
                activation=layer_hparm.activation))

    # compile the network
    network.compile(
        optimizer=network_hparm.optimizer,
        loss=network_hparm.loss,
        metrics=network_hparm.metrics)

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

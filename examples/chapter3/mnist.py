import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from core import network as net
from core import sets
from core.network import LayerType, NetworkOutputType
from utils import history_utils as hutl


def to_array(image_set: np.array) -> np.array:
    image_set_size = len(image_set)
    image_shape = image_set[0].shape
    array_dimensions = image_shape[0] * image_shape[1]
    return image_set.reshape(image_set_size, array_dimensions)


def normalize(image_set: np.array) -> np.array:
    return image_set.astype('float32') / 255


def load_corpus(verbose: bool = True) -> sets.Corpus:
    """Loads the MNIST corpus from public repositories
    """
    if verbose:
        print("Loading MNIST dataset...")

    corpus = sets.Corpus.from_tuple(mnist.load_data())

    train_set_size = corpus.training_set.length
    image_shape = corpus.training_set.input_data.shape
    image_array_dim = image_shape[0] * image_shape[1]
    test_set_size = corpus.test_set.length

    # Prepare the images
    train_images = normalize(to_array(corpus.training_set.input_data))
    test_images = normalize(to_array(corpus.test_set.input_data))

    # Convert the labels to categorical
    train_labels = to_categorical(corpus.training_set.output_data)
    test_labels = to_categorical(corpus.test_set.output_data)

    if verbose:
        print("image dimensions :", image_shape)
        print("train set size   :", train_set_size, "images")
        print("test set size    :", test_set_size, "images")

    corpus = sets.Corpus.from_datasets(train_images, train_labels, test_images, test_labels)
    return corpus


def hyperparameters(input_size: int, output_size) -> (dict, list):
    hidden_layer_units = 16
    hidden_layer_activation = 'relu'
    learning_rate = 0.001

    network_configuration = {
        'input_size': input_size,
        'output_size': output_size,
        'output_type': NetworkOutputType.CATEGORICAL,
        'optimizer': 'rmsprop',
        'learning_rate': learning_rate,
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']}

    layers_configuration = [
        {'layer_type': LayerType.DENSE, 'units': hidden_layer_units, 'activation': hidden_layer_activation,
         'input_shape': (input_size,)},
        {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': 'softmax'}]

    loss = 'categorical_crossentropy'

    return network_configuration, layers_configuration


def run():
    """Runs the MNIST digit recognition example
    """
    num_labels = 10
    corpus = load_corpus()
    input_size = corpus.input_size()
    output_size = corpus.output_size()

    network_configuration, layers_configuration = hyperparameters(input_size, output_size)

    mnist_nnet = net.create_network(network_configuration=network_configuration,
                                    layer_configuration_list=layers_configuration)

    epochs = 20
    batch_size = 128

    history = net.train_network(network=mnist_nnet,
                                training_set=corpus.training_set,
                                epochs=epochs,
                                shuffle=True,
                                batch_size=batch_size)

    hutl.plot_loss_dict(history.history, title='MNIST: Training Loss')
    (test_loss, test_accuracy) = net.test_network(mnist_nnet, corpus.test_set)

    print("Test loss     =", test_loss)
    print("Test accuracy = {:.2%}".format(test_accuracy))

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from core import network as net
from core import sets
from core.hyperparameters import NetworkHyperparameters, LayerHyperparameters, OutputType, LayerPosition
from utils import history_utils as hutl


def to_array(image_set: np.array) -> np.array:
    image_set_size = len(image_set)
    image_shape = image_set[0].shape
    array_dimensions = image_shape[0] * image_shape[1]
    return image_set.reshape(image_set_size, array_dimensions)


def normalize(image_set: np.array) -> np.array:
    return image_set.astype('float32') / 255


def load_corpus(verbose: bool = True) -> sets.Corpus:
    if verbose:
        print("Loading MNIST dataset...")

    corpus = sets.Corpus.from_tuple(mnist.load_data())

    # (train_images, train_labels), (test_images, test_labels) = dsu.split_corpus(corpus)
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


def hyperparameters() -> NetworkHyperparameters:
    input_size = 28 * 28
    hidden_layer_units = 16
    num_labels = 10
    learning_rate = 0.001

    input_layer_hparm = LayerHyperparameters(input_size, LayerPosition.INPUT, 'linear')
    hidden_layer_hparm = LayerHyperparameters(hidden_layer_units, LayerPosition.HIDDEN, 'relu')
    output_layer_hparm = LayerHyperparameters(num_labels, LayerPosition.OUTPUT, 'softmax')
    layer_hparm_list = [input_layer_hparm, output_layer_hparm]
    loss = 'categorical_crossentropy'

    mnist_hparm = NetworkHyperparameters(input_size=input_size, output_size=num_labels,
                                             output_type=OutputType.CATEGORICAL,
                                             layer_hyperparameters_list=layer_hparm_list,
                                             optimizer='rmsprop',
                                             learning_rate=learning_rate,
                                             loss='categorical_crossentropy',
                                             metrics=['accuracy'])
    return mnist_hparm


def run():
    hidden_layer_units = 512
    epochs = 20
    batch_size = 128

    corpus = load_corpus()
    mnist_hyperparameters = hyperparameters()
    mnist_nnet = net.create_network(mnist_hyperparameters)

    history = net.train_network(network=mnist_nnet,
                                training_set=corpus.training_set,
                                epochs=epochs,
                                shuffle=True,
                                batch_size=batch_size)

    hutl.plot_loss_dict(history.history, title='MNIST: Training Loss')
    (test_loss, test_accuracy) = net.test_network(mnist_nnet, corpus.test_set)
    print("Test loss     =", test_loss)
    print("Test accuracy = {:.2%}".format(test_accuracy))

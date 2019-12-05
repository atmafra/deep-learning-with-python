import numpy as np
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical

from core import sets
from core.experiment import Experiment
from core.network import LayerType, ValidationStrategy
from core.sets import Corpus


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

    corpus = Corpus.from_datasets(train_images, train_labels, test_images, test_labels)
    return corpus


def load_experiment(corpus: Corpus):
    """Loads the experiment hyperparameters
    """
    # layer parameters
    input_size = corpus.input_size
    output_size = corpus.output_size
    hidden_layer_units = 16
    hidden_layer_activation = 'relu'

    # compile parameters
    learning_rate = 0.001
    optimizer = optimizers.RMSprop(lr=learning_rate)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    # training parameters
    epochs = 20
    batch_size = 128
    shuffle = True

    layers_configuration = [
        {'layer_type': 'Dense', 'units': hidden_layer_units, 'activation': hidden_layer_activation,
         'input_shape': (input_size,)},
        {'layer_type': 'Dense', 'units': output_size, 'activation': 'softmax'}]

    training_configuration = {
        'keras': {
            'compile': {
                'optimizer': optimizer,
                'loss': loss,
                'metrics': metrics},
            'fit': {
                'epochs': epochs,
                'batch_size': batch_size,
                'shuffle': shuffle}},
        'validation': {
            'strategy': ValidationStrategy.NO_VALIDATION}}

    loss = 'categorical_crossentropy'

    experiment = Experiment(name="MNIST",
                            corpus=corpus,
                            layers_configuration=layers_configuration,
                            training_configuration=training_configuration)

    return experiment


def run():
    """Runs the MNIST digit recognition example
    """
    num_labels = 10
    corpus = load_corpus()
    experiment = load_experiment(corpus=corpus)
    experiment.run(print_results=True, plot_history=True, display_progress_bars=True)

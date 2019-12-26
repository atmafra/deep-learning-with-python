from enum import Enum

import numpy as np
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical

from core.corpus import Corpus, CorpusType
from core.experiment import Experiment
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration


class ModelSource(Enum):
    CONFIGURATION = 0
    FILE = 1


def to_array(image_set: np.array) -> np.array:
    image_set_size = len(image_set)
    image_shape = image_set[0].shape
    array_dimensions = image_shape[0] * image_shape[1]
    return image_set.reshape(image_set_size, array_dimensions)


def normalize(image_set: np.array) -> np.array:
    return image_set.astype('float32') / 255


def load_corpus(verbose: bool = True) -> Corpus:
    """Loads the MNIST corpus from public repositories
    """
    if verbose:
        print("Loading MNIST dataset...")

    corpus = Corpus.from_tuple(mnist.load_data())

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


def create_neural_network(corpus: Corpus, model_source: ModelSource):
    """Loads the experiment hyperparameters
    """
    # layer parameters
    input_size = corpus.input_size
    output_size = corpus.output_size
    hidden_layer_units = 16
    hidden_layer_activation = 'relu'


    layers_configuration = [
        {'layer_type': 'Dense', 'name': 'Dense-1', 'units': hidden_layer_units, 'activation': hidden_layer_activation,
         'input_shape': (input_size,)},
        {'layer_type': 'Dense', 'name': 'Dense-2', 'units': output_size, 'activation': 'softmax'}]

    if model_source == ModelSource.CONFIGURATION:
        return NeuralNetwork.from_configurations(name='MNIST: MLP, 2 dense layers',
                                                 layers_configuration=layers_configuration)

    elif model_source == ModelSource.FILE:
        return NeuralNetwork.from_file(path='models/mnist',
                                       filename='mnist-mlp-2-dense-layers-short.json',
                                       verbose=True)

    raise RuntimeError('Unknown model source')


def create_experiment(corpus: Corpus,
                      neural_network: NeuralNetwork):
    # compile parameters
    learning_rate = 0.001
    optimizer = optimizers.RMSprop(lr=learning_rate)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    # train parameters
    epochs = 20
    batch_size = 128
    shuffle = True
    training_parameters = {
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

    training_configuration = TrainingConfiguration(configuration=training_parameters)

    return Experiment(name="MNIST",
                      corpus_type=CorpusType.CORPUS_DATASET,
                      corpus=corpus,
                      neural_network=neural_network,
                      training_configuration=training_configuration)


def run():
    """Runs the MNIST digit recognition example
    """
    num_labels = 10
    corpus = load_corpus()
    neural_network = create_neural_network(corpus=corpus, model_source=ModelSource.FILE)
    experiment = create_experiment(corpus, neural_network)
    path = 'models/mnist'
    experiment.run(print_results=True, plot_history=True, display_progress_bars=True)
    experiment.save_model(path='models/mnist')

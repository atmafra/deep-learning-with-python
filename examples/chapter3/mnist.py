from enum import Enum

import numpy as np
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical

from core.corpus import Corpus, CorpusType
from core.experiment import Experiment
from core.file_structures import CorpusFileStructure
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from utils.file_utils import str_to_filename


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


def build_corpus(save: bool, verbose: bool = True) -> Corpus:
    """Loads the MNIST corpus from public repositories
    """
    if verbose:
        print("Loading MNIST dataset...")

    corpus_raw = Corpus.from_tuple(corpus=mnist.load_data(), name='MNIST')

    train_set_size = corpus_raw.training_set.length
    image_shape = corpus_raw.training_set.input_data.shape
    image_array_dim = image_shape[0] * image_shape[1]
    test_set_size = corpus_raw.test_set.length

    # Prepare the images
    train_images = normalize(to_array(corpus_raw.training_set.input_data))
    test_images = normalize(to_array(corpus_raw.test_set.input_data))

    # Convert the labels to categorical
    train_labels = to_categorical(corpus_raw.training_set.output_data)
    test_labels = to_categorical(corpus_raw.test_set.output_data)

    if verbose:
        print("image dimensions :", image_shape)
        print("train set size   :", train_set_size, "images")
        print("test set size    :", test_set_size, "images")

    corpus = Corpus.from_datasets(training_input=train_images, training_output=train_labels,
                                  test_input=test_images, test_output=test_labels,
                                  validation_input=None, validation_output=None,
                                  name='MNIST')

    if save:
        save_corpus(corpus)

    return corpus


def save_corpus(corpus: Corpus,
                corpus_file_structure: CorpusFileStructure = None):
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus.name, base_path='data/mnist')
    corpus_file_structure.save_corpus(corpus=corpus)


def load_corpus(corpus_name: str,
                corpus_file_structure: CorpusFileStructure = None) -> Corpus:
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus_name, base_path='data/mnist')
    return corpus_file_structure.load_corpus(corpus_name=corpus_name, datasets_base_name=corpus_name)


def create_neural_network(corpus: Corpus,
                          network_name: str,
                          model_source: ModelSource,
                          model_path: str,
                          model_filename: str = ''):
    """Loads the experiment hyperparameters
    """
    # layer parameters
    input_size = corpus.input_size
    output_size = corpus.output_size
    hidden_layer_units = 16
    hidden_layer_activation = 'relu'

    layers_configuration = [
        {'class_name': 'Dense', 'name': 'Dense-1', 'units': hidden_layer_units, 'activation': hidden_layer_activation,
         'input_shape': (input_size,)},
        {'class_name': 'Dense', 'name': 'Dense-2', 'units': output_size, 'activation': 'softmax'}]

    if model_source == ModelSource.CONFIGURATION:
        return NeuralNetwork.from_configurations(name=network_name,
                                                 layers_configuration=layers_configuration)

    elif model_source == ModelSource.FILE:
        if not model_filename:
            model_filename = str_to_filename(network_name) + '.json'

        return NeuralNetwork.from_architecture_and_weights(path=model_path,
                                                           filename=model_filename,
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

    return Experiment(name=neural_network.name,
                      corpus_type=CorpusType.CORPUS_DATASET,
                      corpus=corpus,
                      neural_network=neural_network,
                      training_configuration=training_configuration)


def run(build: bool = True,
        load_model_configuration: bool = True):
    """
    Runs the MNIST digit recognition example
    :param build: build (or load, if False) the corpus before training
    :param load_model_configuration: load configuration from JSON file
    """
    model_path = 'models/mnist'

    if build:
        corpus = build_corpus(save=True)
    else:
        corpus = load_corpus(corpus_name='MNIST')

    network_name = 'MNIST: MLP, 2 dense layers'
    if load_model_configuration:
        model_source = ModelSource.FILE
    else:
        model_source = ModelSource.CONFIGURATION

    neural_network = create_neural_network(corpus=corpus,
                                           network_name=network_name,
                                           model_source=model_source,
                                           model_path=model_path)

    experiment = create_experiment(corpus, neural_network)

    experiment.run(train=True,
                   display_progress_bars=True,
                   print_training_results=True,
                   plot_training_loss=True,
                   save=True,
                   model_path=model_path)

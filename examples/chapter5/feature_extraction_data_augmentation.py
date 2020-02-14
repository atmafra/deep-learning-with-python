from keras import Model, optimizers, layers
from keras.applications import VGG16

from core.corpus import CorpusType
from core.experiment import Experiment
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from examples.chapter5.cats_and_dogs_files import *

classifier = [
    {'layer_type': 'Dense', 'units': 256, 'activation': 'relu', 'input_dim': 4 * 4 * 512},
    {'layer_type': 'Dropout', 'rate': 0.5},
    {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}]

learning_rate = 2e-5
training_parameters = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=learning_rate),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit_generator': {
            'epochs': 30,
            'steps_per_epoch': 100,
            'validation_steps': 50}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION}}


def get_convolutional_base() -> Model:
    return VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))


def build_network(name: str):
    neural_network = NeuralNetwork(name=name)
    convolutional_base = get_convolutional_base()
    convolutional_base.trainable = False
    neural_network.append_model(model=convolutional_base)
    neural_network.append_model(layers.Flatten())
    neural_network.append_layers(layers_configuration=classifier)
    return neural_network


def load_experiment(experiment_name: str,
                    corpus_files: CorpusFiles):
    """Loads the experiment
    """
    neural_network = build_network(name=experiment_name)
    training_configuration = TrainingConfiguration(training_parameters)

    experiment = Experiment(name=experiment_name,
                            neural_network=neural_network,
                            training_configuration=training_configuration,
                            corpus_type=CorpusType.CORPUS_GENERATOR,
                            corpus_files=corpus_files)
    return experiment


def run(build_dataset: bool = False):
    dirs = prepare_directories()
    copy_files(dirs=dirs, check=True)
    corpus_files = load_corpus_files(dirs=dirs, use_augmented=False, check=True)
    corpus_files_augmented = load_corpus_files(dirs=dirs, use_augmented=True, check=True)

    corpus_name = 'Cats and Dogs'
    experiment_name = corpus_name + ' - Feature Extraction with Data Augmentation'

    experiment = load_experiment(experiment_name=experiment_name,
                                 corpus_files=corpus_files_augmented)

    experiment.run(print_results=True, plot_history=False, display_progress_bars=True)

    experiment.save_architecture(path='models/feature_extraction_data_augmentation')
    experiment.save_weights()

    experiment.plot_loss()
    experiment.plot_accuracy()

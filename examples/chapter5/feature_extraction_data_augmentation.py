from keras import Model, optimizers, layers
from keras.applications import VGG16

from core.convolutional_neural_network import ConvolutionalNeuralNetwork
from core.corpus import CorpusType
from core.experiment import Experiment
import core.network as net
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from examples.chapter5.cats_and_dogs_files import *

classifier_configuration = [
    {'layer_type': 'Dense', 'units': 256, 'activation': 'relu', 'input_dim': 4 * 4 * 512},
    {'layer_type': 'Dropout', 'rate': 0.5},
    {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}]

training_parameters = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=2e-5),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit_generator': {
            'epochs': 30,
            'steps_per_epoch': 100,
            'validation_steps': 50}},
    'validation': {
        'strategy': net.ValidationStrategy.CROSS_VALIDATION}}

fine_tuning_parameters = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=1e-5),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit_generator': {
            'epochs': 100,
            'steps_per_epoch': 100,
            'validation_steps': 50}},
    'validation': {
        'strategy': net.ValidationStrategy.CROSS_VALIDATION}}


def build_network(name: str):
    """Builds the convolutional neural network
    """
    convolutional_base = VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(150, 150, 3))

    classifier = net.create_model(name='Classifier',
                                  layer_configuration_list=classifier_configuration)

    neural_network = ConvolutionalNeuralNetwork(name=name,
                                                convolutional_base=convolutional_base,
                                                classifier=classifier)
    return neural_network


def load_experiment(experiment_name: str,
                    corpus_files: CorpusFiles):
    """Loads the experiment
    """
    neural_network = build_network(name=experiment_name)
    training_configuration = TrainingConfiguration(training_parameters)
    fine_tuning_configuration = TrainingConfiguration(fine_tuning_parameters)

    experiment = Experiment(name=experiment_name,
                            neural_network=neural_network,
                            training_configuration=training_configuration,
                            fine_tuning_configuration=fine_tuning_configuration,
                            corpus_type=CorpusType.CORPUS_GENERATOR,
                            corpus_files=corpus_files)
    return experiment


def run(fine_tune: bool = True):
    """Run the experiment
    """
    dirs = prepare_files(check=True)
    corpus_files_augmented = load_corpus_files(dirs=dirs, use_augmented=True, check=False)
    corpus_name = 'Cats and Dogs'
    experiment_name = corpus_name + ' - Feature Extraction with Data Augmentation and Fine Tuning'

    experiment = load_experiment(experiment_name=experiment_name,
                                 corpus_files=corpus_files_augmented)

    experiment.run(train=True,
                   test_after_training=True,
                   print_training_results=True,
                   fine_tune=fine_tune,
                   unfreeze_layers={'block5_conv1'},
                   plot_training_loss=True,
                   plot_training_accuracy=True,
                   plot_fine_tuning_loss=True,
                   plot_fine_tuning_accuracy=True,
                   training_plot_smooth_factor=0.,
                   validation_plot_smooth_factor=0.8,
                   test=True,
                   print_test_results=True,
                   save=True,
                   model_path='models/cats_and_dogs',
                   display_progress_bars=True)


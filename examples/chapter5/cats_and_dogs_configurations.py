from keras import optimizers

from core.corpus import CorpusGenerator
from core.experiment import Experiment, CorpusType, ExperimentPlan
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration

input_shape = (150, 150, 3)
kernel_size = (3, 3)
pool_size = (2, 2)
output_size = 1

# network architecture
cats_and_dogs = [
    {'class_name': 'Conv2D', 'filters': 32, 'kernel_size': kernel_size, 'activation': 'relu',
     'input_shape': input_shape},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Conv2D', 'filters': 64, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Conv2D', 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Conv2D', 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Flatten'},
    {'class_name': 'Dense', 'units': 512, 'activation': 'relu'},
    {'class_name': 'Dense', 'units': output_size, 'activation': 'sigmoid'}]

cats_and_dogs_dropout = [
    {'class_name': 'Conv2D', 'filters': 32, 'kernel_size': kernel_size, 'activation': 'relu',
     'input_shape': input_shape},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Conv2D', 'filters': 64, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Conv2D', 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Conv2D', 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'class_name': 'MaxPooling2D', 'pool_size': pool_size},
    {'class_name': 'Flatten'},
    {'class_name': 'Dropout', 'rate': 0.5},
    {'class_name': 'Dense', 'units': 512, 'activation': 'relu'},
    {'class_name': 'Dense', 'units': output_size, 'activation': 'sigmoid'}]

# train configuration
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

training_configuration = TrainingConfiguration(configuration=training_parameters)


def load_experiment_plan(corpus_generator: CorpusGenerator,
                         corpus_generator_augmented: CorpusGenerator) -> ExperimentPlan:
    """Loads the Cats & Dogs experiment plan
    """
    # Neural Networks
    neural_network_cats_and_dogs = \
        NeuralNetwork.from_configurations(name='Cats and Dogs - convolutional',
                                          layers_configuration=cats_and_dogs)

    neural_network_cats_and_dogs_dropout = \
        NeuralNetwork.from_configurations(name='Cats and Dogs - convolutional with dropout',
                                          layers_configuration=cats_and_dogs_dropout)

    regular = Experiment(name='Cats and Dogs',
                         neural_network=neural_network_cats_and_dogs,
                         training_configuration=training_configuration,
                         corpus_type=CorpusType.CORPUS_GENERATOR,
                         corpus_files=corpus_generator)

    dropout = Experiment(name='Cats and Dogs (dropout rate: 0.5)',
                         neural_network=neural_network_cats_and_dogs_dropout,
                         training_configuration=training_configuration,
                         corpus_type=CorpusType.CORPUS_GENERATOR,
                         corpus_files=corpus_generator)

    dropout_augmented = Experiment(name='Cats and Dogs (dropout rate: 0.5, data augmentation)',
                                   neural_network=neural_network_cats_and_dogs_dropout,
                                   training_configuration=training_configuration,
                                   corpus_type=CorpusType.CORPUS_GENERATOR,
                                   corpus_files=corpus_generator_augmented)

    experiment_list = [regular, dropout, dropout_augmented]
    experiment_plan = ExperimentPlan(name='Cats and Dogs', experiments=experiment_list)

    return experiment_plan

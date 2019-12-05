from keras import optimizers

from core.experiment import Experiment, CorpusType
from core.network import ValidationStrategy
from core.sets import CorpusGenerator

input_shape = (150, 150, 3)
kernel_size = (3, 3)
pool_size = (2, 2)
output_size = 1

# network architecture
cats_and_dogs = [
    {'layer_type': 'Conv2D', 'filters': 32, 'kernel_size': kernel_size, 'activation': 'relu',
     'input_shape': input_shape},
    {'layer_type': 'MaxPooling2D', 'pool_size': pool_size},
    {'layer_type': 'Conv2D', 'filters': 64, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'layer_type': 'MaxPooling2D', 'pool_size': pool_size},
    {'layer_type': 'Conv2D', 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'layer_type': 'MaxPooling2D', 'pool_size': pool_size},
    {'layer_type': 'Conv2D', 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'layer_type': 'MaxPooling2D', 'pool_size': pool_size},
    {'layer_type': 'Flatten'},
    {'layer_type': 'Dense', 'units': 512, 'activation': 'relu'},
    {'layer_type': 'Dense', 'units': output_size, 'activation': 'sigmoid'}]

# training configuration
learning_rate = 1e-4

training_configuration = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=learning_rate),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit_generator': {
            'epochs': 30,
            'steps_per_epoch': 100,
            'validation_steps': 50,
            'shuffle': True}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION}}


def load_experiment(corpus_generator: CorpusGenerator) -> Experiment:
    experiment_cats_and_dogs = Experiment(name='Cats and Dogs',
                                          layers_configuration=cats_and_dogs,
                                          training_configuration=training_configuration,
                                          corpus_type=CorpusType.CORPUS_GENERATOR,
                                          corpus_generator=corpus_generator)
    return experiment_cats_and_dogs

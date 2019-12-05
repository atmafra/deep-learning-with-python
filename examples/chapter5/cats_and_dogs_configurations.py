from keras import optimizers

from core.experiment import Experiment, CorpusType
from core.network import LayerType, ValidationStrategy
from core.sets import CorpusGenerator

input_shape = (150, 150, 3)
kernel_size = (3, 3)
pool_size = (2, 2)
output_size = 1

# compile parameters
learning_rate = 1e-4
optimizer = optimizers.RMSprop(lr=learning_rate)
loss = 'binary_crossentropy'
metrics = ['accuracy']

# training parameters
epochs = 5
batch_size = 128
shuffle = True

cats_and_dogs = [
    {'layer_type': LayerType.CONV_2D, 'filters': 32, 'kernel_size': kernel_size, 'activation': 'relu',
     'input_shape': input_shape},
    {'layer_type': LayerType.MAX_POOLING_2D, 'pool_size': pool_size},
    {'layer_type': LayerType.CONV_2D, 'filters': 64, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'layer_type': LayerType.MAX_POOLING_2D, 'pool_size': pool_size},
    {'layer_type': LayerType.CONV_2D, 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'layer_type': LayerType.MAX_POOLING_2D, 'pool_size': pool_size},
    {'layer_type': LayerType.CONV_2D, 'filters': 128, 'kernel_size': kernel_size, 'activation': 'relu'},
    {'layer_type': LayerType.MAX_POOLING_2D, 'pool_size': pool_size},
    {'layer_type': LayerType.FLATTEN},
    {'layer_type': LayerType.DENSE, 'units': 512, 'activation': 'relu'},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': 'sigmoid'}]

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


def load_experiment(corpus_generator: CorpusGenerator) -> Experiment:
    experiment_cats_and_dogs = Experiment(name='Cats and Dogs',
                                          layers_configuration_list=cats_and_dogs,
                                          training_configuration=training_configuration,
                                          corpus_type=CorpusType.CORPUS_GENERATOR,
                                          corpus_generator=corpus_generator)
    return experiment_cats_and_dogs

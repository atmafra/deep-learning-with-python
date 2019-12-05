from keras import optimizers

from core.experiment import Experiment, ExperimentPlan
from core.network import LayerType, ValidationStrategy
from core.sets import Corpus

# layer parameters
image_width = 28
image_height = 28
channels = 1
input_size = image_width * image_height
output_size = 10
input_shape_array = (input_size,)
input_shape_image = (image_width, image_height, channels)

# compile parameters
learning_rate = 0.001
optimizer = optimizers.RMSprop(lr=learning_rate)
loss = 'categorical_crossentropy'
metrics = ['accuracy']

# training parameters
epochs = 5
batch_size = 128
shuffle = True

# Network configurations
mnist_dense = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': 'relu',
     'input_shape': input_shape_array},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': 'softmax'}]

mnist_conv_max_pooling = [
    {'layer_type': LayerType.CONV_2D, 'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu',
     'input_shape': input_shape_image},
    {'layer_type': LayerType.MAX_POOLING_2D, 'pool_size': (2, 2)},
    {'layer_type': LayerType.CONV_2D, 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'layer_type': LayerType.MAX_POOLING_2D, 'pool_size': (2, 2)},
    {'layer_type': LayerType.CONV_2D, 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'layer_type': LayerType.FLATTEN},
    {'layer_type': LayerType.DENSE, 'units': 64, 'activation': 'relu'},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': 'softmax'}]

mnist_conv_avg_pooling = [
    {'layer_type': LayerType.CONV_2D, 'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu',
     'input_shape': input_shape_image},
    {'layer_type': LayerType.AVERAGE_POOLING_2D, 'pool_size': (2, 2)},
    {'layer_type': LayerType.CONV_2D, 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'layer_type': LayerType.AVERAGE_POOLING_2D, 'pool_size': (2, 2)},
    {'layer_type': LayerType.CONV_2D, 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'layer_type': LayerType.FLATTEN},
    {'layer_type': LayerType.DENSE, 'units': 64, 'activation': 'relu'},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': 'softmax'}]

mnist_conv_strided = [
    {'layer_type': LayerType.CONV_2D, 'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu',
     'strides': 2, 'input_shape': input_shape_image},
    {'layer_type': LayerType.CONV_2D, 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'strides': 2},
    {'layer_type': LayerType.CONV_2D, 'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'strides': 2},
    {'layer_type': LayerType.FLATTEN},
    {'layer_type': LayerType.DENSE, 'units': 64, 'activation': 'relu'},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': 'softmax'}]

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


def load_experiment_plan(corpus: Corpus) -> ExperimentPlan:
    """Loads the experiment hyperparameters
    """
    # experiment_dense = Experiment(name="MNIST Dense",
    #                               corpus=corpus,
    #                               layers_configuration_list=mnist_dense,
    #                               training_configuration=training_configuration)
    experiment_conv_max_pooling = Experiment(name="MNIST Convolutional (2x2 Max Polling)",
                                             corpus=corpus,
                                             layers_configuration=mnist_conv_max_pooling,
                                             training_configuration=training_configuration)

    experiment_conv_avg_pooling = Experiment(name="MNIST Convolutional (2x2 Average Polling)",
                                             corpus=corpus,
                                             layers_configuration=mnist_conv_avg_pooling,
                                             training_configuration=training_configuration)

    experiment_conv_strided = Experiment(name="MNIST Convolutional (2x2 Strided)",
                                         corpus=corpus,
                                         layers_configuration=mnist_conv_strided,
                                         training_configuration=training_configuration)

    experiment_list = [experiment_conv_max_pooling,
                       experiment_conv_avg_pooling,
                       experiment_conv_strided]

    return ExperimentPlan(name="MNIST Convolutional Experiments", experiments=experiment_list)

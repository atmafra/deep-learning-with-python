from keras import regularizers, optimizers, activations

from core.network import NetworkOutputType, LayerType

# global corpus parameters
num_words = 10000
input_size = num_words
output_size = 1

# global network parameters
hidden_activation = 'relu'
output_activation = 'sigmoid'

# global training parameters
epochs = 20
batch_size = 512
validation_set_size = 10000
shuffle = True

network_configuration_global = {
    'input_size': input_size,
    'output_size': output_size,
    'output_type': NetworkOutputType.BOOLEAN}

training_configuration_global = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=0.001),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit': {
            'epochs': epochs,
            'batch_size': batch_size,
            'shuffle': shuffle}},
    'validation_set_size': validation_set_size}

imdb_small = [
    {'layer_type': LayerType.DENSE, 'units': 4, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 4, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_large = [
    {'layer_type': LayerType.DENSE, 'units': 512, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DENSE, 'units': 512, 'activation': hidden_activation},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_dropout = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l1 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1(0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1(0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l1_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1_l2(l1=0.001, l2=0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1_l2(l1=0.001, l2=0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_dropout_wreg_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

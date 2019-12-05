from keras import regularizers, optimizers

from core.experiment import Experiment, ExperimentPlan
from core.network import LayerType, ValidationStrategy
from core.sets import Corpus

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
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION,
        'set_size': validation_set_size}}

# Network Configuration
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

imdb_medium_dropout_10 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.10},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.10},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_dropout_20 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.20},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.20},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_dropout_30 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.30},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.30},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_dropout_40 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.40},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.40},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_dropout_50 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.50},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.50},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l1_0001 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1(0.0001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1(0.0001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l1_001 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1(0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1(0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l1_01 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1(0.01)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1(0.01)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l1_1 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1(0.1)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1(0.1)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l2_0001 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.0001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.0001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l2_001 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l2_01 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.01)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.01)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l2_1 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.1)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.1)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_wreg_l1_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l1_l2(l1=0.0001, l2=0.0001)},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l1_l2(l1=0.0001, l2=0.0001)},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

imdb_medium_dropout_wreg_l2 = [
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,),
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation,
     'kernel_regularizer': regularizers.l2(0.001)},
    {'layer_type': LayerType.DROPOUT, 'rate': 0.5},
    {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]


def load_experiments(corpus: Corpus):
    _small = Experiment(name='small', corpus=corpus,
                        layers_configuration=imdb_small,
                        training_configuration=training_configuration_global)

    _medium = Experiment(name='medium', corpus=corpus,
                         layers_configuration=imdb_medium,
                         training_configuration=training_configuration_global)

    _large = Experiment(name='large', corpus=corpus,
                        layers_configuration=imdb_large,
                        training_configuration=training_configuration_global)

    _medium_dropout_10 = Experiment(name='medium with dropout (rate = 10%)', corpus=corpus,
                                    layers_configuration=imdb_medium_dropout_10,
                                    training_configuration=training_configuration_global)

    _medium_dropout_20 = Experiment(name='medium with dropout (rate = 20%)', corpus=corpus,
                                    layers_configuration=imdb_medium_dropout_20,
                                    training_configuration=training_configuration_global)

    _medium_dropout_30 = Experiment(name='medium with dropout (rate = 30%)', corpus=corpus,
                                    layers_configuration=imdb_medium_dropout_30,
                                    training_configuration=training_configuration_global)

    _medium_dropout_40 = Experiment(name='medium with dropout (rate = 40%)', corpus=corpus,
                                    layers_configuration=imdb_medium_dropout_40,
                                    training_configuration=training_configuration_global)

    _medium_dropout_50 = Experiment(name='medium with dropout (rate = 50%)', corpus=corpus,
                                    layers_configuration=imdb_medium_dropout_50,
                                    training_configuration=training_configuration_global)

    _medium_wreg_l1_0001 = Experiment(name='medium, weight regularization (L1 = 0.0001)', corpus=corpus,
                                      layers_configuration=imdb_medium_wreg_l1_0001,
                                      training_configuration=training_configuration_global)

    _medium_wreg_l1_001 = Experiment(name='medium, weight regularization (L1 = 0.001)', corpus=corpus,
                                     layers_configuration=imdb_medium_wreg_l1_001,
                                     training_configuration=training_configuration_global)

    _medium_wreg_l1_01 = Experiment(name='medium, weight regularization (L1 = 0.01)', corpus=corpus,
                                    layers_configuration=imdb_medium_wreg_l1_01,
                                    training_configuration=training_configuration_global)

    _medium_wreg_l1_1 = Experiment(name='medium, weight regularization (L1 = 0.1)', corpus=corpus,
                                   layers_configuration=imdb_medium_wreg_l1_1,
                                   training_configuration=training_configuration_global)

    _medium_wreg_l2_0001 = Experiment(name='medium, weight regularization (L2 = 0.0001)', corpus=corpus,
                                      layers_configuration=imdb_medium_wreg_l2_0001,
                                      training_configuration=training_configuration_global)

    _medium_wreg_l2_001 = Experiment(name='medium, weight regularization (L2 = 0.001)', corpus=corpus,
                                     layers_configuration=imdb_medium_wreg_l2_001,
                                     training_configuration=training_configuration_global)

    _medium_wreg_l2_01 = Experiment(name='medium, weight regularization (L2 = 0.01)', corpus=corpus,
                                    layers_configuration=imdb_medium_wreg_l2_01,
                                    training_configuration=training_configuration_global)

    _medium_wreg_l2_1 = Experiment(name='medium, weight regularization (L2 = 0.1)', corpus=corpus,
                                   layers_configuration=imdb_medium_wreg_l2_1,
                                   training_configuration=training_configuration_global)

    _medium_wreg_l1_l2 = Experiment(name='medium, weight regularization (L1 = 0.0001, L2 = 0.0001)', corpus=corpus,
                                    layers_configuration=imdb_medium_wreg_l1_l2,
                                    training_configuration=training_configuration_global)

    _medium_dropout_wreg_l2 = Experiment(name='medium with dropout and L2 weight regularization', corpus=corpus,
                                         layers_configuration=imdb_medium_dropout_wreg_l2,
                                         training_configuration=training_configuration_global)

    # Experiment: effect of dropout with different rates
    trials_dropout = [_medium_dropout_10, _medium_dropout_20, _medium_dropout_30, _medium_dropout_40,
                      _medium_dropout_50]
    experiment_dropout = ExperimentPlan(name="Dropout: effect of dropout rate", experiments=trials_dropout)

    # Experiment: effects of L1 weight regularization
    trials_list_wreg_l1 = [_medium_wreg_l1_0001, _medium_wreg_l1_001, _medium_wreg_l1_01, _medium_wreg_l1_1]
    experiment_wreg_l1 = ExperimentPlan(name="Weight Regularization: effect of L1", experiments=trials_list_wreg_l1)

    # Experiment: effects of L2 weight regularization
    trials_list_wreg_l2 = [_medium_wreg_l2_0001, _medium_wreg_l2_001, _medium_wreg_l2_01, _medium_wreg_l2_1]
    experiment_wreg_l2 = ExperimentPlan(name="Weight Regularization: effect of L2", experiments=trials_list_wreg_l2)

    # Experiment: effects of combining L1 and L2 weight regularization methods
    trials_list_comparison = [_medium, _medium_dropout_40, _medium_wreg_l1_0001, _medium_wreg_l2_0001,
                              _medium_wreg_l1_l2]
    experiment_comparison = ExperimentPlan(name="Comparison: Overfitting Techniques",
                                           experiments=trials_list_comparison)

    experiments = {"dropout": experiment_dropout,
                   "weight_regularization_l1": experiment_wreg_l1,
                   "weight_regularization_l2": experiment_wreg_l2,
                   "comparison": experiment_comparison}

    return experiments

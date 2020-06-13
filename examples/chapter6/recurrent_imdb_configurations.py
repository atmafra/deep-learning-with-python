from keras import optimizers

from core.corpus import Corpus
from core.experiment import Experiment, ExperimentPlan
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration

global_epochs = 10
global_batch_size = 128
global_shuffle = False


def get_training_configuration(set_size: int, validation_split: float = 0.2):
    validation_set_size = int(validation_split * set_size)
    training_parameters = {
        'keras': {
            'compile': {
                'optimizer': optimizers.RMSprop(lr=0.001),
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy']},
            'fit': {
                'epochs': global_epochs,
                'batch_size': global_batch_size,
                'shuffle': global_shuffle}},
        'validation': {
            'strategy': ValidationStrategy.CROSS_VALIDATION,
            'set_size': validation_set_size}}

    return TrainingConfiguration(configuration=training_parameters)


def get_neural_network(vocabulary_size: int,
                       embeddings_dimension: int,
                       input_length: int,
                       layer_type: str = 'SimpleRNN'):
    configurations = [
        {'class_name': 'Embedding', 'input_dim': vocabulary_size, 'output_dim': embeddings_dimension,
         'input_length': input_length, 'name': 'embeddings'},
        {'class_name': layer_type, 'units': embeddings_dimension},
        {'class_name': 'Dense', 'units': 1, 'activation': 'sigmoid'}
    ]

    return NeuralNetwork.from_configurations(name='SimpleRNN',
                                             layers_configuration=configurations)


def load_experiment_plan(corpus: Corpus):
    maxlen = 500
    vocabulary_size = 10000
    embeddings_dimension = 32

    set_size = corpus.size
    validation_split = 0.2

    training_configuration = get_training_configuration(set_size=set_size, validation_split=validation_split)

    simple_rnn_neural_network = get_neural_network(vocabulary_size=vocabulary_size,
                                                   embeddings_dimension=embeddings_dimension,
                                                   input_length=maxlen,
                                                   layer_type='SimpleRNN')

    simple_rnn_experiment = Experiment(name='SimpleRNN',
                                       id='simple-rnn',
                                       corpus=corpus,
                                       neural_network=simple_rnn_neural_network,
                                       training_configuration=training_configuration)

    lstm_neural_network = get_neural_network(vocabulary_size=vocabulary_size,
                                             embeddings_dimension=embeddings_dimension,
                                             input_length=maxlen,
                                             layer_type='LSTM')

    lstm_experiment = Experiment(name='LSTM',
                                 id='lstm',
                                 corpus=corpus,
                                 neural_network=lstm_neural_network,
                                 training_configuration=training_configuration)

    experiment_list = [
        simple_rnn_experiment,
        lstm_experiment
    ]

    return ExperimentPlan(name='Recurrent Neural Networks',
                               experiments=experiment_list)

from keras import regularizers, optimizers, layers

from core.corpus import Corpus
from core.experiment import Experiment, ExperimentPlan
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from embeddings.glove import glove

num_words = 10000
input_size = num_words
output_size = 1

# global network parameters
hidden_activation = 'relu'
output_activation = 'sigmoid'

# global train parameters
global_epochs = 10
global_batch_size = 1000
shuffle = True

# Training configuration
training_parameters_embeddings = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=0.001),
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']},
        'fit': {
            'epochs': global_epochs,
            'batch_size': global_batch_size,
            'shuffle': shuffle}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION}}

# training_parameters_glove = {
#     'keras': {
#         'compile': {
#             'optimizer': optimizers.RMSprop(lr=0.001),
#             'loss': 'binary_crossentropy',
#             'metrics': ['accuracy']},
#         'fit': {
#             'epochs': global_epochs,
#             'batch_size': global_batch_size,
#             'shuffle': shuffle}},
#     'validation': {
#         'strategy': ValidationStrategy.CROSS_VALIDATION,
#         'set_size': global_validation_set_size}}

training_configuration_embeddings = TrainingConfiguration(configuration=training_parameters_embeddings)


# training_configuration_glove = TrainingConfiguration(configuration=training_parameters_glove)


# def get_network_configuration(input_length: int,
#                               vocabulary_size: int,
#                               embeddings_dimension: int,
#                               output_categories: int):
#     return [
#         {'class_name': 'Embedding', 'input_dim': vocabulary_size, 'output_dim': embeddings_dimension,
#          'input_length': input_length, 'name': 'embeddings'},
#         {'class_name': 'Flatten'},
#         {'class_name': 'Dense', 'units': output_categories, 'activation': 'softmax'}
#     ]


def get_neural_network(name: str,
                       input_length: int,
                       vocabulary_size: int,
                       embeddings_dimension: int,
                       output_categories: int,
                       recurrent_layer_type: str = 'SimpleRNN',
                       recurrent_units: int = 100,
                       dropout_rate: float = 0.):
    """ Configure and build the neural network

    :param name: neural network name
    :param input_length: maximum number of tokens in the input
    :param vocabulary_size: vocabulary size, for one-hot encoding
    :param embeddings_dimension: dimension of the embeddings
    :param output_categories: number of distinct categories (intentions) in the output
    :param recurrent_layer_type: Recurrent Neural Network architecture (Flatten, SimpleRNN, GRU, LSTM, ...)
    :param recurrent_units: number of processing units of the recurrent layer
    :param dropout_rate: dropout rate
    :return: list of neural network layers configuration
    """
    if recurrent_layer_type == 'Flatten':
        recurrent_layer_configuration = {'class_name': recurrent_layer_type}
    elif recurrent_layer_type == 'BILSTM':
        recurrent_layer_configuration = {
            'class_name': 'Bidirectional',
            'layer': {
                'class_name': 'LSTM',
                'units': recurrent_units,
                'dropout': dropout_rate,
                'recurrent_dropout': dropout_rate,
            }
        }
    else:
        recurrent_layer_configuration = {
            'class_name': recurrent_layer_type,
            'units': recurrent_units,
            'dropout': dropout_rate,
            'recurrent_dropout': dropout_rate
        }

    configuration = [
        {'class_name': 'Embedding', 'input_dim': vocabulary_size, 'output_dim': embeddings_dimension,
         'input_length': input_length, 'name': 'embeddings'},
        recurrent_layer_configuration,
        {'class_name': 'Dense', 'units': output_categories, 'activation': 'softmax'}
    ]

    return NeuralNetwork.from_configurations(name=name, layers_configuration=configuration)


def load_neural_network():
    pass


def load_experiment(corpus: Corpus,
                    network_name: str,
                    experiment_id: str,
                    vocabulary_size: int,
                    embeddings_dimension: int,
                    recurrent_layer_type: str = 'Flatten',
                    recurrent_units: int = 0,
                    dropout_rate: float = 0.) -> Experiment:
    """ Loads one Rutger Intent Detection experiment

    :param corpus: corpus (common to all experiments)
    :param network_name: neural network name
    :param experiment_id: experiment id for experiment plan indexing
    :param vocabulary_size: vocabulary size (in words)
    :param embeddings_dimension: dimension of the embeddings layer
    :param recurrent_layer_type: Recurrent Neural Network architecture (Flatten, SimpleRNN, GRU, LSTM, ...)
    :param recurrent_units: number of processing units of the recurrent layer
    :param dropout_rate: dropout rate for the recurrent network
    :return: the Word Embeddings experiment
    """
    experiment_name = 'Rutger {} INPL = {}, VOCS = {}, EDIM = {}, RTYP = {} ({} units, DR = {})'. \
        format(experiment_id, corpus.input_size, vocabulary_size, embeddings_dimension, recurrent_layer_type,
               recurrent_units, dropout_rate)

    network = get_neural_network(name=network_name,
                                 input_length=corpus.input_size,
                                 vocabulary_size=vocabulary_size,
                                 embeddings_dimension=embeddings_dimension,
                                 output_categories=corpus.output_size,
                                 recurrent_layer_type=recurrent_layer_type,
                                 recurrent_units=recurrent_units,
                                 dropout_rate=dropout_rate)

    experiment = Experiment(name=experiment_name,
                            id=experiment_id,
                            corpus=corpus,
                            neural_network=network,
                            training_configuration=training_configuration_embeddings)
    return experiment


# def load_glove_experiment(corpus: Corpus,
#                           input_length: int,
#                           vocabulary_size: int,
#                           embeddings_dimension: int,
#                           inject_embeddings: bool = True) -> Experiment:
#     """ Loads the GloVe experiment
#
#     :param corpus: corpus (common to all experiments)
#     :param input_length: number of tokens in the input
#     :param vocabulary_size: vocabulary size (in words)
#     :param embeddings_dimension: dimension of the embeddings layer
#     :return: the GloVe experiment
#     """
#     glove_experiment_name = 'GloVe (input length = {}, embeddings dimension = {}, inject = {})'. \
#         format(input_length, embeddings_dimension, inject_embeddings)
#
#     glove_network = get_neural_network(input_length=input_length,
#                                        vocabulary_size=vocabulary_size,
#                                        embeddings_dimension=embeddings_dimension)
#
#     if inject_embeddings:
#         glove_embeddings = glove.load_embeddings_matrix(embeddings_dimension=embeddings_dimension)
#         glove_network.set_weights(layer_name='embeddings', weights=[glove_embeddings])
#         glove_network.freeze_layer('embeddings')
#
#     glove_corpus = corpus.copy()
#     glove_corpus.get_validation_from_training_set(validation_size=training_configuration_glove.validation_set_size)
#     glove_corpus.resize(training_set_size=200)
#
#     glove_experiment = Experiment(name=glove_experiment_name,
#                                   id='glove',
#                                   corpus=glove_corpus,
#                                   neural_network=glove_network,
#                                   training_configuration=training_configuration_glove)
#
#     return glove_experiment


def load_experiment_plan(corpus: Corpus,
                         vocabulary_size: int,
                         embeddings_dimension: int) -> ExperimentPlan:
    """ Loads all experiment configurations

    :param corpus: corpus (common to all experiments)
    :param input_length: number of tokens in the input
    :param vocabulary_size: vocabulary size (in words)
    :param embeddings_dimension: dimension of the embeddings layer
    :return: the Word Embeddings Experiment Plan
    """
    flatten_experiment = load_experiment(corpus=corpus,
                                         experiment_id='flatten',
                                         network_name='Flatten',
                                         vocabulary_size=vocabulary_size,
                                         embeddings_dimension=embeddings_dimension,
                                         recurrent_layer_type='Flatten')

    simplernn_experiment = load_experiment(corpus=corpus,
                                           experiment_id='simplernn',
                                           network_name='SimpleRNN',
                                           vocabulary_size=vocabulary_size,
                                           embeddings_dimension=embeddings_dimension,
                                           recurrent_layer_type='SimpleRNN',
                                           recurrent_units=100,
                                           dropout_rate=.4)

    lstm_experiment = load_experiment(corpus=corpus,
                                      experiment_id='lstm',
                                      network_name='LSTM',
                                      vocabulary_size=vocabulary_size,
                                      embeddings_dimension=embeddings_dimension,
                                      recurrent_layer_type='LSTM',
                                      recurrent_units=100,
                                      dropout_rate=.4)

    # bilstm_experiment = load_experiment(corpus=corpus,
    #                                     experiment_id='bilstm',
    #                                     network_name='BILSTM',
    #                                     vocabulary_size=vocabulary_size,
    #                                     embeddings_dimension=embeddings_dimension,
    #                                     recurrent_layer_type='BILSTM',
    #                                     recurrent_units=100,
    #                                     dropout_rate=0.)

    bilstm_experiment = load_experiment(corpus=corpus,
                                        experiment_id='bilstm-dropout',
                                        network_name='BILSTM with Dropout',
                                        vocabulary_size=vocabulary_size,
                                        embeddings_dimension=embeddings_dimension,
                                        recurrent_layer_type='BILSTM',
                                        recurrent_units=100,
                                        dropout_rate=0.4)

    # glove_experiment_1 = load_glove_experiment(corpus=corpus,
    #                                          input_length=input_length,
    #                                          vocabulary_size=vocabulary_size,
    #                                          embeddings_dimension=embeddings_dimension,
    #                                          inject_embeddings=True)
    #
    # glove_experiment_2 = load_glove_experiment(corpus=corpus,
    #                                            input_length=input_length,
    #                                            vocabulary_size=vocabulary_size,
    #                                            embeddings_dimension=embeddings_dimension,
    #                                            inject_embeddings=False)

    # Experiment: effect of dropout with different rates
    experiment_list = [
        flatten_experiment,
        simplernn_experiment,
        lstm_experiment,
        bilstm_experiment
    ]

    experiment_plan = ExperimentPlan(name='Rutger Intent Detection', experiments=experiment_list)
    return experiment_plan

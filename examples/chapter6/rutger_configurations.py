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
global_batch_size = 100
global_validation_set_size = 10000
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

training_parameters_glove = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=0.001),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit': {
            'epochs': global_epochs,
            'batch_size': global_batch_size,
            'shuffle': shuffle}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION,
        'set_size': global_validation_set_size}}

training_configuration_embeddings = TrainingConfiguration(configuration=training_parameters_embeddings)
training_configuration_glove = TrainingConfiguration(configuration=training_parameters_glove)


# def get_network_configuration(input_length: int,
#                               vocabulary_size: int,
#                               embeddings_dimension: int,
#                               output_categories: int):
#     return [
#         {'layer_type': 'Embedding', 'input_dim': vocabulary_size, 'output_dim': embeddings_dimension,
#          'input_length': input_length, 'name': 'embeddings'},
#         {'layer_type': 'Flatten'},
#         {'layer_type': 'Dense', 'units': output_categories, 'activation': 'softmax'}
#     ]


def get_network_configuration(input_length: int,
                              vocabulary_size: int,
                              embeddings_dimension: int,
                              output_categories: int,
                              recurrent_layer_type: str = 'SimpleRNN',
                              recurrent_units: int = 100):
    """ Network Configurations

    :param input_length: maximum number of tokens in the input
    :param vocabulary_size: vocabulary size, for one-hot encoding
    :param embeddings_dimension: dimension of the embeddings
    :param output_categories: number of distinct categories (intentions) in the output
    :param recurrent_layer_type: Recurrent Neural Network architecture (Flatten, SimpleRNN, GRU, LSTM, ...)
    :param recurrent_units: number of processing units of the recurrent layer
    :return: list of neural network layers configuration
    """
    if recurrent_layer_type == 'Flatten':
        recurrent_layer_configuration = {'layer_type': recurrent_layer_type}
    else:
        recurrent_layer_configuration = {'layer_type': recurrent_layer_type, 'units': recurrent_units}

    return [
        {'layer_type': 'Embedding', 'input_dim': vocabulary_size, 'output_dim': embeddings_dimension,
         'input_length': input_length, 'name': 'embeddings'},
        # {'layer_type': recurrent_layer_type, 'units': recurrent_units},
        recurrent_layer_configuration,
        {'layer_type': 'Dense', 'units': output_categories, 'activation': 'softmax'}
    ]


def get_neural_network(input_length: int,
                       vocabulary_size: int,
                       embeddings_dimension: int,
                       output_categories: int,
                       recurrent_layer_type: str,
                       recurrent_units: int):
    """ Creates the neural network model according to the configurations

    :param input_length: maximum number of tokens in the input
    :param vocabulary_size: vocabulary size, for one-hot encoding
    :param embeddings_dimension: dimension of the embeddings
    :param output_categories: number of distinct categories (intentions) in the output
    :param recurrent_layer_type: Recurrent Neural Network architecture (Flatten, SimpleRNN, GRU, LSTM, ...)
    :param recurrent_units: number of processing units of the recurrent layer
    :return: neural network created according to the configurations
    """
    embeddings = get_network_configuration(input_length=input_length,
                                           vocabulary_size=vocabulary_size,
                                           embeddings_dimension=embeddings_dimension,
                                           output_categories=output_categories,
                                           recurrent_layer_type=recurrent_layer_type,
                                           recurrent_units=recurrent_units)
    return NeuralNetwork.from_configurations(name='Embeddings', layers_configuration=embeddings)


def load_experiment(corpus: Corpus,
                    experiment_id: str,
                    input_length: int,
                    vocabulary_size: int,
                    embeddings_dimension: int,
                    recurrent_layer_type: str = 'Flatten',
                    recurrent_units: int = 0) -> Experiment:
    """ Loads the Word Embeddings network experiment

    :param corpus: corpus (common to all experiments)
    :param input_length: number of tokens in the input
    :param vocabulary_size: vocabulary size (in words)
    :param embeddings_dimension: dimension of the embeddings layer
    :param recurrent_layer_type: Recurrent Neural Network architecture (Flatten, SimpleRNN, GRU, LSTM, ...)
    :param recurrent_units: number of processing units of the recurrent layer
    :return: the Word Embeddings experiment
    """
    experiment_name = 'Embeddings (input length = {}, embeddings dimension = {}, recurrent type = {})'. \
        format(input_length, embeddings_dimension, recurrent_layer_type)

    network = get_neural_network(input_length=input_length,
                                 vocabulary_size=vocabulary_size,
                                 embeddings_dimension=embeddings_dimension,
                                 output_categories=corpus.training_set.output_size,
                                 recurrent_layer_type=recurrent_layer_type,
                                 recurrent_units=recurrent_units)

    experiment = Experiment(name=experiment_name,
                            id=experiment_id,
                            corpus=corpus,
                            neural_network=network,
                            training_configuration=training_configuration_embeddings)
    return experiment


def load_glove_experiment(corpus: Corpus,
                          input_length: int,
                          vocabulary_size: int,
                          embeddings_dimension: int,
                          inject_embeddings: bool = True) -> Experiment:
    """ Loads the GloVe experiment

    :param corpus: corpus (common to all experiments)
    :param input_length: number of tokens in the input
    :param vocabulary_size: vocabulary size (in words)
    :param embeddings_dimension: dimension of the embeddings layer
    :return: the GloVe experiment
    """
    glove_experiment_name = 'GloVe (input length = {}, embeddings dimension = {}, inject = {})'. \
        format(input_length, embeddings_dimension, inject_embeddings)

    glove_network = get_neural_network(input_length=input_length,
                                       vocabulary_size=vocabulary_size,
                                       embeddings_dimension=embeddings_dimension)

    if inject_embeddings:
        glove_embeddings = glove.load_embeddings_matrix(embeddings_dimension=embeddings_dimension)
        glove_network.set_weights(layer_name='embeddings', weights=[glove_embeddings])
        glove_network.freeze_layer('embeddings')

    glove_corpus = corpus.copy()
    glove_corpus.get_validation_from_training_set(validation_size=training_configuration_glove.validation_set_size)
    glove_corpus.resize(training_set_size=200)

    glove_experiment = Experiment(name=glove_experiment_name,
                                  id='glove',
                                  corpus=glove_corpus,
                                  neural_network=glove_network,
                                  training_configuration=training_configuration_glove)

    return glove_experiment


def load_experiment_plan(corpus: Corpus,
                         input_length: int,
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
                                         input_length=input_length,
                                         vocabulary_size=vocabulary_size,
                                         embeddings_dimension=embeddings_dimension,
                                         recurrent_layer_type='Flatten')

    simplernn_experiment = load_experiment(corpus=corpus,
                                           experiment_id='simplernn',
                                           input_length=input_length,
                                           vocabulary_size=vocabulary_size,
                                           embeddings_dimension=embeddings_dimension,
                                           recurrent_layer_type='SimpleRNN',
                                           recurrent_units=100)

    lstm_experiment = load_experiment(corpus=corpus,
                                      experiment_id='simplernn',
                                      input_length=input_length,
                                      vocabulary_size=vocabulary_size,
                                      embeddings_dimension=embeddings_dimension,
                                      recurrent_layer_type='LSTM',
                                      recurrent_units=100)

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
        lstm_experiment
    ]

    experiment_plan = ExperimentPlan(name='Rutger Intent Detection', experiments=experiment_list)

    return experiment_plan

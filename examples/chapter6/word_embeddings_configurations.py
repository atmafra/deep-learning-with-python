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
global_batch_size = 32
global_validation_set_size = 10000
shuffle = True

# Training configuration
training_parameters_embeddings = {
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
        'set_size': 5000}}

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


def get_network_configuration(input_length: int,
                              vocabulary_size: int,
                              embeddings_dimension: int):
    """
    Network Configurations

    :param input_length: maximum number of tokens in the input
    :param vocabulary_size: vocabulary size, for one-hot encoding
    :param embeddings_dimension: dimension of the embeddings
    :return: list of neural network layers configuration
    """
    return [
        {'layer_type': 'Embedding', 'input_dim': vocabulary_size, 'output_dim': embeddings_dimension,
         'input_length': input_length, 'name': 'embeddings'},
        {'layer_type': 'Flatten'},
        {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
    ]


def get_neural_network(input_length: int,
                       vocabulary_size: int,
                       embeddings_dimension: int):
    """ Creates the neural network model according to the configurations

    :param input_length: maximum number of tokens in the input
    :param vocabulary_size: vocabulary size, for one-hot encoding
    :param embeddings_dimension: dimension of the embeddings
    :return: neural network created according to the configurations
    """
    embeddings = get_network_configuration(input_length=input_length,
                                           vocabulary_size=vocabulary_size,
                                           embeddings_dimension=embeddings_dimension)
    return NeuralNetwork.from_configurations(name='Embeddings', layers_configuration=embeddings)


def load_embeddings_experiment(corpus: Corpus,
                               input_length: int,
                               vocabulary_size: int,
                               embeddings_dimension: int) -> Experiment:
    """ Loads the Word Embeddings network experiment

    :param corpus: corpus (common to all experiments)
    :param input_length: number of tokens in the input
    :param vocabulary_size: vocabulary size (in words)
    :param embeddings_dimension: dimension of the embeddings layer
    :return: the Word Embeddings experiment
    """
    embeddings_experiment_name = 'Embeddings (input length = {}, embeddings dimension = {})'. \
        format(input_length, embeddings_dimension)

    embeddings_network = get_neural_network(input_length=input_length,
                                            vocabulary_size=vocabulary_size,
                                            embeddings_dimension=embeddings_dimension)

    embeddings_corpus = corpus.copy()
    embeddings_experiment = Experiment(name=embeddings_experiment_name,
                                       id='embeddings',
                                       corpus=embeddings_corpus,
                                       neural_network=embeddings_network,
                                       training_configuration=training_configuration_embeddings)
    return embeddings_experiment


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
    embeddings_experiment = load_embeddings_experiment(corpus=corpus,
                                                       input_length=input_length,
                                                       vocabulary_size=vocabulary_size,
                                                       embeddings_dimension=embeddings_dimension)

    glove_experiment_1 = load_glove_experiment(corpus=corpus,
                                             input_length=input_length,
                                             vocabulary_size=vocabulary_size,
                                             embeddings_dimension=embeddings_dimension,
                                             inject_embeddings=True)

    glove_experiment_2 = load_glove_experiment(corpus=corpus,
                                               input_length=input_length,
                                               vocabulary_size=vocabulary_size,
                                               embeddings_dimension=embeddings_dimension,
                                               inject_embeddings=False)

    # Experiment: effect of dropout with different rates
    experiment_list = [
        # embeddings_experiment,
        glove_experiment_1,
        glove_experiment_2
    ]

    experiment_plan = ExperimentPlan(name='Word Embeddings', experiments=experiment_list)

    return experiment_plan

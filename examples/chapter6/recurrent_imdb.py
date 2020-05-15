from keras import optimizers, layers

from core.experiment import Experiment
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from corpora.imdb import imdb

global_epochs = 10
global_batch_size = 128
global_shuffle = False


def load_imdb_corpus(maximum_tokens_per_text: int,
                     vocabulary_size: int):
    return imdb.build_corpus(imdb_path='../../corpora/imdb/data/aclImdb',
                             maximum_tokens_per_text=maximum_tokens_per_text,
                             vocabulary_size=vocabulary_size,
                             randomize=True)


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
                       input_length: int):
    configurations = [
        {'layer_type': 'Embedding', 'input_dim': vocabulary_size, 'output_dim': embeddings_dimension,
         'input_length': input_length, 'name': 'embeddings'},
        {'layer_type': 'SimpleRNN', 'units': embeddings_dimension},
        {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
    ]

    return NeuralNetwork.from_configurations(name='SimpleRNN',
                                             layers_configuration=configurations)


def run():
    maxlen = 500
    vocabulary_size = 10000
    embeddings_dimension = 32

    corpus = load_imdb_corpus(maximum_tokens_per_text=maxlen,
                              vocabulary_size=vocabulary_size)

    set_size = corpus.size
    validation_split = 0.2

    training_configuration = get_training_configuration(set_size=set_size, validation_split=validation_split)

    neural_network = get_neural_network(vocabulary_size=vocabulary_size,
                                        embeddings_dimension=embeddings_dimension,
                                        input_length=maxlen)

    experiment = Experiment(name='SimpleRNN',
                            id='simple-rnn',
                            corpus=corpus,
                            neural_network=neural_network,
                            training_configuration=training_configuration)

    experiment.run(train=True,
                   plot_training_loss=True,
                   plot_training_accuracy=True,
                   test=True,
                   print_test_results=True,
                   save=False)

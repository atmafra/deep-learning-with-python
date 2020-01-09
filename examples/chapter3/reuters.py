import numpy as np
from keras import optimizers
from keras.datasets import reuters

from core.corpus import Corpus, CorpusType
from core.experiment import Experiment
from core.file_structures import CorpusFileStructure
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from utils import dataset_utils as dsu

if __name__ == '__main__':
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def build_corpus(num_words: int, encoding_schema: str, save: bool = True, verbose: bool = True):
    if verbose:
        print('Loading Reuters dataset...')

    corpus_datasets = dsu.separate_corpus(reuters.load_data(num_words=num_words))
    train_data = corpus_datasets[0][0]
    test_data = corpus_datasets[1][0]
    train_labels = corpus_datasets[0][1]
    test_labels = corpus_datasets[1][1]

    # vectorization of the input data
    train_data = dsu.one_hot_encode(train_data, num_words)
    test_data = dsu.one_hot_encode(test_data, num_words)

    # vectorization of the labels
    categories = dsu.count_unique_values(train_labels)

    if encoding_schema == 'one-hot':
        train_labels = dsu.one_hot_encode(train_labels, categories)
        test_labels = dsu.one_hot_encode(test_labels, categories)

    elif encoding_schema == 'int-array':
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

    corpus_name = 'Reuters ({})'.format(encoding_schema)
    corpus = Corpus.from_datasets(training_input=train_data,
                                  training_output=train_labels,
                                  test_input=test_data,
                                  test_output=test_labels,
                                  name=corpus_name)

    if verbose:
        print('Training phrases:', corpus.training_set.length)
        print('Test phrases    :', corpus.test_set.length)
        print('Input size      :', corpus.input_size)
        print('Output size:    :', corpus.output_size)
        print('Categories      :', categories)

    if save:
        save_corpus(corpus)

    return corpus


def save_corpus(corpus: Corpus,
                corpus_file_structure: CorpusFileStructure = None):
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus.name,
                                                                  base_path='data/reuters')
    corpus_file_structure.save_corpus(corpus)


def load_corpus(encoding_schema: str,
                corpus_file_structure: CorpusFileStructure = None):
    corpus_name = 'Reuters ({})'.format(encoding_schema)
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus_name,
                                                                  base_path='data/reuters')

    return corpus_file_structure.load_corpus(corpus_name=corpus_name,
                                             datasets_base_name=corpus_name)


def load_experiment(corpus: Corpus, encoding_schema):
    """ Reuters neural network hyperparameters and experiments configuration

    Args:
        corpus: the Reuters corpus, encoded according to the given encoding schema
        encoding_schema: one-hot or int-array

    """
    # network hyperparameters
    input_size = corpus.input_size
    output_size = -1
    if encoding_schema == 'one-hot':
        output_size = corpus.output_size
    elif encoding_schema == 'int-array':
        output_size = corpus.count_categories
    hidden_activation = 'relu'
    output_activation = 'softmax'

    # optimization hyperparameters
    optimizer = 'rmsprop'
    learning_rate = 0.001
    metrics = ['accuracy']
    loss = None
    if encoding_schema == 'one-hot':
        loss = 'categorical_crossentropy'
    elif encoding_schema == 'int-array':
        loss = 'sparse_categorical_crossentropy'

    # train parameters
    epochs = 20
    batch_size = 512
    shuffle = True
    validation_set_size = 1000

    layers_configuration = [
        {'layer_type': 'Dense', 'units': 64, 'activation': hidden_activation, 'input_shape': (input_size,)},
        {'layer_type': 'Dense', 'units': 64, 'activation': hidden_activation},
        {'layer_type': 'Dense', 'units': 64, 'activation': hidden_activation},
        {'layer_type': 'Dense', 'units': output_size, 'activation': output_activation}]

    training_parameters = {
        'keras': {
            'compile': {
                'optimizer': optimizers.RMSprop(lr=learning_rate),
                'loss': loss,
                'metrics': metrics},
            'fit': {
                'epochs': epochs,
                'batch_size': batch_size,
                'shuffle': shuffle}},
        'validation': {
            'strategy': ValidationStrategy.CROSS_VALIDATION,
            'set_size': validation_set_size}}

    name = 'Reuters (encoding schema: {})'.format(encoding_schema)
    neural_network = NeuralNetwork.from_configurations(name=name, layers_configuration=layers_configuration)
    training_configuration = TrainingConfiguration(configuration=training_parameters)

    return Experiment(name=name,
                      neural_network=neural_network,
                      training_configuration=training_configuration,
                      corpus_type=CorpusType.CORPUS_DATASET,
                      corpus=corpus)


def run(num_words: int = 10000, encoding_schema: str = 'one-hot', build: bool = True):
    corpus = None
    if build:
        corpus = build_corpus(num_words=num_words, encoding_schema=encoding_schema, save=True)
    else:
        corpus = load_corpus(encoding_schema=encoding_schema)
    experiment = load_experiment(corpus=corpus, encoding_schema=encoding_schema)
    experiment.run(print_results=True, plot_history=True)
    experiment.save_model(path='models/reuters')

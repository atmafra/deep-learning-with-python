import numpy as np
from keras import optimizers
from keras.datasets import reuters

from core import sets
from core.experiment import Experiment
from core.network import LayerType, ValidationStrategy
from core.sets import Corpus
from utils import dataset_utils as dsu

if __name__ == '__main__':
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def load_corpus(num_words: int, encoding_schema: str, verbose: bool = True):
    if verbose:
        print('Loading Reuters dataset...')

    dataset = reuters.load_data(num_words=num_words)
    (train_data, train_labels), (test_data, test_labels) = dsu.separate_corpus(dataset)

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

    corpus = sets.Corpus.from_datasets(train_data, train_labels, test_data, test_labels)

    if verbose:
        print('Training phrases:', len(train_data))
        print('Test phrases    :', len(test_data))
        print('Input size      :', corpus.input_size)
        print('Output size:    :', corpus.output_size)
        print('Categories      :', categories)

    return corpus


def load_experiment(corpus: Corpus, encoding_schema):
    """ Reuters neural network hyperparameters and experiments configuration

    Args:
        corpus: the Reuters corpus, encoded according to the given encoding schema
        encoding_schema: one-hot or int-array

    """
    # network hyperparameters
    input_size = corpus.input_size
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

    # training parameters
    epochs = 20
    batch_size = 512
    shuffle = True
    validation_set_size = 1000

    layers_configuration = [
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': hidden_activation, 'input_shape': (input_size,)},
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': hidden_activation},
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': hidden_activation},
        {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

    training_configuration = {
        'keras': {
            'compile': {
                'optimizer': optimizers.RMSprop(lr=learning_rate),
                'loss': loss,
                'metrics': metrics},
            'fit': {
                'epochs': epochs,
                'batch_size': batch_size,
                'shuffle': shuffle}},
        'validation' : {
            'strategy': ValidationStrategy.CROSS_VALIDATION,
            'set_size': validation_set_size}}

    experiment = Experiment(name="Reuters (encoding schema: {})".format(encoding_schema),
                            corpus=corpus,
                            layers_configuration=layers_configuration,
                            training_configuration=training_configuration)

    return experiment


def run(num_words: int = 10000, encoding_schema: str = 'one-hot'):
    if encoding_schema == 'one-hot':
        # one-hot encoding experiment
        corpus_one_hot = load_corpus(num_words=num_words, encoding_schema='one-hot')
        experiment_one_hot = load_experiment(corpus=corpus_one_hot, encoding_schema='one-hot')
        experiment_one_hot.run(print_results=True, plot_history=True)

    elif encoding_schema == 'int-array':
        # int-array encoding experiment
        corpus_int_array = load_corpus(num_words=num_words, encoding_schema='int-array')
        experiment_int_array = load_experiment(corpus=corpus_int_array, encoding_schema='int-array')
        experiment_int_array.run(print_results=True, plot_history=True)

import numpy as np
from keras.datasets import reuters

from core import network as net, sets
from core.network import LayerType, NetworkOutputType
from utils import dataset_utils as dsu
from utils import history_utils as hplt

if __name__ == '__main__':
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def load(num_words: int = 10000, encoding_schema: str = 'one-hot', verbose: bool = True):
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
        print('Input size      :', corpus.input_size())
        print('Categories      :', categories)

    return corpus


def hyperparameters(input_size: int, output_size: int):
    """ IMDB neural network hyperparameters
    """
    # network hyperparameters
    loss = 'categorical_crossentropy'

    network_configuration = {
        'input_size': input_size,
        'output_size': output_size,
        'output_type': NetworkOutputType.CATEGORICAL,
        'optimizer': 'rmsprop',
        'learning_rate': 0.001,
        'loss': loss,
        'metrics': ['accuracy']}

    # layers hyperparameters
    hidden_activation = 'relu'
    output_activation = 'softmax'

    layers_configuration = [
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': hidden_activation, 'input_shape': (input_size,)},
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': hidden_activation},
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': hidden_activation},
        {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

    return network_configuration, layers_configuration


def run(num_words: int = 10000, encoding_schema: str = 'one-hot'):
    # load corpus
    if encoding_schema == 'one-hot':
        loss = 'categorical_crossentropy'

    elif encoding_schema == 'int-array':
        loss = 'sparse_categorical_crossentropy'

    # corpus = sets.Corpus.from_tuple(load(num_words=num_words, encoding_schema=encoding_schema))
    corpus = load(num_words=num_words, encoding_schema=encoding_schema)

    # load hyperparameters
    input_size = corpus.input_size()
    output_size = corpus.output_size()
    network_configuration, layers_configuration = hyperparameters(input_size, output_size)

    # create the neural network
    reuters_nnet = net.create_network(network_configuration=network_configuration,
                                      layer_configuration_list=layers_configuration)

    # train the neural network
    validation_set_size = 1000
    validation_set, training_set_remaining = corpus.training_set.split(size=validation_set_size)

    history = net.train_network(network=reuters_nnet,
                                training_set=training_set_remaining,
                                epochs=20,
                                batch_size=128,
                                shuffle=True,
                                validation_set=validation_set)

    (test_loss, test_accuracy) = net.test_network(reuters_nnet, corpus.test_set)
    print("loss     =", test_loss)
    print("accuracy = {:.2%}".format(test_accuracy))

    hplt.plot_accuracy_dict(history.history,
                            title='Reuters {}: Training and Validation Accuracies'.format(encoding_schema))

import numpy as np
from keras import optimizers
from keras.datasets import imdb

from core.experiment import Experiment, ExperimentPlan
from core.network import LayerType
from core.sets import Corpus
from utils import dataset_utils as dsu

num_words = 10000


def load_corpus(words: int = 10000, verbose: bool = True) -> Corpus:
    """"Loads the IMDB dataset into a corpus object

    Args:
        words (int): word limit in the reverse index
        verbose (bool): outputs progress messages

    """
    if verbose:
        print("Loading IMDB dataset...")

    corpus = imdb.load_data(num_words=words)
    (train_samples, train_labels), (test_samples, test_labels) = dsu.separate_corpus(corpus)

    # one-hot encode the phrases
    # global vector_dimension
    vector_dimension = words
    training_inputs = dsu.one_hot_encode(train_samples, vector_dimension)
    test_inputs = dsu.one_hot_encode(test_samples, vector_dimension)

    # vectorize the labels
    training_outputs = np.asarray(train_labels).astype('float32')
    test_outputs = np.asarray(test_labels).astype('float32')

    if verbose:
        print("{} train reviews loaded".format(len(train_samples)))
        print("{} test reviews loaded".format(len(test_samples)))

    # create the corpus
    return Corpus.from_datasets(training_inputs, training_outputs, test_inputs, test_outputs)


def load_experiments(corpus: Corpus):
    """Loads the experiment hyperparameters
    """
    # network parameters
    input_size = num_words
    output_size = 1
    hidden_activation = 'relu'
    output_activation = 'sigmoid'

    # optimization parameters
    learning_rate = 0.001
    optimizer = optimizers.RMSprop(lr=learning_rate)
    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    # training parameters
    epochs = 20
    batch_size = 512
    shuffle = True
    validation_set_size = 10000

    imdb_1 = [
        {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
        {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
        {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

    imdb_2 = [
        {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
        {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
        {'layer_type': LayerType.DENSE, 'units': 16, 'activation': hidden_activation},
        {'layer_type': LayerType.DENSE, 'units': output_size, 'activation': output_activation}]

    training_configuration = {
        'keras': {
            'compile': {
                'optimizer': optimizer,
                'loss': loss,
                'metrics': metrics},
            'fit': {
                'epochs': epochs,
                'batch_size': batch_size,
                'shuffle': shuffle}},
        'validation_set_size': validation_set_size}

    experiment1 = Experiment(name='2 Hidden Layers, 16 units each',
                             corpus=corpus,
                             layers_configuration_list=imdb_1,
                             training_configuration=training_configuration)

    experiment2 = Experiment(name='3 Hidden Layers, 16 units each',
                             corpus=corpus,
                             layers_configuration_list=imdb_2,
                             training_configuration=training_configuration)

    plan = ExperimentPlan(name='Effect of the number of hidden layers',
                          experiments=[experiment1, experiment2])

    return plan


def run():
    corpus = load_corpus(words=num_words)
    experiment_plan = load_experiments(corpus=corpus)
    experiment_plan.run(print_results=True, plot_training_loss=False, plot_training_accuracy=False)

    experiment_plan.plot_loss("Training Loss", training=True, validation=False)
    experiment_plan.plot_loss("Validation Loss", training=False, validation=True)
    experiment_plan.plot_accuracy("Training Accuracy", training=True, validation=False)
    experiment_plan.plot_accuracy("Validation Accuracy", training=False, validation=True)

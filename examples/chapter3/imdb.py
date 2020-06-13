import numpy as np
from keras import optimizers
from keras.datasets import imdb

from core.corpus import Corpus, CorpusType
from core.experiment import Experiment, ExperimentPlan
from core.file_structures import CorpusFileStructure
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from utils import dataset_utils as dsu

num_words = 10000


def build_corpus(words: int = 10000,
                 save: bool = True,
                 verbose: bool = True) -> Corpus:
    """"
    Loads the IMDB dataset into a corpus object
    :param words: word limit in the reverse index
    :param save: save corpus after loading and pre-processing data
    :param verbose: outputs progress messages
    """
    if verbose:
        print("Loading IMDB dataset...")

    corpus_datasets = dsu.separate_corpus(imdb.load_data(num_words=words))
    train_samples = corpus_datasets[0][0]
    test_samples = corpus_datasets[1][0]
    train_labels = corpus_datasets[0][1]
    test_labels = corpus_datasets[1][1]

    # one-hot encode the phrases
    vector_dimension = words
    training_inputs = dsu.one_hot_encode(train_samples, vector_dimension)
    test_inputs = dsu.one_hot_encode(test_samples, vector_dimension)

    # vectorize the labels
    training_outputs = np.asarray(train_labels).astype('float32')
    test_outputs = np.asarray(test_labels).astype('float32')

    # create the corpus
    corpus = Corpus.from_datasets(training_input=training_inputs,
                                  training_output=training_outputs,
                                  test_input=test_inputs,
                                  test_output=test_outputs,
                                  name='IMDB')

    if verbose:
        print("{} train reviews loaded".format(corpus.training_set.length))
        print("{} test reviews loaded".format(corpus.test_set.length))

    if save:
        save_corpus(corpus)

    return corpus


def save_corpus(corpus: Corpus,
                corpus_file_structure: CorpusFileStructure = None):
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus.name, base_path='data/imdb')
    corpus_file_structure.save_corpus(corpus)


def load_corpus(corpus_file_structure: CorpusFileStructure = None):
    corpus_name = 'IMDB'
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus_name, base_path='data/imdb')
    return corpus_file_structure.load_corpus(corpus_name='IMDB', datasets_base_name='imdb')


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

    # train parameters
    epochs = 5
    batch_size = 512
    shuffle = True
    validation_set_size = 10000

    imdb_1 = [
        {'class_name': 'Dense', 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
        {'class_name': 'Dense', 'units': 16, 'activation': hidden_activation},
        {'class_name': 'Dense', 'units': output_size, 'activation': output_activation}]

    imdb_2 = [
        {'class_name': 'Dense', 'units': 16, 'activation': hidden_activation, 'input_shape': (input_size,)},
        {'class_name': 'Dense', 'units': 16, 'activation': hidden_activation},
        {'class_name': 'Dense', 'units': 16, 'activation': hidden_activation},
        {'class_name': 'Dense', 'units': output_size, 'activation': output_activation}]

    training_parameters = {
        'keras': {
            'compile': {
                'optimizer': optimizer,
                'loss': loss,
                'metrics': metrics},
            'fit': {
                'epochs': epochs,
                'batch_size': batch_size,
                'shuffle': shuffle}},
        'validation': {
            'strategy': ValidationStrategy.CROSS_VALIDATION,
            'set_size': validation_set_size}}

    training_configuration = TrainingConfiguration(configuration=training_parameters)

    network1 = NeuralNetwork.from_configurations(name='IMDB - 2 Hidden Layers, 16 units',
                                                 layers_configuration=imdb_1)

    experiment1 = Experiment(name=network1.name,
                             neural_network=network1,
                             training_configuration=training_configuration,
                             corpus_type=CorpusType.CORPUS_DATASET,
                             corpus=corpus)

    network2 = NeuralNetwork.from_configurations(name='IMDB - 3 Hidden Layers, 16 units',
                                                 layers_configuration=imdb_2)

    experiment2 = Experiment(name=network2.name,
                             neural_network=network2,
                             training_configuration=training_configuration,
                             corpus_type=CorpusType.CORPUS_DATASET,
                             corpus=corpus)

    plan = ExperimentPlan(name='Effect of the number of hidden layers',
                          experiments=[experiment1, experiment2])

    return plan


def run(build: bool = True):
    if build:
        corpus = build_corpus(words=num_words)
    else:
        corpus = load_corpus()

    experiment_plan = load_experiments(corpus=corpus)
    experiment_plan.run(print_results=True,
                        plot_training_loss=True,
                        plot_training_accuracy=True,
                        plot_validation_loss=True,
                        plot_validation_accuracy=True,
                        display_progress_bars=True)

    experiment_plan.save_models('models/imdb')
    # experiment_plan.plot_loss("Training Loss", plot_training_series=True, plot_validation_series=False)
    # experiment_plan.plot_loss("Validation Loss", plot_training_series=False, plot_validation_series=True)
    # experiment_plan.plot_accuracy("Training Accuracy", plot_training_series=True, plot_validation_series=False)
    # experiment_plan.plot_accuracy("Validation Accuracy", plot_training_series=False, plot_validation_series=True)

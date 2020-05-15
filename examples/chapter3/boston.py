from keras import optimizers
from keras.datasets import boston_housing as boston

from core.corpus import Corpus
from core.experiment import Experiment
from core.file_structures import CorpusFileStructure
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from utils import dataset_utils as dsu
from utils.history_utils import plot_mae


def build_corpus(name: str,
                 num_words: int = 10000,
                 save: bool = True, verbose: bool = True) -> Corpus:
    if verbose:
        print('Loading Boston Housing dataset...')

    corpus = boston.load_data()
    corpus_datasets = dsu.separate_corpus(corpus)
    training_inputs = corpus_datasets[0][0]
    training_outputs = corpus_datasets[0][1]
    test_inputs = corpus_datasets[1][0]
    test_outputs = corpus_datasets[1][1]

    # normalization of the train and test data
    dsu.normalize(training_inputs, test_inputs)

    # create the corpus
    corpus = Corpus.from_datasets(training_input=training_inputs,
                                  training_output=training_outputs,
                                  test_input=test_inputs,
                                  test_output=test_outputs,
                                  name=name)

    if verbose:
        print('Training examples:', corpus.training_set.length)
        print('Test examples    :', corpus.test_set.length)
        print('Minimum price    : {:.2f}'.format(corpus.min_output))
        print('Average price    : {:.2f}'.format(corpus.average_output))
        print('Maximum price    : {:.2f}'.format(corpus.max_output))

    if save:
        save_corpus(corpus=corpus)

    return corpus


def save_corpus(corpus: Corpus,
                corpus_file_structure: CorpusFileStructure = None):
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus.name, base_path='data/boston')
    corpus_file_structure.save_corpus(corpus=corpus)


def load_corpus(name: str,
                corpus_file_structure: CorpusFileStructure = None):
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=name,
                                                                  base_path='data/boston')
    return corpus_file_structure.load_corpus(corpus_name=name,
                                             datasets_base_name=name)


def load_experiment(corpus: Corpus) -> Experiment:
    """ Boston Housing neural network hyperparameters
    """
    # network hyperparameters
    input_size = corpus.input_size
    output_size = corpus.output_size
    hidden_activation = 'relu'
    output_activation = 'linear'

    # optimization hyperparameters
    optimizer = 'rmsprop'
    learning_rate = 0.001
    metrics = ['mae']
    loss = 'mse'

    # train parameters
    epochs = 80
    batch_size = 16
    k = 5
    shuffle = True

    layers_configuration = [
        {'layer_type': 'Dense', 'units': 64, 'activation': hidden_activation, 'input_shape': (input_size,)},
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
            'strategy': ValidationStrategy.K_FOLD_CROSS_VALIDATION,
            'shuffle': True,
            'k': 5}}

    neural_network = NeuralNetwork.from_configurations(name='Boston - MLP 2 layers, 64 units',
                                                       layers_configuration=layers_configuration)
    training_configuration = TrainingConfiguration(configuration=training_parameters)
    experiment = Experiment(name=neural_network.name,
                            corpus=corpus,
                            neural_network=neural_network,
                            training_configuration=training_configuration)

    return experiment


def run(build: bool = True):
    corpus_name = 'Boston Housing'
    corpus = None
    if build:
        corpus = build_corpus(name=corpus_name, num_words=10000, verbose=True)
    else:
        corpus = load_corpus(name=corpus_name)
    experiment = load_experiment(corpus=corpus)

    experiment.run(train=True,
                   print_training_results=True,
                   plot_training_loss=True,
                   plot_training_accuracy=False,
                   test=True,
                   print_test_results=True,
                   save=True,
                   model_path='models/boston')

    plot_mae(history=experiment.training_history,
             title='Boston Housing: Training and Validation Mean Absolute Error (MAE)')

from keras import optimizers
from keras.datasets import boston_housing as boston

from core.experiment import Experiment
from core.network import LayerType, ValidationStrategy
from core.sets import Corpus
from utils import dataset_utils as dsu
from utils.history_utils import plot_mae_dict, plot_loss_dict


def load_corpus(num_words: int = 10000, verbose: bool = True) -> Corpus:
    if verbose:
        print('Loading Boston Housing dataset...')

    corpus = boston.load_data()
    (training_inputs, training_outputs), (test_inputs, test_outputs) = dsu.separate_corpus(corpus)

    # normalization of the training and test data
    dsu.normalize(training_inputs, test_inputs)

    # create the corpus
    corpus = Corpus.from_datasets(training_inputs=training_inputs,
                                  training_outputs=training_outputs,
                                  test_inputs=test_inputs,
                                  test_outputs=test_outputs)

    if verbose:
        print('Training examples:', len(training_inputs))
        print('Test examples    :', len(test_inputs))
        print('Minimum price    : {:.2f}'.format(corpus.min_ouptut))
        print('Average price    : {:.2f}'.format(corpus.average_output))
        print('Maximum price    : {:.2f}'.format(corpus.max_output))

    return corpus


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

    # training parameters
    epochs = 80
    batch_size = 16
    k = 5
    shuffle = True

    layers_configuration = [
        {'layer_type': LayerType.DENSE, 'units': 64, 'activation': hidden_activation, 'input_shape': (input_size,)},
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
        'validation': {
            'strategy': ValidationStrategy.K_FOLD_CROSS_VALIDATION,
            'shuffle': True,
            'k': 5}}

    experiment = Experiment(name='Boston', corpus=corpus,
                            layers_configuration_list=layers_configuration,
                            training_configuration=training_configuration)

    return experiment


def run():
    corpus = load_corpus(num_words=10000, verbose=True)
    experiment = load_experiment(corpus=corpus)
    experiment.run(print_results=True, plot_history=False)

    plot_loss_dict(experiment.history,
                   title='Boston Housing: Training and Validation Mean Squared Error (MSE)')

    plot_mae_dict(experiment.history,
                  title='Boston Housing: Training and Validation Mean Absolute Error (MAE)')

